"""Train the sign recognition models.

Usage::

    python -m training.train --kind static --model rf
    python -m training.train --kind sequence
    python -m training.train --kind both

Static model    -> RandomForest (default) or MLP classifier on per-frame features.
Sequence model  -> a bidirectional GRU over the padded frame window for dynamic signs.

Runs ``preprocess`` first if the processed dataset is missing or empty.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import RecognitionConfig, load_config, ensure_dirs


def load_dataset(cfg: RecognitionConfig) -> dict:
    processed = cfg.paths["processed_data"] / "dataset.npz"
    if not processed.exists():
        from training.preprocess import preprocess

        preprocess(cfg)
    data = np.load(processed, allow_pickle=False)
    return {k: v for k, v in data.items()}


def save_sign_index(cfg: RecognitionConfig, static_classes, seq_classes,
                    static_model: str | None, seq_model: str | None,
                    accuracy: dict) -> None:
    ds = load_dataset(cfg)
    feat = ds.get("feature_dim")
    feature_dim = int(np.asarray(feat).item()) if feat is not None else 142
    index = {
        "feature_dim": feature_dim,
        "static_classes": static_classes,
        "sequence_classes": seq_classes,
        "static_model": static_model,
        "sequence_model": seq_model,
        "accuracy": accuracy,
    }
    cfg.paths["sign_index"].parent.mkdir(parents=True, exist_ok=True)
    cfg.paths["sign_index"].write_text(json.dumps(index, indent=2), encoding="utf-8")
    print(f"  wrote model index -> {cfg.paths['sign_index']}")


def train_static(cfg: RecognitionConfig, model_name: str) -> dict | None:
    data = load_dataset(cfg)
    if "X_static" not in data or data["X_static"].shape[0] == 0:
        print("  no static samples; skipping static model.")
        return None
    X, y = data["X_static"], data["y_static"]
    classes = [str(c) for c in data["static_labels"]]
    if len(classes) < 2:
        print("  static modelling needs >= 2 classes.")
        return None

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    import joblib

    if model_name == "mlp":
        clf = make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(256, 128), activation="relu",
                          max_iter=600, early_stopping=True, random_state=0),
        )
    else:
        clf = RandomForestClassifier(n_estimators=400, class_weight="balanced",
                                     random_state=0, n_jobs=-1)

    min_class = int(np.bincount(y).min())
    cv = StratifiedKFold(n_splits=min(5, min_class), shuffle=True, random_state=0)
    scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
    print(f"  static {model_name.upper()} cross-val accuracy: "
          f"{scores.mean():.3f} +- {scores.std():.3f}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=0
    )
    clf.fit(X_train, y_train)
    test_acc = float(clf.score(X_test, y_test))
    print(f"  hold-out test accuracy: {test_acc:.3f}")

    out = cfg.paths["static_model"]
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": clf, "classes": classes, "feature_dim": int(X.shape[1])}, out)
    print(f"  saved static model -> {out}")
    return {"kind": model_name, "cv_acc": float(scores.mean()), "test_acc": test_acc}


def train_sequence(cfg: RecognitionConfig) -> dict | None:
    data = load_dataset(cfg)
    if "X_seq" not in data or data["X_seq"].shape[0] == 0:
        print("  no sequence samples; skipping sequence model.")
        return None
    X = data["X_seq"].astype(np.float32)
    mask = data["mask_seq"].astype(np.float32)
    y = data["y_seq"]
    feat_dim, L = X.shape[2], X.shape[1]
    classes = [str(c) for c in data["seq_labels"]]
    if len(classes) < 2:
        print("  sequence modelling needs >= 2 classes.")
        return None

    import torch

    torch.set_num_threads(max(1, os.cpu_count() // 2))
    from app.sequence_model import SequenceClassifier

    def fit_model(model, Xt, mt, yt, epochs: int = 60, batch: int = 32):
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = torch.nn.CrossEntropyLoss()
        n = len(Xt)
        for epoch in range(epochs):
            perm = torch.randperm(n)
            total = 0.0
            for i in range(0, n, batch):
                idx = perm[i:i + batch]
                opt.zero_grad()
                logits = model(Xt[idx], mt[idx])
                loss = loss_fn(logits, yt[idx].long())
                loss.backward()
                opt.step()
                total += loss.item() * len(idx)
            if epoch % 20 == 0 or epoch == epochs - 1:
                print(f"    epoch {epoch + 1}/{epochs} loss={total / n:.4f}")
        return model

    from sklearn.model_selection import train_test_split

    X_tr, X_te, m_tr, m_te, y_tr, y_te = train_test_split(
        X, mask, y, test_size=0.2, stratify=y, random_state=0
    )
    tX, tm, ty = map(torch.from_numpy, (X_tr, m_tr, y_tr))
    model = fit_model(SequenceClassifier(feat_dim, len(classes)), tX, tm, ty)

    model.eval()
    with torch.no_grad():
        preds = model(torch.from_numpy(X_te), torch.from_numpy(m_te)).argmax(1).numpy()
    test_acc = float((preds == y_te).mean())
    print(f"  sequence GRU hold-out test accuracy: {test_acc:.3f}")

    users = data.get("users_seq", np.asarray([]))
    if len(users) and len(set(u for u in users)) > 1:
        accs = []
        for usr in sorted({u for u in users}):
            test_idx = users == usr
            tr_idx = ~test_idx
            if np.unique(y[tr_idx]).size < 2 or test_idx.sum() == 0:
                continue
            m2 = fit_model(SequenceClassifier(feat_dim, len(classes)),
                           torch.from_numpy(X[tr_idx]), torch.from_numpy(mask[tr_idx]),
                           torch.from_numpy(y[tr_idx]), epochs=25)
            m2.eval()
            with torch.no_grad():
                lg = m2(torch.from_numpy(X[test_idx]), torch.from_numpy(mask[test_idx]))
                accs.append(float((lg.argmax(1).numpy() == y[test_idx]).mean()))
            del m2
        if accs:
            print(f"  leave-one-user-out accuracy: {np.mean(accs):.3f} "
                  f"(users: {sorted({u for u in users})})")

    torch.save(
        {"state_dict": model.state_dict(), "classes": classes,
         "feature_dim": feat_dim, "seq_len": int(L), "hidden": 128},
        cfg.paths["sequence_model"],
    )
    print(f"  saved sequence model -> {cfg.paths['sequence_model']}")
    return {"kind": "gru", "test_acc": test_acc}


def run_cli(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Train sign recognition models.")
    parser.add_argument("--kind", choices=["static", "sequence", "both"], default="both")
    parser.add_argument("--model", choices=["rf", "mlp"], default="rf",
                        help="Static classifier family")
    args = parser.parse_args(argv)
    cfg = load_config()
    ensure_dirs(cfg)
    print(f"Training [{args.kind}] model(s) with data from {cfg.paths['processed_data']}")

    result: dict = {}
    if args.kind in ("static", "both"):
        r = train_static(cfg, args.model)
        if r:
            result["static"] = r
    if args.kind in ("sequence", "both"):
        r = train_sequence(cfg)
        if r:
            result["sequence"] = r

    data = load_dataset(cfg)
    static_classes = [str(c) for c in data.get("static_labels", [])]
    seq_classes = [str(c) for c in data.get("seq_labels", [])]
    static_model = (result.get("static", {}).get("kind")
                    if "static" in result else None)
    seq_model = result.get("sequence", {}).get("kind") if "sequence" in result else None
    save_sign_index(cfg, static_classes, seq_classes, static_model, seq_model, result)
    print("\nDone. Evaluate next: python -m training.evaluate")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_cli())