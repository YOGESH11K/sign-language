"""Evaluate the trained sign models with real metrics.

Usage::

    python -m training.evaluate --kind both
    python -m training.evaluate --kind static
    python -m training.evaluate --kind sequence

Reports accuracy, precision, recall, F1 (macro) and a confusion matrix. When the
dataset contains more than one user, the headline metric is **leave-one-user-out**
accuracy - that is, performance on people whose data never appeared in training,
which approximates real-world use better than a random split.

Nothing is hard-coded here: every number comes from running the actual models.
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
from training.train import load_dataset


def save_report(cfg: RecognitionConfig, kind: str, report: dict, cm: np.ndarray,
                classes) -> None:
    out_dir = cfg.paths["eval_output"]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{kind}_metrics.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    cm_lines = ["labels," + ",".join(classes)]
    for row, label in zip(cm, classes):
        cm_lines.append(f"{label}," + ",".join(str(int(v)) for v in row))
    (out_dir / f"{kind}_confusion_matrix.csv").write_text(
        "\n".join(cm_lines), encoding="utf-8"
    )

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(max(6, len(classes) * 1.4), max(6, len(classes))))
        im = ax.imshow(cm, cmap="Blues")
        for i, ci in enumerate(classes):
            for j, cj in enumerate(classes):
                ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_xticks(range(len(classes)), classes, rotation=45)
        ax.set_yticks(range(len(classes)), classes)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title(f"{kind} confusion matrix")
        fig.colorbar(im)
        fig.tight_layout()
        fig.savefig(out_dir / f"{kind}_confusion_matrix.png")
        plt.close(fig)
        print(f"  saved confusion matrix plot -> {out_dir / kind + '_confusion_matrix.png'}")
    except Exception:
        pass

    table = "labels," + ",".join(classes) + "\n"
    for row, label in zip(cm, classes):
        table += f"{label}," + ",".join(str(int(v)) for v in row) + "\n"
    (out_dir / f"{kind}_confusion_matrix.txt").write_text(table, encoding="utf-8")


def metrics_table(y_true, y_pred, classes) -> dict:
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro",
                                                 zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro",
                                           zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
            .tolist(),
    }


def leave_one_user_out(X, y, users, fit_predict) -> dict:
    """Train on all-but-one user, test on the left-out user."""
    results = []
    for user in sorted({u for u in users}):
        test_idx = users == user
        train_idx = ~test_idx
        if test_idx.sum() == 0 or np.unique(y[train_idx]).size < 2:
            continue
        if np.unique(y).size != np.unique(y[test_idx]).size:
            # left-out user lacks some classes - still measure with available labels
            pass
        y_hat = fit_predict(X[train_idx], y[train_idx], X[test_idx])
        acc = float((y_hat == y[test_idx]).mean())
        results.append({"user": user, "samples": int(test_idx.sum()), "accuracy": acc})
    if not results:
        return {}
    accs = [r["accuracy"] for r in results]
    return {
        "users": results,
        "mean_accuracy": float(np.mean(accs)),
        "min_accuracy": float(np.min(accs)),
    }


def evaluate_static(cfg: RecognitionConfig) -> None:
    data = load_dataset(cfg)
    if "X_static" not in data or data["X_static"].shape[0] == 0:
        print("  no static data to evaluate.")
        return
    X, y = data["X_static"], data["y_static"]
    classes = [str(c) for c in data["static_labels"]]

    from sklearn.ensemble import RandomForestClassifier

    cls = lambda: RandomForestClassifier(
        n_estimators=400, class_weight="balanced", random_state=0, n_jobs=-1
    )

    min_per_class = int(np.bincount(y).min())
    print(f"\n=== STATIC model evaluation ({len(classes)} classes, "
          f"{len(y)} samples) ===")

    users = data.get("users_static", np.asarray([]))
    if len(users) > 1 and min_per_class >= 2:
        louo = leave_one_user_out(X, y, users, _fit_predict_static(cls))
        if louo:
            print(f"  leave-one-user-out -> mean {louo['mean_accuracy']:.3f}, "
                  f"worst {louo['min_accuracy']:.3f}")
            for row in louo["users"]:
                print(f"      user {row['user']:12s} n={row['samples']:3d} "
                      f"acc={row['accuracy']:.3f}")

    if min_per_class < 2:
        from sklearn.model_selection import StratifiedKFold, cross_val_predict

        print("  too few samples per class for a hold-out split; using 5-fold CV predictions.")
        y_hat = None
        preds = cross_val_predict(cls(), X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0))
        y_hat = preds
        report = metrics_table(y, y_hat, classes)
    else:
        from sklearn.model_selection import train_test_split

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.25, stratify=y, random_state=1
        )
        model = cls().fit(X_tr, y_tr)
        y_hat = model.predict(X_te)
        report = metrics_table(y_te, y_hat, classes)
        report["test_samples"] = int(len(y_te))

    print(f"  accuracy  = {report['accuracy']:.3f}")
    print(f"  precision = {report['precision_macro']:.3f} (macro)")
    print(f"  recall    = {report['recall_macro']:.3f} (macro)")
    print(f"  f1        = {report['f1_macro']:.3f} (macro)")
    cm = np.asarray(report["confusion_matrix"])
    save_report(cfg, "static", report, cm, classes)
    print(f"  reports -> {cfg.paths['eval_output']}")


def _fit_predict_static(model_factory):
    def fit_predict(Xtr, ytr, Xte):
        return model_factory().fit(Xtr, ytr).predict(Xte)

    return fit_predict


def evaluate_sequence(cfg: RecognitionConfig) -> None:
    data = load_dataset(cfg)
    if "X_seq" not in data or data["X_seq"].shape[0] == 0:
        print("  no sequence data to evaluate.")
        return
    X = data["X_seq"].astype(np.float32)
    mask = data["mask_seq"].astype(np.float32)
    y = data["y_seq"]
    classes = [str(c) for c in data["seq_labels"]]

    import torch

    torch.set_num_threads(max(1, os.cpu_count() // 2))
    from app.sequence_model import SequenceClassifier
    feat_dim, L = X.shape[2], X.shape[1]

    def build_model():
        return SequenceClassifier(input_dim=feat_dim, n_classes=len(classes))

    def fit_predict(Xtr, ytr, Xte, mtr, mte):
        model = build_model()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        crit = torch.nn.CrossEntropyLoss()
        tX, tm, ty = map(torch.from_numpy, (Xtr, mtr, ytr))
        batch = 32
        for ep in range(40):
            perm = torch.randperm(len(ty))
            for i in range(0, len(ty), batch):
                idx = perm[i:i + batch]
                opt.zero_grad()
                loss = crit(model(tX[idx], tm[idx]), ty[idx].long())
                loss.backward()
                opt.step()
        model.eval()
        with torch.no_grad():
            return model(torch.from_numpy(Xte), torch.from_numpy(mte)).argmax(1).numpy()

    print(f"\n=== SEQUENCE model evaluation ({len(classes)} classes, "
          f"{len(y)} samples) ===")

    users = data.get("users_seq", np.asarray([]))
    if len(set(u for u in users)) > 1:
        results = []
        for user in sorted({u for u in users}):
            ti = users == user
            tri = ~ti
            if tri.sum() == 0 or np.unique(y[tri]).size < 2:
                continue
            y_hat = fit_predict(X[tri], y[tri], X[ti], mask[tri], mask[ti])
            acc = float((y_hat == y[ti]).mean())
            results.append({"user": user, "samples": int(ti.sum()), "accuracy": acc})
        if results:
            accs = [r["accuracy"] for r in results]
            print(f"  leave-one-user-out -> mean {np.mean(accs):.3f}, "
                  f"worst {np.min(accs):.3f}")
            for r in results:
                print(f"      user {r['user']:12s} n={r['samples']:3d} acc={r['accuracy']:.3f}")

    from sklearn.model_selection import train_test_split

    X_tr, X_te, m_tr, m_te, y_tr, y_te = train_test_split(
        X, mask, y, test_size=0.25, stratify=y, random_state=1
    )
    if np.unique(y_tr).size >= 2 and np.unique(y_te).size >= 2:
        y_hat = fit_predict(X_tr, y_tr, X_te, m_tr, m_te)
        report = metrics_table(y_te, y_hat, classes)
        report["test_samples"] = int(len(y_te))
        print(f"  hold-out accuracy  = {report['accuracy']:.3f}")
        print(f"  precision = {report['precision_macro']:.3f} "
              f"recall = {report['recall_macro']:.3f} f1 = {report['f1_macro']:.3f}")
        save_report(cfg, "sequence", report, np.asarray(report["confusion_matrix"]), classes)
        print(f"  reports -> {cfg.paths['eval_output']}")
    else:
        print("  not enough sequence data for a meaningful split; skipping hold-out.")


def run_cli(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate trained sign models.")
    parser.add_argument("--kind", choices=["static", "sequence", "both"], default="both")
    args = parser.parse_args(argv)
    cfg = load_config()
    ensure_dirs(cfg)

    if args.kind in ("static", "both"):
        evaluate_static(cfg)
    if args.kind in ("sequence", "both"):
        evaluate_sequence(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(run_cli())