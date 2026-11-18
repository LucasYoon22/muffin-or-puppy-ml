"""
a3_main.py: CSCC11 Mini Project 3 - Classification (Tasks 1-4 + optional E.1)

Pipeline
--------
1. Load the 1000 images and split the RAW images 800 / 200 (stratified).
2. Build three validation sets from the same 200 images:
       T1_clean     : original images                     (Task 1)
       T2_contrast  : contrast reduced, factor = 0.7       (Task 2)
       T3_rotation  : rotated by 30 degrees                (Task 3)
3. Extract PIXELS / HOG / CLIP features for the training set and for each
   validation set. CLIP features are cached to disk because they are slow.
4. Scale with statistics computed on the TRAINING set only
   (PIXELS: min-max, HOG: z-score, CLIP: none - already L2-normalized).
5. Tasks 1-3: the training set never changes, so each model is trained ONCE
   per hyperparameter value and scored on all three validation sets.
6. Task 4: CLIP features, training labels flipped at several ratios,
   every model retrained for each ratio, validation labels stay clean.

All tables (.csv) and plots (.png) are written to OUT_DIR.
"""
import os
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.exceptions import ConvergenceWarning
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import util_a3 as util
import util_a3_stu as util_stu

DATA_DIR = "/Users/junseoyoon/Desktop/2026_Fall/CSCC11/A3/dogfood"
OUT_DIR = "results"          # tables and plots
CACHE_DIR = "cache"          # cached CLIP features (delete if you change the split)

RANDOM_STATE = 42
TEST_SIZE = 0.2              # 800 train / 200 validation
CONTRAST_FACTOR = 0.7        # Task 2
ROTATION_ANGLE = 30          # Task 3
FLIP_RATIOS = [0, 0.05, 0.1, 0.2, 0.3, 0.5]   # Task 4 (0 = clean baseline)

FEATURES = ["PIXELS", "HOG", "CLIP"]
CLASSIFIERS = ["KNN", "RF", "LR"]
HP_NAME = {"KNN": "k", "RF": "s (min_samples_leaf)", "LR": "C"}
LOG_X = {"KNN": True, "RF": False, "LR": True}    # x-axis scale in plots

VAL_LABEL = {
    "T1_clean": "val (clean)",
    "T2_contrast": f"val (contrast {CONTRAST_FACTOR})",
    "T3_rotation": f"val (rotated {ROTATION_ANGLE} deg)",
}

# Scaling per feature. RF does not need scaling (trees split on per-feature
# thresholds, and min-max / z-score are monotone per feature), so feeding the
# scaled features to RF gives the same result and keeps the code simple.
SCALERS = {
    "PIXELS": util_stu.min_max_scaler,
    "HOG": util_stu.standard_scaler,
    "CLIP": None,
}

np.random.seed(RANDOM_STATE)
warnings.filterwarnings("ignore", category=ConvergenceWarning)


# ---------------------------- GRIDS & MODELS -------------------------------

def make_grids(n: int) -> dict[str, np.ndarray]:
    """Return the hyperparameter grids from the handout for n training samples."""
    sqrt_n = int(np.sqrt(n))
    k_values = np.unique(np.concatenate([
        np.arange(1, sqrt_n, 2),
        np.logspace(np.log10(sqrt_n), np.log10(n // 2), 20),
    ]).astype(int))
    s_values = np.arange(5, n // 2 + 1, 5)
    C_values = np.logspace(-3, 3, 20)
    return {"KNN": k_values, "RF": s_values, "LR": C_values}


def make_model(clf: str, hp: float):
    """Return an untrained classifier with hyperparameter hp."""
    if clf == "KNN":
        return KNeighborsClassifier(n_neighbors=int(hp))
    if clf == "RF":
        return RandomForestClassifier(min_samples_leaf=int(hp),
                                      random_state=RANDOM_STATE, n_jobs=-1)
    if clf == "LR":
        return LogisticRegression(C=float(hp), max_iter=1000)
    raise ValueError(f"Unknown classifier: {clf}")


# ---------------------------- FEATURES -------------------------------------

def extract_features(feature: str, X_raw: np.ndarray, cache_name: str) -> np.ndarray:
    """Return the feature matrix of X_raw for one representation."""
    if feature == "PIXELS":
        return X_raw.astype(np.float64)
    if feature == "HOG":
        return util.extract_hog(X_raw)
    if feature == "CLIP":
        path = os.path.join(CACHE_DIR, f"clip_{cache_name}_rs{RANDOM_STATE}.npy")
        if os.path.exists(path):
            return np.load(path)
        F = util.extract_clip(X_raw)
        if F is None:
            raise RuntimeError("CLIP extraction failed - check torch / clip install.")
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.save(path, F)
        return F
    raise ValueError(f"Unknown feature: {feature}")


def build_feature_sets(X_tr_raw: np.ndarray, val_raw: dict[str, np.ndarray]) -> dict:
    """Return {feature: (F_train, {val_name: F_val})}, scaled with train stats only."""
    feature_sets = {}
    for feature in FEATURES:
        print(f"Extracting {feature} features ...")
        F_tr = extract_features(feature, X_tr_raw, "train")
        F_vals = {name: extract_features(feature, Xv, name) for name, Xv in val_raw.items()}

        scaler = SCALERS[feature]
        if scaler is not None:
            # scaler(train, other) -> (train_scaled, other_scaled); other uses train stats
            F_vals = {name: scaler(F_tr, Fv)[1] for name, Fv in F_vals.items()}
            F_tr = scaler(F_tr)
        feature_sets[feature] = (F_tr, F_vals)
    return feature_sets


# ---------------------------- SWEEP ----------------------------------------

def sweep(clf: str, hp_values: np.ndarray, X_tr: np.ndarray, y_tr: np.ndarray,
          X_vals: dict[str, np.ndarray], y_val: np.ndarray) -> dict[str, np.ndarray]:
    """Train one model per hyperparameter value and score it on every val set.

    Returns {"train": acc array, <val_name>: acc array, ...}, each of len(hp_values).
    Train accuracy is measured against y_tr (the labels the model was fit on).
    """
    res = {"train": np.zeros(len(hp_values))}
    res.update({name: np.zeros(len(hp_values)) for name in X_vals})
    for i, hp in enumerate(hp_values):
        model = make_model(clf, hp).fit(X_tr, y_tr)
        res["train"][i] = accuracy_score(y_tr, model.predict(X_tr))
        for name, Xv in X_vals.items():
            res[name][i] = accuracy_score(y_val, model.predict(Xv))
    return res


# ---------------------------- TABLES ---------------------------------------

def fmt_hp(clf: str, hp: float) -> str:
    return f"{hp:.4g}" if clf == "LR" else str(int(hp))


def make_table1(results: dict, grids: dict) -> pd.DataFrame:
    """Table 1: best hyperparameter on the clean validation set."""
    rows = []
    for f in FEATURES:
        for c in CLASSIFIERS:
            r = results[f][c]
            i = int(np.argmax(r["T1_clean"]))
            rows.append({
                "Feature": f, "Classifier": c, "HP Name": HP_NAME[c][0] if c == "RF" else HP_NAME[c],
                "Best HP": fmt_hp(c, grids[c][i]),
                "Train Acc": r["train"][i], "Val Acc": r["T1_clean"][i],
            })
    return pd.DataFrame(rows)


def make_table2(results: dict, grids: dict, pert: str) -> pd.DataFrame:
    """Table 2: clean (T1) vs perturbed (T2 = pert) validation, columns as in Table 3."""
    rows = []
    for f in FEATURES:
        for c in CLASSIFIERS:
            r, hp = results[f][c], grids[c]
            i1 = int(np.argmax(r["T1_clean"]))
            i2 = int(np.argmax(r[pert]))
            rows.append({
                "Feature": f, "Classifier": c,
                "HP Name": HP_NAME[c][0] if c == "RF" else HP_NAME[c],
                "Best HP T1": fmt_hp(c, hp[i1]),
                "Val Acc T1": r["T1_clean"][i1],
                "Val Acc T2 @HP T1": r[pert][i1],
                "Best HP T2": fmt_hp(c, hp[i2]),
                "Train Acc @HP T2": r["train"][i2],
                "Val Acc T2 @HP T2": r[pert][i2],
                "HP Shift": fmt_hp(c, hp[i2] - hp[i1]) if c != "LR" else f"{hp[i2] - hp[i1]:.4g}",
                "Best Acc. Chg": r[pert][i2] - r["T1_clean"][i1],
                "Acc. Chg": r[pert][i2] - r[pert][i1],
            })
    return pd.DataFrame(rows)


def save_table(df: pd.DataFrame, name: str) -> None:
    """Print a table and save it as CSV."""
    path = os.path.join(OUT_DIR, f"{name}.csv")
    df.to_csv(path, index=False, float_format="%.4f")
    print(f"\n===== {name} =====")
    print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"(saved to {path})")


# ---------------------------- PLOTS ----------------------------------------

def plot_3x3(results: dict, grids: dict, val_keys: list[str], title: str, fname: str) -> None:
    """3x3 grid (rows = features, cols = classifiers) of train/val accuracy vs HP.
    A dotted vertical line marks the best HP for each validation curve."""
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    for i, f in enumerate(FEATURES):
        for j, c in enumerate(CLASSIFIERS):
            ax, r, x = axes[i, j], results[f][c], grids[c]
            ax.plot(x, r["train"], label="train", color="black", lw=1.5)
            for v in val_keys:
                line, = ax.plot(x, r[v], label=VAL_LABEL[v], lw=1.5)
                ax.axvline(x[np.argmax(r[v])], ls=":", color=line.get_color())
            if LOG_X[c]:
                ax.set_xscale("log")
            ax.set_title(f"{f} + {c}")
            ax.set_xlabel(HP_NAME[c])
            ax.set_ylabel("accuracy")
            ax.grid(alpha=0.3)
    axes[0, 0].legend(loc="lower left", fontsize=9)
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, fname)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"(plot saved to {path})")


def plot_task4_curves(res4: dict, grids: dict) -> None:
    """Q4.1: one subplot per classifier, one validation curve per flip ratio."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    cmap = plt.get_cmap("viridis")
    for j, c in enumerate(CLASSIFIERS):
        ax, x = axes[j], grids[c]
        for k, ratio in enumerate(FLIP_RATIOS):
            ax.plot(x, res4[c][ratio]["val"], color=cmap(k / (len(FLIP_RATIOS) - 1)),
                    label=f"flip={ratio}", lw=1.5)
        if LOG_X[c]:
            ax.set_xscale("log")
        ax.set_title(f"CLIP + {c}: clean val accuracy")
        ax.set_xlabel(HP_NAME[c])
        ax.set_ylabel("accuracy")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "task4_val_curves.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)

    # train curves too (the handout asks for train + val curves in every task)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for j, c in enumerate(CLASSIFIERS):
        ax, x = axes[j], grids[c]
        for k, ratio in enumerate(FLIP_RATIOS):
            ax.plot(x, res4[c][ratio]["train"], color=cmap(k / (len(FLIP_RATIOS) - 1)),
                    label=f"flip={ratio}", lw=1.5)
        if LOG_X[c]:
            ax.set_xscale("log")
        ax.set_title(f"CLIP + {c}: train accuracy (on noisy labels)")
        ax.set_xlabel(HP_NAME[c])
        ax.set_ylabel("accuracy")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    path2 = os.path.join(OUT_DIR, "task4_train_curves.png")
    fig.savefig(path2, dpi=150)
    plt.close(fig)
    print(f"(plots saved to {path} and {path2})")


def plot_task4_best_hp(table4: pd.DataFrame) -> None:
    """Q4.2: optimal hyperparameter vs flip ratio, one subplot per classifier."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for j, c in enumerate(CLASSIFIERS):
        sub = table4[table4["Classifier"] == c]
        ax = axes[j]
        ax.plot(sub["Flip Ratio"], sub["Best HP value"], marker="o")
        if c == "LR":
            ax.set_yscale("log")
        ax.set_title(f"CLIP + {c}")
        ax.set_xlabel("flip ratio")
        ax.set_ylabel(f"best {HP_NAME[c]}")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "task4_best_hp_vs_flip.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"(plot saved to {path})")


# ---------------------------- TASK 4 ---------------------------------------

def run_task4(F_tr: np.ndarray, F_val: np.ndarray, y_tr: np.ndarray,
              y_val: np.ndarray, grids: dict) -> tuple[dict, pd.DataFrame]:
    """Retrain every model on CLIP features with flipped training labels."""
    res4 = {c: {} for c in CLASSIFIERS}
    rows = []
    for ratio in FLIP_RATIOS:
        y_noisy = y_tr.copy() if ratio == 0 else util.flip_labels(y_tr, ratio, RANDOM_STATE)
        for c in CLASSIFIERS:
            print(f"[Task 4] flip={ratio} CLIP + {c}")
            r = sweep(c, grids[c], F_tr, y_noisy, {"val": F_val}, y_val)
            res4[c][ratio] = r
            i = int(np.argmax(r["val"]))
            rows.append({
                "Classifier": c, "Flip Ratio": ratio,
                "Best HP": fmt_hp(c, grids[c][i]), "Best HP value": grids[c][i],
                "Train Acc (noisy)": r["train"][i], "Val Acc (clean)": r["val"][i],
            })
    return res4, pd.DataFrame(rows)


# ---------------------------- EXTENSION E.1 --------------------------------

def extension_e1(results: dict, grids: dict, n_val: int) -> None:
    """Count CLIP hyperparameters within 1 or 2 validation images of the best."""
    print("\n===== Extension E.1 (CLIP, clean val) =====")
    for c in CLASSIFIERS:
        acc = results["CLIP"][c]["T1_clean"]
        best = acc.max()
        for n_img in (1, 2):
            tol = n_img / n_val + 1e-9
            close = grids[c][acc >= best - tol]
            print(f"{c}: best={best:.3f}; {len(close)} values within {n_img} image(s): "
                  f"{[fmt_hp(c, h) for h in close]}")


# ---------------------------- MAIN -----------------------------------------

def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1. load + split raw images
    X, y = util.load_dataset_muffin_puppy(DATA_DIR)
    X_tr_raw, X_val_raw, y_tr, y_val = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
    print(f"train: {len(y_tr)} {np.bincount(y_tr)}, val: {len(y_val)} {np.bincount(y_val)}")

    # 2. three validation sets (perturb validation images only)
    val_raw = {
        "T1_clean": X_val_raw,
        "T2_contrast": util.reduce_contrast(X_val_raw, factor=CONTRAST_FACTOR),
        "T3_rotation": util.add_rotation(X_val_raw, angle=ROTATION_ANGLE),
    }

    # 3-4. features + scaling
    grids = make_grids(len(y_tr))
    feature_sets = build_feature_sets(X_tr_raw, val_raw)

    # 5. Tasks 1-3: train once per HP, score on all three validation sets
    results = {}
    for f in FEATURES:
        F_tr, F_vals = feature_sets[f]
        results[f] = {}
        for c in CLASSIFIERS:
            print(f"[Tasks 1-3] {f} + {c} ({len(grids[c])} values)")
            results[f][c] = sweep(c, grids[c], F_tr, y_tr, F_vals, y_val)

    with open(os.path.join(OUT_DIR, "results_task1_3.pkl"), "wb") as fh:
        pickle.dump({"results": results, "grids": grids}, fh)

    # Task 1
    save_table(make_table1(results, grids), "table1_task1_clean")
    plot_3x3(results, grids, ["T1_clean"],
             "Task 1: train / clean validation accuracy", "task1_curves.png")

    # Task 2
    save_table(make_table2(results, grids, "T2_contrast"), "table2_task2_contrast")
    plot_3x3(results, grids, ["T1_clean", "T2_contrast"],
             f"Task 2: clean vs contrast-reduced ({CONTRAST_FACTOR}) validation",
             "task2_curves.png")

    # Task 3
    save_table(make_table2(results, grids, "T3_rotation"), "table2_task3_rotation")
    plot_3x3(results, grids, ["T1_clean", "T3_rotation"],
             f"Task 3: clean vs rotated ({ROTATION_ANGLE} deg) validation",
             "task3_curves.png")

    # 6. Task 4: CLIP + noisy training labels
    F_tr_clip, F_vals_clip = feature_sets["CLIP"]
    res4, table4 = run_task4(F_tr_clip, F_vals_clip["T1_clean"], y_tr, y_val, grids)
    save_table(table4.drop(columns="Best HP value"), "table4_task4_label_noise")
    plot_task4_curves(res4, grids)
    plot_task4_best_hp(table4)
    with open(os.path.join(OUT_DIR, "results_task4.pkl"), "wb") as fh:
        pickle.dump(res4, fh)

    # optional extension
    extension_e1(results, grids, len(y_val))


if __name__ == "__main__":
    main()
