"""
train_models.py  (part 1: linear baselines)
===========================================
Train simple ML models on the TTM data set (data/data.csv, made by
generate_data.py) and measure how well they predict unseen inputs.

Inputs (features): x1 = log10(F_abs), x2 = log10(tp)
Targets:
  - Tl_peak (K): peak surface lattice temperature   -> regression
  - melted (0/1): lattice reached Tm somewhere       -> classification

Workflow
--------
1. Split the data once into 80 % training / 20 % test (fixed seed,
   stratified on melted so both parts keep the same ~21 % melted fraction).
   The test set is used ONLY for the final scores, never for fitting.
2. Regression baseline: linear regression, Tl_peak ~ w0 + w1 x1 + w2 x2.
   Scores: MAE (K) and R^2 on training and test sets; parity plot.
3. Classification baseline: logistic regression on standardised inputs.
   Scores on the test set: confusion matrix, accuracy, recall (melted).

Run with:  conda run -n ttm-ml python train_models.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                 # write PNG files only, never open windows
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (mean_absolute_error, r2_score, confusion_matrix,
                             accuracy_score, recall_score)

from ttm_solver import GOLD


# ---------------------------------------------------------------------------
# 1. Settings
# ---------------------------------------------------------------------------
CSV_PATH = "data/data.csv"
TEST_FRACTION = 0.2       # 20 % of the samples are held back for testing
SEED = 42                 # fixed seed -> the same split on every run
PARITY_PLOT = "parity_linear.png"


# ---------------------------------------------------------------------------
# 2. Load the data and build the inputs
# ---------------------------------------------------------------------------
def load_data(path):
    """Return X (n x 2: log10 F_abs, log10 tp), y_reg (Tl_peak), y_cls (melted)."""
    df = pd.read_csv(path)
    X = np.column_stack([np.log10(df["F_abs"]), np.log10(df["tp"])])
    return X, df["Tl_peak"].to_numpy(), df["melted"].to_numpy()


# ---------------------------------------------------------------------------
# 3. Regression baseline: linear regression on Tl_peak
# ---------------------------------------------------------------------------
def fit_linear_regression(X_train, X_test, y_train, y_test):
    # Least-squares fit of a plane: Tl_peak = w0 + w1*log10(F) + w2*log10(tp).
    # No input scaling needed: without a penalty term, scaling the inputs
    # rescales the weights but leaves the predictions unchanged.
    model = LinearRegression().fit(X_train, y_train)
    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)

    print("\n=== Linear regression: Tl_peak ===")
    print(f"Fitted: Tl_peak = {model.intercept_:.1f} "
          f"{model.coef_[0]:+.1f} * log10(F_abs) {model.coef_[1]:+.1f} * log10(tp)   [K]")
    scores = {}
    for name, y, p in [("training", y_train, pred_train), ("test", y_test, pred_test)]:
        mae = mean_absolute_error(y, p)     # average |prediction - truth|, K
        r2 = r2_score(y, p)                 # fraction of variance explained
        scores[name] = (mae, r2)
        print(f"  {name:8s} set ({len(y):3d} samples): MAE = {mae:6.1f} K,  R^2 = {r2:.4f}")
    return model, pred_train, pred_test, scores


def parity_plot(y_train, pred_train, y_test, pred_test, scores, path):
    """Predicted vs true Tl_peak; perfect predictions lie on the diagonal."""
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(y_train, pred_train, s=14, alpha=0.5, color="tab:blue",
               label=f"training (MAE {scores['training'][0]:.0f} K, "
                     f"$R^2$ {scores['training'][1]:.3f})")
    ax.scatter(y_test, pred_test, s=22, color="tab:orange", edgecolor="k", lw=0.4,
               label=f"test (MAE {scores['test'][0]:.0f} K, "
                     f"$R^2$ {scores['test'][1]:.3f})")

    lo = min(y_train.min(), pred_train.min(), y_test.min(), pred_test.min())
    hi = max(y_train.max(), pred_train.max(), y_test.max(), pred_test.max())
    pad = 0.05 * (hi - lo)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=1, label="perfect prediction")
    # Melting temperature: points right of the vertical line truly melted,
    # points above the horizontal line are predicted to have melted.
    ax.axvline(GOLD["Tm"], color="gray", ls=":", lw=1)
    ax.axhline(GOLD["Tm"], color="gray", ls=":", lw=1)
    ax.text(GOLD["Tm"], lo - pad, " $T_m$", color="gray", va="bottom")

    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_aspect("equal")
    ax.set_xlabel("True peak $T_l$ from TTM solver (K)")
    ax.set_ylabel("Predicted peak $T_l$ (K)")
    ax.set_title("Linear regression on log10($F_{abs}$), log10($t_p$)")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"  Parity plot saved to {path}")


# ---------------------------------------------------------------------------
# 4. Classification baseline: logistic regression on melted
# ---------------------------------------------------------------------------
def fit_logistic_regression(X_train, X_test, y_train, y_test):
    # StandardScaler: shift/scale each input to mean 0, std 1 using the
    # TRAINING data only (the test set must not influence the model).
    # This matters because LogisticRegression penalises large weights
    # (regularisation), which is only fair if the inputs have similar scales.
    model = make_pipeline(StandardScaler(), LogisticRegression())
    model.fit(X_train, y_train)
    pred_test = model.predict(X_test)        # 1 if probability(melted) >= 0.5

    cm = confusion_matrix(y_test, pred_test, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    accuracy = accuracy_score(y_test, pred_test)
    recall = recall_score(y_test, pred_test, pos_label=1)   # TP / (TP + FN)

    # What an "always predict not melted" model would score, for comparison
    trivial_accuracy = np.mean(y_test == 0)

    print("\n=== Logistic regression: melted (test set) ===")
    print(f"Test set: {len(y_test)} samples, {y_test.sum()} melted "
          f"({y_test.mean() * 100:.1f} %)")
    print("Confusion matrix (rows = truth, columns = prediction):")
    print("                      pred. not melted   pred. melted")
    print(f"  truly not melted    {tn:16d}   {fp:12d}   (false alarms: {fp})")
    print(f"  truly melted        {fn:16d}   {tp:12d}   (missed melting: {fn})")
    print(f"Accuracy           : {accuracy * 100:5.1f} %")
    print(f"Recall (melted)    : {recall * 100:5.1f} %   ({tp} of {tp + fn} melted cases caught)")
    print(f"For comparison, always predicting 'not melted' gives "
          f"{trivial_accuracy * 100:.1f} % accuracy and 0 % recall.")

    # The decision boundary is the line where probability = 0.5. Convert the
    # fitted weights back to physical inputs and report the threshold
    # fluence at a few pulse durations.
    scaler, clf = model.named_steps["standardscaler"], model.named_steps["logisticregression"]
    w = clf.coef_[0] / scaler.scale_                        # weights on raw log inputs
    b = clf.intercept_[0] - np.sum(w * scaler.mean_)
    print("Predicted melting threshold (probability = 0.5):")
    for tp_s in (50e-15, 100e-15, 1e-12, 2e-12):
        logF = -(b + w[1] * np.log10(tp_s)) / w[0]
        print(f"  tp = {tp_s * 1e15:6.0f} fs: F_abs = {10 ** logF:6.0f} J/m^2 "
              f"({10 ** logF / 10:.0f} mJ/cm^2)")
    return model


# ---------------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    X, y_reg, y_cls = load_data(CSV_PATH)

    # One split for both tasks, stratified on melted.
    (X_train, X_test, yr_train, yr_test,
     yc_train, yc_test) = train_test_split(X, y_reg, y_cls,
                                           test_size=TEST_FRACTION,
                                           random_state=SEED, stratify=y_cls)
    print(f"Data: {len(X)} samples -> {len(X_train)} training, {len(X_test)} test")
    print(f"Melted fraction: training {yc_train.mean() * 100:.1f} %, "
          f"test {yc_test.mean() * 100:.1f} %")

    _, p_train, p_test, scores = fit_linear_regression(X_train, X_test, yr_train, yr_test)
    parity_plot(yr_train, p_train, yr_test, p_test, scores, PARITY_PLOT)

    fit_logistic_regression(X_train, X_test, yc_train, yc_test)
