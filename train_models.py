"""
train_models.py
===============
Train ML models on the TTM data set (data/data.csv, made by generate_data.py)
and measure how well they predict unseen inputs.

Inputs (features): x1 = log10(F_abs), x2 = log10(tp)
Targets:
  - Tl_peak (K): peak surface lattice temperature   -> regression
  - melted (0/1): lattice reached Tm somewhere       -> classification

Workflow
--------
1. Split the data once into 80 % training / 20 % test (fixed seed,
   stratified on melted so both parts keep the same ~21 % melted fraction).
   The test set is used ONLY for the final scores, never for fitting or for
   choosing settings.
2. Part 1, baselines:
   - linear regression on Tl_peak;
   - logistic regression on melted (default C = 1).
3. Part 2, tuned models. Settings are chosen by 5-fold cross-validation
   (GridSearchCV) on the TRAINING set only:
   - random forest regressor on Tl_peak   (max_depth, min_samples_leaf),
     CV score: mean absolute error;
   - logistic regression on melted         (C),
   - random forest classifier on melted    (max_depth, min_samples_leaf),
     CV score: balanced accuracy = mean of the recalls of both classes.
4. Part 3:
   - physics-informed model: linear regression on log10(Tl_peak - 300 K),
     i.e. a power law Tl_peak - 300 K = A * F_abs^a * tp^b; predictions
     are converted back with Tl = 300 + 10**z, so always above 300 K;
   - extrapolation test: train linear, physics-informed, random forest and
     hybrid on F_abs < 1500 J/m^2 only, test on F_abs >= 1500 J/m^2.
   Part 4, hybrid (residual learning): a random forest (settings by CV)
   learns the residuals Tl_peak - power-law prediction; the hybrid
   predicts power law + forest residual.
5. One comparison table of all models; parity plots (parity_linear.png,
   parity_physics.png, parity_rf.png); extrapolation.png; feature
   importances; predictions for the two samples closest to the melting
   boundary.

Run with:  conda run -n ttm-ml python train_models.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                 # write PNG files only, never open windows
import matplotlib.pyplot as plt
from sklearn.model_selection import (train_test_split, GridSearchCV,
                                     KFold, StratifiedKFold, cross_val_score)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.compose import TransformedTargetRegressor
from sklearn.base import BaseEstimator, RegressorMixin
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
SEED = 42                 # fixed seed -> same split, folds and forests every run
N_FOLDS = 5               # k in k-fold cross-validation
N_TREES = 300             # trees per forest (more never hurts accuracy, only time)
FEATURES = ["log10(F_abs)", "log10(tp)"]

# Candidate settings tried by cross-validation
FOREST_GRID = {"max_depth": [3, 5, 8, 12, None],          # None = grow until leaves are pure
               "min_samples_leaf": [1, 2, 5, 10, 20]}
LOGISTIC_GRID = {"logisticregression__C": [0.01, 0.1, 1, 10, 100, 1e3, 1e4, 1e5]}

# Samples on either side of the melting boundary (found in part 1)
BOUNDARY_CASES = [(1231.0, 1082e-15), (1278.0, 69e-15)]   # (F_abs J/m^2, tp s)

T0 = 300.0                # initial temperature of every solver run, K
F_SPLIT = 1500.0          # extrapolation test: train below, test at/above (J/m^2)


# ---------------------------------------------------------------------------
# 2. Load the data and build the inputs
# ---------------------------------------------------------------------------
def load_data(path):
    """Return the table, X (n x 2: log10 F_abs, log10 tp), Tl_peak, melted."""
    df = pd.read_csv(path)
    X = np.column_stack([np.log10(df["F_abs"]), np.log10(df["tp"])])
    return df, X, df["Tl_peak"].to_numpy(), df["melted"].to_numpy()


# ---------------------------------------------------------------------------
# 3. Regression models for Tl_peak
# ---------------------------------------------------------------------------
def regression_scores(name, settings, model, cv_mae, X_train, X_test, y_train, y_test):
    """Collect the numbers for one row of the regression table."""
    pred_train, pred_test = model.predict(X_train), model.predict(X_test)
    return {"model": name, "settings": settings, "cv_mae": cv_mae,
            "train_mae": mean_absolute_error(y_train, pred_train),
            "test_mae": mean_absolute_error(y_test, pred_test),
            "train_r2": r2_score(y_train, pred_train),
            "test_r2": r2_score(y_test, pred_test),
            "pred_train": pred_train, "pred_test": pred_test}


def fit_linear_regression(X_train, X_test, y_train, y_test, cv):
    # Least-squares fit of a plane: Tl_peak = w0 + w1*log10(F) + w2*log10(tp).
    # Nothing to tune, but its CV score is computed so it can be compared
    # with the forest on the same footing.
    model = LinearRegression().fit(X_train, y_train)
    cv_mae = -cross_val_score(LinearRegression(), X_train, y_train, cv=cv,
                              scoring="neg_mean_absolute_error").mean()
    print(f"Linear regression: Tl_peak = {model.intercept_:.1f} "
          f"{model.coef_[0]:+.1f} * log10(F_abs) {model.coef_[1]:+.1f} * log10(tp)  [K]")
    return regression_scores("Linear regression", "-", model, cv_mae,
                             X_train, X_test, y_train, y_test)


def fit_forest_regressor(X_train, X_test, y_train, y_test, cv):
    # GridSearchCV: for every combination in FOREST_GRID, train on 4 folds of
    # the training set, score MAE on the 5th, repeat for all 5 folds, average.
    # The best combination is then refitted on the whole training set.
    search = GridSearchCV(RandomForestRegressor(n_estimators=N_TREES, random_state=SEED),
                          FOREST_GRID, cv=cv, scoring="neg_mean_absolute_error",
                          n_jobs=-1)
    search.fit(X_train, y_train)
    p = search.best_params_
    settings = f"max_depth={p['max_depth']}, min_samples_leaf={p['min_samples_leaf']}"
    row = regression_scores("Random forest", settings, search.best_estimator_,
                            -search.best_score_, X_train, X_test, y_train, y_test)
    row["estimator"] = search.best_estimator_
    return row


def to_log_rise(T):
    """Tl_peak (K) -> log10 of the temperature rise above T0."""
    return np.log10(T - T0)


def from_log_rise(z):
    """log10(temperature rise) -> Tl_peak (K). Always > T0, since 10**z > 0."""
    return T0 + 10.0 ** z


def make_physics_model():
    # Power law  Tl_peak - T0 = A * F_abs^a * tp^b  becomes linear after
    # taking log10:  log10(Tl_peak - T0) = log10(A) + a*log10(F) + b*log10(tp).
    # TransformedTargetRegressor applies to_log_rise before fitting and
    # from_log_rise after predicting, so predictions and scores are in K.
    return TransformedTargetRegressor(regressor=LinearRegression(),
                                      func=to_log_rise, inverse_func=from_log_rise)


def fit_physics_regression(X_train, X_test, y_train, y_test, cv):
    model = make_physics_model().fit(X_train, y_train)
    cv_mae = -cross_val_score(make_physics_model(), X_train, y_train, cv=cv,
                              scoring="neg_mean_absolute_error").mean()
    a, b = model.regressor_.coef_            # exponents of F_abs and tp
    # Prefactor quoted at reference values F = 1000 J/m^2, tp = 1 ps, so the
    # number is readable (with tp in seconds it would be ~10^-12 scaled)
    dT_ref = 10.0 ** (model.regressor_.intercept_ + a * 3.0 + b * (-12.0))
    print(f"Physics-informed: Tl_peak - {T0:.0f} K = {dT_ref:.0f} K "
          f"* (F_abs / 1000 J/m^2)^{a:.3f} * (tp / 1 ps)^{b:.3f}")
    row = regression_scores("Physics-informed", f"power law: F^{a:.3f}, tp^{b:.3f}",
                            model, cv_mae, X_train, X_test, y_train, y_test)
    row["exponents"] = (a, b)
    return row


class HybridRegressor(RegressorMixin, BaseEstimator):
    """
    Residual learning: physics-informed power law + random forest correction.

    fit:     1. fit the power law to (X, y);
             2. residuals r = y - power-law prediction (what the physics misses);
             3. fit a random forest to (X, r).
    predict: power-law prediction + forest-predicted residual.

    Written as one scikit-learn estimator so that GridSearchCV refits BOTH
    stages on every training fold; fitting the power law once on all data
    would leak the validation folds into the residuals.
    """

    def __init__(self, max_depth=None, min_samples_leaf=1,
                 n_estimators=N_TREES, random_state=SEED):
        # scikit-learn convention: __init__ only stores the settings
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.n_estimators = n_estimators
        self.random_state = random_state

    def fit(self, X, y):
        self.physics_ = make_physics_model().fit(X, y)
        residual = y - self.physics_.predict(X)
        self.forest_ = RandomForestRegressor(n_estimators=self.n_estimators,
                                             max_depth=self.max_depth,
                                             min_samples_leaf=self.min_samples_leaf,
                                             random_state=self.random_state)
        self.forest_.fit(X, residual)
        return self

    def predict(self, X):
        return self.physics_.predict(X) + self.forest_.predict(X)


def fit_hybrid_regression(X_train, X_test, y_train, y_test, cv):
    # Same grid as the plain forest, but the forest now learns the residuals
    search = GridSearchCV(HybridRegressor(), FOREST_GRID, cv=cv,
                          scoring="neg_mean_absolute_error", n_jobs=-1)
    search.fit(X_train, y_train)
    p = search.best_params_
    best = search.best_estimator_
    residual = y_train - best.physics_.predict(X_train)
    print(f"Hybrid: power-law residuals on training set: mean |r| {np.abs(residual).mean():.1f} K, "
          f"range {residual.min():+.0f} to {residual.max():+.0f} K")
    settings = f"max_depth={p['max_depth']}, min_samples_leaf={p['min_samples_leaf']}"
    return regression_scores("Hybrid (power+RF)", settings, best, -search.best_score_,
                             X_train, X_test, y_train, y_test)


def parity_plot(row, y_train, y_test, path):
    """Predicted vs true Tl_peak; perfect predictions lie on the diagonal."""
    pred_train, pred_test = row["pred_train"], row["pred_test"]
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(y_train, pred_train, s=14, alpha=0.5, color="tab:blue",
               label=f"training (MAE {row['train_mae']:.0f} K, $R^2$ {row['train_r2']:.3f})")
    ax.scatter(y_test, pred_test, s=22, color="tab:orange", edgecolor="k", lw=0.4,
               label=f"test (MAE {row['test_mae']:.0f} K, $R^2$ {row['test_r2']:.3f})")

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
    ax.set_title(f"{row['model']} on log10($F_{{abs}}$), log10($t_p$)")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"Parity plot saved to {path}")


# ---------------------------------------------------------------------------
# 4. Classification models for melted
# ---------------------------------------------------------------------------
def classification_scores(name, settings, model, cv_bal_acc, X_test, y_test):
    """Collect the numbers for one row of the classification table."""
    pred = model.predict(X_test)                 # 1 if probability(melted) >= 0.5
    tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
    return {"model": name, "settings": settings, "cv_bal_acc": cv_bal_acc,
            "tn": tn, "fp": fp, "fn": fn, "tp": tp,
            "accuracy": accuracy_score(y_test, pred),
            "recall": recall_score(y_test, pred, pos_label=1),   # TP / (TP + FN)
            "estimator": model}


def logistic_threshold(model, tp_s):
    """Fluence (J/m^2) where the logistic model gives probability 0.5."""
    scaler = model.named_steps["standardscaler"]
    clf = model.named_steps["logisticregression"]
    w = clf.coef_[0] / scaler.scale_                 # weights on raw log inputs
    b = clf.intercept_[0] - np.sum(w * scaler.mean_)
    return 10.0 ** (-(b + w[1] * np.log10(tp_s)) / w[0])


def fit_logistic(X_train, X_test, y_train, y_test, cv):
    # StandardScaler: shift/scale each input to mean 0, std 1. Inside a
    # pipeline it is refitted on the training folds only, so neither the CV
    # validation fold nor the test set influences the scaling.
    # C is the inverse regularisation strength: small C = strong penalty on
    # large weights = smoother, more gradual probability curve.
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=10000))

    # Part 1 baseline: default C = 1 (CV score computed for comparison only)
    base = pipe.fit(X_train, y_train)
    base_cv = cross_val_score(make_pipeline(StandardScaler(), LogisticRegression(max_iter=10000)),
                              X_train, y_train, cv=cv, scoring="balanced_accuracy").mean()
    rows = [classification_scores("Logistic regression", "C=1 (default)", base,
                                  base_cv, X_test, y_test)]

    # Part 2: choose C by cross-validation. When several C values tie for the
    # best score, GridSearchCV keeps the first one in the list, i.e. the
    # smallest C (strongest regularisation) that reaches the best score.
    search = GridSearchCV(make_pipeline(StandardScaler(), LogisticRegression(max_iter=10000)),
                          LOGISTIC_GRID, cv=cv, scoring="balanced_accuracy", n_jobs=-1)
    search.fit(X_train, y_train)
    C = search.best_params_["logisticregression__C"]
    rows.append(classification_scores("Logistic regression", f"C={C:g} (CV)",
                                      search.best_estimator_, search.best_score_,
                                      X_test, y_test))
    return rows


def fit_forest_classifier(X_train, X_test, y_train, y_test, cv):
    search = GridSearchCV(RandomForestClassifier(n_estimators=N_TREES, random_state=SEED),
                          FOREST_GRID, cv=cv, scoring="balanced_accuracy", n_jobs=-1)
    search.fit(X_train, y_train)
    p = search.best_params_
    settings = f"max_depth={p['max_depth']}, min_samples_leaf={p['min_samples_leaf']}"
    return classification_scores("Random forest", settings, search.best_estimator_,
                                 search.best_score_, X_test, y_test)


# ---------------------------------------------------------------------------
# 5. Reporting
# ---------------------------------------------------------------------------
def print_regression_table(rows):
    print("\n=== Tl_peak (regression): MAE in K ===")
    print(f"| {'Model':19s} | {'Settings (chosen by CV)':36s} | {'CV MAE':>6s} "
          f"| {'Train MAE':>9s} | {'Test MAE':>8s} | {'Train R^2':>9s} | {'Test R^2':>8s} |")
    print("|" + "---|" * 7)
    for r in rows:
        print(f"| {r['model']:19s} | {r['settings']:36s} | {r['cv_mae']:6.1f} "
              f"| {r['train_mae']:9.1f} | {r['test_mae']:8.1f} "
              f"| {r['train_r2']:9.4f} | {r['test_r2']:8.4f} |")


def print_classification_table(rows, y_test):
    print(f"\n=== melted (classification), test set: {len(y_test)} samples, "
          f"{y_test.sum()} melted ===")
    print("Confusion matrix as TN / FP / FN / TP "
          "(FP = false alarm, FN = missed melting)")
    print(f"| {'Model':19s} | {'Settings':36s} | {'CV bal. acc.':>12s} "
          f"| {'TN / FP / FN / TP':>17s} | {'Accuracy':>8s} | {'Recall':>6s} |")
    print("|" + "---|" * 6)
    for r in rows:
        cm = f"{r['tn']} / {r['fp']} / {r['fn']} / {r['tp']}"
        print(f"| {r['model']:19s} | {r['settings']:36s} | {r['cv_bal_acc'] * 100:11.1f} % "
              f"| {cm:>17s} | {r['accuracy'] * 100:7.1f} % | {r['recall'] * 100:5.1f} % |")
    print(f"For comparison, always predicting 'not melted' gives "
          f"{np.mean(y_test == 0) * 100:.1f} % accuracy, 0 % recall, "
          f"50 % balanced accuracy.")


def print_feature_importances(forest, target):
    # Impurity-based importance: the share of the total error reduction
    # achieved by each input's split questions, summed over all trees.
    # It is a SHARE: an input with a real but small effect (tp changes
    # Tl_peak by ~20 K, F by ~2000 K) gets a tiny value, not zero.
    print(f"Feature importances, random forest for {target}: "
          + ", ".join(f"{f} {v:.4f}" for f, v in zip(FEATURES, forest.feature_importances_)))


def check_boundary_cases(df, idx_train, idx_test, X, reg_forest, cls_rows):
    """Report where the near-boundary samples ended up and their predictions."""
    print("\n=== Samples closest to the melting boundary ===")
    for F_abs, tp in BOUNDARY_CASES:
        # Locate the sample in the table (values in the CSV are not rounded)
        i = int(np.argmin(np.abs(df["F_abs"] - F_abs) / F_abs + np.abs(df["tp"] - tp) / tp))
        where = "TEST" if i in set(idx_test) else "TRAINING"
        print(f"F_abs = {df['F_abs'][i]:.1f} J/m^2, tp = {df['tp'][i] * 1e15:.0f} fs: "
              f"true Tl_peak {df['Tl_peak'][i]:.0f} K, melted {df['melted'][i]} "
              f"-> in the {where} set")
        x = X[i:i + 1]
        preds = [f"RF Tl_peak {reg_forest.predict(x)[0]:.0f} K"]
        for r in cls_rows:
            p_melt = r["estimator"].predict_proba(x)[0, 1]
            preds.append(f"{r['model']} {r['settings']}: P(melted) {p_melt:.2f}")
        note = ("" if where == "TEST" else
                "  (training sample: the model has seen it, so this is not a fair test)")
        print("   " + "; ".join(preds) + note)


# ---------------------------------------------------------------------------
# 6. Extrapolation test: train at low fluence, predict at high fluence
# ---------------------------------------------------------------------------
def extrapolation_test(df, X, y, path):
    """
    Train on all runs with F_abs < F_SPLIT and test on all runs with
    F_abs >= F_SPLIT. This is separate from the 80/20 split: here the test
    inputs lie OUTSIDE the training range, which a random split never does.

    A random forest predicts averages of training targets, so it can never
    predict a Tl_peak above the largest one it was trained on. The linear
    and physics-informed models are formulas and continue beyond the data.
    The hybrid's forest also flattens, but only its correction to the
    power law, not the temperature itself.
    """
    F = df["F_abs"].to_numpy()
    low, high = F < F_SPLIT, F >= F_SPLIT
    X_low, y_low, X_high, y_high = X[low], y[low], X[high], y[high]

    # Forest settings re-chosen by CV on the low-fluence runs only, so no
    # high-fluence information reaches any model
    cv = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    search = GridSearchCV(RandomForestRegressor(n_estimators=N_TREES, random_state=SEED),
                          FOREST_GRID, cv=cv, scoring="neg_mean_absolute_error", n_jobs=-1)
    search.fit(X_low, y_low)
    p = search.best_params_
    hybrid = GridSearchCV(HybridRegressor(), FOREST_GRID, cv=cv,
                          scoring="neg_mean_absolute_error", n_jobs=-1)
    hybrid.fit(X_low, y_low)
    ph = hybrid.best_params_
    models = [("Linear regression", LinearRegression().fit(X_low, y_low)),
              ("Physics-informed", make_physics_model().fit(X_low, y_low)),
              (f"Random forest (max_depth={p['max_depth']}, "
               f"min_samples_leaf={p['min_samples_leaf']})", search.best_estimator_),
              (f"Hybrid (max_depth={ph['max_depth']}, "
               f"min_samples_leaf={ph['min_samples_leaf']})", hybrid.best_estimator_)]

    print(f"\n=== Extrapolation test: train on F_abs < {F_SPLIT:.0f} J/m^2 "
          f"({low.sum()} runs, Tl_peak up to {y_low.max():.0f} K), "
          f"test on F_abs >= {F_SPLIT:.0f} ({high.sum()} runs, "
          f"Tl_peak {y_high.min():.0f}-{y_high.max():.0f} K) ===")
    print(f"| {'Model':52s} | {'MAE, training range':>19s} | "
          f"{'MAE, extrapolation':>18s} | {'Max predicted Tl':>16s} |")
    print("|" + "---|" * 4)

    fig, axes = plt.subplots(1, len(models), figsize=(4.3 * len(models), 4.6), sharey=True)
    for ax, (name, model) in zip(axes, models):
        pred_low, pred_high = model.predict(X_low), model.predict(X_high)
        mae_low = mean_absolute_error(y_low, pred_low)
        mae_high = mean_absolute_error(y_high, pred_high)
        print(f"| {name:52s} | {mae_low:17.1f} K | {mae_high:16.1f} K | "
              f"{pred_high.max():14.0f} K |")

        ax.axvspan(F_SPLIT, F.max() * 1.05, color="tab:red", alpha=0.07, lw=0)
        ax.axvline(F_SPLIT, color="tab:red", ls="--", lw=1)
        ax.axhline(GOLD["Tm"], color="gray", ls=":", lw=1)
        ax.scatter(F, y, s=10, color="0.65", label="TTM solver (true)")
        ax.scatter(F[low], pred_low, s=8, color="tab:blue",
                   label="prediction, training range")
        ax.scatter(F[high], pred_high, s=14, color="tab:red", edgecolor="k", lw=0.3,
                   label="prediction, extrapolation")
        ax.set_xscale("log")
        ax.set_xlabel("Absorbed fluence $F_{abs}$ (J/m$^2$)")
        ax.set_title(f"{name.split(' (')[0]}\nextrapolation MAE {mae_high:.0f} K", fontsize=10)
    axes[0].set_ylabel("Peak surface $T_l$ (K)")
    axes[0].legend(loc="upper left", fontsize=8)
    axes[1].text(F_SPLIT * 1.05, y.min(), "test only", color="tab:red", fontsize=8)
    fig.suptitle(f"Extrapolation: trained on $F_{{abs}}$ < {F_SPLIT:.0f} J/m$^2$ only",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"Extrapolation plot saved to {path}")


# ---------------------------------------------------------------------------
# 7. Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    df, X, y_reg, y_cls = load_data(CSV_PATH)

    # One split for all models, stratified on melted. Row indices are split
    # too, so we can tell later which samples ended up in the test set.
    (X_train, X_test, yr_train, yr_test, yc_train, yc_test,
     idx_train, idx_test) = train_test_split(X, y_reg, y_cls, np.arange(len(df)),
                                             test_size=TEST_FRACTION,
                                             random_state=SEED, stratify=y_cls)
    print(f"Data: {len(X)} samples -> {len(X_train)} training, {len(X_test)} test; "
          f"melted fraction {yc_train.mean() * 100:.1f} % / {yc_test.mean() * 100:.1f} %")

    # Cross-validation folds (within the training set only). Stratified folds
    # for classification keep ~21 % melted in every fold.
    cv_reg = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    cv_cls = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    # --- Regression: Tl_peak ---
    lin = fit_linear_regression(X_train, X_test, yr_train, yr_test, cv_reg)
    phys = fit_physics_regression(X_train, X_test, yr_train, yr_test, cv_reg)
    rf_reg = fit_forest_regressor(X_train, X_test, yr_train, yr_test, cv_reg)
    hybrid = fit_hybrid_regression(X_train, X_test, yr_train, yr_test, cv_reg)
    print_regression_table([lin, phys, rf_reg, hybrid])
    parity_plot(lin, yr_train, yr_test, "parity_linear.png")
    parity_plot(phys, yr_train, yr_test, "parity_physics.png")
    parity_plot(rf_reg, yr_train, yr_test, "parity_rf.png")
    print_feature_importances(rf_reg["estimator"], "Tl_peak")
    extrapolation_test(df, X, y_reg, "extrapolation.png")

    # --- Classification: melted ---
    cls_rows = fit_logistic(X_train, X_test, yc_train, yc_test, cv_cls)
    cls_rows.append(fit_forest_classifier(X_train, X_test, yc_train, yc_test, cv_cls))
    print_classification_table(cls_rows, yc_test)
    print_feature_importances(cls_rows[-1]["estimator"], "melted")

    print("\nLogistic-regression melting threshold (P = 0.5), J/m^2:")
    for r in cls_rows[:2]:
        print(f"  {r['settings']:15s}: " + ", ".join(
            f"{tp * 1e15:.0f} fs -> {logistic_threshold(r['estimator'], tp):.0f}"
            for tp in (50e-15, 100e-15, 1e-12, 2e-12)))

    check_boundary_cases(df, idx_train, idx_test, X, rf_reg["estimator"], cls_rows)
