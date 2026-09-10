"""
Diabetes prediction on the Pima Indians dataset.

Compare a few ML pipelines against a small ANN under a leak-free evaluation:
model + threshold picked via cross-validation on the training set, test set
touched exactly once at the end. Clinical framing throughout — a missed
diabetic matters more than a false alarm.

"""

import os
# Silence TF's startup chatter — must be set before importing tensorflow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.calibration import calibration_curve
from sklearn.metrics import (accuracy_score, f1_score, recall_score, precision_score,
                             roc_auc_score, roc_curve, confusion_matrix,
                             classification_report, precision_recall_curve,
                             brier_score_loss)

# imblearn's Pipeline applies SMOTE only during fit(), so oversampling
# does NOT leak into CV validation folds. Using sklearn's Pipeline here
# would be a common but subtle mistake.
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from xgboost import XGBClassifier
import tensorflow as tf


# ------------------------------------------------------------
# Setup
# ------------------------------------------------------------
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

PLOTS = Path("plots")
PLOTS.mkdir(exist_ok=True)

def save(name):
    """Save current figure to plots/ at README-friendly quality."""
    plt.savefig(PLOTS / name, dpi=150, bbox_inches="tight")

# GPU memory growth so a small VRAM card doesn't get everything grabbed at
# once. If no GPU is visible we just run on CPU — the ANN is tiny anyway.
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for g in gpus:
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except RuntimeError:
            pass
    print(f"[hw] GPU: {gpus[0].name}")
else:
    print("[hw] no GPU visible, running on CPU")


# ============================================================
# Load data
# ============================================================
df = pd.read_csv("diabetes.csv")
print(f"\nLoaded {df.shape[0]} rows, {df.shape[1]} columns")
print(df.head())

# 0 is biologically impossible in these columns — it's a missing-value
# stand-in in the raw Pima dataset. If the CSV is already pre-imputed
# (some Kaggle mirrors are), this is a no-op and no NaNs get created.
zero_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
df[zero_cols] = df[zero_cols].replace(0, np.nan)


# ============================================================
# Exploratory data analysis
# ============================================================
print("\nDescribe:")
print(df.describe().round(2))

missing = (df.isnull().mean() * 100).round(1).sort_values(ascending=False)
missing = missing[missing > 0]
print("\nMissing %:")
print(missing.to_string() if len(missing) else "(none — dataset is pre-imputed)")

print("\nClass balance:")
print(df["Outcome"].value_counts(normalize=True).round(3).to_string())

print("\nMeans by outcome:")
print(df.groupby("Outcome").mean().round(2))


# --- Missing + class balance ---
fig, axes = plt.subplots(1, 2, figsize=(14, 4))

if len(missing):
    missing.plot(kind="bar", color="steelblue", ax=axes[0])
    axes[0].set_title("Missing values (%)")
    axes[0].set_ylabel("% missing")
    axes[0].tick_params(axis="x", rotation=30)
else:
    axes[0].text(0.5, 0.5, "No missing values\n(dataset pre-imputed)",
                 ha="center", va="center", fontsize=13,
                 transform=axes[0].transAxes)
    axes[0].set_title("Missing values")
    axes[0].set_xticks([]); axes[0].set_yticks([])

sns.countplot(data=df, x="Outcome", hue="Outcome",
              palette={0: "green", 1: "red"}, legend=False, ax=axes[1])
axes[1].set_title("Class balance")
for p in axes[1].patches:
    axes[1].annotate(f"{int(p.get_height())}",
                     (p.get_x() + p.get_width() / 2, p.get_height()),
                     ha="center", va="bottom")

plt.tight_layout()
save("01_missing_and_balance.png")
plt.show()


feature_cols = df.drop(columns="Outcome").columns.tolist()

# --- Boxplots by outcome (the key "which features separate" plot) ---
fig, axes = plt.subplots(2, 4, figsize=(18, 9))
for ax, col in zip(axes.ravel(), feature_cols):
    sns.boxplot(data=df, x="Outcome", y=col, hue="Outcome",
                palette={0: "green", 1: "red"}, legend=False, ax=ax)
    ax.set_title(col)
plt.suptitle("Feature distributions by outcome", fontsize=13, y=1.00)
plt.tight_layout()
save("02_boxplots_by_outcome.png")
plt.show()


# --- Histograms + KDE ---
fig, axes = plt.subplots(2, 4, figsize=(18, 9))
for ax, col in zip(axes.ravel(), feature_cols):
    sns.histplot(data=df, x=col, hue="Outcome", bins=30, kde=True,
                 palette={0: "green", 1: "red"}, alpha=0.5, ax=ax)
    ax.set_title(col)
plt.suptitle("Feature histograms by outcome", fontsize=13, y=1.00)
plt.tight_layout()
save("03_histograms_by_outcome.png")
plt.show()


print("\nSkew:")
print(df[feature_cols].skew().sort_values(ascending=False).round(2).to_string())


# --- Correlation heatmap ---
plt.figure(figsize=(9, 6))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title("Feature correlations")
plt.tight_layout()
save("04_correlation_heatmap.png")
plt.show()


# ============================================================
# Cleaning
# ============================================================
# Insulin was ~49% missing in the raw dataset. Even when a mirror has
# imputed it, most of those values are just the median — dropping it
# keeps the story clean.
df = df.drop(columns=["Insulin"])

# Rows still missing 3+ features are too incomplete to trust after
# imputation — on a pre-cleaned CSV this line drops 0 rows.
before = len(df)
df = df[df.isnull().sum(axis=1) < 3].reset_index(drop=True)
print(f"\nAfter cleaning: {len(df)} rows (-{before - len(df)}), "
      f"{df.shape[1] - 1} features")


# ============================================================
# Train / test split
# ============================================================
# Note: no imputation or scaling has happened yet. Everything numerical
# lives inside the pipelines below so it fits per CV fold, never on the
# whole dataset — that's how the test set stays truly held out.
X = df.drop(columns="Outcome")
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED)
print(f"Train: {len(X_train)}   Test: {len(X_test)}")


# ============================================================
# Candidate pipelines
# ============================================================
def make_pipe(model, smote=True):
    steps = [("impute", SimpleImputer(strategy="median")),
             ("scale", StandardScaler())]
    if smote:
        steps.append(("smote", SMOTE(random_state=SEED)))
    steps.append(("model", model))
    return ImbPipeline(steps)

# XGBoost's version of class_weight='balanced'
xgb_spw = (y_train == 0).sum() / (y_train == 1).sum()

candidates = {
    "LogReg + SMOTE": make_pipe(
        LogisticRegression(max_iter=5000, random_state=SEED)),

    "LogReg (class_weight)": make_pipe(
        LogisticRegression(class_weight="balanced", max_iter=5000, random_state=SEED),
        smote=False),

    "RandomForest + SMOTE": make_pipe(
        RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=SEED)),

    "GradientBoosting": make_pipe(
        GradientBoostingClassifier(random_state=SEED), smote=False),

    "XGBoost (scale_pos_weight)": make_pipe(
        XGBClassifier(eval_metric="logloss", scale_pos_weight=xgb_spw,
                      n_jobs=-1, random_state=SEED), smote=False),
}


# ============================================================
# Model selection — 5-fold CV on train only
# ============================================================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

cv_results, oof = [], {}
for name, pipe in candidates.items():
    proba = cross_val_predict(pipe, X_train, y_train, cv=cv,
                              method="predict_proba", n_jobs=-1)[:, 1]
    oof[name] = proba
    yhat = (proba >= 0.5).astype(int)
    cv_results.append({
        "Model":  name,
        "AUC":    round(roc_auc_score(y_train, proba), 3),
        "F1":     round(f1_score(y_train, yhat), 3),
        "Recall": round(recall_score(y_train, yhat), 3),
        "Prec":   round(precision_score(y_train, yhat), 3),
    })

cv_df = (pd.DataFrame(cv_results)
           .sort_values("F1", ascending=False)
           .reset_index(drop=True))
print("\nCV results (5-fold, train only):")
print(cv_df.to_string(index=False))

best_name = cv_df.iloc[0]["Model"]
best_pipe = candidates[best_name]
print(f"Winner by CV F1: {best_name}")


# ============================================================
# Threshold tuning — on out-of-fold train predictions
# ============================================================
# Recall floor of 0.75 — missing a diabetic is worse than a false alarm,
# but I still want the highest-F1 threshold above that floor, not just
# max recall (which would degrade precision to nothing).
proba = oof[best_name]

rows = []
for t in np.arange(0.20, 0.65, 0.05):
    yhat = (proba >= t).astype(int)
    rows.append({
        "Threshold": round(t, 2),
        "F1":     round(f1_score(y_train, yhat), 3),
        "Recall": round(recall_score(y_train, yhat), 3),
        "Prec":   round(precision_score(y_train, yhat), 3),
    })
thr_df = pd.DataFrame(rows)
print("\nThreshold tuning:")
print(thr_df.to_string(index=False))

ok = thr_df[thr_df["Recall"] >= 0.75]
chosen = (ok if len(ok) else thr_df).sort_values("F1", ascending=False).iloc[0]
BEST_T = chosen["Threshold"]
print(f"Chosen threshold: {BEST_T}  "
      f"(CV F1={chosen['F1']}, Recall={chosen['Recall']})")


# ============================================================
# Fit on full train, evaluate on test (once)
# ============================================================
best_pipe.fit(X_train, y_train)

y_proba_ml = best_pipe.predict_proba(X_test)[:, 1]
y_pred_ml = (y_proba_ml >= BEST_T).astype(int)

print(f"\n=== Test: {best_name} @ threshold {BEST_T} ===")
print(f"Accuracy : {accuracy_score(y_test, y_pred_ml):.3f}")
print(f"ROC-AUC  : {roc_auc_score(y_test, y_proba_ml):.3f}")
print(f"F1       : {f1_score(y_test, y_pred_ml):.3f}")
print(f"Recall   : {recall_score(y_test, y_pred_ml):.3f}")
print(f"Precision: {precision_score(y_test, y_pred_ml):.3f}\n")
print(classification_report(y_test, y_pred_ml,
                            target_names=["Non-Diabetic", "Diabetic"]))

cm_ml = confusion_matrix(y_test, y_pred_ml)


# ============================================================
# ANN — same held-out discipline
# ============================================================
# Preprocess with train-fit imputer/scaler
prep = SkPipeline([("impute", SimpleImputer(strategy="median")),
                   ("scale", StandardScaler())])
X_train_p = prep.fit_transform(X_train)
X_test_p = prep.transform(X_test)

# Carve a validation slice off TRAIN — used for both early stopping AND
# threshold selection, so no test-set peeking anywhere.
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_p, y_train, test_size=0.2, stratify=y_train, random_state=SEED)

w = compute_class_weight("balanced", classes=np.unique(y_tr), y=y_tr)
class_weights = dict(zip(np.unique(y_tr), w))

# Small net on purpose — 7 features and ~500 training rows don't justify
# anything deeper. The original 128-64-32 was overfitting bait.
ann = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_tr.shape[1],)),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid"),
])
ann.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
            loss="binary_crossentropy", metrics=["AUC"])

es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=20,
                                      restore_best_weights=True)
history = ann.fit(X_tr, y_tr, epochs=200, batch_size=32,
                  validation_data=(X_val, y_val), callbacks=[es],
                  class_weight=class_weights, verbose=0)
print(f"\nANN trained for {len(history.history['loss'])} epochs")

# Pick ANN threshold on val — training predictions would be optimistic
# because the model has already fit to them.
p_val = ann.predict(X_val, verbose=0).ravel()
pr_v, rc_v, thr_v = precision_recall_curve(y_val, p_val)
f1_v = 2 * pr_v * rc_v / (pr_v + rc_v + 1e-9)
best_thr_ann = float(thr_v[np.argmax(f1_v[:-1])])
print(f"ANN threshold from val PR curve: {best_thr_ann:.3f}")

y_proba_ann = ann.predict(X_test_p, verbose=0).ravel()
y_pred_ann = (y_proba_ann >= best_thr_ann).astype(int)
cm_ann = confusion_matrix(y_test, y_pred_ann)


# ============================================================
# Comparison
# ============================================================
comparison = pd.DataFrame({
    "Metric": ["Accuracy", "ROC-AUC", "Recall (Diab)",
               "Precision (Diab)", "F1 (Diab)", "False Negatives"],
    best_name: [round(accuracy_score(y_test, y_pred_ml), 3),
                round(roc_auc_score(y_test, y_proba_ml), 3),
                round(recall_score(y_test, y_pred_ml), 3),
                round(precision_score(y_test, y_pred_ml), 3),
                round(f1_score(y_test, y_pred_ml), 3),
                int(cm_ml[1][0])],
    "ANN":     [round(accuracy_score(y_test, y_pred_ann), 3),
                round(roc_auc_score(y_test, y_proba_ann), 3),
                round(recall_score(y_test, y_pred_ann), 3),
                round(precision_score(y_test, y_pred_ann), 3),
                round(f1_score(y_test, y_pred_ann), 3),
                int(cm_ann[1][0])],
}).set_index("Metric")

print("\nFinal comparison (test set, evaluated once):")
print(comparison.to_string())


# ============================================================
# Diagnostic plots — confusion, ROC, PR
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

sns.heatmap(cm_ml, annot=True, fmt="d", cmap="Reds",
            xticklabels=["Non-Diab", "Diab"],
            yticklabels=["Non-Diab", "Diab"], ax=axes[0])
axes[0].set_title(f"{best_name}\nthreshold={BEST_T}")
axes[0].set_ylabel("Actual"); axes[0].set_xlabel("Predicted")

for label, p, c in [(best_name, y_proba_ml, "red"),
                    ("ANN", y_proba_ann, "blue")]:
    fpr, tpr, _ = roc_curve(y_test, p)
    axes[1].plot(fpr, tpr, lw=2, color=c,
                 label=f"{label} (AUC={roc_auc_score(y_test, p):.3f})")
axes[1].plot([0, 1], [0, 1], "k--", alpha=0.4)
axes[1].set_xlabel("False Positive Rate"); axes[1].set_ylabel("True Positive Rate")
axes[1].set_title("ROC"); axes[1].legend()

for label, p, c in [(best_name, y_proba_ml, "red"),
                    ("ANN", y_proba_ann, "blue")]:
    pr, rc, _ = precision_recall_curve(y_test, p)
    axes[2].plot(rc, pr, lw=2, color=c, label=label)
axes[2].set_xlabel("Recall"); axes[2].set_ylabel("Precision")
axes[2].set_title("Precision–Recall"); axes[2].legend()

plt.tight_layout()
save("05_confusion_roc_pr.png")
plt.show()


# ============================================================
# Feature importance
# ============================================================
# Tree models expose .feature_importances_, linear models .coef_
model = best_pipe.named_steps["model"]
if hasattr(model, "feature_importances_"):
    importances = model.feature_importances_
    kind = "Gini importance"
elif hasattr(model, "coef_"):
    importances = np.abs(model.coef_).ravel()
    kind = "|coef| (scaled features)"
else:
    importances = None

if importances is not None:
    imp = (pd.DataFrame({"Feature": X_train.columns, "Importance": importances})
             .sort_values("Importance", ascending=False)
             .reset_index(drop=True))
    print(f"\nFeature importance ({kind}):")
    print(imp.round(4).to_string(index=False))

    plt.figure(figsize=(9, 4.5))
    sns.barplot(data=imp, y="Feature", x="Importance",
                hue="Feature", palette="rocket_r", legend=False)
    plt.title(f"Feature importance — {best_name}")
    plt.tight_layout()
    save("06_feature_importance.png")
    plt.show()


# ============================================================
# Calibration — do the probabilities mean what they claim?
# ============================================================
# Brier score = mean squared error between predicted proba and outcome.
# Lower is better. Reliability curve = predicted vs actual per bin;
# a well-calibrated model sits on the diagonal.
brier_ml = brier_score_loss(y_test, y_proba_ml)
brier_ann = brier_score_loss(y_test, y_proba_ann)
print(f"\nBrier score (lower = better calibration):")
print(f"  {best_name}: {brier_ml:.4f}")
print(f"  ANN: {brier_ann:.4f}")

fig, ax = plt.subplots(figsize=(7, 6))
for label, p, c in [(best_name, y_proba_ml, "red"),
                    ("ANN", y_proba_ann, "blue")]:
    frac_pos, mean_pred = calibration_curve(y_test, p, n_bins=8, strategy="quantile")
    ax.plot(mean_pred, frac_pos, "o-", color=c, lw=2, label=label)
ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="perfect")
ax.set_xlabel("Mean predicted probability")
ax.set_ylabel("Actual fraction positive")
ax.set_title("Reliability curve")
ax.legend()
plt.tight_layout()
save("07_calibration_curve.png")
plt.show()


# ============================================================
# Clinical impact — what the numbers mean in practice
# ============================================================
tn, fp, fn, tp = cm_ml.ravel()
total = tn + fp + fn + tp
positives = tp + fn
flagged = tp + fp

print(f"\n--- Clinical impact (on {total} test patients) ---")
print(f"Actual diabetics       : {positives}")
print(f"Flagged for follow-up  : {flagged}  (TP={tp}, FP={fp})")
print(f"MISSED diabetics (FN)  : {fn}")
print(f"Sensitivity            : {tp / max(tp + fn, 1):.1%}")
print(f"Specificity            : {tn / max(tn + fp, 1):.1%}")
print(f"PPV                    : {tp / max(tp + fp, 1):.1%}")
print(f"NPV                    : {tn / max(tn + fn, 1):.1%}")


# ============================================================
# Save the pipeline artifact for deployment
# ============================================================
artifact = {
    "pipeline":      best_pipe,        # imputer + scaler + SMOTE + model
    "threshold":     float(BEST_T),
    "feature_order": X_train.columns.tolist(),
    "model_name":    best_name,
    "test_metrics": {
        "accuracy":  round(accuracy_score(y_test, y_pred_ml), 4),
        "roc_auc":   round(roc_auc_score(y_test, y_proba_ml), 4),
        "recall":    round(recall_score(y_test, y_pred_ml), 4),
        "precision": round(precision_score(y_test, y_pred_ml), 4),
        "f1":        round(f1_score(y_test, y_pred_ml), 4),
        "brier":     round(brier_ml, 4),
    },
}
joblib.dump(artifact, "diabetes_model.pkl")
print("\nSaved model bundle to diabetes_model.pkl")
print(f"Saved {len(list(PLOTS.glob('*.png')))} plots to {PLOTS}/")


# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 60)
print(" SUMMARY")
print("=" * 60)
print(f" Model         : {best_name}")
print(f" Threshold     : {BEST_T}")
print(f" Test accuracy : {accuracy_score(y_test, y_pred_ml):.3f}")
print(f" Test AUC      : {roc_auc_score(y_test, y_proba_ml):.3f}")
print(f" Test recall   : {recall_score(y_test, y_pred_ml):.3f}")
print(f" FN on test    : {fn} of {positives} diabetics missed")
print("=" * 60)
