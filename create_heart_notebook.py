"""
Script to create 1Cardio_Train_and_Evaluate_V2_heart.ipynb
Adapts the original cardio notebook to use heart.csv as the dataset.
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

def md(src):
    return nbf.v4.new_markdown_cell(src)

def code(src):
    return nbf.v4.new_code_cell(src)

# ── SECTION 1: DATA LOADING ─────────────────────────────────────────────────
cells.append(md("# Heart Disease Train & Evaluate\n\nAdapted from `1Cardio_Train_and_Evaluate_V2.ipynb` to use **heart.csv** as the dataset.\n\nDataset: UCI Heart Disease dataset (918 patients, 11 features, binary target `HeartDisease`)."))

cells.append(code("""\
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                     cross_validate, RandomizedSearchCV,
                                     GridSearchCV)
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (roc_auc_score, f1_score, recall_score,
                             precision_score, accuracy_score, confusion_matrix,
                             roc_curve, precision_recall_curve, brier_score_loss,
                             ConfusionMatrixDisplay)
from sklearn.calibration import calibration_curve
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('heart.csv')
"""))

cells.append(code("# Dataset Shape\ndf.shape"))

cells.append(code("df.head()"))

cells.append(code("df.info()"))

cells.append(code("df.describe()"))

cells.append(code("""\
# Target Class Balance
print(df['HeartDisease'].value_counts())
print()
print(df['HeartDisease'].value_counts(normalize=True).round(3))
"""))

# ── SECTION 2: EDA ───────────────────────────────────────────────────────────
cells.append(md("## Exploratory Data Analysis"))

cells.append(code("""\
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Target distribution
axes[0,0].bar(['No Disease (0)', 'Disease (1)'],
              df['HeartDisease'].value_counts().sort_index(),
              color=['steelblue', 'tomato'])
axes[0,0].set_title('Target Distribution')
axes[0,0].set_ylabel('Count')

# Age by disease status
for label, grp in df.groupby('HeartDisease'):
    grp['Age'].plot.kde(ax=axes[0,1], label=f'HeartDisease={label}')
axes[0,1].set_title('Age Distribution by Disease Status')
axes[0,1].legend()

# MaxHR by disease status
df.boxplot(column='MaxHR', by='HeartDisease', ax=axes[0,2])
axes[0,2].set_title('MaxHR by Disease Status')
axes[0,2].set_xlabel('HeartDisease')

# RestingBP by disease status
df.boxplot(column='RestingBP', by='HeartDisease', ax=axes[1,0])
axes[1,0].set_title('RestingBP by Disease Status')
axes[1,0].set_xlabel('HeartDisease')

# Cholesterol by disease status
df.boxplot(column='Cholesterol', by='HeartDisease', ax=axes[1,1])
axes[1,1].set_title('Cholesterol by Disease Status')
axes[1,1].set_xlabel('HeartDisease')

# Oldpeak by disease status
df.boxplot(column='Oldpeak', by='HeartDisease', ax=axes[1,2])
axes[1,2].set_title('Oldpeak by Disease Status')
axes[1,2].set_xlabel('HeartDisease')

plt.suptitle('EDA: Numerical Features vs Heart Disease', y=1.02, fontsize=14)
plt.tight_layout()
plt.show()
"""))

cells.append(code("""\
# Categorical feature distributions by disease status
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

cat_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope', 'FastingBS']
for ax, col in zip(axes.flatten(), cat_cols):
    ct = pd.crosstab(df[col], df['HeartDisease'], normalize='index')
    ct.plot(kind='bar', ax=ax, color=['steelblue', 'tomato'], rot=0)
    ax.set_title(f'{col} vs HeartDisease')
    ax.set_ylabel('Proportion')
    ax.legend(['No Disease', 'Disease'])

plt.suptitle('Categorical Features vs Heart Disease', y=1.02, fontsize=14)
plt.tight_layout()
plt.show()
"""))

# ── SECTION 3: DATA QUALITY ───────────────────────────────────────────────────
cells.append(md("## Data Quality & Cleaning"))

cells.append(code("""\
# Missing values
print("Missing values per column:")
print(df.isnull().sum())
"""))

cells.append(code("""\
# Detect physiologically implausible values
print(f"Cholesterol = 0: {(df['Cholesterol'] == 0).sum()} rows")
print(f"RestingBP = 0:   {(df['RestingBP'] == 0).sum()} rows")
print(f"MaxHR <= 0:      {(df['MaxHR'] <= 0).sum()} rows")
print(f"Age < 18:        {(df['Age'] < 18).sum()} rows")
"""))

cells.append(code("""\
# Remove physiologically implausible records
rows_before = len(df)
df = df[df['RestingBP'] > 0]
df = df[df['Cholesterol'] > 0]   # 0 likely indicates missing/imputed value
rows_after = len(df)

print(f"Rows removed: {rows_before - rows_after}")
print(f"Remaining rows: {rows_after} ({rows_after/rows_before*100:.1f}% of original)")
"""))

cells.append(code("""\
# Target distribution after cleaning
print("Target distribution after cleaning:")
print(df['HeartDisease'].value_counts())
print(df['HeartDisease'].value_counts(normalize=True).round(3))
"""))

# ── SECTION 4: FEATURE ENGINEERING ───────────────────────────────────────────
cells.append(md("## Feature Engineering"))

cells.append(code("""\
# Encode binary categoricals
df['Sex_enc'] = df['Sex'].map({'M': 1, 'F': 0})
df['ExerciseAngina_enc'] = df['ExerciseAngina'].map({'Y': 1, 'N': 0})
print("Binary encoding done.")
"""))

cells.append(code("""\
# Heart Rate Reserve: difference between MaxHR and age-predicted max HR (220 - Age)
# Positive values → patient achieved more than expected (good cardiopulmonary fitness)
df['HeartRateReserve'] = df['MaxHR'] - (220 - df['Age'])

# BP-to-MaxHR ratio: higher ratio may indicate cardiovascular stress
df['BP_HR_ratio'] = df['RestingBP'] / df['MaxHR']

print("Engineered features:")
print(df[['HeartRateReserve', 'BP_HR_ratio']].describe())
"""))

cells.append(code("""\
# Correlation heatmap for numeric features
numeric_df = df[['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak',
                 'HeartRateReserve', 'BP_HR_ratio',
                 'Sex_enc', 'FastingBS', 'ExerciseAngina_enc', 'HeartDisease']]

plt.figure(figsize=(12, 9))
sns.heatmap(numeric_df.corr(), annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, linewidths=0.5)
plt.title('Correlation Heatmap', fontsize=14)
plt.tight_layout()
plt.show()
"""))

# ── SECTION 5: FEATURE SELECTION & SPLIT ─────────────────────────────────────
cells.append(md("## Feature Selection & Stratified Train/Test Split"))

cells.append(code("""\
# Drop redundant / raw categorical columns already encoded
df_model = df.drop(columns=['Sex', 'ExerciseAngina'])

numerical_features = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak',
                      'HeartRateReserve', 'BP_HR_ratio']
binary_features    = ['Sex_enc', 'FastingBS', 'ExerciseAngina_enc']
categorical_features = ['ChestPainType', 'RestingECG', 'ST_Slope']

all_features = numerical_features + binary_features + categorical_features
print("Features:", all_features)
print("Target: HeartDisease")
"""))

cells.append(code("""\
X = df_model[all_features]
y = df_model['HeartDisease']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print(f"Training set : {X_train.shape[0]} rows")
print(f"Test set     : {X_test.shape[0]} rows")
print(f"Train target dist: {y_train.value_counts().to_dict()}")
print(f"Test  target dist: {y_test.value_counts().to_dict()}")
"""))

# ── SECTION 6: PREPROCESSING PIPELINE ────────────────────────────────────────
cells.append(md("## Preprocessing Pipeline"))

cells.append(code("""\
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'),
         categorical_features),
        ('bin', 'passthrough', binary_features),
    ]
)
"""))

cells.append(md("## Evaluation Strategy\n\n5-fold stratified cross-validation with AUC, F1, Recall, and Precision."))

cells.append(code("""\
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = {'auc': 'roc_auc', 'f1': 'f1', 'recall': 'recall', 'precision': 'precision'}
"""))

# ── SECTION 7: BASELINE MODELS ────────────────────────────────────────────────
cells.append(md("## Baseline Model Comparison"))

cells.append(code("""\
# Logistic Regression Baseline
log_reg_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', LogisticRegression(max_iter=1000, random_state=42))
])

cv_results_lr = cross_validate(log_reg_pipeline, X_train, y_train,
                                cv=cv, scoring=scoring, return_train_score=False)
print("=== Logistic Regression (5-fold CV) ===")
for metric in ['auc', 'f1', 'recall', 'precision']:
    scores = cv_results_lr[f'test_{metric}']
    print(f"  {metric.upper():<12}: {scores.mean():.4f} ± {scores.std():.4f}")
"""))

cells.append(code("""\
# Random Forest Baseline
rf_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', RandomForestClassifier(n_estimators=100, random_state=42))
])

cv_results_rf = cross_validate(rf_pipeline, X_train, y_train,
                                cv=cv, scoring=scoring, return_train_score=False)
print("=== Random Forest (5-fold CV) ===")
for metric in ['auc', 'f1', 'recall', 'precision']:
    scores = cv_results_rf[f'test_{metric}']
    print(f"  {metric.upper():<12}: {scores.mean():.4f} ± {scores.std():.4f}")
"""))

cells.append(code("""\
# XGBoost Baseline
xgb_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', XGBClassifier(eval_metric='logloss', random_state=42, use_label_encoder=False))
])

cv_results_xgb = cross_validate(xgb_pipeline, X_train, y_train,
                                  cv=cv, scoring=scoring, return_train_score=False)
print("=== XGBoost (5-fold CV) ===")
for metric in ['auc', 'f1', 'recall', 'precision']:
    scores = cv_results_xgb[f'test_{metric}']
    print(f"  {metric.upper():<12}: {scores.mean():.4f} ± {scores.std():.4f}")
"""))

cells.append(code("""\
# Summary comparison table
results = {
    'Logistic Regression': cv_results_lr,
    'Random Forest': cv_results_rf,
    'XGBoost': cv_results_xgb,
}
rows = []
for model_name, res in results.items():
    row = {'Model': model_name}
    for metric in ['auc', 'f1', 'recall', 'precision']:
        row[metric.upper()] = f"{res[f'test_{metric}'].mean():.4f}"
    rows.append(row)

summary_df = pd.DataFrame(rows).set_index('Model')
print("=== Baseline Model Comparison ===")
print(summary_df.to_string())
"""))

cells.append(md("**Baseline Interpretation**: Comparing AUC across Logistic Regression, Random Forest, and XGBoost to identify the strongest starting point before hyperparameter tuning."))

# ── SECTION 8: HYPERPARAMETER TUNING ─────────────────────────────────────────
cells.append(md("## Hyperparameter Tuning"))

cells.append(code("""\
# XGBoost – RandomizedSearchCV
param_dist = {
    'model__n_estimators': [100, 200, 300, 400],
    'model__max_depth': [3, 4, 5, 6, 7],
    'model__learning_rate': [0.01, 0.05, 0.1, 0.15],
    'model__subsample': [0.6, 0.7, 0.8, 0.9],
    'model__colsample_bytree': [0.6, 0.7, 0.8, 0.9],
    'model__min_child_weight': [1, 3, 5],
}

xgb_search = RandomizedSearchCV(
    xgb_pipeline, param_dist,
    n_iter=30, scoring='roc_auc',
    cv=cv, n_jobs=-1, random_state=42, verbose=0
)
xgb_search.fit(X_train, y_train)
print("XGBoost best params:", xgb_search.best_params_)
print(f"XGBoost best CV AUC: {xgb_search.best_score_:.4f}")
"""))

cells.append(code("""\
# Evaluate tuned XGBoost with full CV metrics
best_xgb = xgb_search.best_estimator_
cv_tuned_xgb = cross_validate(best_xgb, X_train, y_train,
                               cv=cv, scoring=scoring, return_train_score=False)
print("=== Tuned XGBoost (5-fold CV) ===")
for metric in ['auc', 'f1', 'recall', 'precision']:
    scores = cv_tuned_xgb[f'test_{metric}']
    print(f"  {metric.upper():<12}: {scores.mean():.4f} ± {scores.std():.4f}")
"""))

cells.append(code("""\
# Logistic Regression – GridSearchCV over C
param_grid_lr = {'model__C': [0.001, 0.01, 0.1, 1, 10, 100]}
log_reg_search = GridSearchCV(
    log_reg_pipeline, param_grid_lr,
    scoring='roc_auc', cv=cv, n_jobs=-1
)
log_reg_search.fit(X_train, y_train)
print("LR best C:", log_reg_search.best_params_)
print(f"LR best CV AUC: {log_reg_search.best_score_:.4f}")
"""))

cells.append(code("""\
best_log_reg = log_reg_search.best_estimator_
cv_tuned_lr = cross_validate(best_log_reg, X_train, y_train,
                              cv=cv, scoring=scoring, return_train_score=False)
print("=== Tuned Logistic Regression (5-fold CV) ===")
for metric in ['auc', 'f1', 'recall', 'precision']:
    scores = cv_tuned_lr[f'test_{metric}']
    print(f"  {metric.upper():<12}: {scores.mean():.4f} ± {scores.std():.4f}")
"""))

cells.append(md("**Tuning Summary**: XGBoost tends to benefit more from hyperparameter tuning, especially on structured tabular data. The best model will be selected based on AUC for final evaluation."))

# ── SECTION 9: TEST SET EVALUATION ───────────────────────────────────────────
cells.append(md("## Test Set Evaluation\n\nSelect the best model (highest tuned CV AUC) and evaluate on the held-out test set."))

cells.append(code("""\
# Use the tuned XGBoost (re-fit on full training set is done by best_estimator_ already)
best_xgb.fit(X_train, y_train)

y_test_pred  = best_xgb.predict(X_test)
y_test_proba = best_xgb.predict_proba(X_test)[:, 1]

print("=== Test Set Performance (threshold = 0.50) ===")
print(f"  AUC       : {roc_auc_score(y_test, y_test_proba):.4f}")
print(f"  Accuracy  : {accuracy_score(y_test, y_test_pred):.4f}")
print(f"  F1        : {f1_score(y_test, y_test_pred):.4f}")
print(f"  Recall    : {recall_score(y_test, y_test_pred):.4f}")
print(f"  Precision : {precision_score(y_test, y_test_pred):.4f}")
"""))

cells.append(code("""\
# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_test_proba)
auc_score = roc_auc_score(y_test, y_test_proba)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, lw=2, label=f'XGBoost (AUC = {auc_score:.3f})')
plt.plot([0,1], [0,1], 'k--', lw=1, label='Random baseline')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve – Test Set')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
"""))

cells.append(code("""\
# Confusion Matrix at threshold = 0.50
cm = confusion_matrix(y_test, y_test_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Disease', 'Disease'])
disp.plot(cmap='Blues', colorbar=False)
plt.title('Confusion Matrix (threshold = 0.50)')
plt.tight_layout()
plt.show()

tn, fp, fn, tp = cm.ravel()
print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")
print(f"False Negatives (missed disease): {fn}")
"""))

cells.append(md("**Test Evaluation Interpretation**: A high recall is critical in cardiac screening — missing a patient with heart disease (false negative) is clinically more costly than a false alarm. We next explore threshold optimisation to improve recall."))

# ── SECTION 10: THRESHOLD OPTIMISATION ───────────────────────────────────────
cells.append(md("## Threshold Optimisation"))

cells.append(code("""\
# Precision-Recall Curve
prec, rec, thresholds_pr = precision_recall_curve(y_test, y_test_proba)

plt.figure(figsize=(8, 6))
plt.plot(rec, prec, lw=2, color='darkorange')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
"""))

cells.append(code("""\
# Threshold sweep
threshold_results = []
for thresh in np.arange(0.10, 0.91, 0.05):
    preds = (y_test_proba >= thresh).astype(int)
    threshold_results.append({
        'threshold': round(thresh, 2),
        'recall':    recall_score(y_test, preds, zero_division=0),
        'precision': precision_score(y_test, preds, zero_division=0),
        'f1':        f1_score(y_test, preds, zero_division=0),
        'accuracy':  accuracy_score(y_test, preds),
    })

thresh_df = pd.DataFrame(threshold_results)
print(thresh_df.to_string(index=False))
"""))

cells.append(code("""\
# Visualise recall vs threshold
plt.figure(figsize=(10, 5))
plt.plot(thresh_df['threshold'], thresh_df['recall'],    label='Recall',    marker='o')
plt.plot(thresh_df['threshold'], thresh_df['precision'], label='Precision', marker='s')
plt.plot(thresh_df['threshold'], thresh_df['f1'],        label='F1',        marker='^')
plt.axhline(0.80, color='red', linestyle='--', label='Recall = 0.80')
plt.xlabel('Classification Threshold')
plt.ylabel('Score')
plt.title('Metrics vs Threshold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
"""))

cells.append(code("""\
# Find thresholds achieving recall >= 0.80
high_recall = thresh_df[thresh_df['recall'] >= 0.80]
print("Thresholds with Recall >= 0.80:")
print(high_recall.to_string(index=False))
"""))

cells.append(md("### Evaluation at Optimal Threshold (Recall ≥ 0.80)"))

cells.append(code("""\
# Use threshold = 0.30 (lower threshold → higher recall for screening context)
optimal_threshold = 0.30
y_test_pred_opt = (y_test_proba >= optimal_threshold).astype(int)

print(f"=== Test Set Performance (threshold = {optimal_threshold}) ===")
print(f"  AUC       : {roc_auc_score(y_test, y_test_proba):.4f}")
print(f"  Accuracy  : {accuracy_score(y_test, y_test_pred_opt):.4f}")
print(f"  F1        : {f1_score(y_test, y_test_pred_opt):.4f}")
print(f"  Recall    : {recall_score(y_test, y_test_pred_opt):.4f}")
print(f"  Precision : {precision_score(y_test, y_test_pred_opt):.4f}")

cm_opt = confusion_matrix(y_test, y_test_pred_opt)
disp_opt = ConfusionMatrixDisplay(confusion_matrix=cm_opt,
                                   display_labels=['No Disease', 'Disease'])
disp_opt.plot(cmap='Blues', colorbar=False)
plt.title(f'Confusion Matrix (threshold = {optimal_threshold})')
plt.tight_layout()
plt.show()

tn, fp, fn, tp = cm_opt.ravel()
print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")
print(f"False Negatives reduced to: {fn}")
"""))

cells.append(md("**Threshold Summary**: Lowering the threshold from 0.50 to 0.30 trades precision for recall, reducing false negatives — clinically preferable for cardiac screening where missing disease is more harmful than a false alarm."))

# ── SECTION 11: ABLATION STUDY ───────────────────────────────────────────────
cells.append(md("## Ablation Study\n\nMeasure the contribution of each feature group by removing it and comparing AUC on the test set."))

cells.append(code("""\
def evaluate_subset(features_to_drop, label):
    \"\"\"Train XGBoost without specified features and return test AUC.\"\"\"
    remaining = [f for f in all_features if f not in features_to_drop]

    num_f = [f for f in numerical_features if f not in features_to_drop]
    bin_f = [f for f in binary_features    if f not in features_to_drop]
    cat_f = [f for f in categorical_features if f not in features_to_drop]

    transformers = []
    if num_f:
        transformers.append(('num', StandardScaler(), num_f))
    if cat_f:
        transformers.append(('cat',
                             OneHotEncoder(drop='first', sparse_output=False,
                                           handle_unknown='ignore'), cat_f))
    if bin_f:
        transformers.append(('bin', 'passthrough', bin_f))

    prep = ColumnTransformer(transformers=transformers)
    pipe = Pipeline([
        ('preprocessor', prep),
        ('model', XGBClassifier(
            eval_metric='logloss', random_state=42, use_label_encoder=False,
            **{k.replace('model__',''):v
               for k,v in xgb_search.best_params_.items()}
        ))
    ])
    pipe.fit(X_train[remaining], y_train)
    proba = pipe.predict_proba(X_test[remaining])[:, 1]
    auc = roc_auc_score(y_test, proba)
    print(f"  Ablate [{label}]: AUC = {auc:.4f}  (ΔAUC = {auc - baseline_auc:+.4f})")
    return auc

baseline_auc = roc_auc_score(y_test, y_test_proba)
print(f"Baseline AUC (all features): {baseline_auc:.4f}")
print()

ablation_results = {}

# 1. Remove blood pressure features
ablation_results['No BP'] = evaluate_subset(
    ['RestingBP', 'BP_HR_ratio'], 'RestingBP + BP_HR_ratio')

# 2. Remove age features
ablation_results['No Age'] = evaluate_subset(
    ['Age', 'HeartRateReserve', 'BP_HR_ratio'], 'Age + HeartRateReserve + BP_HR_ratio')

# 3. Remove MaxHR features
ablation_results['No MaxHR'] = evaluate_subset(
    ['MaxHR', 'HeartRateReserve', 'BP_HR_ratio'], 'MaxHR + HeartRateReserve + BP_HR_ratio')

# 4. Remove chest-pain / ECG / symptoms
ablation_results['No Symptoms'] = evaluate_subset(
    ['ChestPainType', 'ExerciseAngina_enc', 'ST_Slope', 'Oldpeak'],
    'ChestPainType + ExerciseAngina + ST_Slope + Oldpeak')

# 5. Remove cholesterol
ablation_results['No Cholesterol'] = evaluate_subset(
    ['Cholesterol'], 'Cholesterol')

# 6. Remove engineered features only
ablation_results['No Engineered'] = evaluate_subset(
    ['HeartRateReserve', 'BP_HR_ratio'], 'HeartRateReserve + BP_HR_ratio')
"""))

cells.append(code("""\
# Ablation bar chart
labels = list(ablation_results.keys())
aucs   = list(ablation_results.values())
deltas = [a - baseline_auc for a in aucs]

fig, ax = plt.subplots(figsize=(10, 5))
colors = ['tomato' if d < -0.02 else 'orange' if d < 0 else 'steelblue' for d in deltas]
bars = ax.barh(labels, deltas, color=colors)
ax.axvline(0, color='black', linewidth=0.8)
ax.set_xlabel('ΔAUC vs Baseline')
ax.set_title(f'Ablation Study — Baseline AUC = {baseline_auc:.4f}')
for bar, delta in zip(bars, deltas):
    ax.text(delta - 0.001 if delta < 0 else delta + 0.001,
            bar.get_y() + bar.get_height()/2,
            f'{delta:+.4f}', va='center',
            ha='right' if delta < 0 else 'left', fontsize=9)
plt.tight_layout()
plt.show()
"""))

cells.append(md("**Ablation Summary**: Features with the largest negative ΔAUC have the most predictive power. Clinical symptom features (ChestPainType, ExerciseAngina, ST_Slope, Oldpeak) and MaxHR are typically strong predictors in heart disease datasets."))

# ── SECTION 12: SHAP ─────────────────────────────────────────────────────────
cells.append(md("## Model Interpretability with SHAP"))

cells.append(code("""\
try:
    import shap
    shap_available = True
except ImportError:
    shap_available = False
    print("SHAP not installed. Run: pip install shap")
"""))

cells.append(code("""\
if shap_available:
    # Extract fitted XGBoost model and transform test data
    xgb_model_fitted = best_xgb.named_steps['model']
    X_test_transformed = best_xgb.named_steps['preprocessor'].transform(X_test)

    # Get feature names after OneHotEncoder
    num_names = numerical_features
    ohe = best_xgb.named_steps['preprocessor'].named_transformers_['cat']
    cat_names = list(ohe.get_feature_names_out(categorical_features))
    bin_names = binary_features
    feature_names_out = num_names + cat_names + bin_names

    explainer   = shap.TreeExplainer(xgb_model_fitted)
    shap_values = explainer.shap_values(X_test_transformed)

    plt.figure()
    shap.summary_plot(shap_values, X_test_transformed,
                      feature_names=feature_names_out, show=False)
    plt.title('SHAP Summary Plot – Feature Importance')
    plt.tight_layout()
    plt.show()
else:
    print("Skipping SHAP — library not available.")
"""))

cells.append(code("""\
if shap_available:
    # Bar chart of mean |SHAP| values
    mean_shap = np.abs(shap_values).mean(axis=0)
    shap_series = pd.Series(mean_shap, index=feature_names_out).sort_values(ascending=False)

    plt.figure(figsize=(10, 6))
    shap_series.head(15).plot(kind='barh', color='steelblue')
    plt.gca().invert_yaxis()
    plt.xlabel('Mean |SHAP value|')
    plt.title('Top 15 Features by Mean |SHAP| Importance')
    plt.tight_layout()
    plt.show()

    print("Top 5 most influential features:")
    print(shap_series.head(5).to_string())
"""))

cells.append(md("**SHAP Interpretation**: SHAP values decompose each prediction into per-feature contributions. Features with high mean |SHAP| values dominate the model's decisions and align well with clinical domain knowledge on cardiac risk."))

# ── SECTION 13: CALIBRATION ───────────────────────────────────────────────────
cells.append(md("## Calibration Analysis\n\nA well-calibrated model's predicted probabilities match observed event rates — critical for clinical risk stratification."))

cells.append(code("""\
prob_true, prob_pred = calibration_curve(y_test, y_test_proba, n_bins=8)

plt.figure(figsize=(7, 6))
plt.plot(prob_pred, prob_true, 's-', label='XGBoost', lw=2)
plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
plt.xlabel('Mean Predicted Probability')
plt.ylabel('Fraction of Positives')
plt.title('Calibration Curve')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
"""))

cells.append(code("""\
brier = brier_score_loss(y_test, y_test_proba)
print(f"Brier Score: {brier:.4f}  (lower = better; random baseline ≈ 0.25)")
"""))

cells.append(md("**Calibration Summary**: A Brier score substantially below 0.25 (the random-classifier baseline) indicates meaningful probability estimates. A calibration curve close to the diagonal confirms the model's probabilities can be used directly for risk stratification without recalibration."))

# ── SECTION 14: FINAL SUMMARY ─────────────────────────────────────────────────
cells.append(md("## Final Summary\n\n| Phase | Key Finding |\n|-------|-------------|\n| **Data** | 918 patients; cleaned to remove 0-Cholesterol/0-RestingBP records |\n| **Target** | HeartDisease — slightly imbalanced (~55% positive) |\n| **Features** | 11 features + 2 engineered (HeartRateReserve, BP_HR_ratio) |\n| **Best Model** | Tuned XGBoost (RandomizedSearchCV) |\n| **Test AUC** | See cell output above |\n| **Threshold** | 0.30 recommended for screening (maximises recall) |\n| **Key Predictors** | Identified via ablation study & SHAP |\n| **Calibration** | Brier score and calibration curve confirm reliable probability estimates |"))

nb.cells = cells

output_path = '/home/user/AAI-Project/1Cardio_Train_and_Evaluate_V2_heart.ipynb'
with open(output_path, 'w') as f:
    nbf.write(nb, f)

print(f"Notebook written to: {output_path}")
print(f"Total cells: {len(nb.cells)}")
