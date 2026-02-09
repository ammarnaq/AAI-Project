"""
Heart Disease Prediction Analysis
===================================
Comprehensive analysis of the Heart Disease dataset including:
- Data exploration and preprocessing
- Exploratory Data Analysis (EDA)
- Feature engineering
- Multiple ML model training and evaluation
- Model comparison and insights
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc, roc_auc_score
)

# Create output directory for plots
os.makedirs('plots', exist_ok=True)

# ============================================================
# 1. DATA LOADING AND EXPLORATION
# ============================================================
print("=" * 60)
print("1. DATA LOADING AND EXPLORATION")
print("=" * 60)

df = pd.read_csv('heart.csv')

print(f"\nDataset Shape: {df.shape[0]} rows x {df.shape[1]} columns")
print(f"\nColumn Names:\n{list(df.columns)}")
print(f"\nData Types:\n{df.dtypes}")
print(f"\nFirst 5 Rows:")
print(df.head())
print(f"\nStatistical Summary (Numerical):")
print(df.describe())
print(f"\nStatistical Summary (Categorical):")
print(df.describe(include='object'))
print(f"\nMissing Values:\n{df.isnull().sum()}")
print(f"\nDuplicate Rows: {df.duplicated().sum()}")

# Target variable distribution
print(f"\nTarget Variable (HeartDisease) Distribution:")
print(df['HeartDisease'].value_counts())
print(f"\nPercentage:")
print(df['HeartDisease'].value_counts(normalize=True).round(4) * 100)

# ============================================================
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================
print("\n" + "=" * 60)
print("2. EXPLORATORY DATA ANALYSIS")
print("=" * 60)

# Define column types
categorical_cols = ['Sex', 'ChestPainType', 'FastingBS', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
numerical_cols = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']

# --- Plot 1: Target Distribution ---
fig, ax = plt.subplots(figsize=(8, 5))
colors = ['#2ecc71', '#e74c3c']
counts = df['HeartDisease'].value_counts()
bars = ax.bar(['No Heart Disease (0)', 'Heart Disease (1)'], counts.values, color=colors, edgecolor='black')
for bar, count in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 10,
            f'{count} ({count/len(df)*100:.1f}%)', ha='center', fontsize=12, fontweight='bold')
ax.set_title('Heart Disease Distribution', fontsize=14, fontweight='bold')
ax.set_ylabel('Count', fontsize=12)
plt.tight_layout()
plt.savefig('plots/01_target_distribution.png', dpi=150)
plt.close()
print("  Saved: plots/01_target_distribution.png")

# --- Plot 2: Numerical Feature Distributions ---
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()
for i, col in enumerate(numerical_cols):
    ax = axes[i]
    ax.hist(df[df['HeartDisease'] == 0][col], bins=30, alpha=0.6, color='#2ecc71', label='No Disease', edgecolor='black')
    ax.hist(df[df['HeartDisease'] == 1][col], bins=30, alpha=0.6, color='#e74c3c', label='Disease', edgecolor='black')
    ax.set_title(f'{col} Distribution by Heart Disease', fontsize=11, fontweight='bold')
    ax.set_xlabel(col)
    ax.set_ylabel('Count')
    ax.legend()
axes[-1].set_visible(False)
plt.suptitle('Numerical Feature Distributions', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('plots/02_numerical_distributions.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: plots/02_numerical_distributions.png")

# --- Plot 3: Categorical Features vs Heart Disease ---
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()
for i, col in enumerate(categorical_cols):
    ax = axes[i]
    ct = pd.crosstab(df[col], df['HeartDisease'], normalize='index') * 100
    ct.plot(kind='bar', stacked=True, ax=ax, color=colors, edgecolor='black')
    ax.set_title(f'{col} vs Heart Disease', fontsize=11, fontweight='bold')
    ax.set_xlabel(col)
    ax.set_ylabel('Percentage (%)')
    ax.legend(['No Disease', 'Disease'], loc='upper right', fontsize=8)
    ax.tick_params(axis='x', rotation=0)
plt.suptitle('Categorical Features vs Heart Disease', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('plots/03_categorical_vs_target.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: plots/03_categorical_vs_target.png")

# --- Plot 4: Correlation Heatmap ---
# Encode categorical columns temporarily for correlation
df_encoded = df.copy()
for col in df.select_dtypes(include='object').columns:
    df_encoded[col] = LabelEncoder().fit_transform(df[col])

fig, ax = plt.subplots(figsize=(12, 9))
corr = df_encoded.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            ax=ax, square=True, linewidths=0.5, vmin=-1, vmax=1)
ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('plots/04_correlation_heatmap.png', dpi=150)
plt.close()
print("  Saved: plots/04_correlation_heatmap.png")

# --- Plot 5: Box Plots for Outlier Detection ---
fig, axes = plt.subplots(1, 5, figsize=(20, 5))
for i, col in enumerate(numerical_cols):
    sns.boxplot(data=df, x='HeartDisease', y=col, ax=axes[i], palette=colors)
    axes[i].set_title(f'{col}', fontsize=11, fontweight='bold')
    axes[i].set_xlabel('Heart Disease')
plt.suptitle('Box Plots: Numerical Features by Heart Disease', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('plots/05_boxplots.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: plots/05_boxplots.png")

# --- Plot 6: Age vs MaxHR Scatter ---
fig, ax = plt.subplots(figsize=(10, 6))
scatter = ax.scatter(df['Age'], df['MaxHR'], c=df['HeartDisease'], cmap='RdYlGn_r',
                     alpha=0.6, edgecolors='black', linewidths=0.5)
ax.set_xlabel('Age', fontsize=12)
ax.set_ylabel('Max Heart Rate', fontsize=12)
ax.set_title('Age vs Max Heart Rate (colored by Heart Disease)', fontsize=14, fontweight='bold')
legend = ax.legend(*scatter.legend_elements(), title='Heart Disease')
ax.add_artist(legend)
plt.tight_layout()
plt.savefig('plots/06_age_vs_maxhr.png', dpi=150)
plt.close()
print("  Saved: plots/06_age_vs_maxhr.png")

# Print key EDA insights
print("\n--- Key EDA Insights ---")
print(f"  - Males in dataset: {(df['Sex']=='M').sum()} ({(df['Sex']=='M').mean()*100:.1f}%)")
print(f"  - Females in dataset: {(df['Sex']=='F').sum()} ({(df['Sex']=='F').mean()*100:.1f}%)")
print(f"  - Age range: {df['Age'].min()} - {df['Age'].max()} (mean: {df['Age'].mean():.1f})")
print(f"  - Cholesterol = 0 count: {(df['Cholesterol']==0).sum()} (likely missing values)")
print(f"  - RestingBP = 0 count: {(df['RestingBP']==0).sum()} (likely missing values)")
print(f"  - Most common chest pain type: {df['ChestPainType'].mode()[0]}")
top_corr = corr['HeartDisease'].drop('HeartDisease').abs().sort_values(ascending=False)
print(f"  - Top correlated features with HeartDisease:")
for feat, val in top_corr.head(5).items():
    print(f"      {feat}: {corr['HeartDisease'][feat]:.3f}")

# ============================================================
# 3. DATA PREPROCESSING
# ============================================================
print("\n" + "=" * 60)
print("3. DATA PREPROCESSING")
print("=" * 60)

df_processed = df.copy()

# Handle zero values in Cholesterol and RestingBP (replace with median)
for col in ['Cholesterol', 'RestingBP']:
    zero_count = (df_processed[col] == 0).sum()
    if zero_count > 0:
        median_val = df_processed[df_processed[col] > 0][col].median()
        df_processed[col] = df_processed[col].replace(0, median_val)
        print(f"  Replaced {zero_count} zero values in '{col}' with median ({median_val})")

# One-hot encode categorical columns
cat_cols_to_encode = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
df_processed = pd.get_dummies(df_processed, columns=cat_cols_to_encode, drop_first=False)

# Convert boolean columns to int
bool_cols = df_processed.select_dtypes(include='bool').columns
df_processed[bool_cols] = df_processed[bool_cols].astype(int)

print(f"\n  Processed dataset shape: {df_processed.shape}")
print(f"  Features after encoding: {df_processed.shape[1] - 1}")
print(f"  Columns: {list(df_processed.columns)}")

# Split features and target
X = df_processed.drop('HeartDisease', axis=1)
y = df_processed['HeartDisease']

# Train-test split (80/20 stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n  Training set: {X_train.shape[0]} samples")
print(f"  Test set: {X_test.shape[0]} samples")
print(f"  Training target distribution: {dict(y_train.value_counts())}")
print(f"  Test target distribution: {dict(y_test.value_counts())}")

# Scale numerical features
scaler = StandardScaler()
num_features = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak', 'FastingBS']
X_train[num_features] = scaler.fit_transform(X_train[num_features])
X_test[num_features] = scaler.transform(X_test[num_features])
print(f"\n  Scaled numerical features: {num_features}")

# ============================================================
# 4. MODEL BUILDING AND TRAINING
# ============================================================
print("\n" + "=" * 60)
print("4. MODEL BUILDING AND TRAINING")
print("=" * 60)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Support Vector Machine': SVC(kernel='rbf', probability=True, random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
}

results = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    print(f"\n  Training: {name}...")

    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')

    # Fit on full training set
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None

    results[name] = {
        'model': model,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'roc_auc': roc,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
    }

    print(f"    CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    print(f"    Test Accuracy: {acc:.4f}")
    print(f"    Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")
    if roc is not None:
        print(f"    ROC AUC: {roc:.4f}")

# ============================================================
# 5. MODEL EVALUATION AND COMPARISON
# ============================================================
print("\n" + "=" * 60)
print("5. MODEL EVALUATION AND COMPARISON")
print("=" * 60)

# --- Results Table ---
results_df = pd.DataFrame({
    name: {
        'CV Accuracy (mean)': f"{r['cv_mean']:.4f} ± {r['cv_std']:.4f}",
        'Test Accuracy': f"{r['accuracy']:.4f}",
        'Precision': f"{r['precision']:.4f}",
        'Recall': f"{r['recall']:.4f}",
        'F1 Score': f"{r['f1']:.4f}",
        'ROC AUC': f"{r['roc_auc']:.4f}" if r['roc_auc'] else 'N/A',
    }
    for name, r in results.items()
}).T

print("\n  Model Comparison Table:")
print(results_df.to_string())

# --- Plot 7: Model Accuracy Comparison ---
fig, ax = plt.subplots(figsize=(12, 6))
model_names = list(results.keys())
test_accs = [results[m]['accuracy'] for m in model_names]
cv_accs = [results[m]['cv_mean'] for m in model_names]
x = np.arange(len(model_names))
width = 0.35

bars1 = ax.bar(x - width/2, cv_accs, width, label='CV Accuracy (mean)', color='#3498db', edgecolor='black')
bars2 = ax.bar(x + width/2, test_accs, width, label='Test Accuracy', color='#e74c3c', edgecolor='black')

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
            f'{bar.get_height():.3f}', ha='center', fontsize=9, fontweight='bold')
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
            f'{bar.get_height():.3f}', ha='center', fontsize=9, fontweight='bold')

ax.set_ylabel('Accuracy', fontsize=12)
ax.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(model_names, rotation=15, ha='right')
ax.legend()
ax.set_ylim(0.7, 1.0)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('plots/07_model_accuracy_comparison.png', dpi=150)
plt.close()
print("\n  Saved: plots/07_model_accuracy_comparison.png")

# --- Plot 8: ROC Curves ---
fig, ax = plt.subplots(figsize=(10, 8))
colors_roc = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
for i, (name, r) in enumerate(results.items()):
    if r['y_pred_proba'] is not None:
        fpr, tpr, _ = roc_curve(y_test, r['y_pred_proba'])
        ax.plot(fpr, tpr, color=colors_roc[i], lw=2,
                label=f"{name} (AUC = {r['roc_auc']:.3f})")

ax.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Classifier')
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=10)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('plots/08_roc_curves.png', dpi=150)
plt.close()
print("  Saved: plots/08_roc_curves.png")

# --- Plot 9: Confusion Matrices ---
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()
for i, (name, r) in enumerate(results.items()):
    cm = confusion_matrix(y_test, r['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                xticklabels=['No Disease', 'Disease'],
                yticklabels=['No Disease', 'Disease'])
    axes[i].set_title(f'{name}\nAcc: {r["accuracy"]:.3f}', fontsize=11, fontweight='bold')
    axes[i].set_ylabel('Actual')
    axes[i].set_xlabel('Predicted')
plt.suptitle('Confusion Matrices for All Models', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('plots/09_confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: plots/09_confusion_matrices.png")

# --- Plot 10: Feature Importance (Random Forest) ---
rf_model = results['Random Forest']['model']
feature_importance = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(10, 8))
feature_importance.plot(kind='barh', ax=ax, color='#3498db', edgecolor='black')
ax.set_title('Feature Importance (Random Forest)', fontsize=14, fontweight='bold')
ax.set_xlabel('Importance', fontsize=12)
plt.tight_layout()
plt.savefig('plots/10_feature_importance.png', dpi=150)
plt.close()
print("  Saved: plots/10_feature_importance.png")

# --- Plot 11: Metrics Comparison Heatmap ---
metrics_data = pd.DataFrame({
    name: {
        'Accuracy': r['accuracy'],
        'Precision': r['precision'],
        'Recall': r['recall'],
        'F1 Score': r['f1'],
        'ROC AUC': r['roc_auc'] if r['roc_auc'] else 0,
    }
    for name, r in results.items()
}).T

fig, ax = plt.subplots(figsize=(12, 6))
sns.heatmap(metrics_data, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax,
            linewidths=0.5, vmin=0.7, vmax=1.0)
ax.set_title('Model Performance Metrics Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('plots/11_metrics_heatmap.png', dpi=150)
plt.close()
print("  Saved: plots/11_metrics_heatmap.png")

# ============================================================
# 6. SUMMARY AND CONCLUSIONS
# ============================================================
print("\n" + "=" * 60)
print("6. SUMMARY AND CONCLUSIONS")
print("=" * 60)

# Find best model
best_model_name = max(results, key=lambda k: results[k]['f1'])
best = results[best_model_name]

print(f"""
  Dataset: Heart Disease Prediction (918 samples, 11 features)
  Target: HeartDisease (binary: 0 = No Disease, 1 = Disease)

  Preprocessing Steps:
    - Replaced zero values in Cholesterol and RestingBP with median
    - One-hot encoded categorical variables
    - StandardScaler applied to numerical features
    - 80/20 stratified train-test split

  Best Performing Model: {best_model_name}
    - Test Accuracy:  {best['accuracy']:.4f}
    - Precision:      {best['precision']:.4f}
    - Recall:         {best['recall']:.4f}
    - F1 Score:       {best['f1']:.4f}
    - ROC AUC:        {best['roc_auc']:.4f}
    - CV Accuracy:    {best['cv_mean']:.4f} (+/- {best['cv_std']:.4f})

  Key Findings:
    - ST_Slope, ChestPainType, and Oldpeak are the most important predictors
    - Males have a higher prevalence of heart disease in this dataset
    - Asymptomatic (ASY) chest pain type is the strongest indicator of heart disease
    - Higher MaxHR is associated with lower heart disease risk
    - Age has a moderate positive correlation with heart disease

  All visualizations saved to the 'plots/' directory.
""")

# Print classification report for best model
print(f"  Classification Report ({best_model_name}):")
print(classification_report(y_test, best['y_pred'],
                            target_names=['No Disease', 'Disease']))

print("=" * 60)
print("Analysis complete. All plots saved to 'plots/' directory.")
print("=" * 60)
