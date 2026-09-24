"""
Generate Publication-Quality Visualizations for EDA, Model Evaluation, and Business Risk
Saves high-res figures to figures/ directory.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, classification_report

# Global Plot Style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['figure.dpi'] = 300

os.makedirs('figures', exist_ok=True)
df = pd.read_csv('data/cleaned_dataset.csv')

CLASS_COLORS = {
    'Healthy': '#2e7d32',               # Forest Green
    'At-Risk': '#f57c00',               # Vibrant Orange
    'Abandonment-Imminent': '#c62828'   # Crimson Red
}
CLASS_ORDER = ['Healthy', 'At-Risk', 'Abandonment-Imminent']

# ==============================================================================
# 1. Risk-Class Distribution
# ==============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

counts = df['risk_class'].value_counts()[CLASS_ORDER]
pcts = (df['risk_class'].value_counts(normalize=True)[CLASS_ORDER] * 100).round(1)

bars = ax1.bar(CLASS_ORDER, counts, color=[CLASS_COLORS[c] for c in CLASS_ORDER], width=0.55, edgecolor='black', linewidth=1)
for bar, pct in zip(bars, pcts):
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 4, f"{int(yval)} ({pct}%)", ha='center', va='bottom', fontweight='bold', fontsize=11)
ax1.set_ylim(0, max(counts) * 1.18)
ax1.set_title("Ecosystem Risk Class Distribution (N=443)", pad=15)
ax1.set_ylabel("Number of Packages")
ax1.set_xlabel("Assigned Risk Class")

# Ecosystem Breakdown Stacked Bar
eco_risk = pd.crosstab(df['ecosystem'], df['risk_class'])[CLASS_ORDER]
eco_risk_pct = eco_risk.div(eco_risk.sum(axis=1), axis=0) * 100
eco_risk_pct.plot(kind='bar', stacked=True, ax=ax2, color=[CLASS_COLORS[c] for c in CLASS_ORDER], edgecolor='black', linewidth=0.8)
ax2.set_title("Risk Class Proportion Across npm and PyPI", pad=15)
ax2.set_ylabel("Percentage (%)")
ax2.set_xlabel("Package Ecosystem")
ax2.set_xticklabels(['npm', 'PyPI'], rotation=0)
ax2.legend(title="Risk Class", frameon=True, loc='lower right')
for c in ax2.containers:
    ax2.bar_label(c, fmt='%.1f%%', label_type='center', color='white', fontweight='bold', fontsize=9)

plt.tight_layout()
plt.savefig('figures/eda_risk_distribution.png', dpi=300)
plt.close()
print("Saved figures/eda_risk_distribution.png")

# ==============================================================================
# 2. Maintenance Inactivity vs Risk
# ==============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

sns.boxplot(x='risk_class', y='days_since_last_release', data=df, order=CLASS_ORDER,
            palette=CLASS_COLORS, hue='risk_class', legend=False, ax=ax1, width=0.5, fliersize=3)
ax1.axhline(365, color='#c62828', linestyle='--', linewidth=1.5, label='1-Year Abandonment Threshold')
ax1.axhline(180, color='#f57c00', linestyle=':', linewidth=1.5, label='6-Month Warning Threshold')
ax1.set_yscale('log')
ax1.set_title("Days Elapsed Since Last Release (Log Scale)", pad=15)
ax1.set_ylabel("Inactivity Duration (Days, Log Scale)")
ax1.set_xlabel("Risk Classification")
ax1.legend(loc='lower left', frameon=True)

# Release Cadence Annual
sns.boxplot(x='risk_class', y='release_cadence_annual', data=df, order=CLASS_ORDER,
            palette=CLASS_COLORS, hue='risk_class', legend=False, ax=ax2, width=0.5, fliersize=3)
ax2.set_title("Annual Release Cadence by Risk Class", pad=15)
ax2.set_ylabel("Releases per Year of Existence")
ax2.set_xlabel("Risk Classification")
ax2.set_ylim(-1, 35)

plt.tight_layout()
plt.savefig('figures/eda_maintenance_inactivity.png', dpi=300)
plt.close()
print("Saved figures/eda_maintenance_inactivity.png")

# ==============================================================================
# 3. Contributor Concentration / Bus Factor vs Risk
# ==============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

sns.boxplot(x='risk_class', y='contributor_gini', data=df, order=CLASS_ORDER,
            palette=CLASS_COLORS, hue='risk_class', legend=False, ax=ax1, width=0.5)
ax1.set_title("Contributor Gini Coefficient (The Bus Factor Index)", pad=15)
ax1.set_ylabel("Gini Coefficient of Commits (1.0 = Extreme Bottleneck)")
ax1.set_xlabel("Risk Classification")
ax1.set_ylim(0.4, 1.05)

# Top Contributor Share vs Issue Resolution
sns.scatterplot(x='top_contributor_share', y='issue_resolution_ratio', hue='risk_class',
                hue_order=CLASS_ORDER, palette=CLASS_COLORS, data=df, ax=ax2, alpha=0.75, s=65, edgecolor='black', linewidth=0.5)
ax2.set_title("Lead Contributor Share vs Issue Resolution Velocity", pad=15)
ax2.set_xlabel("Share of Commits by Lead Contributor (Top-1 Share)")
ax2.set_ylabel("Issue Resolution Ratio (Closed / Total)")
ax2.axvline(0.80, color='grey', linestyle='--', alpha=0.7, label='80% Solo Burden')
ax2.axhline(0.60, color='grey', linestyle=':', alpha=0.7, label='60% Resolution Benchmark')
ax2.legend(title="Risk Class", frameon=True, loc='lower left')

plt.tight_layout()
plt.savefig('figures/eda_contributor_concentration.png', dpi=300)
plt.close()
print("Saved figures/eda_contributor_concentration.png")

# ==============================================================================
# 4. Downloads vs Risk
# ==============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

for r_class in CLASS_ORDER:
    subset = df[df['risk_class'] == r_class]
    sns.kdeplot(subset['log_downloads_monthly'], label=r_class, color=CLASS_COLORS[r_class],
                linewidth=2.5, ax=ax1, fill=True, alpha=0.18)
ax1.set_title("Monthly Download Volume Density across Risk Classes", pad=15)
ax1.set_xlabel("Log10(Monthly Downloads + 1)")
ax1.set_ylabel("Kernel Density Estimate (KDE)")
ax1.legend(title="Risk Class", frameon=True)

sns.boxplot(x='risk_class', y='log_downloads_monthly', data=df, order=CLASS_ORDER,
            palette=CLASS_COLORS, hue='risk_class', legend=False, ax=ax2, width=0.5, fliersize=3)
ax2.set_title("Monthly Downloads Distribution (Log Scale)", pad=15)
ax2.set_ylabel("Log10(Monthly Downloads + 1)")
ax2.set_xlabel("Risk Classification")

plt.tight_layout()
plt.savefig('figures/eda_downloads_vs_risk.png', dpi=300)
plt.close()
print("Saved figures/eda_downloads_vs_risk.png")

# ==============================================================================
# 5. Dependents vs Risk
# ==============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

sns.boxplot(x='risk_class', y='log_dependents', data=df, order=CLASS_ORDER,
            palette=CLASS_COLORS, hue='risk_class', legend=False, ax=ax1, width=0.5, fliersize=3)
ax1.set_title("Ecosystem Adoption / Dependents Distribution", pad=15)
ax1.set_ylabel("Log10(Dependents / Users Count + 1)")
ax1.set_xlabel("Risk Classification")

# Blast Radius Score Distribution
sns.boxplot(x='risk_class', y='blast_radius_score', data=df, order=CLASS_ORDER,
            palette=CLASS_COLORS, hue='risk_class', legend=False, ax=ax2, width=0.5, fliersize=3)
ax2.set_title("Downstream Blast Radius Score (0–100 Scale)", pad=15)
ax2.set_ylabel("Calculated Blast Radius Score")
ax2.set_xlabel("Risk Classification")
ax2.axhline(60, color='#c62828', linestyle='--', label='Critical Impact Line (>=60)')
ax2.axhline(40, color='#f57c00', linestyle=':', label='High Impact Line (>=40)')
ax2.legend(loc='lower left', frameon=True)

plt.tight_layout()
plt.savefig('figures/eda_dependents_vs_risk.png', dpi=300)
plt.close()
print("Saved figures/eda_dependents_vs_risk.png")

# ==============================================================================
# 6. Feature Correlation Heatmap
# ==============================================================================
corr_cols = [
    'contributor_gini', 'top_contributor_share', 'bus_factor_approx',
    'releases_count', 'release_cadence_annual', 'repository_age_years',
    'days_since_last_release', 'issue_resolution_ratio', 'open_issues',
    'log_downloads_monthly', 'log_dependents', 'stars_count', 'forks_count',
    'blast_radius_score'
]
corr_labels = [
    'Contributor Gini', 'Top-1 Commit %', 'Bus Factor',
    'Releases Count', 'Release Cadence', 'Repo Age (Yrs)',
    'Days Since Release', 'Issue Resolution %', 'Open Issues',
    'Log Downloads', 'Log Dependents', 'GitHub Stars', 'GitHub Forks',
    'Blast Radius'
]

corr_matrix = df[corr_cols].corr()

plt.figure(figsize=(12, 9.5))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='vlag', vmin=-0.8, vmax=0.8,
            xticklabels=corr_labels, yticklabels=corr_labels, cbar_kws={'label': 'Pearson Correlation (r)'},
            linewidths=0.6, linecolor='white')
plt.title("Correlation Matrix of Contributor Health, Inactivity, and Ecosystem Blast Radius", pad=20)
plt.tight_layout()
plt.savefig('figures/eda_feature_correlation.png', dpi=300)
plt.close()
print("Saved figures/eda_feature_correlation.png")

# ==============================================================================
# 7. Model Training & Evaluation (Decision Tree)
# ==============================================================================
feature_cols = [
    'contributor_gini', 'top_contributor_share', 'top_2_contributor_share',
    'bus_factor_approx', 'contributors_count', 'maintainers_count',
    'total_commits', 'releases_count', 'repository_age_years',
    'release_cadence_annual', 'issue_resolution_ratio', 'open_issues',
    'dependencies_count', 'dev_dependencies_count', 'stars_count',
    'forks_count', 'log_downloads_monthly', 'log_dependents', 'has_test_script',
    'is_permissive'
]

X = df[feature_cols]
y = df['risk_class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

dt_model = DecisionTreeClassifier(
    max_depth=4,
    min_samples_leaf=6,
    min_samples_split=10,
    criterion='entropy',
    class_weight='balanced',
    random_state=42
)
dt_model.fit(X_train, y_train)

y_pred = dt_model.predict(X_test)
report = classification_report(y_test, y_pred, target_names=CLASS_ORDER, output_dict=True)

# 7A. Decision Tree Plot
plt.figure(figsize=(18, 9))
plot_tree(
    dt_model,
    feature_names=[c.replace('_', ' ').title() for c in feature_cols],
    class_names=CLASS_ORDER,
    filled=True,
    rounded=True,
    fontsize=9,
    precision=2
)
plt.title("Decision Tree Architecture for Predicting Open-Source Abandonment Risk (Max Depth = 4)", pad=20)
plt.tight_layout()
plt.savefig('figures/model_decision_tree.png', dpi=300)
plt.close()
print("Saved figures/model_decision_tree.png")

# 7B. Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=CLASS_ORDER)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_ORDER, yticklabels=CLASS_ORDER,
            cbar=False, ax=ax1, linewidths=0.8, linecolor='white')
ax1.set_title("Test Set Confusion Matrix (Raw Counts)", pad=15)
ax1.set_ylabel("True Risk Class")
ax1.set_xlabel("Predicted Risk Class")

sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues', xticklabels=CLASS_ORDER, yticklabels=CLASS_ORDER,
            cbar=True, ax=ax2, linewidths=0.8, linecolor='white')
ax2.set_title("Test Set Normalized Confusion Matrix (%)", pad=15)
ax2.set_ylabel("True Risk Class")
ax2.set_xlabel("Predicted Risk Class")

plt.tight_layout()
plt.savefig('figures/model_confusion_matrix.png', dpi=300)
plt.close()
print("Saved figures/model_confusion_matrix.png")

# 7C. Feature Importances
imp = pd.Series(dt_model.feature_importances_, index=[c.replace('_', ' ').title() for c in feature_cols]).sort_values(ascending=True)
imp_active = imp[imp > 0]

plt.figure(figsize=(10, 5.5))
bars = plt.barh(imp_active.index, imp_active.values, color='#1976d2', edgecolor='black', linewidth=0.8)
for bar in bars:
    w = bar.get_width()
    plt.text(w + 0.005, bar.get_y() + bar.get_height()/2, f"{w*100:.1f}%", va='center', fontweight='bold', fontsize=9.5)
plt.xlim(0, max(imp_active.values) * 1.18)
plt.title("Decision Tree Feature Importance: Non-Leaking Predictors of Abandonment", pad=15)
plt.xlabel("Gini / Entropy Importance Weight")
plt.ylabel("Predictive Feature")
plt.tight_layout()
plt.savefig('figures/model_feature_importance.png', dpi=300)
plt.close()
print("Saved figures/model_feature_importance.png")

# ==============================================================================
# 8. Business Risk & Blast Radius Strategic Action Matrix
# ==============================================================================
# Predict on entire dataset for enterprise risk audit
df['predicted_risk'] = dt_model.predict(X)

# Define 4-Quadrant Strategic Recommendations
# X-axis: Blast Radius Score (0-100)
# Y-axis: Risk Probability / Class Index
def assign_recommendation(row):
    r = row['predicted_risk']
    blast = row['blast_radius_score']
    
    if r == 'Abandonment-Imminent':
        if blast >= 50:
            return 'FORK / REPLACE IMMEDIATELY (Critical Exposure)'
        else:
            return 'REPLACE / RETIRE (Low Blast Radius)'
    elif r == 'At-Risk':
        if blast >= 50:
            return 'PRIORITIZE FUNDING & BACKUP MAINTAINERS'
        else:
            return 'MONITOR CADENCE & MITIGATE'
    else: # Healthy
        if blast >= 50:
            return 'CONTINUOUS MONITORING (Core Dependency)'
        else:
            return 'STANDARD VENDOR ACCEPTANCE'

df['strategic_action'] = df.apply(assign_recommendation, axis=1)

fig, ax = plt.subplots(figsize=(13, 8))

ACTION_COLORS = {
    'FORK / REPLACE IMMEDIATELY (Critical Exposure)': '#b71c1c',     # Deep Maroon Red
    'PRIORITIZE FUNDING & BACKUP MAINTAINERS': '#e65100',           # Amber Orange
    'REPLACE / RETIRE (Low Blast Radius)': '#d32f2f',               # Bright Red
    'MONITOR CADENCE & MITIGATE': '#fbc02d',                        # Yellow Gold
    'CONTINUOUS MONITORING (Core Dependency)': '#1976d2',           # Blue
    'STANDARD VENDOR ACCEPTANCE': '#388e3c'                         # Green
}

# Jitter points slightly on Y for clarity
y_map = {'Healthy': 1, 'At-Risk': 2, 'Abandonment-Imminent': 3}
y_jitter = df['predicted_risk'].map(y_map) + np.random.uniform(-0.18, 0.18, size=len(df))

for action, color in ACTION_COLORS.items():
    mask = df['strategic_action'] == action
    sub = df[mask]
    if len(sub) > 0:
        ax.scatter(sub['blast_radius_score'], y_jitter[mask], label=f"{action} (n={len(sub)})",
                   color=color, alpha=0.75, s=65, edgecolors='black', linewidth=0.5)

# Quadrant dividing lines
ax.axvline(50, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
ax.axhline(2.5, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
ax.axhline(1.5, color='black', linestyle=':', linewidth=1.0, alpha=0.5)

# Quadrant Labels
ax.text(25, 3.42, "LOW IMPACT × IMMINENT ABANDONMENT\nAction: Replace / Retire Stale Dependency",
        ha='center', va='center', bbox=dict(boxstyle="round,pad=0.4", fc="#ffebee", ec="#ef5350", lw=1.2), fontsize=9.5, fontweight='bold')
ax.text(75, 3.42, "CRITICAL EXPOSURE (High Blast × Imminent Abandonment)\nAction: FORK IMMEDIATELY / SECURE ENTERPRISE VENDOR",
        ha='center', va='center', bbox=dict(boxstyle="round,pad=0.4", fc="#ffcdd2", ec="#c62828", lw=1.5), fontsize=9.5, fontweight='bold')
ax.text(75, 2.0, "HIGH IMPACT × AT-RISK (Single Maintainer / Bottleneck)\nAction: CO-SPONSOR, CONTRIBUTE & FUND MAINTAINERS",
        ha='center', va='center', bbox=dict(boxstyle="round,pad=0.4", fc="#fff8e1", ec="#ffa000", lw=1.5), fontsize=9.5, fontweight='bold')
ax.text(75, 0.65, "CORE STABLE DEPENDENCIES (High Blast × Healthy)\nAction: Continuous SBOM Vulnerability Monitoring",
        ha='center', va='center', bbox=dict(boxstyle="round,pad=0.4", fc="#e8f5e9", ec="#4caf50", lw=1.2), fontsize=9.5, fontweight='bold')

# Annotate famous key packages
notable = ['express', 'lodash', 'chalk', 'request', 'core-js', 'left-pad', 'theano', 'nose', 'minimatch', 'commander']
for pkg in notable:
    p_row = df[df['package_name'] == pkg]
    if not p_row.empty:
        idx = p_row.index[0]
        x_val = p_row['blast_radius_score'].values[0]
        y_val = y_jitter[idx]
        ax.annotate(pkg, (x_val, y_val), xytext=(x_val + 1.2, y_val + 0.05),
                    fontweight='bold', fontsize=9, bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black", lw=0.6),
                    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0", color="black", lw=0.8))

ax.set_yticks([1, 2, 3])
ax.set_yticklabels(['Healthy', 'At-Risk', 'Abandonment-Imminent'], fontweight='bold', fontsize=11)
ax.set_ylim(0.4, 3.6)
ax.set_xlim(-2, 102)
ax.set_xlabel("Downstream Blast Radius Score (Downloads & Dependents Centrality, 0–100)", fontsize=11)
ax.set_ylabel("Predicted Maintainer Risk Level", fontsize=11)
ax.set_title("The Bus-Factor Blast Radius Action Matrix: Strategic Dependency Governance", pad=15)
ax.legend(loc='lower left', bbox_to_anchor=(0.0, -0.28), ncol=2, frameon=True, fontsize=8.5)

plt.subplots_adjust(bottom=0.22)
plt.savefig('figures/business_blast_radius_matrix.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved figures/business_blast_radius_matrix.png")

print("All figures generated successfully!")
