"""
Generate Publication-Quality Visualizations for EDA, Model Evaluation, and Business Risk
Saves high-res figures to figures/ directory.
Eliminates repetitive box plots in favor of diverse, mathematically rigorous charts:
Donut chart, Grouped bar, Kernel violin, ECDF, Bus Factor tiers, Non-linear scatter,
KDE density, 2D Bubble centrality, 100% Stacked severity, Heatmaps, and Decision trees.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, classification_report
from numpy.polynomial import Polynomial

# Global Plot Style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12.5
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['figure.dpi'] = 300

os.makedirs('figures', exist_ok=True)
df = pd.read_csv('data/cleaned_dataset.csv')

CLASS_ORDER = ['Abandonment-Imminent', 'At-Risk', 'Healthy']
CLASS_COLORS = {
    'Abandonment-Imminent': '#b71c1c',  # Crimson Red
    'At-Risk': '#e65100',               # Dark Orange
    'Healthy': '#1b5e20'                # Deep Green
}

# ==============================================================================
# 1. Risk-Class Distribution (Donut Chart + Ecosystem Grouped Bar Chart)
# ==============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2))

counts = df['risk_class'].value_counts()[CLASS_ORDER]
pcts = (counts / len(df) * 100).round(1)

# Donut Chart
wedges, texts, autotexts = ax1.pie(
    counts,
    labels=CLASS_ORDER,
    autopct='%1.1f%%',
    pctdistance=0.75,
    colors=[CLASS_COLORS[c] for c in CLASS_ORDER],
    startangle=140,
    wedgeprops=dict(width=0.45, edgecolor='white', linewidth=2)
)
for text in texts:
    text.set_fontsize(10)
    text.set_fontweight('bold')
for autotext in autotexts:
    autotext.set_fontsize(10.5)
    autotext.set_fontweight('bold')
    autotext.set_color('white')

ax1.text(0, 0.08, f"N = {len(df)}", ha='center', va='center', fontsize=14, fontweight='bold', color='#0f172a')
ax1.text(0, -0.12, "Packages", ha='center', va='center', fontsize=11, color='#475569')
ax1.set_title("Ecosystem Risk Class Distribution (Donut Breakdown)", pad=15)

# Ecosystem Grouped Bar Chart
eco_ct = pd.crosstab(df['ecosystem'], df['risk_class'])[CLASS_ORDER]
eco_pct = (pd.crosstab(df['ecosystem'], df['risk_class'], normalize='index')[CLASS_ORDER] * 100).round(1)

x = np.arange(len(eco_ct.index))
width = 0.25

for idx, r_class in enumerate(CLASS_ORDER):
    bars = ax2.bar(x + (idx - 1)*width, eco_ct[r_class], width=width,
                   label=r_class, color=CLASS_COLORS[r_class], edgecolor='black', linewidth=0.8)
    for bar, pct in zip(bars, eco_pct[r_class]):
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, h + 3, f"{pct}%", ha='center', va='bottom', fontsize=9.5, fontweight='bold')

ax2.set_xticks(x)
ax2.set_xticklabels(['npm (Node.js)\n[n=386]', 'PyPI (Python)\n[n=57]'], fontweight='bold', fontsize=10.5)
ax2.set_ylabel("Number of Packages")
ax2.set_ylim(0, max(eco_ct.max()) * 1.18)
ax2.set_title("Risk Class Distribution by Registry Ecosystem", pad=15)
ax2.legend(title="Risk Class", frameon=True, loc='upper right')

plt.tight_layout()
plt.savefig('figures/eda_risk_distribution.png', dpi=300)
plt.close()
print("Saved figures/eda_risk_distribution.png")

# ==============================================================================
# 2. Maintenance Inactivity vs Risk (Kernel Violin Plot + ECDF Curves)
# ==============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2))

# Kernel Violin Plot with Quartiles on Log Scale
sns.violinplot(x='risk_class', y='days_since_last_release', data=df, order=CLASS_ORDER,
               palette=CLASS_COLORS, hue='risk_class', legend=False, inner='quartile', cut=0, ax=ax1)
ax1.axhline(365, color='#c62828', linestyle='--', linewidth=1.5, label='1-Year Abandonment Boundary (365d)')
ax1.axhline(180, color='#f57c00', linestyle=':', linewidth=1.5, label='6-Month Warning Boundary (180d)')
ax1.set_yscale('log')
ax1.set_title("Inactivity Duration Probability Density (Violin Plot, Log Scale)", pad=15)
ax1.set_ylabel("Days Elapsed Since Last Release (Log Scale)")
ax1.set_xlabel("Risk Classification")
ax1.legend(loc='lower left', frameon=True, fontsize=9.5)

# Empirical Cumulative Distribution Function (ECDF) for Cadence
for r_class in CLASS_ORDER:
    subset = df[df['risk_class'] == r_class]['release_cadence_annual']
    sns.ecdfplot(subset, label=f"{r_class} (Median: {subset.median():.1f}/yr)",
                 color=CLASS_COLORS[r_class], linewidth=2.5, ax=ax2)

ax2.axvline(1.5, color='#b71c1c', linestyle='--', alpha=0.7, label='1.5 Releases/Yr Critical Threshold')
ax2.set_xlim(0, 25)
ax2.set_title("Empirical Cumulative Distribution (ECDF) of Annual Release Cadence", pad=15)
ax2.set_xlabel("Annualized Release Cadence (Releases / Year)")
ax2.set_ylabel("Cumulative Probability P(Cadence ≤ x)")
ax2.legend(title="Risk Class", frameon=True, loc='lower right', fontsize=9.5)

plt.tight_layout()
plt.savefig('figures/eda_maintenance_inactivity.png', dpi=300)
plt.close()
print("Saved figures/eda_maintenance_inactivity.png")

# ==============================================================================
# 3. Contributor Concentration & Bus Factor (Bus Factor Tiers + Scatter Plot)
# ==============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2))

# Bus Factor Tier Grouped Breakdown
def get_bf_tier(bf):
    if bf == 1: return 'BF = 1 (Monopoly)'
    elif bf == 2: return 'BF = 2 (Duopoly)'
    elif bf <= 5: return 'BF = 3–5 (Small Core)'
    else: return 'BF > 5 (Distributed)'

df['bf_tier'] = df['bus_factor_approx'].apply(get_bf_tier)
bf_tiers = ['BF = 1 (Monopoly)', 'BF = 2 (Duopoly)', 'BF = 3–5 (Small Core)', 'BF > 5 (Distributed)']
bf_pct = pd.crosstab(df['risk_class'], df['bf_tier'], normalize='index')[bf_tiers].loc[CLASS_ORDER] * 100

x = np.arange(len(CLASS_ORDER))
width = 0.20
TIER_COLORS = ['#991b1b', '#ea580c', '#3b82f6', '#16a34a']

for idx, tier in enumerate(bf_tiers):
    bars = ax1.bar(x + (idx - 1.5)*width, bf_pct[tier], width=width,
                   label=tier, color=TIER_COLORS[idx], edgecolor='black', linewidth=0.8)
    for bar in bars:
        h = bar.get_height()
        if h > 5:
            ax1.text(bar.get_x() + bar.get_width()/2.0, h + 1.2, f"{int(round(h))}%", ha='center', va='bottom', fontsize=8.5, fontweight='bold')

ax1.set_xticks(x)
ax1.set_xticklabels(CLASS_ORDER, fontweight='bold', fontsize=9.5)
ax1.set_ylabel("Percentage of Packages within Class (%)")
ax1.set_ylim(0, 65)
ax1.set_title("Bus Factor (Developers for 80% Commits) Tier Breakdown", pad=15)
ax1.legend(title="Bus Factor Tier", frameon=True, loc='upper right', fontsize=8.5)

# Scatter Plot with Bottleneck Zone
sns.scatterplot(x='top_contributor_share', y='issue_resolution_ratio', hue='risk_class',
                hue_order=CLASS_ORDER, palette=CLASS_COLORS, data=df, ax=ax2, alpha=0.80, s=70, edgecolor='black', linewidth=0.5)

p_fit = Polynomial.fit(df['top_contributor_share'], df['issue_resolution_ratio'], deg=2)
x_line = np.linspace(df['top_contributor_share'].min(), df['top_contributor_share'].max(), 100)
ax2.plot(x_line, p_fit(x_line), color='#0f172a', linestyle='-', linewidth=2.5, label='Non-Linear Triage Decay Trend')

ax2.axvspan(0.80, 1.02, color='#fee2e2', alpha=0.5, label='Solo Bottleneck Zone (>80% Lead Share)')
ax2.axhline(0.40, color='#dc2626', linestyle=':', linewidth=1.5, label='40% Resolution Collapse Line')
ax2.set_xlim(0.15, 1.03)
ax2.set_ylim(-0.05, 1.05)
ax2.set_title("Lead Contributor Share vs Issue Resolution Velocity", pad=15)
ax2.set_xlabel("Share of Commits by Lead Contributor (Top-1 Share)")
ax2.set_ylabel("Issue Resolution Ratio (Closed / Total)")
ax2.legend(title="Risk Class & Guides", frameon=True, loc='lower left', fontsize=8.5)

plt.tight_layout()
plt.savefig('figures/eda_contributor_concentration.png', dpi=300)
plt.close()
print("Saved figures/eda_contributor_concentration.png")

# ==============================================================================
# 4. Downloads vs Risk (KDE Density + High-Exposure Dormant Packages)
# ==============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2))

for r_class in CLASS_ORDER:
    subset = df[df['risk_class'] == r_class]
    sns.kdeplot(subset['log_downloads_monthly'], label=r_class, color=CLASS_COLORS[r_class],
                linewidth=2.5, ax=ax1, fill=True, alpha=0.18)
    median_val = subset['log_downloads_monthly'].median()
    ax1.axvline(median_val, color=CLASS_COLORS[r_class], linestyle=':', linewidth=1.8,
                label=f"{r_class} Med: 10^{median_val:.1f}")

ax1.set_title("Monthly Download Volume Density across Risk Classes (KDE)", pad=15)
ax1.set_xlabel("Log10(Monthly Downloads + 1)")
ax1.set_ylabel("Kernel Density Estimate (KDE)")
ax1.legend(title="Risk Class & Median", frameon=True, loc='upper left', fontsize=8.5)

# Horizontal Bar Chart of Landmark Abandoned Packages
notable_dormant = df[
    (df['risk_class'] == 'Abandonment-Imminent') &
    (df['downloads_monthly'] >= 100000)
].sort_values(by='downloads_monthly', ascending=True).tail(8)

y_pos = np.arange(len(notable_dormant))
dl_millions = notable_dormant['downloads_monthly'] / 1e6

bars = ax2.barh(y_pos, dl_millions, color='#b71c1c', edgecolor='black', linewidth=0.8, height=0.55)
for idx, (bar, days) in enumerate(zip(bars, notable_dormant['days_since_last_release'])):
    val = bar.get_width()
    ax2.text(val + 0.3, bar.get_y() + bar.get_height()/2.0,
             f"{val:.1f}M/mo  ({int(days)}d inactive)", va='center', fontsize=9, fontweight='bold', color='#1e293b')

ax2.set_yticks(y_pos)
ax2.set_yticklabels(notable_dormant['package_name'], fontweight='bold', fontsize=10)
ax2.set_xlabel("Monthly Downloads (Millions)")
ax2.set_xlim(0, max(dl_millions) * 1.35)
ax2.set_title("The Vanity Paradox: Dormant Packages Retaining Millions of Downloads", pad=15)

plt.tight_layout()
plt.savefig('figures/eda_downloads_vs_risk.png', dpi=300)
plt.close()
print("Saved figures/eda_downloads_vs_risk.png")

# ==============================================================================
# 5. Dependents vs Risk (2D Bubble Centrality + 100% Stacked Severity Tiers)
# ==============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2))

# 2D Ecosystem Exposure Bubble Plot
sns.scatterplot(
    data=df,
    x='log_downloads_monthly',
    y='log_dependents',
    hue='risk_class',
    hue_order=CLASS_ORDER,
    palette=CLASS_COLORS,
    size='blast_radius_score',
    sizes=(25, 250),
    alpha=0.75,
    edgecolor='black',
    linewidth=0.5,
    ax=ax1
)

ax1.axvspan(5.5, 9.5, ymin=0.45, ymax=1.0, color='#fee2e2', alpha=0.35)
ax1.axvline(5.5, color='#991b1b', linestyle='--', alpha=0.6)
ax1.axhline(2.0, color='#991b1b', linestyle='--', alpha=0.6)

annot_pkgs = ['request', 'left-pad', 'core-js', 'express', 'lodash', 'nomnom', 'colors']
for _, row in df[df['package_name'].isin(annot_pkgs)].iterrows():
    ax1.annotate(row['package_name'],
                 (row['log_downloads_monthly'], row['log_dependents']),
                 xytext=(6, 5), textcoords='offset points',
                 fontsize=8.5, fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='#cbd5e1'))

ax1.set_title("2D Ecosystem Centrality: Downloads vs Downstream Dependents", pad=15)
ax1.set_xlabel("Log10(Monthly Downloads + 1)")
ax1.set_ylabel("Log10(Downstream Dependents + 1)")
ax1.legend(loc='lower left', frameon=True, fontsize=8, ncol=2)

# Horizontal 100% Stacked Bar Chart for Blast Radius Tiers
def get_blast_tier(score):
    if score >= 65: return 'Critical (≥65)'
    elif score >= 50: return 'High (50–64)'
    elif score >= 35: return 'Moderate (35–49)'
    else: return 'Low (<35)'

df['blast_tier'] = df['blast_radius_score'].apply(get_blast_tier)
BLAST_TIERS = ['Low (<35)', 'Moderate (35–49)', 'High (50–64)', 'Critical (≥65)']
BLAST_COLORS = ['#388e3c', '#fbc02d', '#f57c00', '#d32f2f']

tier_pct = pd.crosstab(df['risk_class'], df['blast_tier'], normalize='index')[BLAST_TIERS].loc[CLASS_ORDER] * 100

y = np.arange(len(CLASS_ORDER))
left = np.zeros(len(CLASS_ORDER))

for idx, tier in enumerate(BLAST_TIERS):
    values = tier_pct[tier].values
    bars = ax2.barh(y, values, left=left, height=0.55, label=tier,
                    color=BLAST_COLORS[idx], edgecolor='white', linewidth=1.2)
    for bar_idx, val in enumerate(values):
        if val > 6:
            ax2.text(left[bar_idx] + val/2.0, y[bar_idx], f"{val:.1f}%",
                     ha='center', va='center', color='white' if idx in [0, 2, 3] else 'black',
                     fontweight='bold', fontsize=9.5)
    left += values

ax2.set_yticks(y)
ax2.set_yticklabels(CLASS_ORDER, fontweight='bold', fontsize=10)
ax2.set_xlabel("Percentage of Packages (%)")
ax2.set_xlim(0, 100)
ax2.set_title("Downstream Blast Radius Exposure Tier Breakdown (%)", pad=15)
ax2.legend(loc='lower left', bbox_to_anchor=(0.0, 1.02), ncol=4, frameon=True, fontsize=8.5)

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
    max_depth=3,
    min_samples_leaf=4,
    min_samples_split=5,
    criterion='entropy',
    class_weight=None,
    random_state=42
)
dt_model.fit(X_train, y_train)

y_pred = dt_model.predict(X_test)
MODEL_CLASSES = list(dt_model.classes_) # ['Abandonment-Imminent', 'At-Risk', 'Healthy']
report = classification_report(y_test, y_pred, labels=MODEL_CLASSES, target_names=MODEL_CLASSES, output_dict=True)

# 7A. Decision Tree Plot
plt.figure(figsize=(18, 9))
plot_tree(
    dt_model,
    feature_names=[c.replace('_', ' ').title() for c in feature_cols],
    class_names=MODEL_CLASSES,
    filled=True,
    rounded=True,
    fontsize=10,
    precision=2
)
plt.title("Decision Tree Architecture for Predicting Open-Source Abandonment Risk (Depth = 3)", pad=20)
plt.tight_layout()
plt.savefig('figures/model_decision_tree.png', dpi=300)
plt.close()
print("Saved figures/model_decision_tree.png")

# 7B. Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=MODEL_CLASSES)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=MODEL_CLASSES, yticklabels=MODEL_CLASSES,
            cbar=False, ax=ax1, linewidths=0.8, linecolor='white')
ax1.set_title("Test Set Confusion Matrix (Raw Counts)", pad=15)
ax1.set_ylabel("True Risk Class")
ax1.set_xlabel("Predicted Risk Class")

sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues', xticklabels=MODEL_CLASSES, yticklabels=MODEL_CLASSES,
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
df['predicted_risk'] = dt_model.predict(X)

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

y_map = {'Healthy': 1, 'At-Risk': 2, 'Abandonment-Imminent': 3}
y_jitter = df['predicted_risk'].map(y_map) + np.random.uniform(-0.18, 0.18, size=len(df))

for action, color in ACTION_COLORS.items():
    mask = df['strategic_action'] == action
    sub = df[mask]
    if len(sub) > 0:
        ax.scatter(sub['blast_radius_score'], y_jitter[mask], label=f"{action} (n={len(sub)})",
                   color=color, alpha=0.75, s=65, edgecolors='black', linewidth=0.5)

ax.axvline(50, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
ax.axhline(2.5, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
ax.axhline(1.5, color='black', linestyle=':', linewidth=1.0, alpha=0.5)

ax.text(25, 3.42, "LOW IMPACT × IMMINENT ABANDONMENT\nAction: Replace / Retire Stale Dependency",
        ha='center', va='center', bbox=dict(boxstyle="round,pad=0.4", fc="#ffebee", ec="#ef5350", lw=1.2), fontsize=9.5, fontweight='bold')
ax.text(75, 3.42, "CRITICAL EXPOSURE (High Blast × Imminent Abandonment)\nAction: FORK IMMEDIATELY / SECURE ENTERPRISE VENDOR",
        ha='center', va='center', bbox=dict(boxstyle="round,pad=0.4", fc="#ffcdd2", ec="#c62828", lw=1.5), fontsize=9.5, fontweight='bold')
ax.text(75, 2.0, "HIGH IMPACT × AT-RISK (Single Maintainer / Bottleneck)\nAction: CO-SPONSOR, CONTRIBUTE & FUND MAINTAINERS",
        ha='center', va='center', bbox=dict(boxstyle="round,pad=0.4", fc="#fff8e1", ec="#ffa000", lw=1.5), fontsize=9.5, fontweight='bold')
ax.text(75, 0.65, "CORE STABLE DEPENDENCIES (High Blast × Healthy)\nAction: Continuous SBOM Vulnerability Monitoring",
        ha='center', va='center', bbox=dict(boxstyle="round,pad=0.4", fc="#e8f5e9", ec="#4caf50", lw=1.2), fontsize=9.5, fontweight='bold')

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
