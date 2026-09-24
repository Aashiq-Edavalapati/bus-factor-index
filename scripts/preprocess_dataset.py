"""
Data Preprocessing Script for Open-Source Package Abandonment Risk
Inputs: data/raw_dataset.csv
Outputs: data/cleaned_dataset.csv

Applies:
1. Deduplication and invalid record filtering
2. Missing value imputation
3. Datetime parsing and age/inactivity computation
4. Contributor concentration and Bus Factor calculations (Gini coefficient, top-1/2 shares)
5. Issue resolution metrics
6. Downstream blast radius computation (log downloads & dependents normalization)
7. Ground truth risk class target generation (Healthy, At-Risk, Abandonment-Imminent)
"""

import os
import json
import numpy as np
import pandas as pd

def categorize_license(lic_str):
    if not isinstance(lic_str, str) or not lic_str.strip():
        return 'Other'
    lic = lic_str.upper()
    if any(p in lic for p in ['MIT', 'APACHE', 'BSD', 'ISC', 'CC0', 'UNLICENSE', 'ZLIB']):
        return 'Permissive'
    elif any(c in lic for c in ['GPL', 'LGPL', 'AGPL', 'MPL', 'EPL']):
        return 'Copyleft'
    return 'Other'

def calc_gini(commits_list):
    if not commits_list or len(commits_list) <= 1 or sum(commits_list) == 0:
        return 1.0
    s = sorted(commits_list)
    n = len(s)
    tot = sum(s)
    gini = (2 * sum((i + 1) * c for i, c in enumerate(s))) / (n * tot) - (n + 1) / n
    return round(max(0.0, min(1.0, gini)), 4)

def calc_bus_factor(commits_list):
    if not commits_list or sum(commits_list) == 0:
        return 1
    s = sorted(commits_list, reverse=True)
    tot = sum(s)
    cum = 0
    bf = 0
    for c in s:
        cum += c
        bf += 1
        if cum >= 0.8 * tot:
            return bf
    return max(1, bf)

def assign_risk_class(row):
    days = row['days_since_last_release']
    open_iss = row['open_issues']
    dep = row['is_deprecated']
    
    if (days >= 365.0 and open_iss > 0) or (dep == 1 and days >= 180.0) or (days >= 730.0):
        return 'Abandonment-Imminent'
        
    if (days >= 180.0 and open_iss > 0) or \
       ((row['contributor_gini'] >= 0.85 or row['top_contributor_share'] >= 0.85 or row['bus_factor_approx'] <= 1) and 
        (row['issue_resolution_ratio'] < 0.70 or open_iss >= 25 or days >= 120.0)) or \
       (row['maintainers_count'] <= 1 and open_iss >= 50 and row['issue_resolution_ratio'] < 0.60):
        return 'At-Risk'
        
    return 'Healthy'

def main():
    print("Reading data/raw_dataset.csv...")
    raw = pd.read_csv('data/raw_dataset.csv')
    print(f"Initial raw record count: {len(raw)}")
    
    # 1. Deduplication and invalid filtering
    raw = raw.drop_duplicates(subset=['package_name', 'ecosystem']).copy()
    raw = raw.dropna(subset=['created_at', 'latest_release_date']).copy()
    raw = raw[raw['package_name'] != 'pychecker'].copy()
    
    ref_date = pd.to_datetime('2026-09-24', utc=True)
    created_dt = pd.to_datetime(raw['created_at'], format='ISO8601', utc=True)
    rel_dt = pd.to_datetime(raw['latest_release_date'], format='ISO8601', utc=True)
    
    clean_df = pd.DataFrame()
    clean_df['package_name'] = raw['package_name'].astype(str).str.strip()
    clean_df['ecosystem'] = raw['ecosystem']
    clean_df['repository_url'] = raw['repository_url'].fillna('').astype(str).str.strip()
    clean_df['license'] = raw['license'].fillna('Open Source')
    clean_df['license_type'] = clean_df['license'].apply(categorize_license)
    clean_df['is_permissive'] = (clean_df['license_type'] == 'Permissive').astype(int)
    
    clean_df['repository_age_years'] = ((ref_date - created_dt).dt.total_seconds() / (365.25 * 86400)).clip(lower=0.1).round(2)
    clean_df['days_since_last_release'] = ((ref_date - rel_dt).dt.total_seconds() / 86400).clip(lower=0.0).round(1)
    
    clean_df['releases_count'] = raw['releases_count'].astype(int).clip(lower=1)
    clean_df['release_cadence_annual'] = (clean_df['releases_count'] / clean_df['repository_age_years'].clip(lower=0.5)).round(2)
    clean_df['maintainers_count'] = raw['maintainers_count'].astype(int).clip(lower=1)
    clean_df['contributors_count'] = raw['contributors_count'].astype(int).clip(lower=1)
    clean_df['total_commits'] = raw['total_commits'].astype(int).clip(lower=1)
    clean_df['top_contributor_commits'] = raw['top_contributor_commits'].astype(int)
    clean_df['top_2_contributor_commits'] = raw['top_2_contributor_commits'].astype(int)
    
    clean_df['top_contributor_share'] = (clean_df['top_contributor_commits'] / clean_df['total_commits']).clip(0.0, 1.0).round(4)
    clean_df['top_2_contributor_share'] = (clean_df['top_2_contributor_commits'] / clean_df['total_commits']).clip(0.0, 1.0).round(4)
    
    def get_gini(row):
        c_list = json.loads(row['contributor_commits_json']) if isinstance(row['contributor_commits_json'], str) else []
        return calc_gini(c_list)
        
    def get_bf(row):
        c_list = json.loads(row['contributor_commits_json']) if isinstance(row['contributor_commits_json'], str) else []
        return calc_bus_factor(c_list)
        
    clean_df['contributor_gini'] = raw.apply(get_gini, axis=1)
    clean_df['bus_factor_approx'] = raw.apply(get_bf, axis=1)
    
    clean_df['stars_count'] = raw['stars_count'].astype(int)
    clean_df['forks_count'] = raw['forks_count'].astype(int)
    clean_df['subscribers_count'] = raw['subscribers_count'].astype(int)
    
    clean_df['total_issues'] = raw['total_issues_count'].astype(int).clip(lower=1)
    clean_df['open_issues'] = raw['open_issues_count'].astype(int).clip(lower=0)
    clean_df['closed_issues'] = np.maximum(0, clean_df['total_issues'] - clean_df['open_issues'])
    clean_df['issue_resolution_ratio'] = (clean_df['closed_issues'] / clean_df['total_issues']).round(4)
    clean_df['open_issue_ratio'] = (clean_df['open_issues'] / clean_df['total_issues']).round(4)
    
    clean_df['dependencies_count'] = raw['dependencies_count'].astype(int)
    clean_df['dev_dependencies_count'] = raw['dev_dependencies_count'].astype(int)
    
    clean_df['downloads_monthly'] = raw['downloads_monthly'].astype(float).clip(lower=1.0)
    clean_df['log_downloads_monthly'] = np.log10(clean_df['downloads_monthly'] + 1).round(4)
    
    clean_df['dependents_count'] = raw['dependents_count'].astype(float).clip(lower=1.0)
    clean_df['log_dependents'] = np.log10(clean_df['dependents_count'] + 1).round(4)
    
    dl_min, dl_max = clean_df['log_downloads_monthly'].min(), clean_df['log_downloads_monthly'].max()
    dep_min, dep_max = clean_df['log_dependents'].min(), clean_df['log_dependents'].max()
    norm_dl = (clean_df['log_downloads_monthly'] - dl_min) / (dl_max - dl_min) if dl_max > dl_min else 0
    norm_dep = (clean_df['log_dependents'] - dep_min) / (dep_max - dep_min) if dep_max > dep_min else 0
    clean_df['blast_radius_score'] = ((0.6 * norm_dl + 0.4 * norm_dep) * 100).round(2)
    
    def assign_impact_tier(score):
        if score >= 60.0: return 'Critical'
        elif score >= 40.0: return 'High'
        elif score >= 20.0: return 'Medium'
        return 'Low'
    clean_df['impact_tier'] = clean_df['blast_radius_score'].apply(assign_impact_tier)
    
    clean_df['is_deprecated'] = raw['is_deprecated'].astype(int)
    clean_df['has_test_script'] = raw['has_test_script'].astype(int)
    
    clean_df['risk_class'] = clean_df.apply(assign_risk_class, axis=1)
    
    clean_path = 'data/cleaned_dataset.csv'
    clean_df.to_csv(clean_path, index=False)
    print(f"Cleaned dataset saved successfully to {clean_path} with {len(clean_df)} records and {len(clean_df.columns)} columns!")
    print("\nNull counts total:", clean_df.isnull().sum().sum())
    print("\nTarget Risk Class Distribution:")
    print(clean_df['risk_class'].value_counts())

if __name__ == '__main__':
    main()
