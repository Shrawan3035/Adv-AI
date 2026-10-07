"""
Bank Customer Risk Segmentation Using Expectation-Maximization
Complete Python Script
"""

# ==========================================
# 1. IMPORT LIBRARIES
# ==========================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

import warnings
warnings.filterwarnings("ignore")

# Set seed for reproducibility
np.random.seed(42)
sns.set_theme(style="whitegrid")

# ==========================================
# 2. LOAD DATA
# ==========================================
print("Loading observable feature dataset...")
df_em = pd.read_csv("bank_customer_em.csv")

print("\n--- Dataset Shape ---")
print(df_em.shape)

print("\n--- First 5 Rows ---")
print(df_em.head())

print("\n--- Missing Values ---")
print(df_em.isnull().sum())

# ==========================================
# 3. DATA PREPROCESSING
# ==========================================
# Remove customer_id from feature matrix
df_features = df_em.drop(columns=['customer_id'])

# Handle missing values and duplicates
df_features = df_features.dropna()
df_features = df_features.drop_duplicates()

# Standardize numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_features)
X_scaled_df = pd.DataFrame(X_scaled, columns=df_features.columns)

print("\nData successfully scaled.")

# ==========================================
# 4. EXPLORATORY DATA ANALYSIS (EDA)
# ==========================================
print("\nGenerating EDA plots...")
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

sns.histplot(df_features['annual_income'], bins=30, kde=True, ax=axes[0,0], color='skyblue')
axes[0,0].set_title('Income Distribution')

sns.histplot(df_features['debt_to_income_ratio'], bins=30, kde=True, ax=axes[0,1], color='salmon')
axes[0,1].set_title('Debt-to-Income Ratio Distribution')

sns.histplot(df_features['credit_utilization'], bins=30, kde=True, ax=axes[0,2], color='green')
axes[0,2].set_title('Credit Utilization Distribution')

sns.histplot(df_features['payment_delay_days'], bins=30, kde=True, ax=axes[1,0], color='purple')
axes[1,0].set_title('Payment Delay Distribution')

sns.histplot(df_features['savings_balance'], bins=30, kde=True, ax=axes[1,1], color='orange')
axes[1,1].set_title('Savings Balance Distribution')

sns.heatmap(df_features.corr(), annot=False, cmap='coolwarm', ax=axes[1,2])
axes[1,2].set_title('Feature Correlation Heatmap')

plt.tight_layout()
plt.show()

# ==========================================
# 5. EM ALGORITHM / GMM IMPLEMENTATION
# ==========================================
print("\nTraining Expectation-Maximization (GMM) model...")
gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=42)
gmm.fit(X_scaled_df)
print(f"Final model converged in {gmm.n_iter_} iterations.")

# Show Convergence Plot
gmm_track = GaussianMixture(n_components=3, max_iter=1, warm_start=True, random_state=42)
log_likelihoods = []

for i in range(20):
    gmm_track.fit(X_scaled_df)
    log_likelihoods.append(gmm_track.lower_bound_)

plt.figure(figsize=(8,5))
plt.plot(range(1, 21), log_likelihoods, marker='o', linestyle='-')
plt.title("EM Algorithm Convergence")
plt.xlabel("Iteration")
plt.ylabel("Log-Likelihood")
plt.grid(True)
plt.show()

# ==========================================
# 6. E-STEP RESPONSIBILITIES & M-STEP ASSIGNMENTS
# ==========================================
# E-Step: Calculate probabilities
responsibilities = gmm.predict_proba(X_scaled_df)

# Final cluster assignment
df_em['cluster'] = gmm.predict(X_scaled_df)

print("\nCluster Distribution:")
print(df_em['cluster'].value_counts())

# ==========================================
# 7. INTERPRET THE CLUSTERS
# ==========================================
interpretation_cols = [
    'annual_income', 'monthly_expense', 'savings_balance', 
    'debt_to_income_ratio', 'credit_utilization', 
    'payment_delay_days', 'missed_payment_count', 
    'cluster'
]

cluster_profile = df_em[interpretation_cols].groupby('cluster').mean()
print("\n--- Cluster Profiles (Means) ---")
print(cluster_profile.T)

# Determine mapping based on the profiles
# NOTE: Adjust this dictionary if the cluster means change!
# Based on typical results: 
# Highest debt/delay -> High_Risk
# Lowest debt/delay -> Low_Risk
# Middle -> Medium_Risk

mapping_dict = {
    0: 'Medium_Risk', # Update based on actual means output
    1: 'High_Risk',   # Update based on actual means output
    2: 'Low_Risk'     # Update based on actual means output
}

df_em['predicted_risk_group'] = df_em['cluster'].map(mapping_dict)

# ==========================================
# 8. VISUALIZE DISCOVERED CLUSTERS (PCA)
# ==========================================
print("\nGenerating PCA Cluster Visualization...")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled_df)

df_em['pca_x'] = X_pca[:, 0]
df_em['pca_y'] = X_pca[:, 1]

plt.figure(figsize=(10, 7))
sns.scatterplot(
    x='pca_x', y='pca_y', 
    hue='predicted_risk_group', 
    palette={'Low_Risk': 'green', 'Medium_Risk': 'orange', 'High_Risk': 'red'},
    data=df_em, 
    alpha=0.7
)
plt.title("Customer Segments Discovered by EM (PCA reduced)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend(title='Discovered Cluster')
plt.show()

# ==========================================
# 9. COMPARE WITH GROUND TRUTH
# ==========================================
print("\nLoading ground truth for validation...")
df_truth = pd.read_csv("bank_customer_ground_truth.csv")

# Merge on customer_id
df_final = pd.merge(df_em, df_truth, on='customer_id')

# ==========================================
# 10. METRICS & CONFUSION MATRIX
# ==========================================
labels = ['Low_Risk', 'Medium_Risk', 'High_Risk']

print("\n--- Classification Report ---")
print(classification_report(df_final['true_risk_group'], df_final['predicted_risk_group']))

print(f"Overall Accuracy: {accuracy_score(df_final['true_risk_group'], df_final['predicted_risk_group']):.4f}")
print(f"Adjusted Rand Index (ARI): {adjusted_rand_score(df_final['true_risk_group'], df_final['cluster']):.4f}")
print(f"Normalized Mutual Information (NMI): {normalized_mutual_info_score(df_final['true_risk_group'], df_final['cluster']):.4f}")

cm = confusion_matrix(df_final['true_risk_group'], df_final['predicted_risk_group'], labels=labels)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title("Confusion Matrix: True vs Predicted Risk")
plt.ylabel("True Risk Group")
plt.xlabel("Predicted Risk Group (EM Discovered)")
plt.show()

# ==========================================
# 11. FINAL CUSTOMER RISK PROFILE
# ==========================================
df_final['risk_probability'] = np.max(responsibilities, axis=1)

final_columns = [
    'customer_id', 'predicted_risk_group', 'risk_probability',
    'annual_income', 'debt_to_income_ratio', 'credit_utilization',
    'payment_delay_days', 'missed_payment_count'
]

final_profile_df = df_final[final_columns]

print("\n--- Final Customer Risk Profile (Sample) ---")
print(final_profile_df.sample(10, random_state=42))