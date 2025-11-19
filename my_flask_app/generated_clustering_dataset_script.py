"""
High-Quality Clustering Script
Generated using Pipeline for optimal preprocessing
Model: KMeans
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

# Load Data
print("📄 Loading dataset...")
df = pd.read_csv('test_clustering.csv')
print(f"Dataset shape: {df.shape}")

# Define Features
print("\n🎯 Defining features...")
columns_to_drop = ['customer_id']

# The features X are all columns except ['customer_id']
feature_columns = [col for col in df.columns if col not in columns_to_drop]
X = df[feature_columns]

print(f"Number of features: {len(feature_columns)}")
print(f"Features: {feature_columns}")

# Automatic Preprocessing
print("\n🔄 Setting up preprocessing pipeline...")

# Identify numerical and categorical columns  
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

print(f"Numerical features: {numeric_features}")
print(f"Categorical features: {categorical_features}")

# Create preprocessing pipelines
numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer([
    ('num', numeric_pipeline, numeric_features),
    ('cat', categorical_pipeline, categorical_features)
])

# Create Full Pipeline
print("\n🤖 Creating clustering pipeline...")
model = KMeans(n_init=10, random_state=42)

# Create pipeline that chains preprocessing and clustering
full_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', model)
])

# Train
print("\n🚀 Training clustering model...")
cluster_labels = full_pipeline.fit_predict(X)

print(f"✅ Clustering completed!")
print(f"Number of clusters found: {len(np.unique(cluster_labels))}")

# Evaluate
print("\n📊 Evaluating cluster quality...")

# Get preprocessed data for evaluation
X_preprocessed = full_pipeline.named_steps['preprocessor'].transform(X)

# Calculate Silhouette Score
if len(np.unique(cluster_labels)) > 1:
    silhouette_avg = silhouette_score(X_preprocessed, cluster_labels)
    print(f"Silhouette Score: {silhouette_avg:.4f}")
    
    if silhouette_avg > 0.5:
        print("🏆 Excellent clustering quality!")
    elif silhouette_avg > 0.25:
        print("✅ Good clustering quality")
    else:
        print("⚠️ Clustering quality could be improved")
else:
    print("⚠️ Only one cluster found - consider adjusting parameters")

# Cluster distribution
unique, counts = np.unique(cluster_labels, return_counts=True)
print("\n📈 Cluster Distribution:")
for cluster_id, count in zip(unique, counts):
    percentage = (count / len(cluster_labels)) * 100
    print(f"  Cluster {cluster_id}: {count} points ({percentage:.1f}%)")

# Bonus: Elbow Method (for KMeans)
if 'kmeans' in model_config["class"].lower():
    print("\n🔍 Finding optimal number of clusters (Elbow Method)...")
    
    inertias = []
    k_range = range(1, min(11, len(X)))
    
    for k in k_range:
        kmeans_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', KMeans(n_clusters=k, n_init=10, random_state=42))
        ])
        kmeans_pipeline.fit(X)
        inertias.append(kmeans_pipeline.named_steps['model'].inertia_)
    
    # Plot elbow curve
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, inertias, 'bo-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Optimal k')
    plt.grid(True)
    plt.show()
    
    print("Look for the 'elbow' in the curve to determine optimal k!")

print("\n✅ Clustering analysis completed!")
