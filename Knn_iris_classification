"""
Project 2: Data Classification Using AI
-----------------------------------------
Goal: Build a basic classification model using a small dataset.

Key Requirements covered:
    1. Load and understand a dataset
    2. Split data into training and testing sets
    3. Apply a simple classification algorithm (K-Nearest Neighbors)

Key Skills demonstrated:
    - Data handling (pandas)
    - Supervised learning basics
    - Model training and evaluation (scikit-learn)

Dataset: Iris flower dataset (built into scikit-learn)
    150 samples, 4 numeric features, 3 target classes (species of iris).
Algorithm: K-Nearest Neighbors (KNN)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)

# ---------------------------------------------------------------------------
# STEP 1: LOAD AND UNDERSTAND THE DATASET
# ---------------------------------------------------------------------------
print("=" * 60)
print("STEP 1: LOAD AND UNDERSTAND THE DATASET")
print("=" * 60)

iris = load_iris()

# Build a pandas DataFrame so the data is easy to inspect/manipulate
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["species"] = iris.target
df["species_name"] = df["species"].map(dict(enumerate(iris.target_names)))

print(f"\nDataset shape: {df.shape[0]} rows x {df.shape[1]} columns")
print("\nFirst 5 rows:")
print(df.head())

print("\nColumn info:")
print(df.info())

print("\nStatistical summary of features:")
print(df.describe())

print("\nClass distribution (how many samples per species):")
print(df["species_name"].value_counts())

print("\nMissing values per column:")
print(df.isnull().sum())

# Quick visual understanding of the dataset: pairwise feature relationships
plt.figure(figsize=(9, 7))
sns.pairplot(df, hue="species_name", vars=iris.feature_names, corner=True)
plt.suptitle("Iris Dataset - Feature Relationships by Species", y=1.02)
plt.savefig("/home/claude/01_dataset_pairplot.png", bbox_inches="tight", dpi=120)
plt.close()
print("\nSaved exploratory pairplot -> 01_dataset_pairplot.png")

# ---------------------------------------------------------------------------
# STEP 2: SPLIT DATA INTO TRAINING AND TESTING SETS
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 2: SPLIT DATA INTO TRAINING AND TESTING SETS")
print("=" * 60)

X = df[iris.feature_names].values   # features
y = df["species"].values            # target labels

# 80% training, 20% testing. stratify=y keeps class proportions balanced
# in both splits, and random_state makes the split reproducible.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTotal samples : {len(X)}")
print(f"Training set  : {len(X_train)} samples")
print(f"Testing set   : {len(X_test)} samples")

# Feature scaling: KNN is distance-based, so features should be on the
# same scale. We fit the scaler ONLY on training data to avoid data leakage,
# then apply the same transform to the test data.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("\nFeatures scaled using StandardScaler (mean=0, std=1).")

# ---------------------------------------------------------------------------
# STEP 3: APPLY A SIMPLE CLASSIFICATION ALGORITHM (KNN)
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 3: TRAIN THE KNN CLASSIFICATION MODEL")
print("=" * 60)

k = 5
model = KNeighborsClassifier(n_neighbors=k)
model.fit(X_train_scaled, y_train)
print(f"\nKNN model trained with k = {k} neighbors.")

# Predict on the unseen test set
y_pred = model.predict(X_test_scaled)

# ---------------------------------------------------------------------------
# STEP 4: EVALUATE THE MODEL
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 4: MODEL EVALUATION")
print("=" * 60)

acc = accuracy_score(y_test, y_pred)
print(f"\nAccuracy on test set: {acc * 100:.2f}%")

print("\nClassification report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix:")
print(cm)

plt.figure(figsize=(5.5, 4.5))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=iris.target_names, yticklabels=iris.target_names
)
plt.xlabel("Predicted species")
plt.ylabel("Actual species")
plt.title(f"KNN (k={k}) Confusion Matrix - Accuracy {acc*100:.1f}%")
plt.tight_layout()
plt.savefig("/home/claude/02_confusion_matrix.png", dpi=120)
plt.close()
print("\nSaved confusion matrix heatmap -> 02_confusion_matrix.png")

# ---------------------------------------------------------------------------
# STEP 5 (BONUS): FIND THE BEST VALUE OF K
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 5 (BONUS): CHOOSING THE BEST K")
print("=" * 60)

k_values = range(1, 16)
accuracies = []
for k_val in k_values:
    m = KNeighborsClassifier(n_neighbors=k_val)
    m.fit(X_train_scaled, y_train)
    pred = m.predict(X_test_scaled)
    accuracies.append(accuracy_score(y_test, pred))

best_k = list(k_values)[int(np.argmax(accuracies))]
print(f"\nBest k found: {best_k} (accuracy = {max(accuracies) * 100:.2f}%)")

plt.figure(figsize=(7, 4.5))
plt.plot(list(k_values), accuracies, marker="o")
plt.xlabel("k (number of neighbors)")
plt.ylabel("Test accuracy")
plt.title("KNN Accuracy vs. k")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("/home/claude/03_k_vs_accuracy.png", dpi=120)
plt.close()
print("Saved k-vs-accuracy plot -> 03_k_vs_accuracy.png")

# ---------------------------------------------------------------------------
# STEP 6: PREDICT ON A NEW, UNSEEN SAMPLE
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 6: PREDICT ON A NEW SAMPLE")
print("=" * 60)

new_sample = np.array([[5.1, 3.5, 1.4, 0.2]])  # looks like a setosa
new_sample_scaled = scaler.transform(new_sample)
prediction = model.predict(new_sample_scaled)[0]
print(f"\nNew flower measurements {new_sample.tolist()[0]}")
print(f"Predicted species: {iris.target_names[prediction]}")

print("\n" + "=" * 60)
print("PROJECT COMPLETE")
print("=" * 60)
