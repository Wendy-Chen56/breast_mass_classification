# Breast Mass Classification

from sklearn.datasets import load_breast_cancer
import pandas as pd

# Load the dataset
data = load_breast_cancer()

# Convert the dataset to a DataFrame
df = pd.DataFrame(data.data, columns=data.feature_names)

# Select the two required features
X = df[["mean radius", "mean texture"]]

# Set malignant = 1 and benign = 0
y = (data.target == 0).astype(int)

# Check the dataset
print("BREAST CANCER DATASET")
print("=" * 50)

print("\nSelected features:")
print(X.columns.tolist())

print("\nFirst 5 rows:")
print(X.head())

print("\nClass counts (1 = malignant, 0 = benign):")
print(pd.Series(y).value_counts().sort_index())

print("\nMissing values:")
print(X.isnull().sum())

from sklearn.model_selection import train_test_split
import numpy as np

# Split row indices
indices = np.arange(len(y))

# First split: 80% development, 20% test
development_idx, test_idx = train_test_split(
    indices,
    test_size=0.20,
    stratify=y,
    random_state=42
)

# Second split: 75% training, 25% validation
train_idx, val_idx = train_test_split(
    development_idx,
    test_size=0.25,
    stratify=y[development_idx],
    random_state=42
)

# Display split results
print("\nDATA SPLIT RESULTS")
print("=" * 50)

for name, idx in [
    ("Training", train_idx),
    ("Validation", val_idx),
    ("Test", test_idx)
]:
    print(f"\n{name}: {len(idx)} records")
    print(f"Benign (0): {(y[idx] == 0).sum()}")
    print(f"Malignant (1): {(y[idx] == 1).sum()}")

# Verify no overlap and full coverage
assert len(set(train_idx) & set(val_idx)) == 0
assert len(set(train_idx) & set(test_idx)) == 0
assert len(set(val_idx) & set(test_idx)) == 0
assert len(np.unique(np.concatenate(
    [train_idx, val_idx, test_idx]
))) == len(y)

print("\nData split verification: PASSED")

from sklearn.preprocessing import StandardScaler

# Select features for each dataset
X_train = X.iloc[train_idx]
X_val = X.iloc[val_idx]
X_test = X.iloc[test_idx]

# Select target labels
y_train = y[train_idx]
y_val = y[val_idx]
y_test = y[test_idx]

# Create the scaler
scaler = StandardScaler()

# Fit only on training data
X_train_scaled = scaler.fit_transform(X_train)

# Apply the same scaler to validation and test data
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Check results
print("\nSTANDARDIZATION RESULTS")
print("=" * 50)

print("\nTraining feature means:")
print(X_train_scaled.mean(axis=0))

print("\nTraining feature standard deviations:")
print(X_train_scaled.std(axis=0))

print("\nScaled training data (first 5 rows):")
print(X_train_scaled[:5])

# Part 3: NumPy Logistic Regression

def sigmoid(z):
    return np.exp(-np.logaddexp(0, -z))


# Test the sigmoid function
print("\nSIGMOID FUNCTION TEST")
print("=" * 50)

test_scores = np.array([-2, -1, 0, 1, 2])

test_probabilities = sigmoid(test_scores)

for score, probability in zip(test_scores, test_probabilities):
    print(f"Score: {score:2d} | Probability: {probability:.4f}")


# Binary Cross-Entropy Loss

def binary_cross_entropy(z, y_true):
    return np.mean(
        np.logaddexp(0, z) - y_true * z
    )


# Test the loss function
print("\nBINARY CROSS-ENTROPY TEST")
print("=" * 50)

test_z = np.array([0.0, 1.0, -1.0])
test_y = np.array([0, 1, 0])

test_loss = binary_cross_entropy(test_z, test_y)

print(f"Binary Cross-Entropy Loss: {test_loss:.4f}")


# Train Logistic Regression with Gradient Descent

# Add a column of ones for the intercept
X_train_aug = np.column_stack([
    np.ones(len(X_train_scaled)),
    X_train_scaled
])

# Initialize coefficients to zero
beta = np.zeros(X_train_aug.shape[1])

# Training settings
learning_rate = 0.1
epochs = 20000

# Store training loss
loss_history = []

# Calculate initial loss
initial_z = X_train_aug @ beta
initial_loss = binary_cross_entropy(initial_z, y_train)
loss_history.append(initial_loss)

# Gradient Descent
for epoch in range(epochs):

    # Calculate scores
    z = X_train_aug @ beta

    # Convert scores to probabilities
    p = sigmoid(z)

    # Calculate gradient
    gradient = X_train_aug.T @ (p - y_train) / len(y_train)

    # Update coefficients
    beta = beta - learning_rate * gradient

    # Record loss after the update
    updated_z = X_train_aug @ beta
    loss = binary_cross_entropy(updated_z, y_train)
    loss_history.append(loss)

# Display training results
print("\nGRADIENT DESCENT TRAINING RESULTS")
print("=" * 50)

print(f"Learning Rate: {learning_rate}")
print(f"Epochs: {epochs}")

print(f"\nInitial Training Loss: {loss_history[0]:.6f}")
print(f"Final Training Loss: {loss_history[-1]:.6f}")

print("\nLearned Coefficients (standardized units):")
print(f"Intercept (beta_0): {beta[0]:.6f}")
print(f"Mean Radius (beta_1): {beta[1]:.6f}")
print(f"Mean Texture (beta_2): {beta[2]:.6f}")

# Plot Training Loss

import matplotlib.pyplot as plt
from pathlib import Path

# Create reports folder
reports_dir = Path(__file__).resolve().parent / "reports"
reports_dir.mkdir(exist_ok=True)

# Plot training loss
plt.figure(figsize=(10, 6))

plt.plot(
    range(len(loss_history)),
    loss_history,
    label="Training Loss"
)

plt.title("Logistic Regression Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Binary Cross-Entropy Loss")
plt.legend()
plt.grid(alpha=0.3)

# Save the figure
plt.savefig(
    reports_dir / "training_loss.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nTraining loss graph saved:")
print(reports_dir / "training_loss.png")

# Scikit-learn Logistic Regression

from sklearn.linear_model import LogisticRegression

# Train the scikit-learn model
sklearn_model = LogisticRegression(
    C=np.inf,
    solver="lbfgs",
    max_iter=10000,
    tol=1e-9
)

sklearn_model.fit(X_train_scaled, y_train)

# Display coefficients
print("\nSCIKIT-LEARN LOGISTIC REGRESSION")
print("=" * 50)

print(f"Intercept (beta_0): {sklearn_model.intercept_[0]:.6f}")
print(f"Mean Radius (beta_1): {sklearn_model.coef_[0][0]:.6f}")
print(f"Mean Texture (beta_2): {sklearn_model.coef_[0][1]:.6f}")

# Compare NumPy and scikit-learn coefficients
print("\nCOEFFICIENT COMPARISON")
print("=" * 50)

print(f"{'Coefficient':<20} {'NumPy':>12} {'Scikit-learn':>15}")

coefficient_names = [
    "Intercept",
    "Mean Radius",
    "Mean Texture"
]

sklearn_beta = np.concatenate([
    sklearn_model.intercept_,
    sklearn_model.coef_[0]
])

for name, numpy_value, sklearn_value in zip(
    coefficient_names, beta, sklearn_beta
):
    print(f"{name:<20} {numpy_value:>12.6f} {sklearn_value:>15.6f}")

#Compare Validation Probabilities

# NumPy model probabilities
X_val_aug = np.column_stack([
    np.ones(len(X_val_scaled)),
    X_val_scaled
])

numpy_val_prob = sigmoid(X_val_aug @ beta)

# Scikit-learn model probabilities
# Verify that column 1 represents malignant (class 1)
print("\nScikit-learn classes:", sklearn_model.classes_)

class_1_index = np.where(sklearn_model.classes_ == 1)[0][0]

sklearn_val_prob = sklearn_model.predict_proba(
    X_val_scaled
)[:, class_1_index]

# Calculate mean absolute difference
mean_prob_diff = np.mean(
    np.abs(numpy_val_prob - sklearn_val_prob)
)

print("\nVALIDATION PROBABILITY COMPARISON")
print("=" * 50)

print(f"Mean Absolute Difference: {mean_prob_diff:.10f}")

print("\nFirst 5 Validation Predictions:")
print(f"{'NumPy':>12} {'Scikit-learn':>15}")

for numpy_p, sklearn_p in zip(
    numpy_val_prob[:5],
    sklearn_val_prob[:5]
):
    print(f"{numpy_p:>12.6f} {sklearn_p:>15.6f}")
    
# Majority Baseline Model

from sklearn.dummy import DummyClassifier

# Train baseline using the same training data
baseline_model = DummyClassifier(strategy="prior")
baseline_model.fit(X_train_scaled, y_train)

print("\nBASELINE MODEL RESULTS")
print("=" * 50)

print("Classes:", baseline_model.classes_)

print("\nTraining class probabilities:")
print(baseline_model.class_prior_)

# Get validation probabilities for class 1 (malignant)
baseline_class_1_index = np.where(
    baseline_model.classes_ == 1
)[0][0]

baseline_val_prob = baseline_model.predict_proba(
    X_val_scaled
)[:, baseline_class_1_index]

# Apply default threshold
baseline_val_pred = (
    baseline_val_prob >= 0.50
).astype(int)

print("\nValidation predictions:")
print("Predicted benign (0):", np.sum(baseline_val_pred == 0))
print("Predicted malignant (1):", np.sum(baseline_val_pred == 1))

# Calculate baseline accuracy
baseline_accuracy = np.mean(
    baseline_val_pred == y_val
)

print(f"\nBaseline Validation Accuracy: {baseline_accuracy:.6f}")

# Evaluate Validation Thresholds

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

thresholds = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]

threshold_results = []

for threshold in thresholds:

    # Convert probabilities to predicted classes
    y_pred = (numpy_val_prob >= threshold).astype(int)

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(
        y_val, y_pred, labels=[0, 1]
    ).ravel()

    # Calculate evaluation metrics
    threshold_results.append({
        "threshold": threshold,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "accuracy": accuracy_score(y_val, y_pred),
        "precision": precision_score(
            y_val, y_pred, zero_division=0
        ),
        "recall": recall_score(
            y_val, y_pred, zero_division=0
        ),
        "f1": f1_score(
            y_val, y_pred, zero_division=0
        )
    })

# Create results table
threshold_df = pd.DataFrame(threshold_results)

print("\nVALIDATION THRESHOLD COMPARISON")
print("=" * 90)

print(threshold_df.to_string(index=False, float_format="%.6f"))

# Select threshold with highest F1
# The threshold list is already ordered from smallest to largest,
# so idxmax selects the smallest threshold if F1 scores tie.
best_index = threshold_df["f1"].idxmax()
selected_threshold = float(
    threshold_df.loc[best_index, "threshold"]
)

print(f"\nSelected Threshold: {selected_threshold:.2f}")
print(f"Best Validation F1: {threshold_df.loc[best_index, 'f1']:.6f}")

# Export required CSV
threshold_df.to_csv(
    reports_dir / "threshold_results.csv",
    index=False,
    float_format="%.6f"
)

print("\nSaved: reports/threshold_results.csv")

# Step 16: Final Validation and Test Evaluation

from sklearn.metrics import log_loss

def evaluate_model(split_name, model_name, policy, threshold,
                   y_true, probabilities):

    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()

    return {
        "split": split_name,
        "model": model_name,
        "policy": policy,
        "threshold": threshold,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(
            y_true, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_true, predictions, zero_division=0
        ),
        "f1": f1_score(
            y_true, predictions, zero_division=0
        ),
        "log_loss": log_loss(
            y_true, probabilities, labels=[0, 1]
        )
    }


# Threshold selected using validation data only.
# Do not change it after evaluating the test data.

final_results = []

for split_name, X_scaled, y_true in [
    ("validation", X_val_scaled, y_val),
    ("test", X_test_scaled, y_test)
]:

    # NumPy probabilities
    X_aug = np.column_stack([
        np.ones(len(X_scaled)),
        X_scaled
    ])
    numpy_prob = sigmoid(X_aug @ beta)

    # Scikit-learn probabilities
    sklearn_prob = sklearn_model.predict_proba(
        X_scaled
    )[:, class_1_index]

    # Baseline probabilities
    baseline_prob = baseline_model.predict_proba(
        X_scaled
    )[:, baseline_class_1_index]

    configurations = [
        ("baseline", "default", 0.50, baseline_prob),
        ("numpy", "default", 0.50, numpy_prob),
        ("numpy", "selected", selected_threshold, numpy_prob),
        ("sklearn", "default", 0.50, sklearn_prob),
        ("sklearn", "selected", selected_threshold, sklearn_prob)
    ]

    for model_name, policy, threshold, probabilities in configurations:
        final_results.append(
            evaluate_model(
                split_name,
                model_name,
                policy,
                threshold,
                y_true,
                probabilities
            )
        )


# Create final results table
results_df = pd.DataFrame(final_results)

print("\nFINAL MODEL EVALUATION")
print("=" * 110)

print(
    results_df.to_string(
        index=False,
        float_format="%.6f"
    )
)

# Export results
results_df.to_csv(
    reports_dir / "assignment_results.csv",
    index=False,
    float_format="%.6f"
)

print("\nSaved: reports/assignment_results.csv")

# Verify required row counts
assert len(threshold_df) == 7
assert len(results_df) == 10

print("\nFinal results verification: PASSED")
