import sys
import csv
from pathlib import Path

import numpy as np
import pandas as pd

from utils import DSLR


def _sigmoid(z: np.ndarray) -> np.ndarray:
	"""
	Compute the sigmoid function: sig(z) = 1 / (1 + e^(-z))
	Used to map any real value to a probability between 0 and 1.
	"""
	return 1.0 / (1.0 + np.exp(-z))


def _normalize_features(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""
	Normalize features using standardization (z-score normalization).
	Formula: X_normalized = (X - mean) / std
	
	This ensures all features are on the same scale, which:
	- Makes gradient descent converge faster
	- Prevents features with large values from dominating
	- Improves numerical stability
	
	Args:
		X: Feature matrix (n_samples, n_features) without bias term
	
	Returns:
		X_normalized: Standardized features
		means: Mean of each feature (needed for prediction)
		stds: Standard deviation of each feature (needed for prediction)
	"""
	means = np.mean(X, axis=0)
	stds = np.std(X, axis=0)
	
	# Avoid division by zero for constant features
	# If std is 0, the feature is constant and won't help classification
	stds = np.where(stds == 0, 1.0, stds)
	
	X_normalized = (X - means) / stds
	return X_normalized, means, stds


def _prepare_features(df: pd.DataFrame) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray]:
	"""
	Prepare feature matrix with normalization and bias term.
	
	Steps:
	1. Select numerical features from the dataframe
	2. Fill missing values with column means (imputation)
	3. Normalize features to have mean=0 and std=1
	4. Add bias term (column of 1s) as the first column
	
	Args:
		df: Input dataframe with features and target
	
	Returns:
		X_design: Design matrix (n_samples, n_features+1) with bias
		features: List of feature names in order
		means: Mean values used for normalization
		stds: Standard deviation values used for normalization
	"""
	# Get all numerical features available in the dataset
	features = [f for f in DSLR.getNumericalFeature() if f in df.columns]
	if not features:
		raise RuntimeError("No numeric features available for training")

	# Extract features as float and create a copy to avoid modifying original
	X = df[features].astype(float).copy()
	
	# Impute missing values (NaN) with the mean of each column
	# This is a simple strategy that preserves the distribution
	col_means = X.mean(axis=0)
	X = X.fillna(col_means)

	# Convert to numpy array for numerical operations
	X_np = X.to_numpy()
	
	# Normalize features to same scale
	X_normalized, means, stds = _normalize_features(X_np)
	
	# Add bias term (intercept) as the first column
	# Bias is NOT normalized and always equals 1 for all samples
	bias = np.ones((X_normalized.shape[0], 1), dtype=float)
	X_design = np.concatenate([bias, X_normalized], axis=1)
	
	return X_design, features, means, stds


def _one_vs_all_train(X: np.ndarray, y_labels: np.ndarray, classes: list[str], *,
		alpha: float = 0.1, num_iters: int = 2000) -> dict[str, np.ndarray]:
	"""
	Train one-vs-all (one-vs-rest) logistic regression classifiers.
	
	For multi-class classification with K classes, we train K binary classifiers.
	Each classifier learns to distinguish one class from all others.
	
	Training uses batch gradient descent:
	- theta := theta - alpha * (1/m) * X^T * (h(X*theta) - y)
	where h is the sigmoid function
	
	Args:
		X: Design matrix (n_samples, n_features+1) with bias
		y_labels: Array of class labels for each sample
		classes: List of unique class labels
		alpha: Learning rate (step size for gradient descent)
		num_iters: Number of gradient descent iterations
	
	Returns:
		Dictionary mapping each class to its trained theta vector
	"""
	n_samples, n_features_plus_bias = X.shape
	class_to_theta: dict[str, np.ndarray] = {}

	# Train one binary classifier per class
	for cls in classes:
		# Create binary labels: 1 if sample belongs to current class, 0 otherwise
		y = (y_labels == cls).astype(float)
		
		# Initialize weights to zero
		theta = np.zeros(n_features_plus_bias, dtype=float)
		
		# Gradient descent optimization
		for _ in range(num_iters):
			# Forward pass: compute predictions using sigmoid
			pred = _sigmoid(X @ theta)
			
			# Compute error (difference between prediction and true label)
			error = pred - y
			
			# Compute gradient of cost function with respect to theta
			grad = (X.T @ error) / n_samples
			
			# Update weights in opposite direction of gradient
			theta -= alpha * grad
		
		# Store trained weights for this class
		class_to_theta[cls] = theta.copy()

	return class_to_theta


def _save_weights(weights: dict[str, np.ndarray], feature_names: list[str], 
                  means: np.ndarray, stds: np.ndarray, out_path: Path) -> None:
	"""
	Save trained weights and normalization parameters to CSV.
	
	The CSV format is:
	- First row: header with class, intercept, and all feature names
	- Next K rows: one row per class with its weights
	- Last two rows: normalization parameters (means and stds)
	
	Args:
		weights: Dictionary of class -> theta vector
		feature_names: List of feature names in order
		means: Mean values used for normalization
		stds: Standard deviation values used for normalization
		out_path: Path where to save the weights file
	"""
	headers = ["class", "intercept", *feature_names]
	out_path.parent.mkdir(parents=True, exist_ok=True)
	
	with open(out_path, mode="w", newline="") as f:
		writer = csv.writer(f)
		
		# Write header
		writer.writerow(headers)
		
		# Write weights for each class
		for cls, theta in weights.items():
			row = [cls, *[float(x) for x in theta]]
			writer.writerow(row)
		
		# Save normalization parameters for use during prediction
		# These are needed to normalize test data the same way as training data
		writer.writerow(["__MEAN__", 0.0, *[float(m) for m in means]])
		writer.writerow(["__STD__", 1.0, *[float(s) for s in stds]])


def main(argv: list[str]) -> None:
	"""
	Main training function.
	
	Usage: python -m src.logreg_train <dataset_train.csv>
	
	Reads training data, trains one-vs-all logistic regression,
	and saves weights to weights.csv
	"""
	if len(argv) != 2:
		print("Usage: python -m src.logreg_train <dataset_train.csv>")
		sys.exit(1)

	# Load training dataset
	dataset_path = Path(argv[1])
	try:
		df = pd.read_csv(dataset_path)
	except FileNotFoundError:
		print(f"Error: file '{dataset_path}' not found")
		sys.exit(1)
	except Exception as e:
		print(f"Error reading '{dataset_path}': {e}")
		sys.exit(1)

	# Validate dataset has target column
	if 'Hogwarts House' not in df.columns:
		print("Error: dataset must include 'Hogwarts House' column")
		sys.exit(1)

	# Extract target labels and get unique classes
	y_labels = df['Hogwarts House'].astype(str).to_numpy()
	classes = sorted([c for c in pd.Series(y_labels).dropna().unique()])
	
	if len(classes) < 2:
		print("Error: need at least two classes for training")
		sys.exit(1)

	# Prepare features with normalization
	X, feature_names, means, stds = _prepare_features(df)
	
	print(f"Training on {X.shape[0]} samples with {len(feature_names)} features")
	print(f"Classes: {classes}")
	
	# Train one-vs-all classifiers
	weights = _one_vs_all_train(X, y_labels, classes, alpha=0.1, num_iters=5000)

	# Save trained model
	project_root = Path(DSLR._project_root)
	out_file = project_root / "weights.csv"
	_save_weights(weights, feature_names, means, stds, out_file)
	
	print(f"✓ Training complete!")
	print(f"✓ Saved weights to {out_file}")


if __name__ == "__main__":
	main(sys.argv)