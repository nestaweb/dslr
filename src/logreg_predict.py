import sys
import csv
from pathlib import Path

import numpy as np
import pandas as pd

from utils import DSLR


def _sigmoid(z: np.ndarray) -> np.ndarray:
	"""
	Compute the sigmoid function: sig(z) = 1 / (1 + e^(-z))
	Maps any real value to a probability between 0 and 1.
	"""
	return 1.0 / (1.0 + np.exp(-z))


def _load_weights(weights_path: Path) -> tuple[dict[str, np.ndarray], list[str], np.ndarray, np.ndarray]:
	"""
	Load trained weights and normalization parameters from CSV file.
	
	The weights file contains:
	- Header row with feature names
	- One row per class with trained theta values
	- Two special rows (__MEAN__ and __STD__) with normalization parameters
	
	Args:
		weights_path: Path to the weights.csv file
	
	Returns:
		weights: Dictionary mapping class name -> theta vector
		feature_names: List of feature names (excluding intercept)
		means: Mean values for feature normalization
		stds: Standard deviation values for feature normalization
	"""
	try:
		df = pd.read_csv(weights_path)
	except FileNotFoundError:
		raise FileNotFoundError(f"Weights file not found: {weights_path}")
	except Exception as e:
		raise RuntimeError(f"Error reading weights file: {e}")
	
	if df.shape[0] < 3:  # At least 1 class + mean + std rows
		raise ValueError("Invalid weights file format: too few rows")
	
	# Extract header to get feature names (skip 'class' and 'intercept' columns)
	if 'class' not in df.columns or 'intercept' not in df.columns:
		raise ValueError("Weights file must have 'class' and 'intercept' columns")
	
	feature_names = [col for col in df.columns if col not in ['class', 'intercept']]
	
	# Separate class weights from normalization parameters
	# Normalization rows are identified by special labels
	class_rows = df[~df['class'].isin(['__MEAN__', '__STD__'])]
	mean_row = df[df['class'] == '__MEAN__']
	std_row = df[df['class'] == '__STD__']
	
	if mean_row.empty or std_row.empty:
		raise ValueError("Weights file missing normalization parameters (__MEAN__ and __STD__ rows)")
	
	# Build weights dictionary
	weights: dict[str, np.ndarray] = {}
	for _, row in class_rows.iterrows():
		cls = str(row['class'])
		# Theta includes intercept as first element, then feature weights
		theta = np.array([row['intercept']] + [row[f] for f in feature_names], dtype=float)
		weights[cls] = theta
	
	# Extract normalization parameters (exclude 'class' and 'intercept' columns)
	means = np.array([mean_row.iloc[0][f] for f in feature_names], dtype=float)
	stds = np.array([std_row.iloc[0][f] for f in feature_names], dtype=float)
	
	if len(weights) == 0:
		raise ValueError("No class weights found in weights file")
	
	return weights, feature_names, means, stds


def _normalize_features(X: np.ndarray, means: np.ndarray, stds: np.ndarray) -> np.ndarray:
	"""
	Normalize features using the same parameters from training.
	
	CRITICAL: We must use the exact same mean and std values that were
	computed during training, NOT the statistics of the test set.
	This ensures consistency between training and prediction.
	
	Formula: X_normalized = (X - train_mean) / train_std
	
	Args:
		X: Feature matrix to normalize (n_samples, n_features)
		means: Mean values from training data
		stds: Standard deviation values from training data
	
	Returns:
		Normalized feature matrix
	"""
	return (X - means) / stds


def _prepare_features(df: pd.DataFrame, feature_names: list[str], 
                      means: np.ndarray, stds: np.ndarray) -> np.ndarray:
	"""
	Prepare feature matrix for prediction.
	
	Steps:
	1. Extract the same features that were used during training
	2. Fill missing values with column means (from current data)
	3. Normalize using training data statistics
	4. Add bias term as first column
	
	Args:
		df: Test dataframe
		feature_names: List of feature names to use (from training)
		means: Mean values from training (for normalization)
		stds: Standard deviation values from training (for normalization)
	
	Returns:
		Design matrix ready for prediction (n_samples, n_features+1)
	"""
	# Check that all required features are present
	missing_features = [f for f in feature_names if f not in df.columns]
	if missing_features:
		raise ValueError(f"Test data missing required features: {missing_features}")
	
	# Extract features in the same order as training
	X = df[feature_names].astype(float).copy()
	
	# Impute missing values with column means from test data
	# (Ideally we could use training means, but test set may have different missingness patterns)
	col_means = X.mean(axis=0)
	X = X.fillna(col_means)
	
	# Convert to numpy and normalize using TRAINING statistics
	X_np = X.to_numpy()
	X_normalized = _normalize_features(X_np, means, stds)
	
	# Add bias term as first column
	bias = np.ones((X_normalized.shape[0], 1), dtype=float)
	X_design = np.concatenate([bias, X_normalized], axis=1)
	
	return X_design


def _predict(X: np.ndarray, weights: dict[str, np.ndarray]) -> np.ndarray:
	"""
	Make predictions using one-vs-all logistic regression.
	
	For each sample:
	1. Compute probability for each class using its trained weights
	2. Predict the class with highest probability
	
	This implements the prediction rule:
		y_pred = argmax_c P(y=c|X) = argmax_c σ(X * theta_c)
	
	Args:
		X: Design matrix (n_samples, n_features+1) with bias
		weights: Dictionary mapping class name -> theta vector
	
	Returns:
		Array of predicted class labels (n_samples,)
	"""
	n_samples = X.shape[0]
	classes = sorted(weights.keys())
	
	# Compute probability for each class
	# Shape: (n_samples, n_classes)
	probabilities = np.zeros((n_samples, len(classes)), dtype=float)
	
	for i, cls in enumerate(classes):
		theta = weights[cls]
		# Compute P(y=class|X) = sigmoid(X * theta)
		probabilities[:, i] = _sigmoid(X @ theta)
	
	# For each sample, predict class with maximum probability
	predicted_indices = np.argmax(probabilities, axis=1)
	predictions = np.array([classes[idx] for idx in predicted_indices])
	
	return predictions


def _save_predictions(predictions: np.ndarray, indices: np.ndarray, out_path: Path) -> None:
	"""
	Save predictions to CSV in the required format.
	
	Output format:
		Index,Hogwarts House
		0,Gryffindor
		1,Hufflepuff
		...
	
	Args:
		predictions: Array of predicted class labels
		indices: Array of sample indices (from original dataset)
		out_path: Path where to save predictions
	"""
	out_path.parent.mkdir(parents=True, exist_ok=True)
	
	with open(out_path, mode="w", newline="") as f:
		writer = csv.writer(f)
		
		# Write header
		writer.writerow(["Index", "Hogwarts House"])
		
		# Write predictions with their indices
		for idx, pred in zip(indices, predictions):
			writer.writerow([int(idx), pred])


def main(argv: list[str]) -> None:
	"""
	Main prediction function.
	
	Usage: python -m src.logreg_predict <dataset_test.csv> [weights.csv]
	
	Reads test data and trained weights, makes predictions,
	and saves results to houses.csv
	"""
	if len(argv) < 2 or len(argv) > 3:
		print("Usage: python -m src.logreg_predict <dataset_test.csv> [weights.csv]")
		sys.exit(1)
	
	# Parse command line arguments
	test_path = Path(argv[1])
	
	if len(argv) == 3:
		weights_path = Path(argv[2])
	else:
		# Default weights path
		project_root = Path(DSLR._project_root)
		weights_path = project_root / "weights.csv"
	
	# Load test dataset
	try:
		df = pd.read_csv(test_path)
	except FileNotFoundError:
		print(f"Error: test file '{test_path}' not found")
		sys.exit(1)
	except Exception as e:
		print(f"Error reading test file '{test_path}': {e}")
		sys.exit(1)
	
	# Load trained weights and normalization parameters
	try:
		weights, feature_names, means, stds = _load_weights(weights_path)
	except Exception as e:
		print(f"Error loading weights: {e}")
		sys.exit(1)
	
	print(f"Loaded weights for {len(weights)} classes: {sorted(weights.keys())}")
	print(f"Using {len(feature_names)} features")
	
	# Prepare features for prediction
	try:
		X = _prepare_features(df, feature_names, means, stds)
	except Exception as e:
		print(f"Error preparing features: {e}")
		sys.exit(1)
	
	# Make predictions
	predictions = _predict(X, weights)
	
	# Get indices from dataset (or create them if not present)
	if 'Index' in df.columns:
		indices = df['Index'].to_numpy()
	else:
		indices = np.arange(len(df))
	
	# Save predictions
	project_root = Path(DSLR._project_root)
	out_file = project_root / "houses.csv"
	_save_predictions(predictions, indices, out_file)
	
	print(f"✓ Prediction complete!")
	print(f"✓ Predicted {len(predictions)} samples")
	print(f"✓ Saved predictions to {out_file}")
	
	# Show prediction distribution
	unique, counts = np.unique(predictions, return_counts=True)
	print("\nPrediction distribution:")
	for cls, count in zip(unique, counts):
		print(f"  {cls}: {count} ({100*count/len(predictions):.1f}%)")


if __name__ == "__main__":
	main(sys.argv)