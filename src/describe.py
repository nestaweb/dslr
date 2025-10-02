import pandas as pd
import numpy as np
import sys
import math

def ft_min(data):
	min_val = data[0]
	for feature in data:
		if feature < min_val:
			min_val = feature
	return min_val

def ft_max(data):
	max_val = data[0]
	for feature in data:
		if feature > max_val:
			max_val = feature
	return max_val

def ft_mean(data):
	count = len(data)
	ft_sum = 0
	for feature in data:
		if not pd.isna(feature):
			ft_sum += feature
	return ft_sum / count

def ft_std(data):
	clean_data = [x for x in data if not pd.isna(x)]
	n = len(clean_data)
	if n < 2:
		return float("nan")  # std undefined for < 2 values

	mean = ft_mean(clean_data)
	variance = sum((x - mean) ** 2 for x in clean_data) / (n - 1)  # sample variance
	return math.sqrt(variance)

def ft_percentile(data, p):
	data = sorted(data)
	k = (len(data) - 1) * (p / 100)
	f = int(k)
	c = min(f + 1, len(data) - 1)
	return data[f] + (data[c] - data[f]) * (k - f)

def ft_variance(data):
	mean = ft_mean(data)
	return sum((x - mean) ** 2 for x in data) / len(data) if len(data) > 0 else None

def ft_range(data):
	return ft_max(data) - ft_min(data) if len(data) > 0 else None

def ft_unique_count(data):
	return len(set(data))

def ft_missing_values(data):
	return sum(1 for x in data if pd.isna(x))

def ft_mode(data):
	freq = {}
	for x in data:
		freq[x] = freq.get(x, 0) + 1
	return max(freq, key=freq.get) if freq else None

def ft_describe(df: pd.DataFrame):
	result = {}
	numeric_df = df.select_dtypes(include=[np.number])  # do this to only keep numerical values

	for col in numeric_df.columns: # iterate over the features name list
		data = numeric_df[col].dropna().values # get all the data in the data array
		stats = {}
		stats["Count"] = len(data)
		stats["Mean"] = ft_mean(data)
		stats["Std"] = ft_std(data)
		stats["Min"] = ft_min(data)
		stats["25%"] = ft_percentile(data, 25)
		stats["50%"] = ft_percentile(data, 50)
		stats["75%"] = ft_percentile(data, 75)
		stats["Max"] = ft_max(data)
		stats["Range"] = ft_range(data)
		stats["Unique"] = ft_unique_count(data)
		stats["Missing"] = ft_missing_values(df[col].values)
		stats["Mode"] = ft_mode(data)
		result[col] = stats

	# Convert to DataFrame for nice printing
	return pd.DataFrame(result)

def read_dataset(path: str) -> None:
	try:
		df = pd.read_csv(path)
	except FileNotFoundError:
		print(f"Error: file '{path}' not found.")
		sys.exit(1)
	except Exception as e:
		print(f"Error reading '{path}': {e}")
		sys.exit(1)

	print(ft_describe(df))

if __name__ == "__main__":
	if len(sys.argv) != 2:
		print("Usage: python script.py <dataset.csv>")
		sys.exit(1)

	read_dataset(sys.argv[1])
