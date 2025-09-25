import pandas as pd
import statistics, math
import numpy

def ft_count(data):
	count = 0
	for feature in data:
		if not pd.isna(feature):
			count += 1
	return count

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
	count = ft_count(data)
	ft_sum = 0
	for feature in data:
		if not pd.isna(feature):
			ft_sum += feature
	return ft_sum / count

def ft_std(data):
	count = ft_count(data)
	mean = ft_mean(data)
	ft_sum = 0
	for feature in data:
		if not pd.isna(feature):
			xi = abs(feature - mean)
			ft_sum += xi * xi
	return math.sqrt(ft_sum / count)

def

def read_dataset(path):
	df = pd.read_csv("dataset/dataset_train.csv")
	df_clean = df.dropna(subset=["Arithmancy"])
	print(numpy.std(df_clean["Arithmancy"]))
	print(ft_std(df["Arithmancy"]))

read_dataset("dataset/dataset_train.csv")
