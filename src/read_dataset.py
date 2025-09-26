import pandas as pd
import statistics, math
import numpy
from describe import ft_describe

def read_dataset(path):
	df = pd.read_csv("dataset/dataset_train.csv")
	print(ft_describe(df))

read_dataset("dataset/dataset_train.csv")
