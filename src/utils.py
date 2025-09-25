import pandas as pd
import numpy as np
import csv
from pathlib import Path

class DSLR:
	# Resolve dataset paths relative to project root regardless of CWD
	_project_root = Path(__file__).resolve().parent.parent
	_dataset_dir = _project_root / "dataset"
	trainfilepath = str(_dataset_dir / "dataset_train.csv")
	testfilepath = str(_dataset_dir / "dataset_test.csv")

	def getNumericalFeature():
		df = pd.read_csv(DSLR.trainfilepath)
		numeric_df = df.select_dtypes(include=[np.number])
		columns = list(numeric_df.columns)
		if columns:
			columns = columns[1:]
		return columns