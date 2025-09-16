import importlib
import pandas as pd
import numpy as np
from typing import Optional

_base = importlib.import_module('data_loader')


def load_and_preprocess_data(filepath: Optional[str] = None) -> pd.DataFrame:
	return _base.load_and_preprocess_data(filepath)


def load_fighter_stats(name: str, stats_file: Optional[str] = None):
	return _base.load_fighter_stats(name, stats_file)
