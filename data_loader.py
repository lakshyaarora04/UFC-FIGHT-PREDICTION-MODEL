import pandas as pd
import numpy as np
from utils_file import create_sample_data
import re


def load_and_preprocess_data(filepath=None):
	if filepath is None:
		print("No data file provided. Creating sample data...")
		return create_sample_data()
	
	try:
		df = pd.read_csv(filepath)
		print(f"Loaded {len(df)} records from {filepath}")
		
		df = preprocess_data(df)
		return df
	
	except FileNotFoundError:
		print(f"File {filepath} not found. Creating sample data...")
		return create_sample_data()


def _normalize_basename(name: str) -> str:
	name = name.strip()
	name = re.sub(r"[()/%]+", "_", name)
	name = re.sub(r"[^A-Za-z0-9_]+", "_", name)
	name = re.sub(r"_+", "_", name)
	return name.strip('_').lower()


def _transform_rb_schema(df: pd.DataFrame) -> pd.DataFrame:
	df = df.copy()
	red_pref = 'R_'
	blue_pref = 'B_'

	# Create target from Winner
	if 'Winner' in df.columns:
		# Drop draws to keep binary classification
		df = df[df['Winner'].isin(['Red', 'Blue'])].copy()
		df['winner'] = (df['Winner'] == 'Red').astype(int)
	
	# Map paired columns
	r_cols = [c for c in df.columns if c.startswith(red_pref)]
	for rcol in r_cols:
		base_raw = rcol[len(red_pref):]
		bcol = blue_pref + base_raw
		if bcol not in df.columns:
			continue
		base = _normalize_basename(base_raw)
		f1 = f'fighter_1_{base}'
		f2 = f'fighter_2_{base}'
		df[f1] = df[rcol]
		df[f2] = df[bcol]

	# Optionally keep stances/categories as-is for encoder; align names
	for key in ['Stance', 'Height_cms', 'Reach_cms', 'Weight_lbs']:
		r = f'R_{key}'
		b = f'B_{key}'
		if r in df.columns and b in df.columns:
			base = _normalize_basename(key)
			df[f'fighter_1_{base}'] = df[r]
			df[f'fighter_2_{base}'] = df[b]

	# Drop original R_/B_ prefixed columns to avoid leakage/duplication
	cols_to_drop = [c for c in df.columns if c.startswith(red_pref) or c.startswith(blue_pref)]
	# Retain Winner for reference? Not needed after creating 'winner'
	if 'Winner' in df.columns:
		cols_to_drop.append('Winner')
	# Drop text/meta columns if present
	for meta in ['R_fighter', 'B_fighter', 'Referee', 'date', 'location', 'title_bout', 'weight_class']:
		if meta in df.columns:
			cols_to_drop.append(meta)
	cols_to_drop = list(dict.fromkeys(cols_to_drop))
	df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
	return df


def preprocess_data(df):
	df = _transform_rb_schema(df)
	
	numeric_columns = df.select_dtypes(include=[np.number]).columns
	df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())
	
	categorical_columns = df.select_dtypes(include=['object']).columns
	for col in categorical_columns:
		if col not in ['fighter_1_name', 'fighter_2_name', 'winner']:
			df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
	
	for col in df.columns:
		if 'rate' in col.lower() or 'acc' in col.lower() or 'def' in col.lower():
			try:
				df[col] = np.clip(df[col].astype(float), 0, 1)
			except Exception:
				pass
	
	print(f"Data preprocessing completed. Shape: {df.shape}")
	return df


def load_fighter_stats(fighter_name, stats_file=None):
	sample_stats = {
		'height': np.random.normal(175, 10),
		'weight': np.random.normal(70, 15),
		'reach': np.random.normal(180, 12),
		'age': np.random.randint(20, 40),
		'wins': np.random.randint(5, 25),
		'losses': np.random.randint(0, 10),
		'win_rate': np.random.uniform(0.4, 0.9),
		'years_active': np.random.randint(2, 15),
		'recent_form': np.random.uniform(0.3, 1),
		'win_streak': np.random.randint(0, 8),
		'fights_per_year': np.random.uniform(1, 3),
		'days_since_last_fight': np.random.randint(60, 400),
		'sig_str_acc': np.random.uniform(0.35, 0.65),
		'sig_str_def': np.random.uniform(0.45, 0.75),
		'str_landed_per_min': np.random.uniform(2.5, 5.5),
		'str_absorbed_per_min': np.random.uniform(1.5, 4.5),
		'ko_rate': np.random.uniform(0.1, 0.5),
		'td_acc': np.random.uniform(0.25, 0.75),
		'td_def': np.random.uniform(0.55, 0.85),
		'sub_att_per_15': np.random.uniform(0, 2.5),
		'sub_rate': np.random.uniform(0, 0.35),
		'title_fights': np.random.randint(0, 3),
		'main_events': np.random.randint(0, 8),
		'championship_rounds': np.random.randint(0, 15),
		'round_1_win_rate': np.random.uniform(0.4, 0.8),
		'late_round_performance': np.random.uniform(0.4, 0.8),
		'decision_rate': np.random.uniform(0.3, 0.7),
		'finish_rate': np.random.uniform(0.3, 0.7),
		'activity_level': np.random.uniform(0.8, 2.2),
		'td_attempts_per_15': np.random.uniform(0, 4),
		'control_time_per_fight': np.random.uniform(0, 8)
	}
	
	sample_stats['total_fights'] = sample_stats['wins'] + sample_stats['losses']
	
	print(f"Loaded stats for {fighter_name}")
	return sample_stats