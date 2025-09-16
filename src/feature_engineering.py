import pandas as pd
import numpy as np
from typing import List, Tuple
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin


def _infer_pair_feature_basenames(columns: List[str]) -> List[str]:
	fighter_1_prefix = 'fighter_1_'
	fighter_2_prefix = 'fighter_2_'
	fighter_1_cols = {c[len(fighter_1_prefix):] for c in columns if c.startswith(fighter_1_prefix)}
	fighter_2_cols = {c[len(fighter_2_prefix):] for c in columns if c.startswith(fighter_2_prefix)}
	return sorted(list(fighter_1_cols.intersection(fighter_2_cols)))


def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
	df = df.copy()
	pair_basenames = _infer_pair_feature_basenames(df.columns.tolist())

	frames = []
	target_series = df['winner'] if 'winner' in df.columns else None

	# numeric pairwise ops
	num_blocks = {}
	for base in pair_basenames:
		c1 = f'fighter_1_{base}'
		c2 = f'fighter_2_{base}'
		if np.issubdtype(df[c1].dtype, np.number) and np.issubdtype(df[c2].dtype, np.number):
			block = pd.DataFrame(index=df.index)
			block[f'diff_{base}'] = df[c1] - df[c2]
			block[f'ratio_{base}'] = np.where((df[c2] != 0) & np.isfinite(df[c2]), df[c1] / (df[c2] + 1e-8), 0.0)
			block[f'sum_{base}'] = df[c1] + df[c2]
			block[f'avg_{base}'] = (df[c1] + df[c2]) / 2.0
			block[f'max_{base}'] = np.maximum(df[c1], df[c2])
			block[f'min_{base}'] = np.minimum(df[c1], df[c2])
			block[f'lead_{base}'] = (df[c1] > df[c2]).astype(int)
			num_blocks[base] = block
	if num_blocks:
		frames.append(pd.concat(list(num_blocks.values()), axis=1))

	# skill composites
	skill_blocks = []
	for side in [1, 2]:
		skill_terms = []
		for term in ['win_rate', 'sig_str_acc', 'sig_str_def', 'td_def']:
			col = f'fighter_{side}_{term}'
			if col in df.columns and np.issubdtype(df[col].dtype, np.number):
				skill_terms.append(df[col])
		if len(skill_terms) >= 2:
			comp = pd.Series(np.mean(np.column_stack(skill_terms), axis=1), index=df.index, name=f'skill_composite_{side}')
			skill_blocks.append(comp)
	if skill_blocks:
		comp_df = pd.concat(skill_blocks, axis=1)
		comp_out = pd.DataFrame(index=df.index)
		if 'skill_composite_1' in comp_df.columns and 'skill_composite_2' in comp_df.columns:
			comp_out['diff_skill_composite'] = comp_df['skill_composite_1'] - comp_df['skill_composite_2']
			comp_out['ratio_skill_composite'] = np.where(
				(comp_df['skill_composite_2'] != 0) & np.isfinite(comp_df['skill_composite_2']),
				comp_df['skill_composite_1'] / (comp_df['skill_composite_2'] + 1e-8),
				0.0,
			)
			frames.append(comp_out)

	# momentum
	for metric in ['recent_form', 'win_streak', 'activity_level']:
		c1 = f'fighter_1_{metric}'
		c2 = f'fighter_2_{metric}'
		if c1 in df.columns and c2 in df.columns:
			m = pd.DataFrame(index=df.index)
			m[f'diff_{metric}'] = df[c1] - df[c2]
			m[f'ratio_{metric}'] = np.where((df[c2] != 0) & np.isfinite(df[c2]), df[c1] / (df[c2] + 1e-8), 0.0)
			frames.append(m)

	# physical
	for metric in ['height_cms', 'weight_lbs', 'reach_cms', 'age']:
		c1 = f'fighter_1_{metric}'
		c2 = f'fighter_2_{metric}'
		if c1 in df.columns and c2 in df.columns:
			p = pd.DataFrame(index=df.index)
			p[f'diff_{metric}'] = df[c1] - df[c2]
			p[f'ratio_{metric}'] = np.where((df[c2] != 0) & np.isfinite(df[c2]), df[c1] / (df[c2] + 1e-8), 0.0)
			frames.append(p)

	engineered = pd.concat(frames, axis=1) if frames else pd.DataFrame(index=df.index)
	engineered = engineered.replace([np.inf, -np.inf], np.nan).fillna(0.0)
	if target_series is not None:
		engineered['winner'] = target_series.values
	return engineered


class FeatureEngineerTransformer(BaseEstimator, TransformerMixin):
	def __init__(self):
		self.output_features_ = None

	def fit(self, X, y=None):
		engineered = _engineer_features(pd.DataFrame(X))
		self.output_features_ = [c for c in engineered.columns if c != 'winner']
		return self

	def transform(self, X):
		engineered = _engineer_features(pd.DataFrame(X))
		if self.output_features_ is None:
			self.output_features_ = [c for c in engineered.columns if c != 'winner']
		return engineered[self.output_features_]

	def get_feature_names_out(self, input_features=None):
		return np.array(self.output_features_ if self.output_features_ is not None else [])


def build_feature_pipeline(df_sample: pd.DataFrame) -> Tuple[Pipeline, List[str]]:
	engineer = FeatureEngineerTransformer()
	# Fit on sample to determine columns
	engineered_df = _engineer_features(df_sample)
	feature_cols = [c for c in engineered_df.columns if c != 'winner']

	# Determine column types after engineering
	numeric_cols = [c for c in feature_cols if np.issubdtype(engineered_df[c].dtype, np.number)]
	categorical_cols = [c for c in feature_cols if c not in numeric_cols]

	try:
		encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
	except TypeError:
		encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)

	preprocessor = ColumnTransformer(
		transformers=[
			('num', StandardScaler(with_mean=True, with_std=True), numeric_cols),
			('cat', encoder, categorical_cols),
		],
		remainder='drop',
	)

	pipeline = Pipeline(steps=[
		('engineer', engineer),
		('preprocess', preprocessor),
	])
	return pipeline, feature_cols
