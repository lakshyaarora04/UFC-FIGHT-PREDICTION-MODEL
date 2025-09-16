from typing import Dict, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

from .models import build_models_with_ensemble


class UFCFightPredictor:
	def __init__(self, random_state: int = 42):
		self.random_state = random_state
		self.models = None
		self.fitted_model = None

	def train(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
		self.models = build_models_with_ensemble(df)
		X = df.copy()
		y = None
		if 'winner' in X.columns:
			y = X['winner'].astype(int)
			X = X.drop(columns=['winner'])

		results: Dict[str, Dict[str, float]] = {}
		for name, pipeline in self.models.items():
			if y is None:
				continue
			# Skip evaluating the final 'ensemble' during CV; it can be slow and redundant
			# but we still fit it on full data later
			if name == 'ensemble':
				continue
			cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
			accs, precs, recs, f1s = [], [], [], []
			try:
				for tr_idx, va_idx in cv.split(X, y):
					X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
					y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
					pipeline.fit(X_tr, y_tr)
					pred = pipeline.predict(X_va)
					accs.append(accuracy_score(y_va, pred))
					precs.append(precision_score(y_va, pred, zero_division=0))
					recs.append(recall_score(y_va, pred, zero_division=0))
					f1s.append(f1_score(y_va, pred, zero_division=0))
				results[name] = {
					'accuracy': float(np.mean(accs)) if accs else 0.0,
					'precision': float(np.mean(precs)) if precs else 0.0,
					'recall': float(np.mean(recs)) if recs else 0.0,
					'f1': float(np.mean(f1s)) if f1s else 0.0,
				}
			except Exception:
				# Skip models that fail (e.g., dependency issues) without halting training
				continue

		# Fit final ensemble (or fallback) on full data if available
		if 'ensemble' in self.models and y is not None:
			self.fitted_model = self.models['ensemble']
			self.fitted_model.fit(X, y)
		return results

	def predict_fight(self, fighter_1_stats: Dict[str, Any], fighter_2_stats: Dict[str, Any]) -> Dict[str, Any]:
		if self.fitted_model is None:
			raise ValueError('Model not trained. Call train() or load_model() first.')
		row = {**fighter_1_stats, **fighter_2_stats}
		df = pd.DataFrame([row])
		proba = self.fitted_model.predict_proba(df)[0]
		prediction = int(np.argmax(proba))
		return {
			'prediction': prediction,
			'confidence': float(np.max(proba)),
			'probabilities': {
				'fighter_1': float(proba[1]),
				'fighter_2': float(proba[0]),
			},
		}

	def save_model(self, path: str):
		if self.fitted_model is None:
			raise ValueError('No trained model to save.')
		joblib.dump(self.fitted_model, path)
		return path

	def load_model(self, path: str):
		self.fitted_model = joblib.load(path)
		return self

	def get_feature_importance(self) -> pd.DataFrame:
		if self.fitted_model is None:
			raise ValueError('Model not trained.')
		model = getattr(self.fitted_model.named_steps['model'], 'estimators_', None)
		importances = []
		if model is None:
			base = self.fitted_model.named_steps['model']
			if hasattr(base, 'feature_importances_'):
				vals = base.feature_importances_
				for i, val in enumerate(vals):
					importances.append(('feature_' + str(i), float(val)))
		else:
			agg = None
			count = 0
			for est_name, est in model:
				if hasattr(est, 'feature_importances_'):
					vals = np.asarray(est.feature_importances_)
					agg = vals if agg is None else agg + vals
					count += 1
			if agg is not None and count > 0:
				agg = agg / count
				for i, val in enumerate(agg):
					importances.append(('feature_' + str(i), float(val)))
		return pd.DataFrame(importances, columns=['feature', 'importance']).sort_values('importance', ascending=False)
