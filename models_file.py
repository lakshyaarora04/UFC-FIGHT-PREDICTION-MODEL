from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

try:
	import xgboost as xgb  # type: ignore
	_HAS_XGB = True
except Exception:
	xgb = None
	_HAS_XGB = False

try:
	from lightgbm import LGBMClassifier  # type: ignore
	_HAS_LGBM = True
except Exception:
	LGBMClassifier = None
	_HAS_LGBM = False


class ModelFactory:
	def __init__(self):
		pass
	
	def create_random_forest(self):
		return RandomForestClassifier(
			n_estimators=300,
			max_depth=15,
			min_samples_split=5,
			min_samples_leaf=2,
			max_features='sqrt',
			bootstrap=True,
			random_state=42,
			n_jobs=-1
		)
	
	def create_xgboost(self):
		if not _HAS_XGB:
			raise RuntimeError("XGBoost not available on this system.")
		return xgb.XGBClassifier(
			n_estimators=200,
			max_depth=8,
			learning_rate=0.1,
			subsample=0.8,
			colsample_bytree=0.8,
			random_state=42,
			eval_metric='logloss',
			verbosity=0
		)
	
	def create_lightgbm(self):
		if not _HAS_LGBM:
			raise RuntimeError("LightGBM not available on this system.")
		return LGBMClassifier(
			n_estimators=200,
			max_depth=10,
			learning_rate=0.1,
			num_leaves=31,
			subsample=0.8,
			colsample_bytree=0.8,
			random_state=42,
			verbosity=-1
		)
	
	def create_gradient_boosting(self):
		return GradientBoostingClassifier(
			n_estimators=200,
			max_depth=8,
			learning_rate=0.1,
			subsample=0.8,
			random_state=42
		)
	
	def create_svm(self):
		return SVC(
			kernel='rbf',
			C=1.0,
			gamma='scale',
			probability=True,
			random_state=42
		)
	
	def create_logistic_regression(self):
		return LogisticRegression(
			C=1.0,
			random_state=42,
			max_iter=1000
		)
	
	def create_all_models(self):
		models = {
			'random_forest': self.create_random_forest(),
			'gradient_boosting': self.create_gradient_boosting(),
			'svm': self.create_svm(),
			'logistic': self.create_logistic_regression(),
		}
		if _HAS_XGB:
			models['xgboost'] = self.create_xgboost()
		if _HAS_LGBM:
			models['lightgbm'] = self.create_lightgbm()
		return models