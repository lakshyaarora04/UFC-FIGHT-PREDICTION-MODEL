from typing import Dict
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import Pipeline
from .feature_engineering import build_feature_pipeline
from models_file import ModelFactory


def build_models_with_ensemble(df_sample) -> Dict[str, Pipeline]:
	factory = ModelFactory()
	pipeline, feature_cols = build_feature_pipeline(df_sample)

	models = factory.create_all_models()

	wrapped: Dict[str, Pipeline] = {}
	for name, model in models.items():
		wrapped[name] = Pipeline([
			('features', pipeline),
			('model', model),
		])

	estimators = []
	for key in ['random_forest', 'xgboost', 'lightgbm', 'gradient_boosting']:
		if key in models:
			estimators.append((key[:3], models[key]))

	if len(estimators) >= 2:
		ensemble = VotingClassifier(
			estimators=estimators,
			voting='soft',
			weights=[1.0] * len(estimators),
			n_jobs=-1,
		)
		wrapped['ensemble'] = Pipeline([
			('features', pipeline),
			('model', ensemble),
		])
	else:
		# Fallback: pick the strongest single model available
		fallback_key = 'random_forest' if 'random_forest' in models else next(iter(models.keys()))
		wrapped['ensemble'] = Pipeline([
			('features', pipeline),
			('model', models[fallback_key]),
		])

	return wrapped
