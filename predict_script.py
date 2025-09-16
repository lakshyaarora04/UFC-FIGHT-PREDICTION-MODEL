import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import argparse
from src import UFCFightPredictor, load_fighter_stats


def main():
	parser = argparse.ArgumentParser(description='Predict UFC Fight Outcome')
	parser.add_argument('--fighter1', type=str, required=True, help='Fighter 1 name')
	parser.add_argument('--fighter2', type=str, required=True, help='Fighter 2 name')
	parser.add_argument('--model_path', type=str, default='models/ufc_predictor.pkl', help='Path to trained model')
	
	args = parser.parse_args()
	
	print("\ud83c\udf4a UFC Fight Prediction")
	print("=" * 50)
	print(f"Fighter 1: {args.fighter1}")
	print(f"Fighter 2: {args.fighter2}")
	print("-" * 50)
	
	try:
		predictor = UFCFightPredictor()
		predictor = predictor.load_model(args.model_path)
		
		fighter_1_stats = load_fighter_stats(args.fighter1)
		fighter_2_stats = load_fighter_stats(args.fighter2)
		
		fighter_1_prefixed = {f'fighter_1_{k}': v for k, v in fighter_1_stats.items()}
		fighter_2_prefixed = {f'fighter_2_{k}': v for k, v in fighter_2_stats.items()}
		
		result = predictor.predict_fight(fighter_1_prefixed, fighter_2_prefixed)
		
		print(f"\nPREDICTION: {result['prediction']}")
		print(f"CONFIDENCE: {result['confidence']:.2%}")
		print(f"\nDetailed Probabilities:")
		print(f"  {args.fighter1}: {result['probabilities']['fighter_1']:.2%}")
		print(f"  {args.fighter2}: {result['probabilities']['fighter_2']:.2%}")
		
	except FileNotFoundError:
		print(f"Model not found at {args.model_path}")
		print("Please train a model first using: python train_script.py")
	except Exception as e:
		print(f"Error during prediction: {str(e)}")


if __name__ == "__main__":
	main()