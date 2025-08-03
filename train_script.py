import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from src import UFCFightPredictor, load_and_preprocess_data, print_model_performance

def main():
    parser = argparse.ArgumentParser(description='Train UFC Fight Prediction Model')
    parser.add_argument('--data_path', type=str, default=None, help='Path to training data CSV file')
    parser.add_argument('--output_path', type=str, default='models/', help='Directory to save trained model')
    parser.add_argument('--model_name', type=str, default='ufc_predictor.pkl', help='Name of the saved model file')
    
    args = parser.parse_args()
    
    print("🥊 UFC Fight Prediction Model Training")
    print("=" * 50)
    
    df = load_and_preprocess_data(args.data_path)
    
    predictor = UFCFightPredictor()
    results = predictor.train(df)
    
    print_model_performance(results)
    
    os.makedirs(args.output_path, exist_ok=True)
    model_path = os.path.join(args.output_path, args.model_name)
    predictor.save_model(model_path)
    
    feature_importance = predictor.get_feature_importance()
    print("\nTop 10 Most Important Features:")
    print("-" * 40)
    for i, (_, row) in enumerate(feature_importance.head(10).iterrows(), 1):
        print(f"{i:2d}. {row['feature']:25s} {row['importance']:.4f}")

if __name__ == "__main__":
    main()