from src import UFCFightPredictor, load_and_preprocess_data, print_model_performance

def main():
    print("🥊 UFC Fight Prediction Model - Example Usage")
    print("=" * 60)
    
    df = load_and_preprocess_data()
    print(f"Dataset shape: {df.shape}")
    
    predictor = UFCFightPredictor()
    print("\nTraining models...")
    results = predictor.train(df)
    
    print_model_performance(results)
    
    print("\nFeature Importance (Top 10):")
    print("-" * 40)
    feature_importance = predictor.get_feature_importance(10)
    for i, (_, row) in enumerate(feature_importance.iterrows(), 1):
        print(f"{i:2d}. {row['feature']:25s} {row['importance']:.4f}")
    
    print("\n" + "=" * 60)
    print("EXAMPLE FIGHT PREDICTION")
    print("=" * 60)
    
    fighter_1_stats = {
        'fighter_1_height': 185,
        'fighter_1_weight': 84,
        'fighter_1_reach': 195,
        'fighter_1_age': 28,
        'fighter_1_wins': 20,
        'fighter_1_losses': 2,
        'fighter_1_total_fights': 22,
        'fighter_1_win_rate': 0.91,
        'fighter_1_years_active': 8,
        'fighter_1_recent_form': 0.85,
        'fighter_1_win_streak': 5,
        'fighter_1_fights_per_year': 2.5,
        'fighter_1_days_since_last_fight': 180,
        'fighter_1_sig_str_acc': 0.58,
        'fighter_1_sig_str_def': 0.72,
        'fighter_1_str_landed_per_min': 4.2,
        'fighter_1_str_absorbed_per_min': 2.1,
        'fighter_1_ko_rate': 0.45,
        'fighter_1_td_acc': 0.65,
        'fighter_1_td_def': 0.88,
        'fighter_1_sub_att_per_15': 1.2,
        'fighter_1_sub_rate': 0.18,
        'fighter_1_title_fights': 3,
        'fighter_1_main_events': 8,
        'fighter_1_championship_rounds': 15,
        'fighter_1_round_1_win_rate': 0.75,
        'fighter_1_late_round_performance': 0.68,
        'fighter_1_decision_rate': 0.35,
        'fighter_1_finish_rate': 0.65,
        'fighter_1_activity_level': 2.0,
        'fighter_1_td_attempts_per_15': 2.8,
        'fighter_1_control_time_per_fight': 3.5
    }
    
    fighter_2_stats = {
        'fighter_2_height': 180,
        'fighter_2_weight': 84,
        'fighter_2_reach': 188,
        'fighter_2_age': 32,
        'fighter_2_wins': 18,
        'fighter_2_losses': 5,
        'fighter_2_total_fights': 23,
        'fighter_2_win_rate': 0.78,
        'fighter_2_years_active': 10,
        'fighter_2_recent_form': 0.70,
        'fighter_2_win_streak': 2,
        'fighter_2_fights_per_year': 2.2,
        'fighter_2_days_since_last_fight': 220,
        'fighter_2_sig_str_acc': 0.52,
        'fighter_2_sig_str_def': 0.65,
        'fighter_2_str_landed_per_min': 3.8,
        'fighter_2_str_absorbed_per_min': 2.8,
        'fighter_2_ko_rate': 0.35,
        'fighter_2_td_acc': 0.72,
        'fighter_2_td_def': 0.75,
        'fighter_2_sub_att_per_15': 0.8,
        'fighter_2_sub_rate': 0.22,
        'fighter_2_title_fights': 1,
        'fighter_2_main_events': 5,
        'fighter_2_championship_rounds': 8,
        'fighter_2_round_1_win_rate': 0.65,
        'fighter_2_late_round_performance': 0.72,
        'fighter_2_decision_rate': 0.43,
        'fighter_2_finish_rate': 0.57,
        'fighter_2_activity_level': 1.8,
        'fighter_2_td_attempts_per_15': 3.2,
        'fighter_2_control_time_per_fight': 4.1
    }
    
    print("Fighter 1 Profile:")
    print(f"  Record: {fighter_1_stats['fighter_1_wins']}-{fighter_1_stats['fighter_1_losses']}")
    print(f"  Win Rate: {fighter_1_stats['fighter_1_win_rate']:.1%}")
    print(f"  Height/Reach: {fighter_1_stats['fighter_1_height']}cm / {fighter_1_stats['fighter_1_reach']}cm")
    print(f"  Age: {fighter_1_stats['fighter_1_age']} years")
    
    print("\nFighter 2 Profile:")
    print(f"  Record: {fighter_2_stats['fighter_2_wins']}-{fighter_2_stats['fighter_2_losses']}")
    print(f"  Win Rate: {fighter_2_stats['fighter_2_win_rate']:.1%}")
    print(f"  Height/Reach: {fighter_2_stats['fighter_2_height']}cm / {fighter_2_stats['fighter_2_reach']}cm")
    print(f"  Age: {fighter_2_stats['fighter_2_age']} years")
    
    result = predictor.predict_fight(fighter_1_stats, fighter_2_stats)
    
    print(f"\n🏆 PREDICTION: {result['prediction']}")
    print(f"📊 CONFIDENCE: {result['confidence']:.1%}")
    print(f"\n📈 Win Probabilities:")
    print(f"   Fighter 1: {result['probabilities']['fighter_1']:.1%}")
    print(f"   Fighter 2: {result['probabilities']['fighter_2']:.1%}")
    
    predictor.save_model('models/ufc_predictor.pkl')
    print(f"\n✅ Model saved successfully!")

if __name__ == "__main__":
    main()