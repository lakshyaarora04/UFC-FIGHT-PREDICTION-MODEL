import joblib
import pandas as pd
import numpy as np

def save_model(model, filepath):
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")

def load_model(filepath):
    model = joblib.load(filepath)
    print(f"Model loaded from {filepath}")
    return model

def create_sample_data(n_samples=1000):
    np.random.seed(42)
    
    data = {
        'fighter_1_height': np.random.normal(175, 10, n_samples),
        'fighter_1_weight': np.random.normal(70, 15, n_samples),
        'fighter_1_reach': np.random.normal(180, 12, n_samples),
        'fighter_1_age': np.random.randint(20, 40, n_samples),
        'fighter_1_wins': np.random.randint(0, 30, n_samples),
        'fighter_1_losses': np.random.randint(0, 15, n_samples),
        'fighter_1_total_fights': lambda x: x['fighter_1_wins'] + x['fighter_1_losses'],
        'fighter_1_win_rate': lambda x: x['fighter_1_wins'] / (x['fighter_1_total_fights'] + 1),
        'fighter_1_years_active': np.random.randint(1, 15, n_samples),
        'fighter_1_recent_form': np.random.uniform(0, 1, n_samples),
        'fighter_1_win_streak': np.random.randint(0, 10, n_samples),
        'fighter_1_fights_per_year': np.random.uniform(1, 4, n_samples),
        'fighter_1_days_since_last_fight': np.random.randint(60, 730, n_samples),
        'fighter_1_sig_str_acc': np.random.uniform(0.3, 0.7, n_samples),
        'fighter_1_sig_str_def': np.random.uniform(0.4, 0.8, n_samples),
        'fighter_1_str_landed_per_min': np.random.uniform(2, 6, n_samples),
        'fighter_1_str_absorbed_per_min': np.random.uniform(1, 5, n_samples),
        'fighter_1_ko_rate': np.random.uniform(0, 0.6, n_samples),
        'fighter_1_td_acc': np.random.uniform(0.2, 0.8, n_samples),
        'fighter_1_td_def': np.random.uniform(0.5, 0.9, n_samples),
        'fighter_1_sub_att_per_15': np.random.uniform(0, 3, n_samples),
        'fighter_1_sub_rate': np.random.uniform(0, 0.4, n_samples),
        'fighter_1_title_fights': np.random.randint(0, 5, n_samples),
        'fighter_1_main_events': np.random.randint(0, 10, n_samples),
        'fighter_1_championship_rounds': np.random.randint(0, 20, n_samples),
        'fighter_1_round_1_win_rate': np.random.uniform(0.3, 0.8, n_samples),
        'fighter_1_late_round_performance': np.random.uniform(0.3, 0.8, n_samples),
        'fighter_1_decision_rate': np.random.uniform(0.2, 0.8, n_samples),
        'fighter_1_finish_rate': np.random.uniform(0.2, 0.8, n_samples),
        'fighter_1_activity_level': np.random.uniform(0.5, 2, n_samples),
        'fighter_1_td_attempts_per_15': np.random.uniform(0, 5, n_samples),
        'fighter_1_control_time_per_fight': np.random.uniform(0, 10, n_samples),
    }
    
    for key in list(data.keys()):
        new_key = key.replace('fighter_1_', 'fighter_2_')
        if callable(data[key]):
            continue
        data[new_key] = np.random.normal(data[key].mean() * 0.95, data[key].std(), n_samples) if isinstance(data[key], np.ndarray) and data[key].dtype == float else np.random.randint(int(data[key].min() * 0.9), int(data[key].max() * 1.1), n_samples)
    
    df = pd.DataFrame()
    for key, value in data.items():
        if callable(value):
            continue
        df[key] = value
    
    df['fighter_1_total_fights'] = df['fighter_1_wins'] + df['fighter_1_losses']
    df['fighter_2_total_fights'] = df['fighter_2_wins'] + df['fighter_2_losses']
    df['fighter_1_win_rate'] = df['fighter_1_wins'] / (df['fighter_1_total_fights'] + 1)
    df['fighter_2_win_rate'] = df['fighter_2_wins'] / (df['fighter_2_total_fights'] + 1)
    
    skill_diff = (df['fighter_1_win_rate'] - df['fighter_2_win_rate'] + 
                 (df['fighter_1_sig_str_acc'] - df['fighter_2_sig_str_acc']) * 0.5)
    noise = np.random.normal(0, 0.1, n_samples)
    win_prob = 1 / (1 + np.exp(-(skill_diff * 3 + noise)))
    df['winner'] = (np.random.random(n_samples) < win_prob).astype(int)
    
    return df

def print_model_performance(results):
    print("\n" + "="*60)
    print("MODEL PERFORMANCE SUMMARY")
    print("="*60)
    
    if not results:
        print("No cross-validation results available. Proceeding with final model training only.")
        print("="*60)
        return
    
    for model_name, metrics in results.items():
        print(f"\n{model_name.upper()}:")
        print("-" * 30)
        for metric, value in metrics.items():
            print(f"{metric.capitalize():12}: {value:.4f}")
    
    best_model = max(results.keys(), key=lambda x: results[x]['accuracy'])
    best_accuracy = results[best_model]['accuracy']
    print(f"\nBEST PERFORMING MODEL: {best_model.upper()}")
    print(f"ACCURACY: {best_accuracy:.4f}")
    print("="*60)

def validate_fighter_data(fighter_data):
    required_fields = [
        'height', 'weight', 'reach', 'age', 'wins', 'losses',
        'win_rate', 'sig_str_acc', 'td_acc', 'ko_rate'
    ]
    
    missing_fields = []
    for field in required_fields:
        if f'fighter_1_{field}' not in fighter_data and f'fighter_2_{field}' not in fighter_data:
            missing_fields.append(field)
    
    if missing_fields:
        raise ValueError(f"Missing required fields: {missing_fields}")
    
    return True