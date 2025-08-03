import pandas as pd
import numpy as np
from .utils import create_sample_data

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

def preprocess_data(df):
    df = df.copy()
    
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())
    
    categorical_columns = df.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        if col not in ['fighter_1_name', 'fighter_2_name', 'winner']:
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
    
    for col in df.columns:
        if 'rate' in col.lower() or 'acc' in col.lower() or 'def' in col.lower():
            df[col] = np.clip(df[col], 0, 1)
    
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