"""
Daily MLB Predictions Generator
Runs each morning to generate predictions for today's games
"""

import pandas as pd
import numpy as np
import pickle
from datetime import datetime, timedelta
from pybaseball import statcast
from scipy.stats import poisson
import json
import os
import subprocess

print("="*60)
print("MLB DAILY PREDICTIONS GENERATOR")
print(f"Date: {datetime.now().strftime('%Y-%m-%d %I:%M %p')}")
print("="*60)

# Configuration - UPDATE THESE PATHS
GITHUB_REPO_PATH = r'C:\Users\bruce\path\to\SportsEdgeBet'  # ⚠️ UPDATE THIS
MODEL_PATH = os.path.join(GITHUB_REPO_PATH, 'models', 'mlb_tb_model.pkl')
FEATURE_INFO_PATH = os.path.join(GITHUB_REPO_PATH, 'models', 'feature_info.pkl')
OUTPUT_FILE = os.path.join(GITHUB_REPO_PATH, 'data', 'predictions.json')

# Load trained model
print("\nLoading model...")
try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(FEATURE_INFO_PATH, 'rb') as f:
        feature_info = pickle.load(f)
    feature_cols = feature_info['feature_cols']
    print(f"✓ Model loaded (trained {feature_info['trained_date']})")
    print(f"  MAE: {feature_info['test_mae']:.3f} | RMSE: {feature_info['test_rmse']:.3f}")
except Exception as e:
    print(f"✗ Error loading model: {e}")
    print("Make sure GITHUB_REPO_PATH is set correctly!")
    input("Press Enter to exit...")
    exit(1)

# Get recent data for rolling averages
today = datetime.now()
lookback_days = 15  # Get last 15 days for rolling averages

print(f"\nDownloading recent batter data (last {lookback_days} days)...")
start_date = (today - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
end_date = today.strftime('%Y-%m-%d')

try:
    recent_data = statcast(start_dt=start_date, end_dt=end_date)
    print(f"✓ Downloaded {len(recent_data)} pitches")
except Exception as e:
    print(f"✗ Error downloading data: {e}")
    input("Press Enter to exit...")
    exit(1)

# Process batter-game stats
print("\nProcessing batter stats...")

# Aggregate basic stats per batter-game
batter_games = recent_data.groupby(['batter', 'game_pk', 'game_date']).agg({
    'estimated_ba_using_speedangle': 'mean',
    'estimated_woba_using_speedangle': 'mean',
    'estimated_slg_using_speedangle': 'mean',
    'launch_speed': 'mean',
    'launch_angle': 'mean'
}).reset_index()

batter_games.columns = ['player_id', 'game_pk', 'game_date', 'xba', 'xwoba', 'xslg', 
                        'launch_speed', 'launch_angle']

# Calculate hard-hit rate
batted_balls = recent_data[recent_data['launch_speed'].notna()].copy()
batted_balls['hard_hit'] = (batted_balls['launch_speed'] >= 95).astype(int)
hardhit = batted_balls.groupby(['batter', 'game_pk']).agg({
    'hard_hit': 'mean'
}).reset_index()
hardhit.columns = ['player_id', 'game_pk', 'hardhit_percent']

batter_games = batter_games.merge(hardhit, on=['player_id', 'game_pk'], how='left')

# Calculate outcomes
outcomes = recent_data[recent_data['events'].notna()].copy()
outcomes['is_hit'] = outcomes['events'].isin(['single', 'double', 'triple', 'home_run']).astype(int)
outcomes['is_hr'] = (outcomes['events'] == 'home_run').astype(int)
outcomes['is_k'] = outcomes['events'].str.contains('strikeout', na=False).astype(int)
outcomes['singles'] = (outcomes['events'] == 'single').astype(int)
outcomes['doubles'] = (outcomes['events'] == 'double').astype(int)
outcomes['triples'] = (outcomes['events'] == 'triple').astype(int)
outcomes['tb'] = outcomes['events'].map({
    'single': 1, 'double': 2, 'triple': 3, 'home_run': 4
}).fillna(0)

outcome_stats = outcomes.groupby(['batter', 'game_pk']).agg({
    'is_hit': 'sum',
    'is_hr': 'sum',
    'is_k': 'sum',
    'tb': 'sum'
}).reset_index()
outcome_stats.columns = ['player_id', 'game_pk', 'hits', 'hrs', 'so', 'target_tb']

batter_games = batter_games.merge(outcome_stats, on=['player_id', 'game_pk'], how='left')

# Calculate K%
pa_counts = outcomes.groupby(['batter', 'game_pk']).size().reset_index(name='pa')
batter_games = batter_games.merge(pa_counts, on=['player_id', 'game_pk'], how='left')
batter_games['k_percent'] = (batter_games['so'] / batter_games['pa']).fillna(0)

# Fill missing values
batter_games = batter_games.fillna(0)

# Calculate rolling 10-game averages
print("Calculating rolling averages...")
batter_games = batter_games.sort_values(['player_id', 'game_date'])

metrics = ['hits', 'target_tb', 'hrs', 'so', 'launch_speed', 'launch_angle', 
           'hardhit_percent', 'xwoba', 'xba', 'xslg', 'k_percent']

def calc_rolling(group, window=10):
    for metric in metrics:
        if metric in group.columns:
            group[f'rolling_{metric}_10'] = group[metric].rolling(window=window, min_periods=1).mean().shift(1)
    return group

batter_games = batter_games.groupby('player_id').apply(calc_rolling).reset_index(drop=True)

# Get most recent stats for each batter
latest_batters = batter_games.groupby('player_id').last().reset_index()
latest_batters = latest_batters.dropna(subset=['rolling_hits_10'])

print(f"✓ Calculated rolling averages for {len(latest_batters)} batters")

# Add pitcher stats (league average for now - can be improved)
print("\n⚠️  Using league-average pitcher stats")
print("   (Future: integrate real starting pitcher matchups)")

latest_batters['pitcher_xba'] = 0.245
latest_batters['pitcher_xwoba'] = 0.315
latest_batters['pitcher_xslg'] = 0.400
latest_batters['pitcher_k_rate'] = 0.22

# Calculate matchup differentials
latest_batters['xba_diff'] = latest_batters['rolling_xba_10'] - latest_batters['pitcher_xba']
latest_batters['xwoba_diff'] = latest_batters['rolling_xwoba_10'] - latest_batters['pitcher_xwoba']
latest_batters['xslg_diff'] = latest_batters['rolling_xslg_10'] - latest_batters['pitcher_xslg']

# Prepare features for prediction
print("\nGenerating predictions...")
X_predict = latest_batters[feature_cols]

# Make predictions
predictions = model.predict(X_predict)
latest_batters['pred_tb'] = predictions

# Calculate probabilities
latest_batters['over_0.5_prob'] = (1 - poisson.cdf(0, predictions)) * 100
latest_batters['over_1.5_prob'] = (1 - poisson.cdf(1, predictions)) * 100
latest_batters['over_2.5_prob'] = (1 - poisson.cdf(2, predictions)) * 100
latest_batters['over_3.5_prob'] = (1 - poisson.cdf(3, predictions)) * 100

# Assign confidence levels
def get_confidence(prob):
    if prob > 70: return "VERY HIGH"
    elif prob > 65: return "HIGH"
    elif prob > 60: return "MEDIUM"
    elif prob < 35: return "HIGH (UNDER)"
    elif prob < 40: return "MEDIUM (UNDER)"
    else: return "LOW"

latest_batters['confidence'] = latest_batters['over_1.5_prob'].apply(get_confidence)

# Determine betting recommendation
def get_recommendation(row):
    if row['over_1.5_prob'] > 67.5:
        return {"bet": "OVER 1.5 TB", "confidence": row['confidence']}
    elif row['over_1.5_prob'] < 37.5:
        return {"bet": "UNDER 1.5 TB", "confidence": "HIGH (UNDER)"}
    elif row['over_2.5_prob'] > 60:
        return {"bet": "OVER 2.5 TB", "confidence": "MEDIUM"}
    else:
        return {"bet": "No strong edge", "confidence": "LOW"}

latest_batters['recommendation'] = latest_batters.apply(get_recommendation, axis=1)

# Format for JSON output
predictions_list = []
for _, row in latest_batters.iterrows():
    rec = row['recommendation']
    predictions_list.append({
        "player_id": int(row['player_id']),
        "predicted_tb": round(float(row['pred_tb']), 2),
        "probabilities": {
            "over_0.5": round(float(row['over_0.5_prob']), 1),
            "over_1.5": round(float(row['over_1.5_prob']), 1),
            "over_2.5": round(float(row['over_2.5_prob']), 1),
            "over_3.5": round(float(row['over_3.5_prob']), 1)
        },
        "recommendation": rec['bet'],
        "confidence": rec['confidence'],
        "recent_stats": {
            "avg_tb_last_10": round(float(row['rolling_target_tb_10']), 2),
            "avg_exit_velo": round(float(row['rolling_launch_speed_10']), 1),
            "hard_hit_rate": round(float(row['rolling_hardhit_percent_10']) * 100, 1),
            "xwoba": round(float(row['rolling_xwoba_10']), 3)
        }
    })

# Sort by confidence and probability
def sort_key(x):
    if "OVER" in x['recommendation'] and x['probabilities']['over_1.5'] > 67.5:
        return (0, -x['probabilities']['over_1.5'])
    elif "UNDER" in x['recommendation'] and x['probabilities']['over_1.5'] < 37.5:
        return (1, x['probabilities']['over_1.5'])
    else:
        return (2, -x['probabilities']['over_1.5'])

predictions_list = sorted(predictions_list, key=sort_key)

# Create output JSON
output = {
    "generated_at": datetime.now().isoformat(),
    "model_version": feature_info['model_version'],
    "model_performance": {
        "mae": round(float(feature_info['test_mae']), 3),
        "rmse": round(float(feature_info['test_rmse']), 3)
    },
    "total_players": len(predictions_list),
    "high_confidence_picks": [p for p in predictions_list if "HIGH" in p['confidence']],
    "all_predictions": predictions_list
}

# Save to file
print(f"\nSaving predictions...")
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
with open(OUTPUT_FILE, 'w') as f:
    json.dump(output, f, indent=2)

print(f"✓ Saved {len(predictions_list)} predictions to predictions.json")

# Display top picks
print("\n" + "="*60)
print("TOP 10 HIGH-CONFIDENCE PICKS")
print("="*60)
for i, pred in enumerate(output['high_confidence_picks'][:10], 1):
    print(f"{i}. Player {pred['player_id']}")
    print(f"   {pred['recommendation']} - {pred['confidence']}")
    print(f"   Predicted TB: {pred['predicted_tb']} | Prob: {pred['probabilities']['over_1.5']:.1f}%")
    print()

# Auto-commit and push to GitHub
print("="*60)
print("PUSHING TO GITHUB")
print("="*60)

try:
    os.chdir(GITHUB_REPO_PATH)
    subprocess.run(['git', 'add', 'data/predictions.json'], check=True)
    commit_msg = f"Auto-update predictions - {datetime.now().strftime('%Y-%m-%d %I:%M %p')}"
    subprocess.run(['git', 'commit', '-m', commit_msg], check=True)
    subprocess.run(['git', 'push'], check=True)
    print("✓ Successfully pushed to GitHub")
    print("  Your website will update automatically!")
except subprocess.CalledProcessError as e:
    print(f"✗ Git command failed: {e}")
    print("\nManual steps:")
    print("1. Open Git Bash or Command Prompt")
    print(f"2. cd {GITHUB_REPO_PATH}")
    print("3. git add data/predictions.json")
    print(f"4. git commit -m 'Update predictions'")
    print("5. git push")
except Exception as e:
    print(f"✗ Error: {e}")

print("\n" + "="*60)
print("COMPLETE")
print("="*60)

input("\nPress Enter to exit...")
