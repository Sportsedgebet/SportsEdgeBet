"""
Daily MLB Predictions Generator - Multi-Prop Version
Generates predictions for Total Bases, Hits, and Strikeouts
"""

import pandas as pd
import numpy as np
import pickle
from datetime import datetime, timedelta
from pybaseball import statcast, playerid_reverse_lookup
from scipy.stats import poisson
import json
import os
import subprocess
import sys
import warnings

# Reduce noisy console output (pybaseball uses tqdm for large queries)
os.environ.setdefault("TQDM_DISABLE", "1")
warnings.filterwarnings("ignore", category=FutureWarning)

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)
from starters_from_statsapi import fetch_starters_for_date

print("="*60)
print("MLB DAILY PREDICTIONS GENERATOR")
print(f"Date: {datetime.now().strftime('%Y-%m-%d %I:%M %p')}")
print("="*60)

# Configuration
GITHUB_REPO_PATH = r'C:\Users\bruce\Github\SportsEdgeBet'
MODEL_PATH = os.path.join(GITHUB_REPO_PATH, 'models', 'mlb_tb_model.pkl')
FEATURE_INFO_PATH = os.path.join(GITHUB_REPO_PATH, 'models', 'feature_info.pkl')
OUTPUT_FILE = os.path.join(GITHUB_REPO_PATH, 'data', 'predictions.json')
TODAY_STARTERS_CSV = os.path.join(GITHUB_REPO_PATH, 'data', 'today_starters.csv')
PREDICTIONS_TODAY_CSV = os.path.join(GITHUB_REPO_PATH, 'data', 'predictions_today.csv')
PREDICTIONS_LIVE_LINEUPS_CSV = os.path.join(GITHUB_REPO_PATH, 'data', 'predictions_live_lineups.csv')
# Optional matchup inputs
PITCHER_STATS_FILE = os.path.join(GITHUB_REPO_PATH, 'models', 'pitcher_season_stats.csv')
TODAY_MATCHUPS_FILE = os.path.join(GITHUB_REPO_PATH, 'data', 'today_matchups.csv')

# Calendar date for Statcast window end + StatsAPI schedule (YYYY-MM-DD).
# Override for backtests: set env MLB_PREDICT_DATE=2025-09-28
RUN_DATE_STR = os.environ.get('MLB_PREDICT_DATE', datetime.now().strftime('%Y-%m-%d'))
today = datetime.strptime(RUN_DATE_STR, '%Y-%m-%d')
# Set MLB_SKIP_LINEUP_FILTER=1 to publish all players (ignore starters join)
SKIP_LINEUP_FILTER = os.environ.get('MLB_SKIP_LINEUP_FILTER', '').lower() in ('1', 'true', 'yes')

# Data-quality + display guardrails
# These prevent "fake" 0.0 projections turning into 0% / 100% win rates.
MIN_GAMES_LAST_10 = 3
MIN_PA_LAST_10 = 10
PROB_CLIP_LOW = 1.0
PROB_CLIP_HIGH = 99.0

# Load trained model
print("\nLoading model...")
try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(FEATURE_INFO_PATH, 'rb') as f:
        feature_info = pickle.load(f)
    feature_cols = feature_info['feature_cols']
    print(f"Model loaded (trained {feature_info['trained_date']})")
    print(f"  MAE: {feature_info['test_mae']:.3f} | RMSE: {feature_info['test_rmse']:.3f}")
except Exception as e:
    print(f"Error loading model: {e}")
    input("Press Enter to exit...")
    exit(1)

# Get recent data
lookback_days = 15

print(f"\nDownloading recent batter data (last {lookback_days} days)...")
print(f"  Statcast window ends on {RUN_DATE_STR}")
start_date = (today - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
end_date = today.strftime('%Y-%m-%d')

try:
    recent_data = statcast(start_dt=start_date, end_dt=end_date)
    print(f"Downloaded {len(recent_data)} pitches")
except Exception as e:
    print(f"Error: {e}")
    input("Press Enter to exit...")
    exit(1)

if len(recent_data) == 0:
    print("No data available")
    input("Press Enter to exit...")
    exit(1)

# Process batter-game stats
print("\nProcessing batter stats...")

batter_games = recent_data.groupby(['batter', 'game_pk', 'game_date']).agg({
    'estimated_ba_using_speedangle': 'mean',
    'estimated_woba_using_speedangle': 'mean',
    'estimated_slg_using_speedangle': 'mean',
    'launch_speed': 'mean',
    'launch_angle': 'mean'
}).reset_index()

batter_games.columns = ['player_id', 'game_pk', 'game_date', 'xba', 'xwoba', 'xslg', 
                        'launch_speed', 'launch_angle']

# Hard-hit rate
batted_balls = recent_data[recent_data['launch_speed'].notna()].copy()
batted_balls['hard_hit'] = (batted_balls['launch_speed'] >= 95).astype(int)
hardhit = batted_balls.groupby(['batter', 'game_pk']).agg({'hard_hit': 'mean'}).reset_index()
hardhit.columns = ['player_id', 'game_pk', 'hardhit_percent']
batter_games = batter_games.merge(hardhit, on=['player_id', 'game_pk'], how='left')

# Outcomes
outcomes = recent_data[recent_data['events'].notna()].copy()
outcomes['is_hit'] = outcomes['events'].isin(['single', 'double', 'triple', 'home_run']).astype(int)
outcomes['is_hr'] = (outcomes['events'] == 'home_run').astype(int)
outcomes['is_k'] = outcomes['events'].str.contains('strikeout', na=False).astype(int)
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

# K%
pa_counts = outcomes.groupby(['batter', 'game_pk']).size().reset_index(name='pa')
pa_counts = pa_counts.rename(columns={'batter': 'player_id'})
batter_games = batter_games.merge(pa_counts, on=['player_id', 'game_pk'], how='left')
batter_games['k_percent'] = (batter_games['so'] / batter_games['pa']).fillna(0)

batter_games = batter_games.fillna(0)
if 'pa' not in batter_games.columns:
    batter_games['pa'] = 0
batter_games['game_played'] = 1

# Rolling averages
print("Calculating rolling averages...")
batter_games['game_date'] = pd.to_datetime(batter_games['game_date'])
batter_games = batter_games.sort_values(['player_id', 'game_date'])

metrics = ['hits', 'target_tb', 'hrs', 'so', 'launch_speed', 'launch_angle', 
           'hardhit_percent', 'xwoba', 'xba', 'xslg', 'k_percent']

def calc_rolling(group, window=10):
    group['rolling_games_10'] = group['game_played'].rolling(window=window, min_periods=1).sum().shift(1)
    group['rolling_pa_10'] = group['pa'].rolling(window=window, min_periods=1).sum().shift(1)
    for metric in metrics:
        if metric in group.columns:
            group[f'rolling_{metric}_10'] = group[metric].rolling(window=window, min_periods=1).mean().shift(1)
    return group

batter_games = batter_games.groupby('player_id').apply(calc_rolling).reset_index(drop=True)

latest_batters = batter_games.groupby('player_id').last().reset_index()
latest_batters = latest_batters.dropna(subset=['rolling_hits_10'])

before_filter = len(latest_batters)
latest_batters = latest_batters[
    (latest_batters['rolling_games_10'].fillna(0) >= MIN_GAMES_LAST_10)
    & (latest_batters['rolling_pa_10'].fillna(0) >= MIN_PA_LAST_10)
].copy()
print(f"Sample-size filter: {before_filter} -> {len(latest_batters)} batters (min {MIN_GAMES_LAST_10} games, {MIN_PA_LAST_10} PA)")

print(f"Calculated rolling averages for {len(latest_batters)} batters")

# Add pitcher stats / matchup features
print("\nEnriching with pitcher matchup data (if available)...")
latest_batters['pitcher_xba'] = 0.245
latest_batters['pitcher_xwoba'] = 0.315
latest_batters['pitcher_xslg'] = 0.400
latest_batters['pitcher_k_rate'] = 0.22

# Optional: per-batter opposing pitcher for today
if os.path.exists(TODAY_MATCHUPS_FILE) and os.path.exists(PITCHER_STATS_FILE):
    try:
        matchups = pd.read_csv(TODAY_MATCHUPS_FILE)
        pitcher_stats = pd.read_csv(PITCHER_STATS_FILE)

        # Expect columns: player_id, opposing_pitcher_id, pitcher_throws (optional)
        if 'player_id' in matchups.columns and 'opposing_pitcher_id' in matchups.columns:
            latest_batters = latest_batters.merge(
                matchups[['player_id', 'opposing_pitcher_id']],
                on='player_id',
                how='left'
            )

            pitcher_stats = pitcher_stats.rename(columns={'pitcher_id': 'opposing_pitcher_id'})
            latest_batters = latest_batters.merge(
                pitcher_stats[['opposing_pitcher_id', 'pitcher_xba', 'pitcher_xwoba', 'pitcher_xslg', 'pitcher_k_rate']],
                on='opposing_pitcher_id',
                how='left',
                suffixes=('', '_from_stats')
            )

            # Prefer matchup-specific stats where available
            for col in ['pitcher_xba', 'pitcher_xwoba', 'pitcher_xslg', 'pitcher_k_rate']:
                from_stats = f"{col}_from_stats"
                if from_stats in latest_batters.columns:
                    latest_batters[col] = latest_batters[from_stats].combine_first(latest_batters[col])

            print("Applied matchup-specific pitcher stats from today_matchups.csv")
        else:
            print("WARNING: today_matchups.csv missing required columns (player_id, opposing_pitcher_id); using league-average pitcher profile.")
    except Exception as e:
        print(f"WARNING: Error applying matchup-specific pitcher data: {e}")
        print("  Falling back to league-average pitcher profile.")
else:
    print("INFO: No today_matchups.csv or pitcher_season_stats.csv found; using league-average pitcher profile.")

# Matchup differentials (vs either specific pitcher or generic league profile)
latest_batters['xba_diff'] = latest_batters['rolling_xba_10'] - latest_batters['pitcher_xba']
latest_batters['xwoba_diff'] = latest_batters['rolling_xwoba_10'] - latest_batters['pitcher_xwoba']
latest_batters['xslg_diff'] = latest_batters['rolling_xslg_10'] - latest_batters['pitcher_xslg']

# Get player names
print("\nLooking up player names...")
unique_ids = latest_batters['player_id'].unique()
try:
    player_names = playerid_reverse_lookup(unique_ids, key_type='mlbam')
    name_lookup = dict(zip(player_names['key_mlbam'], 
                          player_names['name_first'] + ' ' + player_names['name_last']))
    latest_batters['player_name'] = latest_batters['player_id'].map(name_lookup)
    latest_batters['player_name'] = latest_batters['player_name'].fillna('Unknown Player')
    print(f"Found names for {player_names.shape[0]} players")
except Exception as e:
    print(f"WARNING: Could not look up names: {e}")
    latest_batters['player_name'] = 'Player ' + latest_batters['player_id'].astype(str)

# Generate predictions
print("\nGenerating predictions...")
X_predict = latest_batters[feature_cols]

# Total Bases predictions
pred_tb = model.predict(X_predict)
latest_batters['pred_tb'] = pred_tb

# Hits predictions (simplified model - can be improved)
latest_batters['pred_hits'] = latest_batters['rolling_hits_10']

# Strikeouts predictions
latest_batters['pred_k'] = latest_batters['rolling_so_10']

# Calculate probabilities for all props
# Total Bases
latest_batters['tb_over_0.5'] = (1 - poisson.cdf(0, pred_tb)) * 100
latest_batters['tb_over_1.5'] = (1 - poisson.cdf(1, pred_tb)) * 100
latest_batters['tb_over_2.5'] = (1 - poisson.cdf(2, pred_tb)) * 100
latest_batters['tb_over_3.5'] = (1 - poisson.cdf(3, pred_tb)) * 100

# Hits
latest_batters['hits_over_0.5'] = (1 - poisson.cdf(0, latest_batters['pred_hits'])) * 100
latest_batters['hits_over_1.5'] = (1 - poisson.cdf(1, latest_batters['pred_hits'])) * 100
latest_batters['hits_over_2.5'] = (1 - poisson.cdf(2, latest_batters['pred_hits'])) * 100

# Strikeouts
latest_batters['k_over_0.5'] = (1 - poisson.cdf(0, latest_batters['pred_k'])) * 100
latest_batters['k_over_1.5'] = (1 - poisson.cdf(1, latest_batters['pred_k'])) * 100
latest_batters['k_over_2.5'] = (1 - poisson.cdf(2, latest_batters['pred_k'])) * 100

# Clip extreme probabilities for realism (display + decision safety)
prob_cols = [
    'tb_over_0.5', 'tb_over_1.5', 'tb_over_2.5', 'tb_over_3.5',
    'hits_over_0.5', 'hits_over_1.5', 'hits_over_2.5',
    'k_over_0.5', 'k_over_1.5', 'k_over_2.5'
]
for col in prob_cols:
    if col in latest_batters.columns:
        latest_batters[col] = latest_batters[col].clip(lower=PROB_CLIP_LOW, upper=PROB_CLIP_HIGH)

# Determine best bets
def get_best_bet(row):
    bets = []
    
    # Total Bases
    if row['tb_over_1.5'] > 67.5:
        bets.append({"prop": "Total Bases", "line": "OVER 1.5", "prob": row['tb_over_1.5'], "confidence": "HIGH"})
    elif row['tb_over_1.5'] < 37.5:
        bets.append({"prop": "Total Bases", "line": "UNDER 1.5", "prob": 100 - row['tb_over_1.5'], "confidence": "HIGH"})
    
    # Hits
    if row['hits_over_0.5'] > 70:
        bets.append({"prop": "Hits", "line": "OVER 0.5", "prob": row['hits_over_0.5'], "confidence": "HIGH"})
    elif row['hits_over_0.5'] < 35:
        bets.append({"prop": "Hits", "line": "UNDER 0.5", "prob": 100 - row['hits_over_0.5'], "confidence": "HIGH"})
    
    # Strikeouts
    if row['k_over_0.5'] > 65:
        bets.append({"prop": "Strikeouts", "line": "OVER 0.5", "prob": row['k_over_0.5'], "confidence": "MEDIUM"})
    
    return bets if bets else [{"prop": "None", "line": "No edge", "prob": 50, "confidence": "LOW"}]

latest_batters['recommended_bets'] = latest_batters.apply(get_best_bet, axis=1)

# Format for JSON
predictions_list = []
for _, row in latest_batters.iterrows():
    predictions_list.append({
        "player_id": int(row['player_id']),
        "player_name": row['player_name'],
        "sample": {
            "games_last_10": int(row.get('rolling_games_10', 0) or 0),
            "pa_last_10": int(row.get('rolling_pa_10', 0) or 0)
        },
        "total_bases": {
            "predicted": round(float(row['pred_tb']), 2),
            "probabilities": {
                "over_0.5": round(float(row['tb_over_0.5']), 1),
                "over_1.5": round(float(row['tb_over_1.5']), 1),
                "over_2.5": round(float(row['tb_over_2.5']), 1),
                "over_3.5": round(float(row['tb_over_3.5']), 1)
            }
        },
        "hits": {
            "predicted": round(float(row['pred_hits']), 2),
            "probabilities": {
                "over_0.5": round(float(row['hits_over_0.5']), 1),
                "over_1.5": round(float(row['hits_over_1.5']), 1),
                "over_2.5": round(float(row['hits_over_2.5']), 1)
            }
        },
        "strikeouts": {
            "predicted": round(float(row['pred_k']), 2),
            "probabilities": {
                "over_0.5": round(float(row['k_over_0.5']), 1),
                "over_1.5": round(float(row['k_over_1.5']), 1),
                "over_2.5": round(float(row['k_over_2.5']), 1)
            }
        },
        "recommended_bets": row['recommended_bets'],
        "recent_stats": {
            "avg_tb": round(float(row['rolling_target_tb_10']), 2),
            "avg_hits": round(float(row['rolling_hits_10']), 2),
            "avg_k": round(float(row['rolling_so_10']), 2),
            "exit_velo": round(float(row['rolling_launch_speed_10']), 1),
            "hard_hit_rate": round(float(row['rolling_hardhit_percent_10']) * 100, 1),
            "xwoba": round(float(row['rolling_xwoba_10']), 3)
        }
    })

# Sort by best probabilities
predictions_list = sorted(predictions_list, key=lambda x: max([b['prob'] for b in x['recommended_bets']]), reverse=True)

predictions_count_all = len(predictions_list)


def attach_matchups_from_starters(pred_list, sdf):
    """Add matchup dict from starters_df (StatsAPI boxscore)."""
    if sdf is None or len(sdf) == 0:
        for p in pred_list:
            p['matchup'] = None
        return
    one = sdf.sort_values(['player_id', 'game_id']).drop_duplicates('player_id', keep='first')
    by_pid = one.set_index('player_id')
    for p in pred_list:
        pid = int(p['player_id'])
        if pid not in by_pid.index:
            p['matchup'] = None
            continue
        r = by_pid.loc[pid]
        if isinstance(r, pd.DataFrame):
            r = r.iloc[0]
        p['matchup'] = {
            'game_id': int(r['game_id']),
            'team': str(r['team_name']),
            'opponent': str(r['opponent_team']),
            'label': str(r['matchup']),
            'away_team': str(r['away_team']),
            'home_team': str(r['home_team']),
            'side': str(r['side']),
            'batting_order': int(r['batting_order']),
        }


def build_games_list(sdf):
    if sdf is None or len(sdf) == 0:
        return []
    g = sdf.drop_duplicates(subset=['game_id'], keep='first')
    out = []
    for _, row in g.iterrows():
        out.append({
            'game_id': int(row['game_id']),
            'matchup': str(row['matchup']),
            'away_team': str(row['away_team']),
            'home_team': str(row['home_team']),
        })
    out.sort(key=lambda x: x['matchup'])
    return out


def _predictions_to_df(pred_list):
    rows = []
    for p in pred_list:
        m = p.get('matchup') or {}
        rows.append({
            'player_id': p['player_id'],
            'player_name': p.get('player_name', ''),
            'game_id': m.get('game_id'),
            'team': m.get('team'),
            'opponent': m.get('opponent'),
            'matchup': m.get('label'),
            'pred_tb': p.get('total_bases', {}).get('predicted'),
            'pred_hits': p.get('hits', {}).get('predicted'),
            'pred_k': p.get('strikeouts', {}).get('predicted'),
            'tb_o15': p.get('total_bases', {}).get('probabilities', {}).get('over_1.5'),
            'hits_o05': p.get('hits', {}).get('probabilities', {}).get('over_0.5'),
            'k_o05': p.get('strikeouts', {}).get('probabilities', {}).get('over_0.5'),
            'games_last_10': p.get('sample', {}).get('games_last_10'),
            'pa_last_10': p.get('sample', {}).get('pa_last_10'),
        })
    return pd.DataFrame(rows)


lineup_meta = {
    'schedule_date': RUN_DATE_STR,
    'filter_applied': False,
    'starters_rows': 0,
    'predictions_before': predictions_count_all,
    'predictions_after': predictions_count_all,
    'note': None,
}

starters_df = pd.DataFrame()
schedule_games_count = 0
if not SKIP_LINEUP_FILTER:
    try:
        print(f"\nFetching starting lineups via StatsAPI for {RUN_DATE_STR}...")
        starters_df, schedule_games_count = fetch_starters_for_date(RUN_DATE_STR, sport_id=1)
        os.makedirs(os.path.dirname(TODAY_STARTERS_CSV), exist_ok=True)
        starters_df.to_csv(TODAY_STARTERS_CSV, index=False)
        lineup_meta['starters_rows'] = int(len(starters_df))
        print(f"Saved {len(starters_df)} starter rows to data/today_starters.csv")
    except ImportError:
        lineup_meta['note'] = 'MLB-StatsAPI not installed; install with: pip install MLB-StatsAPI'
        print("WARNING: " + lineup_meta['note'])
    except Exception as e:
        lineup_meta['note'] = f'StatsAPI lineup fetch failed: {e}'
        print("WARNING: " + lineup_meta['note'])

attach_matchups_from_starters(predictions_list, starters_df)
_predictions_to_df(predictions_list).to_csv(PREDICTIONS_TODAY_CSV, index=False)

if not SKIP_LINEUP_FILTER and len(starters_df) > 0:
    starter_ids = set(int(x) for x in starters_df['player_id'].tolist())
    predictions_filtered = [p for p in predictions_list if int(p['player_id']) in starter_ids]
    lineup_meta['filter_applied'] = True
    lineup_meta['predictions_after'] = len(predictions_filtered)
    predictions_list = predictions_filtered
    print(f"Lineup filter: {lineup_meta['predictions_before']} -> {lineup_meta['predictions_after']} players (starters only)")
elif not SKIP_LINEUP_FILTER and len(starters_df) == 0:
    if schedule_games_count > 0:
        extra = 'scheduled games but no boxscore starters parsed — publishing unfiltered predictions'
        lineup_meta['note'] = extra if not lineup_meta.get('note') else f"{lineup_meta['note']}; {extra}"
        print("WARNING: Games on schedule but no starters in boxscore; publishing all predictions (no lineup filter).")
    else:
        extra = 'no games on slate — publishing unfiltered predictions'
        lineup_meta['note'] = extra if not lineup_meta.get('note') else f"{lineup_meta['note']}; {extra}"
        print("INFO: No games on slate or no starters file; publishing all predictions.")

_predictions_to_df(predictions_list).to_csv(PREDICTIONS_LIVE_LINEUPS_CSV, index=False)

games_on_slate = build_games_list(starters_df)

# Create output
output = {
    "generated_at": datetime.now().isoformat(),
    "model_version": feature_info['model_version'],
    "model_performance": {
        "mae": round(float(feature_info['test_mae']), 3),
        "rmse": round(float(feature_info['test_rmse']), 3)
    },
    "lineup_filter": lineup_meta,
    "games": games_on_slate,
    "total_players": len(predictions_list),
    "all_predictions": predictions_list
}

# Save to file
print(f"\nSaving predictions...")
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
with open(OUTPUT_FILE, 'w') as f:
    json.dump(output, f, indent=2)

print(f"Saved {len(predictions_list)} predictions")

# Display top picks
print("\n" + "="*60)
print("TOP 10 BEST BETTING OPPORTUNITIES")
print("="*60)
for i, pred in enumerate(predictions_list[:10], 1):
    print(f"\n{i}. {pred['player_name']} (ID: {pred['player_id']})")
    for bet in pred['recommended_bets']:
        if bet['confidence'] != 'LOW':
            print(f"   {bet['prop']}: {bet['line']} - {bet['prob']:.1f}% ({bet['confidence']})")

# Push to GitHub
print("\n" + "="*60)
print("PUSHING TO GITHUB")
print("="*60)

try:
    os.chdir(GITHUB_REPO_PATH)
    git_paths = ['data/predictions.json']
    for name in ('today_starters.csv', 'predictions_today.csv', 'predictions_live_lineups.csv'):
        rel = os.path.join('data', name)
        if os.path.isfile(os.path.join(GITHUB_REPO_PATH, rel.replace('/', os.sep))):
            git_paths.append(rel.replace('/', os.sep))
    subprocess.run(['git', 'add'] + git_paths, check=True)
    commit_msg = f"Update predictions - {datetime.now().strftime('%Y-%m-%d %I:%M %p')}"
    subprocess.run(['git', 'commit', '-m', commit_msg], check=True)
    subprocess.run(['git', 'push', 'origin', 'main'], check=True)
    print("Successfully pushed to GitHub")
except Exception as e:
    print(f"WARNING: Git push failed: {e}")
    print("Run manually: git add, git commit, git push")

print("\n" + "="*60)
print("COMPLETE")
print("="*60)

input("\nPress Enter to exit...")
