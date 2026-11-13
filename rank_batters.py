import os
import pandas as pd
import numpy as np

DATA_DIR = r"C:\Users\ASUS\.gemini\antigravity\scratch\RR"
bbb = pd.read_feather(os.path.join(DATA_DIR, "bbb_with_pressure.feather"))

print("Loaded bbb with pressure.")

# --- BATTING ANALYSIS ---
# Batter metrics by delivery
# Striker ID and Striker Name
# A ball counts towards balls faced if not wide
bbb['batter_ball_faced'] = bbb['extras_type'] != 'wide'
bbb['is_dot_batter'] = (bbb['runs'] == 0) & bbb['batter_ball_faced']
bbb['is_boundary_batter'] = bbb['is_four'] | bbb['is_six']
bbb['is_batter_out'] = bbb['is_wicket'] & (bbb['dismissed_player_id'] == bbb['striker_id'])

# Aggregate overall baseline per batter
bat_overall = bbb[bbb['batter_ball_faced']].groupby(['striker_id', 'striker_name']).agg(
    total_balls=('batter_ball_faced', 'sum'),
    total_runs=('runs', 'sum'),
    total_dots=('is_dot_batter', 'sum'),
    total_bndys=('is_boundary_batter', 'sum'),
    total_outs=('is_batter_out', 'sum')
).reset_index()

bat_overall['sr_base'] = (bat_overall['total_runs'] / bat_overall['total_balls']) * 100.0
bat_overall['dot_pct_base'] = (bat_overall['total_dots'] / bat_overall['total_balls']) * 100.0
bat_overall['bndy_pct_base'] = (bat_overall['total_bndys'] / bat_overall['total_balls']) * 100.0

# Aggregate under High Pressure
bat_press = bbb[bbb['batter_ball_faced'] & bbb['is_high_pressure']].groupby(['striker_id', 'striker_name']).agg(
    press_balls=('batter_ball_faced', 'sum'),
    press_runs=('runs', 'sum'),
    press_dots=('is_dot_batter', 'sum'),
    press_bndys=('is_boundary_batter', 'sum'),
    press_outs=('is_batter_out', 'sum'),
    press_matches=('match_id', 'nunique')
).reset_index()

bat_press['sr_press'] = (bat_press['press_runs'] / bat_press['press_balls']) * 100.0
bat_press['dot_pct_press'] = (bat_press['press_dots'] / bat_press['press_balls']) * 100.0
bat_press['bndy_pct_press'] = (bat_press['press_bndys'] / bat_press['press_balls']) * 100.0
bat_press['avg_press'] = bat_press['press_runs'] / np.maximum(1, bat_press['press_outs'])

# Merge overall and pressure
bat_df = bat_overall.merge(bat_press, on=['striker_id', 'striker_name'], how='inner')

# Filter minimum sample size: at least 75 balls in high pressure
bat_filtered = bat_df[bat_df['press_balls'] >= 75].copy()

# Clutch Deltas
bat_filtered['delta_sr'] = bat_filtered['sr_press'] - bat_filtered['sr_base']
bat_filtered['delta_bndy'] = bat_filtered['bndy_pct_press'] - bat_filtered['bndy_pct_base']
bat_filtered['delta_dot'] = bat_filtered['dot_pct_base'] - bat_filtered['dot_pct_press']

# Bayesian Shrunk Strike Rate under pressure (K=25 balls towards population mean SR under pressure)
pop_sr_press = (bbb[bbb['batter_ball_faced'] & bbb['is_high_pressure']]['runs'].sum() /
                bbb[bbb['batter_ball_faced'] & bbb['is_high_pressure']]['batter_ball_faced'].sum()) * 100.0

K_bat = 30
bat_filtered['sr_shrunk'] = ((bat_filtered['press_balls'] / (bat_filtered['press_balls'] + K_bat)) * bat_filtered['sr_press'] +
                             (K_bat / (bat_filtered['press_balls'] + K_bat)) * pop_sr_press)

# Match-level consistency for batters:
# For each match, compute batter's impact in high pressure
match_bat = bbb[bbb['batter_ball_faced'] & bbb['is_high_pressure']].groupby(['match_id', 'striker_id']).agg(
    m_balls=('batter_ball_faced', 'sum'),
    m_runs=('runs', 'sum'),
    m_outs=('is_batter_out', 'sum')
).reset_index()

match_bat['m_sr'] = (match_bat['m_runs'] / match_bat['m_balls']) * 100.0
# A match is successful if: (m_balls >= 6 and m_sr >= 140) or (m_runs >= 20 and m_sr >= 125)
match_bat['is_success'] = ((match_bat['m_balls'] >= 6) & (match_bat['m_sr'] >= 140.0)) | ((match_bat['m_runs'] >= 20) & (match_bat['m_sr'] >= 125.0))

bat_consistency = match_bat.groupby('striker_id').agg(
    qualifying_matches=('match_id', 'count'),
    successful_matches=('is_success', 'sum')
).reset_index()
bat_consistency['bat_consistency_rate'] = bat_consistency['successful_matches'] / bat_consistency['qualifying_matches']

bat_filtered = bat_filtered.merge(bat_consistency[['striker_id', 'bat_consistency_rate']], on='striker_id', how='left')

# Composite Batting Pressure Score (BPS):
# 1. Shrunk SR (40%)
# 2. Delta SR (25%)
# 3. Average per dismissal in pressure (15%)
# 4. Consistency Rate (20%)
# Standardize each feature using z-scores or min-max
def zscore(s):
    return (s - s.mean()) / s.std()

bat_filtered['z_sr_shrunk'] = zscore(bat_filtered['sr_shrunk'])
bat_filtered['z_delta_sr'] = zscore(bat_filtered['delta_sr'])
bat_filtered['z_avg_press'] = zscore(np.clip(bat_filtered['avg_press'], 10, 60))
bat_filtered['z_consistency'] = zscore(bat_filtered['bat_consistency_rate'])

bat_filtered['batting_pressure_index'] = (
    0.35 * bat_filtered['z_sr_shrunk'] +
    0.25 * bat_filtered['z_delta_sr'] +
    0.20 * bat_filtered['z_avg_press'] +
    0.20 * bat_filtered['z_consistency']
)

top15_batters = bat_filtered.sort_values(by='batting_pressure_index', ascending=False).head(15)
print("\n=== TOP 15 BATTERS UNDER PRESSURE ===")
cols_bat = ['striker_name', 'press_balls', 'sr_press', 'sr_base', 'delta_sr', 'avg_press', 'bat_consistency_rate', 'batting_pressure_index']
print(top15_batters[cols_bat].to_string(index=False))
