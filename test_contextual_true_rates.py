import os
import pandas as pd
import numpy as np

DATA_DIR = r"C:\Users\ASUS\.gemini\antigravity\scratch\RR"
bbb = pd.read_feather(os.path.join(DATA_DIR, "bbb_with_pressure.feather"))

print("Computing Contextual True Strike Rate & True Economy Rate...")

bbb['batter_ball_faced'] = bbb['extras_type'] != 'wide'
bbb['is_dot_batter'] = (bbb['runs'] == 0) & bbb['batter_ball_faced']
bbb['is_boundary_batter'] = bbb['is_four'] | bbb['is_six']
bbb['is_batter_out'] = bbb['is_wicket'] & (bbb['dismissed_player_id'] == bbb['striker_id'])

# 1. Compute expected phase/match metrics for high-pressure balls:
# For each match and innings, compute average SR and ER in high pressure
m_inn_p = bbb[bbb['is_high_pressure']].groupby(['match_id', 'innings']).agg(
    hp_balls_faced=('batter_ball_faced', 'sum'),
    hp_runs=('runs', 'sum'),
    hp_legal_bowled=('is_legal', 'sum'),
    hp_bowler_runs=('bowler_runs_conceded', 'sum')
).reset_index()

m_inn_p['match_hp_sr'] = (m_inn_p['hp_runs'] / np.maximum(1, m_inn_p['hp_balls_faced'])) * 100.0
m_inn_p['match_hp_er'] = (m_inn_p['hp_bowler_runs'] / np.maximum(1, m_inn_p['hp_legal_bowled'])) * 6.0

# Merge back
bbb = bbb.merge(m_inn_p[['match_id', 'innings', 'match_hp_sr', 'match_hp_er']], on=['match_id', 'innings'], how='left')

# Global high pressure benchmarks
global_hp_sr = (bbb[bbb['batter_ball_faced'] & bbb['is_high_pressure']]['runs'].sum() /
                bbb[bbb['batter_ball_faced'] & bbb['is_high_pressure']]['batter_ball_faced'].sum()) * 100.0
global_hp_er = (bbb[bbb['is_legal'] & bbb['is_high_pressure']]['bowler_runs_conceded'].sum() /
                bbb[bbb['is_legal'] & bbb['is_high_pressure']]['is_legal'].sum()) * 6.0

print(f"Global High Pressure SR: {global_hp_sr:.2f}, ER: {global_hp_er:.2f}")

# Fill na match benchmarks with global benchmarks
bbb['match_hp_sr'] = bbb['match_hp_sr'].fillna(global_hp_sr)
bbb['match_hp_er'] = bbb['match_hp_er'].fillna(global_hp_er)

# Competition quality tier
is_ipl = bbb['comp_name'].str.contains('IPL', case=False, na=False)
bbb['tier_weight'] = np.where(is_ipl, 1.25, np.where(bbb['match_format'] == 'T20I', 1.00, 0.85))

# ==========================================
# REFINED BATTING ANALYSIS WITH TRUE SR
# ==========================================
# For each batter, compute weighted True SR above match context
bat_hp = bbb[bbb['batter_ball_faced'] & bbb['is_high_pressure']].copy()
bat_hp['runs_above_match_exp'] = bat_hp['runs'] - (bat_hp['match_hp_sr'] / 100.0)

bat_stats = bat_hp.groupby(['striker_id', 'striker_name']).agg(
    press_balls=('batter_ball_faced', 'sum'),
    press_runs=('runs', 'sum'),
    press_outs=('is_batter_out', 'sum'),
    press_dots=('is_dot_batter', 'sum'),
    press_bndys=('is_boundary_batter', 'sum'),
    press_matches=('match_id', 'nunique'),
    ipl_balls=('batter_ball_faced', lambda x: x[is_ipl.loc[x.index]].sum()),
    total_runs_above_match=('runs_above_match_exp', 'sum'),
    avg_tier=('tier_weight', 'mean')
).reset_index()

bat_stats['sr_press'] = (bat_stats['press_runs'] / bat_stats['press_balls']) * 100.0
bat_stats['true_sr_diff'] = (bat_stats['total_runs_above_match'] / bat_stats['press_balls']) * 100.0
bat_stats['avg_press'] = bat_stats['press_runs'] / np.maximum(1, bat_stats['press_outs'])

# Merge baseline overall
bat_all = bbb[bbb['batter_ball_faced']].groupby('striker_id').agg(
    total_balls=('batter_ball_faced', 'sum'),
    total_runs=('runs', 'sum')
).reset_index()
bat_all['sr_base'] = (bat_all['total_runs'] / bat_all['total_balls']) * 100.0

bat_merged = bat_stats.merge(bat_all[['striker_id', 'sr_base']], on='striker_id', how='left')
bat_merged['delta_sr'] = bat_merged['sr_press'] - bat_merged['sr_base']

# Consistency
match_bat = bat_hp.groupby(['match_id', 'striker_id']).agg(
    m_balls=('batter_ball_faced', 'sum'),
    m_runs=('runs', 'sum'),
    m_exp_runs=('match_hp_sr', lambda x: (x / 100.0).sum())
).reset_index()
match_bat['is_success'] = (match_bat['m_runs'] > match_bat['m_exp_runs']) | ((match_bat['m_runs'] >= 20) & (match_bat['m_balls'] <= 14))

bat_const = match_bat.groupby('striker_id').agg(
    qual_matches=('match_id', 'count'),
    succ_matches=('is_success', 'sum')
).reset_index()
bat_const['bat_consistency'] = bat_const['succ_matches'] / bat_const['qual_matches']

bat_merged = bat_merged.merge(bat_const[['striker_id', 'bat_consistency']], on='striker_id', how='left')

# Minimum 100 balls
bat_cand = bat_merged[bat_merged['press_balls'] >= 100].copy()

# Bayesian Shrinkage on True SR diff (K=30 balls)
K_bat = 30
bat_cand['true_sr_shrunk'] = (bat_cand['press_balls'] / (bat_cand['press_balls'] + K_bat)) * bat_cand['true_sr_diff']

def zscore(s):
    return (s - s.mean()) / s.std()

bat_cand['z_true_sr'] = zscore(bat_cand['true_sr_shrunk'])
bat_cand['z_delta_sr'] = zscore(bat_cand['delta_sr'])
bat_cand['z_avg_press'] = zscore(np.clip(bat_cand['avg_press'], 10, 55))
bat_cand['z_consistency'] = zscore(bat_cand['bat_consistency'])

# Final Batting Composite Score:
# 35% True SR above match context, 25% Delta vs player baseline, 20% Dismissal resilience, 20% Consistency
# Multiplied by average tier weight
bat_cand['composite_score'] = (
    0.35 * bat_cand['z_true_sr'] +
    0.25 * bat_cand['z_delta_sr'] +
    0.20 * bat_cand['z_avg_press'] +
    0.20 * bat_cand['z_consistency']
) * bat_cand['avg_tier']

top15_batters = bat_cand.sort_values(by='composite_score', ascending=False).head(15).reset_index(drop=True)
top15_batters['Rank'] = range(1, 16)
print("\n=== TOP 15 BATTERS (CONTEXTUAL TRUE-SR MODEL) ===")
cols_b = ['Rank', 'striker_name', 'press_balls', 'sr_press', 'sr_base', 'delta_sr', 'true_sr_diff', 'avg_press', 'bat_consistency', 'composite_score']
print(top15_batters[cols_b].to_string(index=False))


# ==========================================
# REFINED BOWLING ANALYSIS WITH TRUE ER
# ==========================================
bowl_hp = bbb[bbb['is_legal'] & bbb['is_high_pressure']].copy()
# Expected runs conceded based on match context: (match_hp_er / 6.0)
bowl_hp['runs_saved_vs_match'] = (bowl_hp['match_hp_er'] / 6.0) - bowl_hp['bowler_runs_conceded']

bowl_stats = bowl_hp.groupby(['bowler_id', 'bowler_name']).agg(
    press_balls=('is_legal', 'sum'),
    press_runs=('bowler_runs_conceded', 'sum'),
    press_wkts=('is_bowler_wicket', 'sum'),
    press_dots=('runs', lambda x: ((x == 0) & (bbb.loc[x.index, 'is_legal'])).sum()),
    press_bndys=('runs', lambda x: ((bbb.loc[x.index, 'is_four'] | bbb.loc[x.index, 'is_six']) & (bbb.loc[x.index, 'is_legal'])).sum()),
    press_matches=('match_id', 'nunique'),
    ipl_balls=('is_legal', lambda x: x[is_ipl.loc[x.index]].sum()),
    total_runs_saved=('runs_saved_vs_match', 'sum'),
    avg_tier=('tier_weight', 'mean')
).reset_index()

bowl_stats['er_press'] = (bowl_stats['press_runs'] / bowl_stats['press_balls']) * 6.0
bowl_stats['true_er_saved'] = (bowl_stats['total_runs_saved'] / bowl_stats['press_balls']) * 6.0
bowl_stats['wkt_pct_press'] = (bowl_stats['press_wkts'] / bowl_stats['press_balls']) * 100.0
bowl_stats['dot_pct_press'] = (bowl_stats['press_dots'] / bowl_stats['press_balls']) * 100.0

# Overall baseline
bowl_all = bbb[bbb['is_legal']].groupby('bowler_id').agg(
    total_balls=('is_legal', 'sum'),
    total_runs=('bowler_runs_conceded', 'sum')
).reset_index()
bowl_all['er_base'] = (bowl_all['total_runs'] / bowl_all['total_balls']) * 6.0

bowl_merged = bowl_stats.merge(bowl_all[['bowler_id', 'er_base']], on='bowler_id', how='left')
bowl_merged['delta_er'] = bowl_merged['er_base'] - bowl_merged['er_press']

# Consistency
match_bowl = bowl_hp.groupby(['match_id', 'bowler_id']).agg(
    m_balls=('is_legal', 'sum'),
    m_runs=('bowler_runs_conceded', 'sum'),
    m_wkts=('is_bowler_wicket', 'sum'),
    m_exp_runs=('match_hp_er', lambda x: (x / 6.0).sum())
).reset_index()
match_bowl['is_success'] = (match_bowl['m_runs'] < match_bowl['m_exp_runs']) | (match_bowl['m_wkts'] >= 1)

bowl_const = match_bowl.groupby('bowler_id').agg(
    qual_matches=('match_id', 'count'),
    succ_matches=('is_success', 'sum')
).reset_index()
bowl_const['bowl_consistency'] = bowl_const['succ_matches'] / bowl_const['qual_matches']

bowl_merged = bowl_merged.merge(bowl_const[['bowler_id', 'bowl_consistency']], on='bowler_id', how='left')

# Minimum 100 balls
bowl_cand = bowl_merged[bowl_merged['press_balls'] >= 100].copy()

# Bayesian Shrinkage on True ER Saved (K=30 balls)
K_bowl = 30
bowl_cand['true_er_shrunk'] = (bowl_cand['press_balls'] / (bowl_cand['press_balls'] + K_bowl)) * bowl_cand['true_er_saved']

bowl_cand['z_true_er'] = zscore(bowl_cand['true_er_shrunk'])
bowl_cand['z_wkt'] = zscore(bowl_cand['wkt_pct_press'])
bowl_cand['z_dot'] = zscore(bowl_cand['dot_pct_press'])
bowl_cand['z_consistency'] = zscore(bowl_cand['bowl_consistency'])

# Final Bowling Composite Score:
# 35% True ER saved vs match context, 25% Wicket taking rate, 20% Dot ball %, 20% Match consistency
# Multiplied by average tier weight
bowl_cand['composite_score'] = (
    0.35 * bowl_cand['z_true_er'] +
    0.25 * bowl_cand['z_wkt'] +
    0.20 * bowl_cand['z_dot'] +
    0.20 * bowl_cand['z_consistency']
) * bowl_cand['avg_tier']

top15_bowlers = bowl_cand.sort_values(by='composite_score', ascending=False).head(15).reset_index(drop=True)
top15_bowlers['Rank'] = range(1, 16)
print("\n=== TOP 15 BOWLERS (CONTEXTUAL TRUE-ER MODEL) ===")
cols_bo = ['Rank', 'bowler_name', 'press_balls', 'er_press', 'er_base', 'true_er_saved', 'wkt_pct_press', 'dot_pct_press', 'bowl_consistency', 'composite_score']
print(top15_bowlers[cols_bo].to_string(index=False))
