import os
import pandas as pd
import numpy as np

DATA_DIR = r"C:\Users\ASUS\.gemini\antigravity\scratch\RR"
bbb = pd.read_feather(os.path.join(DATA_DIR, "bbb_with_pressure.feather"))

print("Running production ranking pipeline...")

# Filter legal deliveries for bowling, batter balls for batting
bbb['batter_ball_faced'] = bbb['extras_type'] != 'wide'
bbb['is_dot_batter'] = (bbb['runs'] == 0) & bbb['batter_ball_faced']
bbb['is_boundary_batter'] = bbb['is_four'] | bbb['is_six']
bbb['is_batter_out'] = bbb['is_wicket'] & (bbb['dismissed_player_id'] == bbb['striker_id'])

# Competition Quality Weighting:
# IPL = 1.20, T20I = 1.00, other = 0.85
is_ipl = bbb['comp_name'].str.contains('IPL', case=False, na=False)
bbb['weight'] = np.where(is_ipl, 1.20, np.where(bbb['match_format'] == 'T20I', 1.00, 0.85))

# ==========================================
# 1. BATTING PIPELINE
# ==========================================
# Baseline Overall
bat_all = bbb[bbb['batter_ball_faced']].groupby(['striker_id', 'striker_name']).agg(
    total_balls=('batter_ball_faced', 'sum'),
    total_runs=('runs', 'sum'),
    total_outs=('is_batter_out', 'sum'),
    total_dots=('is_dot_batter', 'sum'),
    total_bndys=('is_boundary_batter', 'sum')
).reset_index()
bat_all['sr_base'] = (bat_all['total_runs'] / bat_all['total_balls']) * 100.0

# Pressure Subset (PI >= 0.70)
bat_p = bbb[bbb['batter_ball_faced'] & bbb['is_high_pressure']].groupby(['striker_id', 'striker_name']).agg(
    press_balls=('batter_ball_faced', 'sum'),
    press_runs=('runs', 'sum'),
    press_outs=('is_batter_out', 'sum'),
    press_dots=('is_dot_batter', 'sum'),
    press_bndys=('is_boundary_batter', 'sum'),
    press_matches=('match_id', 'nunique'),
    ipl_balls=('batter_ball_faced', lambda x: x[is_ipl.loc[x.index]].sum())
).reset_index()

bat_p['sr_press'] = (bat_p['press_runs'] / bat_p['press_balls']) * 100.0
bat_p['dot_pct_press'] = (bat_p['press_dots'] / bat_p['press_balls']) * 100.0
bat_p['bndy_pct_press'] = (bat_p['press_bndys'] / bat_p['press_balls']) * 100.0
bat_p['avg_press'] = bat_p['press_runs'] / np.maximum(1, bat_p['press_outs'])

bat_df = bat_all.merge(bat_p, on=['striker_id', 'striker_name'], how='inner')

# Minimum Sample Threshold: at least 100 balls in high pressure
bat_cand = bat_df[bat_df['press_balls'] >= 100].copy()

# Clutch delta
bat_cand['delta_sr'] = bat_cand['sr_press'] - bat_cand['sr_base']

# Bayesian Shrunk SR
pop_sr_press = (bbb[bbb['batter_ball_faced'] & bbb['is_high_pressure']]['runs'].sum() /
                bbb[bbb['batter_ball_faced'] & bbb['is_high_pressure']]['batter_ball_faced'].sum()) * 100.0
K_bat = 30
bat_cand['sr_shrunk'] = (bat_cand['press_balls'] / (bat_cand['press_balls'] + K_bat)) * bat_cand['sr_press'] + \
                        (K_bat / (bat_cand['press_balls'] + K_bat)) * pop_sr_press

# Match-level consistency
match_bat = bbb[bbb['batter_ball_faced'] & bbb['is_high_pressure']].groupby(['match_id', 'striker_id']).agg(
    m_balls=('batter_ball_faced', 'sum'),
    m_runs=('runs', 'sum'),
    m_outs=('is_batter_out', 'sum')
).reset_index()
match_bat['m_sr'] = (match_bat['m_runs'] / match_bat['m_balls']) * 100.0
match_bat['is_success'] = ((match_bat['m_balls'] >= 6) & (match_bat['m_sr'] >= 140.0)) | \
                          ((match_bat['m_runs'] >= 20) & (match_bat['m_sr'] >= 125.0))

bat_const = match_bat.groupby('striker_id').agg(
    qual_matches=('match_id', 'count'),
    succ_matches=('is_success', 'sum')
).reset_index()
bat_const['bat_consistency'] = bat_const['succ_matches'] / bat_const['qual_matches']

bat_cand = bat_cand.merge(bat_const[['striker_id', 'bat_consistency']], on='striker_id', how='left')

# Standardized Z-scores
def zscore(s):
    return (s - s.mean()) / s.std()

bat_cand['z_sr_shrunk'] = zscore(bat_cand['sr_shrunk'])
bat_cand['z_delta_sr'] = zscore(bat_cand['delta_sr'])
bat_cand['z_avg_press'] = zscore(np.clip(bat_cand['avg_press'], 10, 55))
bat_cand['z_consistency'] = zscore(bat_cand['bat_consistency'])

# Competition Quality Adjustment multiplier (players with significant elite IPL experience get appropriate recognition)
bat_cand['ipl_ratio'] = bat_cand['ipl_balls'] / bat_cand['press_balls']
quality_boost = 1.0 + 0.10 * bat_cand['ipl_ratio']

bat_cand['composite_score'] = (
    0.35 * bat_cand['z_sr_shrunk'] +
    0.25 * bat_cand['z_delta_sr'] +
    0.20 * bat_cand['z_avg_press'] +
    0.20 * bat_cand['z_consistency']
) * quality_boost

# Scale composite score to 0 - 100 for presentation
min_s, max_s = bat_cand['composite_score'].min(), bat_cand['composite_score'].max()
bat_cand['pressure_rating_100'] = ((bat_cand['composite_score'] - min_s) / (max_s - min_s)) * 100.0

top10_batters = bat_cand.sort_values(by='composite_score', ascending=False).head(10).reset_index(drop=True)
top10_batters['Rank'] = range(1, 11)

print("\n================ TOP 10 BATTERS UNDER PRESSURE ================")
cols_bat_display = ['Rank', 'striker_name', 'press_balls', 'sr_press', 'sr_base', 'delta_sr', 'avg_press', 'bat_consistency', 'pressure_rating_100']
print(top10_batters[cols_bat_display].to_string(index=False))


# ==========================================
# 2. BOWLING PIPELINE
# ==========================================
# Baseline Overall
bowl_all = bbb[bbb['is_legal']].groupby(['bowler_id', 'bowler_name']).agg(
    total_balls=('is_legal', 'sum'),
    total_runs=('bowler_runs_conceded', 'sum'),
    total_wkts=('is_bowler_wicket', 'sum')
).reset_index()
bowl_all['er_base'] = (bowl_all['total_runs'] / bowl_all['total_balls']) * 6.0

# Pressure Subset (PI >= 0.70)
bowl_p = bbb[bbb['is_legal'] & bbb['is_high_pressure']].groupby(['bowler_id', 'bowler_name']).agg(
    press_balls=('is_legal', 'sum'),
    press_runs=('bowler_runs_conceded', 'sum'),
    press_wkts=('is_bowler_wicket', 'sum'),
    press_dots=('runs', lambda x: ((x == 0) & (bbb.loc[x.index, 'is_legal'])).sum()),
    press_bndys=('runs', lambda x: ((bbb.loc[x.index, 'is_four'] | bbb.loc[x.index, 'is_six']) & (bbb.loc[x.index, 'is_legal'])).sum()),
    press_matches=('match_id', 'nunique'),
    ipl_balls=('is_legal', lambda x: x[is_ipl.loc[x.index]].sum())
).reset_index()

bowl_p['er_press'] = (bowl_p['press_runs'] / bowl_p['press_balls']) * 6.0
bowl_p['dot_pct_press'] = (bowl_p['press_dots'] / bowl_p['press_balls']) * 100.0
bowl_p['bndy_pct_press'] = (bowl_p['press_bndys'] / bowl_p['press_balls']) * 100.0
bowl_p['wkt_pct_press'] = (bowl_p['press_wkts'] / bowl_p['press_balls']) * 100.0

bowl_df = bowl_all.merge(bowl_p, on=['bowler_id', 'bowler_name'], how='inner')

# Minimum Sample Threshold: at least 100 balls in high pressure
bowl_cand = bowl_df[bowl_df['press_balls'] >= 100].copy()

# Clutch delta & True ER saved
pop_er_press = (bbb[bbb['is_legal'] & bbb['is_high_pressure']]['bowler_runs_conceded'].sum() /
                bbb[bbb['is_legal'] & bbb['is_high_pressure']]['is_legal'].sum()) * 6.0

bowl_cand['true_er_saved'] = pop_er_press - bowl_cand['er_press']

# Bayesian Shrunk ER
K_bowl = 30
bowl_cand['er_shrunk'] = (bowl_cand['press_balls'] / (bowl_cand['press_balls'] + K_bowl)) * bowl_cand['er_press'] + \
                         (K_bowl / (bowl_cand['press_balls'] + K_bowl)) * pop_er_press

# Match-level consistency
match_bowl = bbb[bbb['is_legal'] & bbb['is_high_pressure']].groupby(['match_id', 'bowler_id']).agg(
    m_balls=('is_legal', 'sum'),
    m_runs=('bowler_runs_conceded', 'sum'),
    m_wkts=('is_bowler_wicket', 'sum')
).reset_index()
match_bowl['m_er'] = (match_bowl['m_runs'] / match_bowl['m_balls']) * 6.0
match_bowl['is_success'] = (match_bowl['m_er'] <= 8.5) | ((match_bowl['m_er'] <= 10.0) & (match_bowl['m_wkts'] >= 1))

bowl_const = match_bowl.groupby('bowler_id').agg(
    qual_matches=('match_id', 'count'),
    succ_matches=('is_success', 'sum')
).reset_index()
bowl_const['bowl_consistency'] = bowl_const['succ_matches'] / bowl_const['qual_matches']

bowl_cand = bowl_cand.merge(bowl_const[['bowler_id', 'bowl_consistency']], on='bowler_id', how='left')

# Standardized Z-scores
bowl_cand['z_er'] = -zscore(bowl_cand['er_shrunk']) # lower ER is better
bowl_cand['z_wkt'] = zscore(bowl_cand['wkt_pct_press'])
bowl_cand['z_dot'] = zscore(bowl_cand['dot_pct_press'])
bowl_cand['z_consistency'] = zscore(bowl_cand['bowl_consistency'])

# Competition Quality Adjustment multiplier
bowl_cand['ipl_ratio'] = bowl_cand['ipl_balls'] / bowl_cand['press_balls']
quality_boost_bowl = 1.0 + 0.10 * bowl_cand['ipl_ratio']

bowl_cand['composite_score'] = (
    0.35 * bowl_cand['z_er'] +
    0.25 * bowl_cand['z_wkt'] +
    0.20 * bowl_cand['z_dot'] +
    0.20 * bowl_cand['z_consistency']
) * quality_boost_bowl

min_sb, max_sb = bowl_cand['composite_score'].min(), bowl_cand['composite_score'].max()
bowl_cand['pressure_rating_100'] = ((bowl_cand['composite_score'] - min_sb) / (max_sb - min_sb)) * 100.0

top10_bowlers = bowl_cand.sort_values(by='composite_score', ascending=False).head(10).reset_index(drop=True)
top10_bowlers['Rank'] = range(1, 11)

print("\n================ TOP 10 BOWLERS UNDER PRESSURE ================")
cols_bowl_display = ['Rank', 'bowler_name', 'press_balls', 'er_press', 'er_base', 'true_er_saved', 'wkt_pct_press', 'dot_pct_press', 'bowl_consistency', 'pressure_rating_100']
print(top10_bowlers[cols_bowl_display].to_string(index=False))

# Export top 10 to CSV for reference
top10_batters.to_csv(os.path.join(DATA_DIR, "top10_batters.csv"), index=False)
top10_bowlers.to_csv(os.path.join(DATA_DIR, "top10_bowlers.csv"), index=False)
print("\nExported top10_batters.csv and top10_bowlers.csv successfully.")
