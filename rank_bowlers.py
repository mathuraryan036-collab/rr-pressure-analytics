import os
import pandas as pd
import numpy as np

DATA_DIR = r"C:\Users\ASUS\.gemini\antigravity\scratch\RR"
bbb = pd.read_feather(os.path.join(DATA_DIR, "bbb_with_pressure.feather"))

print("Loaded bbb for bowling analysis.")

# Bowler overall baseline
bowler_overall = bbb[bbb['is_legal']].groupby(['bowler_id', 'bowler_name']).agg(
    total_balls=('is_legal', 'sum'),
    total_runs=('bowler_runs_conceded', 'sum'),
    total_wkts=('is_bowler_wicket', 'sum'),
    total_dots=('runs', lambda x: ((x == 0) & (bbb.loc[x.index, 'is_legal'])).sum()),
    total_bndys=('runs', lambda x: ((bbb.loc[x.index, 'is_four'] | bbb.loc[x.index, 'is_six']) & (bbb.loc[x.index, 'is_legal'])).sum())
).reset_index()

bowler_overall['er_base'] = (bowler_overall['total_runs'] / bowler_overall['total_balls']) * 6.0
bowler_overall['dot_pct_base'] = (bowler_overall['total_dots'] / bowler_overall['total_balls']) * 100.0
bowler_overall['bndy_pct_base'] = (bowler_overall['total_bndys'] / bowler_overall['total_balls']) * 100.0

# Bowler under High Pressure
bowler_press = bbb[bbb['is_legal'] & bbb['is_high_pressure']].groupby(['bowler_id', 'bowler_name']).agg(
    press_balls=('is_legal', 'sum'),
    press_runs=('bowler_runs_conceded', 'sum'),
    press_wkts=('is_bowler_wicket', 'sum'),
    press_dots=('runs', lambda x: ((x == 0) & (bbb.loc[x.index, 'is_legal'])).sum()),
    press_bndys=('runs', lambda x: ((bbb.loc[x.index, 'is_four'] | bbb.loc[x.index, 'is_six']) & (bbb.loc[x.index, 'is_legal'])).sum()),
    press_matches=('match_id', 'nunique')
).reset_index()

bowler_press['er_press'] = (bowler_press['press_runs'] / bowler_press['press_balls']) * 6.0
bowler_press['dot_pct_press'] = (bowler_press['press_dots'] / bowler_press['press_balls']) * 100.0
bowler_press['bndy_pct_press'] = (bowler_press['press_bndys'] / bowler_press['press_balls']) * 100.0
bowler_press['wkt_pct_press'] = (bowler_press['press_wkts'] / bowler_press['press_balls']) * 100.0

# Merge
bowl_df = bowler_overall.merge(bowler_press, on=['bowler_id', 'bowler_name'], how='inner')

# Filter minimum sample size: at least 75 balls in high pressure
bowl_filtered = bowl_df[bowl_df['press_balls'] >= 75].copy()

# In high pressure (overs 16-20 / crunch), average population economy rate is much higher (~9.8 rpo):
pop_er_press = (bbb[bbb['is_legal'] & bbb['is_high_pressure']]['bowler_runs_conceded'].sum() /
                bbb[bbb['is_legal'] & bbb['is_high_pressure']]['is_legal'].sum()) * 6.0
print(f"Population High Pressure Economy Rate: {pop_er_press:.2f}")

# Expected vs Actual Economy:
# How much better is this bowler under pressure compared to expected high pressure ER?
bowl_filtered['true_er_saved'] = pop_er_press - bowl_filtered['er_press']

# Bayesian Shrunk ER (K=30 balls)
K_bowl = 30
bowl_filtered['er_shrunk'] = ((bowl_filtered['press_balls'] / (bowl_filtered['press_balls'] + K_bowl)) * bowl_filtered['er_press'] +
                             (K_bowl / (bowl_filtered['press_balls'] + K_bowl)) * pop_er_press)

# Match-level consistency for bowlers:
# For each match where bowler bowled >= 6 balls in high pressure:
match_bowl = bbb[bbb['is_legal'] & bbb['is_high_pressure']].groupby(['match_id', 'bowler_id']).agg(
    m_balls=('is_legal', 'sum'),
    m_runs=('bowler_runs_conceded', 'sum'),
    m_wkts=('is_bowler_wicket', 'sum')
).reset_index()
match_bowl['m_er'] = (match_bowl['m_runs'] / match_bowl['m_balls']) * 6.0
# Success if ER <= 8.5 OR (ER <= 10.0 and wkts >= 1)
match_bowl['is_success'] = (match_bowl['m_er'] <= 8.5) | ((match_bowl['m_er'] <= 10.0) & (match_bowl['m_wkts'] >= 1))

bowl_consistency = match_bowl.groupby('bowler_id').agg(
    qualifying_matches=('match_id', 'count'),
    successful_matches=('is_success', 'sum')
).reset_index()
bowl_consistency['bowl_consistency_rate'] = bowl_consistency['successful_matches'] / bowl_consistency['qualifying_matches']

bowl_filtered = bowl_filtered.merge(bowl_consistency[['bowler_id', 'bowl_consistency_rate']], on='bowler_id', how='left')

# Composite Bowling Pressure Score (BPS):
# 1. Shrunk ER (lower is better -> inverted z-score) (35%)
# 2. Wicket Taking Rate under pressure (25%)
# 3. Dot Ball % under pressure (20%)
# 4. Match Consistency Rate (20%)
def zscore(s):
    return (s - s.mean()) / s.std()

bowl_filtered['z_er'] = -zscore(bowl_filtered['er_shrunk']) # lower ER is better
bowl_filtered['z_wkt'] = zscore(bowl_filtered['wkt_pct_press'])
bowl_filtered['z_dot'] = zscore(bowl_filtered['dot_pct_press'])
bowl_filtered['z_bndy'] = -zscore(bowl_filtered['bndy_pct_press']) # lower boundary % is better
bowl_filtered['z_consistency'] = zscore(bowl_filtered['bowl_consistency_rate'])

bowl_filtered['bowling_pressure_index'] = (
    0.35 * bowl_filtered['z_er'] +
    0.25 * bowl_filtered['z_wkt'] +
    0.20 * bowl_filtered['z_dot'] +
    0.20 * bowl_filtered['z_consistency']
)

top15_bowlers = bowl_filtered.sort_values(by='bowling_pressure_index', ascending=False).head(15)
print("\n=== TOP 15 BOWLERS UNDER PRESSURE ===")
cols_bowl = ['bowler_name', 'press_balls', 'er_press', 'er_base', 'true_er_saved', 'wkt_pct_press', 'dot_pct_press', 'bowl_consistency_rate', 'bowling_pressure_index']
print(top15_bowlers[cols_bowl].to_string(index=False))
