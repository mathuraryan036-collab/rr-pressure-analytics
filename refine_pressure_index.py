import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

DATA_DIR = r"C:\Users\ASUS\.gemini\antigravity\scratch\RR"
bbb = pd.read_csv(os.path.join(DATA_DIR, "ball_by_ball.csv"))
matches = pd.read_csv(os.path.join(DATA_DIR, "matches.csv"))
competitions = pd.read_csv(os.path.join(DATA_DIR, "competitions.csv"))

bbb = bbb.sort_values(by=['match_id', 'innings', 'overid', 'over_ball_num']).reset_index(drop=True)

# Merge
bbb = bbb.merge(matches[['match_id', 'comp_id', 'knock_out', 'revised_target', 'win_team_id']], on='match_id', how='left')
bbb = bbb.merge(competitions[['comp_id', 'comp_name', 'match_format', 'season']], on='comp_id', how='left')

bbb['is_legal'] = ~bbb['extras_type'].isin(['wide', 'Overstep'])
bbb['total_ball_runs'] = bbb['runs'] + bbb['extras1_runs'].fillna(0) + bbb['extras_2_runs'].fillna(0)

# Bowler runs & wickets
is_bye_legbye = bbb['extras_type'].isin(['byes', 'legbyes', 'Penalty'])
bbb['bowler_runs_conceded'] = bbb['runs'] + np.where(is_bye_legbye, 0, bbb['extras1_runs'].fillna(0) + bbb['extras_2_runs'].fillna(0))
bowler_dismissals = ['Caught', 'Bowled', 'LBW', 'Stump', 'Hit Wicket']
bbb['is_bowler_wicket'] = bbb['is_wicket'] & bbb['dismissal_type'].isin(bowler_dismissals)

grouped = bbb.groupby(['match_id', 'innings'])
bbb['prev_cum_runs'] = grouped['total_ball_runs'].cumsum() - bbb['total_ball_runs']
bbb['prev_cum_wickets'] = grouped['is_wicket'].cumsum() - bbb['is_wicket']
bbb['prev_legal_balls'] = grouped['is_legal'].cumsum() - bbb['is_legal']

bbb['balls_remaining'] = np.maximum(0, 120 - bbb['prev_legal_balls'])
bbb['wickets_in_hand'] = np.maximum(0, 10 - bbb['prev_cum_wickets'])
bbb['batting_won'] = (bbb['win_team_id'] == bbb['batting_team_id']).astype(int)

# Target
inn1_totals = bbb[bbb['innings'] == 1].groupby('match_id')['total_ball_runs'].sum().reset_index()
inn1_totals.columns = ['match_id', 'inn1_total']
bbb = bbb.merge(inn1_totals, on='match_id', how='left')

bbb['target'] = np.where(bbb['revised_target'].notna() & (bbb['revised_target'] > 0),
                         bbb['revised_target'],
                         bbb['inn1_total'] + 1)

# Fit Innings 2 model
inn2_mask = (bbb['innings'] == 2)
bbb.loc[inn2_mask, 'runs_needed'] = np.maximum(0, bbb.loc[inn2_mask, 'target'] - bbb.loc[inn2_mask, 'prev_cum_runs'])
bbb.loc[inn2_mask, 'rrr'] = np.where(bbb.loc[inn2_mask, 'balls_remaining'] > 0,
                                     (bbb.loc[inn2_mask, 'runs_needed'] / bbb.loc[inn2_mask, 'balls_remaining']) * 6.0,
                                     99.0)
bbb.loc[inn2_mask, 'rrr_capped'] = np.clip(bbb.loc[inn2_mask, 'rrr'], 0, 36)

features_inn2 = ['runs_needed', 'balls_remaining', 'wickets_in_hand', 'rrr_capped']
clf_inn2 = LogisticRegression()
clf_inn2.fit(bbb.loc[inn2_mask, features_inn2], bbb.loc[inn2_mask, 'batting_won'])
bbb.loc[inn2_mask, 'win_prob'] = clf_inn2.predict_proba(bbb.loc[inn2_mask, features_inn2])[:, 1]

# Fit Innings 1 model
inn1_mask = (bbb['innings'] == 1)
bbb.loc[inn1_mask, 'projected_score'] = bbb.loc[inn1_mask, 'prev_cum_runs'] + (bbb.loc[inn1_mask, 'balls_remaining'] / 6.0) * (6.0 + 0.5 * bbb.loc[inn1_mask, 'wickets_in_hand'])
features_inn1 = ['prev_cum_runs', 'balls_remaining', 'wickets_in_hand', 'projected_score']
clf_inn1 = LogisticRegression()
clf_inn1.fit(bbb.loc[inn1_mask, features_inn1], bbb.loc[inn1_mask, 'batting_won'])
bbb.loc[inn1_mask, 'win_prob'] = clf_inn1.predict_proba(bbb.loc[inn1_mask, features_inn1])[:, 1]

# --- PRESSURE INDEX REFINED ---
# 1. Base Leverage: 4 * P * (1 - P) [measures match closeness, peaks at 0.5]
# But in a chase, even if win prob is 20-30%, when RRR is 11-14 with overs 15-20, that is massive pressure!
# We define Win Probability Competitiveness:
# High when win_prob is between 0.10 and 0.85
bbb['closeness'] = np.where(
    (bbb['win_prob'] >= 0.05) & (bbb['win_prob'] <= 0.95),
    np.sin(np.pi * bbb['win_prob']), # smooth bell curve peaking at 0.50
    0.1
)

# 2. Stage Urgency:
# Overs 1-6: 0.6, Overs 7-15: 0.8, Overs 16-20: 1.0
bbb['stage_factor'] = 0.5 + 0.5 * ((120 - bbb['balls_remaining']) / 120.0)

# 3. Situational Stress:
# Innings 2:
# RRR strain: RRR relative to 8.5 par
# Wickets pressure: fewer wickets in hand = more pressure
rrr_strain = np.clip(bbb['rrr'] / 8.5, 0.5, 2.5)
wkt_pressure_inn2 = (11.0 - bbb['wickets_in_hand']) / 7.0 # ranges from ~0.4 to 1.4

# Innings 1:
# Collapse: losing wickets early
early_collapse = np.where((bbb['overid'] <= 10) & (bbb['wickets_in_hand'] <= 7), 1.35, 1.0)
death_inn1 = np.where(bbb['overid'] >= 16, 1.4, 1.0)

# Combine into continuous Pressure Score
p_inn2 = bbb['closeness'] * bbb['stage_factor'] * rrr_strain * wkt_pressure_inn2
# Exclude dead rubber junk time in chase (e.g. need > 40 off 6 balls or need 3 off 30 balls)
is_dead_chase = (bbb['win_prob'] < 0.02) | (bbb['win_prob'] > 0.98)
p_inn2 = np.where(is_dead_chase, 0.1, p_inn2)

p_inn1 = bbb['closeness'] * bbb['stage_factor'] * early_collapse * death_inn1

bbb['pressure_raw'] = np.where(bbb['innings'] == 2, p_inn2, p_inn1)

# Knockout multiplier
bbb['pressure_raw'] = bbb['pressure_raw'] * np.where(bbb['knock_out'] == 1, 1.2, 1.0)

# Scale to 0-1
p_min = bbb['pressure_raw'].quantile(0.02)
p_max = bbb['pressure_raw'].quantile(0.98)
bbb['pressure_score'] = np.clip((bbb['pressure_raw'] - p_min) / (p_max - p_min), 0.0, 1.0)

# Print quantiles
print("Pressure Score Distribution:")
print(bbb['pressure_score'].describe(percentiles=[0.25, 0.5, 0.7, 0.75, 0.8, 0.9]))

print("\nMean Pressure by Innings and Phase:")
bbb['phase'] = pd.cut(bbb['overid'], bins=[0, 6, 15, 20], labels=['Powerplay', 'Middle', 'Death'])
print(bbb.groupby(['innings', 'phase'])['pressure_score'].mean())

# Check sample top pressure balls
print("\nSample Highest Pressure Deliveries:")
top_p = bbb.sort_values(by='pressure_score', ascending=False)[
    ['match_id', 'innings', 'overid', 'over_ball_num', 'striker_name', 'bowler_name',
     'runs_needed', 'balls_remaining', 'wickets_in_hand', 'rrr', 'win_prob', 'pressure_score']
].head(10)
print(top_p.to_string())

# Check sample lowest pressure deliveries
print("\nSample Lowest Pressure Deliveries:")
low_p = bbb.sort_values(by='pressure_score', ascending=True)[
    ['match_id', 'innings', 'overid', 'over_ball_num', 'striker_name', 'bowler_name',
     'runs_needed', 'balls_remaining', 'wickets_in_hand', 'rrr', 'win_prob', 'pressure_score']
].head(10)
print(low_p.to_string())
