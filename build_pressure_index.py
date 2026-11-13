import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

DATA_DIR = r"C:\Users\ASUS\.gemini\antigravity\scratch\RR"
bbb = pd.read_csv(os.path.join(DATA_DIR, "ball_by_ball.csv"))
matches = pd.read_csv(os.path.join(DATA_DIR, "matches.csv"))
competitions = pd.read_csv(os.path.join(DATA_DIR, "competitions.csv"))

bbb = bbb.sort_values(by=['match_id', 'innings', 'overid', 'over_ball_num']).reset_index(drop=True)

# Merge matches and competitions
bbb = bbb.merge(matches[['match_id', 'comp_id', 'knock_out', 'revised_target', 'win_team_id']], on='match_id', how='left')
bbb = bbb.merge(competitions[['comp_id', 'comp_name', 'match_format', 'season']], on='comp_id', how='left')

bbb['is_legal'] = ~bbb['extras_type'].isin(['wide', 'Overstep'])
bbb['total_ball_runs'] = bbb['runs'] + bbb['extras1_runs'].fillna(0) + bbb['extras_2_runs'].fillna(0)

is_bye_legbye = bbb['extras_type'].isin(['byes', 'legbyes', 'Penalty'])
bbb['bowler_runs_conceded'] = bbb['runs'] + np.where(is_bye_legbye, 0, bbb['extras1_runs'].fillna(0) + bbb['extras_2_runs'].fillna(0))

bowler_dismissals = ['Caught', 'Bowled', 'LBW', 'Stump', 'Hit Wicket']
bbb['is_bowler_wicket'] = bbb['is_wicket'] & bbb['dismissal_type'].isin(bowler_dismissals)

# Cumulative states
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

# Fit Win Probability models
# 1. Innings 2
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

# 2. Innings 1
inn1_mask = (bbb['innings'] == 1)
bbb.loc[inn1_mask, 'projected_score'] = bbb.loc[inn1_mask, 'prev_cum_runs'] + (bbb.loc[inn1_mask, 'balls_remaining'] / 6.0) * (6.0 + 0.5 * bbb.loc[inn1_mask, 'wickets_in_hand'])
features_inn1 = ['prev_cum_runs', 'balls_remaining', 'wickets_in_hand', 'projected_score']
clf_inn1 = LogisticRegression()
clf_inn1.fit(bbb.loc[inn1_mask, features_inn1], bbb.loc[inn1_mask, 'batting_won'])
bbb.loc[inn1_mask, 'win_prob'] = clf_inn1.predict_proba(bbb.loc[inn1_mask, features_inn1])[:, 1]

# Leverage = 4 * win_prob * (1 - win_prob)
bbb['leverage'] = 4.0 * bbb['win_prob'] * (1.0 - bbb['win_prob'])

# Urgency factor
bbb['urgency'] = 0.5 + 0.5 * ((120 - bbb['balls_remaining']) / 120.0)

# Innings 2 Pressure: Leverage * Urgency * (Strain on resources)
# Strain = 1 + clip((rrr - 8.5) / 10, -0.2, 0.8) + (10 - wickets_in_hand)/20
strain_inn2 = 1.0 + np.clip((bbb['rrr_capped'] - 8.5) / 10.0, -0.3, 0.7) + (10.0 - bbb['wickets_in_hand']) / 25.0
bbb.loc[inn2_mask, 'pressure_raw'] = bbb.loc[inn2_mask, 'leverage'] * bbb.loc[inn2_mask, 'urgency'] * strain_inn2.loc[inn2_mask]

# Innings 1 Pressure: Leverage * Urgency * (Collapse or Death over intensity)
# Collapse factor: losing 3+ wickets in overs 1-10
collapse_factor = np.where((bbb['overid'] <= 10) & (bbb['wickets_in_hand'] <= 7), 1.25, 1.0)
# Death over factor: overs 16-20
death_factor = np.where(bbb['overid'] >= 16, 1.25, 1.0)
bbb.loc[inn1_mask, 'pressure_raw'] = bbb.loc[inn1_mask, 'leverage'] * bbb.loc[inn1_mask, 'urgency'] * collapse_factor[inn1_mask] * death_factor[inn1_mask]

# Knockout multiplier
knockout_mult = np.where(bbb['knock_out'] == 1, 1.15, 1.0)
bbb['pressure_raw'] = bbb['pressure_raw'] * knockout_mult

# Normalize pressure score between 0 and 1 using min-max or percentiles
p_min = bbb['pressure_raw'].quantile(0.01)
p_max = bbb['pressure_raw'].quantile(0.99)
bbb['pressure_score'] = np.clip((bbb['pressure_raw'] - p_min) / (p_max - p_min), 0.0, 1.0)

print("\nPressure Score Distribution:")
print(bbb['pressure_score'].describe(percentiles=[0.25, 0.5, 0.7, 0.75, 0.8, 0.9]))

# Check correlation / relationship with game state
print("\nMean Pressure Score by Innings and Over Range:")
bbb['over_range'] = pd.cut(bbb['overid'], bins=[0, 6, 15, 20], labels=['Powerplay (1-6)', 'Middle (7-15)', 'Death (16-20)'])
print(bbb.groupby(['innings', 'over_range'])['pressure_score'].mean())
