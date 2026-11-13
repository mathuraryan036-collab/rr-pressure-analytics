import os
import pandas as pd
import numpy as np

DATA_DIR = r"C:\Users\ASUS\.gemini\antigravity\scratch\RR"
bbb = pd.read_csv(os.path.join(DATA_DIR, "ball_by_ball.csv"))
matches = pd.read_csv(os.path.join(DATA_DIR, "matches.csv"))
competitions = pd.read_csv(os.path.join(DATA_DIR, "competitions.csv"))

bbb = bbb.sort_values(by=['match_id', 'innings', 'overid', 'over_ball_num']).reset_index(drop=True)

bbb = bbb.merge(matches[['match_id', 'comp_id', 'knock_out', 'revised_target', 'win_team_id']], on='match_id', how='left')
bbb = bbb.merge(competitions[['comp_id', 'comp_name', 'match_format', 'season']], on='comp_id', how='left')

bbb['is_legal'] = ~bbb['extras_type'].isin(['wide', 'Overstep'])
bbb['total_ball_runs'] = bbb['runs'] + bbb['extras1_runs'].fillna(0) + bbb['extras_2_runs'].fillna(0)

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

inn1_totals = bbb[bbb['innings'] == 1].groupby('match_id')['total_ball_runs'].sum().reset_index()
inn1_totals.columns = ['match_id', 'inn1_total']
bbb = bbb.merge(inn1_totals, on='match_id', how='left')

bbb['target'] = np.where(bbb['revised_target'].notna() & (bbb['revised_target'] > 0),
                         bbb['revised_target'],
                         bbb['inn1_total'] + 1)

inn2_mask = (bbb['innings'] == 2)
bbb.loc[inn2_mask, 'runs_needed'] = np.maximum(0, bbb.loc[inn2_mask, 'target'] - bbb.loc[inn2_mask, 'prev_cum_runs'])
bbb.loc[inn2_mask, 'rrr'] = np.where(bbb.loc[inn2_mask, 'balls_remaining'] > 0,
                                     (bbb.loc[inn2_mask, 'runs_needed'] / bbb.loc[inn2_mask, 'balls_remaining']) * 6.0,
                                     99.0)

def get_urgency(overid):
    if overid <= 6:
        return 0.6
    elif overid <= 14:
        return 0.85
    elif overid <= 17:
        return 1.15
    else:
        return 1.45

bbb['urgency'] = bbb['overid'].apply(get_urgency)

# Pressure Index Calculation
rrr_strain = np.clip(bbb['rrr'] / 8.5, 0.3, 2.5)
wkt_stress_inn2 = np.clip((11.0 - bbb['wickets_in_hand']) / 5.0, 0.4, 2.0)
p2_raw = rrr_strain * wkt_stress_inn2 * bbb['urgency']

is_already_won = (bbb['runs_needed'] <= 10) & (bbb['balls_remaining'] >= 18) & (bbb['wickets_in_hand'] >= 6)
is_impossible = (bbb['rrr'] > 28.0) & (bbb['wickets_in_hand'] <= 3) & (bbb['balls_remaining'] <= 18)
p2_raw = np.where(is_already_won | is_impossible, 0.3, p2_raw)

inn1_phase = np.where(bbb['overid'] <= 6, 0.7,
             np.where(bbb['overid'] <= 14, 0.65,
             np.where(bbb['overid'] <= 17, 1.15, 1.5)))

collapse_mult = np.where((bbb['overid'] <= 6) & (bbb['wickets_in_hand'] <= 7), 1.4,
                np.where((bbb['overid'] <= 10) & (bbb['wickets_in_hand'] <= 6), 1.5, 1.0))
p1_raw = inn1_phase * collapse_mult

bbb['pressure_raw'] = np.where(bbb['innings'] == 2, p2_raw, p1_raw)
bbb['pressure_raw'] = bbb['pressure_raw'] * np.where(bbb['knock_out'] == 1, 1.2, 1.0)
bbb['pressure_score'] = bbb['pressure_raw'].rank(pct=True)

# Define high pressure binary flag (top 30% most stressful deliveries)
bbb['is_high_pressure'] = bbb['pressure_score'] >= 0.70

print(f"Total deliveries: {len(bbb)}")
print(f"High pressure deliveries: {bbb['is_high_pressure'].sum()} ({bbb['is_high_pressure'].mean()*100:.1f}%)")

# Save processed dataset with pressure score for analytics
bbb.to_feather(os.path.join(DATA_DIR, "bbb_with_pressure.feather"))
print("Saved bbb_with_pressure.feather successfully.")
