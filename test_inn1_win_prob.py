import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

DATA_DIR = r"C:\Users\ASUS\.gemini\antigravity\scratch\RR"
bbb = pd.read_csv(os.path.join(DATA_DIR, "ball_by_ball.csv"))
matches = pd.read_csv(os.path.join(DATA_DIR, "matches.csv"))

bbb = bbb.sort_values(by=['match_id', 'innings', 'overid', 'over_ball_num']).reset_index(drop=True)
m_meta = matches[['match_id', 'comp_id', 'knock_out', 'revised_target', 'win_team_id']]
bbb = bbb.merge(m_meta, on='match_id', how='left')

bbb['is_legal'] = ~bbb['extras_type'].isin(['wide', 'Overstep'])
bbb['total_ball_runs'] = bbb['runs'] + bbb['extras1_runs'].fillna(0) + bbb['extras_2_runs'].fillna(0)

grouped = bbb.groupby(['match_id', 'innings'])
bbb['prev_cum_runs'] = grouped['total_ball_runs'].cumsum() - bbb['total_ball_runs']
bbb['prev_cum_wickets'] = grouped['is_wicket'].cumsum() - bbb['is_wicket']
bbb['prev_legal_balls'] = grouped['is_legal'].cumsum() - bbb['is_legal']

bbb['balls_remaining'] = np.maximum(0, 120 - bbb['prev_legal_balls'])
bbb['wickets_in_hand'] = np.maximum(0, 10 - bbb['prev_cum_wickets'])
bbb['batting_won'] = (bbb['win_team_id'] == bbb['batting_team_id']).astype(int)

inn1 = bbb[bbb['innings'] == 1].copy()
inn1['current_run_rate'] = np.where(inn1['prev_legal_balls'] > 0, (inn1['prev_cum_runs'] / inn1['prev_legal_balls']) * 6.0, 8.0)

# Projected score assuming 8 rpo from remaining balls with current wickets adjustment
inn1['projected_score'] = inn1['prev_cum_runs'] + (inn1['balls_remaining'] / 6.0) * (6.0 + 0.5 * inn1['wickets_in_hand'])

features_inn1 = ['prev_cum_runs', 'balls_remaining', 'wickets_in_hand', 'projected_score']
clf_inn1 = LogisticRegression()
clf_inn1.fit(inn1[features_inn1], inn1['batting_won'])

print("Innings 1 Win Probability Model:")
for f, c in zip(features_inn1, clf_inn1.coef_[0]):
    print(f"  {f}: {c:.4f}")
print(f"  Intercept: {clf_inn1.intercept_[0]:.4f}")

# Sample states in Innings 1:
test_inn1 = pd.DataFrame([
    # [prev_cum_runs, balls_remaining, wickets_in_hand, projected_score]
    [25, 90, 7, 25 + 15*(6 + 3.5)],  # 25/3 at over 5 (collapse)
    [50, 90, 10, 50 + 15*(6 + 5.0)], # 50/0 at over 5 (flying start)
    [120, 30, 8, 120 + 5*(6 + 4.0)], # 120/2 at over 15 (poised for big finish)
    [100, 30, 4, 100 + 5*(6 + 2.0)], # 100/6 at over 15 (struggling)
    [170, 12, 6, 170 + 2*(6 + 3.0)], # 170/4 at over 18 (death overs, high score)
], columns=features_inn1)

probs_inn1 = clf_inn1.predict_proba(test_inn1)[:, 1]
print("\nSample Innings 1 Win Probabilities:")
for i, row in test_inn1.iterrows():
    print(f"  State: {row['prev_cum_runs']:.0f} runs, {row['balls_remaining']:.0f} balls left, {row['wickets_in_hand']:.0f} wkts in hand -> Win Prob: {probs_inn1[i]*100:.1f}%")
