import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

DATA_DIR = r"C:\Users\ASUS\.gemini\antigravity\scratch\RR"
bbb = pd.read_csv(os.path.join(DATA_DIR, "ball_by_ball.csv"))
matches = pd.read_csv(os.path.join(DATA_DIR, "matches.csv"))

# Sort
bbb = bbb.sort_values(by=['match_id', 'innings', 'overid', 'over_ball_num']).reset_index(drop=True)

# Merge matches
m_meta = matches[['match_id', 'comp_id', 'knock_out', 'revised_target', 'win_team_id']]
bbb = bbb.merge(m_meta, on='match_id', how='left')

bbb['is_legal'] = ~bbb['extras_type'].isin(['wide', 'Overstep'])
bbb['total_ball_runs'] = bbb['runs'] + bbb['extras1_runs'].fillna(0) + bbb['extras_2_runs'].fillna(0)

# Cumulative totals
grouped = bbb.groupby(['match_id', 'innings'])
bbb['prev_cum_runs'] = grouped['total_ball_runs'].cumsum() - bbb['total_ball_runs']
bbb['prev_cum_wickets'] = grouped['is_wicket'].cumsum() - bbb['is_wicket']
bbb['prev_legal_balls'] = grouped['is_legal'].cumsum() - bbb['is_legal']

bbb['balls_remaining'] = np.maximum(0, 120 - bbb['prev_legal_balls'])
bbb['wickets_in_hand'] = np.maximum(0, 10 - bbb['prev_cum_wickets'])

# Innings 1 total
inn1_totals = bbb[bbb['innings'] == 1].groupby('match_id')['total_ball_runs'].sum().reset_index()
inn1_totals.columns = ['match_id', 'inn1_total']
bbb = bbb.merge(inn1_totals, on='match_id', how='left')

bbb['target'] = np.where(bbb['revised_target'].notna() & (bbb['revised_target'] > 0),
                         bbb['revised_target'],
                         bbb['inn1_total'] + 1)

# Chasing team won?
bbb['batting_won'] = (bbb['win_team_id'] == bbb['batting_team_id']).astype(int)

# Innings 2 subset
inn2 = bbb[bbb['innings'] == 2].copy()
inn2['runs_needed'] = np.maximum(0, inn2['target'] - inn2['prev_cum_runs'])
inn2['rrr'] = np.where(inn2['balls_remaining'] > 0, (inn2['runs_needed'] / inn2['balls_remaining']) * 6.0, 99.0)
inn2['rrr_capped'] = np.clip(inn2['rrr'], 0, 36)

# Fit Logistic Regression on Innings 2 Win Probability
features = ['runs_needed', 'balls_remaining', 'wickets_in_hand', 'rrr_capped']
X = inn2[features]
y = inn2['batting_won']

clf = LogisticRegression()
clf.fit(X, y)

print("Logistic Regression Coefficients for Innings 2 Win Probability:")
for f, c in zip(features, clf.coef_[0]):
    print(f"  {f}: {c:.4f}")
print(f"  Intercept: {clf.intercept_[0]:.4f}")

# Check predictions on sample test states
test_states = pd.DataFrame([
    # [runs_needed, balls_remaining, wickets_in_hand, rrr_capped]
    [30, 18, 6, 10.0],  # Close finish, 6 wkts in hand
    [30, 18, 2, 10.0],  # Close finish, only 2 wkts in hand
    [15, 30, 8, 3.0],   # Cruising to win
    [60, 12, 3, 30.0],  # Virtually impossible chase
    [100, 60, 8, 10.0], # Middle overs, balanced chase
    [150, 100, 9, 9.0], # Early chase
], columns=features)

test_probs = clf.predict_proba(test_states)[:, 1]
print("\nSample State Win Probabilities:")
for i, row in test_states.iterrows():
    print(f"  Need {row['runs_needed']:.0f} off {row['balls_remaining']:.0f} balls, {row['wickets_in_hand']:.0f} wkts (RRR {row['rrr_capped']:.1f}) -> Win Prob: {test_probs[i]*100:.1f}%")
