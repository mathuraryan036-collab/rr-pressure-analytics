import os
import pandas as pd

DATA_DIR = r"C:\Users\ASUS\.gemini\antigravity\scratch\RR"

competitions = pd.read_csv(os.path.join(DATA_DIR, "competitions.csv"))
matches = pd.read_csv(os.path.join(DATA_DIR, "matches.csv"))
bbb = pd.read_csv(os.path.join(DATA_DIR, "ball_by_ball.csv"))

print("Total ball-by-ball rows:", len(bbb))
print("Unique matches in BBB:", bbb['match_id'].nunique())
print("Unique matches in matches.csv:", matches['match_id'].nunique())
print("Matches in BBB not in matches.csv:", set(bbb['match_id']) - set(matches['match_id']))
print("Matches in matches.csv not in BBB:", len(set(matches['match_id']) - set(bbb['match_id'])))

# Check unique players
print("Unique strikers:", bbb['striker_id'].nunique())
print("Unique bowlers:", bbb['bowler_id'].nunique())

# Check total match formats / competitions merged
m_comp = matches.merge(competitions, on='comp_id', how='left')
print("\nMatches by competition type:")
m_comp['is_ipl'] = m_comp['comp_name'].str.contains('IPL', case=False, na=False)
print(m_comp.groupby(['is_ipl', 'match_format'])['match_id'].count())
