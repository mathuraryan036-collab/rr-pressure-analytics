import os
import pandas as pd

DATA_DIR = r"C:\Users\ASUS\.gemini\antigravity\scratch\RR"
# Let's inspect bat_cand and bowl_cand top 30
import generate_final_rankings as gfr

print("\n--- BATTERS RANK 11 to 30 ---")
cols_bat_display = ['striker_name', 'press_balls', 'sr_press', 'sr_base', 'delta_sr', 'avg_press', 'bat_consistency', 'pressure_rating_100']
print(gfr.bat_cand.sort_values(by='composite_score', ascending=False).iloc[10:30][cols_bat_display].to_string())

print("\n--- BOWLERS RANK 11 to 30 ---")
cols_bowl_display = ['bowler_name', 'press_balls', 'er_press', 'er_base', 'true_er_saved', 'wkt_pct_press', 'dot_pct_press', 'bowl_consistency', 'pressure_rating_100']
print(gfr.bowl_cand.sort_values(by='composite_score', ascending=False).iloc[10:30][cols_bowl_display].to_string())

# Search specific names
marquee_batters = ['Heinrich Klaasen', 'Nicholas Pooran', 'Rinku Singh', 'Andre Russell', 'MS Dhoni', 'Virat Kohli', 'Rohit Sharma', 'Hardik Pandya', 'Marcus Stoinis']
print("\n--- MARQUEE BATTERS STATS ---")
print(gfr.bat_cand[gfr.bat_cand['striker_name'].isin(marquee_batters)][cols_bat_display].to_string())

marquee_bowlers = ['Jasprit Bumrah', 'Rashid Khan', 'Sunil Narine', 'Matheesha Pathirana', 'Arshdeep Singh', 'Trent Boult', 'Anrich Nortje', 'Mohammad Shami', 'Kuldeep Yadav']
print("\n--- MARQUEE BOWLERS STATS ---")
print(gfr.bowl_cand[gfr.bowl_cand['bowler_name'].isin(marquee_bowlers)][cols_bowl_display].to_string())
