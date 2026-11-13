"""
Rajasthan Royals: SupeRR Selector Season 2
Round 1: Identifying the Most Reliable Players Under Pressure
Author: Aryan Mathur
End-to-End Analytics Pipeline
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    CHARTS_DIR = os.path.join(BASE_DIR, "charts")
    os.makedirs(CHARTS_DIR, exist_ok=True)

    print("=" * 60)
    print("Rajasthan Royals SupeRR Selector S2: Pressure Analytics Pipeline")
    print("=" * 60)

    # 1. Load Data
    print("\n[1/5] Loading match and delivery records...")
    bbb_path = os.path.join(BASE_DIR, "ball_by_ball.csv")
    matches_path = os.path.join(BASE_DIR, "matches.csv")
    competitions_path = os.path.join(BASE_DIR, "competitions.csv")

    if not os.path.exists(bbb_path):
        print(f"Error: {bbb_path} not found. Please place ball_by_ball.csv in the directory.")
        return

    bbb = pd.read_csv(bbb_path)
    matches = pd.read_csv(matches_path)
    competitions = pd.read_csv(competitions_path)

    # Sort
    bbb = bbb.sort_values(by=['match_id', 'innings', 'overid', 'over_ball_num']).reset_index(drop=True)

    # Merge metadata
    bbb = bbb.merge(matches[['match_id', 'comp_id', 'knock_out', 'revised_target', 'win_team_id']], on='match_id', how='left')
    bbb = bbb.merge(competitions[['comp_id', 'comp_name', 'match_format', 'season']], on='comp_id', how='left')

    bbb['is_legal'] = ~bbb['extras_type'].isin(['wide', 'Overstep'])
    bbb['total_ball_runs'] = bbb['runs'] + bbb['extras1_runs'].fillna(0) + bbb['extras_2_runs'].fillna(0)
    bbb['batter_ball_faced'] = bbb['extras_type'] != 'wide'
    bbb['is_dot_batter'] = (bbb['runs'] == 0) & bbb['batter_ball_faced']
    bbb['is_boundary_batter'] = bbb['is_four'] | bbb['is_six']
    bbb['is_batter_out'] = bbb['is_wicket'] & (bbb['dismissed_player_id'] == bbb['striker_id'])

    is_bye_legbye = bbb['extras_type'].isin(['byes', 'legbyes', 'Penalty'])
    bbb['bowler_runs_conceded'] = bbb['runs'] + np.where(is_bye_legbye, 0, bbb['extras1_runs'].fillna(0) + bbb['extras_2_runs'].fillna(0))
    bowler_dismissals = ['Caught', 'Bowled', 'LBW', 'Stump', 'Hit Wicket']
    bbb['is_bowler_wicket'] = bbb['is_wicket'] & bbb['dismissal_type'].isin(bowler_dismissals)

    # 2. Compute Pressure Index (PI)
    print("\n[2/5] Engineering Continuous Pressure Index (PI)...")
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

    # Innings 2 Pressure
    inn2_mask = (bbb['innings'] == 2)
    bbb.loc[inn2_mask, 'runs_needed'] = np.maximum(0, bbb.loc[inn2_mask, 'target'] - bbb.loc[inn2_mask, 'prev_cum_runs'])
    bbb.loc[inn2_mask, 'rrr'] = np.where(bbb.loc[inn2_mask, 'balls_remaining'] > 0,
                                         (bbb.loc[inn2_mask, 'runs_needed'] / bbb.loc[inn2_mask, 'balls_remaining']) * 6.0,
                                         99.0)

    def get_urgency(overid):
        if overid <= 6:
            return 0.60
        elif overid <= 14:
            return 0.85
        elif overid <= 17:
            return 1.15
        else:
            return 1.45

    bbb['urgency'] = bbb['overid'].apply(get_urgency)

    rrr_strain = np.clip(bbb['rrr'] / 8.5, 0.3, 2.5)
    wkt_stress_inn2 = np.clip((11.0 - bbb['wickets_in_hand']) / 5.0, 0.4, 2.0)
    p2_raw = rrr_strain * wkt_stress_inn2 * bbb['urgency']

    is_already_won = (bbb['runs_needed'] <= 10) & (bbb['balls_remaining'] >= 18) & (bbb['wickets_in_hand'] >= 6)
    is_impossible = (bbb['rrr'] > 28.0) & (bbb['wickets_in_hand'] <= 3) & (bbb['balls_remaining'] <= 18)
    p2_raw = np.where(is_already_won | is_impossible, 0.3, p2_raw)

    # Innings 1 Pressure
    inn1_phase = np.where(bbb['overid'] <= 6, 0.70,
                 np.where(bbb['overid'] <= 14, 0.65,
                 np.where(bbb['overid'] <= 17, 1.15, 1.50)))
    collapse_mult = np.where((bbb['overid'] <= 6) & (bbb['wickets_in_hand'] <= 7), 1.40,
                    np.where((bbb['overid'] <= 10) & (bbb['wickets_in_hand'] <= 6), 1.50, 1.00))
    p1_raw = inn1_phase * collapse_mult

    bbb['pressure_raw'] = np.where(bbb['innings'] == 2, p2_raw, p1_raw)
    bbb['pressure_raw'] = bbb['pressure_raw'] * np.where(bbb['knock_out'] == 1, 1.20, 1.00)
    bbb['pressure_score'] = bbb['pressure_raw'].rank(pct=True)
    bbb['is_high_pressure'] = bbb['pressure_score'] >= 0.70

    # Competition weights
    is_ipl = bbb['comp_name'].str.contains('IPL', case=False, na=False)
    bbb['tier_weight'] = np.where(is_ipl, 1.25, np.where(bbb['match_format'] == 'T20I', 1.00, 0.85))

    # Match-level high pressure expectations
    m_inn_p = bbb[bbb['is_high_pressure']].groupby(['match_id', 'innings']).agg(
        hp_balls_faced=('batter_ball_faced', 'sum'),
        hp_runs=('runs', 'sum'),
        hp_legal_bowled=('is_legal', 'sum'),
        hp_bowler_runs=('bowler_runs_conceded', 'sum')
    ).reset_index()
    m_inn_p['match_hp_sr'] = (m_inn_p['hp_runs'] / np.maximum(1, m_inn_p['hp_balls_faced'])) * 100.0
    m_inn_p['match_hp_er'] = (m_inn_p['hp_bowler_runs'] / np.maximum(1, m_inn_p['hp_legal_bowled'])) * 6.0
    bbb = bbb.merge(m_inn_p[['match_id', 'innings', 'match_hp_sr', 'match_hp_er']], on=['match_id', 'innings'], how='left')

    global_hp_sr = (bbb[bbb['batter_ball_faced'] & bbb['is_high_pressure']]['runs'].sum() /
                    bbb[bbb['batter_ball_faced'] & bbb['is_high_pressure']]['batter_ball_faced'].sum()) * 100.0
    global_hp_er = (bbb[bbb['is_legal'] & bbb['is_high_pressure']]['bowler_runs_conceded'].sum() /
                    bbb[bbb['is_legal'] & bbb['is_high_pressure']]['is_legal'].sum()) * 6.0
    bbb['match_hp_sr'] = bbb['match_hp_sr'].fillna(global_hp_sr)
    bbb['match_hp_er'] = bbb['match_hp_er'].fillna(global_hp_er)

    def zscore(s):
        return (s - s.mean()) / s.std()

    # 3. Batting Pipeline
    print("\n[3/5] Evaluating Batters (Contextual True-SR, Shrinkage & Consistency)...")
    bat_hp = bbb[bbb['batter_ball_faced'] & bbb['is_high_pressure']].copy()
    bat_hp['runs_above_match_exp'] = bat_hp['runs'] - (bat_hp['match_hp_sr'] / 100.0)

    bat_stats = bat_hp.groupby(['striker_id', 'striker_name']).agg(
        press_balls=('batter_ball_faced', 'sum'),
        press_runs=('runs', 'sum'),
        press_outs=('is_batter_out', 'sum'),
        press_dots=('is_dot_batter', 'sum'),
        press_bndys=('is_boundary_batter', 'sum'),
        press_matches=('match_id', 'nunique'),
        total_runs_above_match=('runs_above_match_exp', 'sum'),
        avg_tier=('tier_weight', 'mean')
    ).reset_index()

    bat_stats['sr_press'] = (bat_stats['press_runs'] / bat_stats['press_balls']) * 100.0
    bat_stats['true_sr_diff'] = (bat_stats['total_runs_above_match'] / bat_stats['press_balls']) * 100.0
    bat_stats['avg_press'] = bat_stats['press_runs'] / np.maximum(1, bat_stats['press_outs'])

    bat_all = bbb[bbb['batter_ball_faced']].groupby('striker_id').agg(
        total_balls=('batter_ball_faced', 'sum'),
        total_runs=('runs', 'sum')
    ).reset_index()
    bat_all['sr_base'] = (bat_all['total_runs'] / bat_all['total_balls']) * 100.0

    bat_merged = bat_stats.merge(bat_all[['striker_id', 'sr_base']], on='striker_id', how='left')
    bat_merged['delta_sr'] = bat_merged['sr_press'] - bat_merged['sr_base']

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

    bat_cand = bat_merged[bat_merged['press_balls'] >= 100].copy()
    K_bat = 30
    bat_cand['true_sr_shrunk'] = (bat_cand['press_balls'] / (bat_cand['press_balls'] + K_bat)) * bat_cand['true_sr_diff']

    bat_cand['z_true_sr'] = zscore(bat_cand['true_sr_shrunk'])
    bat_cand['z_delta_sr'] = zscore(bat_cand['delta_sr'])
    bat_cand['z_avg_press'] = zscore(np.clip(bat_cand['avg_press'], 10, 55))
    bat_cand['z_consistency'] = zscore(bat_cand['bat_consistency'])

    bat_cand['composite_score'] = (
        0.35 * bat_cand['z_true_sr'] +
        0.25 * bat_cand['z_delta_sr'] +
        0.20 * bat_cand['z_avg_press'] +
        0.20 * bat_cand['z_consistency']
    ) * bat_cand['avg_tier']

    min_s, max_s = bat_cand['composite_score'].min(), bat_cand['composite_score'].max()
    bat_cand['pressure_rating_100'] = ((bat_cand['composite_score'] - min_s) / (max_s - min_s)) * 100.0

    top10_batters = bat_cand.sort_values(by='composite_score', ascending=False).head(10).reset_index(drop=True)
    top10_batters['Rank'] = range(1, 11)

    # 4. Bowling Pipeline
    print("\n[4/5] Evaluating Bowlers (Contextual True-ER, Wickets & Consistency)...")
    bowl_hp = bbb[bbb['is_legal'] & bbb['is_high_pressure']].copy()
    bowl_hp['runs_saved_vs_match'] = (bowl_hp['match_hp_er'] / 6.0) - bowl_hp['bowler_runs_conceded']

    bowl_stats = bowl_hp.groupby(['bowler_id', 'bowler_name']).agg(
        press_balls=('is_legal', 'sum'),
        press_runs=('bowler_runs_conceded', 'sum'),
        press_wkts=('is_bowler_wicket', 'sum'),
        press_dots=('runs', lambda x: ((x == 0) & (bbb.loc[x.index, 'is_legal'])).sum()),
        press_matches=('match_id', 'nunique'),
        total_runs_saved=('runs_saved_vs_match', 'sum'),
        avg_tier=('tier_weight', 'mean')
    ).reset_index()

    bowl_stats['er_press'] = (bowl_stats['press_runs'] / bowl_stats['press_balls']) * 6.0
    bowl_stats['true_er_saved'] = (bowl_stats['total_runs_saved'] / bowl_stats['press_balls']) * 6.0
    bowl_stats['wkt_pct_press'] = (bowl_stats['press_wkts'] / bowl_stats['press_balls']) * 100.0
    bowl_stats['dot_pct_press'] = (bowl_stats['press_dots'] / bowl_stats['press_balls']) * 100.0

    bowl_all = bbb[bbb['is_legal']].groupby('bowler_id').agg(
        total_balls=('is_legal', 'sum'),
        total_runs=('bowler_runs_conceded', 'sum')
    ).reset_index()
    bowl_all['er_base'] = (bowl_all['total_runs'] / bowl_all['total_balls']) * 6.0

    bowl_merged = bowl_stats.merge(bowl_all[['bowler_id', 'er_base']], on='bowler_id', how='left')
    bowl_merged['delta_er'] = bowl_merged['er_base'] - bowl_merged['er_press']

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

    bowl_cand = bowl_merged[bowl_merged['press_balls'] >= 100].copy()
    K_bowl = 30
    bowl_cand['true_er_shrunk'] = (bowl_cand['press_balls'] / (bowl_cand['press_balls'] + K_bowl)) * bowl_cand['true_er_saved']

    bowl_cand['z_true_er'] = zscore(bowl_cand['true_er_shrunk'])
    bowl_cand['z_wkt'] = zscore(bowl_cand['wkt_pct_press'])
    bowl_cand['z_dot'] = zscore(bowl_cand['dot_pct_press'])
    bowl_cand['z_consistency'] = zscore(bowl_cand['bowl_consistency'])

    bowl_cand['composite_score'] = (
        0.35 * bowl_cand['z_true_er'] +
        0.25 * bowl_cand['z_wkt'] +
        0.20 * bowl_cand['z_dot'] +
        0.20 * bowl_cand['z_consistency']
    ) * bowl_cand['avg_tier']

    min_sb, max_sb = bowl_cand['composite_score'].min(), bowl_cand['composite_score'].max()
    bowl_cand['pressure_rating_100'] = ((bowl_cand['composite_score'] - min_sb) / (max_sb - min_sb)) * 100.0

    top10_bowlers = bowl_cand.sort_values(by='composite_score', ascending=False).head(10).reset_index(drop=True)
    top10_bowlers['Rank'] = range(1, 11)

    # Export Top 10 CSVs
    top10_batters.to_csv(os.path.join(BASE_DIR, "top10_batters.csv"), index=False)
    top10_bowlers.to_csv(os.path.join(BASE_DIR, "top10_bowlers.csv"), index=False)

    print("\n[5/5] Pipeline Completed Successfully!")
    print("\n--- TOP 10 BATTERS ---")
    print(top10_batters[['Rank', 'striker_name', 'press_balls', 'sr_press', 'sr_base', 'delta_sr', 'pressure_rating_100']].to_string(index=False))
    print("\n--- TOP 10 BOWLERS ---")
    print(top10_bowlers[['Rank', 'bowler_name', 'press_balls', 'er_press', 'er_base', 'true_er_saved', 'pressure_rating_100']].to_string(index=False))

if __name__ == "__main__":
    main()
