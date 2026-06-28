"""
World Cup 2026 Predictor – main entry point
============================================
Run:  python main.py

Outputs
  • Printed table of Round-1 predictions (all 24 games, 3 models)
  • Sport5 "5 חבר'ה" optimal score recommendations + expected-points analysis
  • Pie-chart PNGs saved to  ./charts/
  • Scoreline heat-maps saved to  ./charts/

To use real data instead of synthetic:
  matches = pd.read_csv("results.csv")   # from Kaggle / martj42
  # must have columns: date, home_team, away_team, home_goals, away_goals,
  #                    venue ("home"/"neutral"), tournament
"""

import time
import pandas as pd

from data   import ALL_TEAMS, WC2026_GROUPS, SPORT5_ODDS
from models import (ELOSystem, DixonColesModel, MLModel, EnsembleModel,
                    load_real_data, build_ml_features,
                    tune_elo_k, validate_ensemble_weights)
from predict import (predict_group_stage, predict_round1,
                     print_group_stage_summary, print_sport5_strategy,
                     predict_knockout_stage, print_knockout_summary,
                     plot_all_round1, plot_scoreline_heatmap)
# בתחילת data.py, אחרי כל ה-imports:
try:
    from players_algorithm.team_strength import get_player_strength
    SQUAD_STRENGTH = get_player_strength()
    print("  [Squad] Using player-based SPI strength.")
except Exception as e:
    print(f"  [Squad] Falling back to manual SQUAD_STRENGTH ({e})")
    # SQUAD_STRENGTH נשאר כמו שהוא


# ── toggle what to generate ───────────────────────────────────────────────────
GENERATE_PIES      = True   # pie charts per game
GENERATE_HEATMAPS  = True   # scoreline heat-maps per game
CHART_DIR          = "charts"
RESULTS_CSV        = "results.csv"   # martj42/international_results dataset
DATA_MIN_DATE      = "2020-01-01"    # 6 years of real match data
VALIDATION_SPLIT   = "2024-01-01"    # train on 2020-2023, test on 2024-2026


def main():
    t0 = time.time()
    sep = "=" * 60

    print(sep)
    print("  FIFA WORLD CUP 2026 – ROUND 1 PREDICTOR")
    print(sep)

    # ── 1. Load real match data ───────────────────────────────────────────────
    print(f"\n[1] Loading real match data from '{RESULTS_CSV}' (since {DATA_MIN_DATE}) …")
    matches = load_real_data(RESULTS_CSV, min_date=DATA_MIN_DATE)
    print(f"    Loaded: {len(matches):,} matches, "
          f"{matches['home_team'].nunique()} unique teams, "
          f"{matches['date'].min().date()} – {matches['date'].max().date()}")

    # ── 2. Train / test split ─────────────────────────────────────────────────
    train_matches = matches[matches["date"] < VALIDATION_SPLIT].copy()
    test_matches  = matches[matches["date"] >= VALIDATION_SPLIT].copy()
    print(f"\n[2] Train/test split at {VALIDATION_SPLIT}:")
    print(f"    Train: {len(train_matches):,} matches  ({train_matches['date'].min().date()} – {train_matches['date'].max().date()})")
    print(f"    Test : {len(test_matches):,} matches   ({test_matches['date'].min().date()} – {test_matches['date'].max().date()})")

    # ── 3. Tune ELO K on training data only (last 12 months of train window) ─
    print("\n[3] Tuning ELO K on training data …")
    best_k = tune_elo_k(train_matches, ALL_TEAMS)
    print(f"    Optimal K = {best_k:.1f}  (default was 40)")

    # ── 4. Validate DC vs ML on all held-out test games → per-team weights ───
    print("\n[4] Validating models on all 2024–2026 test games …")
    dc_w, ml_w, team_weights, val_metrics = validate_ensemble_weights(
        train_matches, test_matches, ALL_TEAMS)
    print(f"\n    ── Global results ({val_metrics['n_test']} test matches) ─────")
    print(f"    DC model  :  accuracy={val_metrics['dc_accuracy']:.1%}  log-loss={val_metrics['dc_logloss']:.4f}  → global weight {val_metrics['dc_weight']:.1%}")
    print(f"    ML model  :  accuracy={val_metrics['ml_accuracy']:.1%}  log-loss={val_metrics['ml_logloss']:.4f}  → global weight {val_metrics['ml_weight']:.1%}")
    print(f"    Global weighting: inverse log-loss  |  Per-team: x/(x+y) accuracy")
    print(f"\n    ── Per-team DC weights (all WC2026 teams) ──────────────")
    for t, v in sorted(team_weights.items(), key=lambda x: x[1]["dc"], reverse=True):
        print(f"    {t:<28}  DC={v['dc']:.0%}  ML={v['ml']:.0%}  ({v['source']})")

    # ── 5. Fit final ELO on ALL data with tuned K ─────────────────────────────
    print("\n[5] Fitting final ELO on full 6-year history …")

    elo = ELOSystem(ALL_TEAMS, k_group=best_k, k_knockout=best_k * 1.25)
    elo.fit_from_history(matches)
    top5 = sorted(
        [(t, v) for t, v in elo.ratings.items() if t in ALL_TEAMS],
        key=lambda x: x[1], reverse=True
    )[:5]
    print("    Top-5 ELO: " + ", ".join(f"{t} {v:.0f}" for t, v in top5))

    # ── 6. Train final DC + ML on full 6-year data ────────────────────────────
    print("\n[6] Fitting final Dixon-Coles model on full data …")
    dc = DixonColesModel()
    dc.fit(matches)

    print("\n[7] Building final ML features and training XGBoost …")
    df_feat = build_ml_features(matches, elo)
    ml = MLModel()
    ml.train(df_feat)

    # ── 7. Build ensemble with data-driven per-team weights ───────────────────
    dc_w, ml_w = 1.00, 0.00   # override: 100% DC, ML over-credits underdogs
    print(f"\n[8] Building ensemble  (global DC {dc_w:.0%} / ML {ml_w:.0%}, per-team from 2024–2026 test) …")
    ensemble = EnsembleModel(dc, ml, dc_weight=dc_w, team_weights={})

    # ── 8b. Feed actual in-tournament results into ELO ────────────────────────
    # Applied after training (not before) so historical ML/DC features aren't
    # contaminated by post-tournament ratings.
    print("\n[8b] Updating ELO with actual WC2026 results played so far (R1+R2+R3) …")
    apply_actual_results(elo)

    # ── 9. Predict all 72 group-stage games ──────────────────────────────────
    print("\n[9] Predicting all 72 group-stage fixtures (3 matchdays × 24 games) …")
    results = predict_group_stage(ensemble, elo, matches)

    # ── 10. Print full table ──────────────────────────────────────────────────
    print_group_stage_summary(results)

    # ── 11. Sport5 strategy ───────────────────────────────────────────────────
    if any(r["sport5_odds"] is not None for r in results):
        print_sport5_strategy(results)
    else:
        print("\n[Sport5] No odds entered in data.py::SPORT5_ODDS yet.")

    # ── 12. Round of 32 knockout predictions ─────────────────────────────────
    print("\n[12] Predicting Round of 32 (16 knockout fixtures) …")
    ko_results = predict_knockout_stage(ensemble, elo, matches)
    print_knockout_summary(ko_results)
    from data import SPORT5_EXACT_BONUS_KNOCKOUT
    print_sport5_strategy(ko_results, exact_bonus=SPORT5_EXACT_BONUS_KNOCKOUT,
                          title="SPORT5 KNOCKOUT STRATEGY – GAMES WORTH TAKING A RISK ON")

    # ── 14. Pie charts (round 1 only) ─────────────────────────────────────────
    if GENERATE_PIES:
        r1 = [r for r in results if r["matchday"] == 1]
        plot_all_round1(r1, save_dir=CHART_DIR)

    # ── 15. Heat-maps (round 1 only) ──────────────────────────────────────────
    if GENERATE_HEATMAPS:
        print(f"\n[Heatmaps] Saving scoreline heat-maps for MD1 to '{CHART_DIR}/' …")
        for r in [r for r in results if r["matchday"] == 1]:
            plot_scoreline_heatmap(r, max_goals=5, save_dir=CHART_DIR)
        print("[Heatmaps] Done.")

    elapsed = time.time() - t0
    print(f"\n✅  Done in {elapsed:.1f}s")


def apply_actual_results(elo: "ELOSystem") -> None:
    """
    Feeds real WC2026 group-stage results into the live ELO ratings, so that
    predictions for matchdays not yet played (e.g. matchday 3) reflect actual
    in-tournament form rather than pre-tournament strength alone.

    Extend this list with each round's real scores as the tournament progresses.
    """
    actual_results = [
        # ── Round 1 (matchday 1) ────────────────────────────────────────────
        ("Mexico",                    "South Africa",           2, 0),
        ("South Korea",               "Czechia",                2, 1),
        ("Canada",                    "Bosnia and Herzegovina", 1, 1),
        ("USA",                       "Paraguay",               4, 1),
        ("Qatar",                     "Switzerland",            1, 1),
        ("Brazil",                    "Morocco",                1, 1),
        ("Haiti",                     "Scotland",               0, 1),
        ("Australia",                 "Turkey",                 2, 0),
        ("Germany",                   "Curaçao",                7, 1),
        ("Ivory Coast",               "Ecuador",                1, 0),
        ("Netherlands",               "Japan",                  2, 2),
        ("Sweden",                    "Tunisia",                5, 1),
        ("Belgium",                   "Egypt",                  1, 1),
        ("Iran",                      "New Zealand",            2, 2),
        ("Spain",                     "Cape Verde",             0, 0),
        ("Saudi Arabia",              "Uruguay",                1, 1),
        ("France",                    "Senegal",                3, 1),
        ("Iraq",                      "Norway",                 1, 4),
        ("Argentina",                 "Algeria",                3, 0),
        ("Austria",                   "Jordan",                 3, 1),
        ("Portugal",                  "DR Congo",               1, 1),
        ("Uzbekistan",                "Colombia",               1, 3),
        ("England",                   "Croatia",                4, 2),
        ("Ghana",                     "Panama",                 1, 0),
        # ── Round 2 (matchday 2) ────────────────────────────────────────────
        ("Czechia",                   "South Africa",           1, 1),
        ("Mexico",                    "South Korea",            1, 0),
        ("Switzerland",               "Bosnia and Herzegovina", 4, 1),
        ("Canada",                    "Qatar",                  6, 0),
        ("Scotland",                  "Morocco",                0, 1),
        ("Brazil",                    "Haiti",                  3, 0),
        ("USA",                       "Australia",              2, 0),
        ("Turkey",                    "Paraguay",                0, 1),
        ("Germany",                   "Ivory Coast",            2, 1),
        ("Ecuador",                   "Curaçao",                0, 0),
        ("Netherlands",               "Sweden",                 5, 1),
        ("Tunisia",                   "Japan",                  0, 4),
        ("Belgium",                   "Iran",                   0, 0),
        ("New Zealand",               "Egypt",                  1, 3),
        ("Spain",                     "Saudi Arabia",           4, 0),
        ("Uruguay",                   "Cape Verde",             2, 2),
        ("France",                    "Iraq",                   3, 0),
        ("Norway",                    "Senegal",                3, 2),
        ("Argentina",                 "Austria",                2, 0),
        ("Jordan",                    "Algeria",                1, 2),
        ("Portugal",                  "Uzbekistan",             5, 0),
        ("Colombia",                  "DR Congo",               1, 0),
        ("England",                   "Ghana",                  0, 0),
        ("Panama",                    "Croatia",                0, 1),
        # ── Round 3 (matchday 3) ────────────────────────────────────────────
        ("Mexico",                    "Czechia",                3, 0),
        ("South Korea",               "South Africa",           0, 1),
        ("Switzerland",               "Canada",                 2, 1),
        ("Bosnia and Herzegovina",    "Qatar",                  3, 1),
        ("Brazil",                    "Scotland",               3, 0),
        ("Morocco",                   "Haiti",                  4, 2),
        ("Turkey",                    "USA",                    3, 2),
        ("Paraguay",                  "Australia",              0, 0),
        ("Ecuador",                   "Germany",                2, 1),
        ("Ivory Coast",               "Curaçao",                2, 0),
        ("Netherlands",               "Tunisia",                3, 1),
        ("Japan",                     "Sweden",                 1, 1),
        ("Belgium",                   "New Zealand",            5, 1),
        ("Egypt",                     "Iran",                   1, 1),
        ("Spain",                     "Uruguay",                1, 0),
        ("Cape Verde",                "Saudi Arabia",           0, 0),
        ("France",                    "Norway",                 4, 1),
        ("Senegal",                   "Iraq",                   5, 0),
        ("Argentina",                 "Jordan",                 3, 1),
        ("Austria",                   "Algeria",                3, 3),
        ("Colombia",                  "Portugal",               0, 0),
        ("DR Congo",                  "Uzbekistan",             3, 1),
        ("England",                   "Panama",                 2, 0),
        ("Croatia",                   "Ghana",                  2, 1),
    ]

    for t1, t2, g1, g2 in actual_results:
        elo.update(t1, t2, g1, g2)

    top5 = sorted(
        [(t, v) for t, v in elo.ratings.items() if t in ALL_TEAMS],
        key=lambda x: x[1], reverse=True
    )[:5]
    print(f"    Applied {len(actual_results)} actual results (R1+R2+R3).")
    print("    Top-5 ELO after R1+R2+R3: " + ", ".join(f"{t} {v:.0f}" for t, v in top5))


if __name__ == "__main__":
    main()
