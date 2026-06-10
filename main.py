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
                    generate_training_data, build_ml_features)
from predict import (predict_group_stage, predict_round1,
                     print_group_stage_summary, print_sport5_strategy,
                     plot_all_round1, plot_scoreline_heatmap)


# ── toggle what to generate ───────────────────────────────────────────────────
GENERATE_PIES      = True   # pie charts per game
GENERATE_HEATMAPS  = True   # scoreline heat-maps per game
CHART_DIR          = "charts"
N_TRAINING_SAMPLES = 3000   # increase for more stable DC/ML training


def main():
    t0 = time.time()
    sep = "=" * 60

    print(sep)
    print("  FIFA WORLD CUP 2026 – ROUND 1 PREDICTOR")
    print(sep)

    # ── 1. ELO initialisation ─────────────────────────────────────────────────
    print("\n[1] Initialising ELO ratings from FIFA rankings …")
    elo = ELOSystem(ALL_TEAMS)
    top5 = sorted(elo.ratings.items(), key=lambda x: x[1], reverse=True)[:5]
    print("    Top-5 ELO: " + ", ".join(f"{t} {v:.0f}" for t, v in top5))

    # ── 2. Training data ───────────────────────────────────────────────────────
    print(f"\n[2] Generating {N_TRAINING_SAMPLES:,} synthetic training matches …")
    print("    (replace generate_training_data() with pd.read_csv() for real data)")
    matches = generate_training_data(ALL_TEAMS, n=N_TRAINING_SAMPLES)
    print(f"    Generated: {len(matches):,} matches, "
          f"{matches['home_team'].nunique()} unique teams")

    # ── 3. Train Dixon-Coles model ────────────────────────────────────────────
    print("\n[3] Fitting Dixon-Coles model …")
    dc = DixonColesModel()
    dc.fit(matches)

    # ── 4. Train ML model ─────────────────────────────────────────────────────
    print("\n[4] Building ML features and training XGBoost …")
    df_feat = build_ml_features(matches, elo)
    ml = MLModel()
    ml.train(df_feat)

    # ── 5. Build ensemble ─────────────────────────────────────────────────────
    print("\n[5] Building ensemble (DC 55% + ML 45%) …")
    ensemble = EnsembleModel(dc, ml, dc_weight=0.55)

    # ── 6. Predict all 72 group-stage games ──────────────────────────────────
    print("\n[6] Predicting all 72 group-stage fixtures (3 matchdays × 24 games) …")
    results = predict_group_stage(ensemble, elo, matches)

    # ── 7. Print full table ───────────────────────────────────────────────────
    print_group_stage_summary(results)

    # ── 8. Sport5 strategy ────────────────────────────────────────────────────
    if any(r["sport5_odds"] is not None for r in results):
        print_sport5_strategy(results)
    else:
        print("\n[Sport5] No odds entered in data.py::SPORT5_ODDS yet.")
        print("  → Fill in the odds you see in the app to get EV-optimised picks.")
        print("  → Without odds, 'Rec.' column shows the most-probable scoreline.")

    # ── 9. Pie charts (round 1 only to avoid generating 72 files) ────────────
    if GENERATE_PIES:
        r1 = [r for r in results if r["matchday"] == 1]
        plot_all_round1(r1, save_dir=CHART_DIR)

    # ── 10. Heat-maps (round 1 only) ─────────────────────────────────────────
    if GENERATE_HEATMAPS:
        print(f"\n[Heatmaps] Saving scoreline heat-maps for MD1 to '{CHART_DIR}/' …")
        for r in [r for r in results if r["matchday"] == 1]:
            plot_scoreline_heatmap(r, max_goals=5, save_dir=CHART_DIR)
        print("[Heatmaps] Done.")

    # ── 11. Dynamic ELO demo ──────────────────────────────────────────────────
    _demo_dynamic_elo(ensemble, elo, matches)

    elapsed = time.time() - t0
    print(f"\n✅  Done in {elapsed:.1f}s")


def _demo_dynamic_elo(ensemble, elo, matches):
    """
    Shows how ELO updates after round-1 results feed into round-2 predictions.
    Uses hypothetical results – replace with real scores as the tournament progresses.
    """
    print("\n" + "=" * 60)
    print("  DYNAMIC ELO DEMO – Round 2 (after hypothetical R1 results)")
    print("=" * 60)

    # Example: enter actual Round-1 scores here after the games are played
    hypothetical_r1_results = [
        ("Mexico",      "South Africa",  2, 0),
        ("South Korea", "Czechia",       1, 1),
        ("Brazil",      "Morocco",       2, 1),
        ("USA",         "Paraguay",      1, 0),
        ("Germany",     "Curaçao",       4, 0),
        ("Argentina",   "Algeria",       3, 0),
        ("France",      "Senegal",       2, 0),
        ("England",     "Croatia",       1, 0),
    ]

    snap = elo.snapshot()   # save pre-tournament ELO
    print("\n  Updating ELO with R1 results …")
    for t1, t2, g1, g2 in hypothetical_r1_results:
        old_elo_t1 = elo.ratings.get(t1, 1700)
        elo.update(t1, t2, g1, g2)
        new_elo_t1 = elo.ratings.get(t1, 1700)
        print(f"    {t1} vs {t2}  {g1}-{g2}  "
              f"→  {t1} ELO {old_elo_t1:.0f} → {new_elo_t1:.0f}")

    print("\n  Sample Round-2 predictions (using updated ELO):")
    sample_r2 = [
        ("Mexico",    "South Korea", "A"),
        ("Brazil",    "Haiti",       "C"),
        ("Argentina", "Austria",     "J"),
    ]
    for t1, t2, grp in sample_r2:
        from predict import is_home_game
        home = is_home_game(t1, t2)
        pred = ensemble.predict_all(t1, t2, home, elo, matches)
        ep   = pred["ens_probs"]
        print(f"    Group {grp}: {t1} vs {t2}  →  "
              f"W:{ep['H']:.0%}  D:{ep['D']:.0%}  L:{ep['A']:.0%}  "
              f"xG: {pred['xg1']:.1f}-{pred['xg2']:.1f}")

    elo.restore(snap)   # reset so further calls use pre-tournament ELO


if __name__ == "__main__":
    main()
