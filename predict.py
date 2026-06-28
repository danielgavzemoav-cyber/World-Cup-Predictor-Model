"""
Round-1 predictions, sport5 expected-value optimiser, and pie-chart visualisation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # headless-safe; swap to "TkAgg" if you want interactive
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

from data import (HOST_NATIONS, ROUND1_FIXTURES, ALL_GROUP_FIXTURES,
                  SPORT5_ODDS, SPORT5_EXACT_BONUS_GROUP, SPORT5_EXACT_BONUS_KNOCKOUT,
                  KNOCKOUT_R32_FIXTURES)
from models import ELOSystem, EnsembleModel


# ══════════════════════════════════════════════════════════════════════════════
# 1.  IS-HOME HELPER
# ══════════════════════════════════════════════════════════════════════════════

def _get_odds(t1: str, t2: str) -> dict | None:
    """
    Look up sport5 odds for a fixture, handling reversed team order.
    If stored as (t2, t1) instead of (t1, t2), swaps H and A automatically.
    """
    odds = SPORT5_ODDS.get((t1, t2))
    if odds is not None:
        return odds
    rev = SPORT5_ODDS.get((t2, t1))
    if rev is not None:
        return {"H": rev["A"], "D": rev["D"], "A": rev["H"]}
    return None


def is_home_game(t1: str, t2: str) -> bool:
    """True only when a host nation plays at their own stadium."""
    return t1 in HOST_NATIONS and t2 not in HOST_NATIONS


# ══════════════════════════════════════════════════════════════════════════════
# 2.  SPORT5 EXPECTED-VALUE OPTIMISER
# ══════════════════════════════════════════════════════════════════════════════

def _outcome(g1: int, g2: int) -> str:
    if g1 > g2:  return "H"
    if g1 == g2: return "D"
    return "A"


def expected_value(g1: int, g2: int,
                   ens_matrix: np.ndarray,
                   ens_probs: dict[str, float],
                   odds: dict[str, float],
                   exact_bonus: float = SPORT5_EXACT_BONUS_GROUP) -> float:
    """
    EV of predicting score (g1, g2) given the sport5 odds.

    Scoring:
      correct result only  → odds[outcome]
      correct exact score  → odds[outcome] + exact_bonus

    EV = odds[outcome] × P(outcome) + exact_bonus × P(g1, g2)
    """
    out  = _outcome(g1, g2)
    p_exact = ens_matrix[g1, g2] if g1 < ens_matrix.shape[0] and g2 < ens_matrix.shape[1] else 0.0
    return odds[out] * ens_probs[out] + exact_bonus * p_exact


def optimal_prediction(ens_matrix: np.ndarray,
                       ens_probs: dict[str, float],
                       odds: dict[str, float] | None,
                       max_goals: int = 7,
                       exact_bonus: float = SPORT5_EXACT_BONUS_GROUP
                       ) -> tuple[tuple[int, int], float, str]:
    """
    Returns (best_score, expected_value, method).
    method = "ev_optimised" | "most_likely_score"
    """
    if odds is None:
        # No sport5 odds → recommend most likely scoreline, EV=None signals "unknown"
        best_g1, best_g2 = np.unravel_index(
            np.argmax(ens_matrix[:max_goals+1, :max_goals+1]), (max_goals+1, max_goals+1)
        )
        return (int(best_g1), int(best_g2)), None, "most_likely_score"

    best_score = (1, 0)
    best_ev    = -1.0
    for g1 in range(max_goals + 1):
        for g2 in range(max_goals + 1):
            ev = expected_value(g1, g2, ens_matrix, ens_probs, odds, exact_bonus)
            if ev > best_ev:
                best_ev    = ev
                best_score = (g1, g2)
    return best_score, best_ev, "ev_optimised"


# ══════════════════════════════════════════════════════════════════════════════
# 3.  PREDICT ALL ROUND-1 GAMES
# ══════════════════════════════════════════════════════════════════════════════

def predict_group_stage(ensemble: EnsembleModel,
                        elo: ELOSystem,
                        matches: pd.DataFrame,
                        as_of: pd.Timestamp | None = None,
                        ) -> list[dict]:
    """Predict all 72 group-stage games across the 3 matchdays."""
    if as_of is None:
        as_of = pd.Timestamp("2026-06-11")
    results = []
    for t1, t2, grp, md in ALL_GROUP_FIXTURES:
        home = is_home_game(t1, t2)
        pred = ensemble.predict_all(t1, t2, home, elo, matches, as_of)
        odds = _get_odds(t1, t2)
        best_score, ev, method = optimal_prediction(
            pred["ens_matrix"], pred["ens_probs"], odds
        )
        results.append({
            "group": grp, "matchday": md,
            "team1": t1, "team2": t2, "is_home": home,
            "ml_H": pred["ml_probs"]["H"], "ml_D": pred["ml_probs"]["D"], "ml_A": pred["ml_probs"]["A"],
            "dc_H": pred["dc_probs"]["H"], "dc_D": pred["dc_probs"]["D"], "dc_A": pred["dc_probs"]["A"],
            "ens_H": pred["ens_probs"]["H"], "ens_D": pred["ens_probs"]["D"], "ens_A": pred["ens_probs"]["A"],
            "xg1": pred["xg1"], "xg2": pred["xg2"],
            "rec_score": best_score, "rec_ev": round(ev, 3) if ev is not None else None,
            "rec_method": method, "sport5_odds": odds,
            "_matrix": pred["ens_matrix"],
        })
    return results


def predict_round1(ensemble: EnsembleModel,
                   elo: ELOSystem,
                   matches: pd.DataFrame,
                   as_of: pd.Timestamp | None = None,
                   ) -> list[dict]:
    """
    Returns a list of prediction dicts, one per round-1 fixture.
    Each dict contains probabilities from all three models plus
    the sport5-optimised recommendation.
    """
    if as_of is None:
        as_of = pd.Timestamp("2026-06-11")

    results = []
    for t1, t2, grp in ROUND1_FIXTURES:
        home = is_home_game(t1, t2)
        pred = ensemble.predict_all(t1, t2, home, elo, matches, as_of)

        odds = _get_odds(t1, t2)
        best_score, ev, method = optimal_prediction(
            pred["ens_matrix"], pred["ens_probs"], odds
        )

        results.append({
            "group":    grp,
            "team1":    t1,
            "team2":    t2,
            "is_home":  home,
            # ── outcome probabilities ─────────────────────────────────────────
            "ml_H":     pred["ml_probs"]["H"],
            "ml_D":     pred["ml_probs"]["D"],
            "ml_A":     pred["ml_probs"]["A"],
            "dc_H":     pred["dc_probs"]["H"],
            "dc_D":     pred["dc_probs"]["D"],
            "dc_A":     pred["dc_probs"]["A"],
            "ens_H":    pred["ens_probs"]["H"],
            "ens_D":    pred["ens_probs"]["D"],
            "ens_A":    pred["ens_probs"]["A"],
            # ── expected goals ───────────────────────────────────────────────
            "xg1":      pred["xg1"],
            "xg2":      pred["xg2"],
            # ── sport5 recommendation ────────────────────────────────────────
            "rec_score":  best_score,
            "rec_ev":     round(ev, 3) if ev is not None else None,
            "rec_method": method,
            "sport5_odds": odds,
            # ── scoreline matrix (keep for charts) ───────────────────────────
            "_matrix":  pred["ens_matrix"],
        })

    return results


# ══════════════════════════════════════════════════════════════════════════════
# 4.  PRINT SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════════════════

def print_group_stage_summary(results: list[dict]) -> None:
    hdr = (f"{'MD':>2}  {'Grp':>3}  {'Team 1':<25}  {'Team 2':<25}  "
           f"{'ML: W/D/L':>14}  {'DC: W/D/L':>14}  {'ENS: W/D/L':>15}  "
           f"{'xG':>7}  {'Rec.':>6}  {'EV':>5}")
    LINE = "=" * len(hdr)

    # Group by matchday
    from itertools import groupby
    results_sorted = sorted(results, key=lambda r: (r["matchday"], r["group"]))

    for md, md_games in groupby(results_sorted, key=lambda r: r["matchday"]):
        md_games = list(md_games)
        print("\n" + LINE)
        print(f"  MATCHDAY {md}  –  PREDICTED OUTCOMES  (W=team1 win, D=draw, L=team2 win)")
        print(LINE)
        print(hdr)
        print("-" * len(hdr))

        prev_grp = None
        for r in md_games:
            if r["group"] != prev_grp:
                if prev_grp is not None:
                    print()
                print(f"  Group {r['group']}")
                prev_grp = r["group"]

            ml  = f"{r['ml_H']:.0%}/{r['ml_D']:.0%}/{r['ml_A']:.0%}"
            dc  = f"{r['dc_H']:.0%}/{r['dc_D']:.0%}/{r['dc_A']:.0%}"
            ens = f"{r['ens_H']:.0%}/{r['ens_D']:.0%}/{r['ens_A']:.0%}"
            xg  = f"{r['xg1']:.1f}-{r['xg2']:.1f}"
            rec      = f"{r['rec_score'][0]}-{r['rec_score'][1]}"
            ev       = f"{r['rec_ev']:.2f}" if r["rec_ev"] is not None else "  N/A"
            ev_star  = "" if r["rec_ev"] is not None else "*"
            home_tag = " (H)" if r["is_home"] else ""
            print(f"  {md:>2}  {r['group']:>3}  "
                  f"{r['team1']+home_tag:<25}  {r['team2']:<25}  "
                  f"{ml:>14}  {dc:>14}  {ens:>15}  "
                  f"{xg:>7}  {rec:>6}  {ev:>6}{ev_star}")

        print(LINE)

    print("\n  (H) = host-nation home advantage")
    print("  Rec = EV-optimised sport5 score (or most probable if * = no odds entered)")
    print("  EV  = expected sport5 points for Rec prediction  (base_pts + 6 for exact)")


# backward-compat alias
def print_round1_summary(results):
    print_group_stage_summary(results)


# ══════════════════════════════════════════════════════════════════════════════
# 5.  SPORT5 STRATEGY ADVICE
# ══════════════════════════════════════════════════════════════════════════════

def print_sport5_strategy(results: list[dict],
                          exact_bonus: float = SPORT5_EXACT_BONUS_GROUP,
                          title: str = "SPORT5 STRATEGY – GAMES WORTH TAKING A RISK ON") -> None:
    """
    Highlights the games where going for the underdog prediction
    has a higher EV than the favourite.
    """
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)

    for r in results:
        if r["sport5_odds"] is None:
            continue
        odds  = r["sport5_odds"]
        mat   = r["_matrix"]
        probs = {"H": r["ens_H"], "D": r["ens_D"], "A": r["ens_A"]}
        n     = mat.shape[0]

        def best_ev_for_outcome(out):
            max_p_exact = max(
                (mat[g1, g2] for g1 in range(n) for g2 in range(n) if _outcome(g1, g2) == out),
                default=0.0
            )
            return odds[out] * probs[out] + exact_bonus * max_p_exact

        evs = {out: best_ev_for_outcome(out) for out in ("H", "D", "A")}

        fav_key  = max(probs, key=probs.__getitem__)
        best_out = max(evs, key=evs.__getitem__)   # outcome whose best score has highest full EV

        if best_out != fav_key:
            t1, t2 = r["team1"], r["team2"]
            label  = {"H": f"{t1} wins", "D": "Draw", "A": f"{t2} wins"}
            rec    = r["rec_score"]
            context = f"Group {r['group']}" if "group" in r else "Knockout"
            print(f"\n  {t1} vs {t2}  ({context})")
            print(f"    Most probable : {label[fav_key]}  "
                  f"({probs[fav_key]:.0%} prob,  full EV={evs[fav_key]:.2f})")
            print(f"    Best EV pick  : {label[best_out]}  "
                  f"({probs[best_out]:.0%} prob,  full EV={evs[best_out]:.2f})  ← RECOMMEND")
            print(f"    Rec. scoreline: {rec[0]}-{rec[1]}  (total EV={r['rec_ev']:.2f})")


# ══════════════════════════════════════════════════════════════════════════════
# 6.  KNOCKOUT STAGE PREDICTIONS
# ══════════════════════════════════════════════════════════════════════════════

def predict_knockout_stage(ensemble: EnsembleModel,
                           elo: ELOSystem,
                           matches: pd.DataFrame,
                           as_of: pd.Timestamp | None = None,
                           ) -> list[dict]:
    """Predict all R32 knockout-stage games using knockout exact-score bonus."""
    if as_of is None:
        as_of = pd.Timestamp("2026-06-28")
    results = []
    for t1, t2 in KNOCKOUT_R32_FIXTURES:
        home = is_home_game(t1, t2)
        pred = ensemble.predict_all(t1, t2, home, elo, matches, as_of)
        odds = _get_odds(t1, t2)
        best_score, ev, method = optimal_prediction(
            pred["ens_matrix"], pred["ens_probs"], odds,
            exact_bonus=SPORT5_EXACT_BONUS_KNOCKOUT,
        )
        results.append({
            "team1": t1, "team2": t2, "is_home": home,
            "ml_H": pred["ml_probs"]["H"], "ml_D": pred["ml_probs"]["D"], "ml_A": pred["ml_probs"]["A"],
            "dc_H": pred["dc_probs"]["H"], "dc_D": pred["dc_probs"]["D"], "dc_A": pred["dc_probs"]["A"],
            "ens_H": pred["ens_probs"]["H"], "ens_D": pred["ens_probs"]["D"], "ens_A": pred["ens_probs"]["A"],
            "xg1": pred["xg1"], "xg2": pred["xg2"],
            "rec_score": best_score, "rec_ev": round(ev, 3) if ev is not None else None,
            "rec_method": method, "sport5_odds": odds,
            "_matrix": pred["ens_matrix"],
        })
    return results


def print_knockout_summary(results: list[dict]) -> None:
    hdr = (f"  {'Team 1':<28}  {'Team 2':<28}  "
           f"{'ENS: W/D/L':>15}  {'xG':>7}  {'Rec.':>6}  {'EV':>6}  {'Odds H/D/A'}")
    LINE = "=" * len(hdr)
    print("\n" + LINE)
    print("  ROUND OF 32  –  PREDICTED OUTCOMES  (W=team1 win, D=draw, L=team2 win)")
    print(LINE)
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        ens = f"{r['ens_H']:.0%}/{r['ens_D']:.0%}/{r['ens_A']:.0%}"
        xg  = f"{r['xg1']:.1f}-{r['xg2']:.1f}"
        rec = f"{r['rec_score'][0]}-{r['rec_score'][1]}"
        ev  = f"{r['rec_ev']:.2f}" if r["rec_ev"] is not None else "  N/A"
        o   = r["sport5_odds"]
        odds_str = f"{o['H']}/{o['D']}/{o['A']}" if o else "—"
        home_tag = " (H)" if r["is_home"] else ""
        print(f"  {r['team1']+home_tag:<28}  {r['team2']:<28}  "
              f"{ens:>15}  {xg:>7}  {rec:>6}  {ev:>6}  {odds_str}")
    print(LINE)
    print("  Rec = EV-optimised scoreline (exact-score bonus = +6)")
    print("  EV  = expected sport5 points  (base_pts + 6 for exact)")


# ══════════════════════════════════════════════════════════════════════════════
# 7.  PIE-CHART VISUALISATION
# ══════════════════════════════════════════════════════════════════════════════

_COLORS = {
    "ML":  ["#4C72B0", "#9FC3E9", "#E57373"],
    "DC":  ["#55A868", "#A8D5B5", "#F4A261"],
    "Ens": ["#C44E52", "#F7C5C5", "#8172B2"],
}

_OUTCOME_LABELS = lambda t1, t2: [f"{t1}\nWins", "Draw", f"{t2}\nWins"]


def plot_match_pie(result: dict, save_dir: str | Path | None = None) -> plt.Figure:
    """
    Three-panel figure for one match:
      left   = ML probs pie
      centre = DC probs pie
      right  = Ensemble probs pie
    Plus a text box with xG and sport5 recommendation.
    """
    t1, t2 = result["team1"], result["team2"]
    labels  = _OUTCOME_LABELS(t1, t2)
    h_tag   = " (home)" if result["is_home"] else ""

    fig = plt.figure(figsize=(14, 5))
    fig.suptitle(f"Group {result['group']}:  {t1}{h_tag}  vs  {t2}",
                 fontsize=14, fontweight="bold", y=1.02)

    gs = gridspec.GridSpec(1, 4, figure=fig, width_ratios=[1, 1, 1, 0.55])

    for col_idx, (model, key_h, key_d, key_a) in enumerate([
        ("ML  (XGBoost)",         "ml_H",  "ml_D",  "ml_A"),
        ("DC  (Dixon-Coles)",     "dc_H",  "dc_D",  "dc_A"),
        ("Ensemble (ML + DC)",    "ens_H", "ens_D", "ens_A"),
    ]):
        probs  = [result[key_h], result[key_d], result[key_a]]
        colors = _COLORS[["ML", "DC", "Ens"][col_idx]]

        ax = fig.add_subplot(gs[col_idx])
        wedges, texts, autotexts = ax.pie(
            probs, labels=labels, colors=colors,
            autopct="%1.1f%%", startangle=90,
            pctdistance=0.78, labeldistance=1.12,
            wedgeprops={"edgecolor": "white", "linewidth": 1.5},
        )
        for at in autotexts:
            at.set_fontsize(9)
        ax.set_title(model, fontsize=10, pad=12)

    # ── info panel ────────────────────────────────────────────────────────────
    ax_info = fig.add_subplot(gs[3])
    ax_info.axis("off")

    rec = result["rec_score"]
    odds_str = ""
    if result["sport5_odds"]:
        o = result["sport5_odds"]
        odds_str = (f"\nSport5 odds:\n"
                    f"  {t1} win : {o['H']}\n"
                    f"  Draw    : {o['D']}\n"
                    f"  {t2} win : {o['A']}")

    ev_str = (f"\nExpected pts : {result['rec_ev']:.3f}"
              if result["sport5_odds"] else "")

    info = (
        f"xG  {t1}: {result['xg1']:.2f}\n"
        f"xG  {t2}: {result['xg2']:.2f}\n"
        f"\n── Sport5 pick ──\n"
        f"Prediction : {rec[0]}-{rec[1]}\n"
        f"({result['rec_method'].replace('_',' ')})"
        f"{ev_str}"
        f"{odds_str}"
    )
    ax_info.text(0.05, 0.95, info, transform=ax_info.transAxes,
                 fontsize=9, verticalalignment="top", fontfamily="monospace",
                 bbox={"boxstyle": "round", "facecolor": "#F0F0F0", "alpha": 0.8})

    plt.tight_layout()

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        fname = save_dir / f"Group{result['group']}_{t1.replace(' ','_')}_vs_{t2.replace(' ','_')}.png"
        fig.savefig(fname, dpi=120, bbox_inches="tight")
        print(f"  Saved → {fname}")

    return fig


def plot_all_round1(results: list[dict], save_dir: str | Path = "charts") -> None:
    """Generate and save a pie chart for every round-1 game."""
    print(f"\n[Charts] Saving {len(results)} pie charts to '{save_dir}/' …")
    for r in results:
        plot_match_pie(r, save_dir=save_dir)
    print("[Charts] Done.")


# ══════════════════════════════════════════════════════════════════════════════
# 7.  SCORELINE HEAT-MAP  (bonus visualisation)
# ══════════════════════════════════════════════════════════════════════════════

def plot_scoreline_heatmap(result: dict, max_goals: int = 5,
                           save_dir: str | Path | None = None) -> plt.Figure:
    t1, t2  = result["team1"], result["team2"]
    mat     = result["_matrix"][:max_goals+1, :max_goals+1]

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(mat * 100, cmap="YlOrRd", aspect="auto")
    plt.colorbar(im, ax=ax, label="Probability (%)")

    ax.set_xticks(range(max_goals + 1))
    ax.set_yticks(range(max_goals + 1))
    ax.set_xticklabels(range(max_goals + 1))
    ax.set_yticklabels(range(max_goals + 1))
    ax.set_xlabel(f"{t2} goals", fontsize=11)
    ax.set_ylabel(f"{t1} goals", fontsize=11)
    ax.set_title(f"Scoreline distribution\n{t1} vs {t2}", fontsize=12)

    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            ax.text(j, i, f"{mat[i,j]*100:.1f}", ha="center", va="center",
                    fontsize=8, color="black" if mat[i,j] < 0.08 else "white")

    # Mark recommended prediction
    rec = result["rec_score"]
    if rec[0] <= max_goals and rec[1] <= max_goals:
        ax.add_patch(plt.Rectangle((rec[1]-0.5, rec[0]-0.5), 1, 1,
                                   fill=False, edgecolor="blue", linewidth=2.5,
                                   label=f"Rec: {rec[0]}-{rec[1]}"))
        ax.legend(fontsize=9)

    plt.tight_layout()
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        fname = save_dir / f"heatmap_{t1.replace(' ','_')}_vs_{t2.replace(' ','_')}.png"
        fig.savefig(fname, dpi=120, bbox_inches="tight")
    return fig
