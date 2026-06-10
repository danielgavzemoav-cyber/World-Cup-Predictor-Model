
"""
Three prediction models + dynamic ELO system.

Model 1 – ML   : XGBoost outcome classifier + Poisson goal regressors
Model 2 – DC   : Dixon-Coles bivariate Poisson with low-score correction
Model 3 – Ens  : Weighted ensemble of ML + DC outcome probs; DC scoreline
                 matrix rescaled to match ensemble outcome totals.

ELO is initialised from FIFA rankings and updated in-tournament after
every match (rounds 2 & 3 of the group stage and all knockout games).
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import optimize
from scipy.stats import poisson
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import classification_report, log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

from data import (SQUAD_STRENGTH, HOST_NATIONS, ALL_TEAMS,
                  get_initial_elo, TEAM_NAME_MAP)


# ══════════════════════════════════════════════════════════════════════════════
# 1.  DYNAMIC ELO SYSTEM
# ══════════════════════════════════════════════════════════════════════════════

class ELOSystem:
    def __init__(self, teams: list[str], k_group: float = 40.0, k_knockout: float = 50.0):
        self.K_GROUP    = k_group
        self.K_KNOCKOUT = k_knockout
        self.ratings: dict[str, float] = {t: get_initial_elo(t) for t in teams}

    # ── public helpers ────────────────────────────────────────────────────────
    def elo_diff(self, t1: str, t2: str) -> float:
        return self.ratings.get(t1, 1700.0) - self.ratings.get(t2, 1700.0)

    def win_prob(self, t1: str, t2: str) -> float:
        """P(t1 beats t2) ignoring draw (useful for knockout)."""
        return 1.0 / (1.0 + 10.0 ** (-self.elo_diff(t1, t2) / 400.0))

    def update(self, t1: str, t2: str, g1: int, g2: int,
               is_knockout: bool = False, k_mult: float = 1.0) -> None:
        """Update ratings after a completed match."""
        K   = (self.K_KNOCKOUT if is_knockout else self.K_GROUP) * k_mult
        r1  = self.ratings.get(t1, 1700.0)
        r2  = self.ratings.get(t2, 1700.0)
        exp = 1.0 / (1.0 + 10.0 ** ((r2 - r1) / 400.0))

        actual = 1.0 if g1 > g2 else (0.0 if g1 < g2 else 0.5)
        # Goal-difference multiplier (Elo variant used by World Football ELO)
        gd_mult = 1.0 + 0.5 * np.log1p(abs(g1 - g2))

        delta = K * gd_mult * (actual - exp)
        self.ratings[t1] = r1 + delta
        self.ratings[t2] = r2 - delta

    def fit_from_history(self, matches: pd.DataFrame,
                         start_elo: float = 1500.0) -> None:
        """Reset all ELOs to start_elo then replay match history chronologically.
        WC finals count 1.5×, friendlies count 0.5× to reflect match importance."""
        all_teams = set(matches["home_team"]) | set(matches["away_team"])
        for t in all_teams | set(self.ratings):
            self.ratings[t] = start_elo

        for _, row in matches.sort_values("date").iterrows():
            trn = str(row.get("tournament", ""))
            if "Friendly" in trn:
                k_mult = 0.5
            elif "World Cup" in trn and "qualif" not in trn.lower():
                k_mult = 1.5
            else:
                k_mult = 1.0
            self.update(row["home_team"], row["away_team"],
                        int(row["home_goals"]), int(row["away_goals"]),
                        k_mult=k_mult)

    def snapshot(self) -> dict[str, float]:
        return dict(self.ratings)

    def restore(self, snapshot: dict[str, float]) -> None:
        self.ratings = dict(snapshot)


# ══════════════════════════════════════════════════════════════════════════════
# 2.  SYNTHETIC TRAINING DATA
#     Replace with real Kaggle CSVs for production use.
# ══════════════════════════════════════════════════════════════════════════════

def generate_training_data(teams: list[str], n: int = 3000,
                           seed: int = 42) -> pd.DataFrame:
    """
    Generates realistic synthetic match data driven by squad-strength ratios
    and Poisson goal sampling.  Swap for real data when available.
    """
    np.random.seed(seed)
    home_teams = np.random.choice(teams, n)
    away_teams = np.array(
        [np.random.choice([t for t in teams if t != h]) for h in home_teams]
    )
    venues = np.random.choice(["home", "neutral"], n, p=[0.30, 0.70])

    home_goals, away_goals = [], []
    for ht, at, v in zip(home_teams, away_teams, venues):
        s1 = SQUAD_STRENGTH.get(ht, 65) / 100.0
        s2 = SQUAD_STRENGTH.get(at, 65) / 100.0
        boost = 0.30 if v == "home" else 0.0
        # Exponent 0.85 creates realistic blow-out scores for big mismatches
        lam1 = max(0.30, 1.5 * (s1 / s2) ** 0.85 + boost)
        lam2 = max(0.10, 1.5 * (s2 / s1) ** 0.85)  # floor 0.10 not 0.30
        home_goals.append(np.random.poisson(lam1))
        away_goals.append(np.random.poisson(lam2))

    return pd.DataFrame({
        "date":       pd.date_range("2020-01-01", "2025-12-31", periods=n),
        "home_team":  home_teams,
        "away_team":  away_teams,
        "home_goals": home_goals,
        "away_goals": away_goals,
        "venue":      venues,
        "tournament": np.random.choice(
            ["FIFA World Cup", "World Cup Qual", "Friendly", "Continental"],
            n, p=[0.05, 0.30, 0.30, 0.35]
        ),
    })


def load_real_data(path: str, min_date: str = "2021-01-01") -> pd.DataFrame:
    """
    Load real international match results from results.csv (martj42/international_results).
    Normalises team names, renames columns, and sets venue flag.
    """
    df = pd.read_csv(path, parse_dates=["date"])
    df = df[df["date"] >= pd.Timestamp(min_date)].copy()
    df["home_team"] = df["home_team"].replace(TEAM_NAME_MAP)
    df["away_team"] = df["away_team"].replace(TEAM_NAME_MAP)
    df = df.rename(columns={"home_score": "home_goals", "away_score": "away_goals"})
    df["venue"] = df["neutral"].apply(
        lambda x: "neutral" if str(x).upper() == "TRUE" else "home"
    )
    df = df.dropna(subset=["home_goals", "away_goals"])
    df["home_goals"] = df["home_goals"].astype(int)
    df["away_goals"] = df["away_goals"].astype(int)
    return (df[["date", "home_team", "away_team",
                "home_goals", "away_goals", "venue", "tournament"]]
            .reset_index(drop=True))


def tune_elo_k(matches: pd.DataFrame, all_teams: list[str],
               k_bounds: tuple = (10.0, 100.0)) -> float:
    """
    Find the K value that minimises Brier score (mean squared error of win
    probability vs actual outcome) on the most-recent 12 months of data.
    Beats vs weaker opponents already produce smaller deltas via the
    expected-score term; tuning K sets the overall learning-rate scale.
    """
    val_cutoff = matches["date"].max() - pd.DateOffset(months=12)
    val = matches[matches["date"] >= val_cutoff]

    def brier(k):
        elo = ELOSystem(all_teams, k_group=float(k), k_knockout=float(k) * 1.25)
        elo.fit_from_history(matches[matches["date"] < val_cutoff])
        errors = []
        for _, row in val.iterrows():
            r1 = elo.ratings.get(row["home_team"], 1500.0)
            r2 = elo.ratings.get(row["away_team"], 1500.0)
            p  = 1.0 / (1.0 + 10.0 ** ((r2 - r1) / 400.0))
            g1, g2 = int(row["home_goals"]), int(row["away_goals"])
            actual = 1.0 if g1 > g2 else (0.0 if g1 < g2 else 0.5)
            errors.append((p - actual) ** 2)
        return float(np.mean(errors)) if errors else 1.0

    result = optimize.minimize_scalar(brier, bounds=k_bounds, method="bounded")
    return float(result.x)


def validate_ensemble_weights(
        train_matches: pd.DataFrame,
        test_matches:  pd.DataFrame,
        all_teams:     list[str],
) -> tuple[float, float, dict[str, dict], dict]:
    """
    Train DC and ML on train_matches, evaluate on test_matches (WC2022 group stage).

    Global weight  : inverse log-loss (standard calibration metric).
    Per-team weight: accuracy-based — for each team, if DC was correct x% of
                     that team's games and ML y%, the team's DC weight = x/(x+y).
                     Teams absent from WC2022 fall back to the global weight.

    Returns (global_dc_w, global_ml_w, team_weights, metrics_dict).
    """
    from sklearn.metrics import log_loss as sk_log_loss

    # ── train a fresh ELO + DC + ML on pre-WC data ───────────────────────────
    print("  [Val] Fitting validation ELO …")
    elo_v = ELOSystem(all_teams)
    elo_v.fit_from_history(train_matches)

    print("  [Val] Fitting validation DC …")
    dc_v = DixonColesModel()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dc_v.fit(train_matches)

    print("  [Val] Building validation ML features …")
    df_feat_v = build_ml_features(train_matches, elo_v)
    ml_v = MLModel()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ml_v._train_silent(df_feat_v)

    # ── evaluate on test matches, record per-match correctness per team ───────
    LABELS = ["A", "D", "H"]
    dc_proba, ml_proba, y_true = [], [], []
    # team → list of (dc_correct, ml_correct) booleans
    from collections import defaultdict
    team_hits: dict[str, list[tuple[int, int]]] = defaultdict(list)

    for _, row in test_matches.iterrows():
        t1, t2  = row["home_team"], row["away_team"]
        is_home = row["venue"] == "home"
        as_of   = pd.Timestamp(row["date"])
        g1, g2  = int(row["home_goals"]), int(row["away_goals"])
        actual  = "H" if g1 > g2 else ("A" if g1 < g2 else "D")
        y_true.append(actual)

        dc_p = dc_v.predict_outcome(t1, t2, is_home, elo_v)
        dc_proba.append([dc_p["A"], dc_p["D"], dc_p["H"]])
        dc_pred = max(dc_p, key=dc_p.get)

        ml_pred_r = ml_v.predict_match(t1, t2, is_home, elo_v, train_matches, as_of)
        ml_proba.append([ml_pred_r["A"], ml_pred_r["D"], ml_pred_r["H"]])
        ml_pred = max(("H", "D", "A"), key=lambda k: ml_pred_r[k])

        dc_ok = int(dc_pred == actual)
        ml_ok = int(ml_pred == actual)
        team_hits[t1].append((dc_ok, ml_ok))
        team_hits[t2].append((dc_ok, ml_ok))

    # ── global weights via inverse log-loss ───────────────────────────────────
    dc_arr = np.clip(np.array(dc_proba), 1e-6, 1 - 1e-6)
    ml_arr = np.clip(np.array(ml_proba), 1e-6, 1 - 1e-6)

    dc_ll  = sk_log_loss(y_true, dc_arr, labels=LABELS)
    ml_ll  = sk_log_loss(y_true, ml_arr, labels=LABELS)

    dc_acc = float(np.mean([LABELS[np.argmax(p)] == a for p, a in zip(dc_proba, y_true)]))
    ml_acc = float(np.mean([LABELS[np.argmax(p)] == a for p, a in zip(ml_proba, y_true)]))

    global_dc_w = (1.0 / dc_ll) / (1.0 / dc_ll + 1.0 / ml_ll)
    global_ml_w = 1.0 - global_dc_w

    # ── per-team weights: accuracy-based  x/(x+y) ─────────────────────────────
    team_weights: dict[str, dict] = {}
    for team in all_teams:
        hits = team_hits.get(team)
        if not hits:  # team wasn't at WC2022 → global fallback
            team_weights[team] = {"dc": global_dc_w, "ml": global_ml_w,
                                  "source": "global_fallback"}
            continue
        x = sum(h[0] for h in hits) / len(hits)   # DC accuracy for this team
        y = sum(h[1] for h in hits) / len(hits)   # ML accuracy for this team
        if x + y == 0:  # both models always wrong → global fallback
            team_weights[team] = {"dc": global_dc_w, "ml": global_ml_w,
                                  "source": "global_fallback"}
        else:
            team_weights[team] = {"dc": x / (x + y), "ml": y / (x + y),
                                  "source": f"wc22 ({len(hits)}g dc={x:.0%} ml={y:.0%})"}

    metrics = {
        "n_test":      len(y_true),
        "dc_logloss":  round(dc_ll,  4),
        "ml_logloss":  round(ml_ll,  4),
        "dc_accuracy": round(dc_acc, 3),
        "ml_accuracy": round(ml_acc, 3),
        "dc_weight":   round(global_dc_w, 3),
        "ml_weight":   round(global_ml_w, 3),
    }
    return global_dc_w, global_ml_w, team_weights, metrics


# ══════════════════════════════════════════════════════════════════════════════
# 3.  DIXON-COLES MODEL
# ══════════════════════════════════════════════════════════════════════════════

class DixonColesModel:
    """
    Dixon & Coles (1997) bivariate Poisson with ρ-correction for low scores,
    augmented with optional ELO scaling for in-tournament predictions.
    """

    def __init__(self):
        self.params: np.ndarray | None = None
        self.teams:  list[str]         = []
        self.tidx:   dict[str, int]    = {}

    # ── static helper ─────────────────────────────────────────────────────────
    @staticmethod
    def _rho_corr(g1: int, g2: int, l1: float, l2: float, rho: float) -> float:
        if   g1 == 0 and g2 == 0: return max(1e-9, 1.0 - l1 * l2 * rho)
        elif g1 == 0 and g2 == 1: return max(1e-9, 1.0 + l1 * rho)
        elif g1 == 1 and g2 == 0: return max(1e-9, 1.0 + l2 * rho)
        elif g1 == 1 and g2 == 1: return max(1e-9, 1.0 - rho)
        return 1.0

    # ── fitting ───────────────────────────────────────────────────────────────
    def _nll(self, params: np.ndarray,
             hi: np.ndarray, ai: np.ndarray,
             venue_home: np.ndarray,
             g1: np.ndarray, g2: np.ndarray) -> float:
        """Fully vectorised negative log-likelihood."""
        n     = len(self.teams)
        atk   = params[:n]
        dfs   = params[n:2*n]
        h_adv = params[2*n]
        rho   = params[2*n + 1]

        bonus = h_adv * venue_home
        l1 = np.exp(atk[hi] + dfs[ai] + bonus)
        l2 = np.exp(atk[ai] + dfs[hi])

        corr = np.ones(len(g1))
        m00 = (g1 == 0) & (g2 == 0)
        m01 = (g1 == 0) & (g2 == 1)
        m10 = (g1 == 1) & (g2 == 0)
        m11 = (g1 == 1) & (g2 == 1)
        corr[m00] = np.maximum(1e-9, 1.0 - l1[m00] * l2[m00] * rho)
        corr[m01] = np.maximum(1e-9, 1.0 + l1[m01] * rho)
        corr[m10] = np.maximum(1e-9, 1.0 + l2[m10] * rho)
        corr[m11] = np.maximum(1e-9, 1.0 - rho)

        ll = (np.log(corr)
              + poisson.logpmf(g1, l1)
              + poisson.logpmf(g2, l2))
        return -float(ll.sum())

    def fit(self, matches: pd.DataFrame) -> None:
        self.teams = sorted(
            set(matches["home_team"].tolist() + matches["away_team"].tolist())
        )
        self.tidx = {t: i for i, t in enumerate(self.teams)}
        n = len(self.teams)

        # Pre-compute index arrays once (avoids per-call dict lookups)
        default = n // 2
        hi  = np.array([self.tidx.get(t, default) for t in matches["home_team"]])
        ai  = np.array([self.tidx.get(t, default) for t in matches["away_team"]])
        vh  = (matches["venue"] == "home").astype(float).values
        g1  = matches["home_goals"].values.astype(int)
        g2  = matches["away_goals"].values.astype(int)

        x0 = np.zeros(2 * n + 2)
        x0[2*n]     = 0.20
        x0[2*n + 1] = 0.05

        bounds = [(-3.0, 3.0)] * n + [(-3.0, 3.0)] * n + [(0.0, 1.0), (-0.5, 0.5)]

        res = optimize.minimize(
            self._nll, x0, args=(hi, ai, vh, g1, g2),
            method="L-BFGS-B", bounds=bounds,
            options={"maxiter": 500, "ftol": 1e-5},
        )
        self.params = res.x
        h_adv = res.x[2*n]
        rho   = res.x[2*n + 1]
        status = "converged" if res.success else "stopped early"
        print(f"  DC model: {status}  home_adv={h_adv:.3f}  rho={rho:.3f}")

    # ── prediction helpers ────────────────────────────────────────────────────
    def _lambdas(self, t1: str, t2: str, is_home: bool,
                 elo: ELOSystem | None = None, elo_weight: float = 0.30):
        n     = len(self.teams)
        atk   = self.params[:n]
        dfs   = self.params[n:2*n]
        h_adv = self.params[2*n]

        i1 = self.tidx.get(t1, n // 2)
        i2 = self.tidx.get(t2, n // 2)

        bonus = h_adv if is_home else 0.0
        l1_dc = np.exp(atk[i1] + dfs[i2] + bonus)
        l2_dc = np.exp(atk[i2] + dfs[i1])

        if elo is not None:
            e1  = elo.ratings.get(t1, 1700.0)
            e2  = elo.ratings.get(t2, 1700.0)
            avg = (e1 + e2) / 2.0
            # Dynamic weight: grows with ELO gap so blowout matchups get
            # much stronger scaling than close games
            dyn_w = min(0.85, elo_weight + abs(e1 - e2) / 1200.0)
            l1_dc *= (e1 / avg) ** dyn_w
            l2_dc *= (e2 / avg) ** dyn_w

        return l1_dc, l2_dc

    def scoreline_matrix(self, t1: str, t2: str, is_home: bool,
                         elo: ELOSystem | None = None,
                         max_goals: int = 8) -> tuple[np.ndarray, float, float]:
        """Returns (matrix, xg1, xg2). Matrix rows = t1 goals, cols = t2 goals."""
        rho = self.params[2*len(self.teams) + 1]
        l1, l2 = self._lambdas(t1, t2, is_home, elo)

        mat = np.zeros((max_goals + 1, max_goals + 1))
        for g1 in range(max_goals + 1):
            for g2 in range(max_goals + 1):
                mat[g1, g2] = (
                    self._rho_corr(g1, g2, l1, l2, rho)
                    * poisson.pmf(g1, l1)
                    * poisson.pmf(g2, l2)
                )
        mat = np.clip(mat, 1e-12, None)
        mat /= mat.sum()
        return mat, l1, l2

    def predict_outcome(self, t1: str, t2: str, is_home: bool,
                        elo: ELOSystem | None = None) -> dict[str, float]:
        mat, _, _ = self.scoreline_matrix(t1, t2, is_home, elo)
        n = mat.shape[0]
        p_h = float(np.sum([mat[g1, g2] for g1 in range(n)
                             for g2 in range(n) if g1 > g2]))
        p_a = float(np.sum([mat[g1, g2] for g1 in range(n)
                             for g2 in range(n) if g1 < g2]))
        p_d = float(np.trace(mat))
        total = p_h + p_a + p_d
        return {"H": p_h/total, "D": p_d/total, "A": p_a/total}


# ══════════════════════════════════════════════════════════════════════════════
# 4.  ML MODEL  (XGBoost + Poisson goal regressors)
# ══════════════════════════════════════════════════════════════════════════════

FEATURE_COLS = [
    "elo_diff",   "squad_diff", "venue_home",
    "h_form_pts", "h_form_gf",  "h_form_ga",  "h_form_gd",  "h_win_rate",
    "a_form_pts", "a_form_gf",  "a_form_ga",  "a_form_gd",  "a_win_rate",
]

_FORM_DEFAULT = {"form_pts": 0.9, "form_gf": 1.2, "form_ga": 1.2,
                 "form_gd": 0.0, "win_rate": 0.33}


def _team_form(matches: pd.DataFrame, team: str,
               as_of: pd.Timestamp, n: int = 10) -> dict:
    past = matches[matches["date"] < as_of]
    tm   = past[(past["home_team"] == team) | (past["away_team"] == team)
                ].sort_values("date").tail(n)
    if len(tm) == 0:
        return _FORM_DEFAULT

    pts, gf, ga = [], [], []
    for _, row in tm.iterrows():
        if row["home_team"] == team:
            gf.append(row["home_goals"]); ga.append(row["away_goals"])
            pts.append(3 if row["home_goals"] > row["away_goals"]
                       else (1 if row["home_goals"] == row["away_goals"] else 0))
        else:
            gf.append(row["away_goals"]); ga.append(row["home_goals"])
            pts.append(3 if row["away_goals"] > row["home_goals"]
                       else (1 if row["away_goals"] == row["home_goals"] else 0))

    wins = sum(1 for p in pts if p == 3)
    return {
        "form_pts": np.mean(pts), "form_gf": np.mean(gf),
        "form_ga": np.mean(ga),   "form_gd": np.mean(gf) - np.mean(ga),
        "win_rate": wins / len(pts),
    }


def _build_feature_row(t1: str, t2: str, is_home: bool,
                       elo: ELOSystem, matches: pd.DataFrame,
                       as_of: pd.Timestamp) -> dict:
    hf = _team_form(matches, t1, as_of)
    af = _team_form(matches, t2, as_of)
    return {
        "elo_diff":    elo.elo_diff(t1, t2),
        "squad_diff":  SQUAD_STRENGTH.get(t1, 65) - SQUAD_STRENGTH.get(t2, 65),
        "venue_home":  1 if is_home else 0,
        "h_form_pts":  hf["form_pts"], "h_form_gf": hf["form_gf"],
        "h_form_ga":   hf["form_ga"],  "h_form_gd": hf["form_gd"],
        "h_win_rate":  hf["win_rate"],
        "a_form_pts":  af["form_pts"], "a_form_gf": af["form_gf"],
        "a_form_ga":   af["form_ga"],  "a_form_gd": af["form_gd"],
        "a_win_rate":  af["win_rate"],
    }


def _precompute_form(matches: pd.DataFrame, n_games: int = 10) -> dict:
    """
    Returns {(team, date_idx): form_dict} – O(n log n) instead of O(n²).
    We compute each team's rolling form once, ordered by date.
    """
    matches = matches.sort_values("date").reset_index(drop=True)

    # Build per-team event list
    from collections import defaultdict
    events: dict[str, list] = defaultdict(list)
    for idx, row in matches.iterrows():
        h, a = row["home_team"], row["away_team"]
        g1, g2 = int(row["home_goals"]), int(row["away_goals"])
        events[h].append((idx, g1, g2, True))   # (match_idx, gf, ga, is_home_side)
        events[a].append((idx, g2, g1, False))

    form_cache: dict[tuple, dict] = {}
    for team, evts in events.items():
        evts.sort(key=lambda x: x[0])
        for pos, (match_idx, _, _, _) in enumerate(evts):
            window = evts[max(0, pos - n_games):pos]
            if not window:
                form_cache[(team, match_idx)] = _FORM_DEFAULT
                continue
            pts, gf, ga = [], [], []
            for _, wgf, wga, _ in window:
                gf.append(wgf); ga.append(wga)
                pts.append(3 if wgf > wga else (1 if wgf == wga else 0))
            wins = sum(1 for p in pts if p == 3)
            form_cache[(team, match_idx)] = {
                "form_pts": float(np.mean(pts)),
                "form_gf":  float(np.mean(gf)),
                "form_ga":  float(np.mean(ga)),
                "form_gd":  float(np.mean(gf)) - float(np.mean(ga)),
                "win_rate": wins / len(pts),
            }
    return form_cache


def _tournament_weight(tournament: str) -> float:
    t = tournament.lower()
    if "world cup" in t and "qualif" not in t:  return 3.0
    if "qualif" in t:                            return 2.0
    if any(x in t for x in ("euro", "copa america", "african cup", "asian cup",
                             "gold cup", "nations cup")):
        return 2.0
    if "nations league" in t:                    return 1.5
    if "friendly" in t:                          return 0.5
    return 1.0


def build_ml_features(matches: pd.DataFrame, elo: ELOSystem) -> pd.DataFrame:
    matches = matches.sort_values("date").reset_index(drop=True)
    form_cache = _precompute_form(matches)
    default_f  = _FORM_DEFAULT

    rows = []
    for idx, row in matches.iterrows():
        h, a = row["home_team"], row["away_team"]
        hf = form_cache.get((h, idx), default_f)
        af = form_cache.get((a, idx), default_f)

        rows.append({
            "elo_diff":    elo.elo_diff(h, a),
            "squad_diff":  SQUAD_STRENGTH.get(h, 65) - SQUAD_STRENGTH.get(a, 65),
            "venue_home":  1 if row["venue"] == "home" else 0,
            "h_form_pts":  hf["form_pts"], "h_form_gf": hf["form_gf"],
            "h_form_ga":   hf["form_ga"],  "h_form_gd": hf["form_gd"],
            "h_win_rate":  hf["win_rate"],
            "a_form_pts":  af["form_pts"], "a_form_gf": af["form_gf"],
            "a_form_ga":   af["form_ga"],  "a_form_gd": af["form_gd"],
            "a_win_rate":  af["win_rate"],
            "label":       ("H" if row["home_goals"] > row["away_goals"]
                            else ("A" if row["home_goals"] < row["away_goals"] else "D")),
            "home_goals":  row["home_goals"],
            "away_goals":  row["away_goals"],
            "weight":      _tournament_weight(str(row.get("tournament", ""))),
        })
    return pd.DataFrame(rows)


class MLModel:
    def __init__(self):
        self.clf      = None
        self.home_reg = None
        self.away_reg = None
        self.le       = LabelEncoder()

    def _fit_core(self, df: pd.DataFrame) -> None:
        """Shared fitting logic used by both train() and _train_silent()."""
        y = self.le.fit_transform(df["label"])
        X = df[FEATURE_COLS]; w = df["weight"]
        X_tr, X_te, y_tr, y_te, w_tr, _ = train_test_split(
            X, y, w, test_size=0.20, random_state=42, stratify=y
        )
        self.clf = xgb.XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="mlogloss", random_state=42, verbosity=0,
        )
        self.clf.fit(X_tr, y_tr, sample_weight=w_tr,
                     eval_set=[(X_te, y_te)], verbose=False)
        self.home_reg = PoissonRegressor(alpha=0.1, max_iter=300)
        self.away_reg = PoissonRegressor(alpha=0.1, max_iter=300)
        self.home_reg.fit(X_tr, df.loc[X_tr.index, "home_goals"])
        self.away_reg.fit(X_tr, df.loc[X_tr.index, "away_goals"])
        return X_te, y_te

    def train(self, df: pd.DataFrame) -> None:
        X_te, y_te = self._fit_core(df)
        preds = self.clf.predict(X_te)
        proba = self.clf.predict_proba(X_te)
        print("\n=== ML Model Performance ===")
        print(classification_report(y_te, preds,
              target_names=self.le.classes_, zero_division=0))
        print(f"  Log-loss : {log_loss(y_te, proba):.4f}")

    def _train_silent(self, df: pd.DataFrame) -> None:
        """Like train() but prints nothing — used during validation."""
        self._fit_core(df)

    def predict(self, feat_row: dict) -> dict:
        X     = pd.DataFrame([feat_row])[FEATURE_COLS]
        proba = self.clf.predict_proba(X)[0]
        probs = {c: proba[i] for i, c in enumerate(self.le.classes_)}
        xg1   = max(0.10, float(self.home_reg.predict(X)[0]))
        xg2   = max(0.10, float(self.away_reg.predict(X)[0]))
        return {"H": probs["H"], "D": probs["D"], "A": probs["A"],
                "xg1": xg1, "xg2": xg2}

    def predict_match(self, t1: str, t2: str, is_home: bool,
                      elo: ELOSystem, matches: pd.DataFrame,
                      as_of: pd.Timestamp | None = None) -> dict:
        if as_of is None:
            as_of = pd.Timestamp("2026-06-11")
        feat = _build_feature_row(t1, t2, is_home, elo, matches, as_of)
        return self.predict(feat)

    def scoreline_matrix(self, t1: str, t2: str, is_home: bool,
                         elo: ELOSystem, matches: pd.DataFrame,
                         max_goals: int = 8,
                         as_of: pd.Timestamp | None = None) -> np.ndarray:
        """Independent Poisson matrix from xG regressors."""
        if as_of is None:
            as_of = pd.Timestamp("2026-06-11")
        res  = self.predict_match(t1, t2, is_home, elo, matches, as_of)
        l1, l2 = res["xg1"], res["xg2"]
        mat = np.outer(
            [poisson.pmf(g, l1) for g in range(max_goals + 1)],
            [poisson.pmf(g, l2) for g in range(max_goals + 1)],
        )
        mat = np.clip(mat, 1e-12, None)
        mat /= mat.sum()
        return mat


# ══════════════════════════════════════════════════════════════════════════════
# 5.  ENSEMBLE MODEL
# ══════════════════════════════════════════════════════════════════════════════

class EnsembleModel:
    """
    Combines DC and ML predictions.
    Outcome probs: weighted average.
    Scoreline matrix: DC matrix rescaled so P(H/D/A) totals match ensemble.
    """

    def __init__(self, dc: DixonColesModel, ml: MLModel,
                 dc_weight: float = 0.55,
                 team_weights: dict | None = None):
        self.dc           = dc
        self.ml           = ml
        self.dc_weight    = dc_weight          # global fallback
        self.ml_weight    = 1.0 - dc_weight
        self.team_weights = team_weights or {}  # per-team override

    def _match_weights(self, t1: str, t2: str) -> tuple[float, float]:
        """Average the two teams' per-team DC/ML weights."""
        w1 = self.team_weights.get(t1, {"dc": self.dc_weight, "ml": self.ml_weight})
        w2 = self.team_weights.get(t2, {"dc": self.dc_weight, "ml": self.ml_weight})
        dc_w = (w1["dc"] + w2["dc"]) / 2.0
        return dc_w, 1.0 - dc_w

    def predict_outcome(self, t1: str, t2: str, is_home: bool,
                        elo: ELOSystem | None = None) -> dict[str, float]:
        dc_p = self.dc.predict_outcome(t1, t2, is_home, elo)
        return dc_p  # filled in predict_all below

    def predict_all(self, t1: str, t2: str, is_home: bool,
                    elo: ELOSystem, matches: pd.DataFrame,
                    as_of: pd.Timestamp | None = None,
                    max_goals: int = 8) -> dict:
        """
        Returns a dict with:
          ml_probs, dc_probs, ens_probs  – outcome dicts {H,D,A}
          ens_matrix                      – (9×9) scoreline probability matrix
          xg1, xg2                        – expected goals (ensemble)
          ml_xg1, ml_xg2                 – from ML regressors
          dc_xg1, dc_xg2                 – from DC lambdas
        """
        if as_of is None:
            as_of = pd.Timestamp("2026-06-11")

        ml_res = self.ml.predict_match(t1, t2, is_home, elo, matches, as_of)
        dc_p   = self.dc.predict_outcome(t1, t2, is_home, elo)
        dc_mat, dc_l1, dc_l2 = self.dc.scoreline_matrix(t1, t2, is_home, elo, max_goals)
        ml_mat = self.ml.scoreline_matrix(t1, t2, is_home, elo, matches, max_goals, as_of)

        ml_p = {k: ml_res[k] for k in ("H", "D", "A")}

        dc_w, ml_w = self._match_weights(t1, t2)

        # Ensemble outcome probs
        ens_p = {k: dc_w * dc_p[k] + ml_w * ml_p[k]
                 for k in ("H", "D", "A")}

        # Ensemble scoreline matrix = weighted average of DC + ML matrices
        raw_mat = dc_w * dc_mat + ml_w * ml_mat
        raw_mat = np.clip(raw_mat, 1e-12, None)

        # Rescale so that H/D/A totals exactly match ens_p
        n = raw_mat.shape[0]
        scaled = np.zeros_like(raw_mat)
        for out, target_p in ens_p.items():
            mask = np.zeros((n, n), dtype=bool)
            for g1 in range(n):
                for g2 in range(n):
                    if out == "H" and g1 > g2: mask[g1, g2] = True
                    elif out == "D" and g1 == g2: mask[g1, g2] = True
                    elif out == "A" and g1 < g2: mask[g1, g2] = True
            cat_sum = raw_mat[mask].sum()
            if cat_sum > 0:
                scaled[mask] = raw_mat[mask] * (target_p / cat_sum)

        scaled = np.clip(scaled, 1e-12, None)
        scaled /= scaled.sum()

        # Ensemble xG = weighted average of individual xGs
        ens_xg1 = dc_w * dc_l1 + ml_w * ml_res["xg1"]
        ens_xg2 = dc_w * dc_l2 + ml_w * ml_res["xg2"]

        return {
            "ml_probs": ml_p,
            "dc_probs": dc_p,
            "ens_probs": ens_p,
            "ens_matrix": scaled,
            "xg1": round(ens_xg1, 2),
            "xg2": round(ens_xg2, 2),
            "ml_xg1": round(ml_res["xg1"], 2),
            "ml_xg2": round(ml_res["xg2"], 2),
            "dc_xg1": round(dc_l1, 2),
            "dc_xg2": round(dc_l2, 2),
        }
