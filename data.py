"""
Static data: WC2026 groups, fixtures, FIFA rankings, squad strength, ELO init.
"""

WC2026_GROUPS = {
    "A": ["Mexico", "South Africa", "South Korea", "Czechia"],
    "B": ["Canada", "Bosnia and Herzegovina", "Qatar", "Switzerland"],
    "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "D": ["USA", "Paraguay", "Australia", "Turkey"],
    "E": ["Germany", "Curaçao", "Ivory Coast", "Ecuador"],
    "F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
    "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "H": ["Spain", "Cape Verde", "Saudi Arabia", "Uruguay"],
    "I": ["France", "Senegal", "Iraq", "Norway"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["Portugal", "DR Congo", "Uzbekistan", "Colombia"],
    "L": ["England", "Croatia", "Ghana", "Panama"],
}

# Only these three nations have home-ground advantage
HOST_NATIONS = {"USA", "Canada", "Mexico"}

# Maps team names used in results.csv → our internal names
TEAM_NAME_MAP: dict[str, str] = {
    "United States":  "USA",
    "Czech Republic": "Czechia",
}

ALL_TEAMS = [t for grp in WC2026_GROUPS.values() for t in grp]


def _generate_group_fixtures():
    """
    All 72 group-stage games (6 per group, 3 matchdays × 2 games).
    Standard round-robin for teams [T1, T2, T3, T4]:
      MD1: T1 vs T2,  T3 vs T4
      MD2: T1 vs T3,  T2 vs T4
      MD3: T1 vs T4,  T2 vs T3
    Host nations (Mexico, Canada, USA) sit at position 1 in their group,
    so they are always team1 (home side) in every game they play.
    Returns list of (team1, team2, group, matchday).
    """
    fixtures = []
    for grp, teams in WC2026_GROUPS.items():
        T1, T2, T3, T4 = teams
        fixtures += [
            (T1, T2, grp, 1), (T3, T4, grp, 1),
            (T1, T3, grp, 2), (T2, T4, grp, 2),
            (T1, T4, grp, 3), (T2, T3, grp, 3),
        ]
    return fixtures


ALL_GROUP_FIXTURES = _generate_group_fixtures()   # 72 games, (t1,t2,grp,md)

# Convenience views by matchday (strip matchday field, keep t1/t2/grp)
ROUND1_FIXTURES = [(t1, t2, g) for t1, t2, g, md in ALL_GROUP_FIXTURES if md == 1]
ROUND2_FIXTURES = [(t1, t2, g) for t1, t2, g, md in ALL_GROUP_FIXTURES if md == 2]
ROUND3_FIXTURES = [(t1, t2, g) for t1, t2, g, md in ALL_GROUP_FIXTURES if md == 3]

# FIFA Rankings – June 2026 official update
FIFA_RANKINGS = {
    "Argentina":               1,
    "Spain":                   2,
    "France":                  3,
    "England":                 4,
    "Portugal":                5,
    "Brazil":                  6,
    "Morocco":                 7,
    "Netherlands":             8,
    "Belgium":                 9,
    "Germany":                10,
    "Croatia":                11,
    "Colombia":               13,
    "Mexico":                 14,
    "Senegal":                15,
    "Uruguay":                16,
    "USA":                    17,
    "Japan":                  18,
    "Switzerland":            19,
    "Iran":                   20,
    "Turkey":                 22,
    "Ecuador":                23,
    "Austria":                24,
    "South Korea":            25,
    "Australia":              27,
    "Algeria":                28,
    "Egypt":                  29,
    "Canada":                 30,
    "Norway":                 31,
    "Ivory Coast":            33,
    "Panama":                 34,
    "Sweden":                 38,
    "Czechia":                39,
    "Paraguay":               40,
    "Scotland":               42,
    "DR Congo":               45,
    "Tunisia":                46,
    "Uzbekistan":             51,
    "Iraq":                   56,
    "Qatar":                  57,
    "South Africa":           60,
    "Saudi Arabia":           61,
    "Jordan":                 63,
    "Bosnia and Herzegovina": 64,
    "Cape Verde":             67,
    "Ghana":                  73,
    "Curaçao":                82,
    "Haiti":                  83,
    "New Zealand":            85,
}

# Squad strength 0–100 (expert estimate blending FIFA rank + squad market value)
SQUAD_STRENGTH = {
    "Argentina":               95,
    "Spain":                   93,
    "France":                  92,
    "England":                 90,
    "Brazil":                  91,
    "Portugal":                89,
    "Netherlands":             86,
    "Germany":                 87,
    "Belgium":                 84,
    "Morocco":                 82,
    "Croatia":                 80,
    "Colombia":                78,
    "Mexico":                  77,
    "Senegal":                 76,
    "Uruguay":                 75,
    "USA":                     74,
    "Japan":                   73,
    "Canada":                  72,
    "Switzerland":             72,
    "Austria":                 70,
    "Norway":                  71,
    "Turkey":                  71,
    "Ecuador":                 68,
    "Ivory Coast":             68,
    "South Korea":             69,
    "Australia":               67,
    "Sweden":                  67,
    "Algeria":                 66,
    "Czechia":                 66,
    "Iran":                    65,
    "Scotland":                65,
    "Egypt":                   64,
    "Paraguay":                62,
    "Ghana":                   62,
    "Saudi Arabia":            58,
    "Panama":                  58,
    "DR Congo":                55,
    "Qatar":                   55,
    "South Africa":            54,
    "Uzbekistan":              52,
    "Cape Verde":              52,
    "Iraq":                    50,
    "Bosnia and Herzegovina":  60,
    "Tunisia":                 60,
    "Jordan":                  48,
    "New Zealand":             48,
    "Curaçao":                 40,
    "Haiti":                   45,
}


def get_initial_elo(team: str) -> float:
    """Initialise ELO from FIFA ranking: top teams ~2190, weakest ~1350."""
    rank = FIFA_RANKINGS.get(team, 85)
    return max(670.0, 2200.0 - rank * 18.0)


# ── sport5 "5 חבר'ה" odds ────────────────────────────────────────────────────
# Fill these in from what you see inside the app before each round.
# Format: {(team1, team2): {"H": pts, "D": pts, "A": pts}}
# "H" = team1 wins,  "D" = draw,  "A" = team2 wins.
# Bonus for exact score (group stage) = +4 pts on top of base odds.
# Leave as None to fall back to highest-probability recommendation.

SPORT5_ODDS: dict = {
    # Format: (team1, team2): {"H": team1-wins pts, "D": draw pts, "A": team2-wins pts}
    # Stored exactly as shown in the sport5 app (team order may differ from fixtures).
    # The lookup in predict.py handles reversed pairs automatically.
    #
    # ── Group A ───────────────────────────────────────────────────────────────
    ("Mexico",                   "South Africa"):          {"H": 2.0, "D": 4.0, "A": 6.0 },
    ("South Korea",              "Czechia"):               {"H": 3.0, "D": 3.5, "A": 3.0 },
    # ── Group B ───────────────────────────────────────────────────────────────
    ("Canada",                   "Bosnia and Herzegovina"):{"H": 2.5, "D": 3.0, "A": 3.0 },
    ("Qatar",                    "Switzerland"):           {"H": 8.0, "D": 5.0, "A": 1.5 },
    # ── Group C ───────────────────────────────────────────────────────────────
    ("Brazil",                   "Morocco"):               {"H": 2.0, "D": 4.0, "A": 5.0 },
    ("Haiti",                    "Scotland"):              {"H": 8.0, "D": 5.0, "A": 1.5 },
    # ── Group D ───────────────────────────────────────────────────────────────
    ("USA",                      "Paraguay"):              {"H": 2.0, "D": 3.5, "A": 4.0 },
    ("Australia",                "Turkey"):                {"H": 4.5, "D": 4.0, "A": 2.0 },
    # ── Group E ───────────────────────────────────────────────────────────────
    ("Germany",                  "Curaçao"):               {"H": 1.5, "D": 10.0,"A": 15.0},
    ("Ivory Coast",              "Ecuador"):               {"H": 3.5, "D": 3.0, "A": 2.5 },
    # ── Group F ───────────────────────────────────────────────────────────────
    ("Netherlands",              "Japan"):                 {"H": 2.0, "D": 3.0, "A": 3.5 },
    ("Sweden",                   "Tunisia"):               {"H": 2.0, "D": 3.5, "A": 4.0 },
    # ── Group G ───────────────────────────────────────────────────────────────
    ("Belgium",                  "Egypt"):                 {"H": 2.0, "D": 4.0, "A": 5.0 },
    ("Iran",                     "New Zealand"):           {"H": 2.0, "D": 4.5, "A": 4.0 },
    # ── Group H ───────────────────────────────────────────────────────────────
    ("Spain",                    "Cape Verde"):            {"H": 1.5, "D": 8.0, "A": 12.0},
    ("Uruguay",                  "Saudi Arabia"):          {"H": 2.0, "D": 5.0, "A": 6.0 },
    # ── Group I ───────────────────────────────────────────────────────────────
    ("France",                   "Senegal"):               {"H": 1.5, "D": 6.0, "A": 8.0 },
    ("Iraq",                     "Norway"):                {"H": 8.0, "D": 6.0, "A": 1.5 },
    # ── Group J ───────────────────────────────────────────────────────────────
    ("Argentina",                "Algeria"):               {"H": 1.5, "D": 4.5, "A": 10.0},
    ("Austria",                  "Jordan"):                {"H": 1.5, "D": 5.5, "A": 8.0 },
    # ── Group K ───────────────────────────────────────────────────────────────
    ("Portugal",                 "DR Congo"):              {"H": 1.5, "D": 6.0, "A": 10.0},
    ("Colombia",                 "Uzbekistan"):            {"H": 1.5, "D": 5.5, "A": 8.0 },
    # ── Group L ───────────────────────────────────────────────────────────────
    ("England",                  "Croatia"):               {"H": 2.0, "D": 4.0, "A": 5.0 },
    ("Ghana",                    "Panama"):                {"H": 2.0, "D": 4.0, "A": 4.0 },
    # ── Round 2 ───────────────────────────────────────────────────────────────
    # ── Group A ───────────────────────────────────────────────────────────────
    ("Mexico",                   "South Korea"):           {"H": 2.0, "D": 4.0, "A": 3.5 },
    ("South Africa",             "Czechia"):               {"H": 4.0, "D": 3.5, "A": 2.0 },
    # ── Group B ───────────────────────────────────────────────────────────────
    ("Canada",                   "Qatar"):                 {"H": 2.0, "D": 4.5, "A": 7.0 },
    ("Bosnia and Herzegovina",   "Switzerland"):           {"H": 5.0, "D": 4.0, "A": 2.0 },
    # ── Group C ───────────────────────────────────────────────────────────────
    ("Brazil",                   "Haiti"):                 {"H": 1.5, "D": 12.0,"A": 15.0},
    ("Morocco",                  "Scotland"):              {"H": 2.0, "D": 3.0, "A": 4.0 },
    # ── Group D ───────────────────────────────────────────────────────────────
    ("USA",                      "Australia"):             {"H": 2.0, "D": 4.0, "A": 5.0 },
    ("Paraguay",                 "Turkey"):                {"H": 3.5, "D": 3.5, "A": 2.5 },
    # ── Group E ───────────────────────────────────────────────────────────────
    ("Germany",                  "Ivory Coast"):           {"H": 1.5, "D": 4.5, "A": 6.0 },
    ("Curaçao",                  "Ecuador"):               {"H": 10.0,"D": 6.0, "A": 1.5 },
    # ── Group F ───────────────────────────────────────────────────────────────
    ("Netherlands",              "Sweden"):                {"H": 2.0, "D": 4.0, "A": 5.0 },
    ("Japan",                    "Tunisia"):               {"H": 2.0, "D": 3.5, "A": 4.0 },
    # ── Group G ───────────────────────────────────────────────────────────────
    ("Belgium",                  "Iran"):                  {"H": 1.5, "D": 5.0, "A": 8.0 },
    ("Egypt",                    "New Zealand"):           {"H": 2.0, "D": 4.0, "A": 5.0 },
    # ── Group H ───────────────────────────────────────────────────────────────
    ("Spain",                    "Saudi Arabia"):          {"H": 1.5, "D": 8.0, "A": 12.0},
    ("Cape Verde",               "Uruguay"):               {"H": 6.0, "D": 4.0, "A": 1.5 },
    # ── Group I ───────────────────────────────────────────────────────────────
    ("France",                   "Iraq"):                  {"H": 1.5, "D": 9.0, "A": 12.0},
    ("Senegal",                  "Norway"):                {"H": 3.5, "D": 3.5, "A": 2.0 },
    # ── Group J ───────────────────────────────────────────────────────────────
    ("Argentina",                "Austria"):               {"H": 2.0, "D": 3.5, "A": 5.5 },
    ("Algeria",                  "Jordan"):                {"H": 2.0, "D": 4.0, "A": 6.0 },
    # ── Group K ───────────────────────────────────────────────────────────────
    ("Portugal",                 "Uzbekistan"):            {"H": 1.5, "D": 8.0, "A": 11.0},
    ("DR Congo",                 "Colombia"):              {"H": 7.0, "D": 4.0, "A": 1.5 },
    # ── Group L ───────────────────────────────────────────────────────────────
    ("England",                  "Ghana"):                 {"H": 1.5, "D": 6.0, "A": 8.0 },
    ("Croatia",                  "Panama"):                {"H": 1.5, "D": 4.0, "A": 7.0 },
}

SPORT5_EXACT_BONUS_GROUP    = 4   # extra pts for exact scoreline in group stage
SPORT5_EXACT_BONUS_KNOCKOUT = 6   # extra pts in knockout rounds
