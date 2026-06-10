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

ALL_TEAMS = [t for grp in WC2026_GROUPS.values() for t in grp]

# (team1, team2, group) - team1 is the "home" side for venue purposes
ROUND1_FIXTURES = [
    ("Mexico",                 "South Africa",          "A"),
    ("South Korea",            "Czechia",               "A"),
    ("Canada",                 "Bosnia and Herzegovina","B"),
    ("Qatar",                  "Switzerland",           "B"),
    ("Brazil",                 "Morocco",               "C"),
    ("Haiti",                  "Scotland",              "C"),
    ("USA",                    "Paraguay",              "D"),
    ("Australia",              "Turkey",                "D"),
    ("Germany",                "Curaçao",               "E"),
    ("Ivory Coast",            "Ecuador",               "E"),
    ("Netherlands",            "Japan",                 "F"),
    ("Sweden",                 "Tunisia",               "F"),
    ("Belgium",                "Egypt",                 "G"),
    ("Iran",                   "New Zealand",           "G"),
    ("Spain",                  "Cape Verde",            "H"),
    ("Saudi Arabia",           "Uruguay",               "H"),
    ("France",                 "Senegal",               "I"),
    ("Iraq",                   "Norway",                "I"),
    ("Argentina",              "Algeria",               "J"),
    ("Austria",                "Jordan",                "J"),
    ("Portugal",               "DR Congo",              "K"),
    ("Uzbekistan",             "Colombia",              "K"),
    ("England",                "Croatia",               "L"),
    ("Ghana",                  "Panama",                "L"),
]

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
    rank = FIFA_RANKINGS.get(team, 60)
    return max(1350.0, 2200.0 - rank * 10.0)


# ── sport5 "5 חבר'ה" odds ────────────────────────────────────────────────────
# Fill these in from what you see inside the app before each round.
# Format: {(team1, team2): {"H": pts, "D": pts, "A": pts}}
# "H" = team1 wins,  "D" = draw,  "A" = team2 wins.
# Bonus for exact score (group stage) = +4 pts on top of base odds.
# Leave as None to fall back to highest-probability recommendation.

SPORT5_ODDS: dict = {
    # Example (replace with real app values):
    # ("Mexico", "South Africa"): {"H": 1.9, "D": 4.2, "A": 7.5},
}

SPORT5_EXACT_BONUS_GROUP    = 4   # extra pts for exact scoreline in group stage
SPORT5_EXACT_BONUS_KNOCKOUT = 6   # extra pts in knockout rounds
