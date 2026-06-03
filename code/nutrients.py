"""
NutriAI — Nutrients Module
===========================
RDA (Recommended Daily Allowance) lookup tables and nutrient gap analysis.

Provides:
    - get_rda(age, sex) → dict of daily nutrient targets
    - compute_daily_totals(meals_df) → dict of nutrient sums
    - analyze_gaps(daily_totals, rda) → list of (nutrient, actual, target, pct)
    - get_gap_score(food_row, remaining_budget) → how well a food fills gaps

RDA values sourced from NIH Dietary Reference Intakes (DRI) tables.
"""

from typing import Optional


# ---------------------------------------------------------------------------
# RDA TABLES — NIH Dietary Reference Intakes
# ---------------------------------------------------------------------------
# Organized by sex and age bracket.
# Values are daily targets. Units match our database columns.
# Source: https://ods.od.nih.gov/HealthInformation/nutrientrecommendations.aspx

RDA_TABLES = {
    "male": {
        # age_min, age_max: RDA dict
        (19, 30): {
            "calories": 2400,
            "protein_g": 56,
            "carbs_g": 130,
            "fat_g": 77,        # ~29% of 2400 kcal
            "fiber_g": 38,
            "iron_mg": 8,
            "calcium_mg": 1000,
            "b12_mcg": 2.4,
            "vitamin_d_mcg": 15,
            "zinc_mg": 11,
            "sodium_mg": 2300,  # Upper limit, not target
            "potassium_mg": 3400,
            "magnesium_mg": 400,
        },
        (31, 50): {
            "calories": 2200,
            "protein_g": 56,
            "carbs_g": 130,
            "fat_g": 73,
            "fiber_g": 38,
            "iron_mg": 8,
            "calcium_mg": 1000,
            "b12_mcg": 2.4,
            "vitamin_d_mcg": 15,
            "zinc_mg": 11,
            "sodium_mg": 2300,
            "potassium_mg": 3400,
            "magnesium_mg": 420,
        },
        (51, 70): {
            "calories": 2000,
            "protein_g": 56,
            "carbs_g": 130,
            "fat_g": 67,
            "fiber_g": 30,
            "iron_mg": 8,
            "calcium_mg": 1000,
            "b12_mcg": 2.4,
            "vitamin_d_mcg": 15,
            "zinc_mg": 11,
            "sodium_mg": 2300,
            "potassium_mg": 3400,
            "magnesium_mg": 420,
        },
        (71, 100): {
            "calories": 1800,
            "protein_g": 56,
            "carbs_g": 130,
            "fat_g": 60,
            "fiber_g": 30,
            "iron_mg": 8,
            "calcium_mg": 1200,
            "b12_mcg": 2.4,
            "vitamin_d_mcg": 20,
            "zinc_mg": 11,
            "sodium_mg": 2300,
            "potassium_mg": 3400,
            "magnesium_mg": 420,
        },
    },
    "female": {
        (19, 30): {
            "calories": 1800,
            "protein_g": 46,
            "carbs_g": 130,
            "fat_g": 60,
            "fiber_g": 25,
            "iron_mg": 18,      # Higher for premenopausal women
            "calcium_mg": 1000,
            "b12_mcg": 2.4,
            "vitamin_d_mcg": 15,
            "zinc_mg": 8,
            "sodium_mg": 2300,
            "potassium_mg": 2600,
            "magnesium_mg": 310,
        },
        (31, 50): {
            "calories": 1800,
            "protein_g": 46,
            "carbs_g": 130,
            "fat_g": 60,
            "fiber_g": 25,
            "iron_mg": 18,
            "calcium_mg": 1000,
            "b12_mcg": 2.4,
            "vitamin_d_mcg": 15,
            "zinc_mg": 8,
            "sodium_mg": 2300,
            "potassium_mg": 2600,
            "magnesium_mg": 320,
        },
        (51, 70): {
            "calories": 1600,
            "protein_g": 46,
            "carbs_g": 130,
            "fat_g": 53,
            "fiber_g": 21,
            "iron_mg": 8,       # Drops post-menopause
            "calcium_mg": 1200,
            "b12_mcg": 2.4,
            "vitamin_d_mcg": 15,
            "zinc_mg": 8,
            "sodium_mg": 2300,
            "potassium_mg": 2600,
            "magnesium_mg": 320,
        },
        (71, 100): {
            "calories": 1600,
            "protein_g": 46,
            "carbs_g": 130,
            "fat_g": 53,
            "fiber_g": 21,
            "iron_mg": 8,
            "calcium_mg": 1200,
            "b12_mcg": 2.4,
            "vitamin_d_mcg": 20,
            "zinc_mg": 8,
            "sodium_mg": 2300,
            "potassium_mg": 2600,
            "magnesium_mg": 320,
        },
    },
}

# Nutrients we analyze (excludes calories — handled separately via calorie_target)
TRACKED_NUTRIENTS = [
    "protein_g", "carbs_g", "fat_g", "fiber_g",
    "iron_mg", "calcium_mg", "b12_mcg", "vitamin_d_mcg", "zinc_mg",
    "sodium_mg", "potassium_mg", "magnesium_mg",
]

# Display-friendly names
NUTRIENT_DISPLAY = {
    "calories": "Calories (kcal)",
    "protein_g": "Protein (g)",
    "carbs_g": "Carbs (g)",
    "fat_g": "Fat (g)",
    "fiber_g": "Fiber (g)",
    "iron_mg": "Iron (mg)",
    "calcium_mg": "Calcium (mg)",
    "b12_mcg": "Vitamin B12 (mcg)",
    "vitamin_d_mcg": "Vitamin D (mcg)",
    "zinc_mg": "Zinc (mg)",
    "sodium_mg": "Sodium (mg)",
    "potassium_mg": "Potassium (mg)",
    "magnesium_mg": "Magnesium (mg)",
}


# ---------------------------------------------------------------------------
# RDA LOOKUP
# ---------------------------------------------------------------------------
def get_rda(age: int, sex: str, calorie_target: Optional[int] = None) -> dict:
    """
    Look up RDA values for a given age and sex.
    
    Args:
        age: Person's age in years
        sex: "male" or "female"
        calorie_target: If provided, overrides the default calorie RDA
    
    Returns:
        Dict of nutrient → daily target value
    """
    sex_lower = sex.lower().strip()
    if sex_lower not in RDA_TABLES:
        sex_lower = "female"  # Default fallback

    brackets = RDA_TABLES[sex_lower]
    rda = None

    for (age_min, age_max), values in brackets.items():
        if age_min <= age <= age_max:
            rda = values.copy()
            break

    if rda is None:
        # Age outside known brackets — use closest
        if age < 19:
            rda = list(brackets.values())[0].copy()  # youngest bracket
        else:
            rda = list(brackets.values())[-1].copy()  # oldest bracket

    # Override calorie target if specified
    if calorie_target is not None:
        rda["calories"] = calorie_target

    return rda


# ---------------------------------------------------------------------------
# NUTRIENT ANALYSIS
# ---------------------------------------------------------------------------
def compute_daily_totals(meals: list[dict]) -> dict:
    """
    Sum nutrient values across all meals for one day.
    
    Args:
        meals: List of food dicts, each with nutrient keys
    
    Returns:
        Dict of nutrient → total value for the day
    """
    totals = {}
    all_keys = ["calories"] + TRACKED_NUTRIENTS

    for key in all_keys:
        totals[key] = sum(
            float(meal.get(key, 0) or 0) for meal in meals
        )
        totals[key] = round(totals[key], 1)

    return totals


def analyze_gaps(daily_totals: dict, rda: dict) -> list[dict]:
    """
    Compare daily nutrient totals against RDA targets.
    
    Returns a list of dicts:
        [{"nutrient": "iron_mg", "actual": 6.2, "target": 18.0,
          "pct": 34.4, "status": "low"}, ...]
    
    Status: "low" (<80%), "ok" (80-120%), "high" (>120%)
    For sodium, "high" is the concern (reverse logic).
    """
    results = []
    all_keys = ["calories"] + TRACKED_NUTRIENTS

    for nutrient in all_keys:
        actual = daily_totals.get(nutrient, 0)
        target = rda.get(nutrient, 0)

        if target > 0:
            pct = (actual / target) * 100
        else:
            pct = 100.0

        # Sodium is special: high is bad, low is fine
        if nutrient == "sodium_mg":
            if pct > 100:
                status = "high"
            else:
                status = "ok"
        else:
            if pct < 80:
                status = "low"
            elif pct > 120:
                status = "high"
            else:
                status = "ok"

        results.append({
            "nutrient": nutrient,
            "display_name": NUTRIENT_DISPLAY.get(nutrient, nutrient),
            "actual": round(actual, 1),
            "target": round(target, 1),
            "pct": round(pct, 1),
            "status": status,
        })

    return results


def get_gap_score(food: dict, remaining_budget: dict, rda: dict) -> float:
    """
    Score how well a food fills the remaining nutrient gaps for the day.
    
    Higher score = food fills more gaps. Used by the meal planner to
    pick the best candidate food at each step.
    
    Scoring logic:
    - For each nutrient, compute what % of the remaining daily need
      this food would fill
    - Weight important nutrients higher (protein, fiber, iron, calcium)
    - Penalize if food would push sodium over budget
    - Return weighted average score (0-100)
    """
    weights = {
        "protein_g": 2.0,
        "carbs_g": 1.0,
        "fat_g": 0.8,
        "fiber_g": 1.5,
        "iron_mg": 1.5,
        "calcium_mg": 1.5,
        "b12_mcg": 1.2,
        "vitamin_d_mcg": 1.2,
        "zinc_mg": 1.0,
        "potassium_mg": 1.0,
        "magnesium_mg": 1.0,
    }

    score = 0.0
    total_weight = 0.0

    for nutrient, weight in weights.items():
        remaining = remaining_budget.get(nutrient, 0)
        food_value = float(food.get(nutrient, 0) or 0)
        target = rda.get(nutrient, 1)

        if remaining > 0 and target > 0:
            # How much of the remaining need does this food fill? (0 to 1)
            fill_ratio = min(food_value / remaining, 1.0)
            score += fill_ratio * weight
        total_weight += weight

    # Sodium penalty: if food has high sodium and budget is tight
    sodium_remaining = remaining_budget.get("sodium_mg", 2300)
    food_sodium = float(food.get("sodium_mg", 0) or 0)
    if sodium_remaining > 0 and food_sodium > sodium_remaining * 0.4:
        score *= 0.7  # 30% penalty for high-sodium foods

    if total_weight > 0:
        score = (score / total_weight) * 100

    return round(score, 2)


# ---------------------------------------------------------------------------
# STANDALONE TEST
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("  NutriAI — Nutrient Module Test")
    print("=" * 60)

    test_cases = [
        ("Priya", 28, "female", 1800),
        ("Ravi", 45, "male", 2200),
        ("Mei", 35, "female", 1600),
        ("James", 55, "male", 2000),
    ]

    for name, age, sex, cal_target in test_cases:
        rda = get_rda(age, sex, calorie_target=cal_target)
        print(f"\n{name} ({age}{sex[0].upper()}, {cal_target} kcal target):")
        print(f"   Protein: {rda['protein_g']}g | Iron: {rda['iron_mg']}mg | "
              f"Calcium: {rda['calcium_mg']}mg | B12: {rda['b12_mcg']}mcg")

    # Test gap analysis with a sample day
    sample_meals = [
        {"calories": 350, "protein_g": 15, "carbs_g": 45, "fat_g": 10,
         "fiber_g": 8, "iron_mg": 3, "calcium_mg": 100, "b12_mcg": 0,
         "vitamin_d_mcg": 0, "zinc_mg": 1.5, "sodium_mg": 200,
         "potassium_mg": 400, "magnesium_mg": 50},
        {"calories": 500, "protein_g": 25, "carbs_g": 55, "fat_g": 15,
         "fiber_g": 6, "iron_mg": 4, "calcium_mg": 200, "b12_mcg": 1.0,
         "vitamin_d_mcg": 2, "zinc_mg": 3, "sodium_mg": 400,
         "potassium_mg": 500, "magnesium_mg": 60},
        {"calories": 600, "protein_g": 30, "carbs_g": 50, "fat_g": 25,
         "fiber_g": 5, "iron_mg": 3, "calcium_mg": 150, "b12_mcg": 2.0,
         "vitamin_d_mcg": 5, "zinc_mg": 4, "sodium_mg": 500,
         "potassium_mg": 600, "magnesium_mg": 70},
    ]

    totals = compute_daily_totals(sample_meals)
    rda = get_rda(28, "female", calorie_target=1800)
    gaps = analyze_gaps(totals, rda)

    print(f"\n\nSample day analysis (Priya's RDA):")
    print(f"{'Nutrient':<22} {'Actual':>8} {'Target':>8} {'%':>7}  Status")
    print(f"{'─' * 55}")
    for g in gaps:
        status_icon = "✅" if g["status"] == "ok" else ("⚠️" if g["status"] == "low" else "🔴")
        print(f"{g['display_name']:<22} {g['actual']:>8.1f} {g['target']:>8.1f} {g['pct']:>6.1f}%  {status_icon}")

    print(f"\n{'=' * 60}")
    print("  Nutrient module working ✅")
    print(f"{'=' * 60}")
