"""
NutriAI — Meal Planner Module
===============================
Generates a personalized 7-day, 3-meal-per-day plan using greedy
nutrient-gap filling with diversity constraints.

Algorithm:
    For each day:
        1. Start with the full daily calorie and nutrient budget (from RDA)
        2. For each meal (breakfast, lunch, dinner):
            a. Compute remaining nutrient gaps
            b. Score every food in the safe pool by how well it fills gaps
            c. Pick the top scorer that hasn't been used today
               and doesn't repeat the same food category too often
            d. Subtract its nutrients from the remaining budget
        3. Log the day's totals and gap analysis

Usage:
    from meal_planner import generate_plan
    plan = generate_plan(safe_foods_df, persona_config)
"""

import time
from typing import Optional

import numpy as np
import pandas as pd

from nutrients import (
    get_rda, compute_daily_totals, analyze_gaps,
    get_gap_score, TRACKED_NUTRIENTS,
)


# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
DAYS = 7
MEALS_PER_DAY = 3
MEAL_NAMES = ["Breakfast", "Lunch", "Dinner"]

# Calorie split across meals (breakfast lighter, dinner heavier)
MEAL_CALORIE_SPLIT = {
    "Breakfast": 0.25,
    "Lunch": 0.35,
    "Dinner": 0.40,
}

# Category diversity constraint: max times a food category can appear per day
MAX_CATEGORY_PER_DAY = 2

# How many top candidates to consider before picking (adds variety)
TOP_K_CANDIDATES = 15


# ---------------------------------------------------------------------------
# PLAN GENERATION
# ---------------------------------------------------------------------------
def generate_plan(
    safe_foods: pd.DataFrame,
    age: int = 30,
    sex: str = "female",
    calorie_target: int = 2000,
    seed: Optional[int] = None,
) -> dict:
    """
    Generate a 7-day meal plan from the safe food pool.
    
    Args:
        safe_foods: DataFrame of foods that passed all filters
        age: Person's age (for RDA lookup)
        sex: Person's sex (for RDA lookup)
        calorie_target: Daily calorie target
        seed: Random seed for reproducibility (None = random)
    
    Returns:
        {
            "days": [
                {
                    "day": 1,
                    "meals": [
                        {"meal_name": "Breakfast", "food": {...}, "gap_score": 85.3},
                        ...
                    ],
                    "daily_totals": {...},
                    "gap_analysis": [...],
                },
                ...
            ],
            "weekly_summary": {...},
            "generation_time_s": 1.23,
            "diversity_score": 0.85,
        }
    """
    start_time = time.time()

    if seed is not None:
        np.random.seed(seed)

    rda = get_rda(age, sex, calorie_target=calorie_target)
    foods_list = safe_foods.to_dict("records")

    if len(foods_list) < DAYS * MEALS_PER_DAY:
        raise ValueError(
            f"Not enough safe foods ({len(foods_list)}) to fill "
            f"{DAYS * MEALS_PER_DAY} meal slots. Loosen filter constraints."
        )

    plan_days = []
    used_globally = set()     # Track food descriptions used across the week
    all_categories_used = []  # For diversity score calculation

    for day_num in range(1, DAYS + 1):
        day_meals = []
        categories_today = []
        used_today = set()

        # Daily nutrient budget starts at full RDA
        remaining_budget = rda.copy()

        for meal_idx, meal_name in enumerate(MEAL_NAMES):
            # Target calories for this meal
            meal_cal_target = calorie_target * MEAL_CALORIE_SPLIT[meal_name]
            cal_tolerance = meal_cal_target * 0.5  # Allow ±50% flexibility

            # Score all candidate foods
            candidates = []
            for food in foods_list:
                desc = food.get("description", "")
                cat = food.get("food_category", "")
                food_cal = float(food.get("calories", 0) or 0)

                # Skip if already used today
                if desc in used_today:
                    continue

                # Skip if this category appeared too many times today
                if categories_today.count(cat) >= MAX_CATEGORY_PER_DAY:
                    continue

                # Prefer foods within calorie range for this meal
                if food_cal <= 0 or food_cal > meal_cal_target + cal_tolerance:
                    continue

                # Score by nutrient gap filling
                score = get_gap_score(food, remaining_budget, rda)

                # Bonus for foods not used this week (diversity)
                if desc not in used_globally:
                    score *= 1.15

                # Bonus for calorie-appropriate foods
                cal_fit = 1.0 - abs(food_cal - meal_cal_target) / (meal_cal_target + 1)
                score *= (0.7 + 0.3 * max(cal_fit, 0))

                candidates.append((score, food))

            # Sort by score descending, pick from top K with some randomness
            candidates.sort(key=lambda x: x[0], reverse=True)
            top_candidates = candidates[:TOP_K_CANDIDATES]

            if not top_candidates:
                # Fallback: pick any unused food
                fallback = [f for f in foods_list if f["description"] not in used_today]
                if fallback:
                    chosen = fallback[np.random.randint(0, len(fallback))]
                    chosen_score = 0.0
                else:
                    # Absolute fallback: reuse a food
                    chosen = foods_list[np.random.randint(0, len(foods_list))]
                    chosen_score = 0.0
            else:
                # Pick from top candidates with weighted random (favor higher scores)
                scores = np.array([s for s, _ in top_candidates])
                if scores.sum() > 0:
                    probs = scores / scores.sum()
                else:
                    probs = np.ones(len(scores)) / len(scores)
                pick_idx = np.random.choice(len(top_candidates), p=probs)
                chosen_score, chosen = top_candidates[pick_idx]

            # Record the meal
            day_meals.append({
                "meal_name": meal_name,
                "food": chosen,
                "gap_score": chosen_score,
            })

            # Update tracking
            desc = chosen.get("description", "")
            cat = chosen.get("food_category", "")
            used_today.add(desc)
            used_globally.add(desc)
            categories_today.append(cat)
            all_categories_used.append(cat)

            # Subtract from remaining budget
            for nutrient in ["calories"] + TRACKED_NUTRIENTS:
                food_val = float(chosen.get(nutrient, 0) or 0)
                remaining_budget[nutrient] = max(
                    remaining_budget.get(nutrient, 0) - food_val, 0
                )

        # Compute daily totals and gap analysis
        meals_for_totals = [m["food"] for m in day_meals]
        daily_totals = compute_daily_totals(meals_for_totals)
        gap_analysis = analyze_gaps(daily_totals, rda)

        plan_days.append({
            "day": day_num,
            "meals": day_meals,
            "daily_totals": daily_totals,
            "gap_analysis": gap_analysis,
        })

    # Weekly summary
    weekly_totals = {}
    for nutrient in ["calories"] + TRACKED_NUTRIENTS:
        weekly_totals[nutrient] = round(
            sum(d["daily_totals"].get(nutrient, 0) for d in plan_days), 1
        )
    weekly_averages = {k: round(v / DAYS, 1) for k, v in weekly_totals.items()}

    # Diversity score: unique food descriptions / total meals
    all_foods_used = [
        m["food"]["description"]
        for d in plan_days
        for m in d["meals"]
    ]
    unique_foods = len(set(all_foods_used))
    total_meals = len(all_foods_used)
    diversity_score = round(unique_foods / total_meals, 3) if total_meals > 0 else 0

    # Category diversity (Shannon entropy)
    cat_counts = pd.Series(all_categories_used).value_counts(normalize=True)
    category_entropy = round(-sum(p * np.log2(p) for p in cat_counts if p > 0), 3)

    generation_time = round(time.time() - start_time, 2)

    return {
        "days": plan_days,
        "weekly_summary": {
            "totals": weekly_totals,
            "daily_averages": weekly_averages,
        },
        "generation_time_s": generation_time,
        "diversity_score": diversity_score,
        "unique_foods": unique_foods,
        "total_meals": total_meals,
        "category_entropy": category_entropy,
    }


# ---------------------------------------------------------------------------
# DISPLAY HELPERS
# ---------------------------------------------------------------------------
def print_plan(plan: dict, persona_name: str = "User"):
    """Pretty-print a meal plan to the console."""
    print(f"\n{'═' * 70}")
    print(f"  7-DAY MEAL PLAN FOR {persona_name.upper()}")
    print(f"  Generated in {plan['generation_time_s']}s")
    print(f"  Diversity: {plan['diversity_score']:.0%} unique meals "
          f"({plan['unique_foods']}/{plan['total_meals']})")
    print(f"{'═' * 70}")

    for day_data in plan["days"]:
        day = day_data["day"]
        totals = day_data["daily_totals"]
        print(f"\n  📅 Day {day}")
        print(f"  {'─' * 66}")

        for meal in day_data["meals"]:
            name = meal["meal_name"]
            food = meal["food"]
            desc = food.get("description", "Unknown")
            cal = food.get("calories", 0)
            cat = food.get("food_category", "")
            print(f"    {name:<12} {desc:<40} {cal:>5.0f} kcal  [{cat}]")

        # Day totals
        print(f"    {'':─<66}")
        print(f"    Day total:  {totals['calories']:.0f} kcal | "
              f"P: {totals['protein_g']:.0f}g | "
              f"C: {totals['carbs_g']:.0f}g | "
              f"F: {totals['fat_g']:.0f}g | "
              f"Fiber: {totals['fiber_g']:.0f}g")

        # Flag nutrients below 80% RDA
        low_nutrients = [
            g for g in day_data["gap_analysis"]
            if g["status"] == "low" and g["nutrient"] != "calories"
        ]
        if low_nutrients:
            low_names = [f"{g['display_name']} ({g['pct']:.0f}%)"
                         for g in low_nutrients[:3]]
            print(f"    ⚠️  Below 80% RDA: {', '.join(low_names)}")

    # Weekly summary
    avg = plan["weekly_summary"]["daily_averages"]
    print(f"\n{'═' * 70}")
    print(f"  WEEKLY AVERAGES")
    print(f"  Calories: {avg['calories']:.0f}/day | "
          f"Protein: {avg['protein_g']:.0f}g | "
          f"Carbs: {avg['carbs_g']:.0f}g | "
          f"Fat: {avg['fat_g']:.0f}g")
    print(f"{'═' * 70}")


# ---------------------------------------------------------------------------
# STANDALONE TEST
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from filters import load_foods, apply_all_filters, PERSONAS

    print("=" * 60)
    print("  NutriAI — Meal Planner Test")
    print("=" * 60)

    df = load_foods()

    # Test with Priya (most restrictive persona)
    persona = PERSONAS["Priya"]
    print(f"\nTesting with: {persona['description']}")

    safe_foods, exclusions = apply_all_filters(
        df,
        conditions=persona["conditions"],
        allergens=persona["allergens"],
        diet=persona["diet"],
        calorie_target=persona["calorie_target"],
    )

    plan = generate_plan(
        safe_foods,
        age=persona["age"],
        sex=persona["sex"],
        calorie_target=persona["calorie_target"],
        seed=42,
    )

    print_plan(plan, "Priya")

    # Verify sub-60s constraint
    if plan["generation_time_s"] < 60:
        print(f"\n  ✅ Generation time: {plan['generation_time_s']}s (under 60s limit)")
    else:
        print(f"\n  ❌ Generation time: {plan['generation_time_s']}s (EXCEEDS 60s limit!)")

    # Test all personas briefly
    print(f"\n\n{'=' * 60}")
    print("  Quick test — all 4 personas")
    print(f"{'=' * 60}")

    for name, p in PERSONAS.items():
        safe, _ = apply_all_filters(
            df,
            conditions=p["conditions"],
            allergens=p["allergens"],
            diet=p["diet"],
        )
        plan = generate_plan(
            safe, age=p["age"], sex=p["sex"],
            calorie_target=p["calorie_target"], seed=42,
        )
        print(f"  {name}: {plan['generation_time_s']}s | "
              f"{plan['unique_foods']}/{plan['total_meals']} unique | "
              f"Avg {plan['weekly_summary']['daily_averages']['calories']:.0f} kcal/day")

    print(f"\n  All 4 personas generated successfully ✅")
