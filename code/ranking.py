"""
NutriAI — Ranking Module (v2)
===============================
FAISS + Bloom filter with serving-size-aware meal planning,
meal-type classification, and improved diversity.

BAX-423 Techniques:
    1. FAISS — nutrient-vector similarity search
    2. Bloom Filter — probabilistic exclusion checking
"""

import time
from typing import Optional

import numpy as np
import pandas as pd
import faiss
from pybloom_live import BloomFilter
from sklearn.preprocessing import MinMaxScaler

from nutrients import (
    get_rda, compute_daily_totals, analyze_gaps,
    TRACKED_NUTRIENTS, NUTRIENT_DISPLAY,
)
from data_sources import MEAL_TYPE_FOODS

EMBEDDING_COLS = [
    "calories", "protein_g", "carbs_g", "fat_g", "fiber_g",
    "iron_mg", "calcium_mg", "b12_mcg", "vitamin_d_mcg", "zinc_mg",
    "sodium_mg", "potassium_mg", "magnesium_mg",
]

# Typical serving sizes by food category (grams)
# Keys cover both our curated categories AND USDA category names
SERVING_SIZES = {
    "grains": 250,
    "cereal": 250,
    "baked": 200,
    "vegetables": 150,
    "vegetable": 150,
    "fruits": 175,
    "fruit": 175,
    "legume": 200,
    "bean": 200,
    "nut": 40,
    "seed": 40,
    "dairy": 200,
    "milk": 200,
    "cheese": 40,
    "egg": 120,
    "poultry": 170,
    "chicken": 170,
    "turkey": 170,
    "beef": 170,
    "pork": 170,
    "lamb": 170,
    "meat": 170,
    "fish": 170,
    "seafood": 170,
    "finfish": 170,
    "shellfish": 150,
    "oil": 15,
    "fat": 15,
    "beverage": 250,
    "drink": 250,
    "condiment": 30,
    "sauce": 30,
    "spice": 5,
    "herb": 5,
    "soup": 300,
    "baby": 0,  # Will be filtered out
    "snack": 40,
    "sweet": 30,
    "candy": 0,
    "prepared": 350,
}

# USDA ingredient entries that aren't standalone foods
USDA_INGREDIENT_KEYWORDS = [
    "raw, frozen", "pasteurized", "dehydrated",
    "concentrate", "isolate", "extract",
    "powder, dry", "dry, powder",
    "industrial", "food service",
    "imitation", "analog",
    "formula,", "formulated",
]

# Foods that need specific serving size overrides regardless of category
FOOD_NAME_SERVING_OVERRIDES = {
    "peanut butter": 32,
    "almond butter": 32,
    "tahini": 30,
    "sesame butter": 30,
    "nutella": 30,
    "peanuts": 40,
    "peanut": 40,
    "cheese": 40,
    "gjetost": 40,
    "tostada": 60,
    "coffee": 10,
    "instant coffee": 5,
    "cocoa powder": 10,
    "french fried": 100,
    "french fries": 100,
    "jam": 30,
    "jelly": 30,
    "honey": 20,
    "syrup": 30,
    "popcorn": 30,
    "trail mix": 40,
    "chips": 30,
    "cracker": 30,
    "pretzels": 30,
    "dried fruit": 40,
    "jerky": 30,
    "granola": 50,
    "muesli": 60,
}

DAYS = 7
MEALS_PER_DAY = 3
MEAL_NAMES = ["Breakfast", "Lunch", "Dinner"]
MEAL_CALORIE_SPLIT = {"Breakfast": 0.25, "Lunch": 0.35, "Dinner": 0.40}
MAX_CATEGORY_PER_DAY = 2
FAISS_TOP_K = 80


def get_serving_multiplier(food_category, food_desc=""):
    """Get the serving size multiplier (serving_g / 100g).
    Checks food name overrides first, then category matching."""
    desc_lower = food_desc.lower() if food_desc else ""
    
    # Check food name overrides first (most specific)
    for food_name, serving_g in FOOD_NAME_SERVING_OVERRIDES.items():
        if food_name in desc_lower:
            return serving_g / 100.0
    
    # Then check category
    cat_lower = food_category.lower() if food_category else ""
    for key, serving_g in SERVING_SIZES.items():
        if key in cat_lower:
            if serving_g == 0:
                return 0
            return serving_g / 100.0
    return 200 / 100.0  # Default: 200g serving


def is_usda_ingredient(food_desc):
    """Check if a USDA entry is an ingredient, not a standalone food."""
    desc_lower = food_desc.lower()
    return any(kw in desc_lower for kw in USDA_INGREDIENT_KEYWORDS)


def get_base_food_name(description):
    """Extract base food name for diversity tracking.
    'Broccoli, steamed' and 'Broccoli, raw' → 'broccoli'"""
    desc = description.lower().strip()
    # Take first part before comma
    base = desc.split(",")[0].strip()
    # Remove common prefixes
    for prefix in ["organic ", "store brand ", "premium ", "farm-fresh ",
                   "all-natural ", "locally sourced ", "imported ",
                   "mediterranean-style ", "asian-inspired ", "mexican-style ",
                   "indian-spiced ", "thai-style ", "japanese-style ",
                   "middle eastern ", "italian-style ", "korean-style ",
                   "ethiopian-style ", "caribbean-style ", "cajun-style ",
                   "grilled ", "baked ", "steamed ", "roasted ", "sauteed ",
                   "braised ", "poached ", "broiled ", "pan-seared ",
                   "stir-fried ", "low-sodium ", "reduced-fat ", "fortified "]:
        if base.startswith(prefix):
            base = base[len(prefix):]
    return base.strip()


def get_meal_type_score(food_desc, food_cat, meal_name):
    """Score how appropriate a food is for this meal type (0.0 to 1.0)."""
    desc_lower = food_desc.lower()
    
    if meal_name == "Breakfast":
        if any(kw in desc_lower for kw in MEAL_TYPE_FOODS["breakfast"]):
            return 1.0
        # Grains, fruits, dairy, eggs are good breakfast foods
        if food_cat in ("Grains", "Fruits", "Dairy", "Eggs"):
            return 0.8
        # Heavy proteins and prepared foods less ideal for breakfast
        if food_cat in ("Red Meat", "Fish and Seafood", "Prepared Foods"):
            return 0.3
        return 0.5
    
    else:  # Lunch or Dinner
        if any(kw in desc_lower for kw in MEAL_TYPE_FOODS["lunch_dinner"]):
            return 1.0
        if food_cat in ("Poultry", "Red Meat", "Fish and Seafood", "Legumes", "Prepared Foods"):
            return 0.9
        if food_cat in ("Grains", "Vegetables"):
            return 0.7
        # Pure nuts/seeds and condiments are less ideal as main meals
        if food_cat in ("Nuts and Seeds", "Condiments", "Oils and Fats"):
            return 0.2
        return 0.5


def scale_food_nutrients(food, multiplier):
    """Return a copy of food dict with nutrients scaled by serving multiplier."""
    scaled = dict(food)
    for col in EMBEDDING_COLS:
        if col in scaled and scaled[col]:
            scaled[col] = round(float(scaled[col]) * multiplier, 2)
    return scaled


class FAISSIndex:
    def __init__(self, foods_df):
        self.foods_df = foods_df.reset_index(drop=True)
        self.n_foods = len(foods_df)
        self.dim = len(EMBEDDING_COLS)
        nutrient_matrix = foods_df[EMBEDDING_COLS].fillna(0).values.astype(np.float32)
        self.scaler = MinMaxScaler()
        self.normalized = self.scaler.fit_transform(nutrient_matrix).astype(np.float32)
        self.index = faiss.IndexFlatL2(self.dim)
        self.index.add(self.normalized)

    def query(self, ideal_profile, k=20):
        query_raw = np.array(
            [ideal_profile.get(col, 0) for col in EMBEDDING_COLS], dtype=np.float32
        ).reshape(1, -1)
        query_norm = self.scaler.transform(query_raw).astype(np.float32)
        k = min(k, self.n_foods)
        distances, indices = self.index.search(query_norm, k)
        return indices[0].tolist()

    def get_food(self, idx):
        return self.foods_df.iloc[idx].to_dict()


class ExclusionBloomFilter:
    def __init__(self, excluded_foods, error_rate=0.01):
        self.n_excluded = len(excluded_foods)
        self.error_rate = error_rate
        self.bloom = BloomFilter(capacity=max(self.n_excluded, 100), error_rate=error_rate)
        for desc in excluded_foods:
            self.bloom.add(desc)

    def is_excluded(self, food_desc):
        return food_desc in self.bloom

    def measure_false_positive_rate(self, safe_foods):
        if not safe_foods:
            return 0.0
        return sum(1 for f in safe_foods if self.is_excluded(f)) / len(safe_foods)


def generate_plan_with_faiss(safe_foods, exclusions, age=30, sex="female",
                              calorie_target=2000, seed=None):
    start_time = time.time()
    if seed is not None:
        np.random.seed(seed)

    rda = get_rda(age, sex, calorie_target=calorie_target)

    # Build FAISS index
    faiss_build_start = time.time()
    faiss_idx = FAISSIndex(safe_foods)
    faiss_build_time = time.time() - faiss_build_start

    # Build Bloom filter
    bloom_build_start = time.time()
    excluded_descs = [desc for desc, reason in exclusions]
    bloom = ExclusionBloomFilter(excluded_descs, error_rate=0.01)
    bloom_build_time = time.time() - bloom_build_start

    faiss_query_times = []
    bloom_check_times = []

    plan_days = []
    used_globally = set()
    base_name_counts = {}  # Track base food names for diversity
    all_categories_used = []

    for day_num in range(1, DAYS + 1):
        day_meals = []
        categories_today = []
        used_today = set()
        remaining_budget = rda.copy()

        for meal_idx, meal_name in enumerate(MEAL_NAMES):
            meal_cal_target = calorie_target * MEAL_CALORIE_SPLIT[meal_name]

            # Ideal nutrient profile: blend of RDA share + remaining gap
            ideal_profile = {}
            for col in EMBEDDING_COLS:
                rda_share = rda.get(col, 0) * MEAL_CALORIE_SPLIT[meal_name]
                remaining = remaining_budget.get(col, 0)
                meals_left = max(MEALS_PER_DAY - meal_idx, 1)
                ideal_profile[col] = 0.4 * rda_share + 0.6 * (remaining / meals_left)

            # FAISS retrieval
            faiss_q_start = time.time()
            candidate_indices = faiss_idx.query(ideal_profile, k=FAISS_TOP_K)
            faiss_query_times.append(time.time() - faiss_q_start)

            # Score candidates
            bloom_start = time.time()
            scored_candidates = []

            for idx in candidate_indices:
                food = faiss_idx.get_food(idx)
                desc = food.get("description", "")
                cat = food.get("food_category", "")

                if bloom.is_excluded(desc):
                    continue
                if desc in used_today:
                    continue
                if categories_today.count(cat) >= MAX_CATEGORY_PER_DAY:
                    continue

                # Skip USDA ingredient entries (not standalone foods)
                if is_usda_ingredient(desc):
                    continue

                # Apply serving size (handles USDA category names)
                multiplier = get_serving_multiplier(cat, desc)
                if multiplier == 0:
                    continue  # Baby food etc.
                food_cal = float(food.get("calories", 0) or 0) * multiplier

                # Must be in a reasonable calorie range for this meal
                # Lunch and dinner need more substantial meals than breakfast
                if meal_name == "Breakfast":
                    cal_floor = meal_cal_target * 0.25
                else:
                    cal_floor = meal_cal_target * 0.4
                if food_cal < cal_floor or food_cal > meal_cal_target * 1.3:
                    continue

                # Base-name diversity: limit any base food to max 3 per week
                base_name = get_base_food_name(desc)
                if base_name_counts.get(base_name, 0) >= 3:
                    continue

                # Nutrient gap score
                gap_score = 0.0
                for col in EMBEDDING_COLS:
                    remaining = remaining_budget.get(col, 0)
                    food_val = float(food.get(col, 0) or 0) * multiplier
                    if remaining > 0:
                        gap_score += min(food_val / remaining, 1.0)

                # Meal-type appropriateness
                meal_fit = get_meal_type_score(desc, cat, meal_name)
                gap_score *= (0.5 + 0.5 * meal_fit)

                # Weekly diversity bonus — stronger for unused foods
                if desc not in used_globally:
                    gap_score *= 1.3
                if base_name_counts.get(base_name, 0) == 0:
                    gap_score *= 1.2

                # Calorie fit — strongly penalize foods far from target
                cal_fit = 1.0 - abs(food_cal - meal_cal_target) / (meal_cal_target + 1)
                gap_score *= (0.3 + 0.7 * max(cal_fit, 0))

                # Penalize condiments/oils/seeds as main dishes
                cat_lower = cat.lower()
                if any(k in cat_lower for k in ["condiment", "sauce", "spice", "oil", "fat"]):
                    gap_score *= 0.05
                if any(k in cat_lower for k in ["nut", "seed"]) and meal_name != "Snack":
                    gap_score *= 0.3

                scored_candidates.append((gap_score, food, multiplier))

            bloom_check_times.append(time.time() - bloom_start)

            # Pick from top candidates
            if scored_candidates:
                scored_candidates.sort(key=lambda x: x[0], reverse=True)
                top = scored_candidates[:10]
                scores_arr = np.array([s for s, _, _ in top])
                if scores_arr.sum() > 0:
                    probs = scores_arr / scores_arr.sum()
                else:
                    probs = np.ones(len(scores_arr)) / len(scores_arr)
                pick = np.random.choice(len(top), p=probs)
                chosen_score, chosen_raw, chosen_mult = top[pick]
                chosen = scale_food_nutrients(chosen_raw, chosen_mult)
                chosen["serving_g"] = round(chosen_mult * 100)
            else:
                # Fallback: pick a random food that meets calorie minimum
                min_cal = meal_cal_target * 0.3
                eligible = [
                    safe_foods.iloc[i].to_dict()
                    for i in np.random.choice(len(safe_foods), min(100, len(safe_foods)), replace=False)
                    if float(safe_foods.iloc[i].get("calories", 0) or 0) * get_serving_multiplier(
                        safe_foods.iloc[i].get("food_category", ""), safe_foods.iloc[i].get("description", "")
                    ) >= min_cal
                ]
                if eligible:
                    chosen_raw = eligible[np.random.randint(0, len(eligible))]
                else:
                    chosen_raw = safe_foods.iloc[np.random.randint(0, len(safe_foods))].to_dict()
                cat = chosen_raw.get("food_category", "Prepared Foods")
                chosen_mult = get_serving_multiplier(cat, chosen_raw.get("description", ""))
                chosen = scale_food_nutrients(chosen_raw, chosen_mult)
                chosen["serving_g"] = round(chosen_mult * 100)
                chosen_score = 0.0

            day_meals.append({
                "meal_name": meal_name,
                "food": chosen,
                "gap_score": chosen_score,
            })

            desc = chosen.get("description", "")
            cat = chosen.get("food_category", "")
            used_today.add(desc)
            used_globally.add(desc)
            categories_today.append(cat)
            all_categories_used.append(cat)
            
            # Track base food name for weekly diversity
            base = get_base_food_name(desc)
            base_name_counts[base] = base_name_counts.get(base, 0) + 1

            for nutrient in ["calories"] + TRACKED_NUTRIENTS:
                food_val = float(chosen.get(nutrient, 0) or 0)
                remaining_budget[nutrient] = max(
                    remaining_budget.get(nutrient, 0) - food_val, 0
                )

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
            sum(d["daily_totals"].get(nutrient, 0) for d in plan_days), 1)
    weekly_averages = {k: round(v / DAYS, 1) for k, v in weekly_totals.items()}

    all_foods_used = [m["food"]["description"] for d in plan_days for m in d["meals"]]
    unique_foods = len(set(all_foods_used))
    total_meals = len(all_foods_used)
    diversity_score = round(unique_foods / total_meals, 3) if total_meals > 0 else 0

    cat_counts = pd.Series(all_categories_used).value_counts(normalize=True)
    category_entropy = round(-sum(p * np.log2(p) for p in cat_counts if p > 0), 3)

    generation_time = round(time.time() - start_time, 2)

    avg_faiss_query = np.mean(faiss_query_times) * 1000
    avg_bloom_check = np.mean(bloom_check_times) * 1000
    fp_rate = bloom.measure_false_positive_rate(
        [f["description"] for _, f in safe_foods.head(500).iterrows()]
    )

    return {
        "days": plan_days,
        "weekly_summary": {"totals": weekly_totals, "daily_averages": weekly_averages},
        "generation_time_s": generation_time,
        "diversity_score": diversity_score,
        "unique_foods": unique_foods,
        "total_meals": total_meals,
        "category_entropy": category_entropy,
        "benchmarks": {
            "faiss_build_time_ms": round(faiss_build_time * 1000, 2),
            "faiss_avg_query_ms": round(avg_faiss_query, 3),
            "faiss_total_queries": len(faiss_query_times),
            "faiss_index_size": faiss_idx.n_foods,
            "bloom_build_time_ms": round(bloom_build_time * 1000, 2),
            "bloom_avg_check_ms": round(avg_bloom_check, 3),
            "bloom_n_excluded": bloom.n_excluded,
            "bloom_false_positive_rate": round(fp_rate * 100, 3),
            "bloom_target_error_rate": 1.0,
        },
    }


if __name__ == "__main__":
    from filters import load_foods, apply_all_filters, PERSONAS
    from nutrients import get_rda

    print("=" * 60)
    print("  NutriAI — Meal Planner v2 Test")
    print("  (Serving sizes + meal-type awareness)")
    print("=" * 60)

    df = load_foods()

    for name, persona in PERSONAS.items():
        print(f"\n{'═' * 60}")
        print(f"  {name} — {persona['description']}")
        print(f"{'═' * 60}")

        safe, exclusions = apply_all_filters(
            df, conditions=persona["conditions"], allergens=persona["allergens"],
            diet=persona["diet"], no_pork=persona.get("no_pork", False),
        )

        plan = generate_plan_with_faiss(
            safe, exclusions, age=persona["age"], sex=persona["sex"],
            calorie_target=persona["calorie_target"], seed=42,
        )

        avg = plan["weekly_summary"]["daily_averages"]
        print(f"\n  ⏱️  Generated in {plan['generation_time_s']}s")
        print(f"  🎯 Avg {avg['calories']:.0f} kcal/day (target: {persona['calorie_target']})")
        print(f"  🔄 {plan['unique_foods']}/{plan['total_meals']} unique meals")
        print(f"  📊 Diversity: {plan['diversity_score']:.0%}")

        # Show Day 1 meals
        print(f"\n  📅 Day 1 sample:")
        for meal in plan["days"][0]["meals"]:
            f = meal["food"]
            serving = f.get("serving_g", "?")
            print(f"     {meal['meal_name']:<12} {f['description']:<40} "
                  f"{f['calories']:.0f} kcal ({serving}g)")

    print(f"\n{'═' * 60}")
    print("  All personas complete ✅")
    print(f"{'═' * 60}")
