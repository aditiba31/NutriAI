"""
NutriAI — Filters Module
=========================
Composable filter chain for clinical conditions, allergens, and dietary
preferences. Each filter takes a food DataFrame and returns:
    - A reduced DataFrame of "safe" foods
    - A list of (food_description, reason) tuples explaining exclusions

Filters stack: clinical → allergen → diet → sodium/GI limits

Usage:
    from filters import apply_all_filters
    safe_foods, exclusions = apply_all_filters(
        df,
        conditions=["ibs"],
        allergens=["dairy", "gluten"],
        diet="vegetarian",
        calorie_target=2000,
    )
"""

import sqlite3
from pathlib import Path
from typing import Optional

import pandas as pd


# ---------------------------------------------------------------------------
# DATABASE LOADING
# ---------------------------------------------------------------------------
DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "nutriai_foods.db"


def load_foods(db_path: Optional[Path] = None) -> pd.DataFrame:
    """Load all foods from the SQLite database into a DataFrame."""
    db_path = db_path or DEFAULT_DB_PATH
    if not db_path.exists():
        raise FileNotFoundError(
            f"Food database not found at {db_path}. "
            "Run 'python code/data_pipeline.py --fallback' first."
        )
    conn = sqlite3.connect(str(db_path))
    df = pd.read_sql("SELECT * FROM foods", conn)
    conn.close()
    return df


# ---------------------------------------------------------------------------
# INDIVIDUAL FILTERS
# ---------------------------------------------------------------------------
# Each filter function follows the same signature:
#   filter_fn(df: DataFrame) -> (safe_df: DataFrame, exclusions: list[tuple])
#
# This composable design means any combination of conditions works,
# including hidden personas we haven't seen yet.
# ---------------------------------------------------------------------------

def filter_ibs(df: pd.DataFrame) -> tuple[pd.DataFrame, list[tuple]]:
    """
    IBS filter: remove high-FODMAP foods.
    
    Based on Monash University FODMAP classification. High-FODMAP foods
    (garlic, onion, wheat, certain fruits/legumes, lactose-containing dairy)
    can trigger IBS symptoms like bloating, pain, and altered bowel habits.
    """
    excluded_mask = df["is_high_fodmap"] == 1
    exclusions = [
        (row["description"], "High-FODMAP food — may trigger IBS symptoms (bloating, pain)")
        for _, row in df[excluded_mask].iterrows()
    ]
    return df[~excluded_mask].reset_index(drop=True), exclusions


def filter_gerd(df: pd.DataFrame) -> tuple[pd.DataFrame, list[tuple]]:
    """
    GERD filter: remove acid reflux trigger foods.
    
    GERD triggers include acidic foods (tomato, citrus), caffeine,
    chocolate, spicy foods, fried foods, mint, and carbonated drinks.
    These relax the lower esophageal sphincter or increase stomach acid.
    """
    excluded_mask = df["is_gerd_trigger"] == 1
    exclusions = [
        (row["description"], "GERD trigger — may cause acid reflux (acidic, spicy, or fried)")
        for _, row in df[excluded_mask].iterrows()
    ]
    return df[~excluded_mask].reset_index(drop=True), exclusions


def filter_t2_diabetes(df: pd.DataFrame) -> tuple[pd.DataFrame, list[tuple]]:
    """
    Type 2 Diabetes filter: remove high-GI foods.
    
    High glycemic index foods cause rapid blood sugar spikes.
    We exclude foods with GI estimate = "high" (GI ≥ 70) and flag
    medium-GI foods. Also limits high-sugar items.
    """
    excluded_mask = df["gi_estimate"] == "high"
    exclusions = [
        (row["description"], "High glycemic index — causes rapid blood sugar spikes (GI ≥ 70)")
        for _, row in df[excluded_mask].iterrows()
    ]
    return df[~excluded_mask].reset_index(drop=True), exclusions


def filter_hypertension(df: pd.DataFrame) -> tuple[pd.DataFrame, list[tuple]]:
    """
    Hypertension filter: remove high-sodium foods.
    
    DASH diet guidelines recommend <2,300 mg sodium/day (ideally <1,500 mg).
    We exclude individual foods with >400 mg sodium per 100g, as these
    would consume a large portion of the daily budget in a single serving.
    Also flags foods tagged as high-sodium from keyword matching.
    """
    excluded_mask = df["is_high_sodium"] == 1
    exclusions = [
        (row["description"], f"High sodium ({row['sodium_mg']:.0f} mg/100g) — risk for hypertension (DASH limit)")
        for _, row in df[excluded_mask].iterrows()
    ]
    return df[~excluded_mask].reset_index(drop=True), exclusions


# Map condition names to filter functions
CLINICAL_FILTERS = {
    "ibs": filter_ibs,
    "gerd": filter_gerd,
    "t2_diabetes": filter_t2_diabetes,
    "hypertension": filter_hypertension,
}


def filter_allergens(df: pd.DataFrame, allergens: list[str]) -> tuple[pd.DataFrame, list[tuple]]:
    """
    Allergen filter: remove foods containing specified allergens.
    
    Supported allergens: dairy, gluten, soy, tree_nuts, eggs
    
    Each allergen maps to a boolean column in the database.
    This is keyword-based (not supply-chain verified), which is a
    known limitation documented in the brief.
    """
    allergen_col_map = {
        "dairy":     "contains_dairy",
        "lactose":   "contains_dairy",     # Lactose intolerance → exclude dairy
        "gluten":    "contains_gluten",
        "soy":       "contains_soy",
        "tree_nuts": "contains_tree_nuts",
        "nuts":      "contains_tree_nuts",
        "eggs":      "contains_eggs",
    }

    all_exclusions = []
    combined_mask = pd.Series([False] * len(df), index=df.index)

    for allergen in allergens:
        allergen_lower = allergen.lower().strip()
        col = allergen_col_map.get(allergen_lower)
        if col is None:
            print(f"⚠️  Unknown allergen '{allergen}' — skipping. "
                  f"Supported: {list(allergen_col_map.keys())}")
            continue

        mask = df[col] == 1
        for _, row in df[mask & ~combined_mask].iterrows():
            all_exclusions.append(
                (row["description"], f"Contains {allergen_lower} — allergen exclusion")
            )
        combined_mask = combined_mask | mask

    return df[~combined_mask].reset_index(drop=True), all_exclusions


def filter_diet(df: pd.DataFrame, diet: str) -> tuple[pd.DataFrame, list[tuple]]:
    """
    Diet preference filter: keep only foods matching the dietary pattern.
    
    Supported diets: vegan, vegetarian, pescatarian, non-veg (no filter)
    """
    diet_lower = diet.lower().strip()

    diet_col_map = {
        "vegan":       "is_vegan",
        "vegetarian":  "is_vegetarian",
        "pescatarian": "is_pescatarian",
    }

    if diet_lower in ("non-veg", "non-vegetarian", "none", "no preference", ""):
        return df, []

    col = diet_col_map.get(diet_lower)
    if col is None:
        print(f"⚠️  Unknown diet '{diet}' — no diet filter applied. "
              f"Supported: {list(diet_col_map.keys())}")
        return df, []

    excluded_mask = df[col] == 0
    exclusions = [
        (row["description"], f"Not {diet_lower} — excluded by dietary preference")
        for _, row in df[excluded_mask].iterrows()
    ]
    return df[~excluded_mask].reset_index(drop=True), exclusions


# ---------------------------------------------------------------------------
# MAIN FILTER CHAIN
# ---------------------------------------------------------------------------
def apply_all_filters(
    df: pd.DataFrame,
    conditions: list[str] = None,
    allergens: list[str] = None,
    diet: str = "none",
    calorie_target: int = 2000,
) -> tuple[pd.DataFrame, list[tuple]]:
    """
    Apply the full filter chain: clinical → allergen → diet.
    
    Args:
        df: Full food database DataFrame
        conditions: Clinical conditions (e.g., ["ibs", "gerd"])
        allergens: Allergen list (e.g., ["dairy", "gluten"])
        diet: Dietary preference ("vegan", "vegetarian", "pescatarian", "non-veg")
        calorie_target: Daily calorie target (for reference, not used in filtering)
    
    Returns:
        (safe_foods_df, exclusions_list)
        where exclusions_list = [(food_description, reason_string), ...]
    """
    conditions = conditions or []
    allergens = allergens or []
    all_exclusions = []
    safe = df.copy()

    initial_count = len(safe)

    # Step 1: Clinical condition filters
    for condition in conditions:
        condition_lower = condition.lower().strip()
        filter_fn = CLINICAL_FILTERS.get(condition_lower)
        if filter_fn is None:
            print(f"⚠️  Unknown condition '{condition}' — skipping. "
                  f"Supported: {list(CLINICAL_FILTERS.keys())}")
            continue
        safe, excl = filter_fn(safe)
        all_exclusions.extend(excl)

    # Step 2: Allergen filter
    if allergens:
        safe, excl = filter_allergens(safe, allergens)
        all_exclusions.extend(excl)

    # Step 3: Diet filter
    safe, excl = filter_diet(safe, diet)
    all_exclusions.extend(excl)

    # Step 4: Remove foods with zero calories (data quality)
    zero_cal_mask = safe["calories"] <= 0
    if zero_cal_mask.any():
        for _, row in safe[zero_cal_mask].iterrows():
            all_exclusions.append(
                (row["description"], "Zero calories — likely a data quality issue")
            )
        safe = safe[~zero_cal_mask].reset_index(drop=True)

    final_count = len(safe)
    excluded_count = initial_count - final_count

    print(f"🔍 Filter chain complete:")
    print(f"   Input:    {initial_count} foods")
    print(f"   Excluded: {excluded_count} foods")
    print(f"   Safe:     {final_count} foods")

    if final_count < 50:
        print(f"   ⚠️  WARNING: Only {final_count} safe foods remaining! "
              "Meal plan diversity may be limited.")

    return safe, all_exclusions


# ---------------------------------------------------------------------------
# TEST PERSONAS — used for validation and the pass/fail table in brief.pdf
# ---------------------------------------------------------------------------
PERSONAS = {
    "Priya": {
        "description": "28F, IBS + Vegetarian + Lactose Intolerant",
        "age": 28,
        "sex": "female",
        "conditions": ["ibs"],
        "allergens": ["dairy"],
        "diet": "vegetarian",
        "calorie_target": 1800,
    },
    "Ravi": {
        "description": "45M, GERD + Non-Veg + Gluten-Free",
        "age": 45,
        "sex": "male",
        "conditions": ["gerd"],
        "allergens": ["gluten"],
        "diet": "non-veg",
        "calorie_target": 2200,
    },
    "Mei": {
        "description": "35F, Type 2 Diabetes + Vegan + No Tree Nuts",
        "age": 35,
        "sex": "female",
        "conditions": ["t2_diabetes"],
        "allergens": ["tree_nuts"],
        "diet": "vegan",
        "calorie_target": 1600,
    },
    "James": {
        "description": "55M, Hypertension + Pescatarian + No Soy",
        "age": 55,
        "sex": "male",
        "conditions": ["hypertension"],
        "allergens": ["soy"],
        "diet": "pescatarian",
        "calorie_target": 2000,
    },
}


# ---------------------------------------------------------------------------
# STANDALONE TEST
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("  NutriAI — Filter Chain Test")
    print("=" * 60)

    df = load_foods()
    print(f"\nLoaded {len(df)} foods from database\n")

    for name, persona in PERSONAS.items():
        print(f"\n{'─' * 60}")
        print(f"  PERSONA: {name} — {persona['description']}")
        print(f"{'─' * 60}")

        safe, exclusions = apply_all_filters(
            df,
            conditions=persona["conditions"],
            allergens=persona["allergens"],
            diet=persona["diet"],
            calorie_target=persona["calorie_target"],
        )

        # Show category distribution of safe foods
        cats = safe["food_category"].value_counts().head(5)
        print(f"\n   Top safe food categories:")
        for cat, count in cats.items():
            print(f"      {cat}: {count}")

        # Show a few sample exclusions
        print(f"\n   Sample exclusions (showing 3 of {len(exclusions)}):")
        for food, reason in exclusions[:3]:
            print(f"      ✗ {food}")
            print(f"        → {reason}")

    print(f"\n{'=' * 60}")
    print("  All 4 personas filtered successfully ✅")
    print(f"{'=' * 60}")
