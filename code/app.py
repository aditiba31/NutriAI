"""
NutriAI — Streamlit Application
=================================
Main entry point. Generates personalized 7-day meal plans
tailored to clinical conditions, allergens, diet preferences,
and nutritional targets.

Run:  streamlit run code/app.py
"""

import sys
import time
from pathlib import Path
from io import BytesIO

import streamlit as st
import pandas as pd

# Ensure code/ is on the import path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from filters import load_foods, apply_all_filters, PERSONAS, CLINICAL_FILTERS
from meal_planner import generate_plan, MEAL_NAMES
from nutrients import get_rda, analyze_gaps, NUTRIENT_DISPLAY, compute_daily_totals


# ---------------------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="NutriAI — Diet Plan Builder",
    page_icon="🥗",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #2B5797;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #666;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        border: 1px solid #e9ecef;
    }
    .status-ok { color: #28a745; font-weight: bold; }
    .status-low { color: #dc3545; font-weight: bold; }
    .status-high { color: #ffc107; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# LOAD DATA (cached so it only runs once)
# ---------------------------------------------------------------------------
@st.cache_data
def get_food_database():
    """Load and cache the food database."""
    return load_foods()


# ---------------------------------------------------------------------------
# SIDEBAR — INPUT FORM
# ---------------------------------------------------------------------------
st.sidebar.markdown("## 🥗 NutriAI")
st.sidebar.markdown("Personalized Diet Plan Builder")
st.sidebar.markdown("---")

# Preset persona selector
preset = st.sidebar.selectbox(
    "Quick start — load a test persona",
    ["Custom"] + list(PERSONAS.keys()),
    help="Select a preset persona or configure your own below",
)

if preset != "Custom":
    p = PERSONAS[preset]
    default_age = p["age"]
    default_sex = p["sex"]
    default_conditions = p["conditions"]
    default_allergens = p["allergens"]
    default_diet = p["diet"]
    default_calories = p["calorie_target"]
else:
    default_age = 30
    default_sex = "female"
    default_conditions = []
    default_allergens = []
    default_diet = "none"
    default_calories = 2000

st.sidebar.markdown("### Personal Info")
age = st.sidebar.slider("Age", 18, 80, default_age)
sex = st.sidebar.radio("Sex", ["female", "male"], index=0 if default_sex == "female" else 1)

st.sidebar.markdown("### Clinical Conditions")
conditions = st.sidebar.multiselect(
    "Select conditions",
    options=list(CLINICAL_FILTERS.keys()),
    default=default_conditions,
    format_func=lambda x: {
        "ibs": "IBS (Irritable Bowel Syndrome)",
        "gerd": "GERD (Acid Reflux)",
        "t2_diabetes": "Type 2 Diabetes",
        "hypertension": "Hypertension (High Blood Pressure)",
    }.get(x, x),
)

st.sidebar.markdown("### Allergens")
allergens = st.sidebar.multiselect(
    "Select allergens to exclude",
    options=["dairy", "gluten", "soy", "tree_nuts", "eggs"],
    default=default_allergens,
    format_func=lambda x: {
        "dairy": "🥛 Dairy / Lactose",
        "gluten": "🌾 Gluten",
        "soy": "🫘 Soy",
        "tree_nuts": "🥜 Tree Nuts",
        "eggs": "🥚 Eggs",
    }.get(x, x),
)

st.sidebar.markdown("### Diet Preference")
diet_options = ["none", "vegetarian", "vegan", "pescatarian"]
diet = st.sidebar.selectbox(
    "Diet type",
    options=diet_options,
    index=diet_options.index(default_diet) if default_diet in diet_options else 0,
    format_func=lambda x: {
        "none": "No restriction",
        "vegetarian": "🥬 Vegetarian",
        "vegan": "🌱 Vegan",
        "pescatarian": "🐟 Pescatarian",
    }.get(x, x),
)

st.sidebar.markdown("### Calorie Target")
calorie_target = st.sidebar.slider("Daily calories (kcal)", 1200, 3500, default_calories, step=100)

st.sidebar.markdown("---")
generate_btn = st.sidebar.button("🚀 Generate 7-Day Plan", use_container_width=True, type="primary")


# ---------------------------------------------------------------------------
# MAIN CONTENT
# ---------------------------------------------------------------------------
st.markdown('<p class="main-header">🥗 NutriAI — Automated Diet Plan Builder</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Personalized 7-day meal plans tailored to clinical conditions, '
    'allergens, dietary preferences, and nutritional targets.</p>',
    unsafe_allow_html=True,
)

if not generate_btn:
    # Landing state
    st.info(
        "👈 Configure your profile in the sidebar and click **Generate 7-Day Plan** to start. "
        "Or select a test persona from the dropdown to auto-fill.",
        icon="ℹ️",
    )

    # Show test persona cards
    st.markdown("### Test Personas")
    cols = st.columns(4)
    for i, (name, p) in enumerate(PERSONAS.items()):
        with cols[i]:
            st.markdown(f"**{name}**")
            st.caption(p["description"])

    st.stop()


# ---------------------------------------------------------------------------
# GENERATE PLAN
# ---------------------------------------------------------------------------
df = get_food_database()

# Step 1: Filter
with st.spinner("Filtering foods for safety..."):
    safe_foods, exclusions = apply_all_filters(
        df,
        conditions=conditions,
        allergens=allergens,
        diet=diet,
        calorie_target=calorie_target,
    )

# Check if enough safe foods remain
if len(safe_foods) < 21:
    st.error(
        f"⚠️ Only **{len(safe_foods)}** safe foods remain after filtering — "
        f"need at least 21 for a 7-day plan. Try loosening your constraints.",
        icon="🚫",
    )
    st.stop()

# Step 2: Generate
with st.spinner("Generating your personalized 7-day meal plan..."):
    plan = generate_plan(
        safe_foods,
        age=age,
        sex=sex,
        calorie_target=calorie_target,
    )

gen_time = plan["generation_time_s"]

# ---------------------------------------------------------------------------
# DISPLAY — SUMMARY METRICS
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown("### 📊 Plan Summary")

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Generation Time", f"{gen_time}s", delta="Under 60s ✅" if gen_time < 60 else "Over 60s ❌")
m2.metric("Safe Foods", f"{len(safe_foods):,}")
m3.metric("Foods Excluded", f"{len(exclusions):,}")
m4.metric("Unique Meals", f"{plan['unique_foods']}/{plan['total_meals']}")
m5.metric("Avg Calories/Day", f"{plan['weekly_summary']['daily_averages']['calories']:.0f}")


# ---------------------------------------------------------------------------
# DISPLAY — 7-DAY PLAN
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown("### 🗓️ Your 7-Day Meal Plan")

# Create plan as a DataFrame for display and export
plan_rows = []
for day_data in plan["days"]:
    for meal in day_data["meals"]:
        food = meal["food"]
        plan_rows.append({
            "Day": day_data["day"],
            "Meal": meal["meal_name"],
            "Food": food.get("description", ""),
            "Category": food.get("food_category", ""),
            "Calories": round(food.get("calories", 0)),
            "Protein (g)": round(food.get("protein_g", 0), 1),
            "Carbs (g)": round(food.get("carbs_g", 0), 1),
            "Fat (g)": round(food.get("fat_g", 0), 1),
            "Fiber (g)": round(food.get("fiber_g", 0), 1),
        })

plan_df = pd.DataFrame(plan_rows)

# Day-by-day tabs
day_tabs = st.tabs([f"Day {d}" for d in range(1, 8)])

for i, tab in enumerate(day_tabs):
    day_data = plan["days"][i]
    with tab:
        day_df = plan_df[plan_df["Day"] == i + 1][["Meal", "Food", "Category", "Calories", "Protein (g)", "Carbs (g)", "Fat (g)", "Fiber (g)"]]
        st.dataframe(day_df, use_container_width=True, hide_index=True)

        # Day totals
        totals = day_data["daily_totals"]
        tc1, tc2, tc3, tc4, tc5 = st.columns(5)
        tc1.metric("Calories", f"{totals['calories']:.0f}")
        tc2.metric("Protein", f"{totals['protein_g']:.0f}g")
        tc3.metric("Carbs", f"{totals['carbs_g']:.0f}g")
        tc4.metric("Fat", f"{totals['fat_g']:.0f}g")
        tc5.metric("Fiber", f"{totals['fiber_g']:.0f}g")

        # Nutrient gap analysis
        gaps = day_data["gap_analysis"]
        low_gaps = [g for g in gaps if g["status"] == "low"]
        if low_gaps:
            st.warning(
                "**Below 80% RDA:** " +
                ", ".join(f"{g['display_name']} ({g['pct']:.0f}%)" for g in low_gaps)
            )


# ---------------------------------------------------------------------------
# DISPLAY — WEEKLY NUTRIENT ANALYSIS
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown("### 📈 Weekly Nutrient Analysis")

rda = get_rda(age, sex, calorie_target=calorie_target)

# Build nutrient analysis table
nutrient_rows = []
weekly_avg = plan["weekly_summary"]["daily_averages"]
for nutrient, display_name in NUTRIENT_DISPLAY.items():
    actual = weekly_avg.get(nutrient, 0)
    target = rda.get(nutrient, 0)
    pct = (actual / target * 100) if target > 0 else 100
    if nutrient == "sodium_mg":
        status = "🔴 High" if pct > 100 else "✅ OK"
    else:
        status = "⚠️ Low" if pct < 80 else ("🔴 High" if pct > 120 else "✅ OK")
    nutrient_rows.append({
        "Nutrient": display_name,
        "Daily Avg": round(actual, 1),
        "RDA Target": round(target, 1),
        "% of RDA": round(pct, 1),
        "Status": status,
    })

nutrient_df = pd.DataFrame(nutrient_rows)
st.dataframe(nutrient_df, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# DISPLAY — WHY EXCLUDED (Signature Deliverable)
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown("### 🚫 Why Excluded? — Exclusion Explanations")
st.caption(f"Showing {len(exclusions):,} excluded foods with reasons")

with st.expander(f"View all {len(exclusions):,} exclusion reasons", expanded=False):
    if exclusions:
        excl_df = pd.DataFrame(exclusions, columns=["Food", "Reason"])
        st.dataframe(excl_df, use_container_width=True, hide_index=True, height=400)
    else:
        st.success("No foods were excluded! All foods in the database are safe for this profile.")


# ---------------------------------------------------------------------------
# EXPORT — CSV & PDF
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown("### 📥 Export Your Plan")

export_c1, export_c2 = st.columns(2)

# CSV Export
with export_c1:
    csv_buffer = plan_df.to_csv(index=False)
    st.download_button(
        label="📄 Download as CSV",
        data=csv_buffer,
        file_name="nutriai_7day_plan.csv",
        mime="text/csv",
        use_container_width=True,
    )

# PDF Export
with export_c2:
    try:
        from fpdf import FPDF

        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 10, "NutriAI - 7-Day Personalized Meal Plan", ln=True, align="C")
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 8, f"Generated for: Age {age}, {sex.capitalize()} | "
                 f"Conditions: {', '.join(conditions) or 'None'} | "
                 f"Diet: {diet} | Target: {calorie_target} kcal/day", ln=True, align="C")
        pdf.ln(5)

        for day_data in plan["days"]:
            pdf.set_font("Helvetica", "B", 12)
            pdf.cell(0, 8, f"Day {day_data['day']}", ln=True)
            pdf.set_font("Helvetica", "", 10)

            for meal in day_data["meals"]:
                food = meal["food"]
                desc = food.get("description", "")
                cal = food.get("calories", 0)
                protein = food.get("protein_g", 0)
                pdf.cell(0, 6, f"  {meal['meal_name']}: {desc} "
                         f"({cal:.0f} kcal, {protein:.0f}g protein)", ln=True)

            totals = day_data["daily_totals"]
            pdf.set_font("Helvetica", "I", 9)
            pdf.cell(0, 6, f"  Day total: {totals['calories']:.0f} kcal | "
                     f"P: {totals['protein_g']:.0f}g | "
                     f"C: {totals['carbs_g']:.0f}g | "
                     f"F: {totals['fat_g']:.0f}g", ln=True)
            pdf.ln(3)

        # Weekly summary
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Weekly Averages", ln=True)
        pdf.set_font("Helvetica", "", 10)
        avg = plan["weekly_summary"]["daily_averages"]
        pdf.cell(0, 6, f"Calories: {avg['calories']:.0f}/day | "
                 f"Protein: {avg['protein_g']:.0f}g | "
                 f"Carbs: {avg['carbs_g']:.0f}g | "
                 f"Fat: {avg['fat_g']:.0f}g", ln=True)

        pdf_bytes = pdf.output()
        st.download_button(
            label="📋 Download as PDF",
            data=bytes(pdf_bytes),
            file_name="nutriai_7day_plan.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
    except Exception as e:
        st.error(f"PDF generation failed: {e}")


# ---------------------------------------------------------------------------
# FOOTER
# ---------------------------------------------------------------------------
st.markdown("---")
st.caption(
    "NutriAI — BAX-423 Big Data · Spring 2026 · UC Davis GSM · "
    f"Database: {len(df):,} foods | Generation time: {gen_time}s"
)
