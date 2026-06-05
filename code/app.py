"""
NutriAI — Streamlit Application (v2)
======================================
Run: streamlit run code/app.py
"""
import sys
import time
from pathlib import Path
import streamlit as st
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from filters import (load_foods, apply_all_filters, PERSONAS, CLINICAL_FILTERS,
                     CONDITION_DISPLAY_NAMES, validate_pass_criteria)
from ranking import generate_plan_with_faiss, MEAL_NAMES
from nutrients import get_rda, analyze_gaps, NUTRIENT_DISPLAY
from data_sources import DATA_SOURCE_CITATIONS

st.set_page_config(page_title="NutriAI — Diet Plan Builder", page_icon="🥗",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""<style>
.main-header { font-size: 2.2rem; font-weight: 700; color: #2B5797; margin-bottom: 0.2rem; }
.sub-header { font-size: 1rem; color: #666; margin-bottom: 1.5rem; }
</style>""", unsafe_allow_html=True)

@st.cache_data
def get_food_database():
    return load_foods()

# ── SIDEBAR ──
st.sidebar.markdown("## 🥗 NutriAI")
st.sidebar.markdown("Personalized Diet Plan Builder")
st.sidebar.markdown("---")

preset = st.sidebar.selectbox("Quick start — load a test persona",
                               ["Custom"] + list(PERSONAS.keys()))

if preset != "Custom":
    p = PERSONAS[preset]
    d_age, d_sex = p["age"], p["sex"]
    d_cond, d_allerg = p["conditions"], p["allergens"]
    d_diet, d_cal = p["diet"], p["calorie_target"]
    d_pork = p.get("no_pork", False)
else:
    d_age, d_sex = 30, "female"
    d_cond, d_allerg = [], []
    d_diet, d_cal, d_pork = "none", 2000, False

st.sidebar.markdown("### Personal Info")
age = st.sidebar.slider("Age", 18, 80, d_age)
sex = st.sidebar.radio("Sex", ["female", "male"], index=0 if d_sex == "female" else 1)

st.sidebar.markdown("### Clinical Conditions")
# All 10 conditions available
condition_keys = [k for k in CLINICAL_FILTERS.keys() if k != "acid_reflux"]  # skip alias
conditions = st.sidebar.multiselect(
    "Select conditions", options=condition_keys, default=d_cond,
    format_func=lambda x: CONDITION_DISPLAY_NAMES.get(x, x))

st.sidebar.markdown("### Allergens")
allergens = st.sidebar.multiselect(
    "Select allergens to exclude",
    options=["dairy", "gluten", "soy", "tree_nuts", "eggs", "shellfish", "peanuts"],
    default=d_allerg,
    format_func=lambda x: {"dairy":"🥛 Dairy/Lactose","gluten":"🌾 Gluten",
        "soy":"🫘 Soy","tree_nuts":"🥜 Tree Nuts","eggs":"🥚 Eggs",
        "shellfish":"🦐 Shellfish","peanuts":"🥜 Peanuts"}.get(x,x))

st.sidebar.markdown("### Diet Preference")
diet_opts = ["none","vegetarian","vegan","pescatarian"]
diet = st.sidebar.selectbox("Diet type", options=diet_opts,
    index=diet_opts.index(d_diet) if d_diet in diet_opts else 0,
    format_func=lambda x: {"none":"No restriction","vegetarian":"🥬 Vegetarian",
        "vegan":"🌱 Vegan","pescatarian":"🐟 Pescatarian"}.get(x,x))

st.sidebar.markdown("### Additional Rules")
no_pork = st.sidebar.checkbox("No pork", value=d_pork)

st.sidebar.markdown("### Calorie Target")
calorie_target = st.sidebar.slider("Daily calories (kcal)", 1200, 3500, d_cal, step=100)

st.sidebar.markdown("---")
generate_btn = st.sidebar.button("🚀 Generate 7-Day Plan", use_container_width=True, type="primary")

# ── MAIN ──
st.markdown('<p class="main-header">🥗 NutriAI — Automated Diet Plan Builder</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Personalized 7-day meal plans tailored to clinical conditions, '
            'allergens, dietary preferences, and nutritional targets.</p>', unsafe_allow_html=True)

if not generate_btn:
    st.info("👈 Configure your profile in the sidebar and click **Generate 7-Day Plan** to start.", icon="ℹ️")
    st.markdown("### Test Personas")
    cols = st.columns(4)
    for i, (name, pp) in enumerate(PERSONAS.items()):
        with cols[i]:
            st.markdown(f"**{name}**")
            st.caption(pp["description"])

    # Data sources section on landing page
    st.markdown("---")
    st.markdown("### 📚 Data Sources")
    for key, src in DATA_SOURCE_CITATIONS.items():
        st.markdown(f"**{src['name']}** — {src['usage']}  \n"
                    f"[{src['url']}]({src['url']})")
    st.stop()

# ── GENERATE ──
df = get_food_database()

with st.spinner("Filtering foods for safety..."):
    safe_foods, exclusions = apply_all_filters(
        df, conditions=conditions, allergens=allergens,
        diet=diet, calorie_target=calorie_target, no_pork=no_pork)

if len(safe_foods) < 21:
    st.error(f"⚠️ Only **{len(safe_foods)}** safe foods remain — need at least 21. "
             "Try loosening constraints.", icon="🚫")
    st.stop()

with st.spinner("Generating your personalized 7-day meal plan..."):
    plan = generate_plan_with_faiss(
        safe_foods, exclusions,
        age=age, sex=sex, calorie_target=calorie_target,
        conditions=conditions, allergens=allergens, diet=diet, no_pork=no_pork
    )

gen_time = plan["generation_time_s"]

# ── SUMMARY METRICS ──
st.markdown("---")
st.markdown("### 📊 Plan Summary")
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Generation Time", f"{gen_time}s",
          delta="Under 60s ✅" if gen_time < 60 else "Over 60s ❌")
m2.metric("Safe Foods", f"{len(safe_foods):,}")
m3.metric("Foods Excluded", f"{len(exclusions):,}")
m4.metric("Unique Meals", f"{plan['unique_foods']}/{plan['total_meals']}")
m5.metric("Avg Calories/Day", f"{plan['weekly_summary']['daily_averages']['calories']:.0f}")

# ── OPTIMIZATION ENGINE ──
if "benchmarks" in plan:
    bm = plan["benchmarks"]
    st.markdown("---")
    st.markdown("### ⚡ Optimization Engine")
    t1, t2 = st.columns(2)
    with t1:
        st.markdown("**Smart Retrieval — FAISS**")
        st.caption("Vector similarity search over nutrient profiles")
        f1, f2, f3 = st.columns(3)
        f1.metric("Index Build", f"{bm['faiss_build_time_ms']:.1f}ms")
        f2.metric("Avg Query", f"{bm['faiss_avg_query_ms']:.2f}ms")
        f3.metric("Foods Indexed", f"{bm['faiss_index_size']:,}")
    with t2:
        st.markdown("**Safety Filter — Bloom Filter**")
        st.caption("Probabilistic exclusion screening")
        b1, b2, b3 = st.columns(3)
        b1.metric("Rules Loaded", f"{bm['bloom_n_excluded']:,}")
        b2.metric("Avg Check", f"{bm['bloom_avg_check_ms']:.2f}ms")
        b3.metric("False Positive Rate", f"{bm['bloom_false_positive_rate']:.2f}%")

# ── 7-DAY PLAN ──
st.markdown("---")
st.markdown("### 🗓️ Your 7-Day Meal Plan")

plan_rows = []
for day_data in plan["days"]:
    for meal in day_data["meals"]:
        food = meal["food"]
        plan_rows.append({
            "Day": day_data["day"], "Meal": meal["meal_name"],
            "Food": food.get("description",""), "Category": food.get("food_category",""),
            "Serving": f"{food.get('serving_g','?')}g",
            "Calories": round(food.get("calories",0)),
            "Protein (g)": round(food.get("protein_g",0),1),
            "Carbs (g)": round(food.get("carbs_g",0),1),
            "Fat (g)": round(food.get("fat_g",0),1),
            "Fiber (g)": round(food.get("fiber_g",0),1),
        })
plan_df = pd.DataFrame(plan_rows)

day_tabs = st.tabs([f"Day {d}" for d in range(1, 8)])
for i, tab in enumerate(day_tabs):
    day_data = plan["days"][i]
    with tab:
        day_df = plan_df[plan_df["Day"]==i+1][["Meal","Food","Serving","Calories",
                                                 "Protein (g)","Carbs (g)","Fat (g)","Fiber (g)"]]
        st.dataframe(day_df, use_container_width=True, hide_index=True)
        totals = day_data["daily_totals"]
        tc1, tc2, tc3, tc4, tc5 = st.columns(5)
        tc1.metric("Calories", f"{totals['calories']:.0f}")
        tc2.metric("Protein", f"{totals['protein_g']:.0f}g")
        tc3.metric("Carbs", f"{totals['carbs_g']:.0f}g")
        tc4.metric("Fat", f"{totals['fat_g']:.0f}g")
        tc5.metric("Fiber", f"{totals['fiber_g']:.0f}g")
        low_gaps = [g for g in day_data["gap_analysis"] if g["status"]=="low"]
        if low_gaps:
            st.warning("**Below 80% RDA:** " +
                       ", ".join(f"{g['display_name']} ({g['pct']:.0f}%)" for g in low_gaps))

# ── WEEKLY NUTRIENT ANALYSIS ──
st.markdown("---")
st.markdown("### 📈 Weekly Nutrient Analysis")
rda = get_rda(age, sex, calorie_target=calorie_target)
nutrient_rows = []
weekly_avg = plan["weekly_summary"]["daily_averages"]
for nutrient, display_name in NUTRIENT_DISPLAY.items():
    actual = weekly_avg.get(nutrient,0)
    target = rda.get(nutrient,0)
    pct = (actual/target*100) if target > 0 else 100
    if nutrient == "sodium_mg":
        status = "🔴 High" if pct > 100 else "✅ OK"
    else:
        status = "⚠️ Low" if pct < 80 else ("🔴 High" if pct > 120 else "✅ OK")
    nutrient_rows.append({"Nutrient":display_name,"Daily Avg":round(actual,1),
                          "RDA Target":round(target,1),"% of RDA":round(pct,1),"Status":status})
st.dataframe(pd.DataFrame(nutrient_rows), use_container_width=True, hide_index=True)

# ── PASS CRITERIA (if test persona selected) ──
if preset != "Custom":
    st.markdown("---")
    st.markdown(f"### ✅ Pass Criteria — {preset}")
    criteria_results = validate_pass_criteria(plan, preset, rda)
    if criteria_results:
        crit_rows = []
        for cr in criteria_results:
            icon = "✅" if cr["passed"] else "❌"
            crit_rows.append({"Status":icon, "Criterion":cr["description"], "Detail":cr["detail"]})
        st.dataframe(pd.DataFrame(crit_rows), use_container_width=True, hide_index=True)
        passed = sum(1 for c in criteria_results if c["passed"])
        total = len(criteria_results)
        if passed == total:
            st.success(f"All {total} criteria passed!")
        else:
            st.warning(f"{passed}/{total} criteria passed. Review failures above.")

# ── WHY EXCLUDED ──
st.markdown("---")
st.markdown("### 🚫 Why Excluded? — Exclusion Explanations")
st.caption(f"{len(exclusions):,} foods excluded with source-cited reasons")
with st.expander(f"View all {len(exclusions):,} exclusion reasons", expanded=False):
    if exclusions:
        excl_df = pd.DataFrame(exclusions, columns=["Food","Reason"])
        st.dataframe(excl_df, use_container_width=True, hide_index=True, height=400)
    else:
        st.success("No foods excluded!")

# ── EXPORT ──
st.markdown("---")
st.markdown("### 📥 Export Your Plan")
export_c1, export_c2 = st.columns(2)
with export_c1:
    st.download_button("📄 Download as CSV", plan_df.to_csv(index=False),
                       "nutriai_7day_plan.csv", "text/csv", use_container_width=True)
with export_c2:
    try:
        from fpdf import FPDF
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.set_font("Helvetica","B",16)
        pdf.cell(0,10,"NutriAI - 7-Day Personalized Meal Plan",ln=True,align="C")
        pdf.set_font("Helvetica","",10)
        pdf.cell(0,8,f"Age {age}, {sex.capitalize()} | Conditions: {', '.join(conditions) or 'None'} | "
                 f"Diet: {diet} | Target: {calorie_target} kcal/day",ln=True,align="C")
        pdf.ln(5)
        for day_data in plan["days"]:
            pdf.set_font("Helvetica","B",12)
            pdf.cell(0,8,f"Day {day_data['day']}",ln=True)
            pdf.set_font("Helvetica","",10)
            for meal in day_data["meals"]:
                f = meal["food"]
                pdf.cell(0,6,f"  {meal['meal_name']}: {f.get('description','')} "
                         f"({f.get('calories',0):.0f} kcal, {f.get('serving_g','?')}g)",ln=True)
            t = day_data["daily_totals"]
            pdf.set_font("Helvetica","I",9)
            pdf.cell(0,6,f"  Total: {t['calories']:.0f} kcal | P:{t['protein_g']:.0f}g | "
                     f"C:{t['carbs_g']:.0f}g | F:{t['fat_g']:.0f}g",ln=True)
            pdf.ln(3)
        pdf_bytes = pdf.output()
        st.download_button("📋 Download as PDF", bytes(pdf_bytes),
                           "nutriai_7day_plan.pdf", "application/pdf", use_container_width=True)
    except Exception as e:
        st.error(f"PDF generation failed: {e}")

# ── DATA SOURCES ──
st.markdown("---")
st.markdown("### 📚 Data Sources")
src_cols = st.columns(3)
for i, (key, src) in enumerate(DATA_SOURCE_CITATIONS.items()):
    with src_cols[i % 3]:
        st.markdown(f"**{src['name']}**")
        st.caption(src['usage'])

# ── FOOTER ──
st.markdown("---")
st.caption(f"NutriAI — Personalized Nutrition Planning · "
           f"Database: {len(df):,} foods | Generation time: {gen_time}s | "
           f"{len(CLINICAL_FILTERS)} conditions supported")
