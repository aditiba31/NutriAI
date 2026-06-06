# 🥗 NutriAI

**AI-powered dietary planning that generates personalized 7-day meal plans in under 60 seconds — tailored to clinical conditions, allergens, and nutritional targets.**

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-Vector_Search-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)

🔗 **Live App:** [https://nutriai-dietplan.streamlit.app](https://nutriai-dietplan.streamlit.app)

📂 **GitHub:** [https://github.com/aditiba31/NutriAI](https://github.com/aditiba31/NutriAI)

---

## The Problem

People with conditions like IBS, diabetes, GERD, or hypertension spend hours trying to figure out what they can safely eat — cross-referencing medical guidelines, allergen lists, and nutritional targets manually. Dietitians are expensive, and generic meal plans ignore individual constraints.

## What NutriAI Does

NutriAI takes a user's clinical profile — conditions, allergens, dietary preferences, age, sex, and calorie target — and generates a complete 7-day, 3-meal-per-day plan that is:

- **Clinically safe** — filters against 9 medical conditions (IBS/FODMAP, GERD, Type 2 Diabetes, Hypertension, Celiac, CKD, Gout, Hypothyroid, PCOS)
- **Allergen-free** — excludes 7 allergen types with cross-contamination risk flagging
- **Nutritionally complete** — tracks 5 macros + 8 micronutrients against age/sex-specific RDA targets
- **Diverse** — no repeated meals across the week, category rotation enforced
- **Fast** — end-to-end generation in 1–4 seconds

Every excluded food comes with a source-cited explanation (e.g., *"Garlic excluded — High-FODMAP, Monash University"*).

---

## How to Run

### Prerequisites
- Python 3.10 or higher
- pip

### Setup & Run

```bash
# 1. Clone the repository
git clone https://github.com/aditiba31/NutriAI.git
cd NutriAI

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the application
cd code
streamlit run app.py
```

The app opens at **http://localhost:8501**.

### Using the App
1. Select a **test persona** from the sidebar dropdown (Priya, Ravi, Mei, or James) — or configure a custom profile
2. Set your **age, sex, conditions, allergens, diet type**, and **calorie target**
3. Click **Generate 7-Day Plan**
4. View the plan by day, review nutrient analysis, check pass criteria, and export as CSV or PDF

> The database ships pre-built with 15,636 foods. No API key needed to run.

### Rebuilding the Database (optional)

```bash
# With USDA API key
echo "USDA_API_KEY=your_key_here" > .env
python code/data_pipeline.py

# Without API key (curated fallback)
python code/data_pipeline.py --fallback
```

---

## Architecture

```
User Profile → Clinical Filter Chain → Allergen Exclusion → Safe Food Pool
     → FAISS Index Build → Bloom Filter Build → Ranked Meal Selection
          → Nutrient Gap Analysis → RDA Flagging → UI + Export
```

| Stage | What it does |
|-------|-------------|
| **Data Pipeline** | 15,636 foods from USDA FoodData Central (SR Legacy + Foundation), tagged with clinical/allergen/dietary labels, deduplicated, stored in SQLite |
| **Filter Chain** | Composable filters for 9 conditions using Monash FODMAP, GI Foundation, DASH/NHLBI, and ACG GERD guidelines |
| **FAISS Retrieval** | 13-dimensional nutrient embeddings indexed with `IndexFlatL2` for sub-millisecond nearest-neighbor candidate retrieval |
| **Bloom Filter** | O(1) probabilistic exclusion screening with configurable false-positive rate |
| **Ranking** | Multi-factor scoring: nutrient-gap fill, meal-type fit, calorie match, serving-size scaling, diversity bonus |
| **Analysis** | Per-meal and daily nutrient totals vs. NIH RDA targets by age/sex — flags anything below 80% |

### Data Sources

- [USDA FoodData Central](https://fdc.nal.usda.gov/api-guide.html) — nutrient profiles
- [Monash University FODMAP](https://www.monashfodmap.com) — IBS-safe food classification
- [NIH Dietary Reference Intakes](https://www.ncbi.nlm.nih.gov/books/NBK56068) — RDA tables
- [Glycaemic Index Foundation](https://www.glycemicindex.com) — diabetes GI values
- [NHLBI DASH Eating Plan](https://www.nhlbi.nih.gov/education/dash-eating-plan) — hypertension guidelines

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Frontend | Streamlit |
| Database | SQLite (indexed) |
| Candidate Retrieval | FAISS (`IndexFlatL2`, 13-D nutrient vectors) |
| Safety Screening | Bloom Filter (`pybloom-live`) |
| Normalization | scikit-learn `MinMaxScaler` |
| Nutrient Data | USDA FoodData Central API |
| Export | CSV + PDF (`fpdf2`) |

---

## Project Structure

```
NutriAI/
├── code/
│   ├── app.py              # Streamlit UI (entry point)
│   ├── data_pipeline.py    # USDA ingestion + curated dataset
│   ├── data_sources.py     # Clinical reference data (FODMAP, GI, DASH, GERD)
│   ├── filters.py          # Condition & allergen filter chain
│   ├── nutrients.py        # RDA tables + gap analysis
│   └── ranking.py          # FAISS + Bloom ranking engine
├── data/
│   └── nutriai_foods.db    # Pre-built food database (15,636 records)
├── requirements.txt
└── README.md
```

---

## Test Personas

| Persona | Profile | Key Constraints |
|---------|---------|-----------------|
| **Priya** | 28F · IBS · Vegetarian · Lactose Intolerant | Zero FODMAP triggers, zero dairy, iron ≥ 80% RDA |
| **Ravi** | 45M · GERD · Gluten-Free · No Pork | Zero GERD triggers, zero gluten, B12 ≥ 80% RDA |
| **Mei** | 35F · T2 Diabetes · Vegan · Tree Nut Allergy | All meals GI ≤ 55, fiber ≥ 25g/day |
| **James** | 55M · Hypertension · Pescatarian · Soy Allergy | Sodium ≤ 1500mg/day, ≥ 3 fish meals/week |

All 4 personas pass all 6 core capabilities and all persona-specific criteria.

---

## Performance

| Metric | Value |
|--------|-------|
| Plan generation time | 1–4 seconds |
| FAISS index build | 3–8 ms |
| FAISS avg query | 0.04–0.15 ms |
| Bloom filter build | 0.5–2.0 ms |
| Bloom avg check | 0.002–0.01 ms |
| Bloom false positive rate | 0.0–0.8% |
| Database size | 15,636 foods |
| Conditions supported | 9 |
| Allergen types | 7 |
