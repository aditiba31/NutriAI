# NutriAI — Automated Diet Plan Builder

**BAX-423 Big Data · Spring 2026 · UC Davis GSM**

NutriAI generates a personalized 7-day meal plan in under 60 seconds, tailored to clinical conditions (IBS, GERD, Type 2 Diabetes, Hypertension), allergens, dietary preferences, and nutritional targets.

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/nutriai.git
cd nutriai

# 2. Install dependencies (Python 3.11+ required)
pip install -r requirements.txt

# 3. Copy environment template and add your USDA API key
cp .env.example .env
# Edit .env and add your key from https://fdc.nal.usda.gov/api-key-signup.html

# 4. (Optional) Rebuild the food database from scratch
python code/data_pipeline.py

# 5. Run the app
streamlit run code/app.py
```

The app ships with a pre-built offline snapshot (`data/nutriai_foods.db`) so it runs without API access.

## Live Demo

🔗 **[Live App URL]** _(Streamlit Cloud)_

## Project Structure

```
nutriai/
├── code/
│   ├── app.py                 # Streamlit UI (entry point)
│   ├── data_pipeline.py       # USDA data ingestion & preprocessing
│   ├── filters.py             # Clinical, allergen, and diet filters
│   ├── ranking.py             # FAISS embeddings + candidate ranking
│   ├── meal_planner.py        # 7-day plan generation engine
│   ├── nutrients.py           # RDA tables + nutrient gap analysis
│   └── export.py              # PDF/CSV export
├── data/
│   └── nutriai_foods.db       # Offline SQLite snapshot (≥10,000 items)
├── brief.pdf                  # Technical brief (≤4 pages)
├── prompts.md                 # AI prompts used
├── requirements.txt
├── .env.example
└── README.md
```

## BAX-423 Techniques

1. **FAISS (Facebook AI Similarity Search)** — Nutrient-vector embeddings for fast candidate food retrieval during plan generation.
2. **Bloom Filter** — Probabilistic membership testing for O(1) allergen/exclusion checking during candidate screening.

## Test Personas

| Persona | Condition | Diet | Key Allergen |
|---------|-----------|------|-------------|
| Priya   | IBS       | Vegetarian | Lactose |
| Ravi    | GERD      | Non-Veg    | Gluten  |
| Mei     | T2 Diabetes | Vegan   | Tree Nuts |
| James   | Hypertension | Pescatarian | Soy |
