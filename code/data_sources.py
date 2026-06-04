"""
NutriAI — Data Sources Module
===============================
Curated reference data from all required data sources,
with proper attribution and citations.

Sources:
    1. USDA FoodData Central (https://fdc.nal.usda.gov)
       → Food nutrient profiles (integrated via API in data_pipeline.py)

    2. Monash University Low-FODMAP Diet
       (https://www.monashfodmap.com)
       → FODMAP classification for IBS management

    3. NIH Dietary Reference Intakes
       (https://www.ncbi.nlm.nih.gov/books/NBK56068)
       → RDA tables (integrated in nutrients.py)

    4. Glycaemic Index Foundation
       (https://www.glycemicindex.com)
       → GI values for diabetes-safe meal planning

    5. NHLBI DASH Eating Plan
       (https://www.nhlbi.nih.gov/education/dash-eating-plan)
       → Sodium/potassium/magnesium guidelines for hypertension
"""

# ═══════════════════════════════════════════════════════════════
# SOURCE 2: MONASH UNIVERSITY LOW-FODMAP CLASSIFICATIONS
# Reference: Monash University FODMAP Diet App & Research
# ═══════════════════════════════════════════════════════════════

# High-FODMAP foods (AVOID for IBS) — categorized by FODMAP type
# Fructans, GOS, Lactose, Fructose, Polyols
MONASH_HIGH_FODMAP = {
    "fructans": [
        "garlic", "onion", "leek", "shallot", "scallion",
        "wheat", "rye", "barley",
        "artichoke", "asparagus", "beetroot", "beet",
        "brussels sprout", "broccoli stem",
        "cabbage savoy", "fennel",
        "chicory", "inulin", "dandelion greens",
        "watermelon", "persimmon", "white peach", "rambutan",
        "dried fruit", "date", "fig", "prune",
        "pistachio", "cashew",
        "chamomile tea", "chicory root",
    ],
    "galacto_oligosaccharides": [
        "chickpea", "lentil", "kidney bean", "black bean",
        "lima bean", "navy bean", "pinto bean", "split pea",
        "baked bean", "soybean", "edamame",
        "hummus", "falafel",
    ],
    "lactose": [
        "milk", "yogurt", "ice cream", "custard",
        "cream cheese", "ricotta", "cottage cheese",
        "condensed milk", "evaporated milk",
        "soft cheese", "cream",
    ],
    "excess_fructose": [
        "apple", "pear", "mango", "watermelon",
        "cherry", "boysenberry", "fig",
        "honey", "agave", "high fructose corn syrup",
        "fruit juice concentrate",
    ],
    "polyols": [
        "mushroom", "cauliflower", "snow pea", "sugar snap",
        "apricot", "avocado", "blackberry", "cherry",
        "lychee", "nectarine", "peach", "plum", "prune",
        "sorbitol", "mannitol", "xylitol", "maltitol",
    ],
}

# Low-FODMAP foods (SAFE for IBS) — Monash University green-rated
MONASH_LOW_FODMAP_SAFE = [
    # Proteins
    "chicken", "turkey", "beef", "lamb", "pork", "fish", "seafood",
    "egg", "tofu firm", "tempeh",
    # Grains
    "rice", "oat", "quinoa", "corn", "buckwheat", "millet",
    "potato", "sweet potato", "polenta",
    "rice noodle", "rice cake", "cornmeal",
    # Vegetables
    "carrot", "zucchini", "cucumber", "lettuce", "spinach",
    "bell pepper", "green bean", "eggplant", "tomato",
    "bok choy", "kale", "swiss chard", "turnip", "radish",
    "bamboo shoot", "bean sprout", "chive", "ginger",
    "olive", "pumpkin", "squash",
    # Fruits
    "banana", "blueberry", "strawberry", "raspberry",
    "grape", "orange", "mandarin", "kiwi",
    "pineapple", "cantaloupe", "honeydew",
    "papaya", "passionfruit", "dragon fruit",
    # Nuts & Seeds (safe ones)
    "almond", "walnut", "pecan", "macadamia",
    "pumpkin seed", "sunflower seed", "sesame seed",
    "chia seed", "flaxseed", "hemp seed", "pine nut",
    # Dairy alternatives
    "almond milk", "rice milk", "coconut milk",
    "lactose-free milk", "hard cheese", "cheddar",
    "parmesan", "mozzarella", "brie",
]

# Flatten high-FODMAP into a single list for matching
ALL_HIGH_FODMAP_KEYWORDS = []
for category, foods in MONASH_HIGH_FODMAP.items():
    ALL_HIGH_FODMAP_KEYWORDS.extend(foods)


# ═══════════════════════════════════════════════════════════════
# SOURCE 4: GLYCAEMIC INDEX DATABASE
# Reference: https://www.glycemicindex.com
# GI Scale: Low ≤ 55, Medium 56-69, High ≥ 70
# ═══════════════════════════════════════════════════════════════

# GI values for common foods (per glycemicindex.com)
GI_DATABASE = {
    # LOW GI (≤ 55) — Safe for T2 Diabetes
    "low": {
        "foods": [
            "lentil", "chickpea", "kidney bean", "black bean",
            "navy bean", "pinto bean", "lima bean", "soybean",
            "split pea", "mung bean",
            "barley", "bulgur", "quinoa", "buckwheat",
            "whole wheat pasta", "oat", "oatmeal",
            "sweet potato", "yam", "taro",
            "apple", "pear", "orange", "peach", "plum",
            "strawberry", "blueberry", "raspberry", "blackberry",
            "cherry", "grapefruit", "kiwi", "mango",
            "avocado", "coconut",
            "broccoli", "cauliflower", "spinach", "kale",
            "lettuce", "cabbage", "zucchini", "cucumber",
            "bell pepper", "tomato", "carrot", "eggplant",
            "asparagus", "green bean", "mushroom", "onion",
            "celery", "radish", "artichoke",
            "almond", "walnut", "peanut", "cashew",
            "chia seed", "flaxseed", "pumpkin seed",
            "milk", "yogurt", "cheese",
            "egg", "chicken", "fish", "beef", "pork", "lamb",
            "tofu", "tempeh",
            "olive oil", "coconut oil",
            "hummus", "tahini",
            "brown rice",
            "soba noodle",
        ],
        "gi_range": (0, 55),
    },
    # MEDIUM GI (56-69)
    "medium": {
        "foods": [
            "banana", "grape", "papaya", "pineapple",
            "raisin", "dried fig", "dried apricot",
            "basmati rice", "couscous",
            "whole wheat bread", "rye bread", "pita bread",
            "corn tortilla", "popcorn",
            "beet", "sweet corn", "pumpkin",
            "honey", "maple syrup",
        ],
        "gi_range": (56, 69),
    },
    # HIGH GI (≥ 70) — AVOID for T2 Diabetes
    "high": {
        "foods": [
            "white rice", "white bread", "bagel",
            "white potato", "baked potato", "mashed potato",
            "french fries", "potato chip",
            "instant oatmeal", "corn flakes", "rice cake",
            "rice cracker", "pretzels",
            "watermelon", "dates",
            "candy", "sugar", "glucose", "maltose",
            "soda", "sports drink", "fruit juice",
            "white pasta", "instant noodle",
            "polenta", "tapioca",
            "doughnut", "waffle", "pancake",
            "jasmine rice", "sticky rice",
        ],
        "gi_range": (70, 100),
    },
}

# Build lookup for quick GI classification
GI_HIGH_KEYWORDS = GI_DATABASE["high"]["foods"]
GI_MEDIUM_KEYWORDS = GI_DATABASE["medium"]["foods"]
GI_LOW_KEYWORDS = GI_DATABASE["low"]["foods"]


# ═══════════════════════════════════════════════════════════════
# SOURCE 5: NHLBI DASH EATING PLAN
# Reference: https://www.nhlbi.nih.gov/education/dash-eating-plan
# ═══════════════════════════════════════════════════════════════

DASH_GUIDELINES = {
    "description": "Dietary Approaches to Stop Hypertension",
    "source": "NHLBI (National Heart, Lung, and Blood Institute)",
    "url": "https://www.nhlbi.nih.gov/education/dash-eating-plan",

    # Daily limits for 2,000 kcal diet
    "daily_limits": {
        "sodium_mg": 1500,          # Strict limit for hypertension
        "sodium_mg_moderate": 2300,  # General population limit
        "saturated_fat_g": 14,      # ≤6% of calories
        "added_sugar_g": 25,        # Limit sweets
        "alcohol_drinks": 1,        # For women; 2 for men
    },

    # Daily targets (DASH emphasizes these)
    "daily_targets": {
        "potassium_mg": 4700,       # Higher than standard RDA
        "magnesium_mg": 500,        # Higher than standard RDA
        "calcium_mg": 1250,         # Higher than standard RDA
        "fiber_g": 30,              # Higher than standard RDA
    },

    # DASH food group servings (per day, 2000 kcal)
    "servings_per_day": {
        "grains": "6-8 (mostly whole grains)",
        "vegetables": "4-5",
        "fruits": "4-5",
        "dairy_lowfat": "2-3",
        "lean_meat_fish": "≤6 oz",
        "nuts_seeds_legumes": "4-5 per week",
        "fats_oils": "2-3",
        "sweets": "≤5 per week",
    },
}

# High-sodium foods to flag for hypertension (DASH exclusions)
DASH_HIGH_SODIUM_FOODS = [
    "soy sauce", "teriyaki", "miso", "fish sauce",
    "pickle", "pickled", "sauerkraut", "kimchi",
    "cured", "smoked salmon", "smoked", "jerky",
    "salted", "brine", "brined",
    "bacon", "ham", "salami", "prosciutto", "pepperoni",
    "hot dog", "sausage", "deli meat", "lunch meat",
    "canned soup", "bouillon", "stock cube", "broth cube",
    "anchovy", "anchovies",
    "parmesan", "feta", "blue cheese", "processed cheese",
    "salted butter", "margarine",
    "ketchup", "bbq sauce", "worcestershire",
    "salad dressing", "ranch", "italian dressing",
    "instant noodle", "ramen seasoning",
    "cracker", "pretzel", "chip", "corn chip",
    "frozen dinner", "tv dinner", "microwave meal",
]

# DASH-recommended high-potassium foods
DASH_HIGH_POTASSIUM_FOODS = [
    "banana", "potato", "sweet potato", "spinach",
    "avocado", "tomato", "orange", "cantaloupe",
    "lima bean", "kidney bean", "lentil", "chickpea",
    "salmon", "tuna", "halibut", "cod",
    "yogurt", "milk",
    "squash", "beet", "broccoli",
    "pomegranate", "kiwi", "apricot",
]


# ═══════════════════════════════════════════════════════════════
# GERD TRIGGER FOODS
# Reference: American College of Gastroenterology (ACG)
# ═══════════════════════════════════════════════════════════════

GERD_TRIGGER_FOODS = {
    "acidic": [
        "tomato", "marinara", "ketchup", "salsa",
        "orange", "lemon", "lime", "grapefruit", "citrus",
        "vinegar", "pickle", "sauerkraut", "kimchi",
        "cranberry", "pineapple",
    ],
    "caffeine": [
        "coffee", "espresso", "cappuccino", "latte",
        "black tea", "green tea", "chai tea", "iced tea",
        "energy drink", "cola", "soda",
    ],
    "chocolate": [
        "chocolate", "cocoa", "cacao", "brownie",
        "chocolate chip", "hot chocolate", "mocha",
    ],
    "spicy": [
        "chili", "jalapeno", "cayenne", "habanero",
        "ghost pepper", "sriracha", "hot sauce", "tabasco",
        "wasabi", "horseradish", "black pepper heavy",
        "curry paste", "chili flake", "red pepper flake",
    ],
    "fried_fatty": [
        "fried", "deep-fried", "french fries", "onion ring",
        "fried chicken", "tempura", "fritter",
    ],
    "mint": [
        "peppermint", "spearmint", "mint",
    ],
    "allium": [
        "garlic", "onion", "leek", "shallot",
    ],
    "carbonated": [
        "soda", "carbonated", "sparkling water", "tonic",
    ],
    "alcohol": [
        "wine", "beer", "vodka", "whiskey", "rum",
        "tequila", "cocktail", "alcohol",
    ],
}

ALL_GERD_TRIGGER_KEYWORDS = []
for category, foods in GERD_TRIGGER_FOODS.items():
    ALL_GERD_TRIGGER_KEYWORDS.extend(foods)


# ═══════════════════════════════════════════════════════════════
# ALLERGEN DATABASES
# ═══════════════════════════════════════════════════════════════

ALLERGEN_KEYWORDS = {
    "dairy": [
        "milk", "cheese", "yogurt", "cream", "butter", "whey",
        "casein", "lactose", "ghee", "ice cream", "custard",
        "pudding", "ricotta", "mozzarella", "parmesan", "cheddar",
        "brie", "camembert", "gouda", "swiss", "feta",
        "cottage cheese", "sour cream", "cream cheese",
        "condensed milk", "evaporated milk", "buttermilk",
        "paneer", "kefir", "quark",
    ],
    "gluten": [
        "wheat", "barley", "rye", "bread", "pasta", "noodle",
        "flour", "tortilla", "couscous", "bulgur", "semolina",
        "cracker", "cookie", "cake", "pastry", "muffin",
        "cereal", "granola bar", "pretzel", "pizza",
        "soy sauce", "teriyaki", "seitan", "beer",
        "pancake", "waffle", "croissant", "bagel", "pita",
        "soba", "udon", "orzo",
    ],
    "soy": [
        "soy", "soya", "tofu", "tempeh", "edamame", "miso",
        "soy sauce", "soy milk", "soy protein", "soybean",
        "teriyaki", "tamari", "natto",
    ],
    "tree_nuts": [
        "almond", "walnut", "cashew", "pistachio", "pecan",
        "hazelnut", "macadamia", "brazil nut", "pine nut",
        "chestnut", "praline", "marzipan", "nougat",
        "nut butter", "nut milk", "almond milk",
        "almond flour", "coconut",  # Classified as tree nut by FDA
    ],
    "eggs": [
        "egg", "eggs", "omelet", "omelette", "frittata", "quiche",
        "meringue", "custard", "mayonnaise", "aioli",
    ],
    "shellfish": [
        "shrimp", "prawn", "crab", "lobster", "crawfish",
        "oyster", "mussel", "clam", "scallop",
        "squid", "octopus", "calamari",
    ],
    "peanuts": [
        "peanut", "peanut butter", "groundnut",
    ],
}


# ═══════════════════════════════════════════════════════════════
# MEAL TYPE CLASSIFICATION
# Used to ensure breakfast foods at breakfast, etc.
# ═══════════════════════════════════════════════════════════════

MEAL_TYPE_FOODS = {
    "breakfast": [
        "oat", "oatmeal", "cereal", "granola", "muesli",
        "pancake", "waffle", "french toast",
        "egg", "scrambled", "omelet", "frittata",
        "toast", "bread", "bagel", "muffin",
        "yogurt", "smoothie", "fruit",
        "banana", "berry", "apple",
        "milk", "almond milk", "oat milk",
        "chia seed", "flaxseed",
        "avocado toast",
        "breakfast",
    ],
    "lunch_dinner": [
        "chicken", "turkey", "beef", "lamb", "pork",
        "salmon", "tuna", "cod", "tilapia", "shrimp",
        "rice", "quinoa", "pasta", "noodle",
        "salad", "soup", "stew", "curry", "stir-fry",
        "sandwich", "wrap", "bowl",
        "broccoli", "spinach", "kale",
        "bean", "lentil", "chickpea", "tofu", "tempeh",
        "potato", "sweet potato",
        "grilled", "roasted", "baked", "braised",
        "lunch", "dinner",
    ],
    "snack": [
        "nut", "seed", "trail mix",
        "fruit", "apple", "banana", "orange",
        "carrot stick", "celery",
        "hummus", "guacamole",
        "yogurt", "cheese",
        "granola bar", "energy bar",
        "popcorn", "cracker",
        "smoothie",
        "snack",
    ],
}


# ═══════════════════════════════════════════════════════════════
# ADDITIONAL CLINICAL CONDITIONS (4 more beyond base 4)
# ═══════════════════════════════════════════════════════════════

ADDITIONAL_CONDITIONS = {
    "celiac": {
        "name": "Celiac Disease",
        "description": "Autoimmune disorder requiring strict gluten avoidance",
        "exclude_keywords": [
            "wheat", "barley", "rye", "spelt", "kamut",
            "bread", "pasta", "couscous", "bulgur",
            "flour", "semolina", "malt", "brewer's yeast",
            "soy sauce", "seitan", "beer",
            "cracker", "cookie", "cake", "pastry",
            "cereal", "granola", "pretzel",
            "pancake", "waffle", "croissant", "bagel",
            "soba", "udon", "orzo",
        ],
        "flag_column": "contains_gluten",
        "reason": "Contains gluten — unsafe for celiac disease (strict avoidance required)",
    },
    "ckd": {
        "name": "Chronic Kidney Disease",
        "description": "Requires limiting potassium, phosphorus, and sodium",
        "high_risk_keywords": [
            "banana", "potato", "tomato", "orange",
            "spinach", "avocado", "chocolate",
            "dairy", "cheese", "yogurt", "milk",
            "nuts", "seeds", "beans", "lentil",
            "bran", "whole wheat",
            "processed meat", "cured", "deli",
            "soda", "cola",
        ],
        "reason": "High in potassium/phosphorus — restricted for chronic kidney disease",
    },
    "gout": {
        "name": "Gout",
        "description": "Requires limiting high-purine foods",
        "high_risk_keywords": [
            "organ meat", "liver", "kidney", "heart",
            "anchovy", "sardine", "mackerel", "herring",
            "mussel", "scallop", "trout",
            "bacon", "veal", "venison", "turkey",
            "beer", "alcohol", "wine",
            "gravy", "bouillon", "broth",
            "yeast", "nutritional yeast",
        ],
        "reason": "High-purine food — may trigger gout flare-ups",
    },
    "hypothyroid": {
        "name": "Hypothyroidism",
        "description": "Requires limiting goitrogens and ensuring iodine intake",
        "high_risk_keywords": [
            "soy", "tofu", "tempeh", "edamame", "soy milk",
            "raw broccoli", "raw cauliflower", "raw cabbage",
            "raw kale", "raw brussels sprout", "raw bok choy",
            "raw spinach", "raw collard",
            "millet",
            "cassava", "sweet potato raw",
            "raw turnip", "raw radish",
        ],
        "reason": "Contains goitrogens — may interfere with thyroid function",
    },
    "pcos": {
        "name": "PCOS (Polycystic Ovary Syndrome)",
        "description": "Requires low GI, anti-inflammatory diet",
        "high_risk_keywords": [
            "white rice", "white bread", "white pasta",
            "sugar", "candy", "soda", "juice",
            "pastry", "cake", "cookie", "doughnut",
            "fried", "deep-fried", "french fries",
            "processed meat", "hot dog", "sausage",
            "margarine", "shortening",
            "instant noodle", "instant oatmeal",
        ],
        "reason": "High GI / inflammatory food — may worsen PCOS symptoms",
    },
    "acid_reflux": {
        "name": "Acid Reflux (GERD)",
        "description": "Alias for GERD — same trigger foods",
        "alias": "gerd",
    },
}


# ═══════════════════════════════════════════════════════════════
# DIETARY PATTERNS (expanded)
# ═══════════════════════════════════════════════════════════════

MEAT_KEYWORDS = [
    "chicken", "turkey", "duck", "beef", "pork", "lamb",
    "venison", "bison", "veal", "goat", "rabbit",
    "bacon", "sausage", "ham", "salami", "prosciutto",
    "steak", "ground meat", "meatball", "meatloaf",
    "ribs", "drumstick", "thigh", "breast", "wing",
    "liver", "kidney", "heart", "tongue",
    "hot dog", "deli meat", "lunch meat", "pepperoni",
]

PORK_KEYWORDS = [
    "pork", "bacon", "ham", "prosciutto", "salami",
    "pepperoni", "sausage pork", "pork chop", "pork loin",
    "pork belly", "pork tenderloin", "pork ribs",
    "pulled pork", "carnitas",
]

FISH_SEAFOOD_KEYWORDS = [
    "salmon", "tuna", "cod", "tilapia", "halibut", "trout",
    "sardine", "mackerel", "anchovy", "catfish", "bass",
    "swordfish", "mahi", "snapper", "herring",
    "shrimp", "prawn", "crab", "lobster", "oyster", "mussel",
    "clam", "scallop", "squid", "octopus", "calamari",
    "fish", "seafood",
]

EGG_KEYWORDS = [
    "egg", "eggs", "omelet", "omelette", "frittata", "quiche",
    "meringue", "custard", "mayonnaise", "aioli",
]


# ═══════════════════════════════════════════════════════════════
# PERSONA PASS CRITERIA (from project instructions)
# ═══════════════════════════════════════════════════════════════

PERSONA_PASS_CRITERIA = {
    "Priya": {
        "checks": [
            ("zero_high_fodmap", "Zero high-FODMAP trigger foods"),
            ("zero_dairy", "Zero dairy in all meals"),
            ("all_meatless", "All 7 days meatless (eggs OK)"),
            ("iron_80pct_rda", "Iron ≥ 80% RDA daily"),
        ],
    },
    "Ravi": {
        "checks": [
            ("zero_gerd_triggers", "Zero GERD trigger foods"),
            ("zero_gluten", "Zero gluten in all meals"),
            ("no_pork", "No pork products"),
            ("diversity_gte_07", "Diversity score ≥ 0.7"),
            ("b12_80pct_rda", "B12 ≥ 80% RDA daily"),
        ],
    },
    "Mei": {
        "checks": [
            ("all_low_gi", "All meals GI ≤ 55"),
            ("zero_animal", "Zero animal products (vegan)"),
            ("zero_tree_nuts", "Zero tree nuts"),
            ("fiber_gte_25g", "Fibre ≥ 25g/day"),
        ],
    },
    "James": {
        "checks": [
            ("sodium_lte_1500", "Sodium ≤ 1,500 mg/day every day"),
            ("zero_soy", "Zero soy products"),
            ("fish_3_meals", "At least 3 fish/seafood meals per week"),
            ("potassium_80pct_rda", "Potassium ≥ 80% RDA daily"),
        ],
    },
}


# ═══════════════════════════════════════════════════════════════
# DATA SOURCE CITATIONS (for technical brief)
# ═══════════════════════════════════════════════════════════════

DATA_SOURCE_CITATIONS = {
    "usda": {
        "name": "USDA FoodData Central",
        "url": "https://fdc.nal.usda.gov/api-guide.html",
        "usage": "Primary food nutrient profiles (calories, macros, micronutrients)",
        "datasets": ["SR Legacy (~7,800 foods)", "Foundation Foods (~2,400 foods)"],
    },
    "monash": {
        "name": "Monash University Low-FODMAP Diet",
        "url": "https://www.monashfodmap.com",
        "usage": "FODMAP classification for IBS-safe meal planning",
        "note": "Gold standard for FODMAP research, used in clinical practice worldwide",
    },
    "nih_rda": {
        "name": "NIH Dietary Reference Intakes",
        "url": "https://www.ncbi.nlm.nih.gov/books/NBK56068",
        "usage": "RDA tables by age and sex for nutrient gap analysis",
    },
    "gi_database": {
        "name": "Glycaemic Index Foundation",
        "url": "https://www.glycemicindex.com",
        "usage": "GI values for diabetes-safe meal planning (low GI ≤ 55)",
    },
    "dash": {
        "name": "NHLBI DASH Eating Plan",
        "url": "https://www.nhlbi.nih.gov/education/dash-eating-plan",
        "usage": "Sodium, potassium, magnesium guidelines for hypertension management",
    },
}
