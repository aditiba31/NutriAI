# NutriAI — Key AI Prompts

1. "These are the project instructions. I'm going with NutriAI. Create a step by step plan." — Initial architecture and module breakdown.
2. "Go through the professor's lecture topics and map them to specific components in the plan." — Mapped FAISS to embeddings lecture, Bloom filter to sketching lecture.
3. "I want to finish by Thursday. Prioritize what to build first." — Sequenced the build order: pipeline → filters → ranking → UI.
4. "At each step explain what we are doing, why, alternatives considered, and constraints." — Documented design tradeoffs for each module.
5. "How do I pull food data from the USDA FoodData Central API?" — Built the API ingestion loop in data_pipeline.py.
6. "The API returns nutrients as a nested array. How to flatten into DataFrame columns?" — Created nutrient ID-to-column mapping.
7. "These are so many datasets mentioned — why are we only using one? I need all of them." — Expanded to all 5 required data sources.
8. "How do I incorporate Monash FODMAP, GI database, DASH guidelines, and NIH RDA into the pipeline?" — Built data_sources.py with curated reference data from each source.
9. "How is this running without me pasting any API key?" — Built curated fallback dataset for offline use.
10. "I need at least 10,000 records. Currently only getting 7,800 from USDA. How to supplement?" — Added regional/cooking-style variations to reach 15,636 records.
11. "How to deduplicate foods that appear in both USDA and the curated dataset?" — Implemented normalized exact-match deduplication.
12. "Give me high-FODMAP food keywords organized by Monash categories." — Built FODMAP classification lists, cross-checked against Monash app.
13. "What foods trigger GERD according to ACG guidelines?" — Built GERD trigger keyword list.
14. "List high, medium, and low glycaemic index foods from glycemicindex.com." — Built GI classification for diabetes filtering.
15. "NHLBI DASH diet — what are the sodium and potassium limits for hypertension?" — Implemented sodium cap and potassium targets.
16. "How to make the filters composable so I can stack multiple conditions?" — Designed filter chain that applies sequentially with combined exclusion tracking.
17. "The filter.py has manually written allergens for all — is there a better way?" — Kept keyword approach but added dual-check with database column flags.
18. "How to flag cross-contamination risks? Oats are gluten-free but often processed with wheat." — Added cross-contamination keyword lists per allergen type.
19. "Set up FAISS over 13 nutrient columns. Scales are completely different." — Used MinMaxScaler normalization before IndexFlatL2 indexing.
20. "Python set lookup is already O(1). Why use a Bloom filter?" — Justified via measurable false-positive rate as a benchmark metric.
21. "FAISS gives nearest foods by nutrients but doesn't know if something is a breakfast food. How to combine?" — Built multiplicative multi-factor scoring.
22. "How to enforce that no meal repeats across all 21 slots?" — Global used-set tracking across the 7-day plan.
23. "Reasonable serving sizes by food category — everything in USDA is per 100g." — Built serving size lookup with category and food-name overrides.
24. "NIH RDA values by age bracket and sex — need all 13 nutrients." — Built RDA lookup tables in nutrients.py.
25. "Iron RDA is different for premenopausal vs postmenopausal women — make sure that's handled." — Corrected iron from 8mg to 18mg for females 19–50.
26. "How to flag days where a nutrient falls below 80% of RDA?" — Built gap analysis with status flags (low/ok/high).
27. "Mei is vegan but needs B12 above 80% RDA. No animal sources." — Added fortified plant drink assumption and supplement note.
28. "Priya keeps failing the FODMAP check. Validator flags 'oat porridge' but Monash says oats are safe." — Added safe-term whitelists in validator.
29. "GI index keeps going above 55 for the diabetes persona. What's leaking through?" — Tightened filter to exclude both high and medium GI keywords.
30. "James sodium is over 1500mg some days. Need a hard daily cap." — Added daily sodium tracking with hard rejection in ranking loop.
31. "This meal plan looks so off — showing raw ingredients like flour as full meals." — Added USDA category whitelisting and meal-type classification.
32. "Ravi's plan has zero gluten but validator still fails — buckwheat triggers wheat keyword." — Added gluten safe-term exceptions for buckwheat, rice, quinoa.
33. "Streamlit layout — day tabs, nutrient table, metric cards, warning if below 80% RDA." — Built the tabbed day view with collapsible exclusion expander.
34. "How to generate a downloadable PDF of the meal plan from Streamlit?" — Integrated fpdf2 for PDF export.
35. "I want the UI to look like a real product, not a college project." — Removed course-specific labels from the interface.
36. "Deploy to Streamlit Community Cloud — what files does it need?" — Configured requirements.txt at root for deployment.
37. "Go through the rubric line by line. I don't want to lose a single mark." — Full audit across all 5 grading dimensions.
38. "Verify the exported meal plans for all 4 personas against their pass criteria." — Validated every meal against constraint rules.
39. "What's still missing from my submission?" — Identified missing brief.pdf, README.md, prompts.md.
40. "Check if the FAISS and Bloom filter benchmarks are showing real numbers, not zeros." — Fixed curated path to genuinely build and query both structures.
