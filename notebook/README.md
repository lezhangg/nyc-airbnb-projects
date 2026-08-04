# NYC Airbnb Projects

Three connected data science projects using Inside Airbnb NYC data 
(36,000+ listings, March 2025 snapshot). Each one builds on the last: 
find what drives occupancy, use it to recommend listings, then check 
whether guest language backs up the story.

## Live demo
Try the recommendation system: [NYC Airbnb Recommender](https://lezhangg-nyc-airbnb-projects-notebookapp-bc9xuk.streamlit.app/)

---

## Project 1: A/B Testing — Does Instant Book increase occupancy?

`01_data_loading_eda.ipynb` · `02_sql_experiment_setup.ipynb` · `03_ab_test.ipynb`

Instant Book isn't randomly assigned — hosts choose it, and the two 
groups differ systematically on price, room type, and review score 
before any comparison. Used propensity score matching on six 
confounders to build comparable groups, then CUPED for variance 
reduction and OLS regression as an independent check.

**Result:** Instant Book listings show 16.4% occupancy vs 12.1% for 
matched Request to Book listings — a 36% relative increase, holding 
across most room types, boroughs, and price tiers after multiple 
testing correction. A guardrail check also found a small but real 
tradeoff: a small decline in review scores for Instant Book listings.

**Limitation:** Associational, not causal — PSM only controls for 
observed confounders.

## Project 2: Recommendation System — Content-based filtering

`04_recsystem_data_prep.ipynb` · `05_recsystem_model.ipynb` · `app.py`

No booking history in this dataset, so the recommender works purely 
off listing characteristics — solving the cold-start problem. Built 
cosine similarity from scratch, then gradient descent (also from 
scratch) to learn which features actually matter instead of weighting 
everything equally. Combined that with real occupancy data into a 
hybrid model, and validated all of it on a held-out 20% of listings 
the model never trained on.

**Result:** The hybrid recommender consistently returns listings with 
higher occupancy than equal-weight or weighted-only versions across 
every persona tested, holding up on data it never saw during training.

[Try it live](https://lezhangg-nyc-airbnb-projects-notebookapp-bc9xuk.streamlit.app/)

## Project 3: Sentiment Analysis — What do guests actually say?

`06_sentiment_analysis.ipynb`

Ran HuggingFace's DistilBERT on 800,000 guest reviews (sampled from 
936K due to free-tier GPU limits) to see if guest language explains 
the occupancy patterns from Project 1.

**Result:** Sentiment is overwhelmingly positive (87.5%) and adds a 
real but small amount of predictive signal beyond Project 1's 
variables (R-squared improves by 0.17 points). Its real value is 
qualitative — pulling out what's actually missing from negative 
reviews surfaced concrete, fixable complaints: no AC/hot water, no 
elevator, no parking.

---

## What ties these together

Project 1 finds what drives occupancy. Project 2 uses those same 
features to build a recommender. Project 3 checks whether guest 
language backs up the story — and finds that Project 1's guardrail 
result (small review score decline for Instant Book) shows up 
independently in the sentiment data too.

## Tools
Python, pandas, scikit-learn, statsmodels, scipy, HuggingFace 
transformers, Streamlit, SQLite

## Notes on methodology
This is observational data throughout, and results are presented as 
associations backed by rigorous controls, not causal claims. Notebooks 
2, 3, and 6 include dedicated limitations sections; caveats for the 
other notebooks are noted inline where relevant.