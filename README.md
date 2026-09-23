# RootCause (live demo)

Ask why a metric changed. RootCause plans the investigation, runs validated read-only SQL, ranks the
segments that changed abnormally, checks for data bugs, pulls related incident documents, and cites
the query behind every number.

**Try:** "Why did the number of orders drop in April 2018 compared to March 2018?" on the *Olist lab*
dataset (real Olist e-commerce data with 5 planted problems whose causes are known).

Demo mode: data is a bundled read-only DuckDB file; the LLM is Gemini (free tier), so an investigation
takes 1-3 minutes. Source code, tests and benchmark: see the main RootCause repository.

## Deploy (Streamlit Community Cloud, free)
1. share.streamlit.io → Create app → this repository, branch `main`, main file `frontend/app.py`
2. Advanced settings → Python 3.12 → Secrets: `GEMINI_API_KEY = "..."`
3. Deploy. Demo mode switches on automatically (no database password, bundled DuckDB file present).

Data: Olist Brazilian E-Commerce Public Dataset (CC BY-NC-SA 4.0), with 5 documented planted anomalies in `olist_lab`.
