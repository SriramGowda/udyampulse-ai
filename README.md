# UdyamPulse AI

An academic dashboard for exploring synthetic MSME financial data, financial risk, anomalies, simple forecasts, and public-reference compliance information. All amounts and business records in the included dataset are synthetic.

## Run locally

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

The app can run without Gemini credentials. To enable Gemini, copy `.env.example` to `.env` and set `GEMINI_API_KEY` (and optionally `GEMINI_MODEL`). Keep `.env` private and never commit a real key. It uses the supported `google-genai` SDK; when Gemini is unavailable, financial explanations are generated locally and the RAG assistant displays the relevant retrieved source passages.

## Architecture and workflow

1. `data/raw/msme_financial_data.csv` contains the synthetic business-month records.
2. `services/data_service.py` loads the data and calculates profitability, liquidity, working-capital, and growth KPIs.
3. Pre-trained Phase 2 artifacts in `models/` classify financial risk and detect unusual financial patterns. SHAP describes feature contributions to the risk model.
4. The dashboard selects one business and presents its latest metrics, historical charts, risk/anomaly results, explanations, forecasts, and downloadable CSV data.
5. `services/ai_service.py` optionally sends only the selected synthetic metrics or retrieved local reference passages to Gemini. Without an API key or when the request fails, local explanations and retrieved excerpts keep the app useful.
6. The compliance assistant chunks local `.txt`/`.md` files, embeds them with TF-IDF, ranks them using cosine similarity, and cites the matching filenames. It does not access private GST, bank, or government databases.

## Models and methods

- **Financial risk:** Existing Random Forest classifier, trained using the project's future-stress target. Its LOW/MEDIUM/HIGH probabilities are indicators, not a guarantee of default.
- **Anomaly detection:** Existing Isolation Forest detects records that differ from learned financial patterns. An anomaly is not necessarily a financial-risk classification.
- **SHAP:** Tree SHAP displays the strongest feature contributions to the current risk prediction.
- **Forecasting:** Revenue and cash balance use a damped linear trend fitted to up to 24 latest chronological monthly observations. Forecasts cover the next six months and are floored at zero. A chronological holdout reports MAE and RMSE separately for each metric; holdout values are not used in the fitted forecast.
- **Gemini:** `google-genai` is optional at runtime and controlled by `GEMINI_API_KEY` and `GEMINI_MODEL`. Prompts constrain answers to supplied synthetic metrics or retrieved reference text.
- **RAG:** Local text extraction, overlapping word chunks, TF-IDF vector embeddings, cosine-similarity retrieval, grounded Gemini generation when configured, and source filename references. Without Gemini, relevant retrieved passages are shown without inventing a generated answer.

## Project map

| Path | Purpose |
| --- | --- |
| `app.py` | Streamlit dashboard and interactions |
| `services/data_service.py` | KPI calculations and forecasting |
| `services/ai_service.py` | Gemini integration and document retrieval |
| `data/raw/` | Synthetic financial history |
| `data/processed/` | Phase 2 training, validation, and explanation outputs |
| `models/` | Existing risk and anomaly model artifacts |
| `documents/` | Local demonstration reference documents |
| `notebooks/` | Reproducible phase experiments |
| `tests/` | Forecasting and retrieval checks |

## Limitations and future scope

This is a college demonstration, not a credit decision or compliance product. Synthetic records do not represent real MSMEs; the dataset is short and the trend forecast does not model seasonality or external events. Model predictions inherit training-data assumptions. Local reference notes are illustrative rather than official legal advice. Gemini responses can be unavailable or incorrect; verify important information with official sources. No private financial, GST, or government data is accessed.

Possible future work includes approved real-world datasets, time-series backtesting across businesses, calibrated risk probabilities, broader document formats and multilingual references, user authentication, and explicit source/version governance.

## Viva questions

1. Which financial indicators are used to calculate the risk target, and why is its target window in the future?
2. How does the dashboard avoid using future observations when forecasting?
3. What do MAE and RMSE measure, and why are they reported separately?
4. How does an Isolation Forest identify an unusual record, and why is an anomaly not the same as high risk?
5. What does a SHAP value explain for an individual prediction?
6. How does TF-IDF represent document chunks, and how does cosine similarity rank them?
7. Why should retrieved context constrain a RAG-generated response?
8. What limitations follow from using synthetic data and a linear-trend forecast?
9. What does the application do when the Gemini API key is missing or a request fails?
10. Why should this dashboard not be used as official financial or compliance advice?

## Tests

Run the service checks with:

```powershell
python -m unittest discover -s tests -v
```
