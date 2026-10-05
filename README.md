# UdyamPulse AI

UdyamPulse AI is a professional financial analytics demonstration for synthetic MSME records. It pairs a React/TypeScript dashboard and FastAPI REST API with the existing financial KPI, Random Forest risk, Isolation Forest anomaly, SHAP and forecast implementations, plus optional Gemini and offline Ollama/RAG support.

> **Academic demonstration only:** all included financial data is synthetic. This app is not a credit decision, financial advice, or official compliance guidance.

## Architecture

```text
frontend/                         React + TypeScript + Vite + Tailwind
  src/components/                 Shared cards, states, tables and charts
  src/pages/                      Dashboard and seven analytics/assistant pages
  src/services/api.ts             Browser-to-API client (no secrets)
  src/hooks/                      API loading and error states
  src/layouts/                    Responsive navigation and business selector
backend/
  main.py                         FastAPI app, CORS and error handling
  routes/api.py                   JSON REST endpoints
  services/analytics.py           Loads and invokes the existing saved models
services/
  data_service.py                  Existing KPIs and forecasts
  ai_service.py                    Gemini-first provider orchestration
  local_llm_service.py             Local-only Ollama detection and inference
  rag_service.py                   Local document loading, chunking and retrieval
models/                            Existing saved Random Forest / Isolation Forest
data/raw/                          Synthetic monthly MSME financial records
documents/                         Local TXT/MD RAG references
```

The backend reuses `services/data_service.py` for KPIs and forecasts and the existing model artifacts in `models/`. The React app calls only FastAPI. Provider credentials remain backend-only and are never sent to the browser.

### RAG and provider architecture

Compliance queries follow this pipeline:

```text
documents/*.txt, *.md
  -> chunk locally
  -> TF-IDF vectors
  -> cosine-similarity ranking
  -> relevant excerpts + source filenames
  -> Gemini (when configured and responding)
  -> Ollama local LLM (when the local runtime and model are ready)
  -> retrieved-document fallback (always local)
```

TF-IDF/cosine similarity is the document retrieval method; Ollama is the local text-generation runtime. Generated explanation and retrieved document information are returned separately so a quoted passage is not mistaken for generated advice. The two local references are `documents/msme_finance_reference_1.txt` and `documents/msme_finance_reference_2.txt`.

Provider order for text generation is Gemini, then a local Ollama model, then a retrieval-only answer for compliance. AI Insights retains a deterministic local financial summary if neither language model is usable; that is an analytical fallback, not LLM generation. Provider outages and missing optional configuration do not prevent the analytics API or local retrieval from working.

## Prerequisites

- Python 3.10+ (the checked project environment uses Python 3.13)
- Node.js 20.19+ or 22.12+
- The existing model artifacts under `models/`

## Install and run

From the project root in PowerShell:

```powershell
# Python environment and project dependencies
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

# Backend API (terminal 1)
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000

# React frontend (terminal 2)
cd frontend
npm install
npm run dev
```

Open the Vite URL, normally `http://localhost:5173`. The API health endpoint is `http://127.0.0.1:8000/api/health`; interactive API documentation is available at `http://127.0.0.1:8000/docs`.

## Frontend

The desktop-first responsive app includes:

1. Dashboard
2. Business Analysis
3. Risk Intelligence
4. Forecasting (revenue and cash tabs, MAE/RMSE, six-month table, CSV)
5. Anomaly Detection
6. AI Insights
7. Compliance Assistant
8. Reports (business history and forecast CSV exports)

The frontend uses Recharts and Lucide icons. Set `VITE_API_URL` at build/dev time only if the API is not at `http://localhost:8000`. Do not place `GEMINI_API_KEY` or any backend secret in a `VITE_*` variable.

## Backend and API endpoints

The FastAPI application loads configuration and secrets on the server. Default CORS origins are `http://localhost:5173` and `http://127.0.0.1:5173`; override them with a comma-separated `CORS_ORIGINS` setting for a deployment.

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/api/health` | API, dataset and Gemini configuration status (boolean only; never returns credentials) |
| `GET` | `/api/ai/status` | Safe provider, local Ollama/model and offline retrieval readiness |
| `GET` | `/api/businesses` | Business selector options and latest periods |
| `GET` | `/api/business/{business_id}` | Latest record and monthly history |
| `GET` | `/api/business/{business_id}/kpis` | Latest financial record and calculated KPIs |
| `GET` | `/api/business/{business_id}/risk` | Risk label, probabilities and model explanation |
| `GET` | `/api/business/{business_id}/anomaly` | Latest anomaly status, score and indicator values |
| `GET` | `/api/business/{business_id}/shap` | Top SHAP feature contributions |
| `GET` | `/api/business/{business_id}/forecast` | Six-month revenue/cash forecasts and holdout metrics |
| `POST` | `/api/ai/summary` | Financial explanation for `{ "business_id": "MSME001" }` |
| `POST` | `/api/compliance/query` | Local-reference RAG for `{ "question": "..." }` |

Unknown businesses return `404`, invalid requests return validation errors, and unexpected server exceptions are logged and return a generic `500` response.

## Online Gemini mode

Gemini is optional. To enable online generation, configure the Gemini credential in the backend process environment or a private, Git-ignored root `.env` file. The backend uses `GEMINI_MODEL` (default `gemini-3.8-flash`). No endpoint returns credential values, and no `VITE_*` frontend variable should contain backend credentials. If Gemini has no configuration or a generation request fails, the request falls through to local Ollama generation and then the relevant local fallback. `/api/ai/status` reports configuration only; it does not make a live Gemini connectivity or credential check.

## Offline mode and local model setup

The React UI uses system fonts and does not require external font downloads. With FastAPI running, local financial data, KPI calculations, the existing Random Forest and Isolation Forest models, SHAP, forecasting and TF-IDF/cosine-similarity RAG do not depend on Gemini or internet access. Ollama provides local LLM generation when installed, running, and holding the configured model. If Ollama or its model is unavailable, compliance still returns retrieved-document fallback information with source filenames. **Do not describe retrieval fallback as LLM generation.**

Install Ollama separately from [ollama.com](https://ollama.com), start the local Ollama service, then choose and download a model on the user's machine. The default is `llama3.2:3b`; the application does not download models automatically. After installing Ollama, run `ollama pull llama3.2:3b`, or set `OLLAMA_MODEL` in the backend environment to another installed model tag. The status endpoint checks the local Ollama tags endpoint and configured model name. Ollama inference is sent only to `127.0.0.1:11434`; it does not use an external model API.

The safe provider status includes whether Gemini is configured (not a live service check), whether Ollama is installed/running, whether the configured model is available, whether local references are available, `offline_ready`, the model tag, and the preferred provider. `offline_ready` means a local model or local references are available; a local LLM specifically requires both Ollama and the configured model.

## Existing ML/AI pipeline

- **Financial KPIs:** `services/data_service.py` computes profitability, liquidity, growth, and working-capital measures.
- **Risk:** the existing Random Forest model in `models/financial_risk_model.pkl` returns LOW/MEDIUM/HIGH class probabilities.
- **Anomalies:** the existing Isolation Forest artifact in `models/anomaly_detection_model.pkl` scores the latest financial feature vector.
- **Explanations:** Tree SHAP computes feature-level contributions to the risk prediction.
- **Forecasting:** the existing data service fits a damped linear trend on up to 24 chronological observations, forecasts six months and evaluates a chronological holdout with MAE and RMSE.
- **Financial explanation:** `services/ai_service.py` tries Gemini, then Ollama; if both are unavailable it returns the existing deterministic financial analysis, explicitly identified as a non-LLM fallback.
- **Compliance RAG:** `services/rag_service.py` preserves local TXT/MD chunking, TF-IDF vectorization and cosine-similarity retrieval. `services/ai_service.py` passes only relevant excerpts to Gemini or Ollama and returns source filenames plus retrieved excerpts. If neither model is available, retrieved passages remain usable without generation.

## Tests and production build

```powershell
python -m unittest discover -s tests -v
Set-Location .\frontend
npm run build
```

The tests cover retrieval and citations, Gemini/local/retrieval fallback order, Ollama model detection and loopback-only requests, safe provider status, AI API routes, and the existing KPI, model/SHAP and forecast behavior. Provider generation is mocked; automated tests do not make real Gemini calls.

## Troubleshooting

- **AI mode reports retrieval fallback:** local RAG remains available. Install and start Ollama, pull a model locally, and set `OLLAMA_MODEL` to the exact installed tag if offline LLM generation is desired.
- **Gemini is configured but generation uses local mode:** status is configuration-only. A Gemini generation failure falls back to Ollama; check backend logs for the failure class and verify backend connectivity independently.
- **The dashboard is unavailable offline:** start both FastAPI and Vite locally as shown above. The frontend requires its local FastAPI service.
- **Model artifacts or data fail to load:** run the backend from the project root and retain `data/` and both existing artifacts under `models/`.
- **CORS errors:** set `CORS_ORIGINS` to the local Vite origin(s) and restart FastAPI.

## Limitations

Local LLM generation requires the separately installed Ollama application, a running local service, and a manually downloaded compatible model; the application does not install or download one. The bundled local reference library is illustrative, limited to TXT/MD content, and may be incomplete or out of date. Retrieval fallback displays relevant excerpts without synthesizing an answer. Gemini and Ollama generated text may be inaccurate; verify important financial and compliance information with qualified professionals and official sources. The data is synthetic, the forecast is a simple damped trend, model probabilities inherit training assumptions, and SHAP explains model behavior rather than causation. Authentication, authorization and user-specific data isolation are not implemented; do not expose this demo API to untrusted users.

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
