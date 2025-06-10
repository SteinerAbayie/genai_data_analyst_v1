# LangGraph → BigQuery → Streamlit Dashboard Demo 🚀

_A full Gen‑AI analytics stack in < 200 lines of Python_

---

## 📊 What it does

| Stage           | Action                                                                |
| --------------- | --------------------------------------------------------------------- |
| **1 Prompt**    | You type a plain‑English business question                            |
| **2 LLM**       | GPT‑4o (default) / Gemini / Llama 3 writes governed **BigQuery SQL**  |
| **3 Warehouse** | Query runs, results stream into a **pandas DataFrame**                |
| **4 LLM**       | Model autogenerates a polished **Streamlit** dashboard                |
| **5 Artifacts** | `dashboard_app.py`, `result.csv`, `query.sql` are saved (git‑ignored) |

---

## ✨ Features

- Governed SQL generation (dataset‑whitelist)
- Handles Drive‑backed Sheets (adds Drive scope automatically)
- One‑click pro‑grade dashboard (KPI cards, Ag‑Grid, auto‑charts, dark‑mode toggle)
- Pluggable LLM (GPT‑4o, Gemini 1.5 Pro, Groq Llama 3)
- CI‑friendly artifacts—just hook the "mock Git push" step

---

## 🗺️ Repo layout

```text
.
├── langgraph_streamlit_dashboard.py   ← main script
├── dashboard_app.py                  ← generated Streamlit app (ignored)
├── result.csv                         ← query result (ignored)
├── query.sql                          ← generated SQL (ignored)
├── .env.example                       ← template for secrets
├── .gitignore
└── README.md
```

---

## 🛠️ Quick start

### 1 · Clone & install

```bash
git clone https://github.com/your‑org/genai‑bq‑dashboard.git
cd genai‑bq‑dashboard
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2 · Create Google Cloud credentials (one‑time)

| Step                           | Console navigation                          | Notes                                                               |
| ------------------------------ | ------------------------------------------- | ------------------------------------------------------------------- |
| **a Create a service account** | **IAM & Admin → Service Accounts → Create** | Name it `genai-dashboard-svc`                                       |
| **b Grant roles**              | Same wizard → **Roles**                     | _BigQuery Job User_ + _BigQuery Data Viewer_ (or dataset‑level ACL) |
| **c Download a JSON key**      | **Keys → Add key → JSON**                   | Saves `your‑svc‑acct‑key.json`                                      |
| **d Enable APIs**              | **APIs & Services → Library**               | Turn on **BigQuery API** **and** **Google Drive API**               |

> **Why Drive API?** If any BigQuery table is an external Google Sheet, BigQuery fetches rows via Drive.

### 3 · (If using Sheets) share them with the service account

Open each Sheet → **Share** → add `your‑svc‑acct@…iam.gserviceaccount.com` with **Viewer** access.

### 4 · Fill in `.env`

```dotenv
# ---- .env  (never commit this!) ----
# LLM keys
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=...
GROQ_API_KEY=...

# Google Cloud
GOOGLE_APPLICATION_CREDENTIALS=/absolute/path/your‑svc‑acct‑key.json
BQ_PROJECT_ID=your-gcp-project          # defaults to key's project
BQ_DATASET=analytics                    # dataset the LLM may query
```

### 5 · Run the generator

```bash
python langgraph_streamlit_dashboard.py
# Prompt appears, e.g.:
#   Which channels drive the highest LTV customers last quarter?
```

### 6 · Launch Streamlit

```bash
streamlit run dashboard_app.py
```

Visit **[http://localhost:8501](http://localhost:8501)**—explore filters, charts, KPI cards.

---

## 🔑 How authentication works

```python
from google.oauth2 import service_account
creds = service_account.Credentials.from_service_account_file(
    KEY_PATH,
    scopes=[
        "https://www.googleapis.com/auth/cloud-platform",  # BigQuery
        "https://www.googleapis.com/auth/drive",           # Sheets external tables
    ],
)
client = bigquery.Client(project=PROJECT_ID, credentials=creds)
```

- **Explicit key path**—never falls back to local gcloud creds.
- **Drive scope**—BigQuery can read Google Sheets.
- **`.gitignore` blocks `*.json` & `.env*`**—keys stay local.

---

## 🤖 Switching LLMs

Un‑comment the block you want in `langgraph_streamlit_dashboard.py`:

```python
# GPT‑4o (default)
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

# Gemini 1.5 Pro
# from langchain_google_genai import ChatGoogleGenerativeAI as ChatLLM
# llm = ChatLLM(model="gemini-1.5-pro-latest",
#               google_api_key=os.getenv("GOOGLE_API_KEY"),
#               temperature=0.0)

# Groq Llama‑3 70B
# from langchain_groq import ChatGroq
# llm = ChatGroq(model_name="llama3-70b-8192",
#                groq_api_key=os.getenv("GROQ_API_KEY"),
#                temperature=0.0)
```

---

## 🩹 Common errors & fixes

| Error                            | Likely cause                                                   | Fix                                                               |
| -------------------------------- | -------------------------------------------------------------- | ----------------------------------------------------------------- |
| **`403 accessDenied`** (Sheets)  | Sheet not shared • Drive API disabled • creds lack Drive scope | Share sheet → enable Drive API → ensure scope list includes Drive |
| **`BigQuery Job User` missing**  | Service account lacks role                                     | Grant _BigQuery Job User_ in IAM                                  |
| **LLM 429 / insufficient_quota** | Free tier exhausted                                            | Wait 24 h, switch model, or add billing cap                       |
