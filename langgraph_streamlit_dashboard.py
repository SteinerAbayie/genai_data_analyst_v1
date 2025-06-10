# langgraph_streamlit_dashboard.py – BigQuery edition (v1.2 – compat patched)
"""
End‑to‑End Gen‑AI Analytics Stack demo
=====================================
This self‑contained script drives the full Gen‑AI flow:

1. ✍️  Collect a business question →
2. 🧠  LLM (via **LangGraph**) writes governed BigQuery SQL →
3. 📊  Query executes, DataFrame returns →
4. 🖥️  LLM autogenerates a Streamlit dashboard →
5. 🚀  Artifacts saved + mock Git deploy.

> **Quick start**  
> ```bash
> # Recommended versions (prevent CONFIG_KEYS import issue)
> pip install "langgraph>=0.0.39,<0.1" \
>             "langchain-core>=0.1.5,<0.2" \
>             "langchain-openai>=0.1.6" \
>             google-cloud-bigquery streamlit pandas
> ```  
> Ensure environment vars: `OPENAI_API_KEY`, `GOOGLE_APPLICATION_CREDENTIALS`, and optional `BQ_PROJECT_ID`, `BQ_DATASET`.
"""

# ---------------------------------------------------------------------------
# 0. Compatibility patch – handle CONFIG_KEYS removal in newer langchain-core
#    This MUST come **before** importing langgraph, which expects CONFIG_KEYS.
# ---------------------------------------------------------------------------
try:
    from langchain_core.runnables import config as _lc_cfg  # type: ignore

    if not hasattr(_lc_cfg, "CONFIG_KEYS"):
        # Derive keys from RunnableConfig dataclass if possible, else stub.
        try:
            from dataclasses import fields
            from langchain_core.runnables.config import RunnableConfig  # type: ignore

            _lc_cfg.CONFIG_KEYS = {f.name for f in fields(RunnableConfig)}  # type: ignore
        except Exception:  # pragma: no cover – fallback stub
            _lc_cfg.CONFIG_KEYS = set()  # type: ignore
except ImportError:
    # If langchain_core itself is missing, the user will get a clearer error later.
    pass

# ---------------------------------------------------------------------------
# 1. Standard imports (delayed langgraph import comes **after** patch)
# ---------------------------------------------------------------------------
import os
from pathlib import Path
from typing import TypedDict, Optional

import pandas as pd
from google.cloud import bigquery
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

load_dotenv()
# ---------------------------------------------------------------------------
# 2. Environment checks
# ---------------------------------------------------------------------------
assert os.getenv("OPENAI_API_KEY"), "➡️  Set OPENAI_API_KEY first!"
assert os.getenv(
    "GOOGLE_APPLICATION_CREDENTIALS"
), "➡️  Set GOOGLE_APPLICATION_CREDENTIALS first!"

# ---------------------------------------------------------------------------
# 3. Helper – fetch BigQuery schema summary (governance layer)
# ---------------------------------------------------------------------------


def fetch_bq_schema(client: bigquery.Client, dataset_id: str) -> str:
    """Return a Markdown list of <table.column : dtype> for the dataset."""
    md_lines = []
    for table in client.list_tables(dataset_id):
        table_ref = f"{dataset_id}.{table.table_id}"
        schema = client.get_table(table_ref).schema
        for field in schema:
            md_lines.append(f"* **{table.table_id}.{field.name}** – {field.field_type}")
    return "\n".join(md_lines)


# ---------------------------------------------------------------------------
# 4. LangGraph state definition
# ---------------------------------------------------------------------------
class DashState(TypedDict):
    user_prompt: str  # business question
    sql_query: Optional[str]  # generated & governed SQL
    df: Optional[pd.DataFrame]  # BigQuery result set
    streamlit_code: Optional[str]  # generated dashboard code


# ---------------------------------------------------------------------------
# 5. Graph nodes
# ---------------------------------------------------------------------------
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

# from langchain_google_genai import ChatGoogleGenerativeAI as ChatLLM

# llm = ChatLLM(
#     model="gemini-1.5-pro-latest",
#     google_api_key=os.getenv("GOOGLE_API_KEY"),
#     temperature=0.0,
# )
from google.oauth2 import service_account
from google.cloud import bigquery

KEY_PATH = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")  # points to your JSON file
assert KEY_PATH and Path(KEY_PATH).exists(), "Service‑account key not found!"

SCOPES = [
    "https://www.googleapis.com/auth/cloud-platform",  # BigQuery + most GCP APIs
    "https://www.googleapis.com/auth/drive",  # 👈 lets BQ read Drive files
]

creds = service_account.Credentials.from_service_account_file(KEY_PATH, scopes=SCOPES)

PROJECT_ID = os.getenv("BQ_PROJECT_ID") or creds.project_id
client = bigquery.Client(project=PROJECT_ID, credentials=creds)
DATASET = os.getenv("BQ_DATASET", "analytics")  # change as needed


def ask_user(state: DashState) -> DashState:
    """Prompt for business question if not provided programmatically."""
    if state["user_prompt"]:
        return state
    print(
        "\n📋  Enter the analytics question you'd like answered (e.g. 'Which channels drive our best customers?')"
    )
    state["user_prompt"] = input("✏️  > ").strip()
    return state


def generate_sql(state: DashState) -> DashState:
    """Generate governed BigQuery SQL via LLM."""
    schema_md = fetch_bq_schema(client, DATASET)
    system = (
        "You are an analytics engineer. Generate **single‑statement** StandardSQL for BigQuery that answers the business question, "
        "using only the tables and columns listed. Wrap identifiers in back‑ticks. Do NOT use unsupported tables. "
        f"Return SQL only, no markdown. Assume dataset is `{DATASET}`."
    )
    user = (
        f"Schema available:\n{schema_md}\n\nBusiness question: {state['user_prompt']}"
    )
    sql = llm.invoke(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
    ).content.strip()
    state["sql_query"] = sql
    return state


def run_query(state: DashState) -> DashState:
    """Execute SQL against BigQuery and store result."""
    query_job = client.query(state["sql_query"])
    state["df"] = query_job.to_dataframe()
    return state


def generate_streamlit(state: DashState) -> DashState:
    """LLM writes Streamlit code for the fetched DataFrame."""
    cols_snippet = ", ".join(state["df"].columns[:5]) + (
        " …" if len(state["df"].columns) > 5 else ""
    )
    system = """Write a **production‑ready Streamlit app** (return *code only*) that:
1. **Loads** `result.csv` into a pandas DataFrame named `df`.
2. **Derives and displays** a row of KPI cards at the top (e.g., total rows, date range, revenue sum/mean if numeric).
3. Uses **st.sidebar** to auto‑generate filters:
   • For each *categorical* column → multiselect.  
   • For each *numeric* column → min/max slider.  
   Apply filters to `df` live.
4. Shows an **Ag‑Grid** (interactive) table of the filtered data.
5. **Auto‑detects column types** and renders:
   • A responsive bar/line chart for numeric time‑series (if a date column exists).  
   • Pie or bar charts for categorical distributions (top 10).  
   • Histogram box for numeric distributions.
6. Adds tabs or expanders so the user can switch between visualizations.
7. Uses a **corporate‑blue theme** with:
   • Clean sans‑serif font.  
   • Rounded corners, subtle shadows.  
   • Hover tooltips for charts.  
   • Light mode by default; allow dark‑mode toggle.
8. Wraps the code in `if __name__ == "__main__":` so it runs as a script.
9. Includes graceful handling of empty filter results (show a warning instead of crashing).
Return only the Streamlit Python code block—no markdown, explanations, or extra text. make sure ```python ```
is not in the code
"""
    user = f"Sample columns: {cols_snippet}"
    state["streamlit_code"] = llm.invoke(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
    ).content
    return state


def persist_files(state: DashState) -> DashState:
    """Save CSV, Streamlit app, and SQL to disk; mock a Git push."""
    state["df"].to_csv("result.csv", index=False)
    Path("dashboard_app.py").write_text(state["streamlit_code"], encoding="utf‑8")
    Path("query.sql").write_text(state["sql_query"], encoding="utf‑8")
    print(
        "\n✅  Created dashboard_app.py, result.csv & query.sql  –  run:  streamlit run dashboard_app.py"
    )
    print("📦  Pretending to commit & push to Git… (hook in your CI/CD here)")
    return state


# ---------------------------------------------------------------------------
# 6. Build and compile the LangGraph
# ---------------------------------------------------------------------------
workflow = StateGraph(DashState)

# real nodes
workflow.add_node("ask", ask_user)
workflow.add_node("sql", generate_sql)
workflow.add_node("run", run_query)
workflow.add_node("st", generate_streamlit)
workflow.add_node("save", persist_files)

workflow.set_entry_point("ask")  # <- just this, no START import
workflow.add_edge("ask", "sql")
workflow.add_edge("sql", "run")
workflow.add_edge("run", "st")
workflow.add_edge("st", "save")
workflow.add_edge("save", END)

dag = workflow.compile()


# 7. Public helper
# ---------------------------------------------------------------------------
def build_dashboard(prompt: str = "") -> None:
    """Kick off the graph with an optional pre-supplied prompt."""
    try:
        # Initialize the state properly with all required fields
        initial_state = {
            "user_prompt": prompt,
            "sql_query": None,
            "df": None,
            "streamlit_code": None,
        }
        dag.invoke(initial_state)
    except Exception as e:
        print(f"Error executing dashboard build: {str(e)}")
        raise


# ---------------------------------------------------------------------------
# 8. Smoke tests (RUN_TESTS=1)
# ---------------------------------------------------------------------------
if os.getenv("RUN_TESTS") == "1":
    import unittest

    class CompatPatchTest(unittest.TestCase):
        def test_config_keys_present(self):
            from langchain_core.runnables import config as _cfg

            self.assertTrue(hasattr(_cfg, "CONFIG_KEYS"))

    unittest.main(exit=False)


# ---------------------------------------------------------------------------
# 9. CLI entry‑point
# ---------------------------------------------------------------------------
if __name__ == "__main__" and os.getenv("RUN_TESTS") != "1":
    print("🛠️  LangGraph‑to‑Streamlit (BigQuery) Demo  —  Dataset:", DATASET)
    build_dashboard()
