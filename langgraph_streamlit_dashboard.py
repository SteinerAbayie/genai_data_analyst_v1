# langgraph_streamlit_dashboard_combined.py – Ultimate BigQuery Dashboard Builder (v4.0)
"""
End-to-End Gen-AI Analytics Stack with Beautiful Visualizations - Combined Edition
=================================================================================
This combines the best features from v2.3 (robust BigQuery integration) and v3.0 (beautiful visualizations)

Key Features
------------
1. **Robust BigQuery Integration** – Full schema detection and error handling
2. **Enhanced Planning** – Deep data analysis with comprehensive visual specs
3. **Beautiful Visualizations** – Modern dashboard design with multiple themes
4. **Dual Self-Repair Loops** – Auto-fix BigQuery errors and UI issues
5. **Smart Chart Selection** – Data-driven visualization recommendations
6. **Modern UI Components** – Contemporary design with animations and interactivity
7. **Comprehensive Error Handling** – Robust error detection and recovery
8. **Multi-Model Support** – Configurable models for different tasks

Quick start
-----------
```bash
pip install "langgraph>=0.0.39,<0.1" \
            "langchain-core>=0.1.5,<0.2" \
            "langchain-openai>=0.1.6" \
            google-cloud-bigquery streamlit pandas python-dotenv plotly seaborn numpy
```
Set `OPENAI_API_KEY`, `GOOGLE_APPLICATION_CREDENTIALS`, and optionally other config vars.
"""

# ---------------------------------------------------------------------------
# 0. Compatibility patch
# ---------------------------------------------------------------------------
try:
    from langchain_core.runnables import config as _lc_cfg

    if not hasattr(_lc_cfg, "CONFIG_KEYS"):
        from dataclasses import fields
        from langchain_core.runnables.config import RunnableConfig

        _lc_cfg.CONFIG_KEYS = {f.name for f in fields(RunnableConfig)}  # type: ignore
except ImportError:
    pass

# ---------------------------------------------------------------------------
# 1. Imports & env
# ---------------------------------------------------------------------------
import json
import os
import re
import subprocess
import textwrap
from pathlib import Path
from typing import Optional, TypedDict, Dict, List, Any

import pandas as pd
import numpy as np
from dotenv import load_dotenv
from google.cloud import bigquery
from google.cloud.exceptions import BadRequest
from google.oauth2 import service_account
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from openai import AuthenticationError

load_dotenv()

# Debug output for environment variables
print(f"DEBUG - BQ_PROJECT_ID from env: {os.getenv('BQ_PROJECT_ID')}")
print(f"DEBUG - BQ_DATASET from env: {os.getenv('BQ_DATASET')}")


def extract_sql_from_markdown(text: str) -> str:
    """Extract a SQL query from a markdown code block."""
    match = re.search(r"```(?:sql)?\n(.*?)\n```", text, re.DOTALL)
    return match.group(1) if match else text


def extract_code_from_markdown(text: str) -> str:
    """Extract Python code from a markdown code block."""
    match = re.search(r"```(?:python)?\n(.*?)\n```", text, re.DOTALL)
    return match.group(1) if match else text


# Environment validation
assert os.getenv("OPENAI_API_KEY"), "➡️  Set OPENAI_API_KEY first!"
assert os.getenv(
    "GOOGLE_APPLICATION_CREDENTIALS"
), "➡️  Set GOOGLE_APPLICATION_CREDENTIALS first!"

MAX_RETRIES = int(os.getenv("MAX_RETRIES", "5"))
TIMEOUT = int(os.getenv("TIMEOUT", "45"))

# ---------------------------------------------------------------------------
# 2. BigQuery client & schema helper (Enhanced from v2.3)
# ---------------------------------------------------------------------------
KEY_PATH = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
SCOPES = [
    "https://www.googleapis.com/auth/cloud-platform",
    "https://www.googleapis.com/auth/drive",
]
creds = service_account.Credentials.from_service_account_file(KEY_PATH, scopes=SCOPES)
PROJECT_ID = os.getenv("BQ_PROJECT_ID") or creds.project_id
DATASET = os.getenv("BQ_DATASET", "test_cap")
client = bigquery.Client(project=PROJECT_ID, credentials=creds)

# Enhanced debug output
print(f"DEBUG - PROJECT_ID being used: {PROJECT_ID}")
print(f"DEBUG - DATASET being used: {DATASET}")
print(f"DEBUG - BigQuery client project: {client.project}")
print(f"DEBUG - Full table reference should be: {PROJECT_ID}.{DATASET}.telco_main")

# Test BigQuery connection
try:
    test_query = (
        f"SELECT COUNT(*) as row_count FROM `{PROJECT_ID}.{DATASET}.telco_main` LIMIT 1"
    )
    print(f"DEBUG - Testing query: {test_query}")
    result = client.query(test_query).result()
    for row in result:
        print(f"DEBUG - Test query successful, row count: {row.row_count}")
except Exception as e:
    print(f"DEBUG - Test query failed: {e}")


def fetch_bq_schema(dataset: str) -> str:
    """Fetch BigQuery schema with enhanced error handling."""
    lines: list[str] = []
    try:
        for tbl in client.list_tables(dataset):
            schema = client.get_table(f"{dataset}.{tbl.table_id}").schema
            for fld in schema:
                lines.append(f"* **{tbl.table_id}.{fld.name}** – {fld.field_type}")
    except Exception as e:
        print(f"❌ Failed to fetch schema: {e}")
        return "Schema unavailable"
    return "\n".join(lines)


SCHEMA_MD = fetch_bq_schema(DATASET)
print(f"🔎 Fetched schema for dataset '{DATASET}':\n{SCHEMA_MD}")

# ---------------------------------------------------------------------------
# 3. Visual Design Templates & Configuration (From v3.0)
# ---------------------------------------------------------------------------
DESIGN_TEMPLATES = {
    "modern_dark": {
        "color_palette": [
            "#FF6B6B",
            "#4ECDC4",
            "#45B7D1",
            "#96CEB4",
            "#FFEAA7",
            "#DDA0DD",
            "#98D8C8",
        ],
        "background_color": "#1E1E1E",
        "paper_color": "#2D2D2D",
        "text_color": "#FFFFFF",
        "grid_color": "#404040",
        "font_family": "Inter, sans-serif",
        "theme": "dark",
    },
    "clean_light": {
        "color_palette": [
            "#3498DB",
            "#E74C3C",
            "#2ECC71",
            "#F39C12",
            "#9B59B6",
            "#1ABC9C",
            "#E67E22",
        ],
        "background_color": "#FFFFFF",
        "paper_color": "#F8F9FA",
        "text_color": "#2C3E50",
        "grid_color": "#ECF0F1",
        "font_family": "Inter, sans-serif",
        "theme": "light",
    },
    "corporate_blue": {
        "color_palette": [
            "#2E86AB",
            "#A23B72",
            "#F18F01",
            "#C73E1D",
            "#592941",
            "#0B4F6C",
            "#F4A261",
        ],
        "background_color": "#F5F7FA",
        "paper_color": "#FFFFFF",
        "text_color": "#34495E",
        "grid_color": "#BDC3C7",
        "font_family": "Roboto, sans-serif",
        "theme": "light",
    },
    "vibrant_gradient": {
        "color_palette": [
            "#FF9A9E",
            "#FECFEF",
            "#FECFEF",
            "#A8E6CF",
            "#FFD3A5",
            "#FFA8A8",
            "#C7CEEA",
        ],
        "background_color": "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
        "paper_color": "rgba(255, 255, 255, 0.95)",
        "text_color": "#2C3E50",
        "grid_color": "#E8EAF6",
        "font_family": "Poppins, sans-serif",
        "theme": "light",
    },
}


# ---------------------------------------------------------------------------
# 4. Data Analysis Helper Functions (Enhanced from v3.0)
# ---------------------------------------------------------------------------
def analyze_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    """Comprehensive data analysis for visualization planning."""
    analysis = {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": df.dtypes.to_dict(),
        "numeric_columns": list(df.select_dtypes(include=[np.number]).columns),
        "categorical_columns": list(
            df.select_dtypes(include=["object", "category"]).columns
        ),
        "datetime_columns": list(df.select_dtypes(include=["datetime64"]).columns),
        "null_counts": df.isnull().sum().to_dict(),
        "unique_counts": {col: df[col].nunique() for col in df.columns},
        "sample_data": df.head(3).to_dict("records"),
        "statistical_summary": {},
    }

    # Statistical summaries for numeric columns
    for col in analysis["numeric_columns"]:
        if len(df[col].dropna()) > 0:
            stats = df[col].describe()
            analysis["statistical_summary"][col] = {
                "mean": stats["mean"],
                "std": stats["std"],
                "min": stats["min"],
                "max": stats["max"],
                "range": stats["max"] - stats["min"],
                "skewness": df[col].skew(),
            }

    # Detect potential relationships
    analysis["suggested_visualizations"] = suggest_chart_types(analysis)
    return analysis


def suggest_chart_types(analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Suggest appropriate chart types based on data characteristics."""
    suggestions = []
    numeric_cols = analysis["numeric_columns"]
    categorical_cols = analysis["categorical_columns"]
    datetime_cols = analysis["datetime_columns"]

    # Time series charts
    if datetime_cols and numeric_cols:
        for dt_col in datetime_cols[:1]:
            for num_col in numeric_cols[:2]:
                suggestions.append(
                    {
                        "type": "line",
                        "x": dt_col,
                        "y": num_col,
                        "title": f"{num_col} Over Time",
                        "rationale": "Time series visualization",
                    }
                )

    # Categorical vs Numeric
    if categorical_cols and numeric_cols:
        for cat_col in categorical_cols[:2]:
            if analysis["unique_counts"][cat_col] <= 20:
                for num_col in numeric_cols[:2]:
                    suggestions.append(
                        {
                            "type": "bar",
                            "x": cat_col,
                            "y": num_col,
                            "title": f"{num_col} by {cat_col}",
                            "rationale": "Categorical comparison",
                        }
                    )

    # Numeric distributions
    for num_col in numeric_cols[:3]:
        suggestions.append(
            {
                "type": "histogram",
                "x": num_col,
                "title": f"Distribution of {num_col}",
                "rationale": "Data distribution analysis",
            }
        )

    # Correlation analysis for multiple numeric columns
    if len(numeric_cols) >= 2:
        suggestions.append(
            {
                "type": "scatter",
                "x": numeric_cols[0],
                "y": numeric_cols[1],
                "title": f"{numeric_cols[1]} vs {numeric_cols[0]}",
                "rationale": "Correlation analysis",
            }
        )

    # Pie charts for categorical with reasonable unique counts
    for cat_col in categorical_cols[:2]:
        if 2 <= analysis["unique_counts"][cat_col] <= 8:
            suggestions.append(
                {
                    "type": "pie",
                    "values": cat_col,
                    "title": f"Distribution of {cat_col}",
                    "rationale": "Categorical distribution",
                }
            )

    return suggestions[:6]


def select_optimal_theme(analysis: Dict[str, Any]) -> str:
    """Select the best theme based on data characteristics."""
    return "modern_dark"  # Default to modern dark theme


# ---------------------------------------------------------------------------
# 5. State definition (Combined)
# ---------------------------------------------------------------------------
class DashState(TypedDict):
    prompt: str
    sql_query: Optional[str]
    df: Optional[pd.DataFrame]
    data_analysis: Optional[Dict[str, Any]]
    visual_plan: Optional[Dict[str, Any]]
    streamlit_code: Optional[str]
    error_log: Optional[str]
    retries: int


# ---------------------------------------------------------------------------
# 6. Model setup (Enhanced)
# ---------------------------------------------------------------------------
PLAN_MODEL = os.getenv("PLAN_MODEL", "gpt-4o")
CODE_MODEL = os.getenv("CODE_MODEL", "gpt-4o")
REPAIR_MODEL = os.getenv("REPAIR_MODEL", CODE_MODEL)

planner_llm = ChatOpenAI(model=PLAN_MODEL, temperature=0.1)
coder_llm = ChatOpenAI(model=CODE_MODEL, temperature=0.3)
repair_llm = ChatOpenAI(model=REPAIR_MODEL, temperature=0.3)


# ---------------------------------------------------------------------------
# 7. Enhanced Graph nodes (Combined best features)
# ---------------------------------------------------------------------------
def ask_user(state: DashState) -> DashState:
    if state["prompt"]:
        return state
    state["prompt"] = input("\n✏️  Business question > ").strip()
    return state


def generate_sql(state: DashState) -> DashState:
    """Enhanced SQL generation with v2.3 robustness and v3.0 optimization."""
    prompt = textwrap.dedent(
        f"""You are an expert Google BigQuery analytics engineer. Your task is to write a single, syntactically correct Standard SQL query to answer the user's business question.

**Instructions:**
1. You MUST use only the tables and columns provided in the schema below.
2. When appropriate, use Common Table Expressions (CTEs) to structure your query for clarity.
3. All table and column names MUST be enclosed in backticks (`).
4. The query MUST be a single SQL statement.
5. Do NOT add any explanations, comments, or markdown formatting. Return only the raw SQL query.
6. You MUST fully qualify all table names as `{PROJECT_ID}.{DATASET}.table_name`.
7. Optimize for rich, analyzable results that will create compelling visualizations.

**Schema:**
{SCHEMA_MD}

**Example Query (using a CTE):**
Here is an example of a good query that calculates a rate from a boolean column.
- **Business Question:** "What is the referral rate by offer type?"
- **SQL Query:**
```sql
WITH ReferralData AS (
  SELECT
    `Offer`,
    CAST(`Referred_a_Friend` AS INT64) AS `is_referral`
  FROM
    `{PROJECT_ID}.{DATASET}.telco_main`
)
SELECT
  `Offer`,
  AVG(`is_referral`) AS `referral_rate`
FROM
  ReferralData
GROUP BY
  `Offer`
ORDER BY
  `referral_rate` DESC
```

**Business Question:**
{state['prompt']}

**SQL Query:**"""
    )

    try:
        raw_output = planner_llm.invoke(
            [{"role": "system", "content": prompt}]
        ).content.strip()
        state["sql_query"] = extract_sql_from_markdown(raw_output)
        print(f"DEBUG - Generated SQL: {state['sql_query']}")
    except AuthenticationError:
        raise RuntimeError("Invalid OPENAI_API_KEY; check env var.")
    return state


def run_query(state: DashState) -> DashState:
    """Enhanced query execution with better error handling."""
    print(f"\n▶️ Running query:\n{state['sql_query']}")
    try:
        state["df"] = client.query(state["sql_query"]).to_dataframe()
        state["error_log"] = None
        print(
            f"✅ Query successful! Retrieved {len(state['df'])} rows, {len(state['df'].columns)} columns"
        )
    except Exception as e:
        state["error_log"] = f"Query failed: {str(e)}"
        print(f"❌ Query failed: {e}")
    return state


def fix_sql(state: DashState) -> DashState:
    """Enhanced SQL repair with v2.3 robustness."""
    print(f"\n🔧 Attempting to fix SQL. Retry {state['retries'] + 1}/{MAX_RETRIES}")
    print(f"  - Error: {state['error_log']}")
    print(f"  - Failed SQL: {state['sql_query']}")

    prompt = textwrap.dedent(
        f"""Fix this BigQuery SQL query. The error is: {state['error_log']}
    
Current SQL:
{state['sql_query']}

Schema for reference:
{SCHEMA_MD}

Instructions:
1. Return ONLY the corrected SQL query
2. Use fully qualified table names: `{PROJECT_ID}.{DATASET}.table_name`
3. Do NOT include any explanations, comments, or markdown
4. The query must be syntactically correct BigQuery Standard SQL

Fixed SQL:"""
    )

    raw_output = repair_llm.invoke(
        [{"role": "system", "content": prompt}]
    ).content.strip()
    state["sql_query"] = extract_sql_from_markdown(raw_output)
    state["retries"] += 1
    print(f"DEBUG - Fixed SQL: {state['sql_query']}")
    return state


def analyze_data(state: DashState) -> DashState:
    """Comprehensive data analysis for visualization planning."""
    print("\n🔍 Analyzing data for optimal visualization...")
    state["data_analysis"] = analyze_dataframe(state["df"])

    analysis = state["data_analysis"]
    print(f"  - Shape: {analysis['shape']}")
    print(f"  - Numeric columns: {len(analysis['numeric_columns'])}")
    print(f"  - Categorical columns: {len(analysis['categorical_columns'])}")
    print(f"  - Suggested visualizations: {len(analysis['suggested_visualizations'])}")
    return state


def create_visual_plan(state: DashState) -> DashState:
    """Create comprehensive visual plan with design specifications."""
    print("\n🎨 Creating comprehensive visual plan...")

    analysis = state["data_analysis"]
    theme_name = select_optimal_theme(analysis)
    theme = DESIGN_TEMPLATES[theme_name]

    visual_plan_prompt = textwrap.dedent(
        f"""You are an expert data visualization designer. Create a comprehensive visual plan for a Streamlit dashboard.
    
**Data Analysis:**
- Shape: {analysis['shape']}
- Columns: {analysis['columns']}
- Numeric columns: {analysis['numeric_columns']}
- Categorical columns: {analysis['categorical_columns']}
- Datetime columns: {analysis['datetime_columns']}
- Suggested visualizations: {json.dumps(analysis['suggested_visualizations'], indent=2)}

**Business Question:** {state['prompt']}

**Selected Theme:** {theme_name}
**Theme Colors:** {theme['color_palette']}

Create a JSON plan with the following structure:
{{
    "title": "Dashboard title",
    "subtitle": "Brief description",
    "theme": "{theme_name}",
    "layout": {{
        "type": "multi_column",
        "columns": [2, 1]
    }},
    "kpis": [
        {{
            "title": "KPI Name",
            "value": "column_name_or_calculation",
            "format": "number|percentage|currency",
            "description": "Brief description"
        }}
    ],
    "charts": [
        {{
            "type": "line|bar|scatter|pie|histogram|heatmap|box",
            "title": "Chart Title",
            "x": "column_name",
            "y": "column_name",
            "color": "optional_color_column",
            "aggregation": "sum|mean|count|none",
            "interactive": true,
            "column_span": 1,
            "height": 400
        }}
    ],
    "filters": [
        {{
            "column": "column_name",
            "type": "selectbox|multiselect|slider|date_range",
            "label": "Filter Label"
        }}
    ]
}}

Return only the JSON plan, no explanations."""
    )

    try:
        raw_plan = planner_llm.invoke(
            [{"role": "system", "content": visual_plan_prompt}]
        ).content.strip()
        json_match = re.search(r"```(?:json)?\n(.*?)\n```", raw_plan, re.DOTALL)
        if json_match:
            raw_plan = json_match.group(1)

        plan = json.loads(raw_plan)
        plan["theme_config"] = theme
        state["visual_plan"] = plan

        print(
            f"✅ Visual plan created with {len(plan.get('charts', []))} charts and {len(plan.get('kpis', []))} KPIs"
        )

    except json.JSONDecodeError as e:
        print(f"❌ Failed to parse visual plan JSON: {e}")
        # Fallback to basic plan
        state["visual_plan"] = {
            "title": "Data Dashboard",
            "theme": theme_name,
            "theme_config": theme,
            "charts": analysis["suggested_visualizations"][:4],
            "kpis": [
                {
                    "title": "Total Records",
                    "value": len(state["df"]),
                    "format": "number",
                }
            ],
        }

    return state


def generate_enhanced_code(state: DashState) -> DashState:
    """Generate beautiful, modern Streamlit code with advanced visualizations."""
    print("\n💻 Generating enhanced Streamlit code...")

    plan = state["visual_plan"]
    theme = plan["theme_config"]

    code_generation_prompt = textwrap.dedent(
        f"""You are an expert Streamlit developer specializing in beautiful, modern data dashboards.
Create a production-ready Streamlit application that implements the visual plan below.

**Visual Plan:**
{json.dumps(plan, indent=2)}

**Requirements:**
1. **Modern Design:** Implement the specified theme with custom CSS
2. **Beautiful Charts:** Use Plotly with custom styling, animations, and interactivity
3. **Responsive Layout:** Multi-column layouts that work on mobile and desktop
4. **Interactive Elements:** Filters, hover effects, and dynamic updates
5. **Performance:** Efficient data loading and caching
6. **Error Handling:** Graceful handling of missing data

**Technical Specifications:**
- Use `pd.read_csv('result.csv')` to load data
- Apply the theme colors: {theme['color_palette']}
- Background: {theme['background_color']}
- Text color: {theme['text_color']}
- Use Plotly for all charts with custom styling
- Include proper error handling
- Add loading states and progress indicators

**Chart Styling Template:**
```python
fig.update_layout(
    template="plotly_white",
    font=dict(family="{theme['font_family']}", color="{theme['text_color']}"),
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    colorway={theme['color_palette']},
    title=dict(font=dict(size=20, color="{theme['text_color']}")),
    margin=dict(t=60, b=50, l=50, r=50),
    hoverlabel=dict(bgcolor="white", font_size=12)
)
```

Create a complete, working Streamlit application. Return only the Python code."""
    )

    try:
        raw_code = coder_llm.invoke(
            [{"role": "system", "content": code_generation_prompt}]
        ).content.strip()
        state["streamlit_code"] = extract_code_from_markdown(raw_code)
        print("✅ Enhanced Streamlit code generated")
    except Exception as e:
        print(f"❌ Code generation failed: {e}")
        state["error_log"] = f"Code generation failed: {str(e)}"

    return state


def smoke_test(state: DashState) -> DashState:
    """Test the generated Streamlit application with v2.3 robustness."""
    print("\n🧪 Testing Streamlit application...")

    # Save files
    Path("result.csv").write_text(state["df"].to_csv(index=False), encoding="utf-8")
    Path("dashboard_app.py").write_text(state["streamlit_code"], encoding="utf-8")

    try:
        # Quick syntax check
        compile(state["streamlit_code"], "dashboard_app.py", "exec")

        # Run streamlit with timeout
        proc = subprocess.run(
            [
                "streamlit",
                "run",
                "dashboard_app.py",
                "--server.headless",
                "true",
                "--server.port",
                "8502",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=TIMEOUT,
        )

        if proc.returncode == 0:
            state["error_log"] = None
            print("✅ Streamlit application test passed")
        else:
            state["error_log"] = f"Streamlit error: {proc.stderr}"
            print(f"❌ Streamlit test failed: {proc.stderr}")

    except subprocess.TimeoutExpired:
        state["error_log"] = f"Streamlit timeout after {TIMEOUT}s"
        print(f"⏱️ Streamlit test timed out after {TIMEOUT}s")
    except SyntaxError as e:
        state["error_log"] = f"Syntax error in generated code: {str(e)}"
        print(f"❌ Syntax error: {e}")
    except Exception as e:
        state["error_log"] = f"Test failed: {str(e)}"
        print(f"❌ Test failed: {e}")

    return state


def repair_code(state: DashState) -> DashState:
    """Repair issues in the generated Streamlit code."""
    print(f"\n🔧 Repairing Streamlit code. Retry {state['retries'] + 1}/{MAX_RETRIES}")
    print(f"  - Error: {state['error_log']}")

    repair_prompt = textwrap.dedent(
        f"""Fix the following Streamlit code. The error log and original code are provided below.

**Error Log:**
{state['error_log']}

**Original Code:**
{state['streamlit_code']}

**Visual Plan for Reference:**
{json.dumps(state['visual_plan'], indent=2)}

**Instructions:**
1. Fix all syntax errors and runtime issues
2. Ensure all imports are correct and available
3. Handle missing data gracefully
4. Maintain the visual design and theme
5. Use only standard libraries: streamlit, pandas, plotly, numpy

Return only the corrected Python code."""
    )

    try:
        raw_code = repair_llm.invoke(
            [{"role": "system", "content": repair_prompt}]
        ).content.strip()
        state["streamlit_code"] = extract_code_from_markdown(raw_code)
        state["retries"] += 1
        print("✅ Code repair completed")
    except Exception as e:
        print(f"❌ Code repair failed: {e}")
        state["retries"] += 1

    return state


def save_artifacts(state: DashState) -> DashState:
    """Save all generated artifacts with comprehensive output."""
    print("\n💾 Saving final artifacts...")

    # Save main files
    Path("query.sql").write_text(state["sql_query"], encoding="utf-8")
    Path("dashboard_app.py").write_text(state["streamlit_code"], encoding="utf-8")
    Path("result.csv").write_text(state["df"].to_csv(index=False), encoding="utf-8")

    # Save additional artifacts
    if state.get("visual_plan"):
        Path("visual_plan.json").write_text(
            json.dumps(state["visual_plan"], indent=2), encoding="utf-8"
        )

    if state.get("data_analysis"):
        Path("data_analysis.json").write_text(
            json.dumps(state["data_analysis"], indent=2, default=str), encoding="utf-8"
        )

    print("✅ All artifacts saved successfully!")
    print("\n🚀 Your beautiful dashboard is ready!")
    print("   Run: streamlit run dashboard_app.py")
    print("   Files created:")
    print("   - dashboard_app.py (main application)")
    print("   - result.csv (data)")
    print("   - query.sql (SQL query)")
    print("   - visual_plan.json (design specifications)")
    print("   - data_analysis.json (data insights)")

    return state


# ---------------------------------------------------------------------------
# 8. Build and compile Enhanced LangGraph (Combined workflow)
# ---------------------------------------------------------------------------
def route_after_sql(state: DashState) -> str:
    """Route to data analysis if SQL is good, fix_sql if fixable, or end."""
    if state.get("error_log"):
        if state.get("retries", 0) < MAX_RETRIES:
            return "fix_sql"
        print("❌ SQL query failed after max retries. Aborting.")
        return END
    return "analyze"


def route_after_test(state: DashState) -> str:
    """Route to repair if there are fixable errors, otherwise save."""
    if state.get("error_log") and state.get("retries", 0) < MAX_RETRIES:
        return "fix_code"
    return "save"


# Build the combined workflow
workflow = StateGraph(DashState)

# Add all nodes
workflow.add_node("ask", ask_user)
workflow.add_node("sql", generate_sql)
workflow.add_node("run", run_query)
workflow.add_node("fix_sql", fix_sql)
workflow.add_node("analyze", analyze_data)
workflow.add_node("plan", create_visual_plan)
workflow.add_node("code", generate_enhanced_code)
workflow.add_node("test", smoke_test)
workflow.add_node("fix_code", repair_code)
workflow.add_node("save", save_artifacts)

# Set entry point
workflow.set_entry_point("ask")

# Add edges
workflow.add_edge("ask", "sql")
workflow.add_edge("sql", "run")

# SQL repair loop
workflow.add_conditional_edges(
    "run",
    route_after_sql,
    {"fix_sql": "fix_sql", "analyze": "analyze", END: END},
)
workflow.add_edge("fix_sql", "run")

# Main flow
workflow.add_edge("analyze", "plan")
workflow.add_edge("plan", "code")
workflow.add_edge("code", "test")

# Code repair loop
workflow.add_conditional_edges(
    "test",
    route_after_test,
    {"fix_code": "fix_code", "save": "save"},
)
workflow.add_edge("fix_code", "test")

# Final edge
workflow.add_edge("save", END)

# Compile the workflow
app = workflow.compile()

# ---------------------------------------------------------------------------
# 9. Run the app
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    initial_state: DashState = {
        "prompt": "",
        "sql_query": None,
        "df": None,
        "data_analysis": None,
        "visual_plan": None,
        "streamlit_code": None,
        "error_log": None,
        "retries": 0,
    }

    final_state = app.invoke(initial_state)
    print("\n🎉 Workflow completed.")
