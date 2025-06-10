<!-- README.md -->
# LangGraph ↔ BigQuery ↔ Streamlit Dashboard Demo &nbsp;🚀

<div align="center">

**End‑to‑end Gen‑AI analytics stack in < 200 lines of Python**

| Stage | What happens |
|-------|--------------|
| **① Prompt** | User asks a business question in the terminal |
| **② LLM** | LangGraph + GPT‑4o (or Gemini / Llama 3) writes governed BigQuery SQL |
| **③ Warehouse** | Query runs, pandas `DataFrame` comes back |
| **④ LLM** | Same model autogenerates a polished Streamlit dashboard |
| **⑤ Artifacts** | `dashboard_app.py`, `result.csv`, `query.sql` and a mock “Git push” |

</div>

---

## ✨ Features

- **Governed SQL generation** – Limits the LLM to the dataset you specify  
- **Drive‑backed sheets ready** – Injects Drive scope so external tables “just work”  
- **One‑click dashboard** – KPI cards, Ag‑Grid table, auto‑charts, corporate‑blue theme  
- **Pluggable LLM** – GPT‑4o by default, drop‑in support for Gemini 1.5 Pro or Llama 3    
- **CI‑friendly** – All artifacts saved to disk; hook into your Git/CI pipeline

---

## 🗺️ Repo layout

```text
.
├── langgraph_streamlit_dashboard.py   ← main script
├── dashboard_app.py                  ← generated Streamlit app (git‑ignored)
├── result.csv                         ← query result (git‑ignored)
├── query.sql                          ← generated SQL (git‑ignored)
├── .env.example                       ← template for secrets
├── .gitignore
└── README.md
