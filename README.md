# GlowAgent 

**GlowAgent** is an AI-powered skincare recommendation assistant built for U.S.-based users aged 18–24 who want effective, budget-conscious skincare routines without hours of research. It combines a local product/ingredient database, real-time web search, and a multi-tool LangGraph agent to deliver personalized recommendations through a clean web UI.

---

## Features

- **Personalized Routine Builder** — Generates step-by-step routines (Cleanser → Treatment → Moisturizer → Sunscreen) based on skin type, concerns, allergies, and budget
- **Product Recommendations** — Searches and ranks products from a curated local database
- **Ingredient Lookup** — Verifies full INCI ingredient lists and allergen flags via Open Beauty Facts
- **Live Price Search** — Fetches current prices from Ulta, Sephora, Amazon, Walmart, and Target via Tavily
- **Product Ranking** — Scores and orders products by relevance to the user's skin type, concerns, and category
- **Conversational Memory** — Maintains per-session context using LangGraph's `InMemorySaver`
- **Web UI** — Served directly from FastAPI via `static/index.html`

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | Google Gemini 2.0 Flash (`gemini-2.0-flash`) |
| Embeddings | Google Gemini Embeddings (`gemini-embedding-001`) |
| Vector Store | ChromaDB (persisted locally at `chroma_db/`) |
| Agent Framework | LangGraph (`create_react_agent`) |
| Web Framework | FastAPI + Uvicorn |
| Web Search | Tavily API |
| Ingredient DB | Open Beauty Facts API (no key required) |
| Data | Pandas (CSV-based product & ingredient datasets) |

---

## Project Structure

```
GlowAgent/
├── app/
│   ├── __init__.py
│   ├── build_vectorstore.py   # One-time script to embed CSVs into ChromaDB
│   ├── glow_agent.py          # Agent definition, tools, LLM, memory
│   └── main.py                # FastAPI app and /chat endpoint
├── chroma_db/                 # Persisted ChromaDB vector store (auto-generated)
│   ├── 50c3068e-9d67-433f-97c3-7d8cee499f47/
│   └── chroma.sqlite3
├── data/
│   ├── skincare_ingredients.csv
│   └── skincare_products.csv
├── evals/
│   ├── README.md
│   ├── __init__.py
│   ├── evaluate.py
│   └── product_ranking_eval.py
├── static/
│   └── index.html             # Chat UI served at /ui
├── .env                       # API keys (not committed)
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Prerequisites

- Python 3.10+
- A **Google AI API key** (for Gemini LLM + embeddings) — [Get one here](https://aistudio.google.com/app/apikey)
- A **Tavily API key** (for live price search) — [Get one here](https://app.tavily.com)

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/glowagent.git
cd glowagent
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_google_ai_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

### 5. Build the vector store

This step only needs to be run **once**. It embeds the product and ingredient CSVs into ChromaDB:

```bash
python -m app.build_vectorstore
```

You should see:
```
Built Chroma DB at: /path/to/glowagent/chroma_db
```

> **Note:** Skip this step if the `chroma_db/` folder is already present in the repo.

---

## Running the App

```bash
uvicorn app.main:app --reload
```

Then open your browser and go to:

```
http://localhost:8000/ui
```

The API docs are available at:

```
http://localhost:8000/docs
```

---

## API Reference

### `POST /chat`

Send a message to GlowAgent and receive a skincare recommendation.

**Request body:**

```json
{
  "message": "Recommend a moisturizer for oily acne-prone skin under $20.",
  "thread_id": "optional-session-id-for-memory"
}
```

**Response:**

```json
{
  "response": "Here are my top recommendations for oily acne-prone skin...",
  "thread_id": "abc123"
}
```

Pass the same `thread_id` across requests to maintain conversation context.

### `GET /health`

Returns `{ "status": "ok" }` — useful for deployment health checks.

---

## Agent Tools

GlowAgent uses four tools in priority order:

| Priority | Tool | Purpose |
|---|---|---|
| 1st | `skincare_database_search` | Searches local ChromaDB for products by skin type, concern, category, budget |
| 2nd | `product_ranking_tool` | Scores and ranks candidates from the local DB by relevance |
| 3rd | `open_beauty_facts_search` | Fetches full ingredient lists and allergen data from Open Beauty Facts |
| 4th | `tavily_search_tool` | Live web search for current prices, availability, and recent reviews |

---

## Data

The agent is powered by two CSV files in the `data/` directory:

- **`skincare_products.csv`** — Product name, brand, type, notable effects, skin type flags (Oily, Dry, Sensitive, Combination, Normal), product link, and image URL
- **`skincare_ingredients.csv`** — Ingredient name, description, what it does, who it suits, who should avoid it, and source URL

These are embedded into ChromaDB at startup via `build_vectorstore.py`.

---

## Evals

The `evals/` directory contains evaluation scripts for testing agent output quality:

- `evaluate.py` — General evaluation runner
- `product_ranking_eval.py` — Evaluates the accuracy and ordering of product ranking results

See `evals/README.md` for instructions on running evaluations.

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GOOGLE_API_KEY` | ✅ Yes | Powers Gemini 2.0 Flash (LLM) and Gemini Embeddings |
| `TAVILY_API_KEY` | ✅ Yes | Powers live web price search via Tavily |

---

## Known Limitations

- The local product database is a static CSV snapshot; it does not update automatically
- `InMemorySaver` resets all conversation history when the server restarts
- Open Beauty Facts data is crowd-sourced and may be incomplete for some products
- Price data via Tavily reflects web search results and may not always be exact
