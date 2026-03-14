## 🆕 Version 2 — LLM & Semantic Search

Version 2 adds two new capabilities on top of the existing ML pipeline:

### 🔍 Semantic Car Search (`pages/semantic_search.py`)

Instead of rule-based keyword filters, the search page uses **sentence-transformer embeddings** to find cars that match a free-text description.

- Model: `sentence-transformers/all-MiniLM-L6-v2`
- Index: FAISS `IndexFlatIP` (cosine similarity on normalised vectors)
- Each car is converted to a natural-language sentence, embedded, and stored in the index
- User query is embedded at search time and compared against all 7,834 cars

Example queries:
```
"cheap electric car with low mileage"
"family SUV automatic diesel"
"recent Toyota under $8,000 in California"
```

### 💬 AI Chatbot (`pages/chatbot.py`)

A RAG (Retrieval-Augmented Generation) chatbot grounded in the dataset.

Architecture:
```
User question
     ↓
Retrieval router (semantic vs aggregate)
     ↓
Context string (top-k cars or dataset statistics)
     ↓
Prompt builder (system prompt + history + context + question)
     ↓
Anthropic Claude API (claude-sonnet-4-20250514)
     ↓
Grounded answer
```

Features:
- Maintains full conversation history (last 6 turns passed to LLM)
- Two retrieval modes: semantic (FAISS) for "find me cars like X" and aggregate (pandas) for "what is the average price of Y"
- Retrieved context shown optionally for transparency
- Hallucination guard: system prompt explicitly instructs the model to only use retrieved context

### New file structure

```
carvalue-ai/
├── src/
│   ├── __init__.py
│   ├── embeddings.py       # FAISS index builder + semantic_search()
│   ├── retrieval.py        # Context retrieval router (semantic / aggregate)
│   ├── chatbot_engine.py   # RAG pipeline + Anthropic API call
│   └── prompts.py          # System prompt + RAG template
├── pages/
│   ├── semantic_search.py  # Streamlit page: embedding search UI
│   └── chatbot.py          # Streamlit page: chatbot UI with history
└── model/
    └── faiss_index.pkl     # Pre-built FAISS index (auto-generated on first run)
```

### Streamlit Cloud setup for Chatbot

The chatbot uses an API Anthropic Key for the Chatbot which is hidden as a secret on Streamlit.

The key is automatically injected as an environment variable and picked up by `chatbot_engine.py`.