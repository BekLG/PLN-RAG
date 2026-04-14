# PLN-RAG

A REST API service for Probabilistic Logic Network (PLN) based retrieval-augmented reasoning.
Ingests natural language text, converts it to PLN atoms via a pluggable semantic parser,
stores facts in a PeTTaChainer atomspace, and answers questions via logical proof.

## Architecture

```
Text → Chunker → SemanticParser → PeTTaChainer (atomspace + reasoning) → AnswerGenerator → Response
                      ↑
              Qdrant context retrieval
```

## Quick start (Docker)

```bash
cp .env.example .env
# Fill in OPENAI_API_KEY in .env

docker compose up --build

# Pull the embedding model (once)
docker compose exec ollama ollama pull nomic-embed-text
```

The API will be available at http://localhost:8000.
Interactive docs at http://localhost:8000/docs.

## API endpoints

### POST /ingest
Ingest texts into the knowledge base.
```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"texts": ["People who eat fish are smart.", "Kebede eats fish."]}'
```

### POST /query
Ask a question against the knowledge base.
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Is Kebede smart?"}'
```

### DELETE /reset
Clear the knowledge base (fully or partially).
```bash
# Clear everything
curl -X DELETE http://localhost:8000/reset \
  -H "Content-Type: application/json" \
  -d '{"scope": "all"}'

# Clear only vector DB (re-index without losing atomspace)
curl -X DELETE http://localhost:8000/reset \
  -H "Content-Type: application/json" \
  -d '{"scope": "vectordb"}'
```

### GET /health
```bash
curl http://localhost:8000/health
```

## Switching parsers

Set `PARSER` in `.env` — no code changes needed:

```bash
# Use NL2PLN (DSPy-based, SIMBA/GEPA optimized)
PARSER=nl2pln
NL2PLN_MODULE_PATH=data/simba_all.json

# Use Manhin's parser (format self-correction + FAISS predicate store)
PARSER=manhin
```

To add a new parser:
1. Create `parsers/your_parser.py` implementing `SemanticParser`
2. Register it in `parsers/__init__.py`
3. Set `PARSER=your_parser` in `.env`

## Local development (without Docker)

```bash
# 1. Install SWI-Prolog 9.x
sudo add-apt-repository ppa:swi-prolog/stable
sudo apt-get install swi-prolog

# 2. Build janus_swi from source (NEVER use pip install janus-swi)
git clone https://github.com/SWI-Prolog/packages-swipy
cd packages-swipy && pip install .
cd ..

# 3. Clone and install PeTTa + PeTTaChainer
git clone https://github.com/trueagi-io/PeTTa.git
git clone https://github.com/rTreutlein/PeTTaChainer.git

cd PeTTa
sed -i "/'janus-swi'/d" setup.py   # remove pip janus-swi dep
pip install -e .
cd ..

cd PeTTaChainer && pip install -e . && cd ..

# 4. Install pln-rag deps
pip install -r requirements.txt

# 5. Configure
cp .env.example .env
# Fill in OPENAI_API_KEY

# 6. Run
uvicorn api.main:app --reload
```

## Dependency notes

**janus_swi must always be built from source.**
The pip wheel is compiled against a specific SWI-Prolog ABI version.
If it does not match the installed SWI-Prolog, you will get:
```
janus_swi.janus.PrologError: <exception str() failed>
```
The fix is always: `git clone https://github.com/SWI-Prolog/packages-swipy && pip install .`

## Data persistence

| Path | Contents | Backed by |
|------|----------|-----------|
| `data/atomspace/kb.metta` | PLN atoms (facts + rules) | file, loaded on startup |
| `data/faiss/` | Predicate embeddings (Manhin parser) | FAISS index files |
| Qdrant volume | NL ↔ PLN sentence mappings | Docker volume |

Data survives container restarts via the `pln_data` Docker volume.
