# PLN-RAG

A REST API service for Probabilistic Logic Network (PLN) based retrieval-augmented reasoning.
Ingests natural language text, converts it to PLN atoms via a pluggable semantic parser,
stores facts in a PeTTaChainer atomspace, and answers questions via logical proof.

## Architecture

```text
Text -> optional coreference -> chunker -> mention prepass -> LangExtract
                                                           |
                                                           v
                                                 PLN postprocessor
                                                           |
                                                           v
                                             SQLite evidence ledger
                                             /                    \
                                Qdrant claim index          PeTTaChainer
                                        |                         |
Question -> evidence retrieval -> gated query target -> proof -> Answer
```

SQLite is the authoritative store for documents, exact evidence spans, claims,
and transformation lineage. Qdrant contains rebuildable natural-language search
records, one per validated claim/evidence target. PeTTaChainer is rebuilt from
accepted and traceable derived claims. Retrieval proposes proof targets but
never establishes truth by similarity.

## Project layout

| Path | Purpose |
|------|---------|
| `api/` | FastAPI routes and response/request models |
| `core/` | Runtime pipeline: extraction, PLN cleanup, query planning, reasoning, and answering |
| `parsers/` | LangExtract parser integration |
| `storage/` | SQLite evidence ledger and Qdrant/Ollama index adapters |
| `debug_ui/` | Streamlit inspection UI |
| `tests/` | Automatic unit and safety tests |
| `tests/manual/` | Manual experiments that may require live LLM/Ollama/Qdrant services |
| `docs/` | Architecture notes, fix notes, and research references |

### Dynamic predicate mapping

Ingestion creates a predicate card for each fact, rule premise, and rule
conclusion. Predicate cards include arity, argument types, source examples, and
the originating atom. They are embedded in a separate Qdrant collection so new
predicates can retrieve semantically similar predicates without a hardcoded
domain synonym list.

An LLM/NLI classifier proposes one typed relation: `exactMatch`,
`source_implies_target`, `target_implies_source`, `broader`, `narrower`,
`related`, `contradiction`, or `unrelated`. Deterministic validation checks
arity, ordered argument types, relation allow-list, confidence threshold, and
explicit negation conflicts. Approved exact/directional entailment mappings are
recorded as metadata by default. They become PeTTa bridge rules only when
`PREDICATE_MAPPING_EMIT_BRIDGES=true`. Related mappings remain retrieval/debug
metadata and cannot enter proofs.

## Prerequisites

For Docker demos, you need Docker Compose and a Gemini API key. The default
Compose profile starts the API, Qdrant, and an Ollama container for embeddings.

For local runs without Docker, install Ollama on the host and pull the embedding
model once:

```bash
ollama pull nomic-embed-text
curl http://localhost:11434
```

## Quick start (Docker)

```bash
cp .env.example .env
# Fill in GEMINI_API_KEY

# Full LangExtract stack: API, Qdrant, and Ollama
docker compose --profile default up --build

# API-only light profile without Qdrant/Ollama retrieval
docker compose --profile light up --build

```

The full API is available at http://localhost:8000. The light profile uses
http://localhost:8001. Interactive docs are available at `/docs` on either
port.

### Rebuild derived indexes

After changing Qdrant schema or recovering a deleted Atomspace/Qdrant index,
rebuild both projections from SQLite:

```bash
curl -X POST http://localhost:8000/rebuild
```

Legacy chunk-level Qdrant points are intentionally excluded from executable
query alignment. Re-ingest source documents to create evidence-linked v2
records in `pln_rag_evidence_v2`.

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

## Debug UI (Streamlit)

Run the Streamlit app to inspect step-by-step outputs (LangExtract post-processed
results, canonicalized PLN, PeTTaChainer atoms, and reasoning proof):

```bash
pip install -r requirements.common.txt
streamlit run debug_ui/app.py
```

Set the API base URL in the UI to match your running profile:
- Default Docker profile: http://localhost:8000
- Light Docker profile: http://localhost:8001

Debug endpoints used by the UI:
- POST /debug/ingest
- POST /debug/query

## LangExtract parser

This branch uses the LangExtract parser path:

```bash
GEMINI_API_KEY=your-gemini-api-key-here
LANGEXTRACT_MODEL_ID=gemini-2.5-flash
LANGEXTRACT_EXAMPLES_PATH=data/langextract_examples.json
MENTION_PREPASS_ENABLED=true
```

The `langextract` parser follows a direct PLN-RAG path:

```text
Natural language
-> LangExtract-style chunker
-> mention prepass
-> LangExtract extraction objects
-> proof-safety and predicate-schema validation
-> source-aware canonical PLN statements
-> PeTTaChainer

Question
-> typed intent (boolean/open/factors/explanation/sufficiency)
-> Qdrant context and vocabulary retrieval
-> parser query candidates
-> intent and arity gate
-> positive and explicit-negative proof checks
-> proof-backed answer and exact source provenance
```

It mirrors the useful parts of the standalone `lang-extract` project inside
PLN-RAG: JSON-backed examples, paragraph/sentence-aware chunking, source-aware
canonicalization, fuzzy/unsafe extraction rejection, predicate vocabulary reuse
across chunks, source metadata for translated statements, and the same shared
PLN postprocessor used by the reasoning pipeline. It intentionally skips the
Hyperon MeTTa runtime because the PLN-RAG reasoner consumes PeTTa-style PLN
directly. Qdrant retrieval proposes evidence-linked query targets from accepted
claim records only; those targets still pass entity, predicate, arity, polarity,
and question-intent gates before execution. Query and debug-query operations
are read-only and cannot add evidence to the atomspace.

The shared PLN postprocessor lives in `core/pln/postprocessor.py`. It performs
the final reasoning-readiness pass for parser outputs: canonicalization,
statement filtering, weak premise pruning, portion-arity repair, conflicting
arity rejection, and query planning. It never materializes missing rule premises.

Query fallback execution can be toggled independently at runtime:

```bash
QUERY_FALLBACK_ENABLED=true
```

When disabled, the service runs only the first validated parser query. When
enabled, it may try later parser candidates, but every retry passes through the
same typed intent gate. Boolean queries return one of four proof states:
`positive`, `negative`, `both`, or `unknown`.

### Optional document-level coreference

LangExtract ingestion can optionally run LingMess coreference once per original
document, project cluster mentions into each chunk by character offsets, and
merge those clusters into the existing mention prepass as prompt hints. It does
not rewrite source text.

```bash
python -m pip install -r requirements-coref.txt
COREFERENCE_ENABLED=true
COREFERENCE_DEVICE=auto
```

For Docker, install the optional LingMess dependency during the image build:

```powershell
$env:PLNRAG_INSTALL_COREF="true"
$env:COREFERENCE_ENABLED="true"
docker compose --profile default up --build pln-rag
```

The default is `COREFERENCE_ENABLED=false`. If LingMess is unavailable or fails
and `COREFERENCE_FAIL_OPEN=true`, ingestion continues with deterministic mention
prepass only. See `docs/architecture/coreference.md` for details.

## Local development (without Docker)

```bash
# 1. Install Ollama and pull the embedding model
curl -fsSL https://ollama.com/install.sh | sh
ollama pull nomic-embed-text

# 2. Install SWI-Prolog 9.x
sudo add-apt-repository ppa:swi-prolog/stable
sudo apt-get install swi-prolog

# 3. Build janus_swi from source (NEVER use pip install janus-swi)
git clone https://github.com/SWI-Prolog/packages-swipy
cd packages-swipy && pip install .
cd ..

# 4. Clone and install PeTTa + PeTTaChainer
git clone https://github.com/trueagi-io/PeTTa.git
git clone https://github.com/rTreutlein/PeTTaChainer.git

cd PeTTa
sed -i "/'janus-swi'/d" setup.py   # remove the broken pip janus-swi dep
pip install -e .
cd ..

cd PeTTaChainer && pip install -e . && cd ..
```

For Docker builds, PeTTaChainer is pinned to commit `6b88df7c903705a38205709151cdd7549fd8d1b0`, which is a reachable ref on the current upstream repository.

```bash
# 5. Install pln-rag deps
pip install -r requirements.txt

# 6. Configure
cp .env.example .env
# Fill in GEMINI_API_KEY
# OLLAMA_URL defaults to http://localhost:11434/api/embeddings

# 7. Run
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

**Ollama is required only when vector retrieval is enabled.**
The default Docker profile runs Ollama as a container and persists model weights
in the `ollama_data` volume. For local runs without Docker, run Ollama on the
host at `http://localhost:11434`.

## Data persistence

| Path | Contents | Backed by |
|------|----------|-----------|
| `data/atomspace/kb.metta` locally, `/app/data/atomspace/kb.metta` in Docker | PLN atoms (facts + rules) | file, loaded on startup |
| `data/predicate_registry.json` | Predicate cards and typed mapping graph | JSON file |
| `data/evidence/evidence.db` locally, `/app/data/evidence/evidence.db` in Docker | Documents, evidence spans, claim lineage, and indexing outbox | SQLite |
| Qdrant evidence collection | Accepted source-linked claim/query-target records | Docker volume |
| Qdrant predicate collection | Embedded predicate cards | Docker volume |
| `ollama_data` Docker volume or host `~/.ollama` | Embedding model weights | Docker or local Ollama |

Data survives container restarts via the `pln_data` Docker volume.
Ollama model weights survive in `ollama_data` for Docker runs or `~/.ollama`
for local runs.
