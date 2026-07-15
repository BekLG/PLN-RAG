# PLN-RAG

A REST API service for Probabilistic Logic Network (PLN) based retrieval-augmented reasoning.
Ingests natural language text, converts it to PLN atoms via a pluggable semantic parser,
stores facts in a PeTTaChainer atomspace, and answers questions via logical proof.

## Architecture

```text
Text -> Chunker -> mention prepass -> LangExtract -> PLN postprocessor
                                                    |
                                                    v
                                      SQLite evidence ledger
                                      /                    \
                         Qdrant claim index          PeTTaChainer
                                 |                         |
Question -> evidence retrieval -> gated target -> proof -> Answer
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
explicit negation conflicts. Only approved exact/directional entailment
mappings become PeTTa bridge rules. Related mappings remain retrieval/debug
metadata and cannot enter proofs.

## Prerequisites

Ollama runs on your **host machine**, not inside Docker. Install it and pull the embedding model before starting the service.
This is only required for the non-light profiles:

```bash
# Install Ollama (Linux)
curl -fsSL https://ollama.com/install.sh | sh

# Pull the embedding model (once — persists in ~/.ollama)
ollama pull nomic-embed-text

# Verify it is running
curl http://localhost:11434    # should return "Ollama is running"
```

## Quick start (Docker)

```bash
cp .env.example .env
# Fill in OPENAI_API_KEY or GEMINI_API_KEY
# OLLAMA_URL can stay as localhost in .env; docker-compose overrides it for containers

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

> **Linux note:** `host.docker.internal` is not automatically available on Linux.
> The `docker-compose.yml` already includes `extra_hosts: host.docker.internal:host-gateway`
> to handle this. No manual action needed.

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
- LangExtract Docker profile: http://localhost:8001
- PLN Docker profile: http://localhost:8000

Debug endpoints used by the UI:
- POST /debug/ingest
- POST /debug/query

## Switching parsers

Set `PARSER` in `.env` for local runs. For Docker Compose, use the profile
plus parser-specific env vars to avoid accidental overrides from a shell-level
`PARSER` variable.

```bash
# Use NL2PLN (DSPy-based, SIMBA/GEPA optimized)
PARSER=nl2pln
PLNRAG_PLN_PARSER=nl2pln
NL2PLN_MODULE_PATH=data/simba_all.json

# Use CanonicalPLN parser (separate tuned SIMBA artifact)
PARSER=canonical_pln
PLNRAG_PLN_PARSER=canonical_pln
CANONICAL_PLN_NL2PLN_MODULE_PATH=data/simba_canonical_pln.json

# Use Manhin's parser (format self-correction + FAISS predicate store)
PARSER=manhin
PLNRAG_PLN_PARSER=manhin

# Use LangExtract parser (NL -> LangExtract objects -> canonical PLN)
PARSER=langextract
LANGEXTRACT_API_KEY=your-gemini-api-key-here
LANGEXTRACT_MODEL_ID=gemini-2.5-flash
LANGEXTRACT_EXAMPLES_PATH=data/langextract_examples.json
```

`nl2pln` and `canonical_pln` intentionally use separate compiled artifacts so baseline
comparisons stay clean. Tune `data/simba_canonical_pln.json` without modifying the
baseline `simba_all.json`.

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
PLN postprocessor used by the canonical parser. It intentionally skips the
Hyperon MeTTa runtime because the PLN-RAG reasoner consumes PeTTa-style PLN
directly. Qdrant retrieval is context-only: retrieved atoms are not executable
query targets. Query and debug-query operations are read-only and cannot add
evidence to the atomspace.

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

To add a new parser:
1. Create `parsers/your_parser.py` implementing `SemanticParser`
2. Register it in `parsers/__init__.py`
3. Set `PARSER=your_parser` in `.env`

## Local parser code (not yet on GitHub)

For parsers still in local development, mount them as volumes rather than cloning:

```yaml
# docker-compose.yml
volumes:
  - ./local-deps/manhin-parser:/deps/manhin-parser:ro
```

On your host, symlink your working directory:
```bash
mkdir -p local-deps
ln -s /path/to/your/manhin-parser local-deps/manhin-parser
```

Changes are reflected immediately without rebuilding the image. When the parser is
published to GitHub, swap the volume mount for a `git clone` in the Dockerfile.

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

For Docker builds, PeTTaChainer is pinned to commit `6b88df7c903705a38205709151cdd7549fd8d1b0`, which is a reachable ref on the current upstream repository.

# 5. Install pln-rag deps
pip install -r requirements.txt

# 6. Configure
cp .env.example .env
# Fill in OPENAI_API_KEY
# OLLAMA_URL defaults to http://localhost:11434/api/embeddings — no change needed

# 7. Run
uvicorn api.main:app --reload
```

## Compare parser outputs

Use `compare_parsers.py` to inspect how `nl2pln`, `canonical_pln`, and `manhin`
translate the same input:

```bash
python compare_parsers.py \
  --mode query \
  --text "Is Kebede smart?" \
  --context "(Inheritance Kebede Human)" \
  --context "(Implication (Inheritance $x Human) (Inheritance $x Smart))"
```

The script prints JSON and marks parsers as unavailable if their dependencies are
not installed in the current environment.

## Dependency notes

**janus_swi must always be built from source.**
The pip wheel is compiled against a specific SWI-Prolog ABI version.
If it does not match the installed SWI-Prolog, you will get:
```
janus_swi.janus.PrologError: <exception str() failed>
```
The fix is always: `git clone https://github.com/SWI-Prolog/packages-swipy && pip install .`

**Ollama must be running on the host before starting the service.**
The container reaches it via `host.docker.internal:11434`. If Ollama is not running,
ingest and query requests will fail with a connection refused error.

## Data persistence

| Path | Contents | Backed by |
|------|----------|-----------|
| `data/atomspace/kb.metta` locally, `/app/data/atomspace/kb.metta` in Docker | PLN atoms (facts + rules) | file, loaded on startup |
| `data/predicate_registry.json` | Predicate cards and typed mapping graph | JSON file |
| `data/faiss/` | Predicate embeddings (Manhin parser) | FAISS index files |
| Qdrant chunk collection | NL ↔ PLN chunk mappings | Docker volume |
| Qdrant predicate collection | Embedded predicate cards | Docker volume |
| `~/.ollama` | Embedding model weights | host machine |

Data survives container restarts via the `pln_data` Docker volume.
Ollama model weights live on your host and never need to be re-pulled.
