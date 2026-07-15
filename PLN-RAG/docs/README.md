# PLN-RAG Codebase Walkthrough

This document explains how the current PLN-RAG codebase works from start to
finish. It follows one dummy example through ingestion, LingMess coreference,
mention hints, LangExtract extraction, PLN conversion, predicate mapping,
storage, query planning, proof search, and final answer generation.

The most important rule is this:

```text
The system does not answer directly from the paragraph.
It answers only from stored PLN atoms and proof traces.
```

The high-level flow is:

```text
POST /ingest or /debug/ingest
-> PLNRAGService
-> optional LingMess document coreference
-> offset-aware chunking
-> mention prepass prompt hints
-> LangExtract extraction
-> PLN translation
-> PLN postprocessing
-> predicate registry and mapping
-> PeTTaChainer atomspace
-> Qdrant vector store

POST /query or /debug/query
-> retrieve context from Qdrant and atomspace
-> parse question intent
-> LangExtract query parsing
-> deterministic query candidates from known atoms
-> candidate filtering by intent and arity
-> positive and explicit-negative proof search
-> answer generated only from proof status
```

## Dummy Example

Use this paragraph:

```text
Alex is a person. The camera is a device.
If a person drops a device, the device becomes damaged.
If a device is damaged, it requires inspection.
Alex bought a camera. He dropped it.
```

Then ask:

```text
Does the camera require inspection?
```

The intended proof chain is:

```text
He -> Alex
it -> camera
Alex dropped camera
camera is a device
if person drops device -> device damaged
if device damaged -> requires inspection
therefore camera requires inspection
```

The system should answer yes only if it can build this proof from atoms.

## Main Files

| File | Responsibility |
| --- | --- |
| `api/main.py` | FastAPI endpoints: `/ingest`, `/query`, `/debug/ingest`, `/debug/query`, `/reset`, `/health`. |
| `api/models.py` | Request and response schemas returned by the API. |
| `config.py` | Runtime settings loaded from `.env` or environment variables. |
| `core/orchestration/service.py` | Main pipeline coordinator. This is the best starting point for understanding the system. |
| `core/discourse/lingmess_coreference.py` | Lazy wrapper around `fastcoref.LingMessCoref`. |
| `core/discourse/coreference.py` | Coreference data classes and safe canonical mention selection. |
| `core/discourse/mention_prepass.py` | Converts coreference and local mentions into prompt hints for LangExtract. |
| `core/extraction/langextract_chunker.py` | Splits long documents while preserving character offsets. |
| `parsers/langextract_pln_parser.py` | Calls LangExtract and passes extracted objects to the PLN translator and postprocessor. |
| `core/extraction/langextract_pln.py` | Converts LangExtract objects into PLN/MeTTa-style statements and queries. |
| `core/pln/postprocessor.py` | Normalizes, repairs, aligns, and filters PLN before it enters the reasoner. |
| `core/pln/predicate_registry.py` | Stores observed predicate cards and validated predicate mappings. |
| `core/pln/predicate_mapping.py` | Builds predicate cards and classifies predicate relationships conservatively. |
| `core/query/intent.py` | Parses question intent and filters unsafe query candidates. |
| `core/query/alignment.py` | Builds possible query targets from Qdrant matches, but does not prove them. |
| `core/reasoning/reasoner.py` | Owns atomspace persistence and proof search through PeTTaChainer plus strict checks. |
| `storage/vector_store.py` | Uses Ollama embeddings and Qdrant for chunk and predicate-card retrieval. |
| `core/answering/answer_generator.py` | Converts proof status into a natural-language answer. |

## Startup

When the API starts, `api/main.py` creates:

```python
parser = LangExtractPLNParser()
service = PLNRAGService(parser)
```

The parser loads:

```text
LangExtract model id
LangExtract API key or model URL
few-shot examples from data/langextract_examples.json
mention-prepass setting
predicate registry/mapping settings
PLN postprocessor
```

The service creates:

```text
Reasoner
VectorStore if enabled
AnswerGenerator
LingMess resolver if COREFERENCE_ENABLED=true
```

Important settings:

```text
LANGEXTRACT_MODEL_ID=gemini-2.5-flash
MENTION_PREPASS_ENABLED=true
COREFERENCE_ENABLED=false by default
COREFERENCE_BACKEND=lingmess
PREDICATE_REGISTRY_ENABLED=true
PREDICATE_MAPPING_ENABLED=true
PREDICATE_MAPPING_ONLINE_ENABLED=false by default
PREDICATE_MAPPING_EMIT_BRIDGES=false by default
STRICT_PROOF_VALIDATION_ENABLED=true
```

LingMess is optional because it is a large neural model. When enabled, it is the
document-level resolver. Mention prepass remains a safety and prompt-hint layer.

## Ingestion Step By Step

### 1. API Receives Text

Request:

```http
POST /debug/ingest
Content-Type: application/json
```

```json
{
  "texts": [
    "Alex is a person. The camera is a device. If a person drops a device, the device becomes damaged. If a device is damaged, it requires inspection. Alex bought a camera. He dropped it."
  ]
}
```

`api/main.py` calls:

```python
svc.debug_ingest_batch(req.texts)
```

For normal ingestion it calls:

```python
svc.ingest_batch(req.texts)
```

The debug endpoint returns the intermediate objects. The normal endpoint returns
only the final atoms and status.

### 2. Service Starts One Document Ingest

`core/orchestration/service.py` receives the text in `_debug_ingest_single()` or
`_ingest_single()`.

It prepares:

```text
text = full paragraph
chunks = service._chunk_document(text)
document_coref = service._resolve_document_coref(text)
```

No facts are added yet. This stage only prepares the document.

### 3. Chunking

The chunker receives the full text:

```text
Alex is a person. The camera is a device. If a person drops a device...
```

It returns `TextChunk` records:

```json
[
  {
    "index": 0,
    "text": "Alex is a person. The camera is a device. If a person drops a device, the device becomes damaged. If a device is damaged, it requires inspection. Alex bought a camera. He dropped it.",
    "start": 0,
    "end": 180
  }
]
```

For a long document there may be many chunks:

```json
[
  {"index": 0, "start": 0, "end": 2000},
  {"index": 1, "start": 1936, "end": 3900}
]
```

The start and end offsets matter because LingMess runs on the full document,
while LangExtract runs per chunk. The system needs offsets to project document
coreference into the correct chunk.

### 4. LingMess Document Coreference

If `COREFERENCE_ENABLED=false`, this step returns an empty result:

```json
{
  "backend": "none",
  "model_name": "",
  "clusters": []
}
```

If `COREFERENCE_ENABLED=true`, `LingMessCoreferenceResolver` receives the full
document:

```python
model.predict(
    texts=[text],
    max_tokens_in_batch=10000
)
```

LingMess returns character-offset clusters. Representative result:

```json
{
  "backend": "lingmess",
  "model_name": "biu-nlp/lingmess-coref",
  "clusters": [
    {
      "cluster_id": "lingmess_c1",
      "mentions": [
        {"text": "Alex", "start": 0, "end": 4, "is_pronoun": false},
        {"text": "Alex", "start": 143, "end": 147, "is_pronoun": false},
        {"text": "He", "start": 165, "end": 167, "is_pronoun": true}
      ],
      "canonical_mention": {"text": "Alex", "start": 0, "end": 4},
      "ambiguous": false
    },
    {
      "cluster_id": "lingmess_c2",
      "mentions": [
        {"text": "camera", "start": 25, "end": 31, "is_pronoun": false},
        {"text": "camera", "start": 157, "end": 163, "is_pronoun": false},
        {"text": "it", "start": 176, "end": 178, "is_pronoun": true}
      ],
      "canonical_mention": {"text": "camera", "start": 25, "end": 31},
      "ambiguous": false
    }
  ]
}
```

The code does not trust every cluster blindly. `coreference.py` chooses a safe
canonical mention:

```text
prefer non-pronouns
prefer proper names when there is no name conflict
mark conflicting names as ambiguous
leave pronoun-only clusters unresolved
```

Example of a risky text:

```text
Alex put the camera near the phone. It was broken.
```

If LingMess cannot safely decide between camera and phone, the cluster is marked
ambiguous and the system tells LangExtract not to guess.

### 5. Project Coreference Into The Chunk

The service calls:

```python
project_coref_to_chunk(document_coref, chunk.start, chunk.end, chunk.text)
```

For the dummy one-chunk document, the chunk result is almost identical to the
document result:

```json
{
  "chunk_start": 0,
  "chunk_end": 180,
  "backend": "lingmess",
  "clusters": [
    "Alex/He cluster",
    "camera/it cluster"
  ]
}
```

For a multi-chunk document, only mentions inside that chunk are included, but
the canonical mention may still come from another part of the document. That is
how cross-sentence and cross-chunk pronoun resolution is preserved.

### 6. Mention Prepass Builds Prompt Hints

`MentionPrepass.build()` receives:

```text
chunk text
chunk-level coreference result
global chunk offset
```

It scans local mentions:

```json
[
  {"id": "alex_m1", "text": "Alex", "kind": "name"},
  {"id": "camera_m2", "text": "camera", "kind": "definite"},
  {"id": "he_m3", "text": "He", "kind": "pronoun"},
  {"id": "it_m4", "text": "it", "kind": "pronoun"}
]
```

Then it merges LingMess metadata:

```json
{
  "resolved_pronouns": [
    {
      "text": "He",
      "resolved_to": "Alex",
      "resolution_source": "lingmess"
    },
    {
      "text": "it",
      "resolved_to": "camera",
      "resolution_source": "lingmess"
    }
  ],
  "ambiguous_pronouns": [],
  "unresolved_pronouns": []
}
```

The important output is the prompt hint sent to LangExtract:

```text
Mention prepass hints:
- Mentions below are deterministic anchors, not facts.
- If a pronoun has multiple candidates, do not replace it with one candidate.
- For unambiguous resolved pronouns, use the canonical entity only as the semantic argument and preserve the original source mention.
- he_m3: pronoun text='He' canonical='alex' resolved_to='Alex' source='lingmess'
- it_m4: pronoun text='it' canonical='camera' resolved_to='camera' source='lingmess'
- The mention 'He' at characters 165-167 belongs to the same entity cluster as 'Alex'. Use 'Alex' as the semantic subject/object when extracting propositions; preserve the original mention text.
- The mention 'it' at characters 176-178 belongs to the same entity cluster as 'camera'. Use 'camera' as the semantic subject/object when extracting propositions; preserve the original mention text.
```

Notice what this does and does not do:

```text
does: guide LangExtract to use Alex/camera as semantic arguments
does: preserve source text and offsets
does: mark ambiguity instead of guessing
does not: rewrite the paragraph
does not: add proof facts directly
does not: prove anything
```

### 7. Retrieve Previous Context

Before calling LangExtract, the service retrieves context from:

```text
Qdrant vector store
recent atoms from data/atomspace/kb.metta
known predicate heads remembered by the parser
```

For the first paragraph in an empty database, context is usually:

```json
[]
```

For a later paragraph, context may contain atoms like:

```text
(: camera_is_device (IsA camera device) (STV 1.0 1.0))
(: requires_inspection_rule (Implication ... ) (STV 1.0 1.0))
```

This context is used only to help LangExtract reuse vocabulary, for example:

```text
Existing predicate vocabulary from previous chunks and the knowledge base:
damaged->Damaged, requires-inspection->RequiresInspection
Reuse these predicate names when the question or sentence means the same relation.
```

### 8. LangExtract Statement Extraction

`LangExtractPLNParser.parse()` builds the final prompt:

```text
statement extraction instructions
+ existing predicate vocabulary hint
+ mention prepass/coreference hint
+ few-shot examples from data/langextract_examples.json
+ chunk text
```

It sends this to LangExtract:

```python
lx.extract(
    text_or_documents=chunk_text,
    prompt_description=prompt,
    examples=statement_examples,
    model_id="gemini-2.5-flash",
    api_key=LANGEXTRACT_API_KEY or GEMINI_API_KEY,
    extraction_passes=1,
    max_workers=1,
    show_progress=False
)
```

Representative LangExtract output:

```json
[
  {
    "extraction_class": "type_decl",
    "extraction_text": "Alex is a person",
    "attributes": {
      "entity": "Alex",
      "type": "person"
    }
  },
  {
    "extraction_class": "type_decl",
    "extraction_text": "The camera is a device",
    "attributes": {
      "entity": "camera",
      "type": "device"
    }
  },
  {
    "extraction_class": "rule",
    "extraction_text": "If a person drops a device, the device becomes damaged",
    "attributes": {
      "head_predicate": "damaged",
      "head_args": "$y",
      "body": "(and (isa $x person) (isa $y device) (drops $x $y))"
    }
  },
  {
    "extraction_class": "rule",
    "extraction_text": "If a device is damaged, it requires inspection",
    "attributes": {
      "head_predicate": "requires inspection",
      "head_args": "$y",
      "body": "(and (isa $y device) (damaged $y))"
    }
  },
  {
    "extraction_class": "fact",
    "extraction_text": "He dropped it",
    "attributes": {
      "predicate": "drops",
      "arguments": ["Alex", "camera"]
    }
  }
]
```

This is still not proof-ready. It is only structured extraction.

### 9. Translate LangExtract Output To PLN

`core/extraction/langextract_pln.py` translates extraction objects into
PLN/MeTTa-style statements.

Representative translation:

```text
(: alex_is_person_type (IsA alex person) (STV 1.0 1.0))
(: camera_is_device_type (IsA camera device) (STV 1.0 1.0))
(: y_damaged_rule (Implication (Premises (IsA $x person) (IsA $y device) (Drops $x $y)) (Conclusions (Damaged $y))) (STV 1.0 1.0))
(: y_requires_inspection_rule (Implication (Premises (IsA $y device) (Damaged $y)) (Conclusions (RequiresInspection $y))) (STV 1.0 1.0))
(: alex_camera_drops_fact (Drops alex camera) (STV 1.0 1.0))
```

The translator also rejects unsafe outputs. Examples:

```text
free variable inside a fact
unsupported extraction class
unparseable rule body
fuzzy alignment when LANGEXTRACT_SKIP_FUZZY=true
epistemic absence turned into direct negation
unencoded modal claims like "may cause" becoming certain rules
```

Example rejection:

```json
{
  "extraction_class": "negation",
  "extraction_text": "Alex is not known to be diagnosed",
  "reason": "epistemic or classification absence cannot become direct negation"
}
```

This is one of the main proof-safety layers.

### 10. PLN Postprocessing

`PLNPostprocessor.process()` receives:

```text
source text
translated statements
context atoms
plan_queries=false for ingest
```

It performs:

```text
dedupe statements
canonicalize symbols
normalize property predicates
repair selected arity issues
repair missing universal variables in rules
enforce predicate arities against context
register predicate cards
prune generic sortal premises when safe
infer simple person types for proper names
normalize typed constraints
suppress proof bridges unless enabled
filter malformed statements
ensure every statement has (: name payload (STV s c)) shape
```

Representative result:

```json
{
  "statements": [
    "(: alex_is_person_type (IsA alex person) (STV 1.0 1.0))",
    "(: camera_is_device_type (IsA camera device) (STV 1.0 1.0))",
    "(: y_damaged_rule (Implication (Premises (IsA $x person) (IsA $y device) (Drops $x $y)) (Conclusions (Damaged $y))) (STV 1.0 1.0))",
    "(: y_requires_inspection_rule (Implication (Premises (IsA $y device) (Damaged $y)) (Conclusions (RequiresInspection $y))) (STV 1.0 1.0))",
    "(: alex_camera_drops_fact (Drops alex camera) (STV 1.0 1.0))"
  ],
  "alignment_decisions": [
    {
      "action": "registry_predicate_observed",
      "predicate": "Drops",
      "arity": 2
    },
    {
      "action": "registry_predicate_observed",
      "predicate": "Damaged",
      "arity": 1
    },
    {
      "action": "semantic_bridges_suppressed",
      "reason": "proof_authority_disabled",
      "proof_safe": true
    }
  ]
}
```

Postprocessing is strict by design. If a statement is not safe or well-shaped,
it should not enter the atomspace.

### 11. Predicate Registry And Mapping

The predicate registry observes the current statements and builds predicate
cards.

Representative predicate cards:

```json
[
  {
    "predicate": "Drops",
    "arity": 2,
    "argument_types": ["person", "device"],
    "label": "drops",
    "definition": "drops is a predicate over person, device.",
    "source_atoms": ["(Drops alex camera)"],
    "roles": ["fact"]
  },
  {
    "predicate": "Damaged",
    "arity": 1,
    "argument_types": ["device"],
    "label": "damaged",
    "definition": "damaged is a predicate over device.",
    "source_atoms": ["(Damaged $y)"],
    "roles": ["conclusion", "premise"]
  },
  {
    "predicate": "RequiresInspection",
    "arity": 1,
    "argument_types": ["device"],
    "label": "requires inspection",
    "definition": "requires inspection is a predicate over device.",
    "source_atoms": ["(RequiresInspection $y)"],
    "roles": ["conclusion"]
  }
]
```

These cards are persisted in:

```text
data/predicate_registry.json
```

If Qdrant is enabled, predicate cards are also embedded and stored in a separate
predicate-card collection. This helps later chunks discover semantically related
predicate names.

#### What Predicate Mapping Does

Suppose a later paragraph says:

```text
Damaged devices need checking.
```

LangExtract may produce:

```text
(NeedsChecking $x)
```

But the earlier rule used:

```text
(RequiresInspection $x)
```

Predicate mapping asks:

```text
Are NeedsChecking/1 and RequiresInspection/1 logically compatible?
```

It may produce a mapping:

```json
{
  "source": "NeedsChecking",
  "target": "RequiresInspection",
  "arity": 1,
  "relation": "source_implies_target",
  "confidence": 0.91,
  "proof_safe": true,
  "reason": "same device-state meaning with compatible argument type"
}
```

But mapping does not automatically become proof authority. By default:

```text
PREDICATE_MAPPING_EMIT_BRIDGES=false
```

So the system records mapping decisions, but does not add bridge rules like:

```text
(: needs_checking_to_requires_inspection_bridge
   (Implication
     (Premises (NeedsChecking $x))
     (Conclusions (RequiresInspection $x)))
   (STV 0.91 0.80))
```

Bridge emission is disabled because semantic similarity is not always safe
enough for formal proof. This keeps the architecture honest: mapping helps
vocabulary alignment, but proof facts still need strict validation.

### 12. Add Statements To The Reasoner

After postprocessing, service calls:

```python
reasoner.add_statements(processed.statements, provenance=statement_to_source)
```

The reasoner writes each atom to:

```text
data/atomspace/kb.metta
```

Example stored lines:

```text
(: alex_is_person_type (IsA alex person) (STV 1.0 1.0))
(: camera_is_device_type (IsA camera device) (STV 1.0 1.0))
(: alex_camera_drops_fact (Drops alex camera) (STV 1.0 1.0))
(: y_damaged_rule (Implication (Premises (IsA $x person) (IsA $y device) (Drops $x $y)) (Conclusions (Damaged $y))) (STV 1.0 1.0))
(: y_requires_inspection_rule (Implication (Premises (IsA $y device) (Damaged $y)) (Conclusions (RequiresInspection $y))) (STV 1.0 1.0))
```

It also writes provenance to:

```text
data/atomspace/kb.metta.provenance.jsonl
```

Representative provenance:

```json
{
  "atom": "(: alex_camera_drops_fact (Drops alex camera) (STV 1.0 1.0))",
  "source": {
    "text": "He dropped it",
    "char_interval": {"start_pos": 165, "end_pos": 179},
    "class": "fact"
  }
}
```

This lets debug views show where a proof atom came from.

### 13. Store Chunk In Qdrant

If vector store is enabled, service calls:

```python
vector_store.store(
    chunk_text,
    added_atoms,
    vector,
    metadata=parse_result.metadata,
    query_targets=extract_query_targets(added_atoms)
)
```

Before storing, Qdrant needs a vector. `storage/vector_store.py` sends the chunk
to Ollama:

```http
POST http://localhost:11434/api/embeddings
```

```json
{
  "model": "nomic-embed-text",
  "prompt": "Alex is a person. The camera is a device..."
}
```

Ollama returns:

```json
{
  "embedding": [0.014, -0.028, 0.092, "..."]
}
```

Then Qdrant stores:

```json
{
  "id": "uuid",
  "vector": [0.014, -0.028, 0.092],
  "payload": {
    "nl": "Alex is a person. The camera is a device...",
    "pln": [
      "(: alex_is_person_type (IsA alex person) (STV 1.0 1.0))",
      "(: alex_camera_drops_fact (Drops alex camera) (STV 1.0 1.0))"
    ],
    "query_targets": [
      "(IsA alex person)",
      "(IsA camera device)",
      "(Drops alex camera)",
      "(Damaged $y)",
      "(RequiresInspection $y)"
    ],
    "metadata": {
      "statement_to_source": {},
      "mention_prepass": {},
      "schema_alignment": [],
      "predicate_registry": []
    }
  }
}
```

Qdrant is retrieval memory. It is not the proof system.

## Query Step By Step

Now ask:

```http
POST /debug/query
Content-Type: application/json
```

```json
{
  "question": "Does the camera require inspection?"
}
```

### 1. Parse Question Intent

`core/query/intent.py` receives:

```text
Does the camera require inspection?
```

Representative intent:

```json
{
  "mode": "boolean",
  "entities": [],
  "terms": ["camera", "require", "inspection"],
  "direction": "",
  "causal": false
}
```

For a named entity question:

```text
Does Alex require inspection?
```

the entity detector may identify:

```json
{"entities": ["alex"]}
```

Intent is used to reject query candidates that answer the wrong question.

### 2. Retrieve Qdrant Context

The service embeds the question through Ollama:

```json
{
  "model": "nomic-embed-text",
  "prompt": "Does the camera require inspection?"
}
```

Then it searches Qdrant:

```json
{
  "vector": [0.021, -0.019, 0.104],
  "limit": 10,
  "with_payload": true
}
```

Representative Qdrant match:

```json
{
  "score": 0.81,
  "nl": "Alex is a person. The camera is a device...",
  "pln": [
    "(: camera_is_device_type (IsA camera device) (STV 1.0 1.0))",
    "(: alex_camera_drops_fact (Drops alex camera) (STV 1.0 1.0))",
    "(: y_requires_inspection_rule (Implication ... (RequiresInspection $y)) (STV 1.0 1.0))"
  ],
  "query_targets": [
    "(RequiresInspection $y)",
    "(Damaged $y)",
    "(Drops alex camera)"
  ]
}
```

The service also reads recent atoms from `data/atomspace/kb.metta` so the parser
has predicate vocabulary even if vector retrieval is imperfect.

### 3. Build Qdrant-Aligned Query Candidates

`core/query/alignment.py` uses retrieved `query_targets` to build possible
queries.

From:

```text
(RequiresInspection $y)
```

and question terms:

```text
camera, require, inspection
```

it can propose:

```text
(: $prf (RequiresInspection camera) $tv)
```

But this is only a candidate. It must still pass intent and arity checks.

### 4. LangExtract Parses The Question

The parser also asks LangExtract to translate the question:

```python
parser.parse_query(question, context)
```

It sends:

```text
query extraction instructions
+ known predicate vocabulary
+ local mention hints if any
+ retrieved context atoms
+ question text
```

Representative LangExtract query extraction:

```json
{
  "extraction_class": "query",
  "extraction_text": "Does the camera require inspection?",
  "attributes": {
    "predicate": "requires inspection",
    "arguments": ["camera"]
  }
}
```

Translated query:

```text
(: $prf (RequiresInspection camera) $tv)
```

### 5. Deterministic Query Candidates

The service also builds deterministic candidates from known atom signatures in
the reasoner.

Known signatures might include:

```json
[
  {"head": "RequiresInspection", "args": ["$y"], "arity": 1},
  {"head": "Damaged", "args": ["$y"], "arity": 1},
  {"head": "Drops", "args": ["alex", "camera"], "arity": 2}
]
```

For the question, it can produce:

```text
(: $prf (RequiresInspection camera) $tv)
```

This helps when LangExtract produces a slightly different wording but the
knowledge base already has the correct predicate.

### 6. Candidate Filtering

The service merges candidates from:

```text
deterministic signatures
LangExtract query parser
Qdrant alignment context
```

Then it filters them:

```text
must match question intent
must mention required entity when present
must cover required content terms
must have an allowed predicate arity
must not target an opposite/status predicate unless asked
```

Accepted:

```text
(: $prf (RequiresInspection camera) $tv)
```

Rejected examples:

```text
(: $prf (Damaged camera) $tv)
```

Reason: related, but the question asked about inspection.

```text
(: $prf (Drops alex camera) $tv)
```

Reason: proof-supporting fact, not the question target.

This prevents the old problem where the system answered a different but nearby
question just because Qdrant retrieved it.

### 7. Proof Search

The service calls:

```python
reasoner.query_polarity("(: $prf (RequiresInspection camera) $tv)")
```

The reasoner checks both:

```text
positive query:
(: $prf (RequiresInspection camera) $tv)

negative query:
(: $prf (Not (RequiresInspection camera)) $tv)
```

It first checks exact facts in the atomspace file. If exact fact is not present,
it runs strict backward chaining over rules.

Proof chain:

```text
Goal: (RequiresInspection camera)

Rule:
(Implication
  (Premises (IsA $y device) (Damaged $y))
  (Conclusions (RequiresInspection $y)))

Bind $y = camera.

Need:
(IsA camera device)
(Damaged camera)

Fact:
(IsA camera device)

Need:
(Damaged camera)

Rule:
(Implication
  (Premises (IsA $x person) (IsA $y device) (Drops $x $y))
  (Conclusions (Damaged $y)))

Bind $x = alex, $y = camera.

Need:
(IsA alex person)
(IsA camera device)
(Drops alex camera)

Facts:
(IsA alex person)
(IsA camera device)
(Drops alex camera)

Therefore:
(Damaged camera)

Therefore:
(RequiresInspection camera)
```

Representative `ProofOutcome`:

```json
{
  "status": "positive",
  "positive_query": "(: $prf (RequiresInspection camera) $tv)",
  "negative_query": "(: $prf (Not (RequiresInspection camera)) $tv)",
  "positive_proof": [
    "(: y_requires_inspection_rule ...)",
    "(: y_damaged_rule ...)",
    "(: camera_is_device_type (IsA camera device) (STV 1.0 1.0))",
    "(: alex_is_person_type (IsA alex person) (STV 1.0 1.0))",
    "(: alex_camera_drops_fact (Drops alex camera) (STV 1.0 1.0))"
  ],
  "negative_proof": [],
  "proof_validated": true,
  "support_kind": "entailed"
}
```

Possible statuses:

```text
positive: proposition proved
negative: explicit negation proved
both: contradiction exists
unknown: neither positive nor negative proved
```

The system does not treat absence of proof as proof of false.

### 8. Source Lookup

If configured, service can reverse-map proof atoms back to source text using
provenance.

Representative source:

```json
[
  "He dropped it",
  "The camera is a device",
  "If a person drops a device, the device becomes damaged",
  "If a device is damaged, it requires inspection"
]
```

This part is for explanation and debugging. It is not proof authority.

### 9. Answer Generation

`AnswerGenerator.generate_from_polarity()` receives:

```text
question
executed_query
proof_status
positive_proof
negative_proof
```

For this example:

```json
{
  "executed_query": "(: $prf (RequiresInspection camera) $tv)",
  "proof_status": "positive"
}
```

Answer:

```text
Yes. The proof establishes (RequiresInspection camera).
```

If the negative query was proved:

```text
No. The proof establishes explicit negation of (RequiresInspection camera).
```

If no proof was found:

```text
I don't know - neither the proposition nor its explicit negation was proved.
```

This is intentionally conservative.

## Full Debug Query Response Shape

`/debug/query` returns a response like:

```json
{
  "question": "Does the camera require inspection?",
  "context": ["...retrieved PLN atoms..."],
  "qdrant_matches": ["...retrieved chunks..."],
  "qdrant_aligned_queries": [
    "(: $prf (RequiresInspection camera) $tv)"
  ],
  "execution_candidates": [
    "(: $prf (RequiresInspection camera) $tv)"
  ],
  "langextract_postprocessed": {
    "queries": [
      "(: $prf (RequiresInspection camera) $tv)"
    ],
    "rejected": [],
    "mention_prepass": {},
    "mention_prompt_hint": ""
  },
  "pln_canonicalized_queries": [
    "(: $prf (RequiresInspection camera) $tv)"
  ],
  "executed_query": "(: $prf (RequiresInspection camera) $tv)",
  "query_source": "deterministic",
  "fallback_used": false,
  "query_status": "well_aligned",
  "proof_status": "positive",
  "support_kind": "entailed",
  "answer": "Yes. The proof establishes (RequiresInspection camera)."
}
```

## What Each Model Or Service Does

| Component | Used For | Not Used For |
| --- | --- | --- |
| LingMess / FastCoref | Document-level coreference: `he`, `she`, `it`, `they` resolution. | It does not create proof atoms. |
| Mention prepass | Converts coreference into safe LangExtract hints and protects ambiguity. | It is not the main resolver when LingMess is enabled. |
| LangExtract | Converts natural language into structured facts, rules, negations, types, and queries. | It does not prove answers. |
| Gemini/OpenAI | Backing model for LangExtract and optionally predicate relation classification or fallback answer phrasing. | It is not proof authority. |
| Ollama | Embedding model host, usually `nomic-embed-text`. | It is not LangExtract and not NLI validation. |
| Qdrant | Vector retrieval for chunks and predicate cards. | It does not prove answers. |
| Predicate mapping | Detects possible predicate equivalence or implication. | It does not add proof bridges by default. |
| PeTTaChainer / Reasoner | Stores atoms and runs proof search. | It does not do natural-language extraction. |

## Why LingMess Comes Before LangExtract

This is the current intended design:

```text
raw document
-> LingMess coreference
-> mention prepass hints
-> LangExtract extraction
```

Reason:

```text
"He dropped it" is hard for LangExtract if it does not know who "He" and "it" refer to.
LingMess can resolve those mentions at document level before extraction.
LangExtract then receives safe hints and can emit (Drops alex camera).
```

Bad design would be:

```text
LangExtract first
-> maybe extracts (Drops he it)
-> try to fix logic later
```

That is harder and less safe because the wrong entities may already be baked
into extracted atoms.

The code still keeps deterministic mention prepass because LingMess can fail,
be unavailable, or return ambiguous clusters. The fallback should preserve
ambiguity rather than hallucinate a referent.

## Why Predicate Mapping Is Conservative

Predicate mapping exists because real text uses different wording:

```text
requires inspection
needs checking
must be reviewed
requires maintenance review
```

But in a proof system, wording similarity is dangerous. These are not always
logically identical.

So mapping must pass gates:

```text
same arity
compatible argument types
safe direction
high confidence
identity argument order
no conflict with explicit negation
```

Even then, bridge emission is disabled by default:

```text
PREDICATE_MAPPING_EMIT_BRIDGES=false
```

This means the project favors correctness over making the architecture look
smart. That is the right tradeoff for proof safety.

## Common Failure Points

### LingMess Not Actually Running

Symptoms:

```text
coreference backend is "none"
resolved_pronouns is empty
He/it remains unresolved
```

Check:

```text
COREFERENCE_ENABLED=true
requirements-coref.txt installed
Docker built with PLNRAG_INSTALL_COREF=true
```

Docker example:

```powershell
$env:PLNRAG_INSTALL_COREF="true"
$env:COREFERENCE_ENABLED="true"
docker compose --profile default up --build pln-rag
```

### LangExtract Extracts A Different Predicate

Example:

```text
Query asks: requires inspection
Stored atom: NeedsChecking camera
```

Possible fixes:

```text
improve LangExtract examples
improve predicate vocabulary hints
allow predicate mapping to record relation
only enable bridge emission if proof safety is acceptable
```

### Missing Rule Premise

Example:

```text
Rule requires (IsA camera device)
but extraction did not produce camera device type.
```

Then proof should fail:

```text
proof_status = unknown
```

This is correct. The system should not invent missing premises.

### Qdrant Retrieves Related But Wrong Context

Qdrant might retrieve:

```text
(Damaged camera)
```

for a question about:

```text
RequiresInspection camera
```

The intent gate should reject the wrong query target. Retrieval is support, not
the final answer authority.

## Best Debug Endpoint

Use:

```http
POST /debug/ingest
POST /debug/query
```

For ingestion, inspect:

```text
coreference
mention_prompt_hint
langextract_postprocessed.statements
langextract_postprocessed.rejected
pln_canonicalized
schema_alignment
predicate_registry
atomspace_added
evidence_records
```

For query, inspect:

```text
qdrant_matches
qdrant_aligned_queries
langextract_postprocessed.queries
pln_canonicalized_queries
execution_candidates
executed_query
query_source
proof_status
positive_proof
negative_proof
answer
```

The shortest way to diagnose accuracy is:

```text
1. Did LingMess resolve pronouns correctly?
2. Did LangExtract produce the right fact/rule/query objects?
3. Did PLN translation keep the intended predicate and arguments?
4. Did postprocessing reject or change anything important?
5. Did the reasoner have all rule premises?
6. Did query planning execute the intended target?
```

## Evidence-Linked Retrieval

The current storage model does not treat a whole extraction chunk as one proof
target. SQLite stores four explicit identities:

```text
document_id -> evidence_id -> claim_id -> claim_evidence_id
```

`document_id` identifies the immutable source document. `evidence_id` identifies
a chunk with exact document offsets. `claim_id` identifies normalized PLN.
`claim_evidence_id` records why that claim is associated with that evidence.

For a direct LangExtract claim, the source text must be an exact chunk substring.
Its grounded arguments must occur in that evidence or in a validated
coreference annotation. Explicitly negated claims also require explicit source
negation. Failed checks produce `validation_state=quarantined`; the claim is not
added to Atomspace or Qdrant.

Normalizer-generated claims use `validation_state=derived`. They may be loaded
into Atomspace when their deterministic transformation is traceable, but they
are not independent Qdrant query targets and do not pretend to have an exact
LangExtract span.

Each accepted query target becomes a separate Qdrant point. Its dense and
lexical vectors are generated from natural language (`retrieval_text`), while
the linked PLN atom remains structured payload. Qdrant proposes the linked
target; entity, predicate, arity, polarity, and question-intent gates decide
whether it can be executed. PeTTaChainer still decides whether proof exists.

SQLite also keeps an indexing outbox. If Ollama or Qdrant is unavailable during
ingestion, the accepted claim remains durable and indexing can be retried. Use:

```http
POST /rebuild
```

to rebuild Atomspace and the v2 Qdrant collection from SQLite. Legacy
chunk-level records are not trusted by the evidence-linked query planner.

## Final Mental Model

Keep this model in your head:

```text
LingMess resolves references.
Mention prepass safely explains those references to LangExtract.
LangExtract turns language into candidate logic.
The PLN translator rejects unsafe extraction shapes.
The postprocessor normalizes and constrains logic.
The evidence ledger validates and records exact claim lineage.
Predicate mapping records vocabulary relationships conservatively.
Qdrant retrieves source-linked candidate targets, never truth.
The reasoner proves or does not prove the target.
The answer generator reports only what the proof established.
```

That is the architecture. It is intentionally conservative because this project
cares more about correctness and proof safety than impressive-looking automatic
inference.
