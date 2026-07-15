# Tests

Automatic tests live directly in this folder and should run without live LLM,
Qdrant, Ollama, or network services:

```bash
python -m unittest discover -s tests
```

The suite includes cross-domain intent contracts, proof-safety checks, polarity
and contradiction handling, guarded alternatives, predicate arity validation,
and exact proof provenance. Live model quality remains a separate Docker-level
evaluation because LangExtract output can vary by provider/model version.

Manual experiments live in `tests/manual/`. They are scripts for local debugging
and may require API keys, Ollama, Qdrant, or the full service stack.
