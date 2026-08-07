# Coreference Architecture

PLN-RAG has two pronoun-handling layers.

## Default layer: mention prepass

The default path uses `MentionPrepass` from `core/discourse/mention_prepass.py`.
It scans each chunk for names, noun mentions, and pronouns before LangExtract runs.
The result is passed into the LangExtract prompt as a hint.

This layer does not assert facts into PLN. It only helps the extractor avoid bad
pronoun guesses. For example:

```text
Alex put the camera near the phone. It was broken.
```

The prepass may see both `camera` and `phone` as possible antecedents for `It`.
Because that is ambiguous, the prompt tells LangExtract not to guess. The proof
system should not silently turn `It was broken` into `Broken phone` or
`Broken camera`.

## Optional layer: LingMess coreference

The optional neural layer is `LingMessCoreferenceResolver` from
`core/discourse/lingmess_coreference.py`. It wraps `fastcoref.LingMessCoref`
with the Hugging Face model configured by:

```env
COREFERENCE_MODEL=biu-nlp/lingmess-coref
```

This layer is disabled by default:

```env
COREFERENCE_ENABLED=false
```

To enable it in Docker, the image must include the optional dependency:

```powershell
$env:PLNRAG_INSTALL_COREF="true"
$env:COREFERENCE_ENABLED="true"
docker compose --profile default up --build
```

If LingMess is unavailable or fails and `COREFERENCE_FAIL_OPEN=true`, the service
continues with the deterministic mention prepass. This keeps ingestion usable and
prevents a neural model failure from blocking proof extraction.

## Safety rule

Coreference is hint-only. It can contextualize retrieval text and guide
LangExtract, but it is not proof authority by itself. A PLN claim still needs
accepted source evidence and must pass evidence validation before it can become a
trusted Qdrant query target.
