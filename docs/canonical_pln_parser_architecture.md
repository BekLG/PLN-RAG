# CanonicalPLN Parser Architecture

This document explains the architecture of `CanonicalPLNParser` as a parsing
component. The focus here is the text-to-PLN side: how input is prepared, how
generation is constrained, how symbolic forms are normalized, and how unsafe
outputs are filtered before they reach the reasoner.

## Purpose

`CanonicalPLNParser` is a controlled parsing layer built on top of the existing
`NL2PLN` module.

Its purpose is to make the generated symbolic output more stable and more usable
for downstream symbolic reasoning. It does this by combining:

1. a separate tuned generation artifact
2. parser-side normalization
3. parser-side structural filtering

It is best understood as a reasoning-oriented parser adapter, not as a fully
independent parser engine.

## Position In The Pipeline

At ingest time, the parser sits in the middle of the symbolic pipeline:

```text
Text
  -> Chunker
  -> symbolic context retrieval
  -> CanonicalPLNParser.parse(...)
  -> normalized statements
  -> Reasoner.add_statements(...)
```

At query time, the same parser is reused in question mode:

```text
Question
  -> symbolic context retrieval
  -> CanonicalPLNParser.parse_query(...)
  -> normalized support statements + query candidates
  -> service execution
```

This means the parser is responsible for two related but distinct tasks:

- converting source text into facts and rules
- converting questions into logically meaningful query forms

## Core Design

The parser is split into two conceptual layers.

### 1. Generation Layer

This is still the underlying `NL2PLN` module.

It performs the actual language-model-driven translation from natural language to
candidate PLN statements and queries.

### 2. Control Layer

This is `CanonicalPLNParser` itself.

It reshapes the generation process in three ways:

- before generation: input and context preparation
- after generation: canonicalization and safety filtering
- during query parsing: ordering and shaping candidate queries

This separation is important because it keeps the baseline generator intact while
allowing the project to improve symbolic stability in a controlled way.

## Artifact Split

The project intentionally uses two separate compiled artifacts.

- baseline parser artifact: `data/simba_all.json`
- canonical parser artifact: `data/simba_canonical_pln.json`

This keeps the baseline `NL2PLNParser` independent from the canonical parser.

That separation matters because it lets the project compare:

- baseline generation behavior
- parser-controlled generation behavior

without mutating the baseline parser configuration.

## Parser Identity

Runtime selector:

```bash
PARSER=canonical_pln
```

Artifact path:

```bash
CANONICAL_PLN_NL2PLN_MODULE_PATH=data/simba_canonical_pln.json
```

## Internal Parsing Stages

`CanonicalPLNParser` can be understood as a sequence of parsing stages.

## Stage 1: Input Preparation

Main methods:

- `_normalize_text(...)`
- `_extract_concepts(...)`
- `_extract_context_predicates(...)`
- `_build_parser_inputs(...)`

This stage does not emit PLN directly. It builds a better input package for the
underlying `NL2PLN` call.

Responsibilities:

- normalize the input into a simpler analysis view
- extract common concepts from the text
- extract predicate heads already present in symbolic context
- build parser hints that encourage reuse of established symbolic vocabulary

The key idea is that the parser should not translate each sentence in isolation.
It should see a symbolic neighborhood and try to stay consistent with it.

## Stage 2: Guided Generation Through NL2PLN

Core call shape:

```python
self._nl2pln(
    sentences=[prepared_text],
    context=prepared_context,
    pln_spec=self._pln_spec,
)
```

This remains the actual text-to-PLN generator.

`CanonicalPLNParser` does not replace this generator. Instead, it narrows and
guides how the generator is used by:

- supplying stronger symbolic context
- using a separate compiled artifact
- enforcing downstream normalization rules

## Stage 3: Canonical Symbol Normalization

Main methods:

- `_canonicalize_outputs(...)`
- `_canonicalize_atom(...)`
- `_normalize_token(...)`
- `_canonical_head(...)`
- `_canonical_symbol(...)`
- `_normalize_isa_classes(...)`

This stage is the center of the parser’s symbolic identity.

Its purpose is to reduce drift between logically equivalent forms so the reasoner
sees one stable symbolic space instead of many near-duplicates.

### Structural Preservation

The parser preserves PLN structural constructors exactly. Examples include:

- `Implication`
- `Premises`
- `Conclusions`
- `STV`
- `And`
- `Or`
- `Not`
- `IsA`

This is intentional. Structural constructors are not treated like ordinary
domain symbols.

### Constant Normalization

Constants are normalized toward lowercase snake_case forms.

Examples of intended direction:

```text
Socrates -> socrates
Dr. Ayele -> dr_ayele
quantum physics -> quantum_physics
```

The reason for this is practical rather than cosmetic: the symbolic reasoner is
case-sensitive and not semantics-aware, so symbol consistency matters directly.

### Lemmatization Direction

Common nouns and verbs are pushed toward a shared canonical form.

Examples:

```text
humans -> human
parents -> parent
children -> child
eats -> eat
drinks -> drink
```

This reduces symbolic fragmentation across ingest and query parsing.

### Canonical Class Membership

The parser uses `IsA` as the canonical membership predicate.

That means it prefers a representation like:

```metta
(IsA socrates human)
```

over allowing multiple competing membership encodings.

### Predicate Head Canonicalization

The parser keeps predicate-head canonicalization intentionally minimal.

It normalizes only structural membership variants such as:

- `isa`, `is_a`, `kind_of`, `type_of` -> `IsA`

Other predicate heads are left to the underlying parser output and context
reuse logic rather than being forced through a hand-curated semantic alias
table. This keeps the wrapper more domain-independent and avoids coupling it to
the test vocabulary.

## Stage 4: Proper Name Protection

Main methods:

- `_extract_protected_constants(...)`
- `_extract_proper_name_map(...)`

One repeated parser failure mode was over-normalization of names, especially
accidental singularization of proper names.

Example unwanted drift:

```text
Socrates -> socrate
```

To reduce this, the parser derives protected-name information from the source
text and uses it during token normalization. This allows it to keep the logical
content anchored to source names more reliably while still normalizing the
general symbol space.

## Stage 5: Statement Safety Filtering

Main methods:

- `_filter_statements(...)`
- `_has_valid_implication_shape(...)`

This stage protects the reasoner from parser output that is structurally unsafe.

Two important categories are filtered.

### A. Malformed Rules

Rules are expected to use explicit PLN structure:

```metta
(Implication
  (Premises ...)
  (Conclusions ...))
```

If the parser emits a malformed implication shape, it is filtered instead of
being passed directly into the reasoner.

### B. Free-Variable Facts

Standalone facts containing free variables are filtered.

For example, a parser output like:

```metta
(: meron_heals (Heals meron $p) (STV 1.0 1.0))
```

is not treated as a safe grounded fact. The parser drops it rather than letting
it create ambiguous or unusable symbolic state.

This filtering is one of the most important architectural differences from a
thin parser wrapper.

## Stage 6: Query Parsing

`CanonicalPLNParser` also handles question parsing, but the key architectural
point is that it reuses the same symbolic discipline established for statement
generation.

That means query parsing inherits:

- the same canonical symbol policy
- the same `IsA` membership policy
- the same proper-name handling
- the same structural caution

The query side is therefore an extension of the parsing architecture, not a
completely separate subsystem.

## Parsing Philosophy

The parser is designed around a simple principle:

**the symbolic form should optimize for consistency and proofability, not for
surface resemblance to the original sentence.**

That leads to several practical decisions:

- use one canonical symbol where possible
- preserve structural PLN forms exactly
- keep class membership stable through `IsA`
- filter structurally unsafe outputs before they enter the KB

## Why CanonicalPLNParser Still Wraps NL2PLN

The project intentionally did not replace `NL2PLN` entirely.

Reasons:

- it preserves a clean baseline for comparison
- it keeps the architecture incremental rather than disruptive
- it allows prompt/program tuning and parser-control logic to evolve separately
- it avoids rebuilding the full text-to-PLN stack from scratch

So the best architectural description is:

```text
NL2PLN generation
  +
Canonical symbolic normalization and filtering
```

## Improvements Over The Baseline Parser

Compared to the baseline `NL2PLNParser`, `CanonicalPLNParser` adds several
parser-side improvements:

- more stable symbolic naming
- stronger `IsA` normalization
- lower risk of malformed implications reaching the reasoner
- lower risk of free-variable facts entering the KB
- better protection against proper-name drift
- cleaner separation between baseline artifact behavior and canonical parser behavior

These are parser architecture improvements, not only prompt tweaks.

## Recommended Mental Model

Think of `CanonicalPLNParser` as a symbolic normalization shell around `NL2PLN`.

Its parser-side role is:

1. prepare text and context for more controlled generation
2. canonicalize the generated symbolic forms
3. reject unsafe statement outputs before they enter the knowledge base

That is the core architecture.

## Related Files

Main parser files:

- `parsers/canonical_pln_parser.py`
- `parsers/canonical_pln_prev_parser.py`
- `parsers/canonical_pln_parser_spec.md`

Artifact/config files:

- `data/simba_canonical_pln.json`
- `config.py`
- `.env.example`

Integration files:

- `parsers/__init__.py`
- `core/service.py`
- `api/models.py`
