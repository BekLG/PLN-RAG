# CanonicalPLN Parser Spec

This document defines the intended normalization and representation policy for
`CanonicalPLNParser`.

The goal is not to preserve surface wording. The goal is to produce stable,
proof-friendly symbols for a case-sensitive symbolic reasoner.

## Goals

1. Use one canonical symbol per semantic concept.
2. Avoid proof failures caused by casing, plurality, or paraphrase drift.
3. Keep predicate signatures stable across ingest and query.
4. Query only what the KB can actually derive.
5. Leave natural-language recovery to the answer generator.

## Normalization Rules

### 1. Canonical symbol form

- All non-variable symbols should be lowercase.
- Multiword symbols should use `snake_case`.
- Replace spaces, hyphens, and punctuation with `_`.
- Collapse repeated `_`.
- Strip leading and trailing `_`.

Examples:

- `Abebe` -> `abebe`
- `Dr. Ayele` -> `dr_ayele`
- `Object-77` -> `object_77`
- `quantum physics` -> `quantum_physics`
- `rack-01` -> `rack_01`

### 2. Lemmatization

Lemmatize common nouns and verbs before symbolization.

Examples:

- `humans` -> `human`
- `dogs` -> `dog`
- `men` -> `man`
- `children` -> `child`
- `eats` -> `eat`
- `drinks` -> `drink`

This applies to common concepts and relation arguments, not to variable syntax.

### 3. Variables

- Preserve variables exactly as `$name`.
- Do not lowercase away the `$` prefix.
- Use descriptive variable names when possible.

Good:

- `$person`
- `$topic`
- `$food`

### 4. Rule and proof identifiers

- Rule names and fact ids should also be lowercase `snake_case`.

Examples:

- `eat_fish_smart_rule`
- `abebe_eats_fish`

## Representation Policy

### 1. Keep one predicate signature per meaning

Do not alternate between structurally different encodings unless you also add a
bridge rule.

Bad:

```metta
(Eats abebe fish)
(EatsFish abebe)
```

Good:

```metta
(Eats abebe fish)
```

Bridge only when needed:

```metta
(: eats_something_rule
  (Implication
    (Premises
      (Eats $p $food))
    (Conclusions
      (EatsSomething $p)))
  (STV 1.0 1.0))
```

### 2. Prefer explicit argument-preserving relations

If an argument may be queried later, keep it in the predicate.

Good:

```metta
(AtSpot kebede rack_01)
(Inside rack_01 vault_07)
(InArea kebede vault_07)
```

Avoid collapsing needed arguments into unary predicates.

### 3. Canonical membership representation

Use one class-membership form consistently.

Preferred form for this project:

```metta
(IsA abebe human)
```

Do not mix these for the same meaning without bridge rules:

```metta
(Human abebe)
(IsA abebe human)
(Inheritance abebe human)
```

### 4. Unary properties vs binary relations

Use unary predicates for properties:

```metta
(Smart kebede)
(Mortal socrates)
(Fragile object_77)
```

Use binary or higher-arity predicates when arguments matter:

```metta
(Eats kebede fish)
(StudentOf kebede dr_ayele)
(ExpertIn dr_ayele quantum_physics)
```

### 5. Rules should conclude the queried form

Rules should produce the same predicate form that queries will ask for.

Good:

```metta
(: eat_fish_smart_rule
  (Implication
    (Premises
      (Eats $person fish))
    (Conclusions
      (Smart $person)))
  (STV 1.0 1.0))
```

Then query:

```metta
(: $prf (Smart kebede) $tv)
```

### 6. Query what the KB can prove

If the rules derive a bound, query the bound. If the rules derive a helper
predicate, query the helper predicate.

Good:

```metta
(: $prf (GreaterThan (HeightDist kebede) 160.0) $tv)
```

Bad if no rule derives `HeightDist` directly:

```metta
(: $prf (HeightDist kebede) $hDist)
```

### 7. Open-question policy

For open questions, prefer variable-bearing queries.

Good:

```metta
(: $prf (Smart $who) $tv)
(: $prf (Eats kebede $food) $tv)
```

Bad:

```metta
(: $prf (Smart kebede) $tv)
```

for the question "Who is smart?"

### 8. Existential helper predicates

For questions like "Did X do anything?", use a helper predicate if needed.

```metta
(: eat_something_rule
  (Implication
    (Premises
      (Eats $p $food))
    (Conclusions
      (EatsSomething $p)))
  (STV 1.0 1.0))
```

Then query:

```metta
(: $prf (EatsSomething kebede) $tv)
```

### 9. Exceptions and inequality

Prefer explicit distinction facts over unsupported implicit negation.

Good:

```metta
(Different rio henry)
```

Better rule premise:

```metta
(Different $p henry)
```

Avoid depending on:

```metta
(Not (Equal $p henry))
```

unless the system already supports it robustly.

### 10. Intermediate predicates for chaining

Use normalized intermediate predicates so one conclusion exactly matches the next
premise.

Good:

```metta
(: freeze_rule
  (Implication
    (Premises
      (IsFreezing $s))
    (Conclusions
      (DecreasingHeat $s)))
  (STV 1.0 1.0))

(: heat_temp_rule
  (Implication
    (Premises
      (DecreasingHeat $s))
    (Conclusions
      (DecreasingTemp $s)))
  (STV 1.0 1.0))

(: temp_motion_rule
  (Implication
    (Premises
      (DecreasingTemp $s))
    (Conclusions
      (MoleculesMoveSlower $s)))
  (STV 1.0 1.0))
```

## Preferred Templates

### Fact

```metta
(: fact_id (Predicate arg1 arg2 ...) (STV 1.0 1.0))
```

### Rule

```metta
(: rule_name
  (Implication
    (Premises
      premise1
      premise2)
    (Conclusions
      conclusion1))
  (STV 1.0 1.0))
```

### Query

```metta
(: $prf (Predicate arg1 arg2 ...) $tv)
```

## Good Output Examples

### Example 1: Class membership

Input:

- `Humans are mortal.`
- `Socrates is a human.`

Preferred output:

```metta
(: human_mortal_rule
  (Implication
    (Premises
      (IsA $person human))
    (Conclusions
      (Mortal $person)))
  (STV 1.0 1.0))

(: socrates_human (IsA socrates human) (STV 1.0 1.0))

(: $prf (Mortal socrates) $tv)
```

### Example 2: Binary relation

Input:

- `Kebede eats fish.`
- `People who eat fish are smart.`

Preferred output:

```metta
(: kebede_eats_fish (Eats kebede fish) (STV 1.0 1.0))

(: eat_fish_smart_rule
  (Implication
    (Premises
      (Eats $person fish))
    (Conclusions
      (Smart $person)))
  (STV 1.0 1.0))

(: $prf (Smart kebede) $tv)
```

### Example 3: Open question

Input:

- `Kebede eats fish.`
- question: `What does Kebede eat?`

Preferred query:

```metta
(: $prf (Eats kebede $food) $tv)
```

## Bad Output Patterns

Avoid:

- mixed casing for the same symbol
- plural and singular concept variants as separate atoms
- changing predicate arity across facts and rules
- querying a structure the KB never derives
- collapsing away arguments needed later for querying
- inventing a new predicate when an existing context predicate already matches

Examples of bad drift:

```metta
(IsA Kebede Human)
(IsA kebede human)
(Human kebede)
(Inheritance kebede human)
```

for the same intended meaning.

## Summary

The parser should optimize for symbolic consistency, not surface faithfulness.

Canonical lowercase symbols, lemmatized common concepts, stable predicate
signatures, and proof-aligned queries are the default policy.
