import re
from typing import List

from config import get_settings


SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on logical proof traces.

Answer only from the executed PLN target and proof trace.
If the proof establishes the target, answer yes.
If the proof establishes explicit negation of the target, answer no.
If neither is present, say you do not know.
Do not add domain knowledge that is not in the proof."""


class AnswerGenerator:
    """
    Translates a PLN proof trace into a natural language response.

    The primary path is deterministic: compare the executed query target against
    proof atoms. LLM calls remain only as a fallback when no executed target was
    supplied.
    """

    def __init__(self):
        cfg = get_settings()
        self._use_gemini = bool(cfg.gemini_api_key)
        self._gemini_model = cfg.gemini_model
        self._gemini_api_key = cfg.gemini_api_key
        self._openai_api_key = cfg.openai_api_key
        self._openai_model = cfg.openai_model

    def generate(
        self,
        question: str,
        proof_traces: List[str],
        executed_query: str = "",
    ) -> str:
        if not proof_traces:
            return "I don't know - no proof was found for this question."

        target = self._extract_query_target(executed_query)
        if target:
            return self._answer_from_target(target, proof_traces)

        proof_str = "\n".join(proof_traces)
        user_prompt = f"""Question: {question}

Proof trace:
{proof_str}

Answer only from the proof trace. Do not add unstated domain knowledge."""

        try:
            if self._use_gemini:
                return self._call_gemini(user_prompt)
            return self._call_openai(user_prompt)
        except Exception as e:
            print(f"[AnswerGenerator] Failed: {e}")
            return self._fallback_answer(question, proof_traces, executed_query)

    def generate_from_polarity(
        self,
        question: str,
        executed_query: str,
        status: str,
        positive_proof: List[str],
        negative_proof: List[str],
        support_kind: str = "unknown",
    ) -> str:
        target = self._extract_query_target(executed_query) or "the requested proposition"
        if status == "both":
            return (
                f"The knowledge base is contradictory: it proves both {target} "
                "and its explicit negation."
            )
        if status == "positive":
            if support_kind == "probabilistic":
                # The proof traversed a rule with strength below 1.0, which is how
                # LLM-derived predicate bridges enter a proof. Reporting that as a
                # flat "yes" would present a similarity judgment as entailment.
                return (
                    f"Probably yes. The proof supports {target}, but it relies on at "
                    "least one probabilistic step rather than strict entailment."
                )
            return f"Yes. The proof establishes {target}."
        if status == "negative":
            return f"No. The proof establishes explicit negation of {target}."
        return "I don't know - neither the proposition nor its explicit negation was proved."

    def generate_open_answer(
        self,
        question: str,
        executed_query: str,
        bindings: List[dict],
        truncated: bool = False,
    ) -> str:
        """
        Answer an open question by listing the values that were proved.

        `generate_from_polarity` can only say yes, no, or I-don't-know, which cannot
        answer "which biomarkers predict severity". Every value here came back with
        its own proof, so listing them adds no claim beyond what was proved.
        """
        target = self._extract_query_target(executed_query) or "the requested pattern"
        if not bindings:
            return (
                "I don't know - no value could be proved for "
                f"{target}."
            )
        values: List[str] = []
        for row in bindings:
            bound = row.get("bindings") or {}
            rendered = ", ".join(str(v) for v in bound.values() if str(v).strip())
            if rendered and rendered not in values:
                values.append(rendered)
        listed = "; ".join(values)
        probabilistic = any(
            row.get("support_kind") == "probabilistic" for row in bindings
        )
        prefix = (
            "Proved, with at least one probabilistic step: "
            if probabilistic
            else "Proved: "
        )
        suffix = " (more answers may exist; results were capped)" if truncated else ""
        noun = "answer" if len(values) == 1 else "answers"
        return f"{prefix}{len(values)} {noun} for {target} - {listed}.{suffix}"

    def _call_gemini(self, user_prompt: str) -> str:
        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=self._gemini_api_key)
            response = client.models.generate_content(
                model=self._gemini_model,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0.1,
                    max_output_tokens=1000,
                ),
            )
            return response.text
        except ImportError:
            return self._call_openai(user_prompt)

    def _call_openai(self, user_prompt: str) -> str:
        from openai import OpenAI

        client = OpenAI(api_key=self._openai_api_key)
        response = client.chat.completions.create(
            model=self._openai_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=1000,
        )
        return response.choices[0].message.content

    def _fallback_answer(
        self,
        question: str,
        proof_traces: List[str],
        executed_query: str = "",
    ) -> str:
        proof = " ".join(proof_traces)
        if not proof:
            return "I don't know - no proof was found for this question."

        target = self._extract_query_target(executed_query)
        if target:
            return self._answer_from_target(target, proof_traces)

        return f"I found a proof, but could not generate a fluent answer. Proof: {proof}"

    def _answer_from_target(self, target: str, proof_traces: List[str]) -> str:
        proof_atoms = [self._normalize_spaces(str(item)) for item in proof_traces]
        target = self._normalize_spaces(target)
        negated = f"(Not {target})"

        if any(self._statement_contains_body(item, negated) for item in proof_atoms):
            return f"No. The proof contains explicit negation of {target}."
        if any(self._statement_contains_body(item, target) for item in proof_atoms):
            return f"Yes. The proof establishes {target}."
        return (
            "I don't know. A proof was found, but it does not establish "
            f"the executed target {target}."
        )

    def _extract_query_target(self, query: str) -> str:
        match = re.fullmatch(
            r"\(:\s+[$?][^\s]+\s+(\(.+\))\s+[$?][^\s]+\)",
            self._normalize_spaces(query),
        )
        return self._normalize_spaces(match.group(1)) if match else ""

    def _statement_contains_body(self, statement: str, body: str) -> bool:
        return body in statement

    def _normalize_spaces(self, text: str) -> str:
        if not text:
            return ""
        return " ".join(str(text).split())
