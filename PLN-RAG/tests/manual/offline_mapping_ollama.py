from __future__ import annotations

import json

import requests


PAIR = {
    "id": 0,
    "source": {
        "predicate": "ConsumesHigherCarbohydrates",
        "arity": 1,
        "argument_types": ["person"],
    },
    "target": {
        "predicate": "HasHighCarbohydrateIntake",
        "arity": 1,
        "argument_types": ["person"],
    },
}


def main() -> None:
    prompt = (
        "Classify each predicate pair for a formal proof system. "
        "Allowed relations: exactMatch, source_implies_target, "
        "target_implies_source, broader, narrower, related, contradiction, "
        "unrelated. Be conservative: topical similarity is not entailment. "
        "Preserve ordered arguments; argument_mapping uses zero-based "
        "positions and must be identity for a direct bridge. Return exactly "
        "one result per id as JSON: "
        '{"results":[{"id":0,"relation":"...","confidence":0.0,'
        '"argument_mapping":[0],"reason":"..."}]}.\nPAIRS:\n'
        + json.dumps([PAIR], separators=(",", ":"))
    )
    response = requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model": "llama3",
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Return strict JSON. Classify predicate semantics "
                        "conservatively for a formal proof system."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "format": "json",
            "stream": False,
        },
        timeout=30,
    )
    response.raise_for_status()
    print(response.text)


if __name__ == "__main__":
    main()
