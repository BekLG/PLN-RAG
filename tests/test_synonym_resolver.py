import json
from types import SimpleNamespace

from core.horn_fallback import HornFallback
from core.synonym_resolver import SynonymResolver


class FakeResponse:
    def __init__(self, payload=None, output_text=""):
        self._payload = payload or {}
        self.output_text = output_text

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class FakeHTTP:
    def __init__(self, embeddings=None):
        self.embeddings = embeddings or {}
        self.get_calls = []
        self.post_calls = []

    def get(self, url, params=None):
        self.get_calls.append((url, params))
        return FakeResponse({"edges": []})

    def post(self, url, json=None):
        self.post_calls.append((url, json))
        term = json["prompt"].replace(" ", "_")
        return FakeResponse({"embedding": self.embeddings.get(term, [])})


class FakeResponses:
    def __init__(self, decisions):
        self.decisions = list(decisions)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return FakeResponse(output_text=json.dumps(self.decisions.pop(0)))


class FakeOpenAI:
    def __init__(self, decisions):
        self.responses = FakeResponses(decisions)


def settings(tmp_path, **overrides):
    values = {
        "synonym_resolution_enabled": True,
        "synonym_cache_path": str(tmp_path / "relations.json"),
        "synonym_request_timeout": 1,
        "synonym_wordnet_enabled": False,
        "synonym_conceptnet_lookup_enabled": False,
        "synonym_conceptnet_url": "https://api.conceptnet.io",
        "synonym_conceptnet_limit": 10,
        "synonym_embedding_enabled": True,
        "synonym_embedding_threshold": 0.7,
        "synonym_embedding_top_k": 2,
        "synonym_max_knowledge_terms": 20,
        "synonym_max_verifications_per_query": 4,
        "synonym_verifier_model": "gpt-4o-mini",
        "synonym_verifier_min_confidence": 0.85,
        "openai_model": "openai/gpt-4o-mini",
        "openai_api_key": "test",
        "ollama_url": "http://ollama/api/embeddings",
        "ollama_model": "nomic-embed-text",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_embedding_only_proposes_and_openai_approves_synonym(tmp_path):
    http = FakeHTTP(
        {
            "attorney": [1.0, 0.0],
            "lawyer": [0.99, 0.01],
            "professional": [0.8, 0.2],
        }
    )
    openai = FakeOpenAI(
        [
            {
                "relation": "same_meaning",
                "confidence": 0.98,
                "reason": "They name the same profession.",
            },
        ]
    )
    resolver = SynonymResolver(settings(tmp_path), http, openai)

    pairs = resolver.discover_equivalences(
        "(Educated attorney)",
        [
            "(: fact1 (Professional lawyer) (STV 1 1))",
            "(: rule1 (Implication (Premises (Professional $x)) "
            "(Conclusions (Educated $x))) (STV 1 1))",
        ],
        "is an attorney educated",
    )

    assert ("attorney", "lawyer") in pairs
    assert ("attorney", "professional") not in pairs
    assert len(openai.responses.calls) == 1


def test_related_embedding_pair_never_becomes_equivalent(tmp_path):
    http = FakeHTTP({"soccer": [1.0, 0.0], "sport": [0.99, 0.01]})
    openai = FakeOpenAI(
        [
            {
                "relation": "narrower",
                "confidence": 0.99,
                "reason": "Soccer is a kind of sport.",
            }
        ]
    )
    resolver = SynonymResolver(settings(tmp_path), http, openai)

    pairs = resolver.discover_equivalences(
        "(Healthy soccer)",
        ["(: fact1 (Healthy sport) (STV 1 1))"],
    )

    assert pairs == set()
    cache = json.loads((tmp_path / "relations.json").read_text())
    assert cache["pairs"]["soccer|sport"]["relation"] == "narrower"


def test_persistent_synonym_cache_skips_external_calls(tmp_path):
    cache_path = tmp_path / "relations.json"
    cache_path.write_text(
        json.dumps(
            {
                "version": 1,
                "pairs": {
                    "attorney|lawyer": {
                        "left": "attorney",
                        "right": "lawyer",
                        "relation": "same_meaning",
                        "confidence": 0.99,
                        "source": "test",
                        "reason": "same",
                        "updated_at": "2026-01-01T00:00:00+00:00",
                    }
                },
            }
        )
    )
    http = FakeHTTP()
    openai = FakeOpenAI([])
    resolver = SynonymResolver(settings(tmp_path), http, openai)

    pairs = resolver.discover_equivalences(
        "(Educated attorney)",
        ["(: fact1 (Educated lawyer) (STV 1 1))"],
    )

    assert ("attorney", "lawyer") in pairs
    assert http.post_calls == []
    assert openai.responses.calls == []


def test_verified_pair_is_used_by_horn_reasoning():
    statements = [
        "(: fact1 (Professional lawyer) (STV 1 1))",
        "(: rule1 (Implication (Premises (Professional $x)) "
        "(Conclusions (Educated $x))) (STV 1 1))",
    ]

    proof = HornFallback(
        statements,
        additional_equivalences={("attorney", "lawyer")},
    ).prove("(Educated attorney)")

    assert len(proof) == 2
