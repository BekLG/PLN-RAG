import uuid
import httpx
import hashlib
import math
import re
from collections import OrderedDict
from typing import Any, List, Tuple
from config import get_settings


class VectorStore:
    """
    Manages NL ↔ PLN atom mappings in Qdrant.

    Stores one vector point per validated claim-evidence target. The vector is
    built from natural-language evidence; PLN remains structured payload.
    """

    def __init__(self):
        cfg = get_settings()
        self._qdrant = cfg.qdrant_url
        self._ollama = cfg.ollama_url
        self._ollama_model = cfg.ollama_model
        self._collection = cfg.qdrant_collection
        self._predicate_collection = cfg.predicate_mapping_collection
        self._client = httpx.Client(timeout=30)
        self._collection_sizes: dict[str, int] = {}
        self._embedding_cache: OrderedDict[str, List[float]] = OrderedDict()
        self._embedding_cache_size = 256
        self._hybrid_enabled = cfg.qdrant_hybrid_enabled

    def embed(self, text: str) -> List[float]:
        cached = self._embedding_cache.get(text)
        if cached is not None:
            self._embedding_cache.move_to_end(text)
            return cached
        resp = self._client.post(self._ollama, json={
            "model": self._ollama_model,
            "prompt": text
        })
        resp.raise_for_status()
        vector = resp.json()["embedding"]
        self._cache_embedding(text, vector)
        return vector

    def embed_many(self, texts: List[str]) -> List[List[float]]:
        """Embed uncached texts in one Ollama request when possible."""
        if not texts:
            return []
        missing = list(dict.fromkeys(
            text for text in texts if text not in self._embedding_cache
        ))
        if len(missing) == 1:
            vector = self.embed(missing[0])
            if missing[0] not in self._embedding_cache:
                self._cache_embedding(missing[0], vector)
        elif missing:
            batch_url = self._ollama.replace("/api/embeddings", "/api/embed")
            try:
                resp = self._client.post(
                    batch_url,
                    json={"model": self._ollama_model, "input": missing},
                )
                resp.raise_for_status()
                vectors = resp.json().get("embeddings", [])
                if len(vectors) != len(missing):
                    raise ValueError("Ollama returned an incomplete embedding batch")
                for text, vector in zip(missing, vectors):
                    self._cache_embedding(text, vector)
            except Exception:
                # Older Ollama versions may not expose /api/embed.
                for text in missing:
                    vector = self.embed(text)
                    if text not in self._embedding_cache:
                        self._cache_embedding(text, vector)
        return [self._embedding_cache[text] for text in texts]

    def _cache_embedding(self, text: str, vector: List[float]) -> None:
        self._embedding_cache[text] = vector
        self._embedding_cache.move_to_end(text)
        while len(self._embedding_cache) > self._embedding_cache_size:
            self._embedding_cache.popitem(last=False)

    def _ensure_collection(self, collection: str, vector_size: int):
        if self._collection_sizes.get(collection) == vector_size:
            return
        try:
            self._client.get(
                f"{self._qdrant}/collections/{collection}"
            ).raise_for_status()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                self._client.put(
                    f"{self._qdrant}/collections/{collection}",
                    json={"vectors": {"size": vector_size, "distance": "Cosine"}}
                ).raise_for_status()
        self._collection_sizes[collection] = vector_size

    def _ensure_evidence_collection(self, vector_size: int) -> None:
        cache_key = f"{self._collection}:evidence_v2"
        if self._collection_sizes.get(cache_key) == vector_size:
            return
        try:
            self._client.get(
                f"{self._qdrant}/collections/{self._collection}"
            ).raise_for_status()
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code != 404:
                raise
            self._client.put(
                f"{self._qdrant}/collections/{self._collection}",
                json={
                    "vectors": {
                        "dense": {"size": vector_size, "distance": "Cosine"}
                    },
                    "sparse_vectors": {"lexical": {}},
                },
            ).raise_for_status()
        self._collection_sizes[cache_key] = vector_size

    def store(
        self,
        sentence: str,
        atoms: List[str],
        vector: List[float],
        metadata: dict[str, Any] | None = None,
        query_targets: List[str] | None = None,
    ):
        payload: dict[str, Any] = {
            "nl": sentence,
            "original_text": sentence,
            "retrieval_text": sentence,
            "pln": atoms,
            "query_targets": query_targets or [],
            "validation_state": "accepted",
            "schema_version": 1,
        }
        if metadata:
            payload["metadata"] = metadata
        payload["index_id"] = str(uuid.uuid4())
        self._upsert_evidence_records([payload], [vector])

    def store_claim_records(self, records: List[dict[str, Any]]) -> None:
        """Index source-linked records emitted by the evidence ledger outbox."""
        prepared = [
            dict(record)
            for record in records
            if record.get("index_id")
            and record.get("validation_state") == "accepted"
            and str(record.get("retrieval_text") or "").strip()
        ]
        if not prepared:
            return
        vectors = self.embed_many(
            [str(record["retrieval_text"]) for record in prepared]
        )
        self._upsert_evidence_records(prepared, vectors)

    def _upsert_evidence_records(
        self,
        records: List[dict[str, Any]],
        vectors: List[List[float]],
    ) -> None:
        if not records or not vectors:
            return
        self._ensure_evidence_collection(len(vectors[0]))
        points: List[dict[str, Any]] = []
        for record, vector in zip(records, vectors):
            index_id = str(record.get("index_id") or uuid.uuid4())
            payload = {key: value for key, value in record.items() if key != "outbox_id"}
            points.append(
                {
                    "id": str(
                        uuid.uuid5(
                            uuid.NAMESPACE_URL,
                            f"{self._collection}:{index_id}",
                        )
                    ),
                    "vector": {
                        "dense": vector,
                        "lexical": self._sparse_vector(
                            str(record.get("retrieval_text") or "")
                        ),
                    },
                    "payload": payload,
                }
            )
        self._client.put(
            f"{self._qdrant}/collections/{self._collection}/points?wait=true",
            json={"points": points},
        ).raise_for_status()

    def search(
        self,
        text: str,
        top_k: int,
        min_score: float | None = None,
    ) -> Tuple[List[dict[str, Any]], List[float]]:
        vector = self.embed(text)
        self._ensure_evidence_collection(len(vector))

        score_kind = "dense_cosine"
        if self._hybrid_enabled:
            resp = self._client.post(
                f"{self._qdrant}/collections/{self._collection}/points/query",
                json={
                    "prefetch": [
                        {
                            "query": vector,
                            "using": "dense",
                            "limit": max(top_k * 3, top_k),
                        },
                        {
                            "query": self._sparse_vector(text),
                            "using": "lexical",
                            "limit": max(top_k * 3, top_k),
                        },
                    ],
                    "query": {"rrf": {}},
                    "limit": top_k,
                    "with_payload": True,
                },
            )
            if resp.status_code == 200:
                score_kind = "hybrid_rrf"
        else:
            resp = self._dense_search(vector, top_k)
        if resp.status_code != 200 and self._hybrid_enabled:
            resp = self._dense_search(vector, top_k)
        if resp.status_code != 200:
            return [], vector

        matches: List[dict[str, Any]] = []
        result = resp.json().get("result", [])
        if isinstance(result, dict):
            result = result.get("points", [])
        for item in result:
            score = item.get("score", 0)
            if min_score is not None and score < min_score:
                continue
            payload = item.get("payload", {}) or {}
            matches.append(
                {
                    "score": score,
                    "score_kind": score_kind,
                    "nl": payload.get("nl", ""),
                    "pln": payload.get("pln", []),
                    "query_targets": payload.get("query_targets", []),
                    "metadata": payload.get("metadata", {}),
                    "claim_id": payload.get("claim_id", ""),
                    "evidence_id": payload.get("evidence_id", ""),
                    "claim_evidence_id": payload.get("claim_evidence_id", ""),
                    "predicate": payload.get("predicate", ""),
                    "arguments": payload.get("arguments", []),
                    "polarity": payload.get("polarity", "positive"),
                    "validation_state": payload.get("validation_state", ""),
                    "source_start": payload.get("source_start"),
                    "source_end": payload.get("source_end"),
                }
            )

        return matches, vector

    def _dense_search(self, vector: List[float], top_k: int) -> httpx.Response:
        return self._client.post(
            f"{self._qdrant}/collections/{self._collection}/points/search",
            json={
                "vector": {"name": "dense", "vector": vector},
                "limit": top_k,
                "with_payload": True,
            },
        )

    def _sparse_vector(self, text: str) -> dict[str, List[float] | List[int]]:
        counts: dict[int, int] = {}
        for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9_-]*", text.lower()):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") & 0x7FFFFFFF
            counts[index] = counts.get(index, 0) + 1
        indices = sorted(counts)
        return {
            "indices": indices,
            "values": [1.0 + math.log(counts[index]) for index in indices],
        }

    def retrieve_context(self, text: str, top_k: int) -> Tuple[List[str], List[float]]:
        """
        Returns (context_atoms, embedding_vector).
        context_atoms: flat list of PLN atom strings from top-k similar sentences.
        """
        matches, vector = self.search(text, top_k=top_k)
        context: List[str] = []
        seen: set[str] = set()
        for item in matches:
            pln = item.get("pln", [])
            if isinstance(pln, list):
                for atom in pln:
                    clean = " ".join(str(atom).split())
                    if clean and clean not in seen:
                        seen.add(clean)
                        context.append(clean)

        return context, vector

    def upsert_predicate_cards(self, cards: List[dict[str, Any]]) -> None:
        """Embed and upsert predicate cards into a dedicated Qdrant collection."""
        prepared: List[tuple[dict[str, Any], str, str]] = []
        for card in cards:
            key = str(card.get("key", "")).strip()
            predicate = str(card.get("predicate", "")).strip()
            if not key or not predicate:
                continue
            search_text = str(card.get("search_text", "")).strip()
            if not search_text:
                search_text = self._predicate_card_search_text(card)
            prepared.append((card, key, search_text))

        vectors = self.embed_many([item[2] for item in prepared])
        points: List[dict[str, Any]] = []
        for (card, key, _search_text), vector in zip(prepared, vectors):
            self._ensure_collection(self._predicate_collection, len(vector))
            points.append(
                {
                    "id": str(
                        uuid.uuid5(
                            uuid.NAMESPACE_URL,
                            f"{self._predicate_collection}:{key}",
                        )
                    ),
                    "vector": vector,
                    "payload": dict(card),
                }
            )
        if not points:
            return
        self._client.put(
            f"{self._qdrant}/collections/{self._predicate_collection}"
            "/points?wait=true",
            json={"points": points},
        ).raise_for_status()

    def search_predicate_cards(
        self,
        search_text: str,
        *,
        arity: int,
        top_k: int,
        min_score: float,
    ) -> List[dict[str, Any]]:
        vector = self.embed(search_text)
        self._ensure_collection(self._predicate_collection, len(vector))
        resp = self._client.post(
            f"{self._qdrant}/collections/{self._predicate_collection}"
            "/points/search",
            json={
                "vector": vector,
                "limit": top_k,
                "with_payload": True,
                "filter": {
                    "must": [
                        {"key": "kind", "match": {"value": "predicate_card"}},
                        {"key": "arity", "match": {"value": int(arity)}},
                    ]
                },
            },
        )
        if resp.status_code != 200:
            return []
        results: List[dict[str, Any]] = []
        for item in resp.json().get("result", []):
            score = float(item.get("score", 0.0) or 0.0)
            if score < min_score:
                continue
            results.append(
                {
                    "id": item.get("id"),
                    "score": score,
                    "payload": item.get("payload", {}) or {},
                }
            )
        return results

    def _predicate_card_search_text(self, card: dict[str, Any]) -> str:
        explicit = str(card.get("search_text", "")).strip()
        if explicit:
            return explicit
        parts = [
            f"predicate: {card.get('predicate', '')}",
            f"labels: {' '.join(card.get('labels', []) or [])}",
            f"argument types: {' '.join(card.get('argument_types', []) or [])}",
            f"definition: {card.get('definition', '')}",
            f"evidence: {' '.join(card.get('examples', [])[:1] or [])[:180]}",
            f"atom: {' '.join(card.get('source_atoms', [])[:1] or [])}",
        ]
        return "; ".join(part for part in parts if part.split(":", 1)[-1].strip())

    def list_points(self, limit: int = 50) -> List[dict[str, Any]]:
        """
        Return raw Qdrant payloads for debugging.
        Vectors are intentionally omitted because they are large and not useful
        for normal inspection.
        """
        try:
            resp = self._client.post(
                f"{self._qdrant}/collections/{self._collection}/points/scroll",
                json={
                    "limit": limit,
                    "with_payload": True,
                    "with_vector": False,
                },
            )
            if resp.status_code != 200:
                return []
            points = resp.json().get("result", {}).get("points", [])
            return [
                {
                    "id": point.get("id"),
                    "payload": point.get("payload", {}),
                }
                for point in points
            ]
        except Exception:
            return []

    def list_predicate_points(self, limit: int = 50) -> List[dict[str, Any]]:
        return self._list_collection_points(self._predicate_collection, limit)

    def _list_collection_points(
        self,
        collection: str,
        limit: int,
    ) -> List[dict[str, Any]]:
        try:
            resp = self._client.post(
                f"{self._qdrant}/collections/{collection}/points/scroll",
                json={
                    "limit": limit,
                    "with_payload": True,
                    "with_vector": False,
                },
            )
            if resp.status_code != 200:
                return []
            points = resp.json().get("result", {}).get("points", [])
            return [
                {
                    "id": point.get("id"),
                    "payload": point.get("payload", {}),
                }
                for point in points
            ]
        except Exception:
            return []

    def reset(self):
        for collection in (self._collection, self._predicate_collection):
            try:
                self._client.delete(f"{self._qdrant}/collections/{collection}")
            except Exception:
                pass
        self._collection_sizes.clear()

    def reset_evidence(self) -> None:
        try:
            self._client.delete(f"{self._qdrant}/collections/{self._collection}")
        except Exception:
            pass
        self._collection_sizes = {
            key: value
            for key, value in self._collection_sizes.items()
            if not key.startswith(self._collection)
        }

    @property
    def count(self) -> int:
        return self._collection_count(self._collection)

    @property
    def predicate_count(self) -> int:
        return self._collection_count(self._predicate_collection)

    def _collection_count(self, collection: str) -> int:
        try:
            resp = self._client.get(f"{self._qdrant}/collections/{collection}")
            return resp.json().get("result", {}).get("points_count", 0)
        except Exception:
            return 0
