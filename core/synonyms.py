import os
import json
import httpx
import dspy
from typing import Dict, List
from config import get_settings


class _CompareWords(dspy.Signature):
    """
    You are a strict linguist. Compare the following two words.
    Are they exactly the same meaning (synonyms), is one broader, narrower, or are they just related/unrelated?
    Note: Treat regional variations of the exact same underlying concept (like "football" and "soccer", or "attorney" and "lawyer") as EXACT synonyms ("same_meaning").
    Respond ONLY with one of the following exact words:
    same_meaning
    broader
    narrower
    related
    unrelated
    """
    word1: str = dspy.InputField()
    word2: str = dspy.InputField()
    relationship: str = dspy.OutputField(desc="Must be exactly one of: same_meaning, broader, narrower, related, unrelated")


class HybridSynonymVerifier:
    """
    Verifies if two words are synonyms using DSPy/LLM with a strict prompt.
    Caches verified relationships in data/synonyms/relations.json.
    """
    def __init__(self):
        cfg = get_settings()
        self._cache_file = "data/synonyms/relations.json"
        self._cache: Dict[str, str] = {}
        self._embedding_cache: Dict[str, List[float]] = {}
        
        self._ollama_url = cfg.ollama_url
        self._ollama_model = cfg.ollama_model
        self._client = httpx.Client(timeout=10)
        
        self._load_cache()

        try:
            if not dspy.settings.lm:
                lm = dspy.LM(cfg.openai_model, api_key=cfg.openai_api_key, cache=False)
                dspy.configure(lm=lm, temperature=0.0)
        except Exception:
            lm = dspy.LM(cfg.openai_model, api_key=cfg.openai_api_key, cache=False)
            dspy.configure(lm=lm, temperature=0.0)
            
        self._predict = dspy.Predict(_CompareWords)

    def _load_cache(self):
        os.makedirs(os.path.dirname(self._cache_file), exist_ok=True)
        if os.path.exists(self._cache_file):
            try:
                with open(self._cache_file, "r") as f:
                    self._cache = json.load(f)
            except Exception:
                self._cache = {}

    def _save_cache(self):
        with open(self._cache_file, "w") as f:
            json.dump(self._cache, f, indent=2)

    def _get_embedding(self, text: str) -> List[float]:
        text = text.lower().replace("_", " ")
        if text in self._embedding_cache:
            return self._embedding_cache[text]
        try:
            resp = self._client.post(self._ollama_url, json={
                "model": self._ollama_model,
                "prompt": text
            })
            resp.raise_for_status()
            emb = resp.json()["embedding"]
            self._embedding_cache[text] = emb
            return emb
        except Exception:
            return []

    def _cosine_similarity(self, v1: List[float], v2: List[float]) -> float:
        if not v1 or not v2 or len(v1) != len(v2):
            return 0.0
        dot = sum(a * b for a, b in zip(v1, v2))
        mag1 = sum(a * a for a in v1) ** 0.5
        mag2 = sum(b * b for b in v2) ** 0.5
        if mag1 == 0 or mag2 == 0:
            return 0.0
        return dot / (mag1 * mag2)

    def get_candidates(self, query_word: str, target_words: List[str], top_k: int = 1, threshold: float = 0.5) -> List[str]:
        q_emb = self._get_embedding(query_word)
        if not q_emb:
            return []
            
        scores = []
        for word in target_words:
            w_emb = self._get_embedding(word)
            if not w_emb:
                continue
            sim = self._cosine_similarity(q_emb, w_emb)
            if sim >= threshold:
                scores.append((word, sim))
                
        scores.sort(key=lambda x: x[1], reverse=True)
        return [word for word, sim in scores[:top_k]]

    def verify_synonym(self, word1: str, word2: str) -> str:
        """Returns 'same_meaning', 'broader', 'narrower', 'related', 'unrelated'"""
        w1, w2 = sorted([word1.lower(), word2.lower()])
        cache_key = f"{w1}::{w2}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            result = self._predict(word1=w1, word2=w2).relationship.strip().lower()
            valid = ["same_meaning", "broader", "narrower", "related", "unrelated"]
            for v in valid:
                if v in result:
                    self._cache[cache_key] = v
                    self._save_cache()
                    return v
        except Exception as e:
            print(f"[HybridVerifier] Error verifying: {e}")
        
        self._cache[cache_key] = "unrelated"
        self._save_cache()
        return "unrelated"
