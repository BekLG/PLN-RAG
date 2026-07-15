from __future__ import annotations

import threading
from typing import Any

from core.discourse.coreference import (
    DocumentCorefResult,
    build_cluster,
)


class LingMessCoreferenceResolver:
    """Lazy, reusable wrapper around fastcoref.LingMessCoref."""

    backend = "lingmess"

    def __init__(
        self,
        *,
        model_name: str = "biu-nlp/lingmess-coref",
        device: str = "auto",
        max_tokens_in_batch: int = 10000,
        include_pair_logits: bool = False,
    ):
        self.model_name = model_name
        self._device = device
        self._max_tokens_in_batch = max_tokens_in_batch
        self._include_pair_logits = include_pair_logits
        self._model: Any | None = None
        self._lock = threading.Lock()

    def resolve_document(self, text: str) -> DocumentCorefResult:
        if not text.strip():
            return DocumentCorefResult(
                text=text,
                backend=self.backend,
                model_name=self.model_name,
            )

        model = self._load_model()
        predictions = model.predict(
            texts=[text],
            max_tokens_in_batch=self._max_tokens_in_batch,
        )
        if not predictions:
            return DocumentCorefResult(
                text=text,
                backend=self.backend,
                model_name=self.model_name,
            )

        prediction = predictions[0]
        raw_clusters = prediction.get_clusters(as_strings=False)
        clusters = []
        for index, spans in enumerate(raw_clusters):
            pair_logits = (
                self._pair_logits(prediction, spans)
                if self._include_pair_logits
                else None
            )
            clusters.append(
                build_cluster(
                    cluster_id=f"lingmess_c{index + 1}",
                    text=text,
                    spans=[(int(start), int(end)) for start, end in spans],
                    pair_logits=pair_logits,
                )
            )

        if hasattr(prediction, "release_logits"):
            prediction.release_logits()

        return DocumentCorefResult(
            text=text,
            clusters=clusters,
            backend=self.backend,
            model_name=self.model_name,
        )

    def _load_model(self):
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is not None:
                return self._model
            from fastcoref import LingMessCoref

            self._model = LingMessCoref(
                model_name_or_path=self.model_name,
                device=self._resolved_device(),
            )
            return self._model

    def _resolved_device(self) -> str:
        device = str(self._device or "auto").lower()
        if device == "auto":
            try:
                import torch

                return "cuda:0" if torch.cuda.is_available() else "cpu"
            except Exception:
                return "cpu"
        if device == "cuda":
            return "cuda:0"
        return device

    def _pair_logits(
        self,
        prediction: Any,
        spans: list[tuple[int, int]],
    ) -> dict[tuple[int, int], float]:
        pair_logits: dict[tuple[int, int], float] = {}
        if not hasattr(prediction, "get_logit") or len(spans) < 2:
            return pair_logits
        canonical = spans[0]
        for index, span in enumerate(spans[1:], start=1):
            try:
                pair_logits[(index, 0)] = float(
                    prediction.get_logit(span_i=canonical, span_j=span)
                )
            except Exception:
                continue
        return pair_logits
