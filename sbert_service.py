from __future__ import annotations

import time
from threading import Lock
from typing import Any


SUPPORTED_SBERT_MODELS = {
    "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
    "Xenova/all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
    "gte-small": "thenlper/gte-small",
    "thenlper/gte-small": "thenlper/gte-small",
    "Xenova/gte-small": "thenlper/gte-small",
}

CANONICAL_SBERT_MODELS = {
    "sentence-transformers/all-MiniLM-L6-v2": "all-MiniLM-L6-v2",
    "thenlper/gte-small": "thenlper/gte-small",
}


def finite_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if number != number or number in (float("inf"), float("-inf")):
        return default
    return number


class SbertSimilarityService:
    def __init__(self) -> None:
        self._models: dict[str, Any] = {}
        self._lock = Lock()

    def calculate_pair(self, payload: dict[str, Any]) -> dict[str, Any]:
        model_name = str(payload.get("model") or "all-MiniLM-L6-v2").strip()
        text_a = str(payload.get("textA") or "").strip()
        text_b = str(payload.get("textB") or "").strip()
        if not text_a or not text_b:
            raise ValueError("textA y textB son obligatorios.")

        embeddings, cost = self.encode_texts(model_name, [text_a, text_b])
        similarity = finite_float(embeddings[0] @ embeddings[1])
        return {
            "model": cost["embeddingModel"],
            "sourceModel": cost["sourceModel"],
            "backend": "python-sentence-transformers",
            "similarity": similarity,
            "distance": 1.0 - similarity,
            "dimension": int(embeddings.shape[1]),
            "elapsedSeconds": cost["embeddingWallClockSeconds"],
            "embeddings": [embeddings[0].tolist(), embeddings[1].tolist()],
        }

    def calculate_embeddings(self, payload: dict[str, Any]) -> dict[str, Any]:
        model_name = str(payload.get("model") or "all-MiniLM-L6-v2").strip()
        texts = payload.get("texts")
        if not isinstance(texts, list) or not all(isinstance(text, str) for text in texts):
            raise ValueError("texts debe ser una lista de strings.")
        if not texts:
            raise ValueError("texts debe contener al menos un texto.")

        embeddings, cost = self.encode_texts(model_name, texts)
        return {
            "model": cost["embeddingModel"],
            "sourceModel": cost["sourceModel"],
            "backend": "python-sentence-transformers",
            "dimension": int(embeddings.shape[1]),
            "elapsedSeconds": cost["embeddingWallClockSeconds"],
            "embeddings": [embedding.tolist() for embedding in embeddings],
        }

    def encode_texts(self, model_name: str, texts: list[str]) -> tuple[Any, dict[str, Any]]:
        if not texts:
            return [], self.cost_summary(model_name, 0, 0.0)
        started = time.perf_counter()
        model = self._model(model_name)
        embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        elapsed = time.perf_counter() - started
        return embeddings, self.cost_summary(model_name, len(texts), elapsed)

    def similarities(self, reference: str, candidates: list[str], model_name: str) -> tuple[list[float], dict[str, Any]]:
        if not candidates:
            return [], self.cost_summary(model_name, 0, 0.0)
        embeddings, cost = self.encode_texts(model_name, [reference, *candidates])
        reference_embedding = embeddings[0]
        candidate_embeddings = embeddings[1:]
        return [finite_float(value) for value in candidate_embeddings @ reference_embedding], cost

    def warmup(self, model_name: str) -> None:
        self._model(model_name)

    def model_wrapper(self, model_name: str) -> "SbertModelWrapper":
        return SbertModelWrapper(self, model_name)

    def cost_summary(self, model_name: str, text_count: int, elapsed_seconds: float) -> dict[str, Any]:
        source_model = self.source_model_name(model_name)
        return {
            "embeddingModel": CANONICAL_SBERT_MODELS[source_model],
            "sourceModel": source_model,
            "embeddingTexts": text_count,
            "embeddingWallClockSeconds": elapsed_seconds,
        }

    def source_model_name(self, model_name: str) -> str:
        if model_name not in SUPPORTED_SBERT_MODELS:
            raise ValueError(f"Modelo SBERT no soportado: {model_name}")
        return SUPPORTED_SBERT_MODELS[model_name]

    def _model(self, model_name: str) -> Any:
        source_model = self.source_model_name(model_name)
        if source_model in self._models:
            return self._models[source_model]
        with self._lock:
            if source_model not in self._models:
                from sentence_transformers import SentenceTransformer

                self._models[source_model] = SentenceTransformer(source_model)
        return self._models[source_model]


class SbertModelWrapper:
    def __init__(self, service: SbertSimilarityService, model_name: str) -> None:
        self.service = service
        self.model_name = service.cost_summary(model_name, 0, 0.0)["embeddingModel"]
        self.source_model = service.source_model_name(model_name)

    def encode(self, texts: list[str], convert_to_numpy: bool = True, normalize_embeddings: bool = True) -> Any:
        if not convert_to_numpy or not normalize_embeddings:
            raise ValueError("El runtime compartido requiere convert_to_numpy=True y normalize_embeddings=True.")
        embeddings, _ = self.service.encode_texts(self.model_name, texts)
        return embeddings


_SHARED_SBERT_SERVICE = SbertSimilarityService()


def shared_sbert_service() -> SbertSimilarityService:
    return _SHARED_SBERT_SERVICE
