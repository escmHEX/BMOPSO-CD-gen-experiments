from __future__ import annotations

import csv
import json
import random
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

from sbert_service import normalize_projection_method, project_embeddings_2d, shared_sbert_service


STATUS_QUEUED = "queued"
STATUS_RUNNING = "running"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"

DEFAULT_REFERENCE_TEXT_SELECTION_CONFIG = {
    "datasetPath": "data/external/covid19_tweets/hydrated/10k_data.tsv",
    "sampleSize": 1000,
    "clusterCount": 4,
    "minWords": 20,
    "maxWords": 80,
    "seed": 42,
    "semanticWeight": 0.7,
    "embeddingModel": "all-MiniLM-L6-v2",
}

RANKED_CANDIDATE_LIMIT = 25
SAMPLED_CANDIDATES_FILE = "sampled_candidates.json"
SAMPLE_EMBEDDINGS_FILE = "sample_embeddings.npy"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_dataset_path(root: Path, value: Any) -> Path:
    raw_path = str(value or "").strip()
    if not raw_path:
        raise ValueError("datasetPath is required.")
    path = Path(raw_path)
    if not path.is_absolute():
        path = root / path
    return path


def int_between(value: Any, label: str, minimum: int, maximum: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} debe ser un entero.") from error
    if number < minimum or number > maximum:
        raise ValueError(f"{label} debe estar entre {minimum} y {maximum}.")
    return number


def float_between(value: Any, label: str, minimum: float, maximum: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} debe ser numerico.") from error
    if number < minimum or number > maximum:
        raise ValueError(f"{label} debe estar entre {minimum} y {maximum}.")
    return number


def normalize_config(payload: dict[str, Any], embedding_service: Any | None = None) -> dict[str, Any]:
    defaults = DEFAULT_REFERENCE_TEXT_SELECTION_CONFIG
    config = {
        "datasetPath": str(payload.get("datasetPath", defaults["datasetPath"]) or "").strip(),
        "sampleSize": int_between(payload.get("sampleSize", defaults["sampleSize"]), "sampleSize", 1, 1_000_000),
        "clusterCount": int_between(payload.get("clusterCount", defaults["clusterCount"]), "clusterCount", 1, 10_000),
        "minWords": int_between(payload.get("minWords", defaults["minWords"]), "minWords", 0, 100_000),
        "maxWords": int_between(payload.get("maxWords", defaults["maxWords"]), "maxWords", 0, 100_000),
        "seed": int_between(payload.get("seed", defaults["seed"]), "seed", 0, 2_147_483_647),
        "semanticWeight": float_between(
            payload.get("semanticWeight", defaults["semanticWeight"]),
            "semanticWeight",
            0.0,
            1.0,
        ),
        "embeddingModel": str(payload.get("embeddingModel", defaults["embeddingModel"]) or "").strip(),
    }
    if not config["datasetPath"]:
        raise ValueError("datasetPath is required.")
    if config["minWords"] > config["maxWords"]:
        raise ValueError("minWords debe ser menor o igual a maxWords.")
    if not config["embeddingModel"]:
        raise ValueError("embeddingModel is required.")
    service = embedding_service or shared_sbert_service()
    service.source_model_name(config["embeddingModel"])
    return config


def word_count(text: str) -> int:
    return len(text.split())


def load_tsv_candidates(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise ValueError(f"No existe el TSV configurado: {path}")
    candidates: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fieldnames = set(reader.fieldnames or [])
        if not {"tweetId", "texto"}.issubset(fieldnames):
            raise ValueError("El TSV debe incluir las columnas tweetId y texto.")
        for original_index, row in enumerate(reader, start=1):
            text = str(row.get("texto") or "").strip()
            candidates.append(
                {
                    "tweetId": str(row.get("tweetId") or "").strip(),
                    "text": text,
                    "originalIndex": original_index,
                    "wordCount": word_count(text),
                }
            )
    return candidates


def min_max_normalize(values: list[float]) -> list[float]:
    minimum = min(values)
    maximum = max(values)
    if maximum == minimum:
        return [0.0 for _value in values]
    return [(value - minimum) / (maximum - minimum) for value in values]


def normalized_embeddings(embeddings: Any) -> Any:
    import numpy as np

    matrix = np.asarray(embeddings, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise ValueError("El modelo de embeddings devolvio una matriz vacia.")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("El modelo de embeddings devolvio un vector con norma cero.")
    return matrix / norms


def select_majority_cluster(labels: Any, centers: Any, embeddings: Any, cluster_count: int) -> int:
    import numpy as np

    best: tuple[int, float, int] | None = None
    best_index = 0
    for cluster_index in range(cluster_count):
        indices = np.where(labels == cluster_index)[0]
        size = int(len(indices))
        inertia = float(np.sum((embeddings[indices] - centers[cluster_index]) ** 2)) if size else float("inf")
        key = (-size, inertia, cluster_index)
        if best is None or key < best:
            best = key
            best_index = cluster_index
    return best_index


def select_reference_text(
    payload: dict[str, Any],
    *,
    root: Path,
    embedding_service: Any | None = None,
) -> dict[str, Any]:
    service = embedding_service or shared_sbert_service()
    config = normalize_config(payload, service)
    dataset_path = resolve_dataset_path(root, config["datasetPath"])
    all_candidates = load_tsv_candidates(dataset_path)
    filtered = [
        candidate
        for candidate in all_candidates
        if config["minWords"] <= int(candidate["wordCount"]) <= config["maxWords"]
    ]
    if not filtered:
        raise ValueError("No existen textos que satisfagan el filtro de longitud.")

    sample_size = min(config["sampleSize"], len(filtered))
    if sample_size < config["clusterCount"]:
        raise ValueError("clusterCount no puede ser mayor que la cantidad de textos de la muestra.")
    sample = random.Random(config["seed"]).sample(filtered, sample_size) if len(filtered) > sample_size else list(filtered)

    import numpy as np
    from sklearn.cluster import KMeans

    texts = [str(candidate["text"]) for candidate in sample]
    embeddings, cost = service.encode_texts(config["embeddingModel"], texts)
    embeddings = normalized_embeddings(embeddings)

    model = KMeans(
        n_clusters=config["clusterCount"],
        random_state=config["seed"],
        n_init=20,
        max_iter=300,
    )
    labels = model.fit_predict(embeddings)
    centers = model.cluster_centers_
    cluster_index = select_majority_cluster(labels, centers, embeddings, config["clusterCount"])
    cluster_indices = [index for index, label in enumerate(labels) if int(label) == cluster_index]
    cluster_candidates = [sample[index] for index in cluster_indices]
    cluster_embeddings = embeddings[cluster_indices]
    centroid = centers[cluster_index]
    centroid_norm = float(np.linalg.norm(centroid))
    if centroid_norm == 0:
        raise ValueError("KMeans devolvio un centroide con norma cero.")
    centroid = centroid / centroid_norm
    median_length = float(median([int(candidate["wordCount"]) for candidate in cluster_candidates]))

    semantic_distances = [float(1.0 - np.dot(embedding, centroid)) for embedding in cluster_embeddings]
    length_distances = [float(abs(int(candidate["wordCount"]) - median_length)) for candidate in cluster_candidates]
    normalized_semantic = min_max_normalize(semantic_distances)
    normalized_length = min_max_normalize(length_distances)
    ranked: list[dict[str, Any]] = []
    for candidate, semantic_distance, length_distance, semantic_hat, length_hat in zip(
        cluster_candidates,
        semantic_distances,
        length_distances,
        normalized_semantic,
        normalized_length,
    ):
        score = config["semanticWeight"] * semantic_hat + (1.0 - config["semanticWeight"]) * length_hat
        ranked.append(
            {
                "tweetId": candidate["tweetId"],
                "text": candidate["text"],
                "originalIndex": candidate["originalIndex"],
                "wordCount": candidate["wordCount"],
                "clusterIndex": int(cluster_index),
                "clusterDisplayIndex": int(cluster_index) + 1,
                "score": float(score),
                "semanticDistance": float(semantic_distance),
                "lengthDistance": float(length_distance),
                "normalizedSemanticDistance": float(semantic_hat),
                "normalizedLengthDistance": float(length_hat),
            }
        )
    ranked.sort(
        key=lambda candidate: (
            candidate["score"],
            candidate["semanticDistance"],
            candidate["lengthDistance"],
            candidate["originalIndex"],
        )
    )
    selected = ranked[0]
    ranked_by_original_index = {
        candidate["originalIndex"]: {**candidate, "rank": rank}
        for rank, candidate in enumerate(ranked, start=1)
    }
    sampled_candidates = []
    for sample_index, candidate in enumerate(sample):
        candidate_cluster_index = int(labels[sample_index])
        ranked_candidate = ranked_by_original_index.get(candidate["originalIndex"])
        sampled_candidates.append(
            {
                "tweetId": candidate["tweetId"],
                "text": candidate["text"],
                "originalIndex": candidate["originalIndex"],
                "wordCount": candidate["wordCount"],
                "sampleIndex": sample_index,
                "clusterIndex": candidate_cluster_index,
                "clusterDisplayIndex": candidate_cluster_index + 1,
                "isMajorityCluster": candidate_cluster_index == int(cluster_index),
                "isSelected": candidate["originalIndex"] == selected["originalIndex"],
                "rank": ranked_candidate["rank"] if ranked_candidate else None,
                "score": ranked_candidate["score"] if ranked_candidate else None,
                "semanticDistance": ranked_candidate["semanticDistance"] if ranked_candidate else None,
                "lengthDistance": ranked_candidate["lengthDistance"] if ranked_candidate else None,
            }
        )
    cluster_sizes = {
        str(index + 1): int(np.sum(labels == index))
        for index in range(config["clusterCount"])
    }

    return {
        "config": config,
        "datasetPath": str(dataset_path),
        "selectedText": selected["text"],
        "selectedTweetId": selected["tweetId"],
        "selectedOriginalIndex": selected["originalIndex"],
        "wordCount": selected["wordCount"],
        "score": selected["score"],
        "semanticDistance": selected["semanticDistance"],
        "lengthDistance": selected["lengthDistance"],
        "clusterIndex": selected["clusterIndex"],
        "clusterDisplayIndex": selected["clusterDisplayIndex"],
        "filteredCount": len(filtered),
        "sampleCount": len(sample),
        "majorityClusterSize": len(cluster_candidates),
        "embeddingCost": cost,
        "clusterSizes": cluster_sizes,
        "rankedCandidates": ranked[:RANKED_CANDIDATE_LIMIT],
        "_sampledCandidates": sampled_candidates,
        "_sampleEmbeddings": embeddings,
    }


class ReferenceTextSelectionService:
    def __init__(self, root: Path, embedding_service: Any | None = None) -> None:
        self.root = root
        self.runs_root = root / "runs" / "reference-text-selection"
        self.embedding_service = embedding_service or shared_sbert_service()
        self._runs: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()

    def default_config(self) -> dict[str, Any]:
        return dict(DEFAULT_REFERENCE_TEXT_SELECTION_CONFIG)

    def start_run(self, payload: dict[str, Any]) -> dict[str, Any]:
        config = normalize_config(payload, self.embedding_service)
        run_id = self._new_run_id()
        run_dir = self.runs_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        write_json(run_dir / "config.json", config)
        run = {
            "runId": run_id,
            "status": STATUS_QUEUED,
            "createdAt": utc_now(),
            "updatedAt": utc_now(),
            "runDir": str(run_dir),
            "config": config,
            "progress": {
                "percent": 0,
                "detail": "Corrida en cola.",
                "stage": STATUS_QUEUED,
                "elapsedSeconds": 0,
            },
            "error": None,
            "startedAtEpoch": None,
        }
        with self._lock:
            self._runs[run_id] = run
            self._write_summary_unlocked(run)
        thread = threading.Thread(target=self._run_worker, args=(run_id,), daemon=True)
        thread.start()
        return self._public_run(run)

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self._lock:
            run = self._runs.get(run_id)
            if run:
                if run.get("status") == STATUS_RUNNING:
                    self._refresh_progress_unlocked(run, "Ejecutando seleccion.", 50)
                return self._public_run(run)
        summary_path = self.runs_root / str(run_id) / "summary.json"
        if summary_path.exists():
            return read_json(summary_path)
        return None

    def _run_worker(self, run_id: str) -> None:
        with self._lock:
            run = self._runs[run_id]
            run["status"] = STATUS_RUNNING
            run["startedAtEpoch"] = time.time()
            run["updatedAt"] = utc_now()
            self._refresh_progress_unlocked(run, "Leyendo TSV y calculando embeddings.", 10)
            self._write_summary_unlocked(run)
        try:
            result = select_reference_text(
                run["config"],
                root=self.root,
                embedding_service=self.embedding_service,
            )
            run_dir = Path(run["runDir"])
            sampled_candidates = result.pop("_sampledCandidates")
            sample_embeddings = result.pop("_sampleEmbeddings")
            write_json(run_dir / "selected_reference_text.json", {key: result[key] for key in (
                "selectedText",
                "selectedTweetId",
                "selectedOriginalIndex",
                "wordCount",
                "score",
                "semanticDistance",
                "lengthDistance",
                "clusterDisplayIndex",
            )})
            write_json(run_dir / "ranked_candidates.json", result["rankedCandidates"])
            write_json(run_dir / SAMPLED_CANDIDATES_FILE, sampled_candidates)
            import numpy as np

            np.save(run_dir / SAMPLE_EMBEDDINGS_FILE, sample_embeddings)
            with self._lock:
                run.update(result)
                run["status"] = STATUS_COMPLETED
                run["error"] = None
                run["updatedAt"] = utc_now()
                self._refresh_progress_unlocked(run, "Texto de referencia seleccionado.", 100)
                self._write_summary_unlocked(run)
        except Exception as error:
            with self._lock:
                run = self._runs.get(run_id)
                if not run:
                    return
                run["status"] = STATUS_FAILED
                run["error"] = str(error)
                run["updatedAt"] = utc_now()
                self._refresh_progress_unlocked(run, str(error), 100)
                self._write_summary_unlocked(run)

    def get_run_embedding_projection(self, run_id: str, method: str = "pca") -> dict[str, Any] | None:
        requested_method = normalize_projection_method(method)
        run = self.get_run(run_id)
        if not run:
            return None

        run_dir = Path(str(run.get("runDir") or self.runs_root / str(run_id)))
        cache_path = run_dir / f"embedding_projection_{requested_method}.json"
        if cache_path.exists():
            return read_json(cache_path)

        sampled_candidates_path = run_dir / SAMPLED_CANDIDATES_FILE
        sample_embeddings_path = run_dir / SAMPLE_EMBEDDINGS_FILE
        if not sampled_candidates_path.exists() or not sample_embeddings_path.exists():
            raise ValueError("La corrida no tiene artefactos de proyeccion; vuelve a ejecutarla.")

        import numpy as np

        sampled_candidates = read_json(sampled_candidates_path)
        embeddings = np.load(sample_embeddings_path)
        if embeddings.ndim != 2 or embeddings.shape[0] != len(sampled_candidates):
            raise ValueError("Los artefactos de proyeccion no coinciden con la muestra persistida.")

        projection = project_embeddings_2d(embeddings, requested_method)
        coordinates = projection["coordinates"]
        points = []
        for candidate, coordinate in zip(sampled_candidates, coordinates):
            points.append(
                {
                    "x": coordinate[0],
                    "y": coordinate[1],
                    "tweetId": candidate.get("tweetId") or "",
                    "text": candidate.get("text") or "",
                    "originalIndex": candidate.get("originalIndex"),
                    "wordCount": candidate.get("wordCount"),
                    "clusterIndex": candidate.get("clusterIndex"),
                    "clusterDisplayIndex": candidate.get("clusterDisplayIndex"),
                    "isMajorityCluster": bool(candidate.get("isMajorityCluster")),
                    "isSelected": bool(candidate.get("isSelected")),
                    "rank": candidate.get("rank"),
                    "score": candidate.get("score"),
                    "semanticDistance": candidate.get("semanticDistance"),
                    "lengthDistance": candidate.get("lengthDistance"),
                }
            )

        cost = run.get("embeddingCost") or {}
        config = run.get("config") or {}
        payload = {
            "runId": run_id,
            "method": projection["method"],
            "effectiveMethod": projection["effectiveMethod"],
            "embeddingModel": cost.get("embeddingModel") or config.get("embeddingModel") or "",
            "embeddingTexts": int(embeddings.shape[0]),
            "sampleCount": len(sampled_candidates),
            "selectedTweetId": run.get("selectedTweetId"),
            "clusterSizes": run.get("clusterSizes") or {},
            "warnings": projection.get("warnings") or [],
            "points": points,
        }
        write_json(cache_path, payload)
        return payload

    def _refresh_progress_unlocked(self, run: dict[str, Any], detail: str, percent: int) -> None:
        started_at = run.get("startedAtEpoch")
        elapsed = max(0.0, time.time() - started_at) if started_at else 0.0
        run["progress"] = {
            "percent": int(max(0, min(100, percent))),
            "detail": detail,
            "stage": run.get("status") or STATUS_RUNNING,
            "elapsedSeconds": elapsed,
            "elapsedLabel": self._format_duration(elapsed),
        }

    def _write_summary_unlocked(self, run: dict[str, Any]) -> None:
        write_json(Path(run["runDir"]) / "summary.json", self._public_run(run))

    def _public_run(self, run: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value
            for key, value in run.items()
            if key != "startedAtEpoch"
        }

    def _new_run_id(self) -> str:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        return f"{timestamp}-{uuid.uuid4().hex[:8]}"

    def _format_duration(self, seconds: float) -> str:
        if seconds < 1:
            return "0s"
        minutes, secs = divmod(int(seconds), 60)
        if minutes:
            return f"{minutes}m {secs:02d}s"
        return f"{secs}s"
