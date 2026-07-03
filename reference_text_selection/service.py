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
    "referenceCount": 1,
    "sampleSize": 1000,
    "minClusterCount": 4,
    "minWords": 20,
    "maxWords": 80,
    "seed": 42,
    "semanticWeight": 0.7,
    "qualityWeight": 0.65,
    "mmrWeight": 0.70,
    "embeddingModel": "all-MiniLM-L6-v2",
}

SELECTED_REFERENCES_FILE = "selected_reference_texts.json"
CLUSTER_REPRESENTATIVES_FILE = "cluster_representatives.json"
SAMPLED_CANDIDATES_FILE = "sampled_candidates.json"
SAMPLE_EMBEDDINGS_FILE = "sample_embeddings.npy"
INCOMPATIBLE_RUN_MESSAGE = (
    "La corrida usa el formato singular anterior y es incompatible con la version actual; "
    "vuelve a ejecutarla para obtener artefactos de seleccion N."
)


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
        "referenceCount": int_between(
            payload.get("referenceCount", defaults["referenceCount"]),
            "referenceCount",
            1,
            10_000,
        ),
        "sampleSize": int_between(payload.get("sampleSize", defaults["sampleSize"]), "sampleSize", 1, 1_000_000),
        "minClusterCount": int_between(
            payload.get("minClusterCount", defaults["minClusterCount"]),
            "minClusterCount",
            1,
            10_000,
        ),
        "minWords": int_between(payload.get("minWords", defaults["minWords"]), "minWords", 0, 100_000),
        "maxWords": int_between(payload.get("maxWords", defaults["maxWords"]), "maxWords", 0, 100_000),
        "seed": int_between(payload.get("seed", defaults["seed"]), "seed", 0, 2_147_483_647),
        "semanticWeight": float_between(
            payload.get("semanticWeight", defaults["semanticWeight"]),
            "semanticWeight",
            0.0,
            1.0,
        ),
        "qualityWeight": float_between(
            payload.get("qualityWeight", defaults["qualityWeight"]),
            "qualityWeight",
            0.0,
            1.0,
        ),
        "mmrWeight": float_between(
            payload.get("mmrWeight", defaults["mmrWeight"]),
            "mmrWeight",
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


def cosine_distance(left: Any, right: Any) -> float:
    import numpy as np

    return float(1.0 - np.dot(left, right))


def effective_cluster_count(config: dict[str, Any], sample_count: int) -> int:
    return min(max(int(config["minClusterCount"]), 2 * int(config["referenceCount"])), sample_count)


def _cluster_size_map(labels: Any, cluster_count: int) -> dict[int, int]:
    import numpy as np

    return {index: int(np.sum(labels == index)) for index in range(cluster_count)}


def _representatives_default_sort_key(representative: dict[str, Any]) -> tuple[float, float, int, int]:
    rank = representative.get("selectionRank")
    rank_value = int(rank) if rank is not None else 1_000_000
    selected_group = 0 if rank is not None else 1
    return (
        selected_group,
        rank_value,
        -float(representative.get("globalRepresentativity") or 0.0),
        int(representative.get("originalIndex") or 0),
    )


def _selected_reference_payload(representative: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "tweetId",
        "text",
        "originalIndex",
        "wordCount",
        "clusterIndex",
        "clusterDisplayIndex",
        "clusterSize",
        "selectionRank",
        "scoreLocal",
        "semanticDistance",
        "lengthDistance",
        "normalizedSemanticDistance",
        "normalizedLengthDistance",
        "globalRepresentativity",
        "mmrScore",
        "minSemanticDistanceToSelected",
        "selectionReason",
    ]
    return {key: representative.get(key) for key in keys}


def _validate_counts(config: dict[str, Any], filtered_count: int, sample_count: int) -> None:
    reference_count = int(config["referenceCount"])
    if filtered_count < reference_count:
        raise ValueError("filteredCount debe ser mayor o igual a referenceCount.")
    if sample_count < reference_count:
        raise ValueError("sampleCount debe ser mayor o igual a referenceCount.")


def _select_representatives(
    representatives: list[dict[str, Any]],
    embeddings: Any,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    reference_count = int(config["referenceCount"])
    if len(representatives) < reference_count:
        raise ValueError("No hay representantes suficientes para seleccionar N_ref textos.")

    for representative in representatives:
        representative["isSelected"] = False
        representative["selectionRank"] = None
        representative["mmrScore"] = None
        representative["minSemanticDistanceToSelected"] = None
        representative["selectionReason"] = None

    first = min(
        representatives,
        key=lambda representative: (
            -float(representative["globalRepresentativity"]),
            -int(representative["clusterSize"]),
            int(representative["originalIndex"]),
        ),
    )
    first["isSelected"] = True
    first["selectionRank"] = 1
    first["mmrScore"] = None
    first["minSemanticDistanceToSelected"] = None
    first["selectionReason"] = "representatividad"
    selected = [first]

    while len(selected) < reference_count:
        candidates = [representative for representative in representatives if not representative["isSelected"]]
        scored_candidates: list[dict[str, Any]] = []
        for candidate in candidates:
            candidate_embedding = embeddings[int(candidate["sampleIndex"])]
            min_distance = min(
                cosine_distance(candidate_embedding, embeddings[int(selected_representative["sampleIndex"])])
                for selected_representative in selected
            )
            mmr_score = (
                float(config["mmrWeight"]) * float(candidate["globalRepresentativity"])
                + (1.0 - float(config["mmrWeight"])) * float(min_distance)
            )
            candidate["minSemanticDistanceToSelected"] = float(min_distance)
            candidate["mmrScore"] = float(mmr_score)
            scored_candidates.append(candidate)
        chosen = min(
            scored_candidates,
            key=lambda representative: (
                -float(representative["mmrScore"]),
                -float(representative["globalRepresentativity"]),
                -int(representative["clusterSize"]),
                int(representative["originalIndex"]),
            ),
        )
        chosen["isSelected"] = True
        chosen["selectionRank"] = len(selected) + 1
        chosen["selectionReason"] = "mmr"
        selected.append(chosen)

    return selected


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
    _validate_counts(config, len(filtered), sample_size)
    sample = random.Random(config["seed"]).sample(filtered, sample_size) if len(filtered) > sample_size else list(filtered)
    cluster_count = effective_cluster_count(config, len(sample))

    import numpy as np
    from sklearn.cluster import KMeans

    texts = [str(candidate["text"]) for candidate in sample]
    embeddings, cost = service.encode_texts(config["embeddingModel"], texts)
    embeddings = normalized_embeddings(embeddings)

    model = KMeans(
        n_clusters=cluster_count,
        random_state=config["seed"],
        n_init=20,
        max_iter=300,
    )
    labels = model.fit_predict(embeddings)
    centers = normalized_embeddings(model.cluster_centers_)
    cluster_sizes = _cluster_size_map(labels, cluster_count)
    max_cluster_size = max(cluster_sizes.values()) if cluster_sizes else 1

    per_sample_scores: dict[int, dict[str, Any]] = {}
    representatives: list[dict[str, Any]] = []
    for cluster_index in range(cluster_count):
        cluster_sample_indices = [int(index) for index in np.where(labels == cluster_index)[0]]
        if not cluster_sample_indices:
            continue
        cluster_candidates = [sample[index] for index in cluster_sample_indices]
        median_length = float(median([int(candidate["wordCount"]) for candidate in cluster_candidates]))
        semantic_distances = [
            cosine_distance(embeddings[sample_index], centers[cluster_index])
            for sample_index in cluster_sample_indices
        ]
        length_distances = [
            float(abs(int(candidate["wordCount"]) - median_length))
            for candidate in cluster_candidates
        ]
        normalized_semantic = min_max_normalize(semantic_distances)
        normalized_length = min_max_normalize(length_distances)
        cluster_ranked: list[dict[str, Any]] = []
        for sample_index, candidate, semantic_distance, length_distance, semantic_hat, length_hat in zip(
            cluster_sample_indices,
            cluster_candidates,
            semantic_distances,
            length_distances,
            normalized_semantic,
            normalized_length,
        ):
            score_local = (
                float(config["semanticWeight"]) * float(semantic_hat)
                + (1.0 - float(config["semanticWeight"])) * float(length_hat)
            )
            item = {
                "tweetId": candidate["tweetId"],
                "text": candidate["text"],
                "originalIndex": candidate["originalIndex"],
                "wordCount": candidate["wordCount"],
                "sampleIndex": sample_index,
                "clusterIndex": int(cluster_index),
                "clusterDisplayIndex": int(cluster_index) + 1,
                "clusterSize": int(cluster_sizes[cluster_index]),
                "semanticDistance": float(semantic_distance),
                "lengthDistance": float(length_distance),
                "normalizedSemanticDistance": float(semantic_hat),
                "normalizedLengthDistance": float(length_hat),
                "scoreLocal": float(score_local),
            }
            per_sample_scores[sample_index] = item
            cluster_ranked.append(item)
        cluster_ranked.sort(
            key=lambda candidate: (
                candidate["scoreLocal"],
                candidate["semanticDistance"],
                candidate["lengthDistance"],
                candidate["originalIndex"],
            )
        )
        representative = dict(cluster_ranked[0])
        representative["globalRepresentativity"] = (
            float(config["qualityWeight"]) * (1.0 - float(representative["scoreLocal"]))
            + (1.0 - float(config["qualityWeight"])) * (representative["clusterSize"] / max_cluster_size)
        )
        representative["isRepresentative"] = True
        representatives.append(representative)

    selected_representatives = _select_representatives(representatives, embeddings, config)
    selected_by_sample_index = {int(item["sampleIndex"]): item for item in selected_representatives}
    representative_by_sample_index = {int(item["sampleIndex"]): item for item in representatives}

    for representative in representatives:
        if not representative.get("isSelected"):
            representative["isSelected"] = False
            representative["selectionRank"] = None
            representative["selectionReason"] = None
            representative.setdefault("mmrScore", None)
            representative.setdefault("minSemanticDistanceToSelected", None)

    representatives.sort(key=_representatives_default_sort_key)
    selected_references = sorted(
        [_selected_reference_payload(representative) for representative in selected_representatives],
        key=lambda reference: int(reference.get("selectionRank") or 0),
    )

    sampled_candidates: list[dict[str, Any]] = []
    for sample_index, candidate in enumerate(sample):
        scored = per_sample_scores[sample_index]
        representative = representative_by_sample_index.get(sample_index)
        selected = selected_by_sample_index.get(sample_index)
        sampled_candidates.append(
            {
                "tweetId": candidate["tweetId"],
                "text": candidate["text"],
                "originalIndex": candidate["originalIndex"],
                "wordCount": candidate["wordCount"],
                "sampleIndex": sample_index,
                "clusterIndex": int(labels[sample_index]),
                "clusterDisplayIndex": int(labels[sample_index]) + 1,
                "clusterSize": int(cluster_sizes[int(labels[sample_index])]),
                "isRepresentative": representative is not None,
                "isSelected": selected is not None,
                "selectionRank": selected.get("selectionRank") if selected else None,
                "scoreLocal": scored["scoreLocal"],
                "semanticDistance": scored["semanticDistance"],
                "lengthDistance": scored["lengthDistance"],
                "normalizedSemanticDistance": scored["normalizedSemanticDistance"],
                "normalizedLengthDistance": scored["normalizedLengthDistance"],
                "globalRepresentativity": representative.get("globalRepresentativity") if representative else None,
                "mmrScore": selected.get("mmrScore") if selected else None,
                "minSemanticDistanceToSelected": (
                    selected.get("minSemanticDistanceToSelected") if selected else None
                ),
            }
        )

    cluster_sizes_payload = {
        str(index + 1): int(cluster_sizes.get(index, 0))
        for index in range(cluster_count)
    }

    return {
        "config": config,
        "datasetPath": str(dataset_path),
        "referenceCount": int(config["referenceCount"]),
        "selectedReferences": selected_references,
        "selectedCount": len(selected_references),
        "representatives": representatives,
        "representativeCount": len(representatives),
        "effectiveClusterCount": int(cluster_count),
        "filteredCount": len(filtered),
        "sampleCount": len(sample),
        "embeddingCost": cost,
        "clusterSizes": cluster_sizes_payload,
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
                    self._refresh_progress_unlocked(run, "Ejecutando seleccion N.", 50)
                return self._public_run(run)
        summary_path = self.runs_root / str(run_id) / "summary.json"
        if summary_path.exists():
            summary = read_json(summary_path)
            self._ensure_current_run_format(summary)
            return summary
        return None

    def _run_worker(self, run_id: str) -> None:
        with self._lock:
            run = self._runs[run_id]
            run["status"] = STATUS_RUNNING
            run["startedAtEpoch"] = time.time()
            run["updatedAt"] = utc_now()
            self._refresh_progress_unlocked(run, "Leyendo TSV, embebiendo muestra y ejecutando K-means.", 10)
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
            write_json(run_dir / SELECTED_REFERENCES_FILE, result["selectedReferences"])
            write_json(run_dir / CLUSTER_REPRESENTATIVES_FILE, result["representatives"])
            write_json(run_dir / SAMPLED_CANDIDATES_FILE, sampled_candidates)
            import numpy as np

            np.save(run_dir / SAMPLE_EMBEDDINGS_FILE, sample_embeddings)
            with self._lock:
                run.update(result)
                run["status"] = STATUS_COMPLETED
                run["error"] = None
                run["updatedAt"] = utc_now()
                self._refresh_progress_unlocked(run, "Textos de referencia seleccionados.", 100)
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
        self._ensure_current_run_format(run)

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
                    "clusterSize": candidate.get("clusterSize"),
                    "isRepresentative": bool(candidate.get("isRepresentative")),
                    "isSelected": bool(candidate.get("isSelected")),
                    "selectionRank": candidate.get("selectionRank"),
                    "scoreLocal": candidate.get("scoreLocal"),
                    "semanticDistance": candidate.get("semanticDistance"),
                    "lengthDistance": candidate.get("lengthDistance"),
                    "globalRepresentativity": candidate.get("globalRepresentativity"),
                    "mmrScore": candidate.get("mmrScore"),
                    "minSemanticDistanceToSelected": candidate.get("minSemanticDistanceToSelected"),
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
            "selectedCount": run.get("selectedCount"),
            "selectedReferences": run.get("selectedReferences") or [],
            "clusterSizes": run.get("clusterSizes") or {},
            "effectiveClusterCount": run.get("effectiveClusterCount"),
            "warnings": projection.get("warnings") or [],
            "points": points,
        }
        write_json(cache_path, payload)
        return payload

    def _ensure_current_run_format(self, run: dict[str, Any]) -> None:
        if "selectedText" in run or "rankedCandidates" in run:
            raise ValueError(INCOMPATIBLE_RUN_MESSAGE)
        if run.get("status") == STATUS_COMPLETED and "selectedReferences" not in run:
            raise ValueError(INCOMPATIBLE_RUN_MESSAGE)

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
        minutes, remaining = divmod(int(seconds), 60)
        if minutes:
            return f"{minutes}m {remaining}s"
        return f"{remaining}s"
