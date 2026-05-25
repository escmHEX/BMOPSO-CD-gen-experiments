from __future__ import annotations

import math
import time
from typing import Any


def finite_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def format_duration(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds) or seconds < 0:
        return "No disponible"
    rounded = int(round(seconds))
    minutes, secs = divmod(rounded, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def objective_label(vector: list[float]) -> str:
    if not vector:
        return "--"
    return "[" + ", ".join(f"{value:.6f}" for value in vector) + "]"


def mark_non_dominated(rows: list[dict[str, Any]], vector_key: str, output_key: str) -> None:
    for row in rows:
        vector = row.get(vector_key) or []
        if len(vector) < 2:
            row[output_key] = False
            continue
        row[output_key] = not any(
            other is not row
            and len(other.get(vector_key) or []) == len(vector)
            and all(a >= b for a, b in zip(other[vector_key], vector))
            and any(a > b for a, b in zip(other[vector_key], vector))
            for other in rows
        )


def pareto_points(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    unique = sorted(set(points), key=lambda point: (point[0], point[1]))
    front: list[tuple[float, float]] = []
    for point in unique:
        if not any(
            other != point
            and other[0] >= point[0]
            and other[1] >= point[1]
            and (other[0] > point[0] or other[1] > point[1])
            for other in unique
        ):
            front.append(point)
    return sorted(front, key=lambda point: point[0])


def calculate_hypervolume(points: list[tuple[float, float]]) -> float | None:
    front = pareto_points(points)
    if not front:
        return None
    collapsed: dict[float, float] = {}
    for x_value, y_value in front:
        collapsed[x_value] = max(collapsed.get(x_value, 0.0), y_value)
    hv = 0.0
    previous_x = 0.0
    for x_value, y_value in sorted(collapsed.items()):
        if x_value > previous_x:
            hv += (x_value - previous_x) * y_value
            previous_x = x_value
    return clamp(hv, 0.0, 1.0)


def calculate_spread(points: list[tuple[float, float]]) -> float | None:
    front = pareto_points(points)
    if len(front) < 3:
        return None
    distances = [
        math.dist(front[index - 1], front[index])
        for index in range(1, len(front))
    ]
    mean_distance = sum(distances) / len(distances)
    if mean_distance <= 0:
        return 0.0
    return sum(abs(distance - mean_distance) for distance in distances) / (len(distances) * mean_distance)


def normalized_common_point(row: dict[str, Any]) -> tuple[float, float] | None:
    vector = row.get("commonObjectiveVector") or []
    if len(vector) < 2:
        return None
    fidelity = clamp((finite_float(vector[0]) + 1.0) / 2.0, 0.0, 1.0)
    diversity = clamp(finite_float(vector[1]), 0.0, 1.0)
    return fidelity, diversity


def attach_common_metrics(
    rows: list[dict[str, Any]],
    reference_text: str,
    embedding_model: str,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    started = time.perf_counter()
    completed = [row for row in rows if str(row.get("generatedText") or "").strip()]
    if not completed:
        metrics = {
            "commonObjectiveNames": ["semantic_fidelity", "semantic_diversity"],
            "completedRows": 0,
            "nonDominatedRows": 0,
            "bestObjectiveVector": [],
            "bestObjectiveLabel": "--",
            "hypervolume": None,
            "hypervolumeLabel": "No aplica",
            "spread": None,
            "spreadLabel": "No aplica",
            "moConvention": "Maximization; common metrics unavailable because no generated texts were available.",
        }
        return rows, metrics, {
            "embeddingModel": embedding_model,
            "embeddingTexts": 0,
            "embeddingWallClockSeconds": 0.0,
            "embeddingWallClockLabel": "0s",
        }

    import numpy as np
    from sbert_service import shared_sbert_service

    texts = [reference_text, *[str(row.get("generatedText") or "") for row in completed]]
    embeddings, embedding_cost = shared_sbert_service().encode_texts(embedding_model, texts)
    reference_embedding = embeddings[0]
    generated_embeddings = embeddings[1:]
    fidelities = np.matmul(generated_embeddings, reference_embedding)
    similarity = np.matmul(generated_embeddings, generated_embeddings.T)

    if len(completed) <= 1:
        diversities = [0.0]
    else:
        diversities = []
        for index in range(len(completed)):
            average_similarity = (float(similarity[index].sum()) - 1.0) / (len(completed) - 1)
            diversities.append(1.0 - average_similarity)

    for row, fidelity, diversity in zip(completed, fidelities, diversities):
        vector = [float(fidelity), float(diversity)]
        row["commonFidelity"] = float(fidelity)
        row["commonDiversity"] = float(diversity)
        row["commonObjectiveVector"] = vector
        row["commonObjectiveLabel"] = objective_label(vector)
        row["commonObjectiveNames"] = ["semantic_fidelity", "semantic_diversity"]

    mark_non_dominated(completed, "commonObjectiveVector", "commonNonDominated")
    for row in rows:
        row.setdefault("commonNonDominated", False)

    normalized_points = [
        point
        for row in completed
        if row.get("commonNonDominated")
        for point in [normalized_common_point(row)]
        if point is not None
    ]
    hypervolume = calculate_hypervolume(normalized_points)
    spread = calculate_spread(normalized_points)
    best_row = max(completed, key=lambda row: finite_float(row.get("commonFidelity"), -2.0), default=None)
    elapsed = time.perf_counter() - started

    metrics = {
        "commonObjectiveNames": ["semantic_fidelity", "semantic_diversity"],
        "completedRows": len(completed),
        "nonDominatedRows": sum(1 for row in completed if row.get("commonNonDominated")),
        "bestObjectiveVector": best_row.get("commonObjectiveVector") if best_row else [],
        "bestObjectiveLabel": best_row.get("commonObjectiveLabel") if best_row else "--",
        "bestFidelity": best_row.get("commonFidelity") if best_row else None,
        "bestDiversity": max((finite_float(row.get("commonDiversity")) for row in completed), default=None),
        "hypervolume": hypervolume,
        "hypervolumeLabel": f"{hypervolume:.6f}" if hypervolume is not None else "No aplica",
        "spread": spread,
        "spreadLabel": f"{spread:.6f}" if spread is not None else "No aplica",
        "moConvention": (
            "Common post-hoc maximization metrics over final generated texts. "
            "Fidelity is cosine similarity against the reference text. "
            "Diversity is 1 - average cosine similarity against the remaining final population. "
            "HV normalizes fidelity with (f + 1) / 2 and clamps diversity to [0, 1] only for the reference [0, 0] calculation."
        ),
    }
    cost = {
        "embeddingModel": embedding_cost["embeddingModel"],
        "sourceModel": embedding_cost["sourceModel"],
        "embeddingTexts": len(texts),
        "embeddingWallClockSeconds": elapsed,
        "embeddingWallClockLabel": format_duration(elapsed),
    }
    return rows, metrics, cost
