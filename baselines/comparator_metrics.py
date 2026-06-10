from __future__ import annotations

import math
from typing import Any


COSINE_DISTANCE_UPPER_BOUND = 2.0
COMPARABLE_OBJECTIVE_NAMES = [
    "fidelity_sbert_normalized",
    "semantic_diversity_normalized",
]


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def finite_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def is_dominated(row: dict[str, Any], rows: list[dict[str, Any]]) -> bool:
    vector = row.get("objectiveVector") or []
    if len(vector) < 2:
        return False
    for other in rows:
        if other is row:
            continue
        other_vector = other.get("objectiveVector") or []
        if len(other_vector) != len(vector):
            continue
        if all(a >= b for a, b in zip(other_vector, vector)) and any(
            a > b for a, b in zip(other_vector, vector)
        ):
            return True
    return False


def mark_non_dominated(rows: list[dict[str, Any]]) -> None:
    valid_rows = [row for row in rows if row.get("status") == "ok"]
    for row in rows:
        row["nonDominated"] = (
            row.get("status") == "ok"
            and len(row.get("objectiveVector") or []) >= 2
            and not is_dominated(row, valid_rows)
        )


def is_diagnostic_dominated(row: dict[str, Any], rows: list[dict[str, Any]]) -> bool:
    vector = row.get("diagnosticObjectiveVector") or []
    if len(vector) < 2:
        return False
    for other in rows:
        if other is row:
            continue
        other_vector = other.get("diagnosticObjectiveVector") or []
        if len(other_vector) != len(vector):
            continue
        if all(a >= b for a, b in zip(other_vector, vector)) and any(
            a > b for a, b in zip(other_vector, vector)
        ):
            return True
    return False


def mark_posthoc_non_dominated(rows: list[dict[str, Any]]) -> None:
    valid_rows = [row for row in rows if row.get("status") == "ok"]
    for row in rows:
        row["postHocNonDominated"] = (
            row.get("status") == "ok"
            and len(row.get("diagnosticObjectiveVector") or []) >= 2
            and not is_diagnostic_dominated(row, valid_rows)
        )


def normalized_objective_vector(vector: Any) -> list[float] | None:
    if not isinstance(vector, list) or len(vector) < 2:
        return None
    fidelity = finite_float(vector[0])
    diversity = finite_float(vector[1])
    return [
        clamp((fidelity + 1.0) / 2.0, 0.0, 1.0),
        clamp(diversity / COSINE_DISTANCE_UPPER_BOUND, 0.0, 1.0),
    ]


def comparable_objective_vector(row: dict[str, Any]) -> list[float] | None:
    vector = row.get("comparableObjectiveVector")
    if isinstance(vector, list) and len(vector) >= 2:
        return [clamp(finite_float(vector[0]), 0.0, 1.0), clamp(finite_float(vector[1]), 0.0, 1.0)]
    return normalized_objective_vector(row.get("diagnosticObjectiveVector") or row.get("objectiveVector") or [])


def comparable_point(row: dict[str, Any]) -> tuple[float, float] | None:
    vector = comparable_objective_vector(row)
    if not vector or len(vector) < 2:
        return None
    return vector[0], vector[1]


def pareto_points(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    unique = sorted(set(points), key=lambda item: (item[0], item[1]))
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
    return sorted(front, key=lambda item: item[0])


def calculate_hypervolume(points: list[tuple[float, float]]) -> float | None:
    front = pareto_points(points)
    if len(front) < 1:
        return None

    collapsed: dict[float, float] = {}
    for x_value, y_value in front:
        collapsed[x_value] = max(collapsed.get(x_value, 0.0), y_value)

    ordered = sorted(collapsed.items())
    hv = 0.0
    previous_x = 0.0
    for x_value, y_value in ordered:
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

    absolute_deviation = sum(abs(distance - mean_distance) for distance in distances)
    return absolute_deviation / (len(distances) * mean_distance)


def entropy_weights(matrix: list[list[float]]) -> list[float]:
    if not matrix:
        return [0.5, 0.5]
    width = min(len(row) for row in matrix)
    if width <= 0:
        return [0.5, 0.5]
    columns = [[finite_float(row[index]) for row in matrix] for index in range(width)]
    shifted_columns: list[list[float]] = []
    for column in columns:
        minimum = min(column)
        shift = max(0.0, -minimum) + 0.0001
        shifted_columns.append([value + shift for value in column])
    if len(matrix) <= 1:
        return [1.0 / width] * width
    divergences: list[float] = []
    for column in shifted_columns:
        total = sum(column) or 1.0
        entropy = 0.0
        for value in column:
            probability = value / total
            if probability > 0:
                entropy -= probability * math.log(probability)
        entropy /= math.log(len(matrix))
        divergences.append(max(0.0, 1.0 - entropy))
    total_divergence = sum(divergences)
    if total_divergence <= 0:
        return [1.0 / width] * width
    return [value / total_divergence for value in divergences]


def topsis_scores(matrix: list[list[float]], weights: list[float]) -> list[float]:
    if not matrix:
        return []
    width = min(len(row) for row in matrix)
    columns = [[finite_float(row[index]) for row in matrix] for index in range(width)]
    norms = [math.sqrt(sum(value * value for value in column)) or 1.0 for column in columns]
    weighted_rows = [
        [
            finite_float(row[index]) / norms[index] * weights[index]
            for index in range(width)
        ]
        for row in matrix
    ]
    ideals = [max(row[index] for row in weighted_rows) for index in range(width)]
    anti_ideals = [min(row[index] for row in weighted_rows) for index in range(width)]
    scores: list[float] = []
    for row in weighted_rows:
        positive = math.sqrt(sum((row[index] - ideals[index]) ** 2 for index in range(width)))
        negative = math.sqrt(sum((row[index] - anti_ideals[index]) ** 2 for index in range(width)))
        denominator = positive + negative
        scores.append(negative / denominator if denominator > 0 else 0.0)
    return scores


def canonical_generated_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip().lower()


def aggregate_series(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[int, dict[str, list[float]]] = {}
    sources: dict[int, str] = {}
    for result in results:
        for point in result.get("series") or []:
            generation = int(finite_float(point.get("generation"), -1))
            if generation < 0:
                continue
            bucket = buckets.setdefault(
                generation,
                {
                    "hypervolume": [],
                    "nonDominatedRows": [],
                    "spread": [],
                    "globalInertia": [],
                    "globalEntropy": [],
                },
            )
            sources[generation] = str(point.get("source") or sources.get(generation) or "aggregated")
            for key in bucket:
                value = point.get(key)
                if value is not None:
                    bucket[key].append(finite_float(value))
    series: list[dict[str, Any]] = []
    for generation in sorted(buckets):
        bucket = buckets[generation]
        series.append(
            {
                "generation": generation,
                "hypervolume": sum(bucket["hypervolume"]) / len(bucket["hypervolume"]) if bucket["hypervolume"] else None,
                "nonDominatedRows": sum(bucket["nonDominatedRows"]) / len(bucket["nonDominatedRows"]) if bucket["nonDominatedRows"] else None,
                "spread": sum(bucket["spread"]) / len(bucket["spread"]) if bucket["spread"] else None,
                "globalInertia": sum(bucket["globalInertia"]) / len(bucket["globalInertia"]) if bucket["globalInertia"] else None,
                "globalEntropy": sum(bucket["globalEntropy"]) / len(bucket["globalEntropy"]) if bucket["globalEntropy"] else None,
                "source": sources.get(generation, "aggregated"),
            }
        )
    return series


def chart_point_from_row(row: dict[str, Any], selected: bool = False) -> dict[str, Any] | None:
    comparable_vector = comparable_objective_vector(row)
    if not comparable_vector or len(comparable_vector) < 2:
        return None
    native_vector = row.get("diagnosticObjectiveVector") or row.get("objectiveVector") or []
    return {
        "x": comparable_vector[0],
        "y": comparable_vector[1],
        "instanceId": row.get("instanceId") or row.get("proposalId") or "",
        "proposalId": row.get("proposalId") or "",
        "displayName": row.get("displayName") or "",
        "baseDisplayName": row.get("baseDisplayName") or "",
        "label": row.get("generatedText") or "",
        "prompt": row.get("prompt") or "",
        "rank": row.get("selectionRank") if selected else row.get("rank"),
        "selected": selected or bool(row.get("selected")),
        "status": row.get("status") or "",
        "sourceIndex": row.get("sourceIndex"),
        "repetitionIndex": row.get("repetitionIndex"),
        "nativeObjectiveVector": native_vector,
        "nativeObjectiveLabel": row.get("diagnosticObjectiveLabel") or row.get("objectiveLabel") or "",
        "comparableObjectiveVector": comparable_vector,
        "comparableObjectiveLabel": row.get("comparableObjectiveLabel") or "",
        "coordinateSpace": "comparable_normalized",
    }


def row_is_chart_non_dominated(row: dict[str, Any]) -> bool:
    if row.get("status") != "ok":
        return False
    proposal_id = str(row.get("proposalId") or "").lower()
    if proposal_id == "evolmd":
        return bool(row.get("postHocNonDominated"))
    return bool(row.get("nonDominated"))


def build_charts_from_rows(rows: list[dict[str, Any]], selected_rows: list[dict[str, Any]], series: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "pareto": [point for row in rows for point in [chart_point_from_row(row)] if point is not None],
        "selected": [point for row in selected_rows for point in [chart_point_from_row(row, selected=True)] if point is not None],
        "nonDominated": [
            point
            for row in rows
            if row_is_chart_non_dominated(row)
            for point in [chart_point_from_row(row)]
            if point is not None
        ],
        "series": series,
    }
