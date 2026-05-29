from __future__ import annotations

import json
import math
import random
import re
import string
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from sbert_service import SbertSimilarityService, shared_sbert_service


TOKEN_RE = re.compile(r"\b[\w'-]+\b", re.UNICODE)
PROMPT_MARKER_RE = re.compile(
    r"```|\*\*|<\s*/?\s*[A-Za-z][^>]*>|^\s*(?:system|user|assistant)\s*:|"
    r"\b(?:based on|here'?s|breakdown|possible components?|current component value|potential next values?)\b",
    re.IGNORECASE,
)
WORDNET_POS = {
    "NOUN": "n",
    "PROPN": "n",
    "VERB": "v",
    "ADJ": "a",
    "ADV": "r",
}
MODIFIABLE_POS = {"NOUN", "PROPN", "VERB", "ADJ", "ADV"}
PREFERRED_POS_BY_COMPONENT = {
    "role": {"NOUN", "PROPN", "ADJ"},
    "topic": {"NOUN", "PROPN", "ADJ"},
    "action": {"VERB", "NOUN", "PROPN"},
}


@dataclass(frozen=True)
class MutableUnit:
    text: str
    lemma: str
    pos: str
    start: int
    end: int
    kind: str


@dataclass(frozen=True)
class CandidateRecord:
    candidate: str
    source: str
    unit: MutableUnit | None = None


@dataclass(frozen=True)
class ScoredCandidate:
    candidate: str
    source: str
    similarity: float
    diversity: float
    reason: str
    is_valid: bool


@dataclass(frozen=True)
class OperatorResult:
    output: str
    selected: ScoredCandidate | None
    candidates: list[ScoredCandidate]
    raw_candidate_count: int
    coverage: bool
    elapsed_seconds: float
    cost: dict[str, Any]
    status: str
    diagnostics: dict[str, Any] | None = None


class EmbeddingRanker:
    def __init__(self, model_name: str, service: SbertSimilarityService | None = None) -> None:
        self.model_name = model_name
        self.service = service or shared_sbert_service()

    def similarities(self, reference: str, candidates: list[str]) -> tuple[list[float], dict[str, Any]]:
        return self.service.similarities(reference, candidates, self.model_name)


class LinguisticAnalyzer:
    def __init__(self, model_name: str = "en_core_web_sm") -> None:
        self.model_name = model_name
        self._nlp: Any = None
        self._load_lock = threading.Lock()

    def _load_model(self) -> Any:
        if self._nlp is None:
            with self._load_lock:
                if self._nlp is None:
                    import spacy

                    self._nlp = spacy.load(self.model_name)
        return self._nlp

    def extract_units(self, text: str, include_phrases: bool = True) -> list[MutableUnit]:
        doc = self._load_model()(text)
        units: list[MutableUnit] = []
        seen: set[tuple[int, int, str]] = set()

        for token in doc:
            if token.is_space or token.is_punct or token.is_stop:
                continue
            if token.pos_ not in MODIFIABLE_POS:
                continue
            if not TOKEN_RE.search(token.text):
                continue
            key = (token.idx, token.idx + len(token.text), "word")
            seen.add(key)
            units.append(
                MutableUnit(
                    text=token.text,
                    lemma=(token.lemma_ or token.text).lower(),
                    pos=token.pos_,
                    start=token.idx,
                    end=token.idx + len(token.text),
                    kind="word",
                )
            )

        if include_phrases:
            for chunk in doc.noun_chunks:
                words = [token for token in chunk if not token.is_stop and not token.is_punct]
                if len(words) < 2 or len(words) > 4:
                    continue
                start = chunk.start_char
                end = chunk.end_char
                key = (start, end, "phrase")
                if key in seen:
                    continue
                seen.add(key)
                units.append(
                    MutableUnit(
                        text=chunk.text,
                        lemma=normalize_text(chunk.text),
                        pos="NOUN",
                        start=start,
                        end=end,
                        kind="phrase",
                    )
                )

        return sorted(units, key=lambda unit: (unit.start, unit.end, unit.kind))


class PPDBIndex:
    def __init__(self, index_path: Path) -> None:
        self.index_path = index_path
        self.available = False
        self.entries: dict[str, list[str]] = {}
        self._load()

    def _load(self) -> None:
        if not self.index_path.exists():
            return
        payload = json.loads(self.index_path.read_text(encoding="utf-8"))
        entries = payload.get("entries") if isinstance(payload, dict) else payload
        if not isinstance(entries, dict):
            raise ValueError(f"PPDB index must be a JSON object: {self.index_path}")
        normalized: dict[str, list[str]] = {}
        for key, values in entries.items():
            if not isinstance(values, list):
                continue
            clean_values = [str(value).strip() for value in values if str(value).strip()]
            if clean_values:
                normalized[normalize_text(str(key))] = sorted(set(clean_values), key=str.lower)
        self.entries = normalized
        self.available = True

    def lookup(self, text: str) -> list[str]:
        return list(self.entries.get(normalize_text(text), []))


class WordNetProvider:
    def __init__(self) -> None:
        self._wn: Any = None
        self._load_lock = threading.Lock()

    def _load(self) -> Any:
        if self._wn is None:
            with self._load_lock:
                if self._wn is None:
                    try:
                        from nltk.corpus import wordnet as wn

                        wn.ensure_loaded()
                    except LookupError:
                        import nltk

                        nltk.download("wordnet", quiet=True)
                        from nltk.corpus import wordnet as wn

                        wn.ensure_loaded()
                    self._wn = wn
        return self._wn

    def synset_replacements(self, lemma: str, pos: str | None = None) -> list[list[str]]:
        wn = self._load()
        wn_pos = WORDNET_POS.get(pos or "")
        synsets = wn.synsets(lemma, pos=wn_pos) if wn_pos else wn.synsets(lemma)
        normalized_original = normalize_text(lemma)
        groups: list[list[str]] = []
        for synset in synsets:
            replacements: list[str] = []
            for lemma_item in synset.lemmas():
                value = lemma_item.name().replace("_", " ").strip()
                if not value:
                    continue
                if normalize_text(value) == normalized_original:
                    continue
                replacements.append(value)
            clean = dedupe(replacements)
            if clean:
                groups.append(clean)
        return groups

    def lookup(self, lemma: str, pos: str | None = None) -> list[str]:
        replacements = [
            replacement
            for group in self.synset_replacements(lemma, pos)
            for replacement in group
        ]
        return sorted(set(replacements), key=str.lower)


class DistilBertProvider:
    def __init__(self, model_name: str, pipeline_factory: Callable[..., Any] | None = None) -> None:
        self.model_name = model_name
        self.pipeline_factory = pipeline_factory
        self._pipeline: Any = None
        self._load_lock = threading.Lock()
        self._predict_lock = threading.Lock()

    def _load(self) -> Any:
        if self._pipeline is None:
            with self._load_lock:
                if self._pipeline is None:
                    if self.pipeline_factory is None:
                        from transformers import pipeline

                        self.pipeline_factory = pipeline
                    self._pipeline = self.pipeline_factory("fill-mask", model=self.model_name)
        return self._pipeline

    def is_single_token(self, text: str) -> bool:
        pipe = self._load()
        tokenizer = getattr(pipe, "tokenizer", None)
        if tokenizer is None:
            return is_simple_distilbert_token(text)
        encoded = tokenizer.encode(text, add_special_tokens=False)
        return len(encoded) == 1

    def predict(self, masked_text: str, top_k: int) -> list[str]:
        pipe = self._load()
        with self._predict_lock:
            payload = pipe(masked_text, top_k=top_k)
        if isinstance(payload, dict):
            payload = [payload]
        values: list[str] = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            token = str(item.get("token_str") or "").strip()
            if is_simple_distilbert_token(token):
                values.append(token.lower())
        return dedupe(values)


class TurbulenceOperator:
    strategy_id = "base"
    display_name = "Base"

    def apply(self, current: str, config: dict[str, Any]) -> OperatorResult:
        raise NotImplementedError

    def score_candidates(
        self,
        current: str,
        candidates: list[CandidateRecord],
        config: dict[str, Any],
        ranker: EmbeddingRanker,
    ) -> tuple[list[ScoredCandidate], dict[str, Any]]:
        unique: list[CandidateRecord] = []
        seen: set[str] = set()
        for record in candidates:
            key = normalize_text(record.candidate)
            if key in seen:
                continue
            seen.add(key)
            unique.append(record)

        similarities, embedding_cost = ranker.similarities(current, [record.candidate for record in unique])
        scored: list[ScoredCandidate] = []
        for record, similarity in zip(unique, similarities):
            base_reason = candidate_validation_reason(
                record.candidate,
                current,
                min_words=int(config["minWords"]),
                max_words=int(config["maxWords"]),
            )
            if base_reason == "valid" and similarity < float(config["turbulenceMinSimilarity"]):
                reason = "too_distant_from_current"
            elif base_reason == "valid" and similarity > float(config["turbulenceMaxSimilarity"]):
                reason = "too_close_to_current"
            else:
                reason = base_reason
            scored.append(
                ScoredCandidate(
                    candidate=record.candidate,
                    source=record.source,
                    similarity=similarity,
                    diversity=1.0 - similarity,
                    reason=reason,
                    is_valid=reason == "valid",
                )
            )

        return scored, embedding_cost

    def select_candidate(self, scored: list[ScoredCandidate]) -> ScoredCandidate | None:
        valid = [candidate for candidate in scored if candidate.is_valid]
        if not valid:
            return None
        return max(valid, key=lambda candidate: (candidate.similarity, candidate.diversity, candidate.candidate))


class WordNetPPDBOperator(TurbulenceOperator):
    strategy_id = "wordnet-ppdb-sbert"
    display_name = "WordNet+PPDB+SBERT"

    def __init__(
        self,
        analyzer: LinguisticAnalyzer,
        ranker: EmbeddingRanker,
        ppdb: PPDBIndex | None,
        wordnet: WordNetProvider | None = None,
    ) -> None:
        self.analyzer = analyzer
        self.ranker = ranker
        self.ppdb = ppdb
        self.wordnet = wordnet or WordNetProvider()

    def apply(self, current: str, config: dict[str, Any]) -> OperatorResult:
        started = time.perf_counter()
        k_candidates = int(config["kCandidates"])
        unit_seed = int(config.get("unitSelectionSeed", config.get("selectionSeed", 0)))
        cost = {
            "wordnetQueries": 0,
            "ppdbQueries": 0,
            "ppdbLookupAttempts": 0,
            "ppdbAvailable": bool(self.ppdb and self.ppdb.available and config.get("usePpdb", True)),
            "distilbertInferences": 0,
            "llmCalls": 0,
        }
        general_units = [unit for unit in self.analyzer.extract_units(current, include_phrases=False) if unit.kind == "word"]
        eligible_units = component_preferred_units(general_units, str(config.get("componentName") or ""))
        diagnostics: dict[str, Any] = {
            "local": {
                "strategy": self.strategy_id,
                "componentName": config.get("componentName"),
                "unitSelectionSeed": unit_seed,
                "unitsGeneral": units_to_diagnostics(general_units),
                "unitsEligible": units_to_diagnostics(eligible_units),
            }
        }
        if not eligible_units:
            return local_failure_result(current, started, cost, "no_units", diagnostics)

        unit = select_uniform_unit(eligible_units, unit_seed)
        diagnostics["local"]["selectedUnit"] = unit_to_diagnostics(unit)
        use_ppdb = bool(config.get("usePpdb", True)) and self.ppdb is not None

        def ppdb_lookup(text: str) -> list[str]:
            if not use_ppdb:
                return []
            cost["ppdbLookupAttempts"] += 1
            if self.ppdb and self.ppdb.available:
                cost["ppdbQueries"] += 1
            return self.ppdb.lookup(text) if self.ppdb else []

        cost["wordnetQueries"] += 1
        synset_groups = wordnet_synset_groups(self.wordnet, unit.lemma or unit.text, unit.pos)
        wordnet_replacements = balanced_lexical_replacements(synset_groups, k_candidates, unit_seed, unit.text)
        replacements: list[tuple[str, str]] = [(value, "wordnet") for value in wordnet_replacements]
        diagnostics["local"]["wordnet"] = {
            "query": unit.lemma or unit.text,
            "pos": unit.pos,
            "synsetGroupCount": len(synset_groups),
            "replacements": wordnet_replacements,
        }

        ppdb_lookup_keys: list[str] = []
        ppdb_replacements: list[str] = []
        if len(replacements) < k_candidates and use_ppdb:
            ppdb_lookup_keys = dedupe([unit.text, unit.lemma])
            ppdb_values = [
                value
                for lookup_key in ppdb_lookup_keys
                for value in ppdb_lookup(lookup_key)
            ]
            ppdb_replacements = ordered_lexical_replacements(
                ppdb_values,
                k_candidates - len(replacements),
                unit_seed,
                unit.text,
                used={value for value, _ in replacements},
            )
            replacements.extend((value, "ppdb") for value in ppdb_replacements)
        diagnostics["local"]["ppdb"] = {
            "enabled": bool(config.get("usePpdb", True)),
            "available": bool(self.ppdb and self.ppdb.available),
            "lookupKeys": ppdb_lookup_keys,
            "replacements": ppdb_replacements,
        }

        candidates = build_candidate_records(current, unit, replacements, config)
        diagnostics["local"]["candidateStage"] = {
            "replacementCount": len(replacements),
            "candidateCount": len(candidates),
        }

        scored, embedding_cost = self.score_candidates(current, candidates, config, self.ranker)
        cost.update(sum_costs(cost, embedding_cost))
        selected = self.select_candidate(scored)
        elapsed = time.perf_counter() - started
        return OperatorResult(
            output=selected.candidate if selected else current,
            selected=selected,
            candidates=scored,
            raw_candidate_count=len(candidates),
            coverage=bool(candidates),
            elapsed_seconds=elapsed,
            cost=cost,
            status=operator_status(candidates, scored, selected),
            diagnostics=diagnostics,
        )


class DistilBertOperator(TurbulenceOperator):
    strategy_id = "distilbert-sbert"
    display_name = "DistilBERT+SBERT"

    def __init__(
        self,
        analyzer: LinguisticAnalyzer,
        ranker: EmbeddingRanker,
        provider: DistilBertProvider,
    ) -> None:
        self.analyzer = analyzer
        self.ranker = ranker
        self.provider = provider

    def apply(self, current: str, config: dict[str, Any]) -> OperatorResult:
        started = time.perf_counter()
        k_candidates = int(config["kCandidates"])
        unit_seed = int(config.get("unitSelectionSeed", config.get("selectionSeed", 0)))
        cost = {
            "wordnetQueries": 0,
            "ppdbQueries": 0,
            "ppdbLookupAttempts": 0,
            "ppdbAvailable": False,
            "distilbertInferences": 0,
            "llmCalls": 0,
        }
        general_units = [unit for unit in self.analyzer.extract_units(current, include_phrases=False) if unit.kind == "word"]
        eligible_units = component_preferred_units(general_units, str(config.get("componentName") or ""))
        compatible_units = [unit for unit in eligible_units if self.provider.is_single_token(unit.text)]
        diagnostics: dict[str, Any] = {
            "local": {
                "strategy": self.strategy_id,
                "componentName": config.get("componentName"),
                "unitSelectionSeed": unit_seed,
                "unitsGeneral": units_to_diagnostics(general_units),
                "unitsEligible": units_to_diagnostics(eligible_units),
                "unitsCompatible": units_to_diagnostics(compatible_units),
            }
        }
        if not general_units:
            return local_failure_result(current, started, cost, "no_units", diagnostics)
        if not compatible_units:
            return local_failure_result(current, started, cost, "no_compatible_units", diagnostics)

        unit = select_uniform_unit(compatible_units, unit_seed)
        masked = replace_span(current, unit, "[MASK]")
        top_k = 3 * k_candidates
        cost["distilbertInferences"] += 1
        predictions = self.provider.predict(masked, top_k)
        replacements = ordered_lexical_replacements(predictions, k_candidates, unit_seed, unit.text, preserve_order=True)
        candidates = build_candidate_records(
            current,
            unit,
            [(replacement, "distilbert") for replacement in replacements],
            config,
        )
        diagnostics["local"].update(
            {
                "selectedUnit": unit_to_diagnostics(unit),
                "distilbert": {
                    "maskedText": masked,
                    "topK": top_k,
                    "predictions": predictions,
                    "replacements": replacements,
                },
                "candidateStage": {
                    "replacementCount": len(replacements),
                    "candidateCount": len(candidates),
                },
            }
        )

        scored, embedding_cost = self.score_candidates(current, candidates, config, self.ranker)
        cost.update(sum_costs(cost, embedding_cost))
        selected = self.select_candidate(scored)
        elapsed = time.perf_counter() - started
        return OperatorResult(
            output=selected.candidate if selected else current,
            selected=selected,
            candidates=scored,
            raw_candidate_count=len(candidates),
            coverage=bool(candidates),
            elapsed_seconds=elapsed,
            cost=cost,
            status=operator_status(candidates, scored, selected),
            diagnostics=diagnostics,
        )


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def word_count(value: str) -> int:
    return len(TOKEN_RE.findall(value))


def dedupe(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = re.sub(r"\s+", " ", value.strip())
        key = normalize_text(text)
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(text)
    return result


def replace_span(text: str, unit: MutableUnit, replacement: str) -> str:
    return re.sub(r"\s+", " ", f"{text[:unit.start]}{replacement}{text[unit.end:]}".strip())


def component_preferred_units(units: list[MutableUnit], component_name: str) -> list[MutableUnit]:
    general = [unit for unit in units if unit.kind == "word" and unit.pos in MODIFIABLE_POS]
    preferred_pos = PREFERRED_POS_BY_COMPONENT.get(component_name)
    if not preferred_pos:
        return general
    preferred = [unit for unit in general if unit.pos in preferred_pos]
    return preferred or general


def select_uniform_unit(units: list[MutableUnit], seed: int) -> MutableUnit:
    return random.Random(seed).choice(units)


def unit_to_diagnostics(unit: MutableUnit | None) -> dict[str, Any] | None:
    if unit is None:
        return None
    return {
        "text": unit.text,
        "lemma": unit.lemma,
        "pos": unit.pos,
        "start": unit.start,
        "end": unit.end,
        "kind": unit.kind,
    }


def units_to_diagnostics(units: list[MutableUnit]) -> list[dict[str, Any]]:
    return [payload for unit in units if (payload := unit_to_diagnostics(unit)) is not None]


def wordnet_synset_groups(provider: Any, lemma: str, pos: str | None) -> list[list[str]]:
    if hasattr(provider, "synset_replacements"):
        return provider.synset_replacements(lemma, pos)
    if hasattr(provider, "lookup"):
        values = provider.lookup(lemma, pos)
        return [values] if values else []
    return []


def is_basic_lexical_replacement(value: str, original: str, used_keys: set[str] | None = None) -> bool:
    used_keys = used_keys or set()
    text = re.sub(r"\s+", " ", value.strip())
    if not text:
        return False
    if PROMPT_MARKER_RE.search(text):
        return False
    if normalize_text(text) == normalize_text(original):
        return False
    if normalize_text(text) in used_keys:
        return False
    if word_count(text) != 1:
        return False
    return is_simple_distilbert_token(text)


def ordered_lexical_replacements(
    values: Iterable[str],
    limit: int,
    seed: int,
    original: str,
    used: set[str] | None = None,
    preserve_order: bool = False,
) -> list[str]:
    used_keys = {normalize_text(item) for item in (used or set())}
    replacements: list[str] = []
    for value in values:
        text = re.sub(r"\s+", " ", str(value).strip())
        if not is_basic_lexical_replacement(text, original, used_keys):
            continue
        used_keys.add(normalize_text(text))
        replacements.append(text)
    if not preserve_order:
        random.Random(seed).shuffle(replacements)
    return replacements[:limit]


def balanced_lexical_replacements(
    synset_groups: list[list[str]],
    limit: int,
    seed: int,
    original: str,
) -> list[str]:
    rng = random.Random(seed)
    groups = [
        ordered_lexical_replacements(group, len(group), seed + index + 1, original)
        for index, group in enumerate(synset_groups)
    ]
    groups = [group for group in groups if group]
    rng.shuffle(groups)

    replacements: list[str] = []
    seen: set[str] = set()
    depth = 0
    while len(replacements) < limit:
        progressed = False
        for group in groups:
            if depth >= len(group):
                continue
            value = group[depth]
            key = normalize_text(value)
            if key in seen:
                continue
            seen.add(key)
            replacements.append(value)
            progressed = True
            if len(replacements) >= limit:
                break
        if not progressed:
            break
        depth += 1
    return replacements


def build_candidate_records(
    current: str,
    unit: MutableUnit,
    replacements: list[tuple[str, str]],
    config: dict[str, Any],
) -> list[CandidateRecord]:
    candidates: list[CandidateRecord] = []
    seen: set[str] = set()
    for replacement, source in replacements:
        candidate = replace_span(current, unit, replacement)
        key = normalize_text(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate_validation_reason(candidate, current, int(config["minWords"]), int(config["maxWords"])) != "valid":
            continue
        candidates.append(CandidateRecord(candidate=candidate, source=source, unit=unit))
        if len(candidates) >= int(config["kCandidates"]):
            break
    return candidates


def operator_status(
    candidates: list[CandidateRecord],
    scored: list[ScoredCandidate],
    selected: ScoredCandidate | None,
) -> str:
    if selected:
        return "ok"
    if not candidates:
        return "no_candidates"
    if scored and all(candidate.reason in {"too_distant_from_current", "too_close_to_current"} for candidate in scored):
        return "semantic_range_failed"
    return "no_valid_candidate"


def local_failure_result(
    current: str,
    started: float,
    cost: dict[str, Any],
    status: str,
    diagnostics: dict[str, Any],
) -> OperatorResult:
    return OperatorResult(
        output=current,
        selected=None,
        candidates=[],
        raw_candidate_count=0,
        coverage=False,
        elapsed_seconds=time.perf_counter() - started,
        cost=cost,
        status=status,
        diagnostics=diagnostics,
    )


def candidate_validation_reason(candidate: str, current: str, min_words: int, max_words: int) -> str:
    cleaned = re.sub(r"\s+", " ", candidate.strip())
    if not cleaned:
        return "empty"
    if "\n" in candidate or "\r" in candidate:
        return "line_break"
    if PROMPT_MARKER_RE.search(cleaned):
        return "prompt_marker"
    if normalize_text(cleaned) == normalize_text(current):
        return "literal_copy_current"
    words = word_count(cleaned)
    if words < min_words:
        return "too_short"
    if words > max_words:
        return "too_long"
    original_words = word_count(current)
    if original_words and abs(words - original_words) > max(2, math.ceil(original_words * 0.5)):
        return "length_drift"
    return "valid"


def is_simple_distilbert_token(token: str) -> bool:
    text = token.strip()
    if not text or text.startswith("##"):
        return False
    if text in {"[MASK]", "[SEP]", "[CLS]", "[PAD]", "[UNK]"}:
        return False
    if any(char in string.punctuation.replace("-", "") for char in text):
        return False
    if any(char.isdigit() for char in text):
        return False
    return bool(re.fullmatch(r"[A-Za-z][A-Za-z-]*", text))


def sum_costs(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in extra.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            result[key] = float(result.get(key, 0.0) or 0.0) + float(value)
        else:
            result[key] = value
    return result
