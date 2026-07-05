from __future__ import annotations

import copy
import json
import math
import os
import queue
import re
import csv
import ctypes
import ast
import shlex
import shutil
import signal
import statistics
import subprocess
import sys
import threading
import time
import uuid
import zipfile
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, as_completed, wait
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from baselines.comparator_metrics import aggregate_series
from baselines.comparator_metrics import build_charts_from_rows
from baselines.comparator_metrics import calculate_contribution
from baselines.comparator_metrics import calculate_extent
from baselines.comparator_metrics import calculate_hypervolume
from baselines.comparator_metrics import calculate_unary_entropy
from baselines.comparator_metrics import canonical_generated_text
from baselines.comparator_metrics import COMPARABLE_OBJECTIVE_NAMES
from baselines.comparator_metrics import comparable_point
from baselines.comparator_metrics import entropy_weights
from baselines.comparator_metrics import mark_non_dominated
from baselines.comparator_metrics import normalized_objective_vector
from baselines.comparator_metrics import topsis_scores
from sbert_service import normalize_projection_method
from sbert_service import project_embeddings_2d
from sbert_service import shared_sbert_service


STATUS_QUEUED = "queued"
STATUS_RUNNING = "running"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"
STATUS_CANCELLED = "cancelled"


class ComparatorRunConflictError(RuntimeError):
    pass


ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
STAGE_RE = re.compile(r"^\s*(\d+)\s*/\s*(\d+)\s+(.+)$")
GENERATION_RE = re.compile(r"(?:Generaci[oó]n|generation)\s+(\d+)\s*/\s*(\d+)", re.IGNORECASE)
GENERATION_STARTED_RE = re.compile(r"\bgeneration\s+\d+\s*/\s*\d+\s+started\b", re.IGNORECASE)
GENERATION_TIME_RE = re.compile(
    r"(?:Tiempo\s+Gen:|Generation\s+\d+\s+complete\.\s+Time:)\s*([0-9]+(?:[\.,][0-9]+)?)s",
    re.IGNORECASE,
)
ELAPSED_RE = re.compile(r"\belapsed=(\d{1,2}:\d{2}(?::\d{2})?)\b", re.IGNORECASE)
ARCHIVE_UPDATES_RE = re.compile(r"\barchive_updates=(\d+)\b", re.IGNORECASE)
ARCHIVE_PRUNES_RE = re.compile(r"\barchive_prunes=(\d+)\b", re.IGNORECASE)
PERCENT_RE = re.compile(r"(\d{1,3})%")
TIMESTAMPED_LOG_RE = re.compile(r"^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\s+\|\s+[A-Z]+\s+\|\s+(.+)$")
PYTHON_EXCEPTION_LOG_RE = re.compile(
    r"^(?:[A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception|Timeout)|KeyboardInterrupt|SystemExit):\s+.+$"
)
GIT_REMOTE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
GIT_BRANCH_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]*$")
INSTANCE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
SUPPORTED_GIT_PULL_MODES = {"ff-only"}
EXECUTION_MODE_FAIR_SEQUENTIAL = "fair_sequential"
EXECUTION_MODE_EXPLORATORY_PARALLEL = "exploratory_parallel"
SUPPORTED_EXECUTION_MODES = {EXECUTION_MODE_FAIR_SEQUENTIAL, EXECUTION_MODE_EXPLORATORY_PARALLEL}
PROPOSAL_TOTALS = {
    proposal_id: total
    for proposal_id, total in (("evolmd", 6), ("mesap", 6), ("evolmd-mo", 5), ("binary-mopso-cd", 6))
}
BINARY_DETAIL_LOG_PREFIXES = ("initial population |", "PPDB SQLite index")


def resolve_comparator_config_path(config_dir: Path | None = None) -> Path:
    configured = os.environ.get("COMPARATOR_CONFIG_PATH")
    if configured:
        return Path(configured)
    config_dir = config_dir or Path(__file__).resolve().parent
    local_config = config_dir / "comparator_config.local.json"
    if local_config.exists():
        return local_config
    return config_dir / "comparator_config.json"


COMPARATOR_CONFIG_PATH = resolve_comparator_config_path()
COMPARATOR_INSTANCE_LABELS_FILENAME = "instance_labels.json"
COMPARATOR_INSTANCE_LABEL_MAX_LENGTH = 240


def load_comparator_config() -> dict[str, Any]:
    if not COMPARATOR_CONFIG_PATH.exists():
        return {}
    try:
        payload = json.loads(COMPARATOR_CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


class ComparatorChartingConfigStore:
    LABEL_FIELDS = ("title", "xAxis", "yAxis")
    MAX_LABEL_LENGTH = 160
    KEY_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}$")

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.RLock()

    def public_config(self) -> dict[str, Any]:
        with self._lock:
            return self._public_charting(self._read_payload().get("charting"))

    def save(self, payload: dict[str, Any]) -> dict[str, Any]:
        labels = self._validated_labels(payload.get("labels"))
        with self._lock:
            config = self._read_payload()
            charting = dict(config.get("charting") if isinstance(config.get("charting"), dict) else {})
            current_labels = dict(charting.get("labels") if isinstance(charting.get("labels"), dict) else {})
            for key, values in labels.items():
                current = dict(current_labels.get(key) if isinstance(current_labels.get(key), dict) else {})
                current.update(values)
                current_labels[key] = current
            charting["labels"] = current_labels
            config["charting"] = charting
            self._write_payload(config)
            return self._public_charting(charting)

    def _read_payload(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"No se pudo leer la configuracion del comparador: {error}") from error
        if not isinstance(payload, dict):
            raise ValueError("La configuracion del comparador debe ser un objeto JSON.")
        return payload

    def _write_payload(self, payload: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(payload, ensure_ascii=False, indent=2)
        last_error: OSError | None = None
        for attempt in range(6):
            temp_path = self.path.with_name(f"{self.path.name}.{os.getpid()}.{time.time_ns()}.tmp")
            try:
                temp_path.write_text(encoded, encoding="utf-8")
                temp_path.replace(self.path)
                return
            except OSError as error:
                last_error = error
                try:
                    temp_path.unlink(missing_ok=True)
                except OSError:
                    pass
                time.sleep(0.05 * (attempt + 1))
        if last_error is not None:
            raise last_error

    def _validated_labels(self, raw_labels: Any) -> dict[str, dict[str, str]]:
        if not isinstance(raw_labels, dict) or not raw_labels:
            raise ValueError("labels debe ser un objeto no vacio.")
        labels: dict[str, dict[str, str]] = {}
        for raw_key, raw_values in raw_labels.items():
            key = str(raw_key or "").strip()
            if not self.KEY_RE.match(key):
                raise ValueError("Cada clave de labels debe ser alfanumerica.")
            if not isinstance(raw_values, dict):
                raise ValueError(f"labels.{key} debe ser un objeto.")
            values: dict[str, str] = {}
            for field in self.LABEL_FIELDS:
                if field not in raw_values:
                    continue
                value = str(raw_values.get(field) or "").strip()
                if not value:
                    raise ValueError(f"labels.{key}.{field} no puede estar vacio.")
                if len(value) > self.MAX_LABEL_LENGTH:
                    raise ValueError(f"labels.{key}.{field} no puede superar {self.MAX_LABEL_LENGTH} caracteres.")
                values[field] = value
            if values:
                labels[key] = values
        if not labels:
            raise ValueError("labels debe incluir al menos title, xAxis o yAxis.")
        return labels

    def _public_charting(self, charting: Any) -> dict[str, Any]:
        source = charting if isinstance(charting, dict) else {}
        labels = source.get("labels") if isinstance(source.get("labels"), dict) else {}
        clean_labels: dict[str, dict[str, str]] = {}
        for key, values in labels.items():
            if not self.KEY_RE.match(str(key or "")) or not isinstance(values, dict):
                continue
            clean_values = {
                field: str(values[field]).strip()
                for field in self.LABEL_FIELDS
                if field in values and str(values[field]).strip()
            }
            if clean_values:
                clean_labels[str(key)] = clean_values
        public = {
            "library": str(source.get("library") or "Apache ECharts"),
            "version": str(source.get("version") or ""),
            "localPath": str(source.get("localPath") or ""),
            "labels": clean_labels,
        }
        return public


def config_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "si"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return default


def normalized_git_url(value: Any) -> str:
    url = str(value or "").strip().replace("\\", "/")
    if url.endswith("/"):
        url = url[:-1]
    if url.endswith(".git"):
        url = url[:-4]
    return url.lower()


COMPARATOR_CONFIG = load_comparator_config()
COMPARATOR_DEFAULTS = COMPARATOR_CONFIG.get("defaults") if isinstance(COMPARATOR_CONFIG.get("defaults"), dict) else {}
COMPARATOR_PROPOSAL_CONFIG = COMPARATOR_CONFIG.get("proposals") if isinstance(COMPARATOR_CONFIG.get("proposals"), dict) else {}
POSTHOC_EMBEDDING_MODEL = str(COMPARATOR_DEFAULTS.get("posthocEmbeddingModel") or "all-MiniLM-L6-v2")
DEFAULT_SELECTED_PROPOSALS = tuple(COMPARATOR_DEFAULTS.get("selectedProposalIds") or ("evolmd", "mesap", "evolmd-mo", "binary-mopso-cd"))
DEFAULT_UPDATE_REPOSITORIES_BEFORE_RUN = config_bool(COMPARATOR_DEFAULTS.get("updateRepositoriesBeforeRun"), False)
DEFAULT_EXECUTION_MODE = str(COMPARATOR_DEFAULTS.get("executionMode") or EXECUTION_MODE_FAIR_SEQUENTIAL)
COMPARATOR_TIMEOUT_MINUTES_MIN = 2400
COMPARATOR_TIMEOUT_MINUTES_MAX = 10800
METRIC_SCHEMA_VERSION = 4
METRIC_COORDINATE_SPACE = "comparable_normalized"
BINARY_INTERNAL_COORDINATE_SPACE = "binary_native_normalized"
POSTHOC_DIAGNOSTIC_KMEANS_CLUSTERS = 5
POSTHOC_ENTITY_ENTROPY_POS = {"NOUN", "VERB", "ADJ"}
PROXY_OBJECTIVE_NAMES = ["fidelity_sbert_proxy", "semantic_diversity_proxy"]
MODULE_ROOT = Path(__file__).resolve().parents[1]
BINARY_PROPOSAL_ID = "binary-mopso-cd"
BINARY_LOCAL_REPOSITORY = "baselines/external/binary-mopso-cd"
BINARY_DEVELOPMENT_REPOSITORY = "../Binary MOPSO-CD"
BINARY_PORTAL_PPDB_SOURCE = Path("data/external/ppdb/ppdb-2.0-s-all")
BINARY_PORTAL_PPDB_SQLITE_INDEX = Path("data/turbulence/ppdb_index.sqlite")
BINARY_DEFAULT_OLLAMA_TIMEOUT_SECONDS = 600
BINARY_ALTERNATIVE_MODEL_REPLACED_DEFAULT = "llama3.1:8b"
BINARY_TASK_MODEL_PREFIX = "router.task_models."
BINARY_TASK_THINKING_PREFIX = "router.task_thinking."
BINARY_THINKING_MODE_CHOICES = ("false", "low", "medium", "high")
BINARY_MANAGED_CONFIG_PATHS = {
    "experiment.n",
    "experiment.iterations",
    "experiment.runs",
    "experiment.seed",
    "runtime.outdir_base",
    "ollama.default_model",
    "initialization.population_input_path",
    "initialization.reference_context_input_path",
}
SAME_INITIAL_POPULATION_SCOPE = "per_repetition"
BINARY_AUTO_PARALLELISM_PATHS = (
    "parallelism.particle_update_max_concurrent",
    "parallelism.initial_text_generation_max_concurrent",
)
BINARY_REMOVED_CLI_FLAGS = {
    "--n",
    "--iterations",
    "--runs",
    "--seed",
    "--model",
    "--bert-model",
    "--outdir-base",
    "--freeze-components",
    "--enable-monitor",
    "--disable-selection",
    "--router-heuristic",
    "--task-model",
    "--ppdb-source",
    "--ppdb-index",
    "--enable-checkpoint",
    "--checkpoint-every",
    "--checkpoint-interval",
    "--resume-from",
}
BINARY_PATH_OPTION_TYPES = {
    "runtime.resume_from": "path",
    "runtime.outdir_base": "path",
    "runtime.embedding_cache_file": "path",
    "checkpoint.directory": "path",
    "models.ppdb.source_path": "path",
    "models.ppdb.index_path": "path",
    "logging.file": "path",
}
BINARY_FORCED_OPTION_TYPES = {
    **BINARY_PATH_OPTION_TYPES,
    "mopso.archive_multiplier": "float",
    "mopso.alpha": "float",
}
BINARY_STANDARD_COMPONENTS = ("role", "topic", "action")
BINARY_OLLAMA_MODEL_CHOICES = ("llama3", "llama3.1:8b", "qwen3.5:2b")
BINARY_GUIDED_LIST_OPTIONS = {
    "experiment.frozen_components": {
        "type": "component_multi_select",
        "choices": BINARY_STANDARD_COMPONENTS,
        "allowCustom": False,
        "valueHelp": "Selecciona componentes semanticas que no podran modificarse durante MOPSO-CD. Puedes congelarlas todas; en ese caso Binary evalua y devuelve la poblacion inicial.",
    },
    "semantic_components.order": {
        "type": "ordered_multi_select",
        "choices": BINARY_STANDARD_COMPONENTS,
        "allowCustom": True,
        "valueHelp": "Define las componentes activas y su orden semantico. El orden tambien determina experiment.components dentro de Binary si difiere.",
    },
    "semantic_components.expansion_order": {
        "type": "ordered_multi_select",
        "choices": BINARY_STANDARD_COMPONENTS,
        "allowCustom": True,
        "valueHelp": "Define en que orden se expanden los pools semanticos. Puedes usar el mismo conjunto que el orden semantico o una permutacion.",
    },
}
BINARY_PATH_CHOICES = {
    "logging.level": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    "models.sbert.default": ["all-MiniLM-L6-v2", "gte-small"],
    "models.distilbert.model": ["distilbert-base-uncased"],
    "models.spacy.model": ["en_core_web_sm"],
    "ollama.alternative_model": BINARY_OLLAMA_MODEL_CHOICES,
}
BINARY_SELECT_OPTION_PATHS = {
    "logging.level",
}
BINARY_VALUE_HELP = {
    "models.sbert.default": "Modelo SBERT usado para embeddings y metricas semanticas. Puedes elegir un alias conocido o escribir un modelo compatible.",
    "ollama.default_model": "Gestionado por el modelo comun del comparador. Las tareas del router conservan sus defaults salvo override explicito.",
    "ollama.alternative_model": "Reemplaza los modelos por tarea que en Binary usan por defecto llama3.1:8b. Los overrides explicitos por tarea tienen prioridad.",
    "ollama.timeout_seconds": "Solo aplica a llamadas Ollama/LLM. No limita la construccion del indice PPDB ni el timeout global del proceso del comparador.",
    "logging.level": "Nivel minimo de logs emitidos por Binary. DEBUG es mas verboso; INFO es el nivel usual.",
    "parallelism.enabled": "Activa paralelismo interno de Binary. Para comparaciones de costo justas, recuerda usar modo secuencial del comparador.",
    "parallelism.particle_update_max_concurrent": "Override opcional. Si lo dejas vacio, el comparador envia auto: N de la comparacion; si escribes un valor, fuerza ese limite.",
    "parallelism.initial_text_generation_max_concurrent": "Override opcional. Si lo dejas vacio, el comparador envia auto: N de la comparacion; si escribes un valor, fuerza ese limite.",
    "selection.enabled": "Activa el modulo de seleccion final de Binary.",
    "selection.k": "Cantidad de soluciones seleccionadas al final. Binary exige un entero positivo.",
    "selection.lambda_mmr": "Peso MMR entre relevancia y diversidad. Binary valida el intervalo [0, 1].",
    "selection.tau_min": "Umbral inferior usado por el modulo de seleccion; Binary valida tau_min <= tau_max.",
    "selection.tau_max": "Umbral superior usado por el modulo de seleccion; Binary valida tau_min <= tau_max.",
    "mopso.dmax": "Maximo de componentes que puede modificar una particula. Binary exige que sea mayor que 0 y no supere las componentes activas.",
    "mopso.alpha": "Parametro real positivo de la funcion de transferencia.",
    "mopso.archive_multiplier": "Multiplicador real positivo del tamano de archivo externo respecto de N.",
    "mopso.k_retry": "Binary valida que este valor permanezca en 0.",
    "mopso.p_anchor_enabled": "Activa probabilidad de anclaje durante actualizaciones discretas.",
    "mopso.p_anchor_min": "Probabilidad minima de anclaje. Binary valida p_anchor_min <= p_anchor_max y p_anchor_max <= 1.",
    "mopso.p_anchor_max": "Probabilidad maxima de anclaje. Binary valida p_anchor_min <= p_anchor_max y p_anchor_max <= 1.",
    "mopso.p_tur_min": "Probabilidad minima de turbulencia. Binary valida p_tur_min <= p_tur_max.",
    "mopso.p_tur_max": "Probabilidad maxima de turbulencia. Binary valida p_tur_min <= p_tur_max.",
    "mopso.tau_tur_min": "Umbral minimo para turbulencia. Binary valida tau_tur_min <= tau_tur_max.",
    "mopso.tau_tur_max": "Umbral maximo para turbulencia. Binary valida tau_tur_min <= tau_tur_max.",
    "mopso.guided_trajectory_validation_enabled": "Activa o desactiva la validacion angular de candidatos guiados.",
    "mopso.guided_trajectory_relative_margin": "Margen relativo no negativo usado por la validacion angular cuando esta activa.",
    "generated_text_validation.tau_gen_min": "Fidelidad minima aceptada para texto generado; Binary valida que este entre -1 y 1.",
    "checkpoint.enabled": "Activa escritura de checkpoints.",
    "checkpoint.interval": "Frecuencia de checkpoints cuando estan activos. Binary exige entero positivo.",
    "models.ppdb.enabled": "Activa candidatos PPDB. Si esta activo, Binary requiere index_path.",
    "models.ppdb.index_path": "Ruta del indice SQLite de PPDB.",
    "models.ppdb.source_path": "Ruta fuente de PPDB para crear/usar indice.",
    "models.distilbert.top_k_multiplier": "Multiplicador de top-k para candidatos DistilBERT. Binary exige valor positivo.",
    "initialization.candidate_multiplier": "Multiplicador de candidatos por componente durante inicializacion. Binary exige valor positivo.",
    "initialization.min_product_multiplier": "Minimo relativo del producto de pools iniciales. Binary exige valor positivo.",
    "initialization.prompt_reduction_multiplier": "Factor para reducir prompts candidatos. Binary exige valor positivo.",
    "runtime.eager_load_models": "Si esta activo, Binary carga modelos semanticos al inicio.",
}
BINARY_PATH_LABELS = {
    "experiment.domain": "Dominio",
    "experiment.components": "Componentes activas",
    "experiment.frozen_components": "Componentes congeladas",
    "runtime.resume_from": "Reanudar desde",
    "runtime.embedding_cache_file": "Cache de embeddings",
    "runtime.eager_load_models": "Carga anticipada de modelos",
    "parallelism.enabled": "Paralelismo interno",
    "parallelism.particle_update_max_concurrent": "Concurrencia por particulas",
    "parallelism.initial_text_generation_max_concurrent": "Concurrencia poblacion inicial",
    "checkpoint.enabled": "Checkpoints",
    "checkpoint.interval": "Intervalo de checkpoint",
    "checkpoint.directory": "Directorio de checkpoints",
    "logging.enabled": "Logging activo",
    "logging.console": "Logs en consola",
    "logging.file": "Archivo de logs",
    "logging.level": "Nivel de logs",
    "models.sbert.default": "Modelo SBERT",
    "models.distilbert.model": "Modelo DistilBERT",
    "models.spacy.model": "Modelo spaCy",
    "models.wordnet.enabled": "WordNet",
    "models.ppdb.enabled": "PPDB",
    "models.ppdb.source_path": "Ruta fuente PPDB",
    "models.ppdb.index_path": "Indice PPDB",
    "ollama.host": "Host Ollama",
    "ollama.timeout_seconds": "Timeout Ollama",
    "ollama.alternative_model": "Modelo alternativo",
    "ollama.stream": "Streaming Ollama",
    "ollama.think": "Modo think",
    "semantic_components.order": "Orden de componentes",
    "semantic_components.expansion_order": "Orden de expansion",
    "initialization.candidate_multiplier": "Multiplicador de candidatos",
    "initialization.min_product_multiplier": "Producto minimo de pools",
    "initialization.prompt_reduction_multiplier": "Reduccion de prompts",
    "initialization.generated_sentences_min": "Minimo de frases generadas",
    "initialization.generated_sentences_max": "Maximo de frases generadas",
    "generated_text_validation.tau_gen_min": "Tau minimo de texto generado",
    "mopso.archive_multiplier": "Multiplicador de archivo",
    "mopso.leader_tournament_size": "Torneo de lider",
    "mopso.dmax": "D maximo",
    "mopso.kcand": "Candidatos por cambio",
    "mopso.omega_max": "Omega maximo",
    "mopso.omega_min": "Omega minimo",
    "mopso.p_anchor_enabled": "Anclaje activo",
    "mopso.k_retry": "Reintentos",
    "mopso.guided_trajectory_validation_enabled": "Validacion trayectoria guiada",
    "mopso.guided_trajectory_relative_margin": "Margen relativo trayectoria guiada",
    "selection.enabled": "Seleccion final activa",
    "selection.k": "Cantidad seleccionada",
    "selection.lambda_mmr": "Lambda MMR",
    "selection.tau_min": "Tau minimo",
    "selection.tau_max": "Tau maximo",
    "monitor.enabled": "Monitor activo",
    "monitor.kmeans_clusters": "Clusters KMeans",
}
BINARY_PATH_HELP = {
    "experiment.n": "Gestionado por el campo comun N del comparador.",
    "experiment.iterations": "Gestionado por el campo comun G del comparador.",
    "experiment.runs": "El comparador ejecuta K externamente; Binary corre una repeticion por proceso.",
    "experiment.seed": "Gestionado por la semilla efectiva de cada repeticion.",
    "runtime.outdir_base": "Gestionado por el comparador para aislar artefactos por corrida.",
    "ollama.default_model": "Gestionado por el campo comun Modelo del comparador; no pisa los modelos por tarea.",
    "ollama.alternative_model": "Si se define, el comparador lo traduce a modelos por tarea para los defaults llama3.1:8b de Binary.",
    "mopso.k_retry": "Debe permanecer en 0 segun la validacion actual de Binary.",
    "mopso.guided_trajectory_validation_enabled": "Override booleano para activar o desactivar la validacion de trayectoria guiada.",
    "mopso.guided_trajectory_relative_margin": "Override numerico no negativo para el margen relativo de trayectoria guiada.",
    "ollama.speculative_decoding_enabled": "Binary bloquea esta opcion durante validacion.",
}


def module_relative_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else MODULE_ROOT / path


def configured_binary_repository_path() -> str:
    binary_config = COMPARATOR_PROPOSAL_CONFIG.get(BINARY_PROPOSAL_ID)
    binary_config = binary_config if isinstance(binary_config, dict) else {}
    configured = str(binary_config.get("repositoryPath") or BINARY_LOCAL_REPOSITORY)
    if module_relative_path(configured).exists():
        return configured
    if configured == BINARY_LOCAL_REPOSITORY and module_relative_path(BINARY_DEVELOPMENT_REPOSITORY).exists():
        return BINARY_DEVELOPMENT_REPOSITORY
    return configured


def binary_portal_ppdb_overrides(root: Path, manual_paths: set[str]) -> list[tuple[str, Any, str]]:
    source_path = (root / BINARY_PORTAL_PPDB_SOURCE).resolve()
    index_path = (root / BINARY_PORTAL_PPDB_SQLITE_INDEX).resolve()
    if not source_path.exists() and not index_path.exists():
        return []
    overrides: list[tuple[str, Any, str]] = []
    if source_path.exists() and "models.ppdb.source_path" not in manual_paths:
        overrides.append(("models.ppdb.source_path", str(source_path), "path"))
    if "models.ppdb.index_path" not in manual_paths:
        overrides.append(("models.ppdb.index_path", str(index_path), "path"))
    return overrides


def flatten_mapping_leaves(value: Any, prefix: str = "") -> list[tuple[str, Any]]:
    if isinstance(value, dict):
        leaves: list[tuple[str, Any]] = []
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            leaves.extend(flatten_mapping_leaves(child, path))
        return leaves
    return [(prefix, value)] if prefix else []


def binary_option_label(path: str) -> str:
    if path in BINARY_PATH_LABELS:
        return BINARY_PATH_LABELS[path]
    if path.startswith(BINARY_TASK_MODEL_PREFIX):
        task_name = path.removeprefix(BINARY_TASK_MODEL_PREFIX).replace("_", " ")
        return f"Modelo tarea: {task_name}"
    if path.startswith(BINARY_TASK_THINKING_PREFIX):
        task_name = path.removeprefix(BINARY_TASK_THINKING_PREFIX).replace("_", " ")
        return f"Thinking tarea: {task_name}"
    return path.split(".")[-1].replace("_", " ")


def binary_option_group(path: str) -> str:
    return path.split(".", 1)[0]


def binary_option_type(path: str, value: Any) -> str:
    guided = BINARY_GUIDED_LIST_OPTIONS.get(path)
    if guided:
        return str(guided["type"])
    if path.startswith(BINARY_TASK_THINKING_PREFIX):
        return "thinking_mode"
    if path in BINARY_FORCED_OPTION_TYPES:
        return BINARY_FORCED_OPTION_TYPES[path]
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int) and not isinstance(value, bool):
        return "int"
    if isinstance(value, float):
        return "float"
    if value is None or isinstance(value, (list, dict)):
        return "yaml"
    return "string"


def binary_default_config_path() -> Path:
    return module_relative_path(configured_binary_repository_path()) / "configs" / "default.yaml"


def load_binary_default_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Binary default.yaml was not found at {path}.")
    text = path.read_text(encoding="utf-8")
    try:
        import yaml

        payload = yaml.safe_load(text) or {}
    except ModuleNotFoundError:
        payload = parse_simple_yaml_mapping(text)
    if not isinstance(payload, dict):
        raise ValueError(f"Binary default.yaml root must be a mapping: {path}.")
    return payload


def binary_model_options_from_config(config: dict[str, Any]) -> list[str]:
    ollama = config.get("ollama") if isinstance(config.get("ollama"), dict) else {}
    raw_options = ollama.get("model_options") if isinstance(ollama, dict) else None
    if not isinstance(raw_options, list):
        return list(BINARY_OLLAMA_MODEL_CHOICES)
    options = [str(model).strip() for model in raw_options if str(model).strip()]
    return options or list(BINARY_OLLAMA_MODEL_CHOICES)


def binary_model_capabilities_from_config(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    ollama = config.get("ollama") if isinstance(config.get("ollama"), dict) else {}
    raw_capabilities = ollama.get("model_capabilities") if isinstance(ollama, dict) else None
    if not isinstance(raw_capabilities, dict):
        return {}
    capabilities: dict[str, dict[str, Any]] = {}
    for model, values in raw_capabilities.items():
        if isinstance(values, dict):
            normalized = dict(values)
            tasks = values.get("validated_thinking_tasks")
            if isinstance(tasks, list):
                normalized["validated_thinking_tasks"] = [
                    str(task).strip()
                    for task in tasks
                    if str(task).strip()
                ]
            else:
                normalized["validated_thinking_tasks"] = []
            capabilities[str(model)] = normalized
    return capabilities


def binary_task_models_from_config(config: dict[str, Any]) -> dict[str, str]:
    router = config.get("router") if isinstance(config.get("router"), dict) else {}
    raw_task_models = router.get("task_models") if isinstance(router, dict) else None
    if not isinstance(raw_task_models, dict):
        return {}
    task_models: dict[str, str] = {}
    for task_name, model in raw_task_models.items():
        model_text = str(model or "").strip()
        if model_text:
            task_models[str(task_name).strip()] = model_text
    return {task: model for task, model in task_models.items() if task}


def load_binary_model_metadata() -> tuple[list[str], dict[str, dict[str, Any]]]:
    config = load_binary_default_config(binary_default_config_path())
    return binary_model_options_from_config(config), binary_model_capabilities_from_config(config)


def comparator_ollama_model_choices() -> tuple[str, ...]:
    try:
        binary_choices, _capabilities = load_binary_model_metadata()
    except Exception:
        binary_choices = list(BINARY_OLLAMA_MODEL_CHOICES)
    return tuple(dict.fromkeys([str(COMPARATOR_DEFAULTS.get("model") or "llama3"), *binary_choices]))


def comparator_ollama_model_capabilities() -> dict[str, dict[str, Any]]:
    try:
        _binary_choices, capabilities = load_binary_model_metadata()
    except Exception:
        return {}
    return capabilities


def comparator_binary_default_task_models() -> dict[str, str]:
    try:
        return binary_task_models_from_config(load_binary_default_config(binary_default_config_path()))
    except Exception:
        return {}


def binary_alternative_task_model_overrides(
    alternative_model: Any,
    manual_paths: set[str],
) -> list[tuple[str, Any, str]]:
    model = str(alternative_model or "").strip()
    if not model or model == BINARY_ALTERNATIVE_MODEL_REPLACED_DEFAULT:
        return []
    overrides: list[tuple[str, Any, str]] = []
    for task_name, default_model in sorted(comparator_binary_default_task_models().items()):
        if default_model != BINARY_ALTERNATIVE_MODEL_REPLACED_DEFAULT:
            continue
        path = f"{BINARY_TASK_MODEL_PREFIX}{task_name}"
        if path in manual_paths:
            continue
        overrides.append((path, model, "string"))
    return overrides


def parse_simple_yaml_mapping(text: str) -> dict[str, Any]:
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any] | list[Any], dict[str, Any] | None, str | None]] = [(-1, root, None, None)]
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        stripped = raw_line.strip()
        if stripped.startswith("-"):
            value_text = stripped[1:].strip()
            while stack and indent < stack[-1][0]:
                stack.pop()
            if not stack:
                raise ValueError(f"Invalid YAML list indentation at line {line_number}.")
            entry_indent, container, parent, parent_key = stack[-1]
            if not isinstance(container, list):
                if parent is None or parent_key is None:
                    raise ValueError(f"Unsupported YAML sequence at line {line_number}.")
                if isinstance(container, dict) and container:
                    raise ValueError(f"Unsupported YAML sequence at line {line_number}.")
                container = []
                parent[parent_key] = container
                stack[-1] = (entry_indent, container, parent, parent_key)
            container.append(parse_simple_yaml_scalar(value_text))
            continue
        key, raw_value = split_simple_yaml_mapping_line(stripped, line_number)
        key = key.strip()
        value_text = raw_value.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        if not stack:
            raise ValueError(f"Invalid YAML indentation at line {line_number}.")
        parent = stack[-1][1]
        if not isinstance(parent, dict):
            raise ValueError(f"Unsupported YAML mapping inside sequence at line {line_number}.")
        if not value_text:
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child, parent, key))
        else:
            parent[key] = parse_simple_yaml_scalar(value_text)
    return root


def split_simple_yaml_mapping_line(stripped: str, line_number: int) -> tuple[str, str]:
    quote: str | None = None
    for index, character in enumerate(stripped):
        if character in {"'", '"'}:
            if quote == character:
                quote = None
            elif quote is None:
                quote = character
            continue
        if character == ":" and quote is None and (index == len(stripped) - 1 or stripped[index + 1].isspace()):
            return stripped[:index], stripped[index + 1:]
    raise ValueError(f"Unsupported YAML syntax at line {line_number}.")


def parse_simple_yaml_scalar(value: str) -> Any:
    text = value.strip()
    lowered = text.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none", "~"}:
        return None
    if text.startswith("[") and text.endswith("]"):
        try:
            return ast.literal_eval(text)
        except (SyntaxError, ValueError):
            return [item.strip().strip("\"'") for item in text[1:-1].split(",") if item.strip()]
    if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
        try:
            return ast.literal_eval(text)
        except (SyntaxError, ValueError):
            return text[1:-1]
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def build_binary_cli_options(config: dict[str, Any] | None = None) -> tuple[tuple[dict[str, Any], ...], str | None]:
    base_options: list[dict[str, Any]] = [
        {"flag": "--reference-text", "type": "string", "source": "managed"},
        {
            "flag": "--config",
            "type": "path",
            "label": "YAML base adicional",
            "help": "Archivo YAML opcional que se mezcla sobre configs/default.yaml antes de aplicar overrides --set.",
        },
    ]
    if config is None:
        try:
            config = load_binary_default_config(binary_default_config_path())
        except Exception as exc:
            return tuple(base_options), str(exc)
    binary_model_choices = binary_model_options_from_config(config)

    for path, default_value in sorted(flatten_mapping_leaves(config), key=lambda item: item[0]):
        if path == "ollama.model_options" or path.startswith("ollama.model_capabilities."):
            continue
        option_type = binary_option_type(path, default_value)
        option: dict[str, Any] = {
            "flag": "--set",
            "key": path,
            "configPath": path,
            "type": option_type,
            "default": default_value,
            "group": binary_option_group(path),
            "label": binary_option_label(path),
            "help": BINARY_PATH_HELP.get(path, "Se envia a Binary como override YAML con --set path=value."),
            "valueHelp": BINARY_VALUE_HELP.get(path),
        }
        guided = BINARY_GUIDED_LIST_OPTIONS.get(path)
        if guided:
            option.update(guided)
        if path in BINARY_SELECT_OPTION_PATHS:
            option["ui"] = "select"
        if path in BINARY_MANAGED_CONFIG_PATHS:
            option["source"] = "managed"
        if option_type == "bool":
            option["allowFalse"] = True
        if path in BINARY_PATH_CHOICES and "choices" not in option:
            option["choices"] = BINARY_PATH_CHOICES[path]
            option["allowCustom"] = path not in BINARY_SELECT_OPTION_PATHS
        if path == "ollama.alternative_model":
            option["choices"] = binary_model_choices
            option["allowCustom"] = True
        if path.startswith(BINARY_TASK_MODEL_PREFIX):
            option["choices"] = binary_model_choices
            option["allowCustom"] = True
        if path.startswith(BINARY_TASK_THINKING_PREFIX):
            task_name = path.removeprefix(BINARY_TASK_THINKING_PREFIX)
            option["pairedModelPath"] = f"{BINARY_TASK_MODEL_PREFIX}{task_name}"
            option["choices"] = list(BINARY_THINKING_MODE_CHOICES)
            option["allowFalse"] = True
            option["valueHelp"] = (
                "Activa thinking si el modelo efectivo de esta tarea lo soporta. "
                "Las tareas validadas se muestran solo como referencia."
            )
        base_options.append(option)
    return tuple(base_options), None


BINARY_CLI_OPTIONS, BINARY_CONFIG_METADATA_ERROR = build_binary_cli_options()


def current_binary_cli_options() -> tuple[tuple[dict[str, Any], ...], str | None]:
    return build_binary_cli_options()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def finite_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def finite_int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return int(number)


COMPARATOR_RUN_LOG_LIMIT = max(250, int(finite_float(COMPARATOR_DEFAULTS.get("runLogTailLimit"), 1000)))
COMPARATOR_LOG_CHUNK_LIMIT = max(1000, int(finite_float(COMPARATOR_DEFAULTS.get("logChunkLimit"), 5000)))
RUN_SNAPSHOT_COPY_ATTEMPTS = 3
RUN_SNAPSHOT_COPY_RETRY_DELAY_SECONDS = 0.05


def objective_label(vector: list[float]) -> str:
    if not vector:
        return "--"
    return "[" + ", ".join(f"{value:.6f}" for value in vector) + "]"


def clean_log_message(message: str) -> str:
    return ANSI_RE.sub("", message).strip()


def progress_log_payload(message: str) -> str:
    clean_message = clean_log_message(message)
    match = TIMESTAMPED_LOG_RE.match(clean_message)
    return match.group(1).strip() if match else clean_message


def parse_elapsed_seconds(value: str) -> float | None:
    parts = value.strip().split(":")
    if len(parts) == 2:
        hours = 0
        minutes, seconds = parts
    elif len(parts) == 3:
        hours, minutes, seconds = parts
    else:
        return None
    try:
        return float(int(hours) * 3600 + int(minutes) * 60 + int(seconds))
    except ValueError:
        return None


def log_elapsed_seconds(message: str) -> float | None:
    match = ELAPSED_RE.search(message)
    return parse_elapsed_seconds(match.group(1)) if match else None


def log_generation_time_seconds(message: str) -> float | None:
    match = GENERATION_TIME_RE.search(message)
    if not match:
        return None
    return finite_float(match.group(1).replace(",", "."), -1.0)


def empty_iteration_timing() -> dict[str, Any]:
    return {
        "completedIterations": 0,
        "totalIterations": None,
        "remainingIterations": None,
        "durationSamples": [],
        "averageIterationSeconds": None,
        "averageIterationLabel": "No disponible",
        "lastIterationSeconds": None,
        "lastIterationLabel": "No disponible",
        "completedIterationKeys": [],
        "activeIterationKey": None,
        "activeStartedAtEpoch": None,
        "activeElapsedSeconds": None,
        "previousElapsedSeconds": None,
    }


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


def comparator_status_message(status: str) -> str:
    return {
        STATUS_QUEUED: "En cola",
        STATUS_RUNNING: "Ejecutando",
        STATUS_COMPLETED: "Completada",
        STATUS_FAILED: "Fallida",
        STATUS_CANCELLED: "Cancelada",
    }.get(status, status)


def calculate_posthoc_semantic_scores(generated_texts: list[str], reference_text: str) -> list[dict[str, float]]:
    if not generated_texts:
        return []
    texts = [text if text.strip() else "[texto vacio]" for text in generated_texts]
    reference = reference_text.strip() or "[texto referencia vacio]"
    embeddings, _ = shared_sbert_service().encode_texts(POSTHOC_EMBEDDING_MODEL, [reference, *texts])
    reference_embedding = embeddings[0]
    text_embeddings = embeddings[1:]
    fidelity_scores = text_embeddings @ reference_embedding
    if len(generated_texts) <= 1:
        return [
            {
                "semanticFidelity": clamp(finite_float(fidelity), -1.0, 1.0),
                "semanticDiversity": 0.0,
            }
            for fidelity in fidelity_scores
        ]

    similarity_matrix = text_embeddings @ text_embeddings.T
    scores: list[dict[str, float]] = []
    for index in range(len(texts)):
        sum_similarity = float(similarity_matrix[index].sum()) - 1.0
        average_similarity = sum_similarity / (len(texts) - 1)
        scores.append(
            {
                "semanticFidelity": clamp(finite_float(fidelity_scores[index]), -1.0, 1.0),
                "semanticDiversity": clamp(1.0 - average_similarity, 0.0, 2.0),
            }
        )
    return scores


class ComparableObjectiveProxy:
    def attach(self, rows: list[dict[str, Any]], reference_text: str, display_name: str) -> None:
        valid_rows = [row for row in rows if row.get("status") == "ok"]
        scores = calculate_posthoc_semantic_scores(
            [row.get("generatedText") or "" for row in valid_rows],
            reference_text,
        )
        for row in rows:
            row["postHocNonDominated"] = False
            if row.get("status") != "ok":
                row.pop("comparableObjectiveVector", None)
                row.pop("comparableObjectiveLabel", None)
                row.pop("comparableObjectiveNames", None)
        for row, score in zip(valid_rows, scores):
            fidelity_score = finite_float(score.get("semanticFidelity"))
            diversity_score = finite_float(score.get("semanticDiversity"))
            proxy_vector = [fidelity_score, diversity_score]
            comparable_vector = normalized_objective_vector(proxy_vector) or []
            row["proxyObjectiveVector"] = proxy_vector
            row["proxyObjectiveLabel"] = objective_label(proxy_vector)
            row["proxyObjectiveNames"] = list(PROXY_OBJECTIVE_NAMES)
            row["diagnosticObjectiveVector"] = proxy_vector
            row["diagnosticObjectiveLabel"] = objective_label(proxy_vector)
            row["diagnosticObjectiveNames"] = list(PROXY_OBJECTIVE_NAMES)
            row["comparableObjectiveVector"] = comparable_vector
            row["comparableObjectiveLabel"] = objective_label(comparable_vector)
            row["comparableObjectiveNames"] = list(COMPARABLE_OBJECTIVE_NAMES)
            row["postHocDiagnostics"] = {
                "semanticFidelity": fidelity_score,
                "semanticDiversity": diversity_score,
                "semanticDiversityModel": POSTHOC_EMBEDDING_MODEL,
                "proxyObjectiveNames": list(PROXY_OBJECTIVE_NAMES),
                "note": f"Common comparable proxy for {display_name}; native objectives are preserved as traceability only.",
            }


def latest_child_directory(path: Path) -> Path | None:
    if not path.exists():
        return None
    children = [child for child in path.iterdir() if child.is_dir()]
    if not children:
        return None
    return max(children, key=lambda child: child.stat().st_mtime)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, ensure_ascii=False, indent=2)
    last_error: OSError | None = None
    for attempt in range(6):
        temp_path = path.with_name(f"{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
        try:
            temp_path.write_text(encoded, encoding="utf-8")
            temp_path.replace(path)
            return
        except OSError as error:
            last_error = error
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass
            time.sleep(0.05 * (attempt + 1))
    if last_error is not None:
        raise last_error


def read_json_or_default(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return read_json(path)
    except (OSError, json.JSONDecodeError):
        return default


def label_from_seconds(seconds: Any) -> str:
    return format_duration(finite_float(seconds, -1.0))


def parse_runtime_file(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    runtime: dict[str, float] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        key, separator, value = line.partition("=")
        if not separator:
            key, separator, value = line.partition(":")
        if not separator:
            continue
        runtime[key.strip()] = finite_float(value.strip())
    return runtime


def empty_cost_metrics() -> dict[str, Any]:
    return {
        "processWallClockSeconds": 0.0,
        "processWallClockLabel": "0s",
        "algorithmRuntimeSeconds": None,
        "algorithmRuntimeLabel": "No disponible",
        "proposalTotalWallClockSeconds": 0.0,
        "proposalTotalWallClockLabel": "0s",
        "postProcessingWallClockSeconds": 0.0,
        "postProcessingWallClockLabel": "0s",
        "metricExtractionSeconds": 0.0,
        "metricExtractionLabel": "0s",
        "plotPreparationSeconds": 0.0,
        "plotPreparationLabel": "0s",
        "llmCalls": 0,
        "llmSuccessfulCalls": 0,
        "llmFailedCalls": 0,
        "llmEmptyContentCalls": 0,
        "llmClientWallClockSeconds": 0.0,
        "llmClientWallClockLabel": "0s",
        "llmAverageCallSeconds": None,
        "llmAverageCallLabel": "No disponible",
        "ollamaTotalDurationSeconds": 0.0,
        "ollamaTotalDurationLabel": "0s",
        "promptEvalCount": 0,
        "evalCount": 0,
        "totalTokens": 0,
        "hasTokenReport": False,
        "hasOllamaDurationReport": False,
        "returnCode": None,
        "timedOut": False,
        "cancelled": False,
        "runtimeBreakdown": {},
    }


def build_cost_metrics(
    process_cost: dict[str, Any],
    llm_payload: dict[str, Any],
    output_dir: Path | None,
    cancelled: bool,
) -> dict[str, Any]:
    summary = llm_payload.get("summary") if isinstance(llm_payload, dict) else {}
    summary = summary if isinstance(summary, dict) else {}
    calls = llm_payload.get("calls") if isinstance(llm_payload, dict) and isinstance(llm_payload.get("calls"), list) else []
    runtime = parse_runtime_file(output_dir / "runtime.txt") if output_dir else {}
    algorithm_runtime = runtime.get("total_sec")
    if algorithm_runtime is None:
        algorithm_runtime = runtime.get("runtime_seconds")
    if algorithm_runtime is None:
        algorithm_runtime = runtime.get("grand_total_sec")
    llm_calls = int(finite_float(summary.get("totalCalls")))
    has_token_report = llm_calls_have_token_report(calls) or int(finite_float(summary.get("totalTokens"))) > 0
    has_ollama_duration_report = (
        llm_calls_have_ollama_duration_report(calls)
        or finite_float(summary.get("ollamaTotalDurationSeconds")) > 0
    )
    llm_client_seconds = finite_float(summary.get("clientWallClockSeconds"))
    average_call_seconds = llm_client_seconds / llm_calls if llm_calls > 0 else None
    cost = empty_cost_metrics()
    cost.update(
        {
            "processWallClockSeconds": finite_float(process_cost.get("processWallClockSeconds")),
            "processWallClockLabel": label_from_seconds(process_cost.get("processWallClockSeconds")),
            "algorithmRuntimeSeconds": algorithm_runtime,
            "algorithmRuntimeLabel": label_from_seconds(algorithm_runtime),
            "proposalTotalWallClockSeconds": finite_float(process_cost.get("processWallClockSeconds")),
            "proposalTotalWallClockLabel": label_from_seconds(process_cost.get("processWallClockSeconds")),
            "postProcessingWallClockSeconds": 0.0,
            "postProcessingWallClockLabel": "0s",
            "metricExtractionSeconds": 0.0,
            "metricExtractionLabel": "0s",
            "plotPreparationSeconds": 0.0,
            "plotPreparationLabel": "0s",
            "llmCalls": llm_calls,
            "llmSuccessfulCalls": int(finite_float(summary.get("successfulCalls"))),
            "llmFailedCalls": int(finite_float(summary.get("failedCalls"))),
            "llmEmptyContentCalls": int(finite_float(summary.get("emptyContentCalls"))),
            "llmClientWallClockSeconds": llm_client_seconds,
            "llmClientWallClockLabel": label_from_seconds(llm_client_seconds),
            "llmAverageCallSeconds": average_call_seconds,
            "llmAverageCallLabel": label_from_seconds(average_call_seconds),
            "ollamaTotalDurationSeconds": finite_float(summary.get("ollamaTotalDurationSeconds")),
            "ollamaTotalDurationLabel": label_from_seconds(summary.get("ollamaTotalDurationSeconds")),
            "promptEvalCount": int(finite_float(summary.get("promptEvalCount"))),
            "evalCount": int(finite_float(summary.get("evalCount"))),
            "totalTokens": int(finite_float(summary.get("totalTokens"))),
            "hasTokenReport": has_token_report,
            "hasOllamaDurationReport": has_ollama_duration_report,
            "returnCode": process_cost.get("returnCode"),
            "timedOut": bool(process_cost.get("timedOut")),
            "cancelled": cancelled,
            "runtimeBreakdown": runtime,
            "metricsPath": process_cost.get("metricsPath"),
        }
    )
    return cost


def read_llm_calls_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    calls: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            calls.append(payload)
    return calls


def llm_calls_have_token_report(calls: list[dict[str, Any]]) -> bool:
    return any(
        any(key in call for key in ("promptEvalCount", "evalCount", "totalTokens", "prompt_eval_count", "eval_count"))
        for call in calls
    )


def llm_calls_have_ollama_duration_report(calls: list[dict[str, Any]]) -> bool:
    return any(
        any(
            key in call
            for key in (
                "ollamaTotalDurationSeconds",
                "ollama_total_duration_seconds",
                "total_duration",
            )
        )
        for call in calls
    )


def call_int_metric(call: dict[str, Any], *keys: str) -> int:
    for key in keys:
        if key in call:
            return int(finite_float(call.get(key)))
    return 0


def call_empty_content(call: dict[str, Any]) -> bool:
    if call.get("empty_content") is True or call.get("emptyContent") is True:
        return True
    if "content_chars" in call:
        return int(finite_float(call.get("content_chars"))) == 0
    if "contentChars" in call:
        return int(finite_float(call.get("contentChars"))) == 0
    return False


def call_ollama_total_duration_seconds(call: dict[str, Any]) -> float:
    if "ollamaTotalDurationSeconds" in call:
        return finite_float(call.get("ollamaTotalDurationSeconds"))
    if "ollama_total_duration_seconds" in call:
        return finite_float(call.get("ollama_total_duration_seconds"))
    if "total_duration" in call:
        return finite_float(call.get("total_duration")) / 1_000_000_000.0
    return 0.0


def build_binary_cost_metrics(
    process_cost: dict[str, Any],
    output_dir: Path | None,
    cancelled: bool,
) -> dict[str, Any]:
    runtime = parse_runtime_file(output_dir / "runtime.txt") if output_dir else {}
    calls = read_llm_calls_jsonl(output_dir / "llm_calls.jsonl") if output_dir else []
    llm_seconds = sum(finite_float(call.get("elapsed_seconds")) for call in calls)
    llm_calls = len(calls)
    prompt_eval_count = sum(call_int_metric(call, "promptEvalCount", "prompt_eval_count") for call in calls)
    eval_count = sum(call_int_metric(call, "evalCount", "eval_count") for call in calls)
    empty_content_calls = sum(1 for call in calls if call_empty_content(call))
    ollama_duration_seconds = sum(call_ollama_total_duration_seconds(call) for call in calls)
    algorithm_runtime = runtime.get("runtime_seconds") or runtime.get("total_sec") or runtime.get("grand_total_sec")
    cost = empty_cost_metrics()
    cost.update(
        {
            "processWallClockSeconds": finite_float(process_cost.get("processWallClockSeconds")),
            "processWallClockLabel": label_from_seconds(process_cost.get("processWallClockSeconds")),
            "algorithmRuntimeSeconds": algorithm_runtime,
            "algorithmRuntimeLabel": label_from_seconds(algorithm_runtime),
            "proposalTotalWallClockSeconds": finite_float(process_cost.get("processWallClockSeconds")),
            "proposalTotalWallClockLabel": label_from_seconds(process_cost.get("processWallClockSeconds")),
            "llmCalls": llm_calls,
            "llmSuccessfulCalls": sum(1 for call in calls if call.get("status", "ok") != "error"),
            "llmFailedCalls": sum(1 for call in calls if call.get("status") == "error"),
            "llmEmptyContentCalls": empty_content_calls,
            "llmClientWallClockSeconds": llm_seconds,
            "llmClientWallClockLabel": label_from_seconds(llm_seconds),
            "llmAverageCallSeconds": (llm_seconds / llm_calls) if llm_calls else None,
            "llmAverageCallLabel": label_from_seconds((llm_seconds / llm_calls) if llm_calls else None),
            "ollamaTotalDurationSeconds": ollama_duration_seconds,
            "ollamaTotalDurationLabel": label_from_seconds(ollama_duration_seconds),
            "promptEvalCount": prompt_eval_count,
            "evalCount": eval_count,
            "totalTokens": prompt_eval_count + eval_count,
            "hasTokenReport": llm_calls_have_token_report(calls),
            "hasOllamaDurationReport": llm_calls_have_ollama_duration_report(calls),
            "returnCode": process_cost.get("returnCode"),
            "timedOut": bool(process_cost.get("timedOut")),
            "cancelled": cancelled,
            "runtimeBreakdown": runtime,
            "metricsPath": process_cost.get("metricsPath"),
            "llmTaskBreakdown": llm_task_breakdown(calls),
        }
    )
    return cost


def llm_task_breakdown(calls: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    breakdown: dict[str, dict[str, Any]] = {}
    for call in calls:
        task = str(call.get("semantic_task") or call.get("task") or "unknown")
        bucket = breakdown.setdefault(
            task,
            {"calls": 0, "elapsedSeconds": 0.0, "contentChars": 0, "emptyContentCalls": 0},
        )
        bucket["calls"] += 1
        bucket["elapsedSeconds"] += finite_float(call.get("elapsed_seconds"))
        bucket["contentChars"] += int(finite_float(call.get("content_chars")))
        if call_empty_content(call):
            bucket["emptyContentCalls"] += 1
    for bucket in breakdown.values():
        bucket["elapsedLabel"] = label_from_seconds(bucket["elapsedSeconds"])
    return breakdown


def add_cost_timing(cost: dict[str, Any], key: str, seconds: float) -> None:
    cost[key] = finite_float(cost.get(key)) + max(0.0, seconds)
    cost[f"{key.removesuffix('Seconds')}Label"] = label_from_seconds(cost[key])
    process = finite_float(cost.get("processWallClockSeconds"))
    post = finite_float(cost.get("postProcessingWallClockSeconds"))
    cost["proposalTotalWallClockSeconds"] = process + post
    cost["proposalTotalWallClockLabel"] = label_from_seconds(process + post)


def summarize_costs(proposals: list[dict[str, Any]], run_elapsed_seconds: float) -> dict[str, Any]:
    costs = [proposal.get("cost") or empty_cost_metrics() for proposal in proposals]
    llm_calls = sum(int(cost_total_metric(cost, "llmCalls")) for cost in costs)
    llm_client_seconds = sum(cost_total_metric(cost, "llmClientWallClockSeconds") for cost in costs)
    process_seconds_sum = sum(cost_total_metric(cost, "processWallClockSeconds") for cost in costs)
    post_seconds_sum = sum(cost_total_metric(cost, "postProcessingWallClockSeconds") for cost in costs)
    metric_seconds_sum = sum(cost_total_metric(cost, "metricExtractionSeconds") for cost in costs)
    plot_seconds_sum = sum(cost_total_metric(cost, "plotPreparationSeconds") for cost in costs)
    proposal_total_seconds_sum = sum(cost_total_metric(cost, "proposalTotalWallClockSeconds", "processWallClockSeconds") for cost in costs)
    algorithm_seconds_sum = sum(
        value
        for cost in costs
        for value in [cost_total_metric_if_present(cost, "algorithmRuntimeSeconds")]
        if value is not None
    )
    prompt_tokens = sum(int(cost_total_metric(cost, "promptEvalCount")) for cost in costs)
    completion_tokens = sum(int(cost_total_metric(cost, "evalCount")) for cost in costs)
    total_tokens = sum(int(cost_total_metric(cost, "totalTokens")) for cost in costs)
    has_token_report = any(bool(cost.get("hasTokenReport")) for cost in costs)
    has_ollama_duration_report = any(bool(cost.get("hasOllamaDurationReport")) for cost in costs)
    average_call_seconds = llm_client_seconds / llm_calls if llm_calls > 0 else None
    ollama_seconds = sum(cost_total_metric(cost, "ollamaTotalDurationSeconds") for cost in costs)
    return {
        "runWallClockSeconds": run_elapsed_seconds,
        "runWallClockLabel": label_from_seconds(run_elapsed_seconds),
        "proposalWallClockSecondsSum": process_seconds_sum,
        "proposalWallClockSumLabel": label_from_seconds(process_seconds_sum),
        "proposalTotalWallClockSecondsSum": proposal_total_seconds_sum,
        "proposalTotalWallClockSumLabel": label_from_seconds(proposal_total_seconds_sum),
        "postProcessingWallClockSecondsSum": post_seconds_sum,
        "postProcessingWallClockSumLabel": label_from_seconds(post_seconds_sum),
        "metricExtractionSecondsSum": metric_seconds_sum,
        "metricExtractionSumLabel": label_from_seconds(metric_seconds_sum),
        "plotPreparationSecondsSum": plot_seconds_sum,
        "plotPreparationSumLabel": label_from_seconds(plot_seconds_sum),
        "algorithmRuntimeSecondsSum": algorithm_seconds_sum,
        "algorithmRuntimeSumLabel": label_from_seconds(algorithm_seconds_sum),
        "llmCalls": llm_calls,
        "llmSuccessfulCalls": sum(int(cost.get("llmSuccessfulCalls") or 0) for cost in costs),
        "llmFailedCalls": sum(int(cost.get("llmFailedCalls") or 0) for cost in costs),
        "llmEmptyContentCalls": sum(int(cost_total_metric(cost, "llmEmptyContentCalls")) for cost in costs),
        "llmClientWallClockSeconds": llm_client_seconds,
        "llmClientWallClockLabel": label_from_seconds(llm_client_seconds),
        "llmAverageCallSeconds": average_call_seconds,
        "llmAverageCallLabel": label_from_seconds(average_call_seconds),
        "ollamaTotalDurationSeconds": ollama_seconds,
        "ollamaTotalDurationLabel": label_from_seconds(ollama_seconds),
        "promptEvalCount": prompt_tokens,
        "evalCount": completion_tokens,
        "totalTokens": total_tokens,
        "hasTokenReport": has_token_report,
        "hasOllamaDurationReport": has_ollama_duration_report,
    }


def average_present(values: list[Any]) -> float | None:
    numbers = [finite_float(value) for value in values if value is not None]
    return sum(numbers) / len(numbers) if numbers else None


def finite_present_numbers(values: list[Any]) -> list[float]:
    numbers: list[float] = []
    for value in values:
        if value is None:
            continue
        number = finite_float(value, None)
        if number is not None:
            numbers.append(number)
    return numbers


def sample_std_dev_present(values: list[Any]) -> float | None:
    numbers = finite_present_numbers(values)
    return statistics.stdev(numbers) if len(numbers) >= 2 else None


def apply_std_dev_field(
    target: dict[str, Any],
    key: str,
    values: list[Any],
    *,
    digits: int = 6,
    label_formatter: Callable[[float], str] | None = None,
) -> None:
    std_dev = sample_std_dev_present(values)
    value_key = f"{key}StdDev"
    label_key = f"{key}StdDevLabel"
    if std_dev is None:
        target.pop(value_key, None)
        target.pop(label_key, None)
        return
    target[value_key] = std_dev
    target[label_key] = label_formatter(std_dev) if label_formatter else f"{std_dev:.{digits}f}"


def average_vector(vectors: list[Any]) -> list[float]:
    normalized = [
        [finite_float(value) for value in vector]
        for vector in vectors
        if isinstance(vector, list) and vector
    ]
    if not normalized:
        return []
    width = min(len(vector) for vector in normalized)
    return [sum(vector[index] for vector in normalized) / len(normalized) for index in range(width)]


def sum_present_metric(metrics_list: list[dict[str, Any]], key: str, fallback_key: str | None = None) -> float | None:
    values: list[float] = []
    for metrics in metrics_list:
        if metrics.get(key) is not None:
            values.append(finite_float(metrics.get(key)))
        elif fallback_key and metrics.get(fallback_key) is not None:
            values.append(finite_float(metrics.get(fallback_key)))
    return sum(values) if values else None


def aggregate_runtime_breakdowns(costs: list[dict[str, Any]]) -> dict[str, float]:
    totals: dict[str, float] = {}
    for cost in costs:
        runtime = cost.get("runtimeBreakdown") if isinstance(cost.get("runtimeBreakdown"), dict) else {}
        for key, value in runtime.items():
            totals[key] = totals.get(key, 0.0) + finite_float(value)
    return totals


def average_runtime_breakdowns(totals: dict[str, float], count: int) -> dict[str, float]:
    if count <= 0:
        return {}
    return {key: value / count for key, value in totals.items()}


def cost_total_metric(cost: dict[str, Any], key: str, fallback_key: str | None = None) -> float:
    total_key = f"{key}Total"
    if cost.get(total_key) is not None:
        return finite_float(cost.get(total_key))
    if cost.get(key) is not None:
        return finite_float(cost.get(key))
    if fallback_key:
        return cost_total_metric(cost, fallback_key)
    return 0.0


def cost_total_metric_if_present(cost: dict[str, Any], key: str) -> float | None:
    if cost.get(f"{key}Total") is not None or cost.get(key) is not None:
        return cost_total_metric(cost, key)
    return None


def aggregate_comparator_costs(results: list[dict[str, Any]]) -> dict[str, Any]:
    costs = [result.get("cost") or empty_cost_metrics() for result in results]
    if not costs:
        return empty_cost_metrics()

    count = len(costs)
    llm_calls_total = sum(int(finite_float(cost.get("llmCalls"))) for cost in costs)
    llm_success_total = sum(int(finite_float(cost.get("llmSuccessfulCalls"))) for cost in costs)
    llm_failed_total = sum(int(finite_float(cost.get("llmFailedCalls"))) for cost in costs)
    llm_empty_content_total = sum(int(finite_float(cost.get("llmEmptyContentCalls"))) for cost in costs)
    llm_client_seconds_total = sum(finite_float(cost.get("llmClientWallClockSeconds")) for cost in costs)
    algorithm_values = [
        finite_float(cost.get("algorithmRuntimeSeconds"))
        for cost in costs
        if cost.get("algorithmRuntimeSeconds") is not None
    ]
    algorithm_total = sum(algorithm_values) if algorithm_values else None
    algorithm_average = (algorithm_total / len(algorithm_values)) if algorithm_values and algorithm_total is not None else None
    process_seconds_total = sum(finite_float(cost.get("processWallClockSeconds")) for cost in costs)
    post_seconds_total = sum(finite_float(cost.get("postProcessingWallClockSeconds")) for cost in costs)
    metric_seconds_total = sum(finite_float(cost.get("metricExtractionSeconds")) for cost in costs)
    plot_seconds_total = sum(finite_float(cost.get("plotPreparationSeconds")) for cost in costs)
    proposal_total_seconds_total = sum(
        finite_float(cost.get("proposalTotalWallClockSeconds"), finite_float(cost.get("processWallClockSeconds")))
        for cost in costs
    )
    ollama_seconds_total = sum(finite_float(cost.get("ollamaTotalDurationSeconds")) for cost in costs)
    prompt_eval_total = sum(int(finite_float(cost.get("promptEvalCount"))) for cost in costs)
    eval_total = sum(int(finite_float(cost.get("evalCount"))) for cost in costs)
    token_total = sum(int(finite_float(cost.get("totalTokens"))) for cost in costs)
    has_token_report = any(bool(cost.get("hasTokenReport")) for cost in costs)
    has_ollama_duration_report = any(bool(cost.get("hasOllamaDurationReport")) for cost in costs)
    average_call_seconds = llm_client_seconds_total / llm_calls_total if llm_calls_total > 0 else None
    runtime_breakdown_total = aggregate_runtime_breakdowns(costs)
    aggregated = {
        "costAggregation": "per_repetition_average",
        "costAggregationRepetitions": count,
        "processWallClockSeconds": process_seconds_total / count,
        "processWallClockLabel": label_from_seconds(process_seconds_total / count),
        "processWallClockSecondsTotal": process_seconds_total,
        "processWallClockTotalLabel": label_from_seconds(process_seconds_total),
        "proposalTotalWallClockSeconds": proposal_total_seconds_total / count,
        "proposalTotalWallClockLabel": label_from_seconds(proposal_total_seconds_total / count),
        "proposalTotalWallClockSecondsTotal": proposal_total_seconds_total,
        "proposalTotalWallClockTotalLabel": label_from_seconds(proposal_total_seconds_total),
        "postProcessingWallClockSeconds": post_seconds_total / count,
        "postProcessingWallClockLabel": label_from_seconds(post_seconds_total / count),
        "postProcessingWallClockSecondsTotal": post_seconds_total,
        "postProcessingWallClockTotalLabel": label_from_seconds(post_seconds_total),
        "metricExtractionSeconds": metric_seconds_total / count,
        "metricExtractionLabel": label_from_seconds(metric_seconds_total / count),
        "metricExtractionSecondsTotal": metric_seconds_total,
        "metricExtractionTotalLabel": label_from_seconds(metric_seconds_total),
        "plotPreparationSeconds": plot_seconds_total / count,
        "plotPreparationLabel": label_from_seconds(plot_seconds_total / count),
        "plotPreparationSecondsTotal": plot_seconds_total,
        "plotPreparationTotalLabel": label_from_seconds(plot_seconds_total),
        "algorithmRuntimeSeconds": algorithm_average,
        "algorithmRuntimeLabel": label_from_seconds(algorithm_average),
        "algorithmRuntimeSecondsTotal": algorithm_total,
        "algorithmRuntimeTotalLabel": label_from_seconds(algorithm_total),
        "llmCalls": llm_calls_total / count,
        "llmCallsTotal": llm_calls_total,
        "llmSuccessfulCalls": llm_success_total / count,
        "llmSuccessfulCallsTotal": llm_success_total,
        "llmFailedCalls": llm_failed_total / count,
        "llmFailedCallsTotal": llm_failed_total,
        "llmEmptyContentCalls": llm_empty_content_total / count,
        "llmEmptyContentCallsTotal": llm_empty_content_total,
        "llmClientWallClockSeconds": llm_client_seconds_total / count,
        "llmClientWallClockLabel": label_from_seconds(llm_client_seconds_total / count),
        "llmClientWallClockSecondsTotal": llm_client_seconds_total,
        "llmClientWallClockTotalLabel": label_from_seconds(llm_client_seconds_total),
        "llmAverageCallSeconds": average_call_seconds,
        "llmAverageCallLabel": label_from_seconds(average_call_seconds),
        "ollamaTotalDurationSeconds": ollama_seconds_total / count,
        "ollamaTotalDurationLabel": label_from_seconds(ollama_seconds_total / count),
        "ollamaTotalDurationSecondsTotal": ollama_seconds_total,
        "ollamaTotalDurationTotalLabel": label_from_seconds(ollama_seconds_total),
        "promptEvalCount": prompt_eval_total / count,
        "promptEvalCountTotal": prompt_eval_total,
        "evalCount": eval_total / count,
        "evalCountTotal": eval_total,
        "totalTokens": token_total / count,
        "totalTokensTotal": token_total,
        "hasTokenReport": has_token_report,
        "hasOllamaDurationReport": has_ollama_duration_report,
        "returnCode": next((cost.get("returnCode") for cost in reversed(costs) if cost.get("returnCode") is not None), None),
        "timedOut": any(bool(cost.get("timedOut")) for cost in costs),
        "cancelled": any(bool(cost.get("cancelled")) for cost in costs),
        "runtimeBreakdown": average_runtime_breakdowns(runtime_breakdown_total, count),
        "runtimeBreakdownTotal": runtime_breakdown_total,
        "metricsPaths": [cost.get("metricsPath") for cost in costs if cost.get("metricsPath")],
    }
    apply_std_dev_field(
        aggregated,
        "processWallClockSeconds",
        [cost.get("processWallClockSeconds") for cost in costs],
        label_formatter=label_from_seconds,
    )
    apply_std_dev_field(aggregated, "llmCalls", [cost.get("llmCalls") for cost in costs], digits=2)
    apply_std_dev_field(aggregated, "promptEvalCount", [cost.get("promptEvalCount") for cost in costs], digits=2)
    apply_std_dev_field(aggregated, "evalCount", [cost.get("evalCount") for cost in costs], digits=2)
    return aggregated


def aggregate_comparator_metrics(results: list[dict[str, Any]], proposal_dir: Path) -> dict[str, Any]:
    metrics_list = [result.get("metrics") or {} for result in results]
    if not metrics_list:
        return {
            "totalRows": 0,
            "completedRows": 0,
            "objectiveNames": [],
            "bestObjectiveVector": [],
            "bestObjectiveLabel": "--",
            "comparableObjectiveNames": list(COMPARABLE_OBJECTIVE_NAMES),
            "bestComparableObjectiveVector": [],
            "bestComparableObjectiveLabel": "--",
            "metricSchemaVersion": METRIC_SCHEMA_VERSION,
            "metricCoordinateSpace": METRIC_COORDINATE_SPACE,
            "nonDominatedRows": 0,
            "hypervolume": None,
            "hypervolumeLabel": "No aplica",
            "extent": None,
            "extentLabel": "No aplica",
            "unaryEntropy": None,
            "unaryEntropyLabel": "No aplica",
            "contribution": None,
            "contributionLabel": "No aplica",
            "outputDir": str(proposal_dir),
            "repetitionAggregation": "Promedio sobre repeticiones K con semillas distintas.",
        }
    best_vector = average_vector([metrics.get("bestObjectiveVector") for metrics in metrics_list])
    best_diagnostic_vector = average_vector([
        metrics.get("bestDiagnosticObjectiveVector")
        for metrics in metrics_list
    ])
    best_comparable_vector = average_vector([
        metrics.get("bestComparableObjectiveVector")
        for metrics in metrics_list
    ])
    hypervolume = average_present([metrics.get("hypervolume") for metrics in metrics_list])
    extent = average_present([metrics.get("extent") for metrics in metrics_list])
    unary_entropy = average_present([metrics.get("unaryEntropy") for metrics in metrics_list])
    contribution = average_present([metrics.get("contribution") for metrics in metrics_list])
    archive_update_average = average_present([
        metrics.get("externalArchiveUpdateCount")
        for metrics in metrics_list
    ])
    archive_prune_average = average_present([
        metrics.get("externalArchivePruneCount")
        for metrics in metrics_list
    ])
    archive_update_total = sum_present_metric(
        metrics_list,
        "externalArchiveUpdateCountTotal",
        "externalArchiveUpdateCount",
    )
    archive_prune_total = sum_present_metric(
        metrics_list,
        "externalArchivePruneCountTotal",
        "externalArchivePruneCount",
    )
    first_metrics = next((metrics for metrics in metrics_list if metrics), {})
    metrics: dict[str, Any] = {
        "totalRows": average_present([metrics.get("totalRows") for metrics in metrics_list]),
        "completedRows": average_present([metrics.get("completedRows") for metrics in metrics_list]),
        "objectiveNames": first_metrics.get("objectiveNames") or [],
        "bestObjectiveVector": best_vector,
        "bestObjectiveLabel": objective_label(best_vector),
        "comparableObjectiveNames": first_metrics.get("comparableObjectiveNames") or list(COMPARABLE_OBJECTIVE_NAMES),
        "bestComparableObjectiveVector": best_comparable_vector,
        "bestComparableObjectiveLabel": objective_label(best_comparable_vector),
        "metricSchemaVersion": METRIC_SCHEMA_VERSION,
        "metricCoordinateSpace": METRIC_COORDINATE_SPACE,
        "nonDominatedRows": average_present([metrics.get("nonDominatedRows") for metrics in metrics_list]),
        "hypervolume": hypervolume,
        "hypervolumeLabel": f"{hypervolume:.6f}" if hypervolume is not None else "No aplica",
        "extent": extent,
        "extentLabel": f"{extent:.6f}" if extent is not None else "No aplica",
        "unaryEntropy": unary_entropy,
        "unaryEntropyLabel": f"{unary_entropy:.6f}" if unary_entropy is not None else "No aplica",
        "contribution": contribution,
        "contributionLabel": f"{contribution:.6f}" if contribution is not None else "No aplica",
        "outputDir": str(proposal_dir),
        "moConvention": first_metrics.get("moConvention"),
        "repetitionAggregation": "Promedio sobre repeticiones K con semillas distintas.",
    }
    apply_std_dev_field(metrics, "nonDominatedRows", [metrics.get("nonDominatedRows") for metrics in metrics_list])
    apply_std_dev_field(metrics, "hypervolume", [metrics.get("hypervolume") for metrics in metrics_list])
    apply_std_dev_field(metrics, "extent", [metrics.get("extent") for metrics in metrics_list])
    apply_std_dev_field(metrics, "unaryEntropy", [metrics.get("unaryEntropy") for metrics in metrics_list])
    apply_std_dev_field(metrics, "contribution", [metrics.get("contribution") for metrics in metrics_list])
    if archive_update_average is not None and archive_prune_average is not None:
        metrics.update(
            {
                "externalArchiveUpdateCount": archive_update_average,
                "externalArchivePruneCount": archive_prune_average,
                "externalArchiveUpdateCountTotal": archive_update_total,
                "externalArchivePruneCountTotal": archive_prune_total,
            }
        )
    if first_metrics.get("postHocDiagnostic"):
        posthoc_values = [
            item.get("postHocNonDominatedRows")
            for item in metrics_list
        ]
        posthoc_non_dominated = average_present([
            item.get("postHocNonDominatedRows")
            for item in metrics_list
        ])
        metrics.update(
            {
                "postHocDiagnostic": True,
                "diagnosticObjectiveNames": first_metrics.get("diagnosticObjectiveNames") or [],
                "bestDiagnosticObjectiveVector": best_diagnostic_vector,
                "bestDiagnosticObjectiveLabel": objective_label(best_diagnostic_vector),
                "postHocNonDominatedRows": posthoc_non_dominated,
                "nonDominatedRows": posthoc_non_dominated,
            }
        )
        apply_std_dev_field(metrics, "nonDominatedRows", posthoc_values)
        apply_std_dev_field(metrics, "postHocNonDominatedRows", posthoc_values)
    if first_metrics.get("proxyDiagnostic"):
        metrics["proxyDiagnostic"] = True
    return metrics


def attach_terminal_series_diagnostics(
    metrics: dict[str, Any],
    series: list[dict[str, Any]],
    repetition_results: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    repetition_results = repetition_results or []
    for key, label_key in (("globalInertia", "globalInertiaLabel"), ("globalEntropy", "globalEntropyLabel")):
        value = latest_finite_series_value(series, key)
        metrics[key] = value
        metrics[label_key] = f"{value:.6f}" if value is not None else "No aplica"
        apply_std_dev_field(
            metrics,
            key,
            [
                latest_finite_series_value(result.get("series") or [], key)
                for result in repetition_results
                if isinstance(result, dict)
            ],
        )
    return metrics


def latest_finite_series_value(series: list[dict[str, Any]], key: str) -> float | None:
    ordered = sorted(
        [point for point in series if isinstance(point, dict)],
        key=lambda point: finite_float(point.get("generation"), -1.0),
    )
    for point in reversed(ordered):
        if point.get(key) is None:
            continue
        value = finite_float(point.get(key), None)
        if value is not None:
            return value
    return None


def embedding_front_rows_from_rows(
    rows: list[dict[str, Any]],
    selected_rows: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    selected_rows = selected_rows or []
    selected_texts = {canonical_generated_text(row.get("generatedText")) for row in selected_rows}
    selected_ranks = {
        canonical_generated_text(row.get("generatedText")): row.get("selectionRank")
        for row in selected_rows
    }
    front_rows: list[dict[str, Any]] = []
    for row in rows:
        text = str(row.get("generatedText") or "").strip()
        if not text or row.get("status") != "ok":
            continue
        if not (row.get("nonDominated") or row.get("postHocNonDominated")):
            continue
        key = canonical_generated_text(text)
        selected = bool(row.get("selected")) or key in selected_texts
        front_rows.append(
            {
                "text": text,
                "selected": selected,
                "rank": row.get("rank"),
                "selectionRank": row.get("selectionRank") or selected_ranks.get(key),
                "repetitionIndex": row.get("repetitionIndex"),
                "repetitionSeed": row.get("repetitionSeed"),
                "sourceIndex": row.get("sourceIndex"),
                "objectiveLabel": row.get("comparableObjectiveLabel")
                or row.get("diagnosticObjectiveLabel")
                or row.get("objectiveLabel")
                or "--",
                "proposalId": row.get("proposalId") or "",
                "instanceId": row.get("instanceId") or row.get("proposalId") or "",
                "displayName": row.get("displayName") or "",
            }
        )
    return front_rows


def proposal_embedding_front_rows(proposal: dict[str, Any]) -> list[dict[str, Any]]:
    existing = proposal.get("embeddingFrontRows")
    if isinstance(existing, list):
        rows = [row for row in existing if isinstance(row, dict) and str(row.get("text") or "").strip()]
        if rows:
            return rows

    fallback_rows: list[dict[str, Any]] = []
    selected_texts = {
        canonical_generated_text(point.get("label") or point.get("text"))
        for point in ((proposal.get("charts") or {}).get("selected") or [])
        if isinstance(point, dict)
    }
    for point in ((proposal.get("charts") or {}).get("nonDominated") or []):
        if not isinstance(point, dict):
            continue
        text = str(point.get("label") or point.get("text") or "").strip()
        if not text:
            continue
        key = canonical_generated_text(text)
        fallback_rows.append(
            {
                "text": text,
                "selected": bool(point.get("selected")) or key in selected_texts,
                "rank": point.get("rank"),
                "selectionRank": point.get("selectionRank"),
                "repetitionIndex": point.get("repetitionIndex"),
                "sourceIndex": point.get("sourceIndex"),
                "objectiveLabel": point.get("comparableObjectiveLabel")
                or point.get("nativeObjectiveLabel")
                or point.get("objectiveLabel")
                or "--",
                "proposalId": point.get("proposalId") or proposal.get("proposalId") or "",
                "instanceId": point.get("instanceId") or proposal.get("instanceId") or proposal.get("proposalId") or "",
                "displayName": point.get("displayName") or proposal.get("displayName") or "",
            }
        )
    if fallback_rows:
        return fallback_rows

    rows = proposal.get("rows") if isinstance(proposal.get("rows"), list) else []
    selected_rows = proposal.get("selectedRows") if isinstance(proposal.get("selectedRows"), list) else []
    return embedding_front_rows_from_rows(
        [row for row in rows if isinstance(row, dict)],
        [row for row in selected_rows if isinstance(row, dict)],
    )


def proposal_is_binary_payload(proposal: dict[str, Any]) -> bool:
    proposal_id = str(proposal.get("proposalId") or proposal.get("baseProposalId") or "")
    instance_id = str(proposal.get("instanceId") or "")
    return (
        proposal_id == BINARY_PROPOSAL_ID
        or instance_id == BINARY_PROPOSAL_ID
        or instance_id.startswith(f"{BINARY_PROPOSAL_ID}:")
        or instance_id.startswith(f"{BINARY_PROPOSAL_ID}-")
    )


def chart_point_text(point: dict[str, Any]) -> str:
    for key in ("label", "text", "labelText", "generatedText", "generated_text"):
        text = str(point.get(key) or "").strip()
        if text:
            return text
    return ""


def selected_chart_point_metadata(charts: dict[str, Any]) -> tuple[set[str], dict[str, Any]]:
    selected_texts: set[str] = set()
    selected_ranks: dict[str, Any] = {}
    for point in charts.get("selected") or []:
        if not isinstance(point, dict):
            continue
        key = canonical_generated_text(chart_point_text(point))
        if not key:
            continue
        selected_texts.add(key)
        selected_ranks[key] = point.get("selectionRank") or point.get("rank")
    return selected_texts, selected_ranks


def visible_front_chart_rows_for_projection(
    proposal: dict[str, Any],
    selected: dict[str, Any],
) -> list[dict[str, Any]]:
    charts = selected.get("charts") if isinstance(selected.get("charts"), dict) else {}
    visible_key = "pareto" if proposal_is_binary_payload(proposal) else "nonDominated"
    selected_texts, selected_ranks = selected_chart_point_metadata(charts)
    rows: list[dict[str, Any]] = []
    for point in charts.get(visible_key) or []:
        if not isinstance(point, dict):
            continue
        text = chart_point_text(point)
        if not text:
            continue
        key = canonical_generated_text(text)
        rows.append(
            {
                "text": text,
                "selected": bool(point.get("selected")) or key in selected_texts,
                "rank": point.get("rank"),
                "selectionRank": point.get("selectionRank") or selected_ranks.get(key),
                "repetitionIndex": point.get("repetitionIndex") or selected.get("repetitionIndex"),
                "repetitionSeed": point.get("repetitionSeed") or selected.get("repetitionSeed"),
                "sourceIndex": point.get("sourceIndex"),
                "objectiveLabel": point.get("comparableObjectiveLabel")
                or point.get("nativeObjectiveLabel")
                or point.get("objectiveLabel")
                or "--",
                "proposalId": point.get("proposalId") or proposal.get("proposalId") or "",
                "instanceId": point.get("instanceId") or proposal.get("instanceId") or proposal.get("proposalId") or "",
                "displayName": point.get("displayName") or proposal.get("displayName") or "",
            }
        )
    return rows


def proposal_has_repetition_chart_payloads(proposal: dict[str, Any]) -> bool:
    return isinstance(proposal.get("pointChartRepetitions"), list) or isinstance(proposal.get("repetitions"), list)


def point_chart_repetition_payload(result: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(result, dict) or result.get("status") != STATUS_COMPLETED:
        return None
    charts = result.get("charts") if isinstance(result.get("charts"), dict) else {}
    metrics = result.get("metrics") if isinstance(result.get("metrics"), dict) else {}
    embedding_front_rows = [
        row for row in (result.get("embeddingFrontRows") or [])
        if isinstance(row, dict)
    ]
    payload: dict[str, Any] = {
        "repetitionIndex": result.get("repetitionIndex"),
        "repetitionSeed": result.get("repetitionSeed"),
        "charts": charts,
        "metrics": metrics,
        "embeddingFrontRows": embedding_front_rows,
    }
    internal_analysis = result.get("internalBmopsoAnalysis")
    if isinstance(internal_analysis, dict):
        payload["internalBmopsoAnalysis"] = internal_analysis
    return payload


def proposal_point_chart_repetitions(proposal: dict[str, Any]) -> list[dict[str, Any]]:
    explicit = proposal.get("pointChartRepetitions")
    if isinstance(explicit, list):
        return [item for item in explicit if isinstance(item, dict)]

    repetitions = proposal.get("repetitions")
    if isinstance(repetitions, list):
        payloads = [
            payload
            for item in repetitions
            for payload in [point_chart_repetition_payload(item)]
            if payload
        ]
        if payloads:
            return payloads

    return [
        {
            "repetitionIndex": None,
            "repetitionSeed": None,
            "charts": proposal.get("charts") if isinstance(proposal.get("charts"), dict) else {},
            "metrics": proposal.get("metrics") if isinstance(proposal.get("metrics"), dict) else {},
            "embeddingFrontRows": proposal_embedding_front_rows(proposal),
            **(
                {"internalBmopsoAnalysis": proposal.get("internalBmopsoAnalysis")}
                if isinstance(proposal.get("internalBmopsoAnalysis"), dict)
                else {}
            ),
        }
    ]


def selected_point_chart_repetition(proposal: dict[str, Any], repetition: int | None = None) -> dict[str, Any] | None:
    repetitions = proposal_point_chart_repetitions(proposal)
    if not repetitions:
        return None
    if repetition is None:
        return repetitions[0]
    for item in repetitions:
        if finite_int_or_none(item.get("repetitionIndex")) == repetition:
            return item
    return None


def proposal_embedding_front_rows_for_repetition(
    proposal: dict[str, Any],
    repetition: int | None = None,
) -> list[dict[str, Any]]:
    selected = selected_point_chart_repetition(proposal, repetition)
    if selected is None:
        return []
    rows = visible_front_chart_rows_for_projection(proposal, selected)
    if rows or proposal_has_repetition_chart_payloads(proposal):
        return rows
    rows = selected.get("embeddingFrontRows")
    if isinstance(rows, list):
        return [row for row in rows if isinstance(row, dict) and str(row.get("text") or "").strip()]
    return proposal_embedding_front_rows(proposal)


def aggregate_proposal_repetitions(
    proposal: ProposalDefinition,
    proposal_dir: Path,
    results: list[dict[str, Any]],
    repetitions_k: int,
    identity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    identity = identity or {
        "instanceId": proposal.proposal_id,
        "proposalId": proposal.proposal_id,
        "displayName": proposal.display_name,
        "baseDisplayName": proposal.display_name,
        "proposalConfig": {"extraArgs": "", "cliValues": {}},
    }
    completed = [result for result in results if result.get("status") == STATUS_COMPLETED]
    if not completed:
        failure = results[-1] if results else {}
        status = STATUS_CANCELLED if any(result.get("status") == STATUS_CANCELLED for result in results) else STATUS_FAILED
        return {
            **identity,
            "status": status,
            "outputDir": str(proposal_dir),
            "rows": [],
            "embeddingFrontRows": [],
            "metrics": aggregate_comparator_metrics([], proposal_dir),
            "cost": aggregate_comparator_costs(results),
            "error": failure.get("error") or "Todas las repeticiones fallaron.",
            "repetitionsK": repetitions_k,
            "completedRepetitions": 0,
            "repetitions": results,
        }

    rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    embedding_front_rows: list[dict[str, Any]] = []
    for result in completed:
        repetition_index = result.get("repetitionIndex")
        repetition_seed = result.get("repetitionSeed")
        for row in result.get("rows") or []:
            if isinstance(row, dict):
                rows.append({
                    **row,
                    "instanceId": row.get("instanceId") or identity.get("instanceId"),
                    "proposalId": row.get("proposalId") or identity.get("proposalId"),
                    "displayName": row.get("displayName") or identity.get("displayName"),
                    "baseDisplayName": row.get("baseDisplayName") or identity.get("baseDisplayName"),
                    "repetitionIndex": repetition_index,
                    "repetitionSeed": repetition_seed,
                })
        for row in result.get("selectedRows") or []:
            if isinstance(row, dict):
                selected_rows.append({
                    **row,
                    "instanceId": row.get("instanceId") or identity.get("instanceId"),
                    "proposalId": row.get("proposalId") or identity.get("proposalId"),
                    "displayName": row.get("displayName") or identity.get("displayName"),
                    "baseDisplayName": row.get("baseDisplayName") or identity.get("baseDisplayName"),
                    "repetitionIndex": repetition_index,
                    "repetitionSeed": repetition_seed,
                })
        for row in result.get("embeddingFrontRows") or []:
            if isinstance(row, dict):
                embedding_front_rows.append({
                    **row,
                    "instanceId": row.get("instanceId") or identity.get("instanceId"),
                    "proposalId": row.get("proposalId") or identity.get("proposalId"),
                    "displayName": row.get("displayName") or identity.get("displayName"),
                    "repetitionIndex": row.get("repetitionIndex") or repetition_index,
                    "repetitionSeed": row.get("repetitionSeed") or repetition_seed,
                })

    series = aggregate_series(completed)
    metrics = attach_terminal_series_diagnostics(aggregate_comparator_metrics(completed, proposal_dir), series, completed)
    if not embedding_front_rows:
        embedding_front_rows = embedding_front_rows_from_rows(rows, selected_rows)
    point_chart_repetitions = [
        payload
        for result in completed
        for payload in [point_chart_repetition_payload(result)]
        if payload
    ]

    return {
        **identity,
        "status": STATUS_COMPLETED,
        "outputDir": str(proposal_dir),
        "rows": rows,
        "selectedRows": selected_rows,
        "embeddingFrontRows": embedding_front_rows,
        "metrics": metrics,
        "series": series,
        "charts": build_charts_from_rows(rows, selected_rows, series),
        "pointChartRepetitions": point_chart_repetitions,
        "cost": aggregate_comparator_costs(completed),
        "error": None if len(completed) == repetitions_k else f"{repetitions_k - len(completed)} repeticion(es) fallaron.",
        "repetitionsK": repetitions_k,
        "completedRepetitions": len(completed),
        "repetitions": results,
    }


def completed_repetition_count(results: list[dict[str, Any]]) -> int:
    return sum(1 for result in results if result.get("status") == STATUS_COMPLETED)


def proposal_config(proposal_id: str) -> dict[str, Any]:
    value = COMPARATOR_PROPOSAL_CONFIG.get(proposal_id)
    return value if isinstance(value, dict) else {}


def proposal_git_defaults(proposal_id: str) -> dict[str, Any]:
    value = proposal_config(proposal_id).get("git")
    return value if isinstance(value, dict) else {}


def tuple_from_config(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(str(item).strip() for item in value if str(item).strip())


@dataclass(frozen=True)
class ProposalDefinition:
    proposal_id: str
    display_name: str
    repository_path: str
    description: str
    objective_names: tuple[str, ...]
    result_file: str
    single_objective: bool
    kind: str = "evolmd"
    entrypoint: str = "main.py"
    final_selection_file: str | None = None
    metrics_series_file: str | None = None
    supports_history_export: bool = False
    cli_options: tuple[dict[str, Any], ...] = ()
    preload_modules: tuple[str, ...] = ()
    python_executable: str = ""
    python_path_entries: tuple[str, ...] = ()
    required_modules: tuple[str, ...] = ()
    git_remote: str = "origin"
    git_branch: str = "main"
    git_pull_mode: str = "ff-only"
    git_expected_remote_url: str = ""


@dataclass(frozen=True)
class ProposalRunInstance:
    instance_id: str
    proposal_id: str
    display_name: str
    base_display_name: str
    proposal: ProposalDefinition
    proposal_config: dict[str, Any]
    runtime_config: dict[str, Any]
    n: int
    repetitions_k: int
    order_index: int


PROPOSALS: tuple[ProposalDefinition, ...] = (
    ProposalDefinition(
        proposal_id="evolmd",
        display_name="EVOLMD",
        repository_path=str(proposal_config("evolmd").get("repositoryPath") or "baselines/external/evolmd"),
        description="Genetic prompt evolution baseline with BERTScore fitness.",
        objective_names=("fitness",),
        result_file="data_final_evaluada.json",
        single_objective=True,
        kind="evolmd",
        final_selection_file="comparator_final_selection.json",
        metrics_series_file="metrics_gen.csv",
        supports_history_export=True,
        cli_options=(
            {"flag": "--n", "type": "int", "source": "common"},
            {"flag": "--generaciones", "type": "int", "source": "common"},
            {"flag": "--k", "type": "int", "default": 3, "step": 1, "label": "K torneo"},
            {"flag": "--prob-crossover", "type": "float", "default": 0.8, "step": 0.01, "label": "Prob. crossover"},
            {"flag": "--prob-mutacion", "type": "float", "default": 0.1, "step": 0.01, "label": "Prob. mutacion"},
            {"flag": "--num-elitismo", "type": "int", "default": 2, "step": 1},
            {"flag": "--model", "type": "string", "source": "common"},
            {"flag": "--bert-model", "type": "string", "default": "bert-base-uncased", "choices": ["bert-base-uncased", "roberta-large", "distilbert-base-uncased"], "allowCustom": True},
            {"flag": "--outdir-base", "type": "path", "source": "managed"},
            {"flag": "--texto-referencia", "type": "path", "source": "managed"},
        ),
        preload_modules=("torch",),
        python_executable=str(proposal_config("evolmd").get("pythonExecutable") or ""),
        python_path_entries=tuple_from_config(proposal_config("evolmd").get("pythonPathEntries")),
        required_modules=tuple_from_config(proposal_config("evolmd").get("requiredModules")),
        git_remote=str(proposal_git_defaults("evolmd").get("remote") or "origin"),
        git_branch=str(proposal_git_defaults("evolmd").get("branch") or "main"),
        git_pull_mode=str(proposal_git_defaults("evolmd").get("pullMode") or "ff-only"),
        git_expected_remote_url=str(proposal_git_defaults("evolmd").get("expectedRemoteUrl") or ""),
    ),
    ProposalDefinition(
        proposal_id="evolmd-mo",
        display_name="EVOLMD-MO",
        repository_path=str(proposal_config("evolmd-mo").get("repositoryPath") or "baselines/external/evolmd-mo"),
        description="NSGA-II prompt evolution baseline with SBERT fidelity and diversity.",
        objective_names=("fidelity_sbert", "diversity_individual"),
        result_file="pareto_front.json",
        single_objective=False,
        kind="evolmd-mo",
        final_selection_file="comparator_final_selection.json",
        metrics_series_file="evolucion_metricas.csv",
        supports_history_export=True,
        cli_options=(
            {"flag": "--n", "type": "int", "source": "common"},
            {"flag": "--generaciones", "type": "int", "source": "common"},
            {"flag": "--k", "type": "int", "default": 3, "step": 1, "label": "K torneo"},
            {"flag": "--prob-crossover", "type": "float", "default": 0.8, "step": 0.01, "label": "Prob. crossover"},
            {"flag": "--prob-mutacion", "type": "float", "default": 0.1, "step": 0.01, "label": "Prob. mutacion"},
            {"flag": "--num-elitismo", "type": "int", "default": 2, "step": 1},
            {"flag": "--model", "type": "string", "source": "common"},
            {"flag": "--bert-model", "type": "string", "default": "all-MiniLM-L6-v2", "choices": ["all-MiniLM-L6-v2", "gte-small"], "allowCustom": True},
            {"flag": "--outdir-base", "type": "path", "source": "managed"},
            {"flag": "--texto-referencia", "type": "path", "source": "managed"},
        ),
        preload_modules=("torch",),
        python_executable=str(proposal_config("evolmd-mo").get("pythonExecutable") or ""),
        python_path_entries=tuple_from_config(proposal_config("evolmd-mo").get("pythonPathEntries")),
        required_modules=tuple_from_config(proposal_config("evolmd-mo").get("requiredModules")),
        git_remote=str(proposal_git_defaults("evolmd-mo").get("remote") or "origin"),
        git_branch=str(proposal_git_defaults("evolmd-mo").get("branch") or "main"),
        git_pull_mode=str(proposal_git_defaults("evolmd-mo").get("pullMode") or "ff-only"),
        git_expected_remote_url=str(proposal_git_defaults("evolmd-mo").get("expectedRemoteUrl") or ""),
    ),
    ProposalDefinition(
        proposal_id="mesap",
        display_name="MESAP",
        repository_path=str(proposal_config("mesap").get("repositoryPath") or "baselines/external/mesap"),
        description="Modelo Evolutivo Semantico Adaptativo para Prompts; GA uniobjetivo con fitness BERTScore penalizada.",
        objective_names=("fitness",),
        result_file="population_final.json",
        single_objective=True,
        kind="mesap",
        final_selection_file="comparator_final_selection.json",
        metrics_series_file="metrics_log.csv",
        supports_history_export=True,
        cli_options=(
            {"flag": "--n", "type": "int", "source": "common"},
            {"flag": "--generations", "type": "int", "source": "common"},
            {"flag": "--k", "type": "int", "default": 3, "step": 1, "label": "K torneo"},
            {"flag": "--elite_size", "type": "int", "default": 2, "step": 1, "label": "Elitismo"},
            {"flag": "--prob_crossover", "type": "float", "default": 0.8, "step": 0.01, "label": "Prob. crossover"},
            {"flag": "--prob_mutation", "type": "float", "default": 0.1, "step": 0.01, "label": "Prob. mutacion"},
            {"flag": "--model", "type": "string", "source": "common"},
            {"flag": "--bert_model", "type": "string", "default": "bert-base-uncased", "choices": ["bert-base-uncased", "roberta-large", "distilbert-base-uncased"], "allowCustom": True},
            {"flag": "--outdir_base", "type": "path", "source": "managed"},
            {"flag": "--reference_text", "type": "path", "source": "managed"},
        ),
        preload_modules=("torch",),
        python_executable=str(proposal_config("mesap").get("pythonExecutable") or ""),
        python_path_entries=tuple_from_config(proposal_config("mesap").get("pythonPathEntries")),
        required_modules=tuple_from_config(proposal_config("mesap").get("requiredModules")),
        git_remote=str(proposal_git_defaults("mesap").get("remote") or "origin"),
        git_branch=str(proposal_git_defaults("mesap").get("branch") or "main"),
        git_pull_mode=str(proposal_git_defaults("mesap").get("pullMode") or "ff-only"),
        git_expected_remote_url=str(proposal_git_defaults("mesap").get("expectedRemoteUrl") or ""),
    ),
    ProposalDefinition(
        proposal_id="binary-mopso-cd",
        display_name="Binary MOPSO-CD",
        repository_path=configured_binary_repository_path(),
        description="Semantic Binary MOPSO-CD optimizer with explicit SBERT fidelity/diversity and Entropy-TOPSIS-MMR selection.",
        objective_names=("f1_fidelity_sbert", "f2_semantic_diversity"),
        result_file="pareto_front.json",
        single_objective=False,
        kind="binary-mopso-cd",
        entrypoint="-m binary_mopso_cd",
        final_selection_file="final_selection_hybrid.json",
        metrics_series_file="evolucion_metricas.csv",
        cli_options=BINARY_CLI_OPTIONS,
        preload_modules=("torch",),
        python_executable=str(proposal_config("binary-mopso-cd").get("pythonExecutable") or ""),
        python_path_entries=tuple_from_config(proposal_config("binary-mopso-cd").get("pythonPathEntries")),
        required_modules=tuple_from_config(proposal_config("binary-mopso-cd").get("requiredModules")),
        git_remote=str(proposal_git_defaults("binary-mopso-cd").get("remote") or "origin"),
        git_branch=str(proposal_git_defaults("binary-mopso-cd").get("branch") or "dev"),
        git_pull_mode=str(proposal_git_defaults("binary-mopso-cd").get("pullMode") or "ff-only"),
        git_expected_remote_url=str(proposal_git_defaults("binary-mopso-cd").get("expectedRemoteUrl") or ""),
    ),
)

PROPOSAL_BY_ID = {proposal.proposal_id: proposal for proposal in PROPOSALS}


def resolve_repository(root: Path, proposal: ProposalDefinition) -> Path:
    path = Path(proposal.repository_path)
    return path if path.is_absolute() else root / path


def resolve_config_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def proposal_entrypoint_exists(root: Path, proposal: ProposalDefinition) -> bool:
    repository = resolve_repository(root, proposal)
    if proposal.kind == "binary-mopso-cd":
        return (repository / "src" / "binary_mopso_cd" / "cli.py").exists()
    return (repository / proposal.entrypoint).exists()


def proposal_python_executable(root: Path, repository: Path, proposal: ProposalDefinition) -> str:
    if proposal.python_executable:
        configured = resolve_config_path(root, proposal.python_executable)
        return str(configured)
    windows_candidates = [
        repository / ".venv" / "Scripts" / "python.exe",
        repository / "venv" / "Scripts" / "python.exe",
        root / "baselines" / "venvs" / proposal.proposal_id / "Scripts" / "python.exe",
    ]
    posix_candidates = [
        repository / ".venv" / "bin" / "python",
        repository / "venv" / "bin" / "python",
        root / "baselines" / "venvs" / proposal.proposal_id / "bin" / "python",
    ]
    candidates = (
        windows_candidates + posix_candidates
        if sys.platform == "win32"
        else posix_candidates + windows_candidates
    )
    for venv_python in candidates:
        if venv_python.exists():
            return str(venv_python)
    return sys.executable


def proposal_python_path_entries(root: Path, proposal: ProposalDefinition) -> list[str]:
    return [str(resolve_config_path(root, entry)) for entry in proposal.python_path_entries]


def check_proposal_dependencies(root: Path, proposal: ProposalDefinition) -> dict[str, Any]:
    repository = resolve_repository(root, proposal)
    executable = proposal_python_executable(root, repository, proposal)
    if not proposal.required_modules:
        return {"ok": True, "missing": [], "pythonExecutable": executable, "error": None}
    if not Path(executable).exists() and Path(executable).is_absolute():
        return {
            "ok": False,
            "missing": list(proposal.required_modules),
            "pythonExecutable": executable,
            "error": "Configured Python executable does not exist.",
        }

    script = (
        "import importlib.util, json, sys;"
        "mods = sys.argv[1:];"
        "missing = [name for name in mods if importlib.util.find_spec(name) is None];"
        "print(json.dumps({'missing': missing}))"
    )
    environment = os.environ.copy()
    entries = proposal_python_path_entries(root, proposal)
    if entries:
        environment["PYTHONPATH"] = os.pathsep.join([*entries, environment.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    try:
        completed = subprocess.run(
            [executable, "-c", script, *proposal.required_modules],
            cwd=str(repository),
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
            check=False,
            **hidden_subprocess_kwargs(),
        )
    except Exception as error:
        return {
            "ok": False,
            "missing": list(proposal.required_modules),
            "pythonExecutable": executable,
            "error": str(error),
        }
    try:
        payload = json.loads(completed.stdout.strip() or "{}")
    except json.JSONDecodeError:
        payload = {}
    missing = payload.get("missing") if isinstance(payload.get("missing"), list) else list(proposal.required_modules)
    return {
        "ok": completed.returncode == 0 and not missing,
        "missing": missing,
        "pythonExecutable": executable,
        "error": completed.stderr.strip() if completed.returncode else None,
    }


def split_cli_args(raw: str) -> list[str]:
    text = str(raw or "").strip()
    if not text:
        return []
    if sys.platform == "win32":
        ctypes.windll.shell32.CommandLineToArgvW.restype = ctypes.POINTER(ctypes.c_wchar_p)
        argc = ctypes.c_int()
        argv = ctypes.windll.shell32.CommandLineToArgvW(text, ctypes.byref(argc))
        if not argv:
            raise ValueError("Could not parse extra CLI arguments.")
        try:
            return [argv[index] for index in range(argc.value)]
        finally:
            ctypes.windll.kernel32.LocalFree(argv)
    return shlex.split(text)


def command_label(command: list[str]) -> str:
    return subprocess.list2cmdline([str(part) for part in command])


def hidden_subprocess_kwargs(detached: bool = False) -> dict[str, Any]:
    if sys.platform != "win32":
        return {}
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    if detached:
        creationflags |= subprocess.CREATE_NEW_PROCESS_GROUP
    kwargs: dict[str, Any] = {}
    if creationflags:
        kwargs["creationflags"] = creationflags
    if hasattr(subprocess, "STARTUPINFO"):
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = 0
        kwargs["startupinfo"] = startupinfo
    return kwargs


class ComparatorService:
    def __init__(self, root: Path, config_path: Path | None = None) -> None:
        self.root = root
        self.runs_root = root / "runs" / "comparator"
        self._runs: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()
        self._posthoc_spacy_model: Any | None = None
        self._comparable_proxy = ComparableObjectiveProxy()
        self._front_point_diagnostic_cache: dict[tuple[str, str], dict[str, Any]] = {}
        self._charting_config_store = ComparatorChartingConfigStore(config_path or COMPARATOR_CONFIG_PATH)

    def list_proposals(self) -> list[dict[str, Any]]:
        proposals = []
        for proposal in PROPOSALS:
            repository = resolve_repository(self.root, proposal)
            entrypoint_available = proposal_entrypoint_exists(self.root, proposal)
            dependencies = check_proposal_dependencies(self.root, proposal) if entrypoint_available else {
                "ok": False,
                "missing": list(proposal.required_modules),
                "pythonExecutable": proposal_python_executable(self.root, repository, proposal),
                "error": "Proposal entrypoint is missing.",
            }
            cli_options, metadata_error = self._proposal_cli_options_with_error(proposal)
            if proposal.kind == "binary-mopso-cd" and metadata_error:
                dependencies = dict(dependencies)
                missing = list(dependencies.get("missing") or [])
                if "default.yaml" not in missing:
                    missing.append("default.yaml")
                dependencies["ok"] = False
                dependencies["missing"] = missing
                dependencies["error"] = metadata_error
            git_config = self._default_git_config(proposal)
            proposals.append(
                {
                    "proposalId": proposal.proposal_id,
                    "displayName": proposal.display_name,
                    "description": proposal.description,
                    "repositoryPath": str(repository),
                    "pythonExecutable": dependencies.get("pythonExecutable"),
                    "objectiveNames": list(proposal.objective_names),
                    "singleObjective": proposal.single_objective,
                    "kind": proposal.kind,
                    "resultFile": proposal.result_file,
                    "finalSelectionFile": proposal.final_selection_file,
                    "metricsSeriesFile": proposal.metrics_series_file,
                    "supportsHistoryExport": proposal.supports_history_export,
                    "entrypointAvailable": entrypoint_available,
                    "dependencyStatus": dependencies,
                    "cliOptions": list(cli_options),
                    "git": {
                        **git_config,
                        "snapshot": self._repository_git_snapshot(repository, git_config),
                    },
                    "available": entrypoint_available and bool(dependencies.get("ok")),
                }
            )
        return proposals

    def _proposal_cli_options_with_error(self, proposal: ProposalDefinition) -> tuple[tuple[dict[str, Any], ...], str | None]:
        if proposal.kind != "binary-mopso-cd":
            return proposal.cli_options, None
        return current_binary_cli_options()

    def _proposal_cli_options(self, proposal: ProposalDefinition) -> tuple[dict[str, Any], ...]:
        options, _error = self._proposal_cli_options_with_error(proposal)
        return options

    def public_defaults(self) -> dict[str, Any]:
        return {
            "model": str(COMPARATOR_DEFAULTS.get("model") or "llama3"),
            "ollamaModelOptions": list(comparator_ollama_model_choices()),
            "ollamaModelCapabilities": comparator_ollama_model_capabilities(),
            "timeoutMinutes": self._int_between(
                COMPARATOR_DEFAULTS.get("timeoutMinutes", COMPARATOR_TIMEOUT_MINUTES_MIN),
                "timeoutMinutes",
                COMPARATOR_TIMEOUT_MINUTES_MIN,
                COMPARATOR_TIMEOUT_MINUTES_MAX,
            ),
            "updateRepositoriesBeforeRun": DEFAULT_UPDATE_REPOSITORIES_BEFORE_RUN,
            "executionMode": DEFAULT_EXECUTION_MODE,
            "metricSchemaVersion": METRIC_SCHEMA_VERSION,
            "metricCoordinateSpace": METRIC_COORDINATE_SPACE,
            "charting": self.charting_config(),
        }

    def charting_config(self) -> dict[str, Any]:
        return self._charting_config_store.public_config()

    def save_charting_config(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._charting_config_store.save(payload)

    def save_instance_labels(self, run_id: str, payload: dict[str, Any]) -> dict[str, Any] | None:
        run_dir = self.resolve_run_directory(run_id)
        if run_dir is None:
            return None

        with self._lock:
            active_run = self._runs.get(run_id)
            source_run = active_run if active_run is not None else read_json(run_dir / "summary.json")
            valid_instance_ids = self._instance_label_target_ids(source_run)
            next_labels = self._validated_instance_labels(payload.get("labels"), valid_instance_ids)
            current_labels = self._read_instance_labels(run_dir)
            current_labels.update(next_labels)
            self._write_instance_labels(run_dir, current_labels)
            run = self.get_run(run_id)
            if run is None:
                return None
            return {
                "labels": current_labels,
                "run": run,
            }

    def _instance_labels_path(self, run_dir: Path) -> Path:
        return run_dir / COMPARATOR_INSTANCE_LABELS_FILENAME

    def _read_instance_labels(self, run_dir: Path) -> dict[str, str]:
        path = self._instance_labels_path(run_dir)
        if not path.exists():
            return {}
        try:
            payload = read_json(path)
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"No se pudieron leer los nombres persistidos de instancias: {error}") from error
        source = payload.get("labels") if isinstance(payload, dict) else None
        if not isinstance(source, dict):
            return {}
        labels: dict[str, str] = {}
        for raw_key, raw_value in source.items():
            key = str(raw_key or "").strip()
            value = str(raw_value or "").strip()
            if key and value:
                labels[key] = value
        return labels

    def _write_instance_labels(self, run_dir: Path, labels: dict[str, str]) -> None:
        write_json_atomic(self._instance_labels_path(run_dir), {"labels": dict(sorted(labels.items()))})

    def _validated_instance_labels(self, raw_labels: Any, valid_instance_ids: set[str]) -> dict[str, str]:
        if not isinstance(raw_labels, dict) or not raw_labels:
            raise ValueError("labels debe ser un objeto no vacio.")
        labels: dict[str, str] = {}
        unknown_ids: list[str] = []
        for raw_key, raw_value in raw_labels.items():
            instance_id = str(raw_key or "").strip()
            if not instance_id:
                raise ValueError("Cada clave de labels debe identificar una instancia.")
            if instance_id not in valid_instance_ids:
                unknown_ids.append(instance_id)
                continue
            display_name = str(raw_value or "").strip()
            if not display_name:
                raise ValueError(f"labels.{instance_id} no puede estar vacio.")
            if len(display_name) > COMPARATOR_INSTANCE_LABEL_MAX_LENGTH:
                raise ValueError(
                    f"labels.{instance_id} no puede superar {COMPARATOR_INSTANCE_LABEL_MAX_LENGTH} caracteres."
                )
            if any(ord(char) < 32 for char in display_name):
                raise ValueError(f"labels.{instance_id} no puede incluir caracteres de control.")
            labels[instance_id] = display_name
        if unknown_ids:
            raise ValueError(f"labels referencia instancias desconocidas: {', '.join(unknown_ids)}.")
        if not labels:
            raise ValueError("labels debe incluir al menos una instancia valida.")
        return labels

    def _instance_label_target_ids(self, run: dict[str, Any]) -> set[str]:
        ids: set[str] = set()

        def collect(value: Any) -> None:
            if isinstance(value, dict):
                instance_id = str(value.get("instanceId") or "").strip()
                proposal_id = str(value.get("proposalId") or "").strip()
                if instance_id:
                    ids.add(instance_id)
                elif proposal_id:
                    ids.add(proposal_id)
                for child in value.values():
                    collect(child)
            elif isinstance(value, list):
                for child in value:
                    collect(child)

        collect((run.get("config") or {}).get("proposalInstances") or [])
        collect(run.get("proposalStates") or {})
        collect(run.get("proposals") or [])
        return ids

    def _run_directory_from_payload(self, run: dict[str, Any]) -> Path | None:
        run_id = str(run.get("runId") or "").strip()
        if run_id:
            return self.resolve_run_directory(run_id)
        run_dir = str(run.get("runDir") or "").strip()
        if not run_dir:
            return None
        path = Path(run_dir)
        try:
            resolved = path.resolve()
            resolved.relative_to(self.runs_root.resolve())
        except (OSError, ValueError):
            return None
        return resolved if resolved.is_dir() else None

    def _with_instance_labels(self, run: dict[str, Any]) -> dict[str, Any]:
        run_dir = self._run_directory_from_payload(run)
        labels = self._read_instance_labels(run_dir) if run_dir is not None else {}
        if not labels:
            return dict(run)
        public = self._apply_instance_labels(copy.deepcopy(run), labels)
        public["instanceLabels"] = labels
        return public

    def _apply_instance_labels(self, value: Any, labels: dict[str, str]) -> Any:
        if isinstance(value, dict):
            updated = {key: self._apply_instance_labels(child, labels) for key, child in value.items()}
            instance_id = str(updated.get("instanceId") or "").strip()
            if not instance_id:
                instance_id = str(updated.get("proposalId") or "").strip()
            if instance_id in labels:
                updated["displayName"] = labels[instance_id]
            return updated
        if isinstance(value, list):
            return [self._apply_instance_labels(item, labels) for item in value]
        return value

    def _initial_proposal_state(
        self,
        proposal: ProposalDefinition | ProposalRunInstance,
        repetitions_k: int | None = None,
    ) -> dict[str, Any]:
        instance = self._coerce_instance(proposal)
        base_proposal = instance.proposal
        total_repetitions = max(1, int(repetitions_k or instance.repetitions_k or 1))
        return {
            "instanceId": instance.instance_id,
            "proposalId": instance.proposal_id,
            "displayName": instance.display_name,
            "baseDisplayName": instance.base_display_name,
            "runtimeConfig": instance.runtime_config,
            "n": instance.n,
            "repetitionsK": total_repetitions,
            "status": STATUS_QUEUED,
            "stageLabel": "En cola",
            "stageIndex": 0,
            "stageTotal": PROPOSAL_TOTALS.get(base_proposal.proposal_id, 1),
            "generationIndex": None,
            "generationTotal": None,
            "currentRepetitionIndex": None,
            "completedRepetitions": 0,
            "totalRepetitions": total_repetitions,
            "iterationTiming": empty_iteration_timing(),
            "progress": 0.0,
            "logs": 0,
            "updatedAt": utc_now(),
        }

    def start_run(self, payload: dict[str, Any]) -> dict[str, Any]:
        config = self._read_config(payload)
        self._validate_selected_proposals_available(config)
        run_id = self._new_run_id()
        run_dir = self.runs_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        selected_instances = self._selected_instances(config)
        run = {
            "runId": run_id,
            "status": STATUS_QUEUED,
            "createdAt": utc_now(),
            "updatedAt": utc_now(),
            "startedAtEpoch": None,
            "runDir": str(run_dir),
            "config": config,
            "metricSchemaVersion": METRIC_SCHEMA_VERSION,
            "metricCoordinateSpace": METRIC_COORDINATE_SPACE,
            "logs": [],
            "proposals": [],
            "proposalStates": {
                instance.instance_id: self._initial_proposal_state(instance)
                for instance in selected_instances
            },
            "progress": {
                "percent": 0,
                "detail": "Corrida en cola.",
                "elapsedSeconds": 0,
                "elapsedLabel": "0s",
                "remainingSeconds": None,
                "remainingLabel": "No disponible",
                "etaBasisLabel": "Esperando primera iteracion completada.",
                "etaScopeLabel": "No aplica",
            },
            "costSummary": summarize_costs([], 0.0),
            "repositoryUpdates": {},
            "sameInitialPopulationArtifacts": {},
            "sameInitialPopulationForBmopso": {
                **config.get(
                    "sameInitialPopulationForBmopso",
                    {"enabled": False, "generatorInstanceId": None, "scope": SAME_INITIAL_POPULATION_SCOPE},
                ),
                "artifacts": {},
            },
            "error": None,
            "cancelRequested": False,
            "activeProcesses": {},
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
                return self._public_run(run)

        summary_path = self.runs_root / run_id / "summary.json"
        if summary_path.exists():
            return self._with_metric_recompute_status(read_json(summary_path))
        return None

    def get_run_embedding_projection(
        self,
        run_id: str,
        method: str = "pca",
        repetition: int | None = None,
    ) -> dict[str, Any] | None:
        requested_method = normalize_projection_method(method)
        run = self.get_run(run_id)
        if not run:
            return None

        reference_text = str((run.get("config") or {}).get("referenceText") or run.get("referenceText") or "").strip()
        proposals: list[dict[str, Any]] = []
        projection_rows: list[tuple[int, dict[str, Any]]] = []
        warnings: list[str] = []

        for proposal_index, proposal in enumerate(run.get("proposals") or []):
            if not isinstance(proposal, dict) or proposal.get("status") != STATUS_COMPLETED:
                continue
            rows = proposal_embedding_front_rows_for_repetition(proposal, repetition)
            proposal_payload = {
                "instanceId": proposal.get("instanceId") or proposal.get("proposalId") or "",
                "proposalId": proposal.get("proposalId") or "",
                "displayName": proposal.get("displayName") or proposal.get("proposalId") or "",
                "baseDisplayName": proposal.get("baseDisplayName") or proposal.get("displayName") or "",
                "repetitionIndex": repetition,
                "points": [],
            }
            proposals.append(proposal_payload)
            for row in rows:
                projection_rows.append((len(proposals) - 1, row))

        if not projection_rows:
            return {
                "runId": run_id,
                "method": requested_method,
                "effectiveMethod": None,
                "embeddingModel": POSTHOC_EMBEDDING_MODEL,
                "repetitionIndex": repetition,
                "reference": {"text": reference_text, "x": None, "y": None},
                "proposals": proposals,
                "warnings": ["No hay textos del frente final disponibles para proyectar."],
            }

        texts = [reference_text or "[texto referencia vacio]"]
        texts.extend(str(row.get("text") or "").strip() or "[texto vacio]" for _index, row in projection_rows)
        embeddings, cost = shared_sbert_service().encode_texts(POSTHOC_EMBEDDING_MODEL, texts)
        projection = project_embeddings_2d(embeddings, requested_method)
        coordinates = projection["coordinates"]
        warnings.extend(projection.get("warnings") or [])

        reference_coordinates = coordinates[0] if coordinates else [None, None]
        for coordinate, (proposal_index, row) in zip(coordinates[1:], projection_rows):
            proposals[proposal_index]["points"].append(
                {
                    "x": coordinate[0],
                    "y": coordinate[1],
                    "text": row.get("text") or "",
                    "selected": bool(row.get("selected")),
                    "rank": row.get("rank"),
                    "selectionRank": row.get("selectionRank"),
                    "repetitionIndex": row.get("repetitionIndex"),
                    "repetitionSeed": row.get("repetitionSeed"),
                    "sourceIndex": row.get("sourceIndex"),
                    "objectiveLabel": row.get("objectiveLabel") or "--",
                    "proposalId": row.get("proposalId") or proposals[proposal_index].get("proposalId") or "",
                    "instanceId": row.get("instanceId") or proposals[proposal_index].get("instanceId") or "",
                    "displayName": row.get("displayName") or proposals[proposal_index].get("displayName") or "",
                }
            )

        return {
            "runId": run_id,
            "method": projection["method"],
            "effectiveMethod": projection["effectiveMethod"],
            "embeddingModel": cost["embeddingModel"],
            "sourceModel": cost["sourceModel"],
            "embeddingTexts": cost["embeddingTexts"],
            "embeddingWallClockSeconds": cost["embeddingWallClockSeconds"],
            "repetitionIndex": repetition,
            "reference": {
                "text": reference_text,
                "x": reference_coordinates[0],
                "y": reference_coordinates[1],
            },
            "proposals": proposals,
            "warnings": warnings,
        }

    def get_run_front_point_diagnostics(self, run_id: str, payload: dict[str, Any]) -> dict[str, Any] | None:
        if not self.get_run(run_id):
            return None
        raw_points = payload.get("points") if isinstance(payload, dict) else None
        points = [
            {
                "key": str(point.get("key") or "").strip(),
                "text": str(point.get("text") or "").strip(),
            }
            for point in (raw_points or [])
            if isinstance(point, dict) and str(point.get("key") or "").strip()
        ]
        if not points:
            return {
                "runId": run_id,
                "embeddingModel": POSTHOC_EMBEDDING_MODEL,
                "points": [],
            }

        missing_texts: list[str] = []
        seen_missing: set[str] = set()
        for point in points:
            text = point["text"] or "[texto vacio]"
            cache_key = (POSTHOC_EMBEDDING_MODEL, text)
            if cache_key not in self._front_point_diagnostic_cache and text not in seen_missing:
                missing_texts.append(text)
                seen_missing.add(text)

        if missing_texts:
            embeddings, _cost = shared_sbert_service().encode_texts(POSTHOC_EMBEDDING_MODEL, missing_texts)
            entity_diagnostics = self._posthoc_entity_term_diagnostics(missing_texts)
            for text, embedding, entity in zip(missing_texts, embeddings, entity_diagnostics):
                self._front_point_diagnostic_cache[(POSTHOC_EMBEDDING_MODEL, text)] = {
                    "embedding": [float(value) for value in embedding],
                    "entityTerms": entity.get("entityTerms"),
                    "entityTokenCount": entity.get("entityTokenCount"),
                }

        diagnostics = []
        for point in points:
            text = point["text"] or "[texto vacio]"
            cached = self._front_point_diagnostic_cache.get((POSTHOC_EMBEDDING_MODEL, text)) or {}
            diagnostics.append(
                {
                    "key": point["key"],
                    "embedding": cached.get("embedding"),
                    "entityTerms": cached.get("entityTerms"),
                    "entityTokenCount": cached.get("entityTokenCount"),
                }
            )

        return {
            "runId": run_id,
            "embeddingModel": POSTHOC_EMBEDDING_MODEL,
            "points": diagnostics,
        }

    def _posthoc_entity_term_diagnostics(self, generated_texts: list[str]) -> list[dict[str, Any]]:
        try:
            if self._posthoc_spacy_model is None:
                import spacy

                self._posthoc_spacy_model = spacy.load("en_core_web_sm")
            diagnostics = []
            for doc in self._posthoc_spacy_model.pipe(generated_texts):
                terms = []
                for token in doc:
                    if token.pos_ in POSTHOC_ENTITY_ENTROPY_POS:
                        lemma = str(token.lemma_ or "").lower().strip()
                        if lemma:
                            terms.append(lemma)
                diagnostics.append({"entityTerms": terms, "entityTokenCount": len(doc)})
            return diagnostics
        except Exception:
            return [{"entityTerms": None, "entityTokenCount": None} for _text in generated_texts]

    def get_run_logs(self, run_id: str, offset: int = 0, limit: int | None = None) -> dict[str, Any] | None:
        run = self.get_run(run_id)
        if not run:
            return None
        safe_offset = max(0, int(offset or 0))
        safe_limit = max(1, min(int(limit or COMPARATOR_LOG_CHUNK_LIMIT), COMPARATOR_LOG_CHUNK_LIMIT))
        log_path = self._run_log_path(run)
        if not log_path.exists():
            logs = list(run.get("logs") or [])
            chunk = logs[safe_offset:safe_offset + safe_limit]
            next_offset = safe_offset + len(chunk)
            return {
                "runId": run_id,
                "offset": safe_offset,
                "nextOffset": next_offset,
                "limit": safe_limit,
                "hasMore": next_offset < len(logs),
                "logs": chunk,
                "source": "summary_tail",
            }

        file_size = log_path.stat().st_size
        safe_offset = min(safe_offset, file_size)
        entries: list[dict[str, Any]] = []
        with log_path.open("rb") as handle:
            handle.seek(safe_offset)
            while len(entries) < safe_limit:
                line_bytes = handle.readline()
                if not line_bytes:
                    break
                line = line_bytes.decode("utf-8", errors="replace")
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    entry = {"at": None, "proposalId": "system", "message": line.rstrip("\n")}
                if isinstance(entry, dict):
                    entries.append(entry)
            next_offset = handle.tell()
        return {
            "runId": run_id,
            "offset": safe_offset,
            "nextOffset": next_offset,
            "limit": safe_limit,
            "hasMore": next_offset < file_size,
            "logs": entries,
            "source": "jsonl",
            "offsetUnit": "bytes",
        }

    def resolve_run_directory(self, run_id: str) -> Path | None:
        requested = str(run_id or "").strip()
        if not requested or requested in {".", ".."} or "/" in requested or "\\" in requested:
            raise ValueError("Invalid run ID.")

        runs_root = self.runs_root.resolve()
        run_dir = (runs_root / requested).resolve()
        try:
            run_dir.relative_to(runs_root)
        except ValueError as error:
            raise ValueError("Invalid run ID.") from error

        if run_dir == runs_root:
            raise ValueError("Invalid run ID.")
        if not run_dir.is_dir():
            return None
        return run_dir

    def snapshot_run_directory(self, run_id: str, snapshot_parent: Path) -> Path | None:
        source_dir = self.resolve_run_directory(run_id)
        if source_dir is None:
            return None

        target_parent = Path(snapshot_parent)
        target_parent.mkdir(parents=True, exist_ok=True)
        snapshot_dir = target_parent / source_dir.name
        snapshot_dir.mkdir()
        self._copy_run_snapshot_tree(source_dir, snapshot_dir)
        return snapshot_dir

    def write_run_snapshot_zip(self, snapshot_dir: Path, zip_path: Path) -> None:
        source_dir = Path(snapshot_dir)
        target_path = Path(zip_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(target_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(source_dir.rglob("*"), key=lambda item: item.relative_to(source_dir).as_posix()):
                if path.is_file():
                    archive.write(path, path.relative_to(source_dir.parent).as_posix())

    def _copy_run_snapshot_tree(self, source_dir: Path, snapshot_dir: Path) -> None:
        manifest = sorted(source_dir.rglob("*"), key=lambda item: item.relative_to(source_dir).as_posix())
        for source_path in manifest:
            relative_path = source_path.relative_to(source_dir)
            target_path = snapshot_dir / relative_path
            if source_path.is_dir():
                target_path.mkdir(parents=True, exist_ok=True)
                continue
            if source_path.is_file():
                self._copy_snapshot_file(source_path, target_path, relative_path)
                continue
            raise RuntimeError(f"Cannot snapshot unsupported path: {relative_path.as_posix()}")

    def _copy_snapshot_file(self, source_path: Path, target_path: Path, relative_path: Path) -> None:
        last_error: BaseException | None = None
        for attempt in range(RUN_SNAPSHOT_COPY_ATTEMPTS):
            try:
                before = source_path.stat()
                if not source_path.is_file():
                    raise RuntimeError(f"Snapshot source is not a file: {relative_path.as_posix()}")
                target_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_path, target_path)
                after = source_path.stat()
                if before.st_size == after.st_size and before.st_mtime_ns == after.st_mtime_ns:
                    return
                last_error = RuntimeError(f"File changed while creating snapshot: {relative_path.as_posix()}")
            except OSError as error:
                last_error = error
            if attempt < RUN_SNAPSHOT_COPY_ATTEMPTS - 1:
                time.sleep(RUN_SNAPSHOT_COPY_RETRY_DELAY_SECONDS)
        raise RuntimeError(f"Could not create stable snapshot for {relative_path.as_posix()}") from last_error

    def recompute_run_metrics(self, run_id: str) -> dict[str, Any] | None:
        with self._lock:
            active_run = self._runs.get(run_id)
            source_run = self._public_run(active_run) if active_run else None

        summary_path = self.runs_root / run_id / "summary.json"
        if source_run is None:
            if not summary_path.exists():
                return None
            source_run = read_json(summary_path)

        if source_run.get("status") != STATUS_COMPLETED:
            raise ValueError("Solo se pueden recalcular metricas de corridas completadas.")

        recomputed_run = self._recompute_run_metrics_payload(source_run)
        with self._lock:
            active_run = self._runs.get(run_id)
            if active_run:
                hidden = {
                    "activeProcesses": active_run.get("activeProcesses", {}),
                    "startedAtEpoch": active_run.get("startedAtEpoch"),
                }
                active_run.clear()
                active_run.update(recomputed_run)
                active_run.update(hidden)
                self._write_summary_unlocked(active_run)
                return self._public_run(active_run)

            write_json(summary_path, recomputed_run)
            return self._with_metric_recompute_status(recomputed_run)

    def recontinue_run(self, run_id: str) -> dict[str, Any] | None:
        summary_path = self.runs_root / run_id / "summary.json"
        with self._lock:
            run = self._runs.get(run_id)
            if run is None:
                if not summary_path.exists():
                    return None
                run = read_json(summary_path)
                run["activeProcesses"] = {}
                run["startedAtEpoch"] = None
                self._runs[run_id] = run

            if run.get("status") == STATUS_COMPLETED:
                raise ValueError("No se puede re-continuar una corrida completada.")
            if self._has_live_processes_unlocked(run):
                raise ComparatorRunConflictError("La corrida todavia tiene procesos activos.")
            run["activeProcesses"] = {}

            remaining_instances = self._prepare_recontinue_unlocked(run)
            if not remaining_instances:
                run["status"] = STATUS_COMPLETED
                run["error"] = None
                run["updatedAt"] = utc_now()
                self._refresh_run_progress_unlocked(run)
                self._refresh_cost_summary_unlocked(run)
                self._write_summary_unlocked(run)
                return self._public_run(run)

            run["status"] = STATUS_RUNNING
            run["cancelRequested"] = False
            run["error"] = None
            run["startedAtEpoch"] = time.time()
            run["updatedAt"] = utc_now()
            self._refresh_run_progress_unlocked(run)
            self._refresh_cost_summary_unlocked(run)
            self._write_summary_unlocked(run)

        thread = threading.Thread(target=self._recontinue_worker, args=(run_id, remaining_instances), daemon=True)
        thread.start()
        with self._lock:
            return self._public_run(self._runs[run_id])

    def _has_live_processes_unlocked(self, run: dict[str, Any]) -> bool:
        for process in (run.get("activeProcesses") or {}).values():
            if process is None:
                continue
            poll = getattr(process, "poll", None)
            if not callable(poll):
                return True
            try:
                if poll() is None:
                    return True
            except Exception:
                return True
        return False

    def _prepare_recontinue_unlocked(self, run: dict[str, Any]) -> list[ProposalRunInstance]:
        config = run.get("config") if isinstance(run.get("config"), dict) else {}
        selected_instances = self._selected_instances(config)
        states = run.setdefault("proposalStates", {})
        completed_ids = {
            str(instance_id)
            for instance_id, state in states.items()
            if isinstance(state, dict) and state.get("status") == STATUS_COMPLETED
        }
        completed_ids.update(
            str(proposal.get("instanceId") or proposal.get("proposalId") or "")
            for proposal in (run.get("proposals") or [])
            if isinstance(proposal, dict) and proposal.get("status") == STATUS_COMPLETED
        )
        completed_ids.discard("")
        for instance in selected_instances:
            total_repetitions = max(1, int(instance.repetitions_k or 1))
            state = states.setdefault(instance.instance_id, self._initial_proposal_state(instance))
            state["runtimeConfig"] = instance.runtime_config
            state["n"] = instance.n
            state["repetitionsK"] = total_repetitions
            state["totalRepetitions"] = total_repetitions
            if instance.instance_id in completed_ids:
                state.update(
                    {
                        "status": STATUS_COMPLETED,
                        "stageLabel": comparator_status_message(STATUS_COMPLETED),
                        "progress": 1.0,
                        "completedRepetitions": total_repetitions,
                        "currentRepetitionIndex": total_repetitions,
                        "totalRepetitions": total_repetitions,
                        "updatedAt": utc_now(),
                    }
                )
        first_incomplete_index = next(
            (
                index
                for index, instance in enumerate(selected_instances)
                if instance.instance_id not in completed_ids
            ),
            None,
        )
        if first_incomplete_index is None:
            return []

        remaining_instances = [
            instance
            for instance in selected_instances[first_incomplete_index:]
            if instance.instance_id not in completed_ids
        ]
        reset_ids = {instance.instance_id for instance in remaining_instances}

        run["proposals"] = [
            proposal
            for proposal in (run.get("proposals") or [])
            if str(proposal.get("instanceId") or proposal.get("proposalId") or "") not in reset_ids
        ]

        run_dir = Path(str(run.get("runDir") or ""))
        for instance_id in reset_ids:
            self._delete_run_instance_dir(run_dir, instance_id)

        for instance in remaining_instances:
            states[instance.instance_id] = self._initial_proposal_state(instance)

        self._filter_run_logs_unlocked(run, reset_ids)
        self._sort_proposals_unlocked(run)
        self._refresh_run_progress_unlocked(run)
        self._refresh_cost_summary_unlocked(run)
        return remaining_instances

    def _delete_run_instance_dir(self, run_dir: Path, instance_id: str) -> None:
        if not run_dir:
            raise ValueError("No existe runDir para limpiar la instancia.")
        run_root = run_dir.resolve(strict=True)
        target = run_dir / instance_id
        target_resolved = target.resolve(strict=False)
        if target_resolved.parent != run_root:
            raise ValueError(f"Ruta de instancia fuera del runDir: {instance_id}")
        if not target.exists():
            return
        if target.is_dir():
            shutil.rmtree(target)
            return
        target.unlink()

    def _filter_run_logs_unlocked(self, run: dict[str, Any], reset_ids: set[str]) -> None:
        run["logs"] = [
            entry
            for entry in (run.get("logs") or [])
            if not isinstance(entry, dict) or str(entry.get("proposalId") or "") not in reset_ids
        ]
        log_path = self._run_log_path(run)
        if not log_path.exists():
            return
        kept_lines: list[str] = []
        with log_path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                stripped = line.rstrip("\n")
                try:
                    entry = json.loads(stripped)
                except json.JSONDecodeError:
                    kept_lines.append(stripped)
                    continue
                if isinstance(entry, dict) and str(entry.get("proposalId") or "") in reset_ids:
                    continue
                kept_lines.append(stripped)
        with log_path.open("w", encoding="utf-8") as handle:
            if kept_lines:
                handle.write("\n".join(kept_lines) + "\n")

    def cancel_run(self, run_id: str) -> dict[str, Any] | None:
        with self._lock:
            run = self._runs.get(run_id)
            if not run:
                return None
            run["cancelRequested"] = True
            run["updatedAt"] = utc_now()
            processes = list((run.get("activeProcesses") or {}).values())
            self._append_log_unlocked(run, "system", "Cancel requested. Terminating active baseline processes.")
            self._mark_queued_as_cancelled_unlocked(run)

        for process in processes:
            self._terminate_process(process)

        with self._lock:
            self._write_summary_unlocked(run)
            return self._public_run(run)

    def _recompute_run_metrics_payload(self, run: dict[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        config = run.get("config") if isinstance(run.get("config"), dict) else {}
        proposals: list[dict[str, Any]] = []
        recomputed_ids: list[str] = []

        for result in run.get("proposals") or []:
            if not isinstance(result, dict) or result.get("status") != STATUS_COMPLETED:
                proposals.append(result)
                continue
            proposal = PROPOSAL_BY_ID.get(str(result.get("proposalId") or ""))
            if proposal is None:
                proposals.append(result)
                continue
            recomputed = self._recompute_proposal_metrics(proposal, result, config)
            proposals.append(recomputed)
            recomputed_ids.append(proposal.proposal_id)

        elapsed = time.perf_counter() - started
        updated = {
            **run,
            "proposals": proposals,
            "metricSchemaVersion": METRIC_SCHEMA_VERSION,
            "metricCoordinateSpace": METRIC_COORDINATE_SPACE,
            "metricsRecomputedAt": utc_now(),
            "metricRecompute": {
                "recomputedAt": utc_now(),
                "wallClockSeconds": elapsed,
                "wallClockLabel": label_from_seconds(elapsed),
                "proposalIds": recomputed_ids,
                "note": "Recalculo desde artefactos Python reales; no reejecuta propuestas ni LLM.",
            },
            "updatedAt": utc_now(),
        }
        self._apply_contribution_metrics_unlocked(updated)
        logs = list(updated.get("logs") or [])
        log_entry = {
            "at": utc_now(),
            "proposalId": "system",
            "message": (
                "Metric recomputation completed using comparable normalized objectives "
                f"for {', '.join(recomputed_ids) or 'no proposals'}."
            ),
        }
        logs.append(log_entry)
        self._append_log_entry_to_file(updated, log_entry)
        updated["logs"] = logs[-COMPARATOR_RUN_LOG_LIMIT:]
        return self._with_metric_recompute_status(updated)

    def _recompute_proposal_metrics(
        self,
        proposal: ProposalDefinition,
        result: dict[str, Any],
        config: dict[str, Any],
    ) -> dict[str, Any]:
        repetitions = result.get("repetitions") if isinstance(result.get("repetitions"), list) else []
        if repetitions:
            recomputed_repetitions: list[dict[str, Any]] = []
            for repetition in repetitions:
                if isinstance(repetition, dict) and repetition.get("status") == STATUS_COMPLETED:
                    recomputed_repetitions.append(
                        self._recompute_single_proposal_result(proposal, repetition, config)
                    )
                elif isinstance(repetition, dict):
                    recomputed_repetitions.append(repetition)
            aggregated = aggregate_proposal_repetitions(
                proposal,
                Path(str(result.get("outputDir") or ".")),
                recomputed_repetitions,
                int(result.get("repetitionsK") or config.get("repetitionsK") or 1),
                self._identity_from_result(proposal, result),
            )
            for key in ("gitRevision", "command", "outputFiles"):
                if result.get(key) is not None:
                    aggregated[key] = result.get(key)
            return aggregated

        return self._recompute_single_proposal_result(proposal, result, config)

    def _identity_from_result(self, proposal: ProposalDefinition, result: dict[str, Any]) -> dict[str, Any]:
        return {
            "instanceId": str(result.get("instanceId") or result.get("proposalId") or proposal.proposal_id),
            "proposalId": proposal.proposal_id,
            "displayName": str(result.get("displayName") or proposal.display_name),
            "baseDisplayName": str(result.get("baseDisplayName") or proposal.display_name),
            "proposalConfig": result.get("proposalConfig") if isinstance(result.get("proposalConfig"), dict) else {"extraArgs": "", "cliValues": {}},
            "runtimeConfig": result.get("runtimeConfig") if isinstance(result.get("runtimeConfig"), dict) else {},
            "n": result.get("n"),
            "repetitionsK": result.get("repetitionsK"),
        }

    def _instance_from_result(self, proposal: ProposalDefinition, result: dict[str, Any]) -> ProposalRunInstance:
        identity = self._identity_from_result(proposal, result)
        runtime_config = identity.get("runtimeConfig") if isinstance(identity.get("runtimeConfig"), dict) else {}
        n = int(identity.get("n") or runtime_config.get("n") or COMPARATOR_DEFAULTS.get("n") or 1)
        repetitions_k = int(identity.get("repetitionsK") or runtime_config.get("repetitionsK") or COMPARATOR_DEFAULTS.get("repetitionsK") or 1)
        return ProposalRunInstance(
            instance_id=identity["instanceId"],
            proposal_id=proposal.proposal_id,
            display_name=identity["displayName"],
            base_display_name=identity["baseDisplayName"],
            proposal=proposal,
            proposal_config=identity["proposalConfig"],
            runtime_config=runtime_config,
            n=n,
            repetitions_k=repetitions_k,
            order_index=0,
        )

    def _recompute_single_proposal_result(
        self,
        proposal: ProposalDefinition,
        result: dict[str, Any],
        config: dict[str, Any],
    ) -> dict[str, Any]:
        output_dir = Path(str(result.get("outputDir") or ""))
        if not output_dir.exists():
            raise ValueError(f"No existe el directorio de salida para {proposal.display_name}: {output_dir}")
        result_path = output_dir / proposal.result_file
        if not result_path.exists():
            raise ValueError(f"No existe {proposal.result_file} para {proposal.display_name}: {result_path}")

        top_k = int(config.get("topK") or 10)
        reference_text = str(config.get("referenceText") or "")
        rows = self._normalize_rows(proposal, read_json(result_path), top_k, reference_text=reference_text)
        self._attach_instance_metadata(rows, self._instance_from_result(proposal, result))
        selected_rows, _selection_seconds = self._select_final_rows(
            proposal,
            rows,
            output_dir,
            write_if_missing=False,
        )
        self._attach_instance_metadata(selected_rows, self._instance_from_result(proposal, result))
        self._mark_selected_rows(rows, selected_rows)
        metrics = self._summarize_rows(proposal, rows, output_dir)
        series = self._build_metric_series(proposal, output_dir, rows, reference_text)
        attach_terminal_series_diagnostics(metrics, series)
        charts = self._build_chart_payload(proposal, rows, selected_rows, series)
        embedding_front_rows = embedding_front_rows_from_rows(rows, selected_rows)
        return {
            **result,
            "status": STATUS_COMPLETED,
            "rows": rows[:top_k],
            "selectedRows": selected_rows,
            "embeddingFrontRows": embedding_front_rows,
            "metrics": metrics,
            "series": series,
            "charts": charts,
            "outputFiles": self._output_files(proposal, output_dir),
            "metricSchemaVersion": METRIC_SCHEMA_VERSION,
            "metricCoordinateSpace": METRIC_COORDINATE_SPACE,
            "error": None,
        }

    def _read_config(self, payload: dict[str, Any]) -> dict[str, Any]:
        reference_text = str(payload.get("referenceText", "")).strip()
        if not reference_text:
            raise ValueError("referenceText is required.")

        top_k = self._int_between(payload.get("topK", COMPARATOR_DEFAULTS.get("topK", 10)), "topK", 1, 200)
        n = self._int_between(payload.get("n", COMPARATOR_DEFAULTS.get("n", 10)), "n", 1, 500)
        generaciones = self._int_between(payload.get("generaciones", COMPARATOR_DEFAULTS.get("generaciones", 3)), "generaciones", 0, 500)
        seed = self._int_between(payload.get("seed", COMPARATOR_DEFAULTS.get("seed", 42)), "seed", 0, 2_147_483_647)
        repetitions_k = self._int_between(
            payload.get("repetitionsK", COMPARATOR_DEFAULTS.get("repetitionsK", 1)),
            "repetitionsK",
            1,
            30,
        )
        model = self._safe_model_name(payload.get("model", COMPARATOR_DEFAULTS.get("model", "llama3")))

        raw_instances = payload.get("proposalInstances")
        if raw_instances is not None:
            proposal_instances = self._proposal_instances_config(raw_instances, n, repetitions_k)
            selected = self._selected_ids_from_instances(proposal_instances)
            proposal_configs = self._legacy_proposal_configs_from_instances(proposal_instances)
        else:
            selected = self._selected_proposal_ids(payload.get("selectedProposalIds"))
            proposal_configs = self._proposal_configs(payload.get("proposalConfigs"), selected)
            proposal_instances = self._legacy_proposal_instances_config(selected, proposal_configs)

        execution_mode = self._execution_mode(payload.get("executionMode", DEFAULT_EXECUTION_MODE))
        requested_parallelism = self._int_between(
            payload.get("proposalParallelism", COMPARATOR_DEFAULTS.get("proposalParallelism", 1)),
            "proposalParallelism",
            1,
            8,
        )
        effective_parallelism = 1 if execution_mode == EXECUTION_MODE_FAIR_SEQUENTIAL else requested_parallelism
        costs_comparable = execution_mode == EXECUTION_MODE_FAIR_SEQUENTIAL
        config = {
            "referenceText": reference_text,
            "topK": top_k,
            "n": n,
            "generaciones": generaciones,
            "seed": seed,
            "repetitionsK": repetitions_k,
            "model": model,
            "executionMode": execution_mode,
            "proposalParallelism": requested_parallelism,
            "effectiveProposalParallelism": effective_parallelism,
            "costsComparable": costs_comparable,
            "executionPolicy": {
                "mode": execution_mode,
                "requestedParallelism": requested_parallelism,
                "effectiveParallelism": effective_parallelism,
                "costsComparable": costs_comparable,
                "label": "Comparacion justa secuencial" if costs_comparable else "Exploratoria paralela",
                "note": (
                    "Costos comparables: las propuestas se ejecutan una por una."
                    if costs_comparable
                    else "Costos no comparables: las propuestas comparten Ollama/CPU/GPU."
                ),
            },
            "timeoutMinutes": self._int_between(
                payload.get("timeoutMinutes", COMPARATOR_DEFAULTS.get("timeoutMinutes", COMPARATOR_TIMEOUT_MINUTES_MIN)),
                "timeoutMinutes",
                COMPARATOR_TIMEOUT_MINUTES_MIN,
                COMPARATOR_TIMEOUT_MINUTES_MAX,
            ),
            "selectedProposalIds": selected,
            "proposalConfigs": proposal_configs,
            "proposalInstances": proposal_instances,
            "sameInitialPopulationForBmopso": self._same_initial_population_config(
                payload.get("sameInitialPopulationForBmopso"),
                proposal_instances,
                n,
                repetitions_k,
            ),
            "updateRepositoriesBeforeRun": self._bool_config_value(
                payload.get("updateRepositoriesBeforeRun", DEFAULT_UPDATE_REPOSITORIES_BEFORE_RUN),
                "updateRepositoriesBeforeRun",
            ),
            "proposalGitConfigs": self._proposal_git_configs(payload.get("proposalGitConfigs"), selected),
        }
        self._validate_binary_task_thinking_config(config)
        return config

    def _same_initial_population_config(
        self,
        value: Any,
        proposal_instances: list[dict[str, Any]],
        default_n: int,
        default_repetitions: int,
    ) -> dict[str, Any]:
        disabled = {"enabled": False, "generatorInstanceId": None, "scope": SAME_INITIAL_POPULATION_SCOPE}
        if value in (None, ""):
            return disabled
        if not isinstance(value, dict):
            raise ValueError("sameInitialPopulationForBmopso must be an object.")
        enabled = self._bool_config_value(value.get("enabled", False), "sameInitialPopulationForBmopso.enabled")
        scope = str(value.get("scope") or SAME_INITIAL_POPULATION_SCOPE).strip()
        if scope != SAME_INITIAL_POPULATION_SCOPE:
            raise ValueError(f"sameInitialPopulationForBmopso.scope must be {SAME_INITIAL_POPULATION_SCOPE}.")
        if not enabled:
            return disabled
        generator_id = str(value.get("generatorInstanceId") or "").strip()
        if not generator_id:
            raise ValueError("sameInitialPopulationForBmopso.generatorInstanceId is required when enabled.")
        instances_by_id = {
            str(instance.get("instanceId") or ""): instance
            for instance in proposal_instances
            if isinstance(instance, dict)
        }
        generator = instances_by_id.get(generator_id)
        if generator is None:
            raise ValueError("sameInitialPopulationForBmopso.generatorInstanceId must reference an existing instance.")
        if str(generator.get("proposalId") or "") != BINARY_PROPOSAL_ID:
            raise ValueError("sameInitialPopulationForBmopso.generatorInstanceId must reference a Binary MOPSO-CD instance.")
        self._validate_same_initial_population_runtime(proposal_instances, default_n, default_repetitions)
        return {"enabled": True, "generatorInstanceId": generator_id, "scope": SAME_INITIAL_POPULATION_SCOPE}

    def _validate_same_initial_population_runtime(
        self,
        proposal_instances: list[dict[str, Any]],
        default_n: int,
        default_repetitions: int,
    ) -> None:
        binary_instances = [
            instance
            for instance in proposal_instances
            if isinstance(instance, dict) and str(instance.get("proposalId") or "") == BINARY_PROPOSAL_ID
        ]
        if len(binary_instances) <= 1:
            return
        first_runtime = self._effective_runtime_tuple(binary_instances[0], default_n, default_repetitions)
        for instance in binary_instances[1:]:
            if self._effective_runtime_tuple(instance, default_n, default_repetitions) != first_runtime:
                raise ValueError(
                    "sameInitialPopulationForBmopso requires Binary MOPSO-CD instances to use the same N and K repeticiones."
                )

    def _effective_runtime_tuple(
        self,
        instance: dict[str, Any],
        default_n: int,
        default_repetitions: int,
    ) -> tuple[int, int]:
        runtime_config = instance.get("runtimeConfig") if isinstance(instance.get("runtimeConfig"), dict) else {}
        n = int(runtime_config.get("n") or default_n)
        repetitions = int(runtime_config.get("repetitionsK") or default_repetitions)
        return n, repetitions

    def _validate_binary_task_thinking_config(self, config: dict[str, Any]) -> None:
        capabilities = comparator_ollama_model_capabilities()
        default_task_models = comparator_binary_default_task_models()
        common_model = str(config.get("model") or COMPARATOR_DEFAULTS.get("model") or "llama3").strip()
        for instance in config.get("proposalInstances") or []:
            if str(instance.get("proposalId") or "") != BINARY_PROPOSAL_ID:
                continue
            proposal_config = instance.get("proposalConfig") if isinstance(instance.get("proposalConfig"), dict) else {}
            cli_values = proposal_config.get("cliValues") if isinstance(proposal_config.get("cliValues"), dict) else {}
            for key, thinking_value in cli_values.items():
                thinking_path = str(key)
                if not thinking_path.startswith(BINARY_TASK_THINKING_PREFIX):
                    continue
                if not self._binary_task_thinking_enabled(thinking_value):
                    continue
                task_name = thinking_path.removeprefix(BINARY_TASK_THINKING_PREFIX)
                task_model = str(
                    cli_values.get(f"{BINARY_TASK_MODEL_PREFIX}{task_name}")
                    or default_task_models.get(task_name)
                    or common_model
                ).strip()
                self._validate_binary_task_thinking_capability(task_model, task_name, capabilities)

    def _binary_task_thinking_enabled(self, value: Any) -> bool:
        if value in ("", None):
            return False
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        return text not in {"false", "0", "no"}

    def _validate_binary_task_thinking_capability(
        self,
        model: str,
        task_name: str,
        capabilities: dict[str, dict[str, Any]],
    ) -> None:
        model_capabilities = capabilities.get(model)
        if not isinstance(model_capabilities, dict):
            raise ValueError(
                f"{model} for {task_name} is not declared in Binary ollama.model_capabilities."
            )
        if model_capabilities.get("thinking") is not True:
            raise ValueError(f"{model} for {task_name} does not support thinking in Binary.")

    def _execution_mode(self, value: Any) -> str:
        mode = str(value or EXECUTION_MODE_FAIR_SEQUENTIAL).strip()
        if mode not in SUPPORTED_EXECUTION_MODES:
            raise ValueError(f"executionMode must be one of: {', '.join(sorted(SUPPORTED_EXECUTION_MODES))}.")
        return mode

    def _selected_proposal_ids(self, value: Any) -> list[str]:
        if value is None:
            return list(DEFAULT_SELECTED_PROPOSALS)
        if not isinstance(value, list):
            raise ValueError("selectedProposalIds must be a list.")
        selected: list[str] = []
        for item in value:
            proposal_id = str(item or "").strip()
            if not proposal_id:
                continue
            if proposal_id not in PROPOSAL_BY_ID:
                raise ValueError(f"Unknown proposalId: {proposal_id}.")
            if proposal_id not in selected:
                selected.append(proposal_id)
        if not selected:
            raise ValueError("Select at least one proposal.")
        return selected

    def _selected_ids_from_instances(self, instances: list[dict[str, Any]]) -> list[str]:
        selected: list[str] = []
        for instance in instances:
            proposal_id = str(instance.get("proposalId") or "").strip()
            if proposal_id and proposal_id not in selected:
                selected.append(proposal_id)
        if not selected:
            raise ValueError("Select at least one proposal instance.")
        return selected

    def _proposal_instances_config(self, value: Any, default_n: int, default_repetitions: int) -> list[dict[str, Any]]:
        if not isinstance(value, list):
            raise ValueError("proposalInstances must be a list.")
        if not value:
            raise ValueError("Agrega al menos una instancia de propuesta.")

        result: list[dict[str, Any]] = []
        seen_instance_ids: set[str] = set()
        per_proposal_counts: dict[str, int] = {}
        canonical_by_proposal: dict[str, dict[str, str]] = {}

        for index, raw in enumerate(value):
            if not isinstance(raw, dict):
                raise ValueError(f"proposalInstances[{index}] must be an object.")
            proposal_id = str(raw.get("proposalId") or "").strip()
            if proposal_id not in PROPOSAL_BY_ID:
                raise ValueError(f"Unknown proposalId: {proposal_id}.")
            proposal = PROPOSAL_BY_ID[proposal_id]
            per_proposal_counts[proposal_id] = per_proposal_counts.get(proposal_id, 0) + 1
            instance_id = self._safe_instance_id(
                raw.get("instanceId"),
                proposal_id,
                per_proposal_counts[proposal_id],
                f"proposalInstances[{index}].instanceId",
            )
            if instance_id in seen_instance_ids:
                raise ValueError(f"proposalInstances contains duplicate instanceId: {instance_id}.")
            seen_instance_ids.add(instance_id)

            display_name = str(raw.get("displayName") or "").strip()
            if not display_name:
                display_name = f"{proposal.display_name} - config {per_proposal_counts[proposal_id]}"
            proposal_config = self._proposal_config_value(
                proposal,
                raw.get("proposalConfig") if "proposalConfig" in raw else raw.get("config"),
                f"proposalInstances[{index}].proposalConfig",
            )
            runtime_config = self._runtime_config_value(
                raw.get("runtimeConfig"),
                default_n,
                default_repetitions,
                f"proposalInstances[{index}].runtimeConfig",
            )
            canonical = self._canonical_instance_config(proposal, proposal_config, runtime_config)
            existing = canonical_by_proposal.setdefault(proposal_id, {})
            if canonical in existing:
                first_name = existing[canonical]
                raise ValueError(
                    "duplicateProposalInstances: No se puede ejecutar porque estas instancias "
                    f"no difieren en configuracion: {first_name} y {display_name}."
                )
            existing[canonical] = display_name

            result.append(
                {
                    "instanceId": instance_id,
                    "proposalId": proposal_id,
                    "displayName": display_name,
                    "baseDisplayName": proposal.display_name,
                    "proposalConfig": proposal_config,
                    "runtimeConfig": runtime_config,
                    "orderIndex": index,
                }
            )
        return result

    def _legacy_proposal_instances_config(
        self,
        selected: list[str],
        proposal_configs: dict[str, dict[str, Any]],
    ) -> list[dict[str, Any]]:
        return [
            {
                "instanceId": proposal_id,
                "proposalId": proposal_id,
                "displayName": PROPOSAL_BY_ID[proposal_id].display_name,
                "baseDisplayName": PROPOSAL_BY_ID[proposal_id].display_name,
                "proposalConfig": proposal_configs.get(proposal_id, {"extraArgs": "", "cliValues": {}}),
                "runtimeConfig": {},
                "orderIndex": index,
            }
            for index, proposal_id in enumerate(selected)
        ]

    def _legacy_proposal_configs_from_instances(self, instances: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        configs: dict[str, dict[str, Any]] = {}
        for instance in instances:
            proposal_id = str(instance.get("proposalId") or "")
            configs.setdefault(proposal_id, instance.get("proposalConfig") or {"extraArgs": "", "cliValues": {}})
        return configs

    def _safe_instance_id(self, value: Any, proposal_id: str, index: int, label: str) -> str:
        instance_id = str(value or "").strip()
        if not instance_id:
            instance_id = f"{proposal_id}-{index}"
        if not INSTANCE_ID_RE.fullmatch(instance_id):
            raise ValueError(f"{label} has unsupported characters. Use letters, numbers, dot, underscore or hyphen.")
        return instance_id

    def _proposal_configs(self, value: Any, selected: list[str]) -> dict[str, dict[str, Any]]:
        if value is None:
            value = {}
        if not isinstance(value, dict):
            raise ValueError("proposalConfigs must be an object.")
        result: dict[str, dict[str, Any]] = {}
        for proposal_id in selected:
            proposal = PROPOSAL_BY_ID[proposal_id]
            result[proposal_id] = self._proposal_config_value(
                proposal,
                value.get(proposal_id),
                f"proposalConfigs.{proposal_id}",
            )
        return result

    def _proposal_config_value(self, proposal: ProposalDefinition, raw: Any, label: str) -> dict[str, Any]:
        raw = raw or {}
        if not isinstance(raw, dict):
            raise ValueError(f"{label} must be an object.")
        extra_args = raw.get("extraArgs", "")
        if extra_args is None:
            extra_args = ""
        if not isinstance(extra_args, str):
            raise ValueError(f"{label}.extraArgs must be text.")
        cli_values = raw.get("cliValues", {})
        if cli_values is None:
            cli_values = {}
        if not isinstance(cli_values, dict):
            raise ValueError(f"{label}.cliValues must be an object.")
        return {
            "extraArgs": extra_args.strip(),
            "cliValues": self._normalize_cli_values(proposal, cli_values),
        }

    def _canonical_proposal_config(self, proposal: ProposalDefinition, proposal_config: dict[str, Any]) -> str:
        cli_values = self._canonical_cli_values_for_duplicates(proposal, proposal_config.get("cliValues") or {})
        canonical = {
            "extraArgs": str(proposal_config.get("extraArgs") or "").strip(),
            "cliValues": cli_values,
        }
        return json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

    def _canonical_instance_config(
        self,
        proposal: ProposalDefinition,
        proposal_config: dict[str, Any],
        runtime_config: dict[str, Any],
    ) -> str:
        return json.dumps(
            {
                "proposalConfig": json.loads(self._canonical_proposal_config(proposal, proposal_config)),
                "runtimeConfig": runtime_config,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )

    def _runtime_config_value(
        self,
        raw: Any,
        default_n: int,
        default_repetitions: int,
        label: str,
    ) -> dict[str, int]:
        if raw in (None, ""):
            return {}
        if not isinstance(raw, dict):
            raise ValueError(f"{label} must be an object.")
        runtime_config: dict[str, int] = {}
        if "n" in raw and raw.get("n") not in (None, ""):
            n = self._int_between(raw.get("n"), f"{label}.n", 1, 500)
            if n != default_n:
                runtime_config["n"] = n
        if "repetitionsK" in raw and raw.get("repetitionsK") not in (None, ""):
            repetitions = self._int_between(raw.get("repetitionsK"), f"{label}.repetitionsK", 1, 30)
            if repetitions != default_repetitions:
                runtime_config["repetitionsK"] = repetitions
        return runtime_config

    def _canonical_cli_values_for_duplicates(self, proposal: ProposalDefinition, values: dict[str, Any]) -> dict[str, Any]:
        options = self._configurable_cli_options(proposal)
        effective: dict[str, Any] = {}
        for key, value in values.items():
            option = options.get(str(key))
            if option is None:
                effective[str(key)] = value
                continue
            if "default" in option and self._cli_value_matches_default(value, option.get("default")):
                continue
            effective[str(key)] = value
        return effective

    def _cli_value_matches_default(self, value: Any, default: Any) -> bool:
        if isinstance(default, list):
            if isinstance(value, list):
                return [str(item) for item in value] == [str(item) for item in default]
            try:
                import yaml

                parsed = yaml.safe_load(str(value))
            except Exception:
                parsed = None
            if isinstance(parsed, list):
                return [str(item) for item in parsed] == [str(item) for item in default]
            return False
        if isinstance(default, bool):
            return isinstance(value, bool) and value is default
        if isinstance(default, (int, float)) and not isinstance(default, bool):
            try:
                return float(value) == float(default)
            except (TypeError, ValueError):
                return False
        return str(value).strip() == str(default).strip()

    def _proposal_git_configs(self, value: Any, selected: list[str]) -> dict[str, dict[str, str]]:
        if value is None:
            value = {}
        if not isinstance(value, dict):
            raise ValueError("proposalGitConfigs must be an object.")

        result: dict[str, dict[str, str]] = {}
        for proposal_id in selected:
            proposal = PROPOSAL_BY_ID[proposal_id]
            raw = value.get(proposal_id) or {}
            if not isinstance(raw, dict):
                raise ValueError(f"proposalGitConfigs.{proposal_id} must be an object.")
            defaults = self._default_git_config(proposal)
            remote = self._safe_git_remote(raw.get("remote", defaults["remote"]), f"proposalGitConfigs.{proposal_id}.remote")
            branch = self._safe_git_branch(raw.get("branch", defaults["branch"]), f"proposalGitConfigs.{proposal_id}.branch")
            pull_mode = str(raw.get("pullMode", defaults["pullMode"]) or "").strip()
            if pull_mode not in SUPPORTED_GIT_PULL_MODES:
                raise ValueError(
                    f"proposalGitConfigs.{proposal_id}.pullMode must be one of: "
                    + ", ".join(sorted(SUPPORTED_GIT_PULL_MODES))
                )
            result[proposal_id] = {
                "remote": remote,
                "branch": branch,
                "pullMode": pull_mode,
                "expectedRemoteUrl": str(raw.get("expectedRemoteUrl", defaults.get("expectedRemoteUrl") or "") or "").strip(),
            }
        return result

    def _default_git_config(self, proposal: ProposalDefinition) -> dict[str, str]:
        return {
            "remote": proposal.git_remote,
            "branch": proposal.git_branch,
            "pullMode": proposal.git_pull_mode,
            "expectedRemoteUrl": proposal.git_expected_remote_url,
        }

    def _bool_config_value(self, value: Any, label: str) -> bool:
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in {"true", "1", "yes", "si"}:
            return True
        if text in {"false", "0", "no"}:
            return False
        raise ValueError(f"{label} must be true or false.")

    def _safe_git_remote(self, value: Any, label: str) -> str:
        remote = str(value or "").strip()
        if not remote:
            raise ValueError(f"{label} is required.")
        if not GIT_REMOTE_RE.fullmatch(remote):
            raise ValueError(f"{label} has unsupported characters.")
        return remote

    def _safe_git_branch(self, value: Any, label: str) -> str:
        branch = str(value or "").strip()
        if not branch:
            raise ValueError(f"{label} is required.")
        invalid = (
            not GIT_BRANCH_RE.fullmatch(branch)
            or ".." in branch
            or "@{" in branch
            or "\\" in branch
            or branch.startswith("/")
            or branch.endswith("/")
            or branch.endswith(".")
        )
        if invalid:
            raise ValueError(f"{label} is not a safe branch name.")
        return branch

    def _configurable_cli_options(self, proposal: ProposalDefinition) -> dict[str, dict[str, Any]]:
        return {
            self._cli_option_key(option): option
            for option in self._proposal_cli_options(proposal)
            if option.get("source") not in {"managed", "common"}
        }

    def _cli_option_key(self, option: dict[str, Any]) -> str:
        return str(option.get("key") or option.get("configPath") or option.get("flag") or "")

    def _normalize_cli_values(self, proposal: ProposalDefinition, values: dict[str, Any]) -> dict[str, Any]:
        options = self._configurable_cli_options(proposal)
        normalized: dict[str, Any] = {}
        for key, raw_value in values.items():
            key_text = str(key)
            option = options.get(key_text)
            if option is None:
                raise ValueError(f"{proposal.display_name}: {key_text} is not configurable from proposalConfigs.")
            option_type = str(option.get("type") or "string")
            if option_type == "bool":
                if raw_value in ("", None) or (raw_value is False and not option.get("allowFalse")):
                    continue
                normalized[key_text] = self._bool_cli_value(raw_value, f"{proposal.display_name}.{key_text}")
            elif option_type == "thinking_mode":
                if raw_value in ("", None):
                    continue
                mode = self._thinking_mode_cli_value(raw_value, f"{proposal.display_name}.{key_text}")
                if mode is False and not option.get("allowFalse"):
                    continue
                normalized[key_text] = mode
            elif option_type in {"int", "float", "string", "path"}:
                text = str(raw_value or "").strip()
                if not text:
                    continue
                if option_type == "int":
                    self._int_cli_value(text, f"{proposal.display_name}.{key_text}", option)
                elif option_type == "float":
                    self._float_cli_value(text, f"{proposal.display_name}.{key_text}", option)
                self._validate_cli_choice(option, text, f"{proposal.display_name}.{key_text}")
                normalized[key_text] = text
            elif option_type == "yaml":
                text = str(raw_value or "").strip()
                if text:
                    normalized[key_text] = text
            elif option_type in {"multi_select", "component_multi_select", "ordered_multi_select"}:
                selected = self._normalize_cli_list_value(proposal, option, raw_value, key_text)
                if selected or option_type == "ordered_multi_select":
                    normalized[key_text] = selected
            elif option_type == "repeatable":
                if not isinstance(raw_value, list):
                    raise ValueError(f"{proposal.display_name}.{key_text} must be a list.")
                selected = [str(item).strip() for item in raw_value if str(item).strip()]
                if selected:
                    normalized[key_text] = selected
            elif option_type == "repeatable_assignment":
                normalized_assignments = self._normalize_assignment_values(proposal, option, raw_value, bool_values=False)
                if normalized_assignments:
                    normalized[key_text] = normalized_assignments
            elif option_type == "repeatable_assignment_bool":
                normalized_assignments = self._normalize_assignment_values(proposal, option, raw_value, bool_values=True)
                if normalized_assignments:
                    normalized[key_text] = normalized_assignments
            else:
                raise ValueError(f"{proposal.display_name}.{key_text} has unsupported option type: {option_type}.")
        return normalized

    def _normalize_cli_list_value(
        self,
        proposal: ProposalDefinition,
        option: dict[str, Any],
        raw_value: Any,
        key_text: str,
    ) -> list[str]:
        if isinstance(raw_value, list):
            raw_items = raw_value
        elif isinstance(raw_value, str):
            text = raw_value.strip()
            if not text:
                return []
            try:
                import yaml

                parsed = yaml.safe_load(text)
            except Exception:
                parsed = None
            raw_items = parsed if isinstance(parsed, list) else [item.strip() for item in text.split(",")]
        else:
            raise ValueError(f"{proposal.display_name}.{key_text} must be a list.")
        selected: list[str] = []
        seen: set[str] = set()
        for item in raw_items:
            text = str(item).strip()
            if not text or text in seen:
                continue
            self._validate_cli_choice(option, text, f"{proposal.display_name}.{key_text}")
            selected.append(text)
            seen.add(text)
        return selected

    def _normalize_assignment_values(
        self,
        proposal: ProposalDefinition,
        option: dict[str, Any],
        raw_value: Any,
        bool_values: bool,
    ) -> dict[str, Any]:
        flag = str(option.get("flag"))
        if not isinstance(raw_value, dict):
            raise ValueError(f"{proposal.display_name}.{flag} must be an object.")
        allowed = {
            str(item.get("name"))
            for item in option.get("assignments", [])
            if isinstance(item, dict) and item.get("name")
        }
        normalized: dict[str, Any] = {}
        for assignment_name, assignment_value in raw_value.items():
            name = str(assignment_name).strip()
            if not name:
                continue
            if allowed and name not in allowed:
                raise ValueError(f"{proposal.display_name}.{flag} has unknown assignment: {name}.")
            if bool_values:
                if assignment_value in ("", None):
                    continue
                normalized[name] = self._bool_cli_value(assignment_value, f"{proposal.display_name}.{flag}.{name}")
            else:
                text = str(assignment_value or "").strip()
                if text:
                    normalized[name] = text
        return normalized

    def _bool_cli_value(self, value: Any, label: str) -> bool:
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in {"true", "1", "yes", "si", "sí"}:
            return True
        if text in {"false", "0", "no"}:
            return False
        raise ValueError(f"{label} must be true or false.")

    def _thinking_mode_cli_value(self, value: Any, label: str) -> bool | str:
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in {"true", "1", "yes", "si", "sÃ­"}:
            return True
        if text in {"false", "0", "no"}:
            return False
        if text in BINARY_THINKING_MODE_CHOICES and text != "false":
            return text
        raise ValueError(f"{label} must be false, low, medium, high, true or false.")

    def _int_cli_value(self, value: Any, label: str, option: dict[str, Any]) -> int:
        try:
            number = int(value)
        except (TypeError, ValueError):
            raise ValueError(f"{label} must be an integer.") from None
        minimum = option.get("min")
        maximum = option.get("max")
        if minimum is not None and number < int(minimum):
            raise ValueError(f"{label} must be at least {minimum}.")
        if maximum is not None and number > int(maximum):
            raise ValueError(f"{label} must be at most {maximum}.")
        return number

    def _float_cli_value(self, value: Any, label: str, option: dict[str, Any]) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{label} must be numeric.") from None
        if not math.isfinite(number):
            raise ValueError(f"{label} must be finite.")
        minimum = option.get("min")
        maximum = option.get("max")
        if minimum is not None and number < float(minimum):
            raise ValueError(f"{label} must be at least {minimum}.")
        if maximum is not None and number > float(maximum):
            raise ValueError(f"{label} must be at most {maximum}.")
        return number

    def _validate_cli_choice(self, option: dict[str, Any], value: str, label: str) -> None:
        choices = option.get("choices")
        if not choices or option.get("allowCustom"):
            return
        allowed = {str(item) for item in choices}
        if value not in allowed:
            raise ValueError(f"{label} must be one of: {', '.join(sorted(allowed))}.")

    def _int_between(self, value: Any, label: str, minimum: int, maximum: int) -> int:
        try:
            number = int(value)
        except (TypeError, ValueError):
            raise ValueError(f"{label} must be an integer.") from None
        if number < minimum or number > maximum:
            raise ValueError(f"{label} must be between {minimum} and {maximum}.")
        return number

    def _float_between(self, value: Any, label: str, minimum: float, maximum: float) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{label} must be numeric.") from None
        if not math.isfinite(number) or number < minimum or number > maximum:
            raise ValueError(f"{label} must be between {minimum} and {maximum}.")
        return number

    def _safe_model_name(self, value: Any) -> str:
        model = str(value or "").strip()
        if not model:
            raise ValueError("model is required.")
        return model

    def _new_run_id(self) -> str:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        return f"{timestamp}-{uuid.uuid4().hex[:8]}"

    def _selected_proposals(self, config: dict[str, Any]) -> list[ProposalDefinition]:
        ids = config.get("selectedProposalIds") or list(DEFAULT_SELECTED_PROPOSALS)
        return [PROPOSAL_BY_ID[proposal_id] for proposal_id in ids]

    def _instance_runtime_values(self, raw_instance: dict[str, Any] | None, config: dict[str, Any] | None) -> tuple[dict[str, Any], int, int]:
        runtime_config = {}
        if isinstance(raw_instance, dict) and isinstance(raw_instance.get("runtimeConfig"), dict):
            runtime_config = dict(raw_instance.get("runtimeConfig") or {})
        config = config or {}
        n = int(runtime_config.get("n") or config.get("n") or COMPARATOR_DEFAULTS.get("n") or 1)
        repetitions_k = int(runtime_config.get("repetitionsK") or config.get("repetitionsK") or COMPARATOR_DEFAULTS.get("repetitionsK") or 1)
        return runtime_config, n, repetitions_k

    def _selected_instances(self, config: dict[str, Any]) -> list[ProposalRunInstance]:
        raw_instances = config.get("proposalInstances") or []
        if not raw_instances:
            proposal_configs = config.get("proposalConfigs") or {}
            return [
                self._proposal_instance_from_parts(
                    proposal=proposal,
                    instance_id=proposal.proposal_id,
                    display_name=proposal.display_name,
                    base_display_name=proposal.display_name,
                    proposal_config=proposal_configs.get(proposal.proposal_id, {"extraArgs": "", "cliValues": {}}),
                    runtime_config={},
                    config=config,
                    order_index=index,
                )
                for index, proposal in enumerate(self._selected_proposals(config))
            ]

        instances: list[ProposalRunInstance] = []
        for index, item in enumerate(raw_instances):
            if not isinstance(item, dict):
                continue
            proposal_id = str(item.get("proposalId") or "").strip()
            proposal = PROPOSAL_BY_ID.get(proposal_id)
            if proposal is None:
                continue
            instance_id = str(item.get("instanceId") or proposal_id).strip() or proposal_id
            display_name = str(item.get("displayName") or proposal.display_name).strip() or proposal.display_name
            runtime_config = item.get("runtimeConfig") if isinstance(item.get("runtimeConfig"), dict) else {}
            instances.append(
                self._proposal_instance_from_parts(
                    proposal=proposal,
                    instance_id=instance_id,
                    display_name=display_name,
                    base_display_name=str(item.get("baseDisplayName") or proposal.display_name),
                    proposal_config=item.get("proposalConfig") if isinstance(item.get("proposalConfig"), dict) else {"extraArgs": "", "cliValues": {}},
                    runtime_config=runtime_config,
                    config=config,
                    order_index=int(finite_float(item.get("orderIndex"), index)),
                )
            )
        return instances

    def _proposal_instance_from_parts(
        self,
        proposal: ProposalDefinition,
        instance_id: str,
        display_name: str,
        base_display_name: str,
        proposal_config: dict[str, Any],
        runtime_config: dict[str, Any],
        config: dict[str, Any] | None,
        order_index: int,
    ) -> ProposalRunInstance:
        runtime_config, n, repetitions_k = self._instance_runtime_values({"runtimeConfig": runtime_config}, config)
        return ProposalRunInstance(
            instance_id=instance_id,
            proposal_id=proposal.proposal_id,
            display_name=display_name,
            base_display_name=base_display_name,
            proposal=proposal,
            proposal_config=proposal_config,
            runtime_config=runtime_config,
            n=n,
            repetitions_k=repetitions_k,
            order_index=order_index,
        )

    def _coerce_instance(self, value: ProposalDefinition | ProposalRunInstance, config: dict[str, Any] | None = None) -> ProposalRunInstance:
        if isinstance(value, ProposalRunInstance):
            return value
        proposal = value
        proposal_config = ((config or {}).get("proposalConfigs") or {}).get(proposal.proposal_id, {"extraArgs": "", "cliValues": {}})
        return self._proposal_instance_from_parts(
            proposal=proposal,
            instance_id=proposal.proposal_id,
            display_name=proposal.display_name,
            base_display_name=proposal.display_name,
            proposal_config=proposal_config,
            runtime_config={},
            config=config,
            order_index=0,
        )

    def _result_identity(self, instance: ProposalRunInstance) -> dict[str, Any]:
        return {
            "instanceId": instance.instance_id,
            "proposalId": instance.proposal_id,
            "displayName": instance.display_name,
            "baseDisplayName": instance.base_display_name,
            "proposalConfig": instance.proposal_config,
            "runtimeConfig": instance.runtime_config,
            "n": instance.n,
        }

    def _attach_instance_metadata(self, rows: list[dict[str, Any]], instance: ProposalRunInstance) -> None:
        for row in rows:
            if not isinstance(row, dict):
                continue
            row["instanceId"] = instance.instance_id
            row["proposalId"] = instance.proposal_id
            row["displayName"] = instance.display_name
            row["baseDisplayName"] = instance.base_display_name

    def _validate_selected_proposals_available(self, config: dict[str, Any]) -> None:
        for proposal in self._selected_proposals(config):
            repository = resolve_repository(self.root, proposal)
            if not proposal_entrypoint_exists(self.root, proposal):
                raise ValueError(f"{proposal.display_name} is not available: missing entrypoint at {repository}.")
            dependency_status = check_proposal_dependencies(self.root, proposal)
            if not dependency_status.get("ok"):
                missing = ", ".join(dependency_status.get("missing") or [])
                detail = f"missing modules: {missing}" if missing else dependency_status.get("error") or "dependency check failed"
                raise ValueError(f"{proposal.display_name} is not available: {detail}.")

    def _run_git(self, repository: Path, args: list[str], timeout_seconds: int = 30) -> dict[str, Any]:
        command = ["git", "-C", str(repository), *args]
        started = time.perf_counter()
        try:
            completed = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_seconds,
                check=False,
                **hidden_subprocess_kwargs(),
            )
            return {
                "command": command_label(command),
                "returnCode": completed.returncode,
                "stdout": completed.stdout.strip(),
                "stderr": completed.stderr.strip(),
                "durationSeconds": time.perf_counter() - started,
            }
        except FileNotFoundError:
            return {
                "command": command_label(command),
                "returnCode": 127,
                "stdout": "",
                "stderr": "git executable was not found.",
                "durationSeconds": time.perf_counter() - started,
            }
        except subprocess.TimeoutExpired as error:
            return {
                "command": command_label(command),
                "returnCode": 124,
                "stdout": (error.stdout or "").strip() if isinstance(error.stdout, str) else "",
                "stderr": (error.stderr or "").strip() if isinstance(error.stderr, str) else "",
                "durationSeconds": time.perf_counter() - started,
            }

    def _git_stdout(self, repository: Path, args: list[str], timeout_seconds: int = 10) -> str | None:
        result = self._run_git(repository, args, timeout_seconds)
        if result["returnCode"] != 0:
            return None
        return str(result.get("stdout") or "").strip() or None

    def _repository_git_snapshot(self, repository: Path, git_config: dict[str, str]) -> dict[str, Any]:
        snapshot: dict[str, Any] = {
            "repositoryPath": str(repository),
            "remote": git_config.get("remote"),
            "configuredBranch": git_config.get("branch"),
            "isGit": False,
            "branch": None,
            "commit": None,
            "shortCommit": None,
            "dirty": False,
            "dirtyCount": 0,
            "upstream": None,
            "remoteUrl": None,
            "error": None,
        }
        if not repository.exists():
            snapshot["error"] = "Repository path does not exist."
            return snapshot

        inside = self._run_git(repository, ["rev-parse", "--is-inside-work-tree"], 10)
        if inside["returnCode"] != 0 or str(inside.get("stdout") or "").strip().lower() != "true":
            snapshot["error"] = inside.get("stderr") or "Path is not a Git work tree."
            return snapshot

        snapshot["isGit"] = True
        snapshot["branch"] = self._git_stdout(repository, ["rev-parse", "--abbrev-ref", "HEAD"], 10)
        snapshot["commit"] = self._git_stdout(repository, ["rev-parse", "HEAD"], 10)
        snapshot["shortCommit"] = self._git_stdout(repository, ["rev-parse", "--short", "HEAD"], 10)
        snapshot["upstream"] = self._git_stdout(repository, ["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"], 10)
        snapshot["remoteUrl"] = self._git_stdout(repository, ["remote", "get-url", str(git_config.get("remote") or "origin")], 10)
        status = self._run_git(repository, ["status", "--porcelain"], 10)
        if status["returnCode"] == 0:
            status_lines = [line for line in str(status.get("stdout") or "").splitlines() if line.strip()]
            snapshot["dirty"] = bool(status_lines)
            snapshot["dirtyCount"] = len(status_lines)
        else:
            snapshot["error"] = status.get("stderr") or "Could not read Git status."
        return snapshot

    def _repository_update_disabled(self, proposal: ProposalDefinition, git_config: dict[str, str]) -> dict[str, Any]:
        repository = resolve_repository(self.root, proposal)
        snapshot = self._repository_git_snapshot(repository, git_config)
        return {
            "proposalId": proposal.proposal_id,
            "displayName": proposal.display_name,
            "repositoryPath": str(repository),
            "remote": git_config["remote"],
            "branch": git_config["branch"],
            "pullMode": git_config["pullMode"],
            "expectedRemoteUrl": git_config.get("expectedRemoteUrl") or "",
            "status": "disabled",
            "message": "Repository update is disabled for this run.",
            "before": snapshot,
            "after": snapshot,
            "fetch": None,
            "pull": None,
            "durationSeconds": 0.0,
        }

    def _update_repository_for_proposal(
        self,
        proposal: ProposalDefinition,
        git_config: dict[str, str],
    ) -> dict[str, Any]:
        repository = resolve_repository(self.root, proposal)
        started = time.perf_counter()
        result: dict[str, Any] = {
            "proposalId": proposal.proposal_id,
            "displayName": proposal.display_name,
            "repositoryPath": str(repository),
            "remote": git_config["remote"],
            "branch": git_config["branch"],
            "pullMode": git_config["pullMode"],
            "expectedRemoteUrl": git_config.get("expectedRemoteUrl") or "",
            "status": "pending",
            "message": "",
            "before": None,
            "after": None,
            "fetch": None,
            "pull": None,
            "durationSeconds": 0.0,
        }

        before = self._repository_git_snapshot(repository, git_config)
        result["before"] = before
        if not before.get("isGit"):
            result["status"] = "skipped_not_git"
            result["message"] = before.get("error") or "Repository is not a Git work tree."
        elif git_config.get("expectedRemoteUrl") and normalized_git_url(before.get("remoteUrl")) != normalized_git_url(git_config.get("expectedRemoteUrl")):
            result["status"] = "skipped_remote_mismatch"
            result["message"] = (
                f"Skipped pull because {git_config['remote']} points to {before.get('remoteUrl') or 'unknown'}, "
                f"not {git_config.get('expectedRemoteUrl')}."
            )
        elif before.get("branch") != git_config["branch"]:
            result["status"] = "skipped_branch_mismatch"
            result["message"] = (
                f"Skipped pull because current branch is {before.get('branch') or 'unknown'}, "
                f"not {git_config['branch']}."
            )
        elif before.get("dirty"):
            result["status"] = "skipped_dirty"
            result["message"] = f"Skipped pull because the repository has {before.get('dirtyCount', 0)} local change(s)."
        else:
            fetch = self._run_git(repository, ["fetch", git_config["remote"], git_config["branch"]], 120)
            result["fetch"] = fetch
            if fetch["returnCode"] != 0:
                result["status"] = "fetch_failed"
                result["message"] = fetch.get("stderr") or "git fetch failed."
            else:
                pull = self._run_git(
                    repository,
                    ["pull", "--ff-only", git_config["remote"], git_config["branch"]],
                    120,
                )
                result["pull"] = pull
                if pull["returnCode"] != 0:
                    result["status"] = "pull_failed"
                    result["message"] = pull.get("stderr") or "git pull --ff-only failed."
                else:
                    output = f"{pull.get('stdout') or ''}\n{pull.get('stderr') or ''}".lower()
                    result["status"] = "up_to_date" if "already up to date" in output or "already up-to-date" in output else "pulled"
                    result["message"] = pull.get("stdout") or pull.get("stderr") or "Repository updated."

        result["after"] = self._repository_git_snapshot(repository, git_config)
        result["durationSeconds"] = time.perf_counter() - started
        return result

    def _prepare_repositories_before_run(self, run: dict[str, Any], proposals: list[ProposalDefinition]) -> None:
        should_update = bool(run["config"].get("updateRepositoriesBeforeRun"))
        git_configs = run["config"].get("proposalGitConfigs") or {}

        for proposal in proposals:
            instance_ids = self._state_instance_ids_for_proposal(run, proposal.proposal_id)
            with self._lock:
                if run.get("cancelRequested"):
                    self._append_log_unlocked(run, "system", "Run cancelled before repository preparation finished.")
                    self._write_summary_unlocked(run)
                    return
                for instance_id in instance_ids:
                    self._set_proposal_state_unlocked(
                        run,
                        instance_id,
                        STATUS_RUNNING,
                        "Actualizando repositorio" if should_update else "Registrando revision Git",
                        0.02,
                    )
                self._write_summary_unlocked(run)

            git_config = git_configs.get(proposal.proposal_id) or self._default_git_config(proposal)
            update = (
                self._update_repository_for_proposal(proposal, git_config)
                if should_update
                else self._repository_update_disabled(proposal, git_config)
            )
            status = update.get("status")
            short_commit = ((update.get("after") or {}).get("shortCommit") or (update.get("before") or {}).get("shortCommit") or "--")

            with self._lock:
                run.setdefault("repositoryUpdates", {})[proposal.proposal_id] = update
                self._append_log_unlocked(
                    run,
                    proposal.proposal_id,
                    f"Git {status}: {update.get('remote')}/{update.get('branch')} at {short_commit}. {update.get('message') or ''}",
                )
                for instance_id in instance_ids:
                    self._set_proposal_state_unlocked(run, instance_id, STATUS_QUEUED, "En cola", 0.0)
                self._write_summary_unlocked(run)

    def _state_instance_ids_for_proposal(self, run: dict[str, Any], proposal_id: str) -> list[str]:
        states = run.get("proposalStates") or {}
        ids = [
            str(instance_id)
            for instance_id, state in states.items()
            if str((state or {}).get("proposalId") or instance_id) == proposal_id
        ]
        return ids or [proposal_id]

    def _run_worker(self, run_id: str) -> None:
        with self._lock:
            run = self._runs[run_id]
            run["status"] = STATUS_RUNNING
            run["startedAtEpoch"] = time.time()
            run["updatedAt"] = utc_now()
            self._append_log_unlocked(run, "system", "Comparator run started.")
            self._write_summary_unlocked(run)

        try:
            selected_instances = self._selected_instances(run["config"])
            selected_proposals = self._selected_proposals(run["config"])
            self._prepare_repositories_before_run(run, selected_proposals)
            policy = run["config"].get("executionPolicy") or {}
            parallelism = min(run["config"].get("effectiveProposalParallelism", 1), len(selected_instances))
            with self._lock:
                self._append_log_unlocked(
                    run,
                    "system",
                    (
                        f"Execution mode {policy.get('mode', run['config'].get('executionMode'))}: "
                        f"requested parallelism={policy.get('requestedParallelism', run['config'].get('proposalParallelism'))}, "
                        f"effective parallelism={parallelism}, costsComparable={bool(policy.get('costsComparable'))}."
                    ),
                )
                self._write_summary_unlocked(run)
            if parallelism <= 1:
                self._run_proposals_sequential(run, selected_instances)
            else:
                self._run_proposals_parallel(run, selected_instances, parallelism)

            with self._lock:
                self._finish_run_execution_unlocked(run, len(selected_instances))
                self._write_summary_unlocked(run)
        except Exception as error:
            with self._lock:
                run["status"] = STATUS_FAILED
                run["error"] = str(error)
                run["updatedAt"] = utc_now()
                self._refresh_cost_summary_unlocked(run)
                self._append_log_unlocked(run, "system", f"Unexpected error: {error}")
                self._write_summary_unlocked(run)

    def _recontinue_worker(self, run_id: str, remaining_instances: list[ProposalRunInstance]) -> None:
        try:
            with self._lock:
                run = self._runs[run_id]
                parallelism = min(run["config"].get("effectiveProposalParallelism", 1), len(remaining_instances))

            if parallelism <= 1:
                self._run_proposals_sequential(run, remaining_instances)
            else:
                self._run_proposals_parallel(run, remaining_instances, parallelism)

            with self._lock:
                total_instances = len(self._selected_instances(run["config"]))
                self._finish_run_execution_unlocked(run, total_instances)
                self._write_summary_unlocked(run)
        except Exception as error:
            with self._lock:
                run = self._runs[run_id]
                run["status"] = STATUS_FAILED
                run["error"] = str(error)
                run["updatedAt"] = utc_now()
                self._refresh_cost_summary_unlocked(run)
                self._append_log_unlocked(run, "system", f"Unexpected error: {error}")
                self._write_summary_unlocked(run)

    def _finish_run_execution_unlocked(self, run: dict[str, Any], total_instances: int) -> None:
        self._sort_proposals_unlocked(run)
        self._apply_contribution_metrics_unlocked(run)
        failed = [item for item in run["proposals"] if item["status"] == STATUS_FAILED]
        if run.get("cancelRequested"):
            run["status"] = STATUS_CANCELLED
        elif failed and len(failed) == total_instances:
            run["status"] = STATUS_FAILED
            run["error"] = "All proposal executions failed."
        elif failed:
            run["status"] = STATUS_FAILED
            failed_names = ", ".join(item["displayName"] for item in failed)
            run["error"] = f"Proposal executions failed: {failed_names}."
        else:
            run["status"] = STATUS_COMPLETED
            run["error"] = None
        run["updatedAt"] = utc_now()
        self._refresh_run_progress_unlocked(run)
        self._refresh_cost_summary_unlocked(run)
        self._append_log_unlocked(run, "system", f"Comparator run finished with status {run['status']}.")

    def _run_proposals_sequential(self, run: dict[str, Any], proposals: list[ProposalDefinition | ProposalRunInstance]) -> None:
        with self._lock:
            self._refresh_same_initial_population_artifacts_unlocked(run)
        for proposal in self._same_initial_population_execution_order(run, proposals):
            with self._lock:
                if run["cancelRequested"]:
                    run["status"] = STATUS_CANCELLED
                    self._append_log_unlocked(run, "system", "Run cancelled before next proposal.")
                    self._write_summary_unlocked(run)
                    return

            result = self._execute_proposal(run, proposal)
            with self._lock:
                self._record_proposal_result_unlocked(run, result)
                self._write_summary_unlocked(run)

    def _run_proposals_parallel(self, run: dict[str, Any], proposals: list[ProposalDefinition | ProposalRunInstance], parallelism: int) -> None:
        with self._lock:
            self._refresh_same_initial_population_artifacts_unlocked(run)
        if self._same_initial_population_enabled(run):
            self._run_proposals_parallel_with_same_initial_population(run, proposals, parallelism)
            return
        self._run_proposals_parallel_unrestricted(run, proposals, parallelism)

    def _run_proposals_parallel_unrestricted(
        self,
        run: dict[str, Any],
        proposals: list[ProposalDefinition | ProposalRunInstance],
        parallelism: int,
    ) -> None:
        with ThreadPoolExecutor(max_workers=parallelism, thread_name_prefix="comparator") as executor:
            futures: dict[Future[dict[str, Any]], ProposalDefinition | ProposalRunInstance] = {
                executor.submit(self._execute_proposal, run, proposal): proposal
                for proposal in proposals
            }

            for future in as_completed(futures):
                result = self._result_from_proposal_future(run, futures[future], future)

                with self._lock:
                    self._record_proposal_result_unlocked(run, result)
                    if run["cancelRequested"]:
                        for pending in futures:
                            if not pending.done():
                                pending.cancel()
                        self._mark_queued_as_cancelled_unlocked(run)
                    self._write_summary_unlocked(run)

    def _run_proposals_parallel_with_same_initial_population(
        self,
        run: dict[str, Any],
        proposals: list[ProposalDefinition | ProposalRunInstance],
        parallelism: int,
    ) -> None:
        generator_id = self._same_initial_population_generator_id(run)
        if not generator_id:
            self._run_proposals_parallel_unrestricted(run, proposals, parallelism)
            return

        generator: ProposalDefinition | ProposalRunInstance | None = None
        dependents: list[ProposalDefinition | ProposalRunInstance] = []
        independent: list[ProposalDefinition | ProposalRunInstance] = []
        for proposal in proposals:
            instance = self._coerce_instance(proposal, run.get("config"))
            if instance.instance_id == generator_id:
                generator = proposal
            elif instance.proposal_id == BINARY_PROPOSAL_ID:
                dependents.append(proposal)
            else:
                independent.append(proposal)

        if generator is None:
            self._run_proposals_parallel_unrestricted(run, proposals, parallelism)
            return

        dependency_resolved = False
        with ThreadPoolExecutor(max_workers=parallelism, thread_name_prefix="comparator") as executor:
            futures: dict[Future[dict[str, Any]], ProposalDefinition | ProposalRunInstance] = {}

            def submit(proposal: ProposalDefinition | ProposalRunInstance) -> None:
                futures[executor.submit(self._execute_proposal, run, proposal)] = proposal

            submit(generator)
            for proposal in independent:
                submit(proposal)

            while futures:
                done, _pending = wait(futures, return_when=FIRST_COMPLETED)
                for future in done:
                    proposal = futures.pop(future)
                    result = self._result_from_proposal_future(run, proposal, future)
                    instance = self._coerce_instance(proposal, run.get("config"))
                    to_submit: list[ProposalDefinition | ProposalRunInstance] = []
                    blocked_dependents: list[dict[str, Any]] = []

                    with self._lock:
                        self._record_proposal_result_unlocked(run, result)
                        if instance.instance_id == generator_id and not dependency_resolved:
                            dependency_resolved = True
                            if result.get("status") == STATUS_COMPLETED and not run.get("cancelRequested"):
                                to_submit = list(dependents)
                            else:
                                blocked_dependents = [
                                    self._same_initial_population_blocked_result(run, dependent, result)
                                    for dependent in dependents
                                ]
                                for blocked in blocked_dependents:
                                    self._record_proposal_result_unlocked(run, blocked)
                        if run["cancelRequested"]:
                            for pending in futures:
                                if not pending.done():
                                    pending.cancel()
                            self._mark_queued_as_cancelled_unlocked(run)
                        self._write_summary_unlocked(run)

                    if to_submit:
                        for proposal_to_submit in to_submit:
                            submit(proposal_to_submit)

    def _result_from_proposal_future(
        self,
        run: dict[str, Any],
        proposal: ProposalDefinition | ProposalRunInstance,
        future: Future[dict[str, Any]],
    ) -> dict[str, Any]:
        instance = self._coerce_instance(proposal, run.get("config"))
        if future.cancelled():
            return self._cancelled_result(instance, Path(run["runDir"]) / instance.instance_id)
        try:
            return future.result()
        except Exception as error:
            return self._failed_result(
                instance,
                Path(run["runDir"]) / instance.instance_id,
                f"Unexpected proposal error: {error}",
            )

    def _same_initial_population_enabled(self, run: dict[str, Any]) -> bool:
        config = run.get("config") or {}
        value = config.get("sameInitialPopulationForBmopso")
        return isinstance(value, dict) and bool(value.get("enabled"))

    def _same_initial_population_generator_id(self, run: dict[str, Any]) -> str | None:
        if not self._same_initial_population_enabled(run):
            return None
        config = run.get("config") or {}
        value = config.get("sameInitialPopulationForBmopso")
        generator_id = str((value or {}).get("generatorInstanceId") or "").strip()
        return generator_id or None

    def _same_initial_population_execution_order(
        self,
        run: dict[str, Any],
        proposals: list[ProposalDefinition | ProposalRunInstance],
    ) -> list[ProposalDefinition | ProposalRunInstance]:
        generator_id = self._same_initial_population_generator_id(run)
        if not generator_id:
            return list(proposals)
        instances = [(proposal, self._coerce_instance(proposal, run.get("config"))) for proposal in proposals]
        generator = next((proposal for proposal, instance in instances if instance.instance_id == generator_id), None)
        if generator is None:
            return list(proposals)

        binary_group = [
            proposal
            for proposal, instance in instances
            if instance.proposal_id == BINARY_PROPOSAL_ID and instance.instance_id != generator_id
        ]
        if not binary_group:
            return list(proposals)

        ordered: list[ProposalDefinition | ProposalRunInstance] = []
        inserted_binary_group = False
        for proposal, instance in instances:
            if instance.proposal_id == BINARY_PROPOSAL_ID:
                if not inserted_binary_group:
                    ordered.append(generator)
                    ordered.extend(binary_group)
                    inserted_binary_group = True
                continue
            ordered.append(proposal)
        return ordered

    def _same_initial_population_blocked_result(
        self,
        run: dict[str, Any],
        proposal: ProposalDefinition | ProposalRunInstance,
        generator_result: dict[str, Any],
    ) -> dict[str, Any]:
        instance = self._coerce_instance(proposal, run.get("config"))
        status = generator_result.get("status")
        reason = "fue cancelada" if status == STATUS_CANCELLED else "fallo"
        return self._failed_result(
            instance,
            Path(run["runDir"]) / instance.instance_id,
            f"Misma poblacion inicial para MOPSO no disponible porque la instancia generadora {reason}.",
        )

    def _same_initial_population_artifacts_from_output_dir(self, output_dir: Path | str | None) -> dict[str, str] | None:
        if not output_dir:
            return None
        directory = Path(output_dir)
        population_path = directory / "data_initial_population.json"
        reference_context_path = directory / "reference_context.json"
        if not population_path.exists() or not reference_context_path.exists():
            return None
        return {
            "populationPath": str(population_path.resolve()),
            "referenceContextPath": str(reference_context_path.resolve()),
        }

    def _same_initial_population_artifacts_from_result(self, result: dict[str, Any]) -> dict[str, Any] | None:
        if result.get("proposalId") != BINARY_PROPOSAL_ID or result.get("status") != STATUS_COMPLETED:
            return None
        direct = self._same_initial_population_artifacts_from_output_dir(result.get("outputDir")) or {}
        repetitions: dict[str, dict[str, str]] = {}
        for index, repetition in enumerate(result.get("repetitions") or [], start=1):
            if not isinstance(repetition, dict) or repetition.get("status") != STATUS_COMPLETED:
                continue
            repetition_artifacts = self._same_initial_population_artifacts_from_output_dir(repetition.get("outputDir"))
            if not repetition_artifacts:
                continue
            repetition_key = str(int(repetition.get("repetitionIndex") or index))
            repetitions[repetition_key] = repetition_artifacts
        if repetitions and not direct:
            first_key = sorted(repetitions, key=lambda item: int(item))[0]
            direct = dict(repetitions[first_key])
        if not direct:
            return None
        return {**direct, "repetitions": repetitions}

    def _refresh_same_initial_population_artifacts_unlocked(self, run: dict[str, Any]) -> None:
        artifacts = dict(run.get("sameInitialPopulationArtifacts") or {})
        for result in run.get("proposals") or []:
            if not isinstance(result, dict):
                continue
            instance_id = str(result.get("instanceId") or result.get("proposalId") or "").strip()
            if not instance_id:
                continue
            result_artifacts = self._same_initial_population_artifacts_from_result(result)
            if result_artifacts:
                artifacts[instance_id] = result_artifacts
        run["sameInitialPopulationArtifacts"] = artifacts
        self._sync_same_initial_population_trace_unlocked(run)

    def _record_same_initial_population_artifacts_unlocked(self, run: dict[str, Any], result: dict[str, Any]) -> None:
        instance_id = str(result.get("instanceId") or result.get("proposalId") or "").strip()
        result_artifacts = self._same_initial_population_artifacts_from_result(result)
        if instance_id and result_artifacts:
            artifacts = run.setdefault("sameInitialPopulationArtifacts", {})
            artifacts[instance_id] = result_artifacts
        self._sync_same_initial_population_trace_unlocked(run)

    def _sync_same_initial_population_trace_unlocked(self, run: dict[str, Any]) -> None:
        config = run.get("config") or {}
        value = config.get("sameInitialPopulationForBmopso")
        if not isinstance(value, dict):
            value = {"enabled": False, "generatorInstanceId": None, "scope": SAME_INITIAL_POPULATION_SCOPE}
        run["sameInitialPopulationForBmopso"] = {
            "enabled": bool(value.get("enabled")),
            "generatorInstanceId": value.get("generatorInstanceId") if value.get("enabled") else None,
            "scope": str(value.get("scope") or SAME_INITIAL_POPULATION_SCOPE),
            "artifacts": run.get("sameInitialPopulationArtifacts") or {},
        }

    def _same_initial_population_repetition_key(self, output_base: Path) -> str | None:
        match = re.fullmatch(r"rep-(\d+)", output_base.parent.name)
        if not match:
            return None
        return str(int(match.group(1)))

    def _same_initial_population_paths_for_command(
        self,
        run: dict[str, Any],
        instance: ProposalRunInstance,
        output_base: Path,
    ) -> dict[str, str] | None:
        generator_id = self._same_initial_population_generator_id(run)
        if (
            not generator_id
            or instance.proposal_id != BINARY_PROPOSAL_ID
            or instance.instance_id == generator_id
        ):
            return None
        artifacts = (run.get("sameInitialPopulationArtifacts") or {}).get(generator_id)
        if not isinstance(artifacts, dict):
            artifacts = None
        repetition_key = self._same_initial_population_repetition_key(output_base)
        if repetition_key and artifacts:
            repetitions = artifacts.get("repetitions") if isinstance(artifacts.get("repetitions"), dict) else {}
            repetition_artifacts = repetitions.get(repetition_key)
            if isinstance(repetition_artifacts, dict):
                artifacts = repetition_artifacts
        if not artifacts:
            raise ValueError(
                "Misma poblacion inicial para MOPSO no disponible: "
                "la instancia generadora aun no produjo artefactos reutilizables."
            )
        raw_population_path = str(artifacts.get("populationPath") or "").strip()
        raw_reference_context_path = str(artifacts.get("referenceContextPath") or "").strip()
        if not raw_population_path or not raw_reference_context_path:
            raise ValueError(
                "Misma poblacion inicial para MOPSO no disponible: "
                "la instancia generadora no produjo data_initial_population.json y reference_context.json."
            )
        population_path = Path(raw_population_path)
        reference_context_path = Path(raw_reference_context_path)
        if not population_path.exists() or not reference_context_path.exists():
            raise ValueError(
                "Misma poblacion inicial para MOPSO no disponible: "
                "la instancia generadora no produjo data_initial_population.json y reference_context.json."
            )
        return {
            "populationPath": str(population_path.resolve()),
            "referenceContextPath": str(reference_context_path.resolve()),
        }

    def _same_initial_population_input_trace(
        self,
        run: dict[str, Any],
        instance: ProposalRunInstance,
        output_base: Path,
    ) -> dict[str, Any] | None:
        paths = self._same_initial_population_paths_for_command(run, instance, output_base)
        if not paths:
            return None
        trace: dict[str, Any] = {
            "generatorInstanceId": self._same_initial_population_generator_id(run),
            "scope": SAME_INITIAL_POPULATION_SCOPE,
            **paths,
        }
        repetition_key = self._same_initial_population_repetition_key(output_base)
        if repetition_key:
            trace["repetitionIndex"] = int(repetition_key)
        return trace

    def _record_proposal_result_unlocked(self, run: dict[str, Any], result: dict[str, Any]) -> None:
        repository_update = (run.get("repositoryUpdates") or {}).get(result["proposalId"])
        if repository_update:
            result["repositoryUpdate"] = repository_update
            snapshot = repository_update.get("after") or repository_update.get("before") or {}
            result["gitRevision"] = {
                "branch": snapshot.get("branch"),
                "commit": snapshot.get("commit"),
                "shortCommit": snapshot.get("shortCommit"),
                "dirty": snapshot.get("dirty"),
                "dirtyCount": snapshot.get("dirtyCount"),
                "remote": repository_update.get("remote"),
                "configuredBranch": repository_update.get("branch"),
                "status": repository_update.get("status"),
            }
        self._record_same_initial_population_artifacts_unlocked(run, result)
        run["proposals"].append(result)
        state_key = result.get("instanceId") or result["proposalId"]
        state = run["proposalStates"].setdefault(
            state_key,
            {
                "instanceId": state_key,
                "proposalId": result["proposalId"],
                "displayName": result.get("displayName", state_key),
                "baseDisplayName": result.get("baseDisplayName", result["proposalId"]),
            },
        )
        state["status"] = result["status"]
        state["stageLabel"] = comparator_status_message(result["status"])
        state["progress"] = 1.0
        if isinstance(result.get("runtimeConfig"), dict):
            state["runtimeConfig"] = result.get("runtimeConfig") or {}
        if result.get("n") is not None:
            state["n"] = int(result.get("n"))
        total_repetitions = max(1, int(result.get("repetitionsK") or run.get("config", {}).get("repetitionsK") or 1))
        raw_completed_repetitions = result.get("completedRepetitions")
        if raw_completed_repetitions is None:
            completed_repetitions = 1 if result.get("status") == STATUS_COMPLETED else 0
        else:
            completed_repetitions = int(raw_completed_repetitions)
        completed_repetitions = max(0, min(total_repetitions, completed_repetitions))
        state["repetitionsK"] = total_repetitions
        state["totalRepetitions"] = total_repetitions
        state["completedRepetitions"] = completed_repetitions
        state["currentRepetitionIndex"] = total_repetitions
        state["updatedAt"] = utc_now()
        run["updatedAt"] = utc_now()
        self._refresh_run_progress_unlocked(run)
        self._refresh_cost_summary_unlocked(run)

    def _sort_proposals_unlocked(self, run: dict[str, Any]) -> None:
        instances = run.get("config", {}).get("proposalInstances") or []
        instance_order = {
            str(instance.get("instanceId")): index
            for index, instance in enumerate(instances)
            if isinstance(instance, dict) and instance.get("instanceId")
        }
        proposal_order = {proposal.proposal_id: index for index, proposal in enumerate(PROPOSALS)}
        run["proposals"].sort(
            key=lambda item: (
                instance_order.get(str(item.get("instanceId") or ""), len(instance_order)),
                proposal_order.get(item.get("proposalId"), len(proposal_order)),
            )
        )

    def _apply_contribution_metrics_unlocked(self, run: dict[str, Any]) -> None:
        completed = [
            proposal
            for proposal in run.get("proposals") or []
            if isinstance(proposal, dict) and proposal.get("status") == STATUS_COMPLETED
        ]
        points_by_proposal = {
            self._proposal_entity_id(proposal): self._proposal_non_dominated_points(proposal)
            for proposal in completed
        }
        contributions = calculate_contribution(points_by_proposal)
        for proposal in completed:
            entity_id = self._proposal_entity_id(proposal)
            contribution = contributions.get(entity_id, 0.0)
            metrics = proposal.setdefault("metrics", {})
            metrics["contribution"] = contribution
            metrics["contributionLabel"] = f"{contribution:.6f}"
        self._apply_repetition_contribution_std_devs(completed)
        self._apply_series_contribution_metrics(completed)

    def _proposal_entity_id(self, proposal: dict[str, Any]) -> str:
        return str(proposal.get("instanceId") or proposal.get("proposalId") or "")

    def _proposal_non_dominated_points(self, proposal: dict[str, Any]) -> list[tuple[float, float]]:
        return self._non_dominated_points_from_charts(proposal.get("charts") or {})

    def _non_dominated_points_from_charts(self, charts: dict[str, Any]) -> list[tuple[float, float]]:
        points: list[tuple[float, float]] = []
        if not isinstance(charts, dict):
            return points
        for point in (charts.get("nonDominated") or []):
            if not isinstance(point, dict):
                continue
            x_value = finite_float(point.get("x"), None)
            y_value = finite_float(point.get("y"), None)
            if x_value is None or y_value is None:
                continue
            points.append((x_value, y_value))
        return points

    def _apply_repetition_contribution_std_devs(self, proposals: list[dict[str, Any]]) -> None:
        points_by_repetition: dict[int, dict[str, list[tuple[float, float]]]] = {}
        for proposal in proposals:
            entity_id = self._proposal_entity_id(proposal)
            if not entity_id:
                continue
            for payload in proposal_point_chart_repetitions(proposal):
                repetition_index = finite_int_or_none(payload.get("repetitionIndex"))
                if repetition_index is None:
                    continue
                charts = payload.get("charts") if isinstance(payload.get("charts"), dict) else {}
                points_by_repetition.setdefault(repetition_index, {})[entity_id] = self._non_dominated_points_from_charts(charts)

        values_by_entity: dict[str, list[float]] = {
            self._proposal_entity_id(proposal): []
            for proposal in proposals
            if self._proposal_entity_id(proposal)
        }
        for points_by_proposal in points_by_repetition.values():
            contributions = calculate_contribution(points_by_proposal)
            for entity_id, contribution in contributions.items():
                values_by_entity.setdefault(entity_id, []).append(contribution)

        for proposal in proposals:
            entity_id = self._proposal_entity_id(proposal)
            metrics = proposal.setdefault("metrics", {})
            apply_std_dev_field(metrics, "contribution", values_by_entity.get(entity_id, []))

    def _apply_series_contribution_metrics(self, proposals: list[dict[str, Any]]) -> None:
        generations = sorted(
            {
                int(finite_float(point.get("generation"), -1))
                for proposal in proposals
                for point in proposal.get("series") or []
                if isinstance(point, dict) and finite_float(point.get("generation"), -1) >= 0
            }
        )
        for generation in generations:
            points_by_proposal: dict[str, list[tuple[float, float]]] = {}
            series_by_proposal: dict[str, dict[str, Any]] = {}
            for proposal in proposals:
                entity_id = self._proposal_entity_id(proposal)
                point = next(
                    (
                        item
                        for item in proposal.get("series") or []
                        if isinstance(item, dict) and int(finite_float(item.get("generation"), -1)) == generation
                    ),
                    None,
                )
                if point is None:
                    continue
                front_points = []
                for raw_point in point.get("frontPoints") or []:
                    if not isinstance(raw_point, list) or len(raw_point) < 2:
                        continue
                    x_value = finite_float(raw_point[0], None)
                    y_value = finite_float(raw_point[1], None)
                    if x_value is None or y_value is None:
                        continue
                    front_points.append((x_value, y_value))
                if front_points:
                    points_by_proposal[entity_id] = front_points
                    series_by_proposal[entity_id] = point
            if not points_by_proposal:
                continue
            contributions = calculate_contribution(points_by_proposal)
            for entity_id, point in series_by_proposal.items():
                point["contribution"] = contributions.get(entity_id, 0.0)

    def _mark_queued_as_cancelled_unlocked(self, run: dict[str, Any]) -> None:
        for state in (run.get("proposalStates") or {}).values():
            if state.get("status") == STATUS_QUEUED:
                state["status"] = STATUS_CANCELLED
                state["stageLabel"] = "Cancelada antes de iniciar"
                state["progress"] = 1.0
                state["updatedAt"] = utc_now()
        self._refresh_run_progress_unlocked(run)

    def _refresh_cost_summary_unlocked(self, run: dict[str, Any]) -> None:
        started_at = run.get("startedAtEpoch")
        elapsed = max(0.0, time.time() - started_at) if started_at else 0.0
        run["costSummary"] = summarize_costs(run.get("proposals") or [], elapsed)

    def _execute_proposal(self, run: dict[str, Any], proposal: ProposalDefinition | ProposalRunInstance) -> dict[str, Any]:
        instance = self._coerce_instance(proposal, run.get("config"))
        base_proposal = instance.proposal
        repetitions_k = int(instance.repetitions_k or run["config"].get("repetitionsK") or 1)
        base_dir = Path(run["runDir"]) / instance.instance_id
        with self._lock:
            state = run["proposalStates"].get(instance.instance_id)
            if state:
                state["runtimeConfig"] = instance.runtime_config
                state["n"] = instance.n
                state["repetitionsK"] = repetitions_k
                state["totalRepetitions"] = repetitions_k
                state["completedRepetitions"] = 0
                state["currentRepetitionIndex"] = 1 if repetitions_k > 0 else None
                state["updatedAt"] = utc_now()
        if repetitions_k <= 1:
            result = self._execute_proposal_once(
                run,
                instance,
                base_dir,
                self._repetition_seed(run["config"].get("seed"), 0),
            )
            result = {**self._result_identity(instance), **result}
            result["completedRepetitions"] = 1 if result.get("status") == STATUS_COMPLETED else 0
            result["repetitionsK"] = 1
            with self._lock:
                state = run["proposalStates"].get(instance.instance_id)
                if state:
                    state["totalRepetitions"] = 1
                    state["repetitionsK"] = 1
                    state["completedRepetitions"] = int(result["completedRepetitions"])
                    state["currentRepetitionIndex"] = 1
                    state["updatedAt"] = utc_now()
            return result

        base_dir.mkdir(parents=True, exist_ok=True)
        results: list[dict[str, Any]] = []
        for repetition_index in range(repetitions_k):
            with self._lock:
                if run.get("cancelRequested"):
                    break
                self._set_proposal_state_unlocked(
                    run,
                    instance.instance_id,
                    STATUS_RUNNING,
                    f"Repeticion {repetition_index + 1}/{repetitions_k}",
                    repetition_index / repetitions_k,
                )
                state = run["proposalStates"].get(instance.instance_id)
                if state:
                    state["currentRepetitionIndex"] = repetition_index + 1
                    state["completedRepetitions"] = completed_repetition_count(results)
                    state["totalRepetitions"] = repetitions_k
                    state["repetitionsK"] = repetitions_k
                    timing = self._iteration_timing(state)
                    timing["activeIterationKey"] = None
                    timing["activeStartedAtEpoch"] = None
                    timing["activeElapsedSeconds"] = None
                    self._refresh_iteration_counts(run, state)
                    self._refresh_run_progress_unlocked(run)

            repetition_seed = self._repetition_seed(run["config"].get("seed"), repetition_index)
            repetition_dir = base_dir / f"rep-{repetition_index + 1:03d}"
            result = self._execute_proposal_once(run, instance, repetition_dir, repetition_seed)
            result["repetitionIndex"] = repetition_index + 1
            result["repetitionSeed"] = repetition_seed
            results.append(result)
            write_json(repetition_dir / "summary.json", result)
            with self._lock:
                state = run["proposalStates"].get(instance.instance_id)
                if state:
                    state["completedRepetitions"] = completed_repetition_count(results)
                    state["currentRepetitionIndex"] = min(len(results) + 1, repetitions_k)
                    state["totalRepetitions"] = repetitions_k
                    state["repetitionsK"] = repetitions_k
                    state["updatedAt"] = utc_now()

        aggregated = aggregate_proposal_repetitions(base_proposal, base_dir, results, repetitions_k, self._result_identity(instance))
        return aggregated

    def _build_command(
        self,
        run: dict[str, Any],
        proposal: ProposalDefinition | ProposalRunInstance,
        repository_dir: Path,
        output_base: Path,
        reference_path: Path,
        random_seed: int | None,
    ) -> list[str]:
        instance = self._coerce_instance(proposal, run.get("config"))
        base_proposal = instance.proposal
        proposal_config_values = instance.proposal_config or {}
        structured_args = self._cli_args_from_values(base_proposal, proposal_config_values.get("cliValues") or {})
        extra_args = split_cli_args(proposal_config_values.get("extraArgs", ""))
        self._validate_extra_args(base_proposal, extra_args)
        if base_proposal.kind == "binary-mopso-cd":
            command = [
                proposal_python_executable(self.root, repository_dir, base_proposal),
                "-m",
                "binary_mopso_cd",
                "--reference-text",
                run["config"]["referenceText"],
            ]
            return (
                command
                + self._binary_managed_set_args(
                    run,
                    instance,
                    output_base,
                    random_seed,
                    proposal_config_values.get("cliValues") or {},
                    extra_args,
                )
                + structured_args
                + extra_args
            )

        if base_proposal.kind == "mesap":
            command = [
                proposal_python_executable(self.root, repository_dir, base_proposal),
                str(self.root / "baselines" / "bootstrap.py"),
                base_proposal.entrypoint,
                "--n",
                str(instance.n),
                "--generations",
                str(run["config"]["generaciones"]),
                "--model",
                run["config"]["model"],
                "--outdir_base",
                str(output_base),
                "--reference_text",
                str(reference_path),
            ]
            return command + structured_args + extra_args

        command = [
            proposal_python_executable(self.root, repository_dir, base_proposal),
            str(self.root / "baselines" / "bootstrap.py"),
            base_proposal.entrypoint,
            "--n",
            str(instance.n),
            "--generaciones",
            str(run["config"]["generaciones"]),
            "--model",
            run["config"]["model"],
            "--outdir-base",
            str(output_base),
            "--texto-referencia",
            str(reference_path),
        ]
        return command + structured_args + extra_args

    def _binary_managed_set_args(
        self,
        run: dict[str, Any],
        instance: ProposalRunInstance,
        output_base: Path,
        random_seed: int | None,
        cli_values: dict[str, Any] | None = None,
        extra_args: list[str] | None = None,
    ) -> list[str]:
        model = str(run["config"]["model"])
        seed = int(random_seed if random_seed is not None else run["config"]["seed"])
        manual_paths = {str(path) for path in (cli_values or {})}
        manual_paths.update(self._binary_set_paths_from_args(extra_args or []))
        overrides: list[tuple[str, Any, str]] = [
            ("experiment.n", int(instance.n), "int"),
            ("experiment.iterations", int(run["config"]["generaciones"]), "int"),
            ("experiment.runs", 1, "int"),
            ("experiment.seed", seed, "int"),
            ("runtime.outdir_base", str(output_base.resolve()), "path"),
            ("ollama.default_model", model, "string"),
        ]
        overrides.extend(
            binary_alternative_task_model_overrides(
                (cli_values or {}).get("ollama.alternative_model"),
                manual_paths,
            )
        )
        if "ollama.timeout_seconds" not in manual_paths:
            overrides.append(("ollama.timeout_seconds", BINARY_DEFAULT_OLLAMA_TIMEOUT_SECONDS, "int"))
        overrides.extend(binary_portal_ppdb_overrides(self.root, manual_paths))
        for path in BINARY_AUTO_PARALLELISM_PATHS:
            if path not in manual_paths:
                overrides.append((path, int(instance.n), "int"))
        same_initial_paths = self._same_initial_population_paths_for_command(run, instance, output_base)
        if same_initial_paths:
            overrides.extend([
                ("initialization.population_input_path", same_initial_paths["populationPath"], "path"),
                ("initialization.reference_context_input_path", same_initial_paths["referenceContextPath"], "path"),
            ])
        args: list[str] = []
        for path, value, value_type in overrides:
            args.extend(["--set", f"{path}={self._yaml_cli_literal(value, value_type)}"])
        return args

    def _cli_args_from_values(self, proposal: ProposalDefinition, values: dict[str, Any]) -> list[str]:
        options = self._configurable_cli_options(proposal)
        args: list[str] = []
        for key, value in values.items():
            option = options.get(key)
            if option is None:
                raise ValueError(f"{proposal.display_name}: {key} is not configurable.")
            flag = str(option.get("flag") or key)
            option_type = str(option.get("type") or "string")
            if flag == "--set":
                config_path = str(option.get("configPath") or key)
                args.extend(["--set", f"{config_path}={self._yaml_cli_literal(value, option_type)}"])
            elif option_type == "bool":
                if value:
                    args.append(flag)
            elif option_type in {"int", "float", "string", "path"}:
                args.extend([flag, str(value)])
            elif option_type == "yaml":
                args.extend([flag, str(value)])
            elif option_type in {"multi_select", "component_multi_select", "ordered_multi_select"}:
                selected = [str(item).strip() for item in value if str(item).strip()]
                if selected:
                    args.extend([flag, ",".join(selected)])
            elif option_type == "repeatable":
                for item in value:
                    text = str(item).strip()
                    if text:
                        args.extend([flag, text])
            elif option_type == "repeatable_assignment":
                for name, assignment_value in value.items():
                    text = str(assignment_value).strip()
                    if text:
                        args.extend([flag, f"{name}={text}"])
            elif option_type == "repeatable_assignment_bool":
                for name, assignment_value in value.items():
                    args.extend([flag, f"{name}={str(bool(assignment_value)).lower()}"])
            else:
                raise ValueError(f"{proposal.display_name}: unsupported option type for {flag}: {option_type}.")
        return args

    def _yaml_cli_literal(self, value: Any, option_type: str) -> str:
        if option_type == "bool":
            return "true" if self._bool_cli_value(value, "yaml bool") else "false"
        if option_type == "thinking_mode":
            mode = self._thinking_mode_cli_value(value, "yaml thinking")
            if isinstance(mode, bool):
                return "true" if mode else "false"
            return json.dumps(mode, ensure_ascii=False)
        if option_type in {"int", "float"}:
            return str(value)
        if option_type in {"yaml", "component_multi_select", "ordered_multi_select", "multi_select"}:
            if isinstance(value, (list, dict)):
                return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
            return str(value).strip()
        return json.dumps(str(value), ensure_ascii=False)

    def _validate_extra_args(self, proposal: ProposalDefinition, extra_args: list[str]) -> None:
        managed_flags = {
            str(option["flag"])
            for option in self._proposal_cli_options(proposal)
            if option.get("source") in {"managed", "common"}
        }
        if proposal.kind == "binary-mopso-cd":
            managed_flags.discard("--set")
        managed_flags.update({"--outdir-base", "--texto-referencia", "--reference-text", "--seed", "--runs"})
        blocked = [
            item
            for item in extra_args
            if any(item == flag or item.startswith(f"{flag}=") for flag in managed_flags)
        ]
        if blocked:
            raise ValueError(
                f"{proposal.display_name}: these CLI flags are managed by the comparator and cannot be overridden: "
                + ", ".join(blocked)
            )
        if proposal.kind == "binary-mopso-cd":
            removed = [
                item
                for item in extra_args
                if any(item == flag or item.startswith(f"{flag}=") for flag in BINARY_REMOVED_CLI_FLAGS)
            ]
            if removed:
                raise ValueError(
                    f"{proposal.display_name}: these CLI flags were removed by the new Binary CLI. "
                    "Use the structured default.yaml controls instead: " + ", ".join(removed)
                )
            self._validate_binary_manual_set_args(extra_args)

    def _validate_binary_manual_set_args(self, extra_args: list[str]) -> None:
        blocked_paths: list[str] = []
        for path in self._binary_set_paths_from_args(extra_args):
            if (
                path in BINARY_MANAGED_CONFIG_PATHS
                or path.startswith(BINARY_TASK_MODEL_PREFIX)
                or path.startswith(BINARY_TASK_THINKING_PREFIX)
            ):
                blocked_paths.append(path)
        if blocked_paths:
            raise ValueError(
                "Binary MOPSO-CD: these YAML paths are managed by the comparator or by the structured UI: "
                + ", ".join(blocked_paths)
            )

    def _binary_set_paths_from_args(self, extra_args: list[str]) -> set[str]:
        paths: set[str] = set()
        index = 0
        while index < len(extra_args):
            item = extra_args[index]
            target = ""
            if item == "--set":
                index += 1
                if index >= len(extra_args):
                    raise ValueError("Binary MOPSO-CD: --set requires path=value.")
                target = extra_args[index]
            elif item.startswith("--set="):
                target = item[len("--set="):]
            if target:
                path = target.split("=", 1)[0].strip()
                if not path or "=" not in target:
                    raise ValueError("Binary MOPSO-CD: --set requires path=value.")
                paths.add(path)
            index += 1
        return paths

    def _execute_proposal_once(
        self,
        run: dict[str, Any],
        proposal: ProposalDefinition | ProposalRunInstance,
        proposal_dir: Path,
        random_seed: int | None,
    ) -> dict[str, Any]:
        instance = self._coerce_instance(proposal, run.get("config"))
        base_proposal = instance.proposal
        with self._lock:
            if run.get("cancelRequested"):
                return self._cancelled_result(instance, proposal_dir)
            self._set_proposal_state_unlocked(
                run,
                instance.instance_id,
                STATUS_RUNNING,
                "Preparando directorio de salida",
                0.01,
            )

        proposal_dir.mkdir(parents=True, exist_ok=True)
        reference_path = proposal_dir / "reference.txt"
        reference_path.write_text(run["config"]["referenceText"], encoding="utf-8")
        output_base = proposal_dir / "exec"
        output_base.mkdir(parents=True, exist_ok=True)
        cost_metrics_path = proposal_dir / "cost_metrics.json"

        repository_dir = resolve_repository(self.root, base_proposal)
        if not proposal_entrypoint_exists(self.root, base_proposal):
            return self._failed_result(
                instance,
                proposal_dir,
                f"Proposal entrypoint is missing at {repository_dir}.",
            )

        try:
            same_initial_input = self._same_initial_population_input_trace(run, instance, output_base)
            command = self._build_command(run, instance, repository_dir, output_base, reference_path, random_seed)
        except ValueError as error:
            return self._failed_result(instance, proposal_dir, str(error))

        self._append_log(run, instance.instance_id, "Starting baseline process.")
        self._append_log(run, instance.instance_id, command_label(command))
        with self._lock:
            self._set_proposal_state_unlocked(
                run,
                instance.instance_id,
                STATUS_RUNNING,
                "Proceso Python iniciado",
                0.05,
            )
            self._write_summary_unlocked(run)
        process_cost = self._run_process(
            run,
            instance.instance_id,
            command,
            repository_dir,
            base_proposal.preload_modules,
            base_proposal.python_path_entries,
            cost_metrics_path,
            random_seed,
            export_history=base_proposal.supports_history_export,
        )
        return_code = int(process_cost["returnCode"])
        llm_payload = read_json_or_default(cost_metrics_path, {})

        if run.get("cancelRequested"):
            cost = self._build_cost(base_proposal, process_cost, llm_payload, None, True)
            return self._cancelled_result(instance, proposal_dir, cost)
        if return_code != 0:
            output_dir = latest_child_directory(output_base)
            message = self._process_failure_message(
                run,
                instance.instance_id,
                return_code,
                bool(process_cost.get("timedOut")),
            )
            cost = self._build_cost(base_proposal, process_cost, llm_payload, output_dir, False)
            return self._failed_result(
                instance,
                output_dir or proposal_dir,
                message,
                cost,
            )

        output_dir = latest_child_directory(output_base)
        if not output_dir:
            cost = self._build_cost(base_proposal, process_cost, llm_payload, None, False)
            return self._failed_result(instance, proposal_dir, "No output directory was created.", cost)

        result_path = output_dir / base_proposal.result_file
        if not result_path.exists():
            cost = self._build_cost(base_proposal, process_cost, llm_payload, output_dir, False)
            return self._failed_result(
                instance,
                proposal_dir,
                f"Expected result file was not found: {result_path.name}.",
                cost,
            )

        try:
            extraction_started = time.perf_counter()
            rows = self._normalize_rows(
                base_proposal,
                read_json(result_path),
                run["config"]["topK"],
                reference_text=run["config"]["referenceText"],
            )
            self._attach_instance_metadata(rows, instance)
            cost = self._build_cost(base_proposal, process_cost, llm_payload, output_dir, False)
            add_cost_timing(cost, "metricExtractionSeconds", time.perf_counter() - extraction_started)

            selected_rows, selection_seconds = self._select_final_rows(base_proposal, rows, output_dir)
            self._attach_instance_metadata(selected_rows, instance)
            if selection_seconds:
                add_cost_timing(cost, "postProcessingWallClockSeconds", selection_seconds)

            metrics_started = time.perf_counter()
            self._mark_selected_rows(rows, selected_rows)
            metrics = self._summarize_rows(base_proposal, rows, output_dir)
            series = self._build_metric_series(base_proposal, output_dir, rows, run["config"]["referenceText"])
            attach_terminal_series_diagnostics(metrics, series)
            add_cost_timing(cost, "metricExtractionSeconds", time.perf_counter() - metrics_started)

            plot_started = time.perf_counter()
            charts = self._build_chart_payload(base_proposal, rows, selected_rows, series)
            add_cost_timing(cost, "plotPreparationSeconds", time.perf_counter() - plot_started)
            embedding_front_rows = embedding_front_rows_from_rows(rows, selected_rows)
            result = {
                **self._result_identity(instance),
                "status": STATUS_COMPLETED,
                "outputDir": str(output_dir),
                "rows": rows[: run["config"]["topK"]],
                "selectedRows": selected_rows,
                "embeddingFrontRows": embedding_front_rows,
                "metrics": metrics,
                "series": series,
                "charts": charts,
                "cost": cost,
                "command": command_label(command),
                "outputFiles": self._output_files(base_proposal, output_dir),
                "error": None,
            }
            if same_initial_input:
                result["sameInitialPopulationInput"] = same_initial_input
            same_initial_artifacts = self._same_initial_population_artifacts_from_output_dir(output_dir)
            if same_initial_artifacts:
                result["sameInitialPopulationArtifacts"] = same_initial_artifacts
            return result
        except Exception as error:
            cost = self._build_cost(base_proposal, process_cost, llm_payload, output_dir, False)
            return self._failed_result(instance, proposal_dir, f"Could not normalize output: {error}", cost)

    def _build_cost(
        self,
        proposal: ProposalDefinition,
        process_cost: dict[str, Any],
        llm_payload: dict[str, Any],
        output_dir: Path | None,
        cancelled: bool,
    ) -> dict[str, Any]:
        if proposal.kind == "binary-mopso-cd":
            return build_binary_cost_metrics(process_cost, output_dir, cancelled)
        return build_cost_metrics(process_cost, llm_payload, output_dir, cancelled)

    def _process_failure_message(
        self,
        run: dict[str, Any],
        instance_id: str,
        return_code: int,
        timed_out: bool,
    ) -> str:
        if timed_out or return_code == 124:
            base = f"Process timed out after {run['config']['timeoutMinutes']} minute(s)."
        elif return_code < 0:
            signum = abs(return_code)
            try:
                signal_label = signal.Signals(signum).name
            except ValueError:
                signal_label = f"signal {signum}"
            base = f"Process terminated by signal {signum} ({signal_label})."
        else:
            base = f"Process exited with code {return_code}."
        detail = self._latest_failure_log_detail(run, instance_id)
        if detail:
            return f"{base} {detail}"
        if return_code == -signal.SIGILL:
            return (
                f"{base} A native Python dependency likely used CPU instructions unsupported by this server node. "
                "Run scripts/diagnose_native_runtime.py to identify the failing runtime."
            )
        if not timed_out and return_code != 124:
            return f"{base} Check dependencies, Ollama, and model availability."
        return base

    def _latest_failure_log_detail(self, run: dict[str, Any], instance_id: str) -> str | None:
        with self._lock:
            messages = [
                str(entry.get("message") or "")
                for entry in run.get("logs", [])
                if entry.get("proposalId") == instance_id
            ]
        fallback: str | None = None
        for message in reversed(messages):
            detail = self._log_detail_text(message)
            if not detail:
                continue
            if PYTHON_EXCEPTION_LOG_RE.match(detail):
                return detail
            lowered = detail.lower()
            if fallback is None and any(marker in lowered for marker in ("error", "exception", "failed")):
                fallback = detail
        return fallback

    def _log_detail_text(self, message: str) -> str:
        text = clean_log_message(message)
        if not text:
            return ""
        match = TIMESTAMPED_LOG_RE.match(text)
        if match:
            text = match.group(1)
        return text.strip()

    def _select_final_rows(
        self,
        proposal: ProposalDefinition,
        rows: list[dict[str, Any]],
        output_dir: Path,
        write_if_missing: bool = True,
    ) -> tuple[list[dict[str, Any]], float]:
        started = time.perf_counter()
        selected_from_file = self._read_selected_rows(proposal, output_dir)
        if selected_from_file:
            selected = self._normalize_selected_rows(proposal, selected_from_file, rows)
            return selected, 0.0

        if proposal.kind in {"evolmd", "mesap", "evolmd-mo"}:
            selected = self._entropy_topsis_mmr_selection(rows, proposal.single_objective)
            if write_if_missing:
                write_json(output_dir / "comparator_final_selection.json", selected)
            return selected, time.perf_counter() - started

        selected = rows[:5]
        return selected, 0.0

    def _read_selected_rows(self, proposal: ProposalDefinition, output_dir: Path) -> list[dict[str, Any]]:
        candidates = []
        if proposal.final_selection_file:
            candidates.append(output_dir / proposal.final_selection_file)
        candidates.extend([
            output_dir / "final_selection_hybrid.json",
            output_dir / "pareto_ranked.json",
            output_dir / "comparator_final_selection.json",
        ])
        for path in candidates:
            payload = read_json_or_default(path, None)
            if isinstance(payload, list) and payload:
                return [item for item in payload if isinstance(item, dict)]
        return []

    def _normalize_selected_rows(
        self,
        proposal: ProposalDefinition,
        selected: list[dict[str, Any]],
        rows: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        by_text = {canonical_generated_text(row.get("generatedText")): row for row in rows}
        normalized: list[dict[str, Any]] = []
        for index, item in enumerate(selected, start=1):
            key = canonical_generated_text(item.get("generatedText") or item.get("generated_text") or item.get("generated_data"))
            row = by_text.get(key)
            if row:
                normalized.append(self._selected_projection(row, index, item.get("topsis_score")))
                continue
            raw_row = self._normalize_any_row(proposal, item, index)
            normalized.append(self._selected_projection(raw_row, index, item.get("topsis_score")))
        return normalized[:5]

    def _entropy_topsis_mmr_selection(self, rows: list[dict[str, Any]], use_diagnostic: bool) -> list[dict[str, Any]]:
        candidates = [
            row
            for row in rows
            if row.get("status") == "ok"
            and row.get("generatedText")
            and (row.get("diagnosticObjectiveVector") if use_diagnostic else row.get("objectiveVector"))
        ]
        if not candidates:
            return []
        vectors = [
            row.get("comparableObjectiveVector")
            or (row["diagnosticObjectiveVector"] if use_diagnostic else row["objectiveVector"])
            for row in candidates
        ]
        matrix = [[finite_float(vector[0]), finite_float(vector[1] if len(vector) > 1 else 0.0)] for vector in vectors]
        weights = entropy_weights(matrix)
        scores = topsis_scores(matrix, weights)
        for source_index, (row, score) in enumerate(zip(candidates, scores)):
            row["_topsis_score"] = score
            row["_selection_source_index"] = source_index
        candidates.sort(key=lambda row: row["_topsis_score"], reverse=True)
        selected: list[dict[str, Any]] = []
        embeddings = None
        if len(candidates) > 1:
            texts = [row.get("generatedText") or "" for row in candidates]
            embeddings, _ = shared_sbert_service().encode_texts(POSTHOC_EMBEDDING_MODEL, texts)
        while candidates and len(selected) < 5:
            if not selected or embeddings is None:
                chosen = candidates.pop(0)
            else:
                best_index = 0
                best_score = -float("inf")
                selected_indices = [row["_selection_source_index"] for row in selected]
                for idx, row in enumerate(candidates):
                    source_idx = row["_selection_source_index"]
                    redundancy = max(float(embeddings[source_idx] @ embeddings[item_idx]) for item_idx in selected_indices)
                    mmr = 0.35 * finite_float(row.get("_topsis_score")) - 0.65 * max(0.0, redundancy)
                    if mmr > best_score:
                        best_index = idx
                        best_score = mmr
                chosen = candidates.pop(best_index)
            selected.append(chosen)
        return [self._selected_projection(row, index + 1, row.get("_topsis_score")) for index, row in enumerate(selected)]

    def _selected_projection(self, row: dict[str, Any], selection_rank: int, topsis_score: Any = None) -> dict[str, Any]:
        return {
            "instanceId": row.get("instanceId"),
            "proposalId": row.get("proposalId"),
            "displayName": row.get("displayName"),
            "baseDisplayName": row.get("baseDisplayName"),
            "selectionRank": selection_rank,
            "rank": row.get("rank"),
            "generatedText": row.get("generatedText"),
            "prompt": row.get("prompt"),
            "objectiveVector": row.get("objectiveVector") or [],
            "objectiveLabel": row.get("objectiveLabel") or "--",
            "diagnosticObjectiveVector": row.get("diagnosticObjectiveVector"),
            "diagnosticObjectiveLabel": row.get("diagnosticObjectiveLabel"),
            "comparableObjectiveVector": row.get("comparableObjectiveVector") or [],
            "comparableObjectiveLabel": row.get("comparableObjectiveLabel") or "--",
            "comparableObjectiveNames": row.get("comparableObjectiveNames") or list(COMPARABLE_OBJECTIVE_NAMES),
            "topsisScore": finite_float(topsis_score, None) if topsis_score is not None else None,
        }

    def _mark_selected_rows(self, rows: list[dict[str, Any]], selected_rows: list[dict[str, Any]]) -> None:
        selected_texts = {canonical_generated_text(row.get("generatedText")) for row in selected_rows}
        selected_ranks = {
            canonical_generated_text(row.get("generatedText")): row.get("selectionRank")
            for row in selected_rows
        }
        for row in rows:
            key = canonical_generated_text(row.get("generatedText"))
            row["selected"] = key in selected_texts
            if row["selected"]:
                row["selectionRank"] = selected_ranks.get(key)
            row.pop("_topsis_score", None)
            row.pop("_selection_source_index", None)

    def _build_metric_series(
        self,
        proposal: ProposalDefinition,
        output_dir: Path,
        final_rows: list[dict[str, Any]],
        reference_text: str,
    ) -> list[dict[str, Any]]:
        history = self._read_population_history(output_dir)
        diagnostic_series = self._read_diagnostic_metric_series(proposal, output_dir, history)
        initial_population_point = (
            self._initial_population_series_point(proposal, output_dir, reference_text)
            if proposal.kind == "binary-mopso-cd"
            else None
        )
        base_series: list[dict[str, Any]] = []
        if proposal.kind == "binary-mopso-cd":
            archive_series = self._read_binary_archive_metric_series(proposal, output_dir, reference_text)
            if archive_series:
                archive_series = self._with_initial_population_series_point(archive_series, initial_population_point)
                return self._merge_series_diagnostics(archive_series, diagnostic_series)
            series = self._read_binary_metric_series(output_dir)
            if series:
                series = self._with_initial_population_series_point(series, initial_population_point)
                return self._merge_series_diagnostics(series, diagnostic_series)
        csv_series = self._read_legacy_metric_series(proposal, output_dir)
        if history:
            history_series = [
                point
                for entry in history
                for point in [self._history_entry_metrics(proposal, entry, reference_text)]
                if point is not None
            ]
            history_series = self._with_initial_population_series_point(history_series, initial_population_point)
            return self._merge_series_diagnostics(history_series, diagnostic_series or csv_series)
        if csv_series:
            base_series = csv_series
        elif initial_population_point:
            base_series = [initial_population_point]
        base_series = self._with_initial_population_series_point(base_series, initial_population_point)
        return self._merge_series_diagnostics(base_series, diagnostic_series)

    def _read_diagnostic_metric_series(
        self,
        proposal: ProposalDefinition,
        output_dir: Path,
        history: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        if proposal.kind != "binary-mopso-cd" and history:
            return self._build_posthoc_diagnostic_series(proposal, history)
        native = self._read_native_diagnostic_metric_series(proposal, output_dir)
        if self._series_has_diagnostics(native):
            return native
        return self._build_posthoc_diagnostic_series(proposal, history or [])

    def _series_has_diagnostics(self, series: list[dict[str, Any]]) -> bool:
        return any(
            point.get("globalInertia") is not None or point.get("globalEntropy") is not None
            for point in series
        )

    def _read_native_diagnostic_metric_series(
        self,
        proposal: ProposalDefinition,
        output_dir: Path,
    ) -> list[dict[str, Any]]:
        if proposal.kind == "binary-mopso-cd":
            return self._read_binary_monitor_metric_series(output_dir)
        if proposal.metrics_series_file:
            return self._read_legacy_metric_series(proposal, output_dir)
        return []

    def _read_binary_monitor_metric_series(self, output_dir: Path) -> list[dict[str, Any]]:
        rows = self._read_csv_dicts(output_dir / "monitor_metrics.csv")
        series: list[dict[str, Any]] = []
        for item in rows:
            generation = item.get("generation") or item.get("Generacion")
            if generation is None:
                continue
            generation_value = finite_float(generation, None)
            if generation_value is None:
                continue
            series.append(
                {
                    "generation": int(generation_value),
                    "hypervolume": None,
                    "nonDominatedRows": None,
                    "extent": None,
                    "unaryEntropy": None,
                    "contribution": None,
                    "globalInertia": finite_float(item.get("kmeans_inertia"), None),
                    "globalEntropy": finite_float(item.get("entity_entropy"), None),
                    "source": "binary_monitor",
                }
            )
        return [item for item in series if item["generation"] >= 0]

    def _build_posthoc_diagnostic_series(
        self,
        proposal: ProposalDefinition,
        history: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not history:
            return []
        series: list[dict[str, Any]] = []
        for entry in history:
            population = entry.get("population") if isinstance(entry.get("population"), list) else []
            if not population:
                continue
            texts = self._history_population_texts(proposal, population)
            diagnostic = self._posthoc_population_diagnostics(texts)
            if not diagnostic:
                continue
            generation = finite_float(entry.get("generation"), None)
            if generation is None:
                continue
            series.append(
                {
                    "generation": int(generation),
                    "hypervolume": None,
                    "nonDominatedRows": None,
                    "extent": None,
                    "unaryEntropy": None,
                    "contribution": None,
                    **diagnostic,
                    "source": "posthoc_diagnostic_history",
                }
            )
        return [item for item in series if item["generation"] >= 0]

    def _history_population_texts(self, proposal: ProposalDefinition, population: list[dict[str, Any]]) -> list[str]:
        texts: list[str] = []
        for index, row in enumerate(population, start=1):
            if not isinstance(row, dict):
                continue
            try:
                normalized = self._normalize_any_row(proposal, row, index)
            except Exception:
                continue
            text = str(normalized.get("generatedText") or "").strip()
            if normalized.get("status") == "ok" and text:
                texts.append(text)
        return texts

    def _posthoc_population_diagnostics(self, generated_texts: list[str]) -> dict[str, float | None] | None:
        texts = [text.strip() for text in generated_texts if text and text.strip()]
        if not texts:
            return None
        try:
            embeddings, _ = shared_sbert_service().encode_texts(POSTHOC_EMBEDDING_MODEL, texts)
            embedding_rows = [[float(value) for value in row] for row in embeddings]
            inertia = 0.0
            if len(embedding_rows) >= 2:
                from sklearn.cluster import KMeans

                clusters = min(POSTHOC_DIAGNOSTIC_KMEANS_CLUSTERS, len(embedding_rows))
                inertia = float(
                    KMeans(n_clusters=clusters, n_init=10, random_state=0).fit(embedding_rows).inertia_
                    / len(embedding_rows)
                )
            entity_entropy = self._posthoc_entity_entropy(texts)
            return {
                "globalInertia": finite_float(inertia, 0.0),
                "globalEntropy": entity_entropy,
            }
        except Exception:
            return None

    def _posthoc_entity_entropy(self, generated_texts: list[str]) -> float | None:
        try:
            if self._posthoc_spacy_model is None:
                import spacy

                self._posthoc_spacy_model = spacy.load("en_core_web_sm")
            nlp = self._posthoc_spacy_model
            concepts: list[str] = []
            total_tokens = 0
            for doc in nlp.pipe(generated_texts):
                total_tokens += len(doc)
                for token in doc:
                    if token.pos_ in POSTHOC_ENTITY_ENTROPY_POS:
                        lemma = str(token.lemma_ or "").lower().strip()
                        if lemma:
                            concepts.append(lemma)
            if not concepts or total_tokens <= 1:
                return 0.0
            counts = Counter(concepts)
            total = sum(counts.values())
            raw_entropy = -sum((count / total) * math.log2(count / total) for count in counts.values())
            return finite_float(raw_entropy / math.log2(total_tokens))
        except Exception:
            return None

    def _read_binary_archive_metric_series(
        self,
        proposal: ProposalDefinition,
        output_dir: Path,
        reference_text: str,
    ) -> list[dict[str, Any]]:
        path = output_dir / "archive_history.jsonl"
        if not path.exists():
            return []
        series: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue
            archive = payload.get("archive") if isinstance(payload.get("archive"), list) else []
            if not archive:
                continue
            rows = self._normalize_rows(proposal, archive, top_k=len(archive), reference_text=reference_text)
            metrics = self._summarize_rows(proposal, rows, Path("."), include_artifact_metrics=False)
            generation = finite_float(payload.get("generation"), None)
            if generation is None:
                continue
            series.append(
                {
                    "generation": int(generation),
                    "hypervolume": metrics.get("hypervolume"),
                    "nonDominatedRows": metrics.get("nonDominatedRows"),
                    "extent": metrics.get("extent"),
                    "unaryEntropy": metrics.get("unaryEntropy"),
                    "contribution": metrics.get("contribution"),
                    "frontPoints": self._front_points_from_rows(rows),
                    "source": "archive_history",
                }
            )
        return [item for item in series if item["generation"] >= 0]

    def _read_binary_metric_series(self, output_dir: Path) -> list[dict[str, Any]]:
        rows = self._read_csv_dicts(output_dir / "evolucion_metricas.csv")
        series = []
        for item in rows:
            generation = finite_float(item.get("generation"), None)
            if generation is None:
                continue
            series.append(
                {
                    "generation": int(generation),
                    "hypervolume": finite_float(item.get("hypervolume"), None),
                    "nonDominatedRows": finite_float(item.get("archive_size"), None),
                    "extent": None,
                    "unaryEntropy": None,
                    "contribution": None,
                    "source": "native",
                }
            )
        return [item for item in series if item["generation"] >= 0]

    def _read_binary_archive_summary_metrics(self, output_dir: Path) -> dict[str, Any]:
        counts = self._read_binary_archive_counts_from_metrics_csv(output_dir)
        if counts is None:
            counts = self._read_binary_archive_counts_from_runtime_log(output_dir)
        if counts is None:
            return {}
        update_count, prune_count = counts
        return {
            "externalArchiveUpdateCount": update_count,
            "externalArchivePruneCount": prune_count,
            "externalArchiveUpdateCountTotal": update_count,
            "externalArchivePruneCountTotal": prune_count,
        }

    def _read_binary_archive_counts_from_metrics_csv(self, output_dir: Path) -> tuple[int, int] | None:
        rows = self._read_csv_dicts(output_dir / "evolucion_metricas.csv")
        for item in reversed(rows):
            generation = finite_int_or_none(item.get("generation"))
            if generation is None or generation <= 0:
                continue
            update_count = finite_int_or_none(item.get("archive_update_count"))
            prune_count = finite_int_or_none(item.get("archive_prune_count"))
            if update_count is not None and prune_count is not None:
                return update_count, prune_count
        return None

    def _read_binary_archive_counts_from_runtime_log(self, output_dir: Path) -> tuple[int, int] | None:
        path = output_dir / "runtime.log"
        if not path.exists():
            return None
        counts: tuple[int, int] | None = None
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            update_match = ARCHIVE_UPDATES_RE.search(line)
            prune_match = ARCHIVE_PRUNES_RE.search(line)
            if update_match and prune_match:
                counts = int(update_match.group(1)), int(prune_match.group(1))
        return counts

    def _read_legacy_metric_series(self, proposal: ProposalDefinition, output_dir: Path) -> list[dict[str, Any]]:
        filename = proposal.metrics_series_file
        if not filename:
            return []
        rows = self._read_csv_dicts(output_dir / filename)
        series: list[dict[str, Any]] = []
        for item in rows:
            generation = item.get("generation") or item.get("Generacion")
            if generation is None:
                continue
            generation_value = finite_float(generation, None)
            if generation_value is None:
                continue
            series.append(
                {
                    "generation": int(generation_value),
                    "hypervolume": None,
                    "nonDominatedRows": None,
                    "extent": None,
                    "unaryEntropy": None,
                    "contribution": None,
                    "globalInertia": None,
                    "globalEntropy": None,
                    "source": "legacy_csv_without_front",
                }
            )
        return series

    def _initial_population_series_point(
        self,
        proposal: ProposalDefinition,
        output_dir: Path,
        reference_text: str,
    ) -> dict[str, Any] | None:
        path = output_dir / "data_initial_population.json"
        if not path.exists():
            return None
        payload = read_json_or_default(path, None)
        if not isinstance(payload, list) or not payload:
            return None
        rows = self._normalize_rows(proposal, payload, top_k=len(payload), reference_text=reference_text)
        metrics = self._summarize_rows(proposal, rows, Path("."), include_artifact_metrics=False)
        texts = [
            str(row.get("generatedText") or "").strip()
            for row in rows
            if row.get("status") == "ok" and str(row.get("generatedText") or "").strip()
        ]
        diagnostics = self._posthoc_population_diagnostics(texts) or {}
        return {
            "generation": 0,
            "hypervolume": metrics.get("hypervolume"),
            "nonDominatedRows": metrics.get("postHocNonDominatedRows") if metrics.get("postHocDiagnostic") else metrics.get("nonDominatedRows"),
            "extent": metrics.get("extent"),
            "unaryEntropy": metrics.get("unaryEntropy"),
            "contribution": metrics.get("contribution"),
            "globalInertia": diagnostics.get("globalInertia"),
            "globalEntropy": diagnostics.get("globalEntropy"),
            "frontPoints": self._front_points_from_rows(rows),
            "source": "initial_population",
        }

    def _with_initial_population_series_point(
        self,
        series: list[dict[str, Any]],
        initial_population_point: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        if not initial_population_point:
            return series
        without_zero = [
            point
            for point in series
            if int(finite_float(point.get("generation"), -1)) != 0
        ]
        return sorted(
            [initial_population_point, *without_zero],
            key=lambda item: int(finite_float(item.get("generation"), -1)),
        )

    def _merge_series_diagnostics(
        self,
        base_series: list[dict[str, Any]],
        diagnostic_series: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not diagnostic_series:
            return base_series
        diagnostics_by_generation = {
            int(finite_float(point.get("generation"), -1)): point
            for point in diagnostic_series
        }
        merged: list[dict[str, Any]] = []
        for point in base_series:
            generation = int(finite_float(point.get("generation"), -1))
            diagnostic = diagnostics_by_generation.get(generation) or {}
            merged_point = dict(point)
            for key in ("globalInertia", "globalEntropy"):
                if merged_point.get(key) is None and diagnostic.get(key) is not None:
                    merged_point[key] = diagnostic.get(key)
            merged.append(merged_point)
        return merged

    def _ensure_final_front_series_point(
        self,
        proposal: ProposalDefinition,
        base_series: list[dict[str, Any]],
        final_rows: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if self._series_has_front_metrics(base_series) or not final_rows:
            return base_series
        final_point = self._final_series_point(proposal, final_rows)
        if not self._series_has_front_metrics([final_point]):
            return base_series
        merged: list[dict[str, Any]] = []
        final_inserted = False
        for point in base_series:
            generation = int(finite_float(point.get("generation"), -1))
            if generation == 0:
                combined = dict(final_point)
                combined.update({key: value for key, value in point.items() if value is not None})
                combined["frontPoints"] = final_point.get("frontPoints") or point.get("frontPoints") or []
                combined["source"] = final_point.get("source")
                merged.append(combined)
                final_inserted = True
            else:
                merged.append(point)
        if not final_inserted:
            merged.append(final_point)
        return sorted(merged, key=lambda item: int(finite_float(item.get("generation"), -1)))

    def _series_has_front_metrics(self, series: list[dict[str, Any]]) -> bool:
        return any(
            bool(point.get("frontPoints"))
            or point.get("extent") is not None
            or point.get("unaryEntropy") is not None
            for point in series
            if isinstance(point, dict)
        )

    def _read_csv_dicts(self, path: Path) -> list[dict[str, str]]:
        if not path.exists():
            return []
        with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
            return list(csv.DictReader(handle))

    def _read_population_history(self, output_dir: Path) -> list[dict[str, Any]]:
        path = output_dir / "population_history.jsonl"
        if not path.exists():
            return []
        entries: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                entries.append(payload)
        return entries

    def _history_entry_metrics(self, proposal: ProposalDefinition, entry: dict[str, Any], reference_text: str) -> dict[str, Any] | None:
        generation = finite_float(entry.get("generation"), None)
        if generation is None:
            return None
        population = entry.get("population") if isinstance(entry.get("population"), list) else []
        rows = self._normalize_rows(proposal, population, top_k=len(population) or 1, reference_text=reference_text)
        metrics = self._summarize_rows(proposal, rows, Path("."), include_artifact_metrics=False)
        return {
            "generation": int(generation),
            "hypervolume": metrics.get("hypervolume"),
            "nonDominatedRows": metrics.get("postHocNonDominatedRows") if metrics.get("postHocDiagnostic") else metrics.get("nonDominatedRows"),
            "extent": metrics.get("extent"),
            "unaryEntropy": metrics.get("unaryEntropy"),
            "contribution": metrics.get("contribution"),
            "frontPoints": self._front_points_from_rows(rows),
            "source": "population_history",
        }

    def _final_series_point(self, proposal: ProposalDefinition, rows: list[dict[str, Any]]) -> dict[str, Any]:
        metrics = self._summarize_rows(proposal, rows, Path("."), include_artifact_metrics=False)
        return {
            "generation": 0,
            "hypervolume": metrics.get("hypervolume"),
            "nonDominatedRows": metrics.get("postHocNonDominatedRows") if metrics.get("postHocDiagnostic") else metrics.get("nonDominatedRows"),
            "extent": metrics.get("extent"),
            "unaryEntropy": metrics.get("unaryEntropy"),
            "contribution": metrics.get("contribution"),
            "frontPoints": self._front_points_from_rows(rows),
            "source": "final_only",
        }

    def _front_points_from_rows(self, rows: list[dict[str, Any]]) -> list[list[float]]:
        return [
            [point[0], point[1]]
            for row in rows
            if row.get("nonDominated") or row.get("postHocNonDominated")
            for point in [comparable_point(row)]
            if point is not None
        ]

    def _build_chart_payload(
        self,
        proposal: ProposalDefinition,
        rows: list[dict[str, Any]],
        selected_rows: list[dict[str, Any]],
        series: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return build_charts_from_rows(rows, selected_rows, series)

    def _output_files(self, proposal: ProposalDefinition, output_dir: Path) -> dict[str, str | None]:
        files = {
            "result": output_dir / proposal.result_file,
            "finalSelection": output_dir / proposal.final_selection_file if proposal.final_selection_file else None,
            "series": output_dir / proposal.metrics_series_file if proposal.metrics_series_file else None,
            "populationHistory": output_dir / "population_history.jsonl",
            "runtime": output_dir / "runtime.txt",
            "runtimeLog": output_dir / "runtime.log",
            "llmCalls": output_dir / "llm_calls.jsonl",
        }
        return {key: str(path) if path and path.exists() else None for key, path in files.items()}

    def _set_proposal_state_unlocked(
        self,
        run: dict[str, Any],
        proposal_id: str,
        status: str,
        stage_label: str,
        progress: float | None = None,
    ) -> None:
        state = run["proposalStates"].setdefault(
            proposal_id,
            {
                "proposalId": proposal_id,
                "displayName": proposal_id,
                "stageTotal": PROPOSAL_TOTALS.get(proposal_id, 1),
                "iterationTiming": empty_iteration_timing(),
            },
        )
        state.setdefault("iterationTiming", empty_iteration_timing())
        state.setdefault("completedRepetitions", 0)
        state.setdefault("totalRepetitions", max(1, int((run.get("config") or {}).get("repetitionsK") or 1)))
        state.setdefault("repetitionsK", state.get("totalRepetitions"))
        state["status"] = status
        state["stageLabel"] = stage_label
        if progress is not None:
            state["progress"] = clamp(progress, 0.0, 1.0)
        state["updatedAt"] = utc_now()
        run["updatedAt"] = utc_now()
        self._refresh_run_progress_unlocked(run)

    def _iteration_timing(self, state: dict[str, Any]) -> dict[str, Any]:
        timing = state.get("iterationTiming")
        if not isinstance(timing, dict):
            timing = empty_iteration_timing()
            state["iterationTiming"] = timing
        defaults = empty_iteration_timing()
        for key, value in defaults.items():
            timing.setdefault(key, value)
        return timing

    def _proposal_iteration_total(
        self,
        run: dict[str, Any],
        state: dict[str, Any],
        generation_total: int | None = None,
    ) -> int:
        config = run.get("config") or {}
        state_repetitions = state.get("totalRepetitions") or state.get("repetitionsK")
        config_repetitions = config.get("repetitionsK")
        if config_repetitions is not None and not config.get("proposalInstances"):
            repetitions = max(1, int(config_repetitions))
        else:
            repetitions = max(1, int(state_repetitions or config_repetitions or 1))
        configured_generations = max(0, int(config.get("generaciones") or 0))
        observed_generations = max(0, int(generation_total or state.get("generationTotal") or 0))
        per_repetition = configured_generations if configured_generations > 0 else observed_generations
        return max(0, per_repetition * repetitions)

    def _iteration_key(self, state: dict[str, Any], generation_index: int) -> str:
        repetition_index = max(1, int(state.get("currentRepetitionIndex") or 1))
        return f"{repetition_index}:{generation_index}"

    def _global_iteration_index(
        self,
        state: dict[str, Any],
        generation_index: int,
        generation_total: int,
    ) -> int:
        repetition_index = max(1, int(state.get("currentRepetitionIndex") or 1))
        per_repetition = max(1, int(generation_total or state.get("generationTotal") or 1))
        return (repetition_index - 1) * per_repetition + generation_index

    def _refresh_iteration_counts(
        self,
        run: dict[str, Any],
        state: dict[str, Any],
        generation_total: int | None = None,
    ) -> None:
        timing = self._iteration_timing(state)
        total = self._proposal_iteration_total(run, state, generation_total)
        completed = max(
            int(timing.get("completedIterations") or 0),
            len(timing.get("completedIterationKeys") or []),
        )
        if total > 0:
            completed = min(completed, total)
        timing["completedIterations"] = completed
        timing["totalIterations"] = total if total > 0 else None
        timing["remainingIterations"] = max(0, total - completed) if total > 0 else None

    def _mark_generation_started_unlocked(
        self,
        run: dict[str, Any],
        state: dict[str, Any],
        generation_index: int,
        generation_total: int,
        elapsed_seconds: float | None,
    ) -> None:
        timing = self._iteration_timing(state)
        key = self._iteration_key(state, generation_index)
        timing["activeIterationKey"] = key
        timing["activeStartedAtEpoch"] = time.time()
        timing["activeElapsedSeconds"] = elapsed_seconds
        self._refresh_iteration_counts(run, state, generation_total)

    def _record_iteration_duration_unlocked(
        self,
        run: dict[str, Any],
        state: dict[str, Any],
        generation_index: int,
        generation_total: int,
        duration_seconds: float | None = None,
        elapsed_seconds: float | None = None,
    ) -> bool:
        timing = self._iteration_timing(state)
        key = self._iteration_key(state, generation_index)
        completed_keys = timing.setdefault("completedIterationKeys", [])
        if key in completed_keys:
            if elapsed_seconds is not None:
                timing["previousElapsedSeconds"] = elapsed_seconds
            self._refresh_iteration_counts(run, state, generation_total)
            return False

        duration = duration_seconds if duration_seconds is not None and duration_seconds >= 0 else None
        active_key = timing.get("activeIterationKey")
        if duration is None and elapsed_seconds is not None:
            active_elapsed = timing.get("activeElapsedSeconds")
            if active_key == key and isinstance(active_elapsed, (int, float)) and elapsed_seconds > active_elapsed:
                duration = elapsed_seconds - active_elapsed
            else:
                previous_elapsed = timing.get("previousElapsedSeconds")
                if isinstance(previous_elapsed, (int, float)) and elapsed_seconds > previous_elapsed:
                    duration = elapsed_seconds - previous_elapsed

        if duration is None and active_key == key:
            started_at = timing.get("activeStartedAtEpoch")
            if isinstance(started_at, (int, float)):
                duration = max(0.0, time.time() - started_at)

        if duration is None or not math.isfinite(duration) or duration < 0:
            if elapsed_seconds is not None:
                timing["previousElapsedSeconds"] = elapsed_seconds
            self._refresh_iteration_counts(run, state, generation_total)
            return False

        samples = timing.setdefault("durationSamples", [])
        samples.append(float(duration))
        completed_keys.append(key)
        timing["completedIterations"] = max(
            int(timing.get("completedIterations") or 0),
            self._global_iteration_index(state, generation_index, generation_total),
        )
        average = sum(samples) / len(samples)
        timing["averageIterationSeconds"] = average
        timing["averageIterationLabel"] = format_duration(average)
        timing["lastIterationSeconds"] = float(duration)
        timing["lastIterationLabel"] = format_duration(duration)
        if active_key == key:
            timing["activeIterationKey"] = None
            timing["activeStartedAtEpoch"] = None
            timing["activeElapsedSeconds"] = None
        if elapsed_seconds is not None:
            timing["previousElapsedSeconds"] = elapsed_seconds
        self._refresh_iteration_counts(run, state, generation_total)
        return True

    def _apply_log_progress_unlocked(self, run: dict[str, Any], proposal_id: str, message: str) -> None:
        if proposal_id == "system":
            self._refresh_run_progress_unlocked(run)
            return

        state = run["proposalStates"].get(proposal_id)
        if not state:
            return

        state["logs"] = int(state.get("logs") or 0) + 1
        state["status"] = STATUS_RUNNING
        state["updatedAt"] = utc_now()
        base_proposal_id = str(state.get("proposalId") or proposal_id)
        progress_message = progress_log_payload(message)

        stage_match = STAGE_RE.match(progress_message)
        if stage_match:
            stage_index = int(stage_match.group(1))
            stage_total = int(stage_match.group(2))
            label = stage_match.group(3).strip()
            state["stageIndex"] = stage_index
            state["stageTotal"] = stage_total
            state["stageLabel"] = label
            state["generationIndex"] = None
            state["generationTotal"] = None
            state["progress"] = clamp((stage_index - 1) / max(stage_total, 1), 0.0, 0.98)
            self._refresh_run_progress_unlocked(run)
            return

        generation_time = log_generation_time_seconds(progress_message)
        if generation_time is not None and generation_time >= 0:
            generation_index = state.get("generationIndex")
            generation_total = state.get("generationTotal")
            if generation_index and generation_total:
                self._record_iteration_duration_unlocked(
                    run,
                    state,
                    int(generation_index),
                    int(generation_total),
                    duration_seconds=generation_time,
                )
            self._refresh_run_progress_unlocked(run)
            return

        generation_match = GENERATION_RE.search(progress_message)
        if generation_match:
            generation_index = int(generation_match.group(1))
            generation_total = int(generation_match.group(2))
            stage_total = max(int(state.get("stageTotal") or PROPOSAL_TOTALS.get(base_proposal_id, 1)), 1)
            stage_index = max(int(state.get("stageIndex") or stage_total), 1)
            base = clamp((stage_index - 1) / stage_total, 0.0, 0.98)
            state["generationIndex"] = generation_index
            state["generationTotal"] = generation_total
            state["stageLabel"] = f"Generacion {generation_index}/{generation_total}"
            state["progress"] = clamp(base + (generation_index / max(generation_total, 1)) / stage_total, 0.0, 0.98)
            elapsed_seconds = log_elapsed_seconds(progress_message)
            if base_proposal_id == "binary-mopso-cd" and not GENERATION_STARTED_RE.search(progress_message):
                self._record_iteration_duration_unlocked(
                    run,
                    state,
                    generation_index,
                    generation_total,
                    elapsed_seconds=elapsed_seconds,
                )
            else:
                self._mark_generation_started_unlocked(
                    run,
                    state,
                    generation_index,
                    generation_total,
                    elapsed_seconds,
                )
            self._refresh_run_progress_unlocked(run)
            return

        percent_match = PERCENT_RE.search(progress_message)
        if percent_match:
            percent = clamp(float(percent_match.group(1)) / 100.0, 0.0, 1.0)
            stage_total = max(int(state.get("stageTotal") or PROPOSAL_TOTALS.get(base_proposal_id, 1)), 1)
            stage_index = max(int(state.get("stageIndex") or 1), 1)
            base = clamp((stage_index - 1) / stage_total, 0.0, 0.98)
            state["progress"] = max(float(state.get("progress") or 0.0), clamp(base + percent / stage_total, 0.0, 0.98))

        if base_proposal_id == "binary-mopso-cd" and progress_message.startswith(BINARY_DETAIL_LOG_PREFIXES):
            state["stageLabel"] = progress_message[:120]

        if not state.get("stageLabel") or state.get("stageLabel") == "En cola":
            state["stageLabel"] = progress_message[:120]
        self._refresh_run_progress_unlocked(run)

    def _state_iteration_eta_seconds(self, state: dict[str, Any]) -> float | None:
        timing = self._iteration_timing(state)
        average = timing.get("averageIterationSeconds")
        remaining_iterations = timing.get("remainingIterations")
        completed_iterations = int(timing.get("completedIterations") or 0)
        if (
            completed_iterations <= 0
            or not isinstance(average, (int, float))
            or not math.isfinite(float(average))
            or remaining_iterations is None
        ):
            return None
        return max(0.0, float(average) * max(0, int(remaining_iterations)))

    def _iteration_basis_label(self, states: list[dict[str, Any]]) -> str:
        if not states:
            return "Sin propuesta activa."
        if len(states) > 1:
            ready = sum(1 for state in states if self._state_iteration_eta_seconds(state) is not None)
            return f"{ready}/{len(states)} propuesta(s) activa(s) con muestras de iteracion."

        timing = self._iteration_timing(states[0])
        completed = int(timing.get("completedIterations") or 0)
        total = timing.get("totalIterations")
        remaining = timing.get("remainingIterations")
        average_label = timing.get("averageIterationLabel") or "No disponible"
        if completed <= 0:
            total_label = f"0/{total}" if total else "0"
            return f"Esperando primera iteracion completada ({total_label})."
        if total:
            return f"{average_label} promedio/iteracion; {completed}/{total} completadas; {remaining or 0} restantes."
        return f"{average_label} promedio/iteracion; {completed} completada(s)."

    def _iteration_eta_fields(
        self,
        run: dict[str, Any],
        active_states: list[dict[str, Any]],
        queued: int,
    ) -> dict[str, Any]:
        base = {
            "remainingSeconds": None,
            "remainingLabel": "No disponible",
            "etaBasisLabel": self._iteration_basis_label(active_states),
            "etaScopeLabel": "No aplica",
        }
        if run.get("status") != STATUS_RUNNING:
            return base
        if not active_states:
            base["etaScopeLabel"] = "Sin propuesta activa."
            return base

        estimates = [self._state_iteration_eta_seconds(state) for state in active_states]
        missing_estimate = any(value is None for value in estimates)
        mode = (run.get("config") or {}).get("executionMode") or EXECUTION_MODE_FAIR_SEQUENTIAL

        if mode == EXECUTION_MODE_EXPLORATORY_PARALLEL:
            if missing_estimate:
                base["remainingLabel"] = "Esperando primera iteracion"
                base["etaScopeLabel"] = "Paralelo exploratorio; faltan muestras de propuestas activas."
                return base
            remaining = max(float(value) for value in estimates if value is not None)
            base["remainingSeconds"] = remaining
            base["remainingLabel"] = format_duration(remaining)
            base["etaScopeLabel"] = (
                "Maximo entre propuestas activas; cola no estimada."
                if queued
                else "Maximo entre propuestas activas."
            )
            return base

        active_remaining = estimates[0]
        if active_remaining is None:
            base["remainingLabel"] = "Esperando primera iteracion"
            base["etaScopeLabel"] = "Propuesta activa sin muestras suficientes."
            return base

        base["remainingSeconds"] = float(active_remaining)
        base["remainingLabel"] = format_duration(float(active_remaining))
        base["etaScopeLabel"] = (
            "Propuesta activa; cola no estimada."
            if queued
            else "Corrida activa estimable por iteraciones."
        )
        return base

    def _refresh_run_progress_unlocked(self, run: dict[str, Any]) -> None:
        states = list((run.get("proposalStates") or {}).values())
        if not states:
            return

        progress_sum = sum(float(state.get("progress") or 0.0) for state in states)
        percent = int(round(100 * progress_sum / len(states)))
        percent = int(clamp(percent, 0, 100))

        active_states = [state for state in states if state.get("status") == STATUS_RUNNING]
        active = active_states[0] if active_states else None
        queued = sum(1 for state in states if state.get("status") == STATUS_QUEUED)
        completed = sum(1 for state in states if state.get("status") == STATUS_COMPLETED)
        failed = sum(1 for state in states if state.get("status") == STATUS_FAILED)
        cancelled = sum(1 for state in states if state.get("status") == STATUS_CANCELLED)

        started_at = run.get("startedAtEpoch")
        elapsed = max(0.0, time.time() - started_at) if started_at else 0.0
        eta_fields = self._iteration_eta_fields(run, active_states, queued)

        if active:
            detail = f"{active.get('displayName', active.get('proposalId'))}: {active.get('stageLabel', 'Ejecutando')}"
        elif run.get("cancelRequested"):
            detail = "Cancelacion solicitada; esperando cierre de procesos activos."
        else:
            detail = f"{completed} completada(s), {failed} fallida(s), {cancelled} cancelada(s), {queued} en cola."

        run["progress"] = {
            "percent": percent,
            "detail": detail,
            "elapsedSeconds": elapsed,
            "elapsedLabel": format_duration(elapsed),
            "remainingSeconds": eta_fields["remainingSeconds"],
            "remainingLabel": eta_fields["remainingLabel"],
            "etaBasisLabel": eta_fields["etaBasisLabel"],
            "etaScopeLabel": eta_fields["etaScopeLabel"],
            "activeInstanceId": active.get("instanceId") if active else None,
            "activeProposalId": active.get("proposalId") if active else None,
            "activeProposalName": active.get("displayName") if active else None,
            "queuedProposals": queued,
            "completedProposals": completed,
            "failedProposals": failed,
            "cancelledProposals": cancelled,
            "totalProposals": len(states),
        }

    def _repetition_seed(self, seed: Any, repetition_index: int) -> int | None:
        if seed is None:
            return None
        return (int(seed) + int(repetition_index)) % 2_147_483_648

    def _run_process(
        self,
        run: dict[str, Any],
        proposal_id: str,
        command: list[str],
        cwd: Path,
        preload_modules: tuple[str, ...],
        python_path_entries: tuple[str, ...],
        cost_metrics_path: Path,
        random_seed: int | None = None,
        export_history: bool = False,
    ) -> dict[str, Any]:
        environment = os.environ.copy()
        environment["PYTHONIOENCODING"] = "utf-8"
        environment["BASELINE_COST_METRICS_PATH"] = str(cost_metrics_path)
        resolved_python_path_entries = [str(resolve_config_path(self.root, entry)) for entry in python_path_entries]
        if resolved_python_path_entries:
            environment["PYTHONPATH"] = os.pathsep.join(
                [*resolved_python_path_entries, environment.get("PYTHONPATH", "")]
            ).rstrip(os.pathsep)
        if random_seed is not None:
            seed_text = str(int(random_seed))
            environment["BASELINE_RANDOM_SEED"] = seed_text
            environment["PYTHONHASHSEED"] = seed_text
        if preload_modules:
            environment["BASELINE_PRELOAD_MODULES"] = ",".join(preload_modules)
        if export_history:
            environment["COMPARATOR_EXPORT_HISTORY"] = "1"
        process_started = time.perf_counter()
        timed_out = False
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=environment,
            start_new_session=sys.platform != "win32",
            **hidden_subprocess_kwargs(detached=True),
        )

        with self._lock:
            run["activeProcesses"][proposal_id] = process

        assert process.stdout is not None
        output_queue: queue.Queue[str | None] = queue.Queue()

        def read_output() -> None:
            try:
                for output_line in process.stdout:
                    output_queue.put(output_line)
            finally:
                output_queue.put(None)

        reader = threading.Thread(target=read_output, daemon=True)
        reader.start()
        timeout_seconds = run["config"]["timeoutMinutes"] * 60
        deadline = time.monotonic() + timeout_seconds
        reader_done = False
        termination_requested = False

        while True:
            try:
                line = output_queue.get(timeout=0.2)
            except queue.Empty:
                line = ""

            if line is None:
                reader_done = True
            elif line:
                message = clean_log_message(line)
                if message:
                    self._append_log(run, proposal_id, message)

            if run.get("cancelRequested") and process.poll() is None and not termination_requested:
                self._append_log(run, proposal_id, "Cancellation requested; terminating process tree.")
                self._terminate_process(process)
                termination_requested = True

            if process.poll() is None and time.monotonic() >= deadline and not termination_requested:
                self._append_log(run, proposal_id, f"Timeout reached after {run['config']['timeoutMinutes']} minute(s).")
                self._terminate_process(process)
                termination_requested = True
                timed_out = True
                return_code = 124
                break

            if reader_done and process.poll() is not None:
                return_code = process.returncode
                break

            if line is None and process.poll() is None:
                continue

        reader.join(timeout=2)
        with self._lock:
            if run.get("activeProcesses", {}).get(proposal_id) is process:
                run["activeProcesses"].pop(proposal_id, None)
        return {
            "returnCode": return_code,
            "processWallClockSeconds": time.perf_counter() - process_started,
            "timedOut": timed_out,
            "metricsPath": str(cost_metrics_path),
        }

    def _terminate_process(self, process: subprocess.Popen[str]) -> None:
        if process.poll() is not None:
            return
        try:
            if sys.platform == "win32":
                subprocess.run(
                    ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
            else:
                os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        except Exception:
            process.terminate()

    def _normalize_rows(
        self,
        proposal: ProposalDefinition,
        raw_rows: Any,
        top_k: int,
        reference_text: str | None = None,
    ) -> list[dict[str, Any]]:
        if not isinstance(raw_rows, list):
            raise ValueError("Expected a JSON list.")

        rows = [
            self._normalize_any_row(proposal, row, index)
            for index, row in enumerate(raw_rows, start=1)
            if isinstance(row, dict)
        ]
        self._attach_comparable_proxy(rows, reference_text or "", proposal.display_name)

        mark_non_dominated(rows)
        for row in rows:
            row["postHocNonDominated"] = bool(row.get("nonDominated"))
        if proposal.single_objective:
            rows.sort(key=lambda row: row["objectiveVector"][0], reverse=True)
        else:
            rows.sort(
                key=lambda row: (
                    1 if row.get("nonDominated") else 0,
                    sum(row.get("objectiveVector") or []),
                ),
                reverse=True,
            )

        for rank, row in enumerate(rows, start=1):
            row["rank"] = rank
            row["shownInTopK"] = rank <= top_k
        return rows

    def _normalize_any_row(
        self,
        proposal: ProposalDefinition,
        row: dict[str, Any],
        index: int,
    ) -> dict[str, Any]:
        if proposal.kind == "binary-mopso-cd":
            return self._normalize_binary_row(proposal, row, index)
        if proposal.single_objective:
            return self._normalize_evolmd_row(proposal, row, index)
        return self._normalize_evolmd_mo_row(proposal, row, index)

    def _attach_comparable_proxy(self, rows: list[dict[str, Any]], reference_text: str, display_name: str) -> None:
        self._comparable_proxy.attach(rows, reference_text, display_name)

    def _attach_evolmd_posthoc_diagnostics(self, rows: list[dict[str, Any]], reference_text: str, display_name: str = "EVOLMD") -> None:
        self._attach_comparable_proxy(rows, reference_text, display_name)
        mark_non_dominated(rows)
        for row in rows:
            row["postHocNonDominated"] = bool(row.get("nonDominated"))

    def _normalize_evolmd_row(
        self,
        proposal: ProposalDefinition,
        row: dict[str, Any],
        index: int,
    ) -> dict[str, Any]:
        vector = [finite_float(row.get("fitness"))]
        return {
            "proposalId": proposal.proposal_id,
            "displayName": proposal.display_name,
            "sourceIndex": index,
            "rank": index,
            "generatedText": str(row.get("generated_data") or ""),
            "prompt": str(row.get("prompt") or ""),
            "objectiveVector": vector,
            "objectiveLabel": objective_label(vector),
            "objectiveNames": list(proposal.objective_names),
            "status": "ok" if row.get("generated_data") else "empty_generated_data",
            "nonDominated": False,
            "raw": {
                "role": row.get("role"),
                "topic": row.get("topic"),
                "keywords": row.get("keywords"),
            },
        }

    def _normalize_evolmd_mo_row(
        self,
        proposal: ProposalDefinition,
        row: dict[str, Any],
        index: int,
    ) -> dict[str, Any]:
        objetivos = row.get("objetivos") if isinstance(row.get("objetivos"), list) else []
        metrics_detail = row.get("metrics_detail") if isinstance(row.get("metrics_detail"), dict) else {}
        vector = [
            finite_float(objetivos[0] if len(objetivos) > 0 else metrics_detail.get("fidelity_sbert")),
            finite_float(objetivos[1] if len(objetivos) > 1 else metrics_detail.get("diversity_individual")),
        ]
        comparable_vector = normalized_objective_vector(vector) or []
        return {
            "proposalId": proposal.proposal_id,
            "displayName": proposal.display_name,
            "sourceIndex": index,
            "rank": index,
            "generatedText": str(row.get("generated_data") or ""),
            "prompt": str(row.get("prompt") or ""),
            "objectiveVector": vector,
            "objectiveLabel": objective_label(vector),
            "objectiveNames": list(proposal.objective_names),
            "comparableObjectiveVector": comparable_vector,
            "comparableObjectiveLabel": objective_label(comparable_vector),
            "comparableObjectiveNames": list(COMPARABLE_OBJECTIVE_NAMES),
            "status": "ok" if row.get("generated_data") else "empty_generated_data",
            "nonDominated": False,
            "raw": {
                "role": row.get("role"),
                "topic": row.get("topic"),
                "keywords": row.get("keywords"),
                "metricsDetail": metrics_detail,
            },
        }

    def _normalize_binary_row(
        self,
        proposal: ProposalDefinition,
        row: dict[str, Any],
        index: int,
    ) -> dict[str, Any]:
        objectives = row.get("objectives") if isinstance(row.get("objectives"), dict) else {}
        vector = [
            finite_float(objectives.get("f1")),
            finite_float(objectives.get("f2")),
        ]
        comparable_vector = normalized_objective_vector(vector) or []
        components = row.get("components") if isinstance(row.get("components"), dict) else {}
        return {
            "proposalId": proposal.proposal_id,
            "displayName": proposal.display_name,
            "sourceIndex": index,
            "rank": index,
            "generatedText": str(row.get("generated_text") or row.get("generated_data") or ""),
            "prompt": str(row.get("prompt") or ""),
            "objectiveVector": vector,
            "objectiveLabel": objective_label(vector),
            "objectiveNames": list(proposal.objective_names),
            "comparableObjectiveVector": comparable_vector,
            "comparableObjectiveLabel": objective_label(comparable_vector),
            "comparableObjectiveNames": list(COMPARABLE_OBJECTIVE_NAMES),
            "status": "ok" if row.get("generated_text") and objectives else "invalid_solution",
            "nonDominated": False,
            "raw": {
                "solutionId": row.get("solution_id"),
                "components": components,
                "generation": row.get("generation"),
                "changed": row.get("changed"),
            },
        }

    def _summarize_rows(
        self,
        proposal: ProposalDefinition,
        rows: list[dict[str, Any]],
        output_dir: Path,
        include_artifact_metrics: bool = True,
    ) -> dict[str, Any]:
        completed = [row for row in rows if row.get("status") == "ok"]
        best_vector = completed[0]["objectiveVector"] if completed else []
        best_comparable_row = max(
            completed,
            key=lambda row: sum(row.get("comparableObjectiveVector") or []),
            default=None,
        )
        best_comparable_vector = (best_comparable_row.get("comparableObjectiveVector") if best_comparable_row else []) or []
        metrics: dict[str, Any] = {
            "totalRows": len(rows),
            "completedRows": len(completed),
            "objectiveNames": list(proposal.objective_names),
            "bestObjectiveVector": best_vector,
            "bestObjectiveLabel": objective_label(best_vector),
            "comparableObjectiveNames": list(COMPARABLE_OBJECTIVE_NAMES),
            "bestComparableObjectiveVector": best_comparable_vector,
            "bestComparableObjectiveLabel": objective_label(best_comparable_vector),
            "metricSchemaVersion": METRIC_SCHEMA_VERSION,
            "metricCoordinateSpace": METRIC_COORDINATE_SPACE,
            "nonDominatedRows": sum(1 for row in rows if row.get("nonDominated") or row.get("postHocNonDominated")),
            "hypervolume": None,
            "hypervolumeLabel": "No aplica",
            "extent": None,
            "extentLabel": "No aplica",
            "unaryEntropy": None,
            "unaryEntropyLabel": "No aplica",
            "contribution": None,
            "contributionLabel": "No aplica",
            "outputDir": str(output_dir),
        }

        diagnostic_rows = [row for row in completed if row.get("diagnosticObjectiveVector") or row.get("proxyObjectiveVector")]
        front_points = [
            point
            for row in rows
            if row.get("nonDominated") or row.get("postHocNonDominated")
            for point in [comparable_point(row)]
            if point is not None
        ]
        hypervolume = calculate_hypervolume(front_points)
        extent = calculate_extent(front_points) if front_points else None
        unary_entropy = calculate_unary_entropy(front_points) if front_points else None
        best_diagnostic_row = max(
            diagnostic_rows,
            key=lambda row: sum(row.get("comparableObjectiveVector") or []),
            default=None,
        )
        best_diagnostic = (
            (best_diagnostic_row.get("proxyObjectiveVector") or best_diagnostic_row.get("diagnosticObjectiveVector") or [])
            if best_diagnostic_row
            else []
        )
        metrics["proxyDiagnostic"] = True
        metrics["postHocDiagnostic"] = proposal.single_objective
        metrics["diagnosticObjectiveNames"] = list(PROXY_OBJECTIVE_NAMES)
        metrics["bestDiagnosticObjectiveVector"] = best_diagnostic
        metrics["bestDiagnosticObjectiveLabel"] = objective_label(best_diagnostic)
        if best_comparable_vector:
            metrics["bestComparableObjectiveVector"] = best_comparable_vector
            metrics["bestComparableObjectiveLabel"] = objective_label(best_comparable_vector)
        metrics["postHocNonDominatedRows"] = metrics["nonDominatedRows"]
        metrics["hypervolume"] = hypervolume
        metrics["hypervolumeLabel"] = f"{hypervolume:.6f}" if hypervolume is not None else "No aplica"
        metrics["extent"] = extent
        metrics["extentLabel"] = f"{extent:.6f}" if extent is not None else "No aplica"
        metrics["unaryEntropy"] = unary_entropy
        metrics["unaryEntropyLabel"] = f"{unary_entropy:.6f}" if unary_entropy is not None else "No aplica"
        metrics["moConvention"] = (
            "Maximization. Comparative metrics use the common SBERT proxy vector "
            "[(semantic_fidelity + 1) / 2, semantic_diversity / 2] for every proposal. "
            "Native objective vectors are preserved only for traceability. HV uses reference point [0, 0]."
        )

        if include_artifact_metrics and proposal.kind == "binary-mopso-cd":
            metrics.update(self._read_binary_archive_summary_metrics(output_dir))

        return metrics

    def _failed_result(
        self,
        proposal: ProposalDefinition | ProposalRunInstance,
        proposal_dir: Path,
        message: str,
        cost: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        instance = self._coerce_instance(proposal)
        base_proposal = instance.proposal
        return {
            **self._result_identity(instance),
            "status": STATUS_FAILED,
            "outputDir": str(proposal_dir),
            "rows": [],
            "embeddingFrontRows": [],
            "metrics": {
                "totalRows": 0,
                "completedRows": 0,
                "objectiveNames": list(base_proposal.objective_names),
                "bestObjectiveVector": [],
                "bestObjectiveLabel": "--",
                "comparableObjectiveNames": list(COMPARABLE_OBJECTIVE_NAMES),
                "bestComparableObjectiveVector": [],
                "bestComparableObjectiveLabel": "--",
                "metricSchemaVersion": METRIC_SCHEMA_VERSION,
                "metricCoordinateSpace": METRIC_COORDINATE_SPACE,
                "nonDominatedRows": 0,
                "hypervolume": None,
                "hypervolumeLabel": "No aplica",
                "extent": None,
                "extentLabel": "No aplica",
                "unaryEntropy": None,
                "unaryEntropyLabel": "No aplica",
                "contribution": None,
                "contributionLabel": "No aplica",
                "outputDir": str(proposal_dir),
            },
            "cost": cost or empty_cost_metrics(),
            "error": message,
        }

    def _cancelled_result(
        self,
        proposal: ProposalDefinition | ProposalRunInstance,
        proposal_dir: Path,
        cost: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        result = self._failed_result(proposal, proposal_dir, "Execution was cancelled.", cost)
        result["status"] = STATUS_CANCELLED
        result["cost"]["cancelled"] = True
        return result

    def _append_log(self, run: dict[str, Any], proposal_id: str, message: str) -> None:
        with self._lock:
            self._append_log_unlocked(run, proposal_id, message)
            self._write_summary_unlocked(run)

    def _append_log_unlocked(self, run: dict[str, Any], proposal_id: str, message: str) -> None:
        clean_message = clean_log_message(message)
        if not clean_message:
            return
        entry = {
            "at": utc_now(),
            "proposalId": proposal_id,
            "message": clean_message,
        }
        self._append_log_entry_to_file(run, entry)
        run["logs"].append(entry)
        run["logs"] = run["logs"][-COMPARATOR_RUN_LOG_LIMIT:]
        run["updatedAt"] = utc_now()
        self._apply_log_progress_unlocked(run, proposal_id, clean_message)

    def _run_log_path(self, run: dict[str, Any]) -> Path:
        return Path(run["runDir"]) / "logs.jsonl"

    def _append_log_entry_to_file(self, run: dict[str, Any], entry: dict[str, Any]) -> None:
        try:
            log_path = self._run_log_path(run)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError:
            pass

    def _public_run(self, run: dict[str, Any]) -> dict[str, Any]:
        public = {
            key: value
            for key, value in run.items()
            if key not in {"activeProcesses", "startedAtEpoch"}
        }
        return self._with_metric_recompute_status(public)

    def _with_metric_recompute_status(self, run: dict[str, Any]) -> dict[str, Any]:
        public = self._with_internal_bmopso_analysis(self._with_instance_labels(run))
        public["metricRecomputeStatus"] = self._metric_recompute_status(public)
        return public

    def _with_internal_bmopso_analysis(self, run: dict[str, Any]) -> dict[str, Any]:
        proposals = run.get("proposals")
        if not isinstance(proposals, list):
            return run
        enriched_proposals: list[Any] = []
        for result in proposals:
            if not isinstance(result, dict):
                enriched_proposals.append(result)
                continue
            enriched = dict(result)
            if enriched.get("proposalId") == BINARY_PROPOSAL_ID and enriched.get("status") == STATUS_COMPLETED:
                repetition_analyses = self._build_internal_bmopso_repetition_analyses(enriched)
                if repetition_analyses:
                    analysis = self._aggregate_internal_bmopso_analyses(enriched, repetition_analyses)
                    enriched["pointChartRepetitions"] = self._attach_internal_bmopso_to_point_repetitions(
                        enriched,
                        repetition_analyses,
                    )
                else:
                    analysis = self._build_single_internal_bmopso_analysis(enriched, enriched)
                if analysis:
                    enriched["internalBmopsoAnalysis"] = analysis
                else:
                    enriched.pop("internalBmopsoAnalysis", None)
            else:
                enriched.pop("internalBmopsoAnalysis", None)
            enriched_proposals.append(enriched)
        run["proposals"] = enriched_proposals
        return run

    def _build_internal_bmopso_analysis(self, result: dict[str, Any]) -> dict[str, Any] | None:
        repetition_analyses = self._build_internal_bmopso_repetition_analyses(result)
        if repetition_analyses:
            return self._aggregate_internal_bmopso_analyses(result, repetition_analyses)
        return self._build_single_internal_bmopso_analysis(result, result)

    def _build_internal_bmopso_repetition_analyses(self, result: dict[str, Any]) -> list[dict[str, Any]]:
        repetitions = result.get("repetitions") if isinstance(result.get("repetitions"), list) else []
        return [
            analysis
            for repetition in repetitions
            if isinstance(repetition, dict) and repetition.get("status") == STATUS_COMPLETED
            for analysis in [self._build_single_internal_bmopso_analysis(repetition, result)]
            if analysis
        ]

    def _attach_internal_bmopso_to_point_repetitions(
        self,
        result: dict[str, Any],
        analyses: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        analyses_by_repetition = {
            finite_int_or_none(analysis.get("repetitionIndex")): analysis
            for analysis in analyses
            if finite_int_or_none(analysis.get("repetitionIndex")) is not None
        }
        repetitions: list[dict[str, Any]] = []
        for item in proposal_point_chart_repetitions(result):
            copied = dict(item)
            analysis = analyses_by_repetition.get(finite_int_or_none(copied.get("repetitionIndex")))
            if analysis:
                copied["internalBmopsoAnalysis"] = analysis
            repetitions.append(copied)
        return repetitions

    def _build_single_internal_bmopso_analysis(
        self,
        result: dict[str, Any],
        identity_source: dict[str, Any],
    ) -> dict[str, Any] | None:
        output_dir = Path(str(result.get("outputDir") or ""))
        if not output_dir.exists():
            return None
        series = self._read_binary_internal_metric_series(output_dir)
        if not series:
            return None
        proposal = PROPOSAL_BY_ID.get(BINARY_PROPOSAL_ID)
        if proposal is None:
            return None
        instance = self._internal_bmopso_instance(identity_source, proposal)
        rows = self._read_binary_internal_front_rows(proposal, instance, output_dir)
        if not rows:
            return None
        selected_rows = self._read_binary_internal_selected_rows(proposal, instance, output_dir, rows)
        self._mark_selected_rows(rows, selected_rows)
        charts = build_charts_from_rows(rows, selected_rows, series)
        self._retag_internal_bmopso_charts(charts)
        hypervolume = latest_finite_series_value(series, "hypervolume")
        return {
            "available": True,
            "instanceId": instance.instance_id,
            "proposalId": instance.proposal_id,
            "displayName": instance.display_name,
            "repetitionIndex": result.get("repetitionIndex"),
            "repetitionSeed": result.get("repetitionSeed"),
            "source": "evolucion_metricas.csv",
            "coordinateSpace": BINARY_INTERNAL_COORDINATE_SPACE,
            "metrics": {
                "hypervolume": hypervolume,
                "hypervolumeLabel": f"{hypervolume:.6f}" if hypervolume is not None else "No aplica",
            },
            "series": series,
            "charts": charts,
        }

    def _internal_bmopso_instance(
        self,
        result: dict[str, Any],
        proposal: ProposalDefinition,
    ) -> ProposalRunInstance:
        runtime_config = result.get("runtimeConfig") if isinstance(result.get("runtimeConfig"), dict) else {}
        return ProposalRunInstance(
            instance_id=str(result.get("instanceId") or result.get("proposalId") or proposal.proposal_id),
            proposal_id=proposal.proposal_id,
            display_name=str(result.get("displayName") or proposal.display_name),
            base_display_name=str(result.get("baseDisplayName") or proposal.display_name),
            proposal=proposal,
            proposal_config=result.get("proposalConfig") if isinstance(result.get("proposalConfig"), dict) else {"extraArgs": "", "cliValues": {}},
            runtime_config=runtime_config,
            n=int(result.get("n") or runtime_config.get("n") or COMPARATOR_DEFAULTS.get("n") or 1),
            repetitions_k=int(result.get("repetitionsK") or runtime_config.get("repetitionsK") or COMPARATOR_DEFAULTS.get("repetitionsK") or 1),
            order_index=int(finite_float(result.get("orderIndex"), 0)),
        )

    def _read_binary_internal_metric_series(self, output_dir: Path) -> list[dict[str, Any]]:
        rows = self._read_csv_dicts(output_dir / "evolucion_metricas.csv")
        series: list[dict[str, Any]] = []
        for item in rows:
            generation = finite_int_or_none(item.get("generation"))
            hypervolume = finite_float(item.get("hypervolume"), None)
            if generation is None or generation < 0 or hypervolume is None:
                continue
            archive_size = finite_int_or_none(item.get("archive_size"))
            point = {
                "generation": generation,
                "hypervolume": hypervolume,
                "source": "evolucion_metricas.csv",
            }
            if archive_size is not None:
                point["archiveSize"] = archive_size
                point["nonDominatedRows"] = archive_size
            series.append(point)
        return series

    def _read_binary_internal_front_rows(
        self,
        proposal: ProposalDefinition,
        instance: ProposalRunInstance,
        output_dir: Path,
    ) -> list[dict[str, Any]]:
        payload = read_json_or_default(output_dir / proposal.result_file, None)
        if not isinstance(payload, list):
            return []
        rows = [
            self._normalize_binary_row(proposal, item, index)
            for index, item in enumerate(payload, start=1)
            if isinstance(item, dict)
        ]
        self._attach_instance_metadata(rows, instance)
        mark_non_dominated(rows)
        for row in rows:
            row["postHocNonDominated"] = False
        rows.sort(
            key=lambda row: (
                1 if row.get("nonDominated") else 0,
                sum(row.get("objectiveVector") or []),
            ),
            reverse=True,
        )
        for rank, row in enumerate(rows, start=1):
            row["rank"] = rank
        return rows

    def _read_binary_internal_selected_rows(
        self,
        proposal: ProposalDefinition,
        instance: ProposalRunInstance,
        output_dir: Path,
        rows: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        selected_payload = read_json_or_default(output_dir / "final_selection_hybrid.json", None)
        selected = [item for item in selected_payload if isinstance(item, dict)] if isinstance(selected_payload, list) else []
        if not selected:
            return []
        selected_rows = self._normalize_selected_rows(proposal, selected, rows)
        self._attach_instance_metadata(selected_rows, instance)
        return selected_rows

    def _retag_internal_bmopso_charts(self, charts: dict[str, Any]) -> None:
        for key in ("pareto", "selected", "nonDominated"):
            for point in charts.get(key) or []:
                if isinstance(point, dict):
                    point["coordinateSpace"] = BINARY_INTERNAL_COORDINATE_SPACE

    def _aggregate_internal_bmopso_analyses(
        self,
        result: dict[str, Any],
        analyses: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        if not analyses:
            return None
        series = self._aggregate_internal_bmopso_series(analyses)
        charts = {
            "pareto": [point for analysis in analyses for point in (analysis.get("charts") or {}).get("pareto", [])],
            "selected": [point for analysis in analyses for point in (analysis.get("charts") or {}).get("selected", [])],
            "nonDominated": self._internal_bmopso_non_dominated_points(
                [point for analysis in analyses for point in (analysis.get("charts") or {}).get("pareto", [])]
            ),
            "series": series,
        }
        hypervolume = latest_finite_series_value(series, "hypervolume")
        return {
            "available": True,
            "instanceId": str(result.get("instanceId") or result.get("proposalId") or BINARY_PROPOSAL_ID),
            "proposalId": BINARY_PROPOSAL_ID,
            "displayName": str(result.get("displayName") or "Binary MOPSO-CD"),
            "source": "evolucion_metricas.csv",
            "coordinateSpace": BINARY_INTERNAL_COORDINATE_SPACE,
            "metrics": {
                "hypervolume": hypervolume,
                "hypervolumeLabel": f"{hypervolume:.6f}" if hypervolume is not None else "No aplica",
            },
            "series": series,
            "charts": charts,
        }

    def _aggregate_internal_bmopso_series(self, analyses: list[dict[str, Any]]) -> list[dict[str, Any]]:
        buckets: dict[int, dict[str, list[float]]] = {}
        for analysis in analyses:
            for point in analysis.get("series") or []:
                generation = finite_int_or_none(point.get("generation"))
                if generation is None or generation < 0:
                    continue
                bucket = buckets.setdefault(generation, {"hypervolume": [], "archiveSize": [], "nonDominatedRows": []})
                for key in bucket:
                    value = finite_float(point.get(key), None)
                    if value is not None:
                        bucket[key].append(value)
        series: list[dict[str, Any]] = []
        for generation in sorted(buckets):
            bucket = buckets[generation]
            item: dict[str, Any] = {
                "generation": generation,
                "hypervolume": sum(bucket["hypervolume"]) / len(bucket["hypervolume"]) if bucket["hypervolume"] else None,
                "source": "evolucion_metricas.csv",
            }
            if bucket["archiveSize"]:
                item["archiveSize"] = sum(bucket["archiveSize"]) / len(bucket["archiveSize"])
            if bucket["nonDominatedRows"]:
                item["nonDominatedRows"] = sum(bucket["nonDominatedRows"]) / len(bucket["nonDominatedRows"])
            series.append(item)
        return series

    def _internal_bmopso_non_dominated_points(self, points: list[dict[str, Any]]) -> list[dict[str, Any]]:
        valid_points = [
            point
            for point in points
            if isinstance(point, dict)
            and finite_float(point.get("x"), None) is not None
            and finite_float(point.get("y"), None) is not None
        ]
        front = []
        for point in valid_points:
            x_value = finite_float(point.get("x"))
            y_value = finite_float(point.get("y"))
            dominated = any(
                other is not point
                and finite_float(other.get("x")) >= x_value
                and finite_float(other.get("y")) >= y_value
                and (finite_float(other.get("x")) > x_value or finite_float(other.get("y")) > y_value)
                for other in valid_points
            )
            if not dominated:
                copied = dict(point)
                copied["coordinateSpace"] = BINARY_INTERNAL_COORDINATE_SPACE
                front.append(copied)
        return front

    def _metric_recompute_status(self, run: dict[str, Any]) -> dict[str, Any]:
        available = run.get("status") == STATUS_COMPLETED
        current_schema = int(finite_float(run.get("metricSchemaVersion"), 0))
        coordinate_space = str(run.get("metricCoordinateSpace") or "")
        legacy_points = any(self._proposal_has_legacy_chart_points(proposal) for proposal in run.get("proposals") or [])
        recommended = available and (
            current_schema != METRIC_SCHEMA_VERSION
            or coordinate_space != METRIC_COORDINATE_SPACE
            or legacy_points
        )
        return {
            "available": available,
            "recommended": recommended,
            "currentSchemaVersion": current_schema or None,
            "expectedSchemaVersion": METRIC_SCHEMA_VERSION,
            "coordinateSpace": coordinate_space or None,
            "expectedCoordinateSpace": METRIC_COORDINATE_SPACE,
            "legacyChartPointsDetected": legacy_points,
        }

    def _proposal_has_legacy_chart_points(self, proposal: Any) -> bool:
        if not isinstance(proposal, dict):
            return False
        for key in ("pareto", "selected", "nonDominated"):
            points = ((proposal.get("charts") or {}).get(key) or [])
            for point in points[:3]:
                if isinstance(point, dict) and point.get("coordinateSpace") != METRIC_COORDINATE_SPACE:
                    return True
        for row in (proposal.get("rows") or [])[:3]:
            if isinstance(row, dict) and not row.get("comparableObjectiveVector"):
                return True
        return False

    def _write_summary_unlocked(self, run: dict[str, Any]) -> None:
        write_json(Path(run["runDir"]) / "summary.json", self._public_run(run))
