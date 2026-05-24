from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

from initial_population.common_metrics import mark_non_dominated, objective_label


PROGRESS_PREFIX = "__INITIAL_POPULATION_COMPARISON_PROGRESS__"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"


class BaselineExecutionError(Exception):
    def __init__(self, stage: str, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.stage = stage
        self.details = details or {}


class Recorder:
    def __init__(self, output_dir: Path, config: dict[str, Any]) -> None:
        self.output_dir = output_dir
        self.config = config
        self.started = time.perf_counter()
        self.logs: list[dict[str, Any]] = []
        self.output_dir.mkdir(parents=True, exist_ok=True)
        write_json(self.output_dir / "config.json", config)

    def log(self, stage: str, message: str) -> None:
        entry = {"at": utc_now(), "stage": stage, "message": message}
        self.logs.append(entry)
        print(f"[{stage}] {message}", flush=True)
        write_json(self.output_dir / "logs.json", self.logs)

    def progress(
        self,
        stage: str,
        detail: str,
        percent: float,
        stage_index: int,
        stage_total: int,
        completed: int | None = None,
        total: int | None = None,
    ) -> None:
        payload = {
            "strategyId": self.config.get("strategyId"),
            "stage": stage,
            "detail": detail,
            "percent": int(max(0, min(100, round(percent)))),
            "stageIndex": stage_index,
            "stageTotal": stage_total,
            "completed": completed,
            "total": total,
            "elapsedSeconds": time.perf_counter() - self.started,
        }
        print(f"{PROGRESS_PREFIX}{json.dumps(payload, ensure_ascii=False)}", flush=True)
        self.log(stage, detail)


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def finite_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number


def add_import_path(path: Path) -> None:
    value = str(path.resolve())
    if value not in sys.path:
        sys.path.insert(0, value)


def role_topic_artifacts(initial_population_module: Any) -> dict[str, Any]:
    return {
        "roles": list(getattr(initial_population_module, "ROLES_DEFAULT", [])),
        "topics": list(getattr(initial_population_module, "TOPICS_DEFAULT", [])),
    }


def prompt_artifacts() -> dict[str, Any]:
    try:
        from agents import generate_data, initial_prompts, keywords
    except Exception as error:
        return {"error": str(error)}

    sample_individual = {
        "role": "{role}",
        "topic": "{topic}",
        "prompt": "{prompt}",
        "keywords": ["{keyword_1}", "{keyword_2}"],
    }
    artifacts: dict[str, Any] = {}
    try:
        artifacts["initialPrompt"] = {
            "systemPrompt": initial_prompts._get_system_prompt(),
            "userPromptTemplate": initial_prompts._get_user_prompt("{reference_text}", "{role}", "{topic}"),
        }
    except Exception as error:
        artifacts["initialPrompt"] = {"error": str(error)}
    try:
        artifacts["keywords"] = {
            "systemPrompt": keywords._get_system_prompt(5, 12),
            "userPromptTemplate": keywords._get_user_prompt("{role}", "{topic}", "{prompt}"),
        }
    except Exception as error:
        artifacts["keywords"] = {"error": str(error)}
    try:
        artifacts["generation"] = {
            "systemPrompt": generate_data._get_system_prompt(),
            "userPromptTemplate": generate_data._get_user_prompt(sample_individual, "{reference_text}"),
        }
    except Exception as error:
        artifacts["generation"] = {"error": str(error)}
    return artifacts


def normalize_evolmd_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        fitness = finite_float(row.get("fitness"))
        normalized.append({
            "sourceIndex": index,
            "rank": index,
            "role": str(row.get("role") or ""),
            "topic": str(row.get("topic") or ""),
            "action": "",
            "prompt": str(row.get("prompt") or ""),
            "generatedText": str(row.get("generated_data") or ""),
            "status": "ok" if row.get("generated_data") else "empty_generated_data",
            "nativeObjectiveVector": [fitness],
            "nativeObjectiveLabel": objective_label([fitness]),
            "nativeObjectiveNames": ["fitness"],
            "native": {
                "fitness": fitness,
                "keywords": row.get("keywords") if isinstance(row.get("keywords"), list) else [],
            },
        })
    normalized.sort(key=lambda item: item["nativeObjectiveVector"][0], reverse=True)
    for rank, row in enumerate(normalized, start=1):
        row["rank"] = rank
    return normalized


def normalize_evolmd_mo_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        objetivos = row.get("objetivos") if isinstance(row.get("objetivos"), list) else []
        metrics_detail = row.get("metrics_detail") if isinstance(row.get("metrics_detail"), dict) else {}
        vector = [
            finite_float(objetivos[0] if len(objetivos) > 0 else metrics_detail.get("fidelity_sbert")),
            finite_float(objetivos[1] if len(objetivos) > 1 else metrics_detail.get("diversity_individual")),
        ]
        normalized.append({
            "sourceIndex": index,
            "rank": index,
            "role": str(row.get("role") or ""),
            "topic": str(row.get("topic") or ""),
            "action": "",
            "prompt": str(row.get("prompt") or ""),
            "generatedText": str(row.get("generated_data") or ""),
            "status": "ok" if row.get("generated_data") else "empty_generated_data",
            "nativeObjectiveVector": vector,
            "nativeObjectiveLabel": objective_label(vector),
            "nativeObjectiveNames": ["fidelity_sbert", "diversity_individual"],
            "native": {
                "objetivos": vector,
                "keywords": row.get("keywords") if isinstance(row.get("keywords"), list) else [],
                "metricsDetail": metrics_detail,
            },
        })
    mark_non_dominated(normalized, "nativeObjectiveVector", "nativeNonDominated")
    normalized.sort(
        key=lambda item: (
            1 if item.get("nativeNonDominated") else 0,
            sum(item.get("nativeObjectiveVector") or []),
        ),
        reverse=True,
    )
    for rank, row in enumerate(normalized, start=1):
        row["rank"] = rank
    return normalized


async def run_evolmd(config: dict[str, Any], output_dir: Path, recorder: Recorder) -> dict[str, Any]:
    repository_dir = Path(config["repositoryDir"]).resolve()
    add_import_path(repository_dir)

    from agents.llm_agent import LLMAgent
    from ga_core import initial_population, setup
    from ga_core.ga import generar_data_para_individuo
    from ga_core.utils import guardar_individuos
    from metrics.bert import bertscore_individuos
    from metrics.reports import append_metrics

    seed = config.get("seed")
    if seed is not None:
        random.seed(int(seed))

    recorder.progress("setup", "Preparing EVOLMD initial-population experiment.", 4, 1, 5)
    reference_path = output_dir / "reference_input.txt"
    reference_path.write_text(config["referenceText"], encoding="utf-8")
    outdir, ref_text = setup.setup_experiment(
        base_dir=output_dir / "exec",
        texto_referencia_arg=str(reference_path),
    )
    llm_agent = LLMAgent(model=config["model"])

    recorder.progress("initial_population", f"EVOLMD generating {config['n']} initial individuals.", 10, 2, 5, 0, config["n"])
    t_initial_population = time.perf_counter()
    individuals = await initial_population.generar_poblacion_inicial(
        n=config["n"],
        llm_agent=llm_agent,
        texto_referencia=ref_text,
        archivo_salida=outdir / "data_initial_population.json",
        temp_prompts=config["tempPrompts"],
        temp_keywords=config["tempKeywords"],
    )
    initial_population_seconds = time.perf_counter() - t_initial_population
    recorder.progress("initial_population", f"EVOLMD created {len(individuals)} initial individuals.", 35, 2, 5, len(individuals), config["n"])

    recorder.progress("generation", "EVOLMD generating data for initial population in upstream batches of 10.", 40, 3, 5, 0, len(individuals))
    t_generation = time.perf_counter()
    batch_size = 10
    with_data = []
    completed = 0
    for start in range(0, len(individuals), batch_size):
        batch = individuals[start:start + batch_size]
        tasks = [
            generar_data_para_individuo(individual, ref_text, llm_agent, temperatura=config["tempGeneration"])
            for individual in batch
        ]
        results = await asyncio.gather(*tasks)
        with_data.extend(results)
        completed += len(results)
        recorder.progress("generation", f"EVOLMD generated data {completed}/{len(individuals)}.", 40 + 35 * completed / max(1, len(individuals)), 3, 5, completed, len(individuals))
    generation_seconds = time.perf_counter() - t_generation

    recorder.progress("native_evaluation", "EVOLMD evaluating initial population with BERTScore.", 80, 4, 5)
    t_evaluation = time.perf_counter()
    evaluated = bertscore_individuos(with_data, ref_text, model_type=config["bertModel"])
    evaluation_seconds = time.perf_counter() - t_evaluation
    guardar_individuos(evaluated, outdir / "data_inicial_evaluada.json")
    append_metrics(outdir, 0, evaluated, duration_sec=evaluation_seconds)

    runtime = {
        "initial_population_sec": initial_population_seconds,
        "initial_gen_sec": generation_seconds,
        "initial_eval_sec": evaluation_seconds,
        "total_sec": time.perf_counter() - recorder.started,
    }
    write_runtime(outdir / "runtime.txt", runtime)
    rows = normalize_evolmd_rows(evaluated)
    write_json(output_dir / "final_rows.json", rows)
    artifacts = {
        "rolesTopics": role_topic_artifacts(initial_population),
        "upstreamPrompts": prompt_artifacts(),
        "runtimeConfig": {
            "initialPopulationBatchSize": getattr(initial_population, "DEFAULT_BATCH_SIZE", 10),
            "dataGenerationBatchSize": 10,
            "ollamaHost": "http://127.0.0.1:11434",
            "bertModel": config["bertModel"],
        },
    }
    write_json(output_dir / "artifacts.json", artifacts)
    recorder.progress("done", "EVOLMD initial population completed.", 100, 5, 5)
    return {
        "strategyId": "evolmd-initial",
        "displayName": "EVOLMD inicial",
        "status": STATUS_COMPLETED,
        "outputDir": str(outdir),
        "rows": rows,
        "nativeMetrics": summarize_native_rows(rows, ["fitness"]),
        "artifacts": artifacts,
        "runtime": runtime,
        "error": None,
    }


async def run_evolmd_mo(config: dict[str, Any], output_dir: Path, recorder: Recorder) -> dict[str, Any]:
    repository_dir = Path(config["repositoryDir"]).resolve()
    add_import_path(repository_dir / "src")
    add_import_path(repository_dir)

    from agents.llm_agent import LLMAgent
    from ga_core import initial_population, setup
    from ga_core.ga import evaluar_poblacion, generar_data_para_individuo
    from ga_core.utils import guardar_individuos

    seed = config.get("seed")
    if seed is not None:
        random.seed(int(seed))

    recorder.progress("setup", "Preparing EVOLMD-MO initial-population experiment.", 4, 1, 5)
    reference_path = output_dir / "reference_input.txt"
    reference_path.write_text(config["referenceText"], encoding="utf-8")
    outdir, ref_text = setup.setup_experiment(
        base_dir=output_dir / "exec",
        texto_referencia_arg=str(reference_path),
    )
    llm_agent = LLMAgent(model=config["model"])

    recorder.progress("initial_population", f"EVOLMD-MO generating {config['n']} initial individuals concurrently.", 10, 2, 5, 0, config["n"])
    t_initial_population = time.perf_counter()
    individuals = await initial_population.generar_poblacion_inicial(
        n=config["n"],
        llm_agent=llm_agent,
        texto_referencia=ref_text,
        archivo_salida=outdir / "data_initial_population.json",
        temp_prompts=config["tempPrompts"],
        temp_keywords=config["tempKeywords"],
    )
    initial_population_seconds = time.perf_counter() - t_initial_population
    recorder.progress("initial_population", f"EVOLMD-MO created {len(individuals)} initial individuals.", 35, 2, 5, len(individuals), config["n"])

    recorder.progress("generation", "EVOLMD-MO generating data for all initial individuals concurrently.", 40, 3, 5, 0, len(individuals))
    t_generation = time.perf_counter()
    tasks = [
        generar_data_para_individuo(individual, ref_text, llm_agent, temperatura=config["tempGeneration"])
        for individual in individuals
    ]
    with_data = await asyncio.gather(*tasks)
    generation_seconds = time.perf_counter() - t_generation
    recorder.progress("generation", f"EVOLMD-MO generated data {len(with_data)}/{len(individuals)}.", 75, 3, 5, len(with_data), len(individuals))

    recorder.progress("native_evaluation", "EVOLMD-MO evaluating initial population with SBERT and individual diversity.", 80, 4, 5)
    t_evaluation = time.perf_counter()
    evaluated = evaluar_poblacion(with_data, ref_text)
    evaluation_seconds = time.perf_counter() - t_evaluation
    guardar_individuos(evaluated, outdir / "data_inicial_evaluada.json")

    runtime = {
        "initial_population_sec": initial_population_seconds,
        "initial_gen_sec": generation_seconds,
        "initial_eval_sec": evaluation_seconds,
        "total_sec": time.perf_counter() - recorder.started,
    }
    write_runtime(outdir / "runtime.txt", runtime)
    rows = normalize_evolmd_mo_rows(evaluated)
    write_json(output_dir / "final_rows.json", rows)
    artifacts = {
        "rolesTopics": role_topic_artifacts(initial_population),
        "upstreamPrompts": prompt_artifacts(),
        "runtimeConfig": {
            "initialPopulationConcurrency": "n asyncio tasks",
            "dataGenerationConcurrency": "n asyncio tasks",
            "ollamaHost": "http://127.0.0.1:11434",
            "nativeEmbeddingModel": "all-MiniLM-L6-v2",
        },
    }
    write_json(output_dir / "artifacts.json", artifacts)
    recorder.progress("done", "EVOLMD-MO initial population completed.", 100, 5, 5)
    return {
        "strategyId": "evolmd-mo-initial",
        "displayName": "EVOLMD-MO inicial",
        "status": STATUS_COMPLETED,
        "outputDir": str(outdir),
        "rows": rows,
        "nativeMetrics": summarize_native_rows(rows, ["fidelity_sbert", "diversity_individual"]),
        "artifacts": artifacts,
        "runtime": runtime,
        "error": None,
    }


def summarize_native_rows(rows: list[dict[str, Any]], objective_names: list[str]) -> dict[str, Any]:
    completed = [row for row in rows if row.get("status") == "ok"]
    vector_key = "nativeObjectiveVector"
    if len(objective_names) >= 2:
        mark_non_dominated(completed, vector_key, "nativeNonDominated")
    best = max(completed, key=lambda row: sum(row.get(vector_key) or []), default=None)
    return {
        "objectiveNames": objective_names,
        "totalRows": len(rows),
        "completedRows": len(completed),
        "nonDominatedRows": sum(1 for row in completed if row.get("nativeNonDominated")),
        "bestObjectiveVector": best.get(vector_key) if best else [],
        "bestObjectiveLabel": best.get("nativeObjectiveLabel") if best else "--",
    }


def write_runtime(path: Path, runtime: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{key}={value:.6f}" for key, value in runtime.items()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


async def run_baseline(config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    recorder = Recorder(output_dir, config)
    strategy_id = config.get("strategyId")
    if strategy_id == "evolmd-initial":
        return await run_evolmd(config, output_dir, recorder)
    if strategy_id == "evolmd-mo-initial":
        return await run_evolmd_mo(config, output_dir, recorder)
    raise BaselineExecutionError("config", f"Unsupported baseline strategy: {strategy_id}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run isolated initial-population baseline strategy.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    try:
        result = asyncio.run(run_baseline(config, output_dir))
        write_json(output_dir / "result.json", result)
        return 0
    except BaselineExecutionError as error:
        failure = {
            "strategyId": config.get("strategyId"),
            "displayName": config.get("displayName") or config.get("strategyId"),
            "status": STATUS_FAILED,
            "rows": [],
            "nativeMetrics": {},
            "artifacts": {},
            "runtime": {},
            "outputDir": str(output_dir),
            "error": {
                "stage": error.stage,
                "message": str(error),
                "details": error.details,
            },
        }
        write_json(output_dir / "result.json", failure)
        return 2
    except Exception as error:
        failure = {
            "strategyId": config.get("strategyId"),
            "displayName": config.get("displayName") or config.get("strategyId"),
            "status": STATUS_FAILED,
            "rows": [],
            "nativeMetrics": {},
            "artifacts": {},
            "runtime": {},
            "outputDir": str(output_dir),
            "error": {
                "stage": "runner",
                "message": str(error),
                "details": {},
            },
        }
        write_json(output_dir / "result.json", failure)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
