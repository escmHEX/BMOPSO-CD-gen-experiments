const embeddingWorker = new Worker(new URL("./embedding-worker.js", import.meta.url), { type: "module" });

const EMBEDDING_MODELS = {
  "Xenova/all-MiniLM-L6-v2": {
    displayName: "all-MiniLM-L6-v2",
    family: "Sentence Transformers / SBERT",
    baseModel: "sentence-transformers/all-MiniLM-L6-v2",
    mtebAverage: "56,26",
    mtebRetrieval: "41,95",
    dimensions: "384",
    maxInput: "256 word pieces",
    approximateCost: "Muy bajo: ONNX quantized aprox. 24,5 MB",
    language: "Inglés",
  },
  "Xenova/gte-small": {
    displayName: "gte-small",
    family: "General Text Embeddings",
    baseModel: "thenlper/gte-small",
    mtebAverage: "61,36",
    mtebRetrieval: "49,46",
    dimensions: "384",
    maxInput: "512 tokens",
    approximateCost: "Muy bajo: ONNX int8 aprox. 33,8 MB",
    language: "Inglés",
  },
};

const PIPELINE_CONFIG = {
  task: "feature-extraction",
  pooling: "mean",
  normalize: "true",
  execution: "Web Worker en navegador",
};

const COMPONENT_PRESETS = {
  role: {
    name: "role",
    definition: "speaker identity or perspective",
    current: "local resident affected by flooding",
    target: "community member reporting flood damage",
    other: "topic = flooded housing; action = report urgent shelter needs",
  },
  topic: {
    name: "topic",
    definition: "subject or event focus",
    current: "flooded housing",
    target: "residential flood damage",
    other: "role = local resident affected by flooding; action = report urgent shelter needs",
  },
  action: {
    name: "action",
    definition: "communicative intent or requested operation",
    current: "report urgent shelter needs",
    target: "request emergency housing assistance",
    other: "role = local resident affected by flooding; topic = flooded housing",
  },
};

const workerRequests = new Map();
let nextWorkerRequestId = 1;
let stopRequested = false;
let latestEmbeddingVectors = null;
let latestCheckerRows = [];

const dom = {
  navItems: document.querySelectorAll(".nav-item"),
  panels: document.querySelectorAll(".tool-panel"),
  lmEndpoint: document.querySelector("#lmEndpoint"),
  lmApiMode: document.querySelector("#lmApiMode"),
  llmModelSelect: document.querySelector("#llmModelSelect"),
  llmModelManual: document.querySelector("#llmModelManual"),
  checkerEmbeddingModel: document.querySelector("#checkerEmbeddingModel"),
  runCount: document.querySelector("#runCount"),
  candidateCount: document.querySelector("#candidateCount"),
  temperature: document.querySelector("#temperature"),
  topP: document.querySelector("#topP"),
  maxTokens: document.querySelector("#maxTokens"),
  requestTimeout: document.querySelector("#requestTimeout"),
  minWords: document.querySelector("#minWords"),
  maxWords: document.querySelector("#maxWords"),
  semanticMargin: document.querySelector("#semanticMargin"),
  targetCopyThreshold: document.querySelector("#targetCopyThreshold"),
  concurrency: document.querySelector("#concurrency"),
  loadModelsButton: document.querySelector("#loadModelsButton"),
  previewPromptButton: document.querySelector("#previewPromptButton"),
  runTemplateButton: document.querySelector("#runTemplateButton"),
  stopTemplateButton: document.querySelector("#stopTemplateButton"),
  componentPreset: document.querySelector("#componentPreset"),
  componentName: document.querySelector("#componentName"),
  componentDefinition: document.querySelector("#componentDefinition"),
  currentComponent: document.querySelector("#currentComponent"),
  targetComponent: document.querySelector("#targetComponent"),
  otherComponents: document.querySelector("#otherComponents"),
  referenceText: document.querySelector("#referenceText"),
  promptTemplate: document.querySelector("#promptTemplate"),
  baselineSimilarity: document.querySelector("#baselineSimilarity"),
  validSuccessRate: document.querySelector("#validSuccessRate"),
  averageBestSimilarity: document.querySelector("#averageBestSimilarity"),
  averageImprovement: document.querySelector("#averageImprovement"),
  completedRuns: document.querySelector("#completedRuns"),
  validCandidates: document.querySelector("#validCandidates"),
  checkerStatusTone: document.querySelector("#checkerStatusTone"),
  checkerStatusTitle: document.querySelector("#checkerStatusTitle"),
  checkerStatusDetail: document.querySelector("#checkerStatusDetail"),
  checkerConnectionDot: document.querySelector("#checkerConnectionDot"),
  checkerConnectionText: document.querySelector("#checkerConnectionText"),
  renderedPromptPreview: document.querySelector("#renderedPromptPreview"),
  bestCandidateDetails: document.querySelector("#bestCandidateDetails"),
  runResultsBody: document.querySelector("#runResultsBody"),
  candidateResultsBody: document.querySelector("#candidateResultsBody"),
  clearCheckerResultsButton: document.querySelector("#clearCheckerResultsButton"),
  embeddingModelLabel: document.querySelector("#embeddingModelLabel"),
  embeddingModelSelect: document.querySelector("#embeddingModelSelect"),
  embeddingTextA: document.querySelector("#embeddingTextA"),
  embeddingTextB: document.querySelector("#embeddingTextB"),
  calculateEmbeddingButton: document.querySelector("#calculateEmbeddingButton"),
  clearEmbeddingButton: document.querySelector("#clearEmbeddingButton"),
  embeddingSimilarity: document.querySelector("#embeddingSimilarity"),
  embeddingDistance: document.querySelector("#embeddingDistance"),
  embeddingDimension: document.querySelector("#embeddingDimension"),
  embeddingStatusTone: document.querySelector("#embeddingStatusTone"),
  embeddingStatusTitle: document.querySelector("#embeddingStatusTitle"),
  embeddingStatusDetail: document.querySelector("#embeddingStatusDetail"),
  embeddingPreviewA: document.querySelector("#embeddingPreviewA"),
  embeddingPreviewB: document.querySelector("#embeddingPreviewB"),
  copyEmbeddingAButton: document.querySelector("#copyEmbeddingAButton"),
  copyEmbeddingBButton: document.querySelector("#copyEmbeddingBButton"),
  embeddingRuntimeDetails: document.querySelector("#embeddingRuntimeDetails"),
  embeddingModelDetails: document.querySelector("#embeddingModelDetails"),
};

function setStatus(toneElement, titleElement, detailElement, title, detail, state = "ready") {
  titleElement.textContent = title;
  detailElement.textContent = detail;
  toneElement.classList.toggle("is-busy", state === "busy");
  toneElement.classList.toggle("is-error", state === "error");
}

function formatNumber(value, digits = 6) {
  return new Intl.NumberFormat("es-CL", {
    maximumFractionDigits: digits,
    minimumFractionDigits: digits,
  }).format(value);
}

function clampNumber(value, min, max) {
  return Math.min(max, Math.max(min, Number(value)));
}

function dotProduct(vectorA, vectorB) {
  return vectorA.reduce((sum, value, index) => sum + value * vectorB[index], 0);
}

function vectorNorm(vector) {
  return Math.sqrt(dotProduct(vector, vector));
}

function cosineSimilarity(vectorA, vectorB) {
  const denominator = vectorNorm(vectorA) * vectorNorm(vectorB);
  if (denominator === 0) {
    throw new Error("No se puede calcular coseno con un vector de norma cero.");
  }
  return Math.min(1, Math.max(-1, dotProduct(vectorA, vectorB) / denominator));
}

function cosineDistance(similarity) {
  return 1 - similarity;
}

function renderDefinitionList(container, entries) {
  container.replaceChildren(
    ...entries.map(([term, description]) => {
      const wrapper = document.createElement("div");
      const dt = document.createElement("dt");
      const dd = document.createElement("dd");
      dt.textContent = term;
      dd.textContent = description;
      wrapper.append(dt, dd);
      return wrapper;
    }),
  );
}

function normalizeEndpoint(endpoint) {
  return endpoint.trim().replace(/\/+$/, "");
}

function endpointForMode(mode) {
  return mode === "openai" ? "/lmstudio/v1" : "/lmstudio/api/v1";
}

function selectedLlmModel() {
  return dom.llmModelSelect.value || dom.llmModelManual.value.trim();
}

function selectedEmbeddingModel() {
  return dom.embeddingModelSelect.value;
}

function renderEmbeddingModelDetails() {
  const model = EMBEDDING_MODELS[selectedEmbeddingModel()];
  dom.embeddingModelLabel.textContent = model.family;
  renderDefinitionList(dom.embeddingRuntimeDetails, [
    ["Pipeline", PIPELINE_CONFIG.task],
    ["Pooling", PIPELINE_CONFIG.pooling],
    ["Normalización", PIPELINE_CONFIG.normalize],
    ["Ejecución", PIPELINE_CONFIG.execution],
  ]);
  renderDefinitionList(dom.embeddingModelDetails, [
    ["Modelo", `${model.displayName} (${selectedEmbeddingModel()})`],
    ["Base", model.baseModel],
    ["Idioma", model.language],
    ["MTEB Average (56)", model.mtebAverage],
    ["MTEB Retrieval (15)", model.mtebRetrieval],
    ["Dimensión", model.dimensions],
    ["Entrada máxima", model.maxInput],
    ["Costo aproximado", model.approximateCost],
  ]);
}

function requestEmbeddings(modelId, texts) {
  const id = nextWorkerRequestId;
  nextWorkerRequestId += 1;

  const promise = new Promise((resolve, reject) => {
    workerRequests.set(id, { resolve, reject });
  });

  embeddingWorker.postMessage({ id, modelId, texts });
  return promise;
}

function vectorPreview(vector) {
  return `${JSON.stringify(vector.slice(0, 16), null, 2)}\n\nPrimeros 16 valores de ${vector.length}.`;
}

function resetEmbeddingResults() {
  latestEmbeddingVectors = null;
  dom.embeddingSimilarity.textContent = "--";
  dom.embeddingDistance.textContent = "--";
  dom.embeddingDimension.textContent = "--";
  dom.embeddingPreviewA.textContent = "Ejecuta el cálculo para ver el vector.";
  dom.embeddingPreviewB.textContent = "Ejecuta el cálculo para ver el vector.";
  dom.copyEmbeddingAButton.disabled = true;
  dom.copyEmbeddingBButton.disabled = true;
}

async function calculateEmbeddingPair() {
  const textA = dom.embeddingTextA.value.trim();
  const textB = dom.embeddingTextB.value.trim();
  if (!textA || !textB) {
    setStatus(dom.embeddingStatusTone, dom.embeddingStatusTitle, dom.embeddingStatusDetail, "Falta información", "Ingresa ambos textos antes de calcular.", "error");
    return;
  }

  dom.calculateEmbeddingButton.disabled = true;
  resetEmbeddingResults();
  setStatus(dom.embeddingStatusTone, dom.embeddingStatusTitle, dom.embeddingStatusDetail, "Calculando", "Preparando embeddings.", "busy");

  try {
    latestEmbeddingVectors = await requestEmbeddings(selectedEmbeddingModel(), [textA, textB]);
    const similarity = cosineSimilarity(latestEmbeddingVectors[0], latestEmbeddingVectors[1]);
    dom.embeddingSimilarity.textContent = formatNumber(similarity);
    dom.embeddingDistance.textContent = formatNumber(cosineDistance(similarity));
    dom.embeddingDimension.textContent = String(latestEmbeddingVectors[0].length);
    dom.embeddingPreviewA.textContent = vectorPreview(latestEmbeddingVectors[0]);
    dom.embeddingPreviewB.textContent = vectorPreview(latestEmbeddingVectors[1]);
    dom.copyEmbeddingAButton.disabled = false;
    dom.copyEmbeddingBButton.disabled = false;
    setStatus(dom.embeddingStatusTone, dom.embeddingStatusTitle, dom.embeddingStatusDetail, "Cálculo completado", `Embeddings generados con ${EMBEDDING_MODELS[selectedEmbeddingModel()].displayName}.`);
  } catch (error) {
    setStatus(dom.embeddingStatusTone, dom.embeddingStatusTitle, dom.embeddingStatusDetail, "Error al calcular", error.message, "error");
  } finally {
    dom.calculateEmbeddingButton.disabled = false;
  }
}

function resetCheckerMetrics() {
  dom.baselineSimilarity.textContent = "--";
  dom.validSuccessRate.textContent = "--";
  dom.averageBestSimilarity.textContent = "--";
  dom.averageImprovement.textContent = "--";
  dom.completedRuns.textContent = "--";
  dom.validCandidates.textContent = "--";
  dom.bestCandidateDetails.replaceChildren();
}

function setCheckerRunning(isRunning) {
  dom.runTemplateButton.disabled = isRunning;
  dom.stopTemplateButton.disabled = !isRunning;
  dom.loadModelsButton.disabled = isRunning;
  dom.previewPromptButton.disabled = isRunning;
}

function getTemplateBody(template) {
  const match = template.match(/^\s*[A-Za-z_][A-Za-z0-9_]*\s*=\s*([\"']{3})([\s\S]*?)\1\s*$/);
  return (match ? match[2] : template).trim();
}

function checkerVariables() {
  return {
    num_candidates: dom.candidateCount.value.trim(),
    component_name: dom.componentName.value.trim(),
    component_definition: dom.componentDefinition.value.trim(),
    current_component: dom.currentComponent.value.trim(),
    target_component: dom.targetComponent.value.trim(),
    other_components: dom.otherComponents.value.trim(),
    reference_text: dom.referenceText.value.trim(),
  };
}

function renderPrompt() {
  const variables = checkerVariables();
  return getTemplateBody(dom.promptTemplate.value).replace(/\{([A-Za-z0-9_]+)\}/g, (match, key) => (
    Object.hasOwn(variables, key) ? variables[key] : match
  ));
}

function wordCount(text) {
  return text.trim().split(/\s+/).filter(Boolean).length;
}

function normalizeCandidate(line) {
  return line
    .trim()
    .replace(/^```[A-Za-z]*\s*/g, "")
    .replace(/```$/g, "")
    .replace(/^\s*(?:[-*]|\d+[\).\]:-])\s*/u, "")
    .replace(/^candidate\s*\d*\s*[:.-]\s*/iu, "")
    .replace(/^["'`]+|["'`]+$/g, "")
    .replace(/^<|>$/g, "")
    .trim();
}

function parseCandidates(rawOutput, expectedCount) {
  const cleaned = rawOutput
    .replace(/```[A-Za-z]*\n?/g, "")
    .replace(/```/g, "")
    .trim();
  const seen = new Set();
  const candidates = [];

  cleaned.split(/\r?\n/).forEach((line) => {
    const candidate = normalizeCandidate(line);
    const key = candidate.toLocaleLowerCase();
    if (candidate && !seen.has(key)) {
      seen.add(key);
      candidates.push(candidate);
    }
  });

  return candidates.slice(0, expectedCount);
}

function validationReason(candidate, runCandidates, config) {
  const lower = candidate.toLocaleLowerCase();
  const current = config.current.toLocaleLowerCase();
  const target = config.target.toLocaleLowerCase();
  const words = wordCount(candidate);

  if (words < config.minWords || words > config.maxWords) {
    return "invalid_length";
  }
  if (lower === current) {
    return "literal_copy_current";
  }
  if (lower === target) {
    return "literal_copy_target";
  }
  if (runCandidates.filter((item) => item.toLocaleLowerCase() === lower).length > 1) {
    return "duplicate_output";
  }
  return "valid";
}

async function callLmStudio(prompt, runNumber, config) {
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), config.timeoutMs);
  const started = performance.now();
  const isNativeApi = config.apiMode === "native";
  const requestUrl = isNativeApi ? `${config.endpoint}/chat` : `${config.endpoint}/chat/completions`;
  const requestBody = isNativeApi
    ? {
        model: config.model,
        input: prompt,
        temperature: config.temperature,
        top_p: config.topP,
        max_output_tokens: config.maxTokens,
        stream: false,
        store: false,
      }
    : {
        model: config.model,
        messages: [{ role: "user", content: prompt }],
        temperature: config.temperature,
        top_p: config.topP,
        max_tokens: config.maxTokens,
        stream: false,
      };

  try {
    const response = await fetch(requestUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      signal: controller.signal,
      body: JSON.stringify(requestBody),
    });

    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(payload.error?.message || `LM Studio respondió HTTP ${response.status}`);
    }

    return {
      raw: extractLmStudioText(payload, config.apiMode),
      elapsedMs: performance.now() - started,
      usage: payload.usage || payload.stats || null,
      runNumber,
    };
  } finally {
    window.clearTimeout(timeout);
  }
}

function extractLmStudioText(payload, apiMode) {
  if (apiMode === "native") {
    const output = Array.isArray(payload.output) ? payload.output : [];
    return output
      .filter((item) => item?.type === "message" && typeof item.content === "string")
      .map((item) => item.content)
      .join("\n")
      .trim();
  }

  return payload.choices?.[0]?.message?.content || "";
}

async function fetchLmStudioModels() {
  const endpoint = normalizeEndpoint(dom.lmEndpoint.value);
  dom.lmEndpoint.value = endpoint;
  dom.loadModelsButton.disabled = true;
  dom.checkerConnectionDot.classList.add("is-busy");
  dom.checkerConnectionDot.classList.remove("is-error");
  dom.checkerConnectionText.textContent = "Consultando LM Studio";

  try {
    const response = await fetch(`${endpoint}/models`);
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.error?.message || `HTTP ${response.status}`);
    }

    const models = parseLmStudioModels(payload, dom.lmApiMode.value);
    dom.llmModelSelect.replaceChildren(
      ...models.map((model) => {
        const option = document.createElement("option");
        option.value = model.id;
        option.textContent = model.label;
        return option;
      }),
    );

    if (models.length > 0) {
      const loaded = models.find((model) => model.loaded) || models[0];
      dom.llmModelSelect.value = loaded.id;
      dom.llmModelManual.value = loaded.id;
    }

    dom.checkerConnectionText.textContent = `${models.length} modelo(s) disponibles`;
  } catch (error) {
    dom.checkerConnectionDot.classList.add("is-error");
    dom.checkerConnectionText.textContent = "Error de conexión";
    setStatus(dom.checkerStatusTone, dom.checkerStatusTitle, dom.checkerStatusDetail, "No se pudo conectar a LM Studio", error.message, "error");
  } finally {
    dom.checkerConnectionDot.classList.remove("is-busy");
    dom.loadModelsButton.disabled = false;
  }
}

function parseLmStudioModels(payload, apiMode) {
  if (apiMode === "native") {
    return (Array.isArray(payload.models) ? payload.models : [])
      .filter((model) => model.type === "llm")
      .map((model) => {
        const loadedInstance = Array.isArray(model.loaded_instances) ? model.loaded_instances[0] : null;
        const id = loadedInstance?.id || model.key;
        const label = `${model.display_name || model.key}${loadedInstance ? " (cargado)" : ""}`;
        return { id, label, loaded: Boolean(loadedInstance) };
      });
  }

  return (Array.isArray(payload.data) ? payload.data : [])
    .map((model) => ({ id: model.id, label: model.id, loaded: true }));
}

function readCheckerConfig() {
  const endpoint = normalizeEndpoint(dom.lmEndpoint.value);
  const model = selectedLlmModel();
  const variables = checkerVariables();

  if (!endpoint) throw new Error("Define el endpoint de LM Studio.");
  if (!model) throw new Error("Selecciona o escribe el modelo LLM.");
  if (!variables.current_component || !variables.target_component) {
    throw new Error("Current y Target son obligatorios.");
  }

  return {
    endpoint,
    apiMode: dom.lmApiMode.value,
    model,
    embeddingModel: dom.checkerEmbeddingModel.value,
    runs: clampNumber(dom.runCount.value, 1, 200),
    expectedCandidates: clampNumber(dom.candidateCount.value, 1, 20),
    temperature: clampNumber(dom.temperature.value, 0, 2),
    topP: clampNumber(dom.topP.value, 0, 1),
    maxTokens: clampNumber(dom.maxTokens.value, 8, 2048),
    timeoutMs: clampNumber(dom.requestTimeout.value, 5, 600) * 1000,
    minWords: clampNumber(dom.minWords.value, 1, 20),
    maxWords: clampNumber(dom.maxWords.value, 1, 40),
    margin: clampNumber(dom.semanticMargin.value, 0, 1),
    targetCopyThreshold: clampNumber(dom.targetCopyThreshold.value, 0, 1),
    concurrency: clampNumber(dom.concurrency.value, 1, 8),
    current: variables.current_component,
    target: variables.target_component,
    prompt: renderPrompt(),
  };
}

function clearCheckerTables() {
  latestCheckerRows = [];
  dom.runResultsBody.innerHTML = '<tr><td colspan="6">Sin ejecuciones todavía.</td></tr>';
  dom.candidateResultsBody.innerHTML = '<tr><td colspan="6">Sin candidatos todavía.</td></tr>';
}

function appendCandidateRows(rows) {
  if (latestCheckerRows.length === 0) {
    dom.candidateResultsBody.replaceChildren();
  }

  latestCheckerRows.push(...rows);
  rows.forEach((row) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${row.runNumber}</td>
      <td>${escapeHtml(row.candidate)}</td>
      <td>${formatNumber(row.similarity)}</td>
      <td>${formatNumber(cosineDistance(row.similarity))}</td>
      <td>${formatSigned(row.improvement)}</td>
      <td><span class="${row.isValidImprovement ? "valid" : "invalid"}">${row.reason}</span></td>
    `;
    dom.candidateResultsBody.append(tr);
  });
}

function renderRunRows(runResults) {
  if (runResults.length === 0) {
    dom.runResultsBody.innerHTML = '<tr><td colspan="6">Sin ejecuciones todavía.</td></tr>';
    return;
  }

  dom.runResultsBody.replaceChildren(
    ...runResults.map((run) => {
      const tr = document.createElement("tr");
      const best = run.bestValidCandidate;
      tr.innerHTML = `
        <td>${run.runNumber}</td>
        <td>${best ? escapeHtml(best.candidate) : "--"}</td>
        <td>${best ? formatNumber(best.similarity) : "--"}</td>
        <td>${best ? formatSigned(best.improvement) : "--"}</td>
        <td>${run.validImprovementCount}/${run.candidateCount}</td>
        <td>${run.error ? `<span class="invalid">${escapeHtml(run.error)}</span>` : '<span class="valid">ok</span>'}</td>
      `;
      return tr;
    }),
  );
}

function formatSigned(value) {
  const sign = value > 0 ? "+" : "";
  return `${sign}${formatNumber(value)}`;
}

function escapeHtml(value) {
  return value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function summarizeCheckerResults(runResults, baseline) {
  const completed = runResults.filter((run) => !run.error);
  const validSuccesses = completed.filter((run) => run.bestValidCandidate);
  const validCandidates = latestCheckerRows.filter((row) => row.reason === "valid").length;
  const bestCandidates = validSuccesses.map((run) => run.bestValidCandidate);
  const bestOverall = bestCandidates.reduce((best, row) => (!best || row.similarity > best.similarity ? row : best), null);
  const avgBestSimilarity = bestCandidates.length
    ? bestCandidates.reduce((sum, row) => sum + row.similarity, 0) / bestCandidates.length
    : null;
  const avgImprovement = bestCandidates.length
    ? bestCandidates.reduce((sum, row) => sum + row.improvement, 0) / bestCandidates.length
    : null;

  dom.baselineSimilarity.textContent = formatNumber(baseline);
  dom.validSuccessRate.textContent = completed.length ? `${formatNumber((validSuccesses.length / completed.length) * 100, 2)}%` : "--";
  dom.averageBestSimilarity.textContent = avgBestSimilarity === null ? "--" : formatNumber(avgBestSimilarity);
  dom.averageImprovement.textContent = avgImprovement === null ? "--" : formatSigned(avgImprovement);
  dom.completedRuns.textContent = `${completed.length}/${runResults.length}`;
  dom.validCandidates.textContent = String(validCandidates);

  renderDefinitionList(dom.bestCandidateDetails, bestOverall ? [
    ["Candidato", bestOverall.candidate],
    ["Run", String(bestOverall.runNumber)],
    ["Similitud", formatNumber(bestOverall.similarity)],
    ["Distancia", formatNumber(cosineDistance(bestOverall.similarity))],
    ["Mejora vs current", formatSigned(bestOverall.improvement)],
  ] : [
    ["Resultado", "Ningún candidato válido superó el baseline con el margen configurado."],
  ]);
}

async function scoreRunCandidates(runNumber, rawOutput, baseline, config) {
  const candidates = parseCandidates(rawOutput, config.expectedCandidates);
  if (candidates.length === 0) {
    return {
      runResult: {
        runNumber,
        candidateCount: 0,
        validImprovementCount: 0,
        bestValidCandidate: null,
        error: "sin candidatos parseables",
      },
      rows: [],
    };
  }

  const embeddings = await requestEmbeddings(config.embeddingModel, [config.target, ...candidates]);
  const targetEmbedding = embeddings[0];
  const rows = candidates.map((candidate, index) => {
    const similarity = cosineSimilarity(embeddings[index + 1], targetEmbedding);
    const improvement = similarity - baseline;
    const baseReason = validationReason(candidate, candidates, config);
    const hasSemanticProgress = improvement >= config.margin;
    const reason = baseReason !== "valid"
      ? baseReason
      : similarity >= config.targetCopyThreshold ? "semantic_copy_target"
      : hasSemanticProgress ? "valid" : "insufficient_semantic_progress";
    const isValidImprovement = reason === "valid";
    return { runNumber, candidate, similarity, improvement, reason, isValidImprovement };
  });

  const bestValidCandidate = rows
    .filter((row) => row.isValidImprovement)
    .sort((a, b) => b.similarity - a.similarity)[0] || null;

  return {
    runResult: {
      runNumber,
      candidateCount: rows.length,
      validImprovementCount: rows.filter((row) => row.isValidImprovement).length,
      bestValidCandidate,
      error: null,
    },
    rows,
  };
}

async function runTemplateEvaluation() {
  let config;
  try {
    config = readCheckerConfig();
  } catch (error) {
    setStatus(dom.checkerStatusTone, dom.checkerStatusTitle, dom.checkerStatusDetail, "Configuración incompleta", error.message, "error");
    return;
  }

  stopRequested = false;
  clearCheckerTables();
  resetCheckerMetrics();
  dom.renderedPromptPreview.textContent = config.prompt;
  setCheckerRunning(true);
  setStatus(dom.checkerStatusTone, dom.checkerStatusTitle, dom.checkerStatusDetail, "Calculando baseline", "Generando embeddings de Current y Target.", "busy");

  const runResults = [];
  let nextRun = 1;

  try {
    const [currentEmbedding, targetEmbedding] = await requestEmbeddings(config.embeddingModel, [config.current, config.target]);
    const baseline = cosineSimilarity(currentEmbedding, targetEmbedding);
    dom.baselineSimilarity.textContent = formatNumber(baseline);

    async function workerLoop() {
      while (!stopRequested && nextRun <= config.runs) {
        const runNumber = nextRun;
        nextRun += 1;
        setStatus(dom.checkerStatusTone, dom.checkerStatusTitle, dom.checkerStatusDetail, "Ejecutando plantillas", `Ejecución ${runNumber} de ${config.runs}.`, "busy");

        try {
          const response = await callLmStudio(config.prompt, runNumber, config);
          const scored = await scoreRunCandidates(runNumber, response.raw, baseline, config);
          runResults.push(scored.runResult);
          appendCandidateRows(scored.rows);
        } catch (error) {
          runResults.push({
            runNumber,
            candidateCount: 0,
            validImprovementCount: 0,
            bestValidCandidate: null,
            error: error.name === "AbortError" ? "timeout" : error.message,
          });
        }

        runResults.sort((a, b) => a.runNumber - b.runNumber);
        renderRunRows(runResults);
        summarizeCheckerResults(runResults, baseline);
      }
    }

    await Promise.all(Array.from({ length: config.concurrency }, () => workerLoop()));
    const finalState = stopRequested ? "Evaluación detenida" : "Evaluación completada";
    setStatus(dom.checkerStatusTone, dom.checkerStatusTitle, dom.checkerStatusDetail, finalState, `${runResults.length} ejecución(es) procesada(s).`);
  } catch (error) {
    setStatus(dom.checkerStatusTone, dom.checkerStatusTitle, dom.checkerStatusDetail, "Error en evaluación", error.message, "error");
  } finally {
    setCheckerRunning(false);
  }
}

function applyComponentPreset() {
  const preset = COMPONENT_PRESETS[dom.componentPreset.value];
  if (!preset) return;
  dom.componentName.value = preset.name;
  dom.componentDefinition.value = preset.definition;
  dom.currentComponent.value = preset.current;
  dom.targetComponent.value = preset.target;
  dom.otherComponents.value = preset.other;
}

embeddingWorker.addEventListener("message", (event) => {
  const { id, type, embeddings, message, progress } = event.data;
  const request = workerRequests.get(id);

  if (type === "progress") {
    const file = progress?.file ? `: ${progress.file}` : "";
    const pct = typeof progress?.progress === "number" ? ` (${Math.round(progress.progress)}%)` : "";
    setStatus(dom.embeddingStatusTone, dom.embeddingStatusTitle, dom.embeddingStatusDetail, "Descargando modelo", `${progress?.status || "progreso"}${file}${pct}`, "busy");
    return;
  }

  if (!request) return;
  workerRequests.delete(id);

  if (type === "result") {
    request.resolve(embeddings);
    return;
  }

  request.reject(new Error(message));
});

embeddingWorker.addEventListener("error", (event) => {
  workerRequests.forEach(({ reject }) => reject(new Error(event.message)));
  workerRequests.clear();
});

dom.navItems.forEach((button) => {
  button.addEventListener("click", () => {
    dom.navItems.forEach((item) => item.classList.toggle("is-active", item === button));
    dom.panels.forEach((panel) => panel.classList.toggle("is-active", panel.id === button.dataset.tool));
  });
});

dom.loadModelsButton.addEventListener("click", fetchLmStudioModels);
dom.previewPromptButton.addEventListener("click", () => {
  dom.renderedPromptPreview.textContent = renderPrompt();
});
dom.runTemplateButton.addEventListener("click", runTemplateEvaluation);
dom.stopTemplateButton.addEventListener("click", () => {
  stopRequested = true;
  setStatus(dom.checkerStatusTone, dom.checkerStatusTitle, dom.checkerStatusDetail, "Deteniendo", "Se terminarán las solicitudes ya iniciadas.", "busy");
});
dom.clearCheckerResultsButton.addEventListener("click", () => {
  clearCheckerTables();
  resetCheckerMetrics();
});
dom.componentPreset.addEventListener("change", applyComponentPreset);
dom.llmModelSelect.addEventListener("change", () => {
  if (dom.llmModelSelect.value) {
    dom.llmModelManual.value = dom.llmModelSelect.value;
  }
});
dom.lmApiMode.addEventListener("change", () => {
  dom.lmEndpoint.value = endpointForMode(dom.lmApiMode.value);
  dom.llmModelSelect.replaceChildren(new Option("Cargar modelos desde LM Studio", ""));
  dom.llmModelManual.value = dom.lmApiMode.value === "native" ? "meta-llama-3.1-8b-instruct" : "meta-llama-3.1-8b-instruct";
  dom.checkerConnectionText.textContent = "Sin probar conexión";
  dom.checkerConnectionDot.classList.remove("is-error", "is-busy");
});

dom.calculateEmbeddingButton.addEventListener("click", calculateEmbeddingPair);
dom.clearEmbeddingButton.addEventListener("click", () => {
  dom.embeddingTextA.value = "";
  dom.embeddingTextB.value = "";
  resetEmbeddingResults();
  setStatus(dom.embeddingStatusTone, dom.embeddingStatusTitle, dom.embeddingStatusDetail, "Listo para calcular", "Ingresa dos textos para generar sus embeddings.");
});
dom.embeddingModelSelect.addEventListener("change", () => {
  resetEmbeddingResults();
  renderEmbeddingModelDetails();
  setStatus(dom.embeddingStatusTone, dom.embeddingStatusTitle, dom.embeddingStatusDetail, "Modelo cambiado", `Seleccionaste ${EMBEDDING_MODELS[selectedEmbeddingModel()].displayName}.`);
});
dom.copyEmbeddingAButton.addEventListener("click", () => {
  if (latestEmbeddingVectors) navigator.clipboard.writeText(JSON.stringify(latestEmbeddingVectors[0], null, 2));
});
dom.copyEmbeddingBButton.addEventListener("click", () => {
  if (latestEmbeddingVectors) navigator.clipboard.writeText(JSON.stringify(latestEmbeddingVectors[1], null, 2));
});

renderEmbeddingModelDetails();
dom.renderedPromptPreview.textContent = renderPrompt();
