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
  batching: "Embedding canónico por texto individual",
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

const PSO_COMPONENTS = [
  {
    key: "rol",
    promptName: "role",
    definition: "speaker identity or perspective",
  },
  {
    key: "topico",
    promptName: "topic",
    definition: "subject or event focus",
  },
  {
    key: "accion",
    promptName: "action",
    definition: "communicative intent or requested operation",
  },
];

const PSO_OPERATORS = {
  influence: {
    label: "Influencia pbest/gbest",
    statusLabel: "pbest/gbest",
  },
  turbulence: {
    label: "Turbulencia",
    statusLabel: "turbulencia",
  },
};

const SOLUTION_GENERATION_SYSTEM_PROMPT = `You are a plain-text generator for natural-disaster scenario messages. You will receive one text-generation instruction from the user. Follow the instruction and generate exactly one final text message.

Output rules:
- Return plain text only.
- Return the dataset content itself, not a prompt, explanation, title, label, list, code, or metadata.
- Do not describe the task.
- Do not add unsolicited safety advice.
- Do not use quotation marks, hashtags, URLs, usernames, placeholders, tags, or special markers.
- Limit the message to between 1 and 4 sentences.
- Return only the final message.`;

const SOLUTION_REFERENCE_PRESETS = [
  {
    id: "nicolas_food",
    label: "Nicolas: racionamiento de alimentos",
    source: "Memoria de Nicolas Meneses, escenario de calibracion 1.",
    referenceText: "We're rationing food. Only one store open and it's chaos",
    roles: [
      "affected resident seeking supplies",
      "parent worried about food access",
      "store customer reporting scarcity",
      "neighborhood volunteer coordinating aid",
      "local organizer monitoring supply lines",
      "community member facing rationing",
      "family caregiver near crowded stores",
      "resident documenting food shortages",
    ],
    topics: [
      "food rationing during emergency",
      "limited grocery access",
      "crowded supply distribution",
      "household food insecurity",
      "chaotic store conditions",
      "scarce essential supplies",
      "neighborhood aid coordination",
      "urgent food access",
    ],
    actions: [
      "report supply shortage",
      "request coordinated food aid",
      "warn about store crowding",
      "ask for emergency distribution",
      "summarize rationing impact",
      "share local scarcity update",
      "request support for families",
      "document urgent food needs",
    ],
  },
  {
    id: "nicolas_cleanup",
    label: "Nicolas: coordinacion de limpieza",
    source: "Memoria de Nicolas Meneses, escenario de calibracion 2.",
    referenceText: "We're organizing a cleanup drive in the park at 9 AM",
    roles: [
      "community volunteer coordinator",
      "neighbor organizing cleanup work",
      "local resident mobilizing helpers",
      "park volunteer team leader",
      "municipal liaison sharing cleanup plans",
      "student volunteer coordinating attendance",
      "resident reporting community action",
      "aid organizer preparing a work crew",
    ],
    topics: [
      "park cleanup coordination",
      "volunteer mobilization after damage",
      "scheduled community cleanup",
      "debris removal planning",
      "neighborhood recovery work",
      "morning cleanup logistics",
      "public space restoration",
      "community disaster response",
    ],
    actions: [
      "invite volunteers to attend",
      "announce cleanup schedule",
      "coordinate tools and helpers",
      "request local participation",
      "share recovery logistics",
      "organize debris removal",
      "confirm meeting time",
      "summarize cleanup plan",
    ],
  },
  {
    id: "nicolas_water",
    label: "Nicolas: tratamiento de agua",
    source: "Memoria de Nicolas Meneses, escenario de calibracion 3.",
    referenceText: "Please conserve water as our treatment facilities are still offline",
    roles: [
      "local authority issuing service update",
      "water utility representative",
      "municipal emergency communicator",
      "public works official",
      "community leader sharing water guidance",
      "resident relay for official alerts",
      "infrastructure coordinator",
      "emergency operations spokesperson",
    ],
    topics: [
      "water conservation during outage",
      "offline treatment facilities",
      "critical service disruption",
      "public water restrictions",
      "infrastructure recovery delay",
      "safe water management",
      "municipal utility outage",
      "household conservation guidance",
    ],
    actions: [
      "ask residents to conserve water",
      "report treatment facility outage",
      "share official service update",
      "warn about limited water supply",
      "request reduced household use",
      "explain ongoing infrastructure issue",
      "summarize conservation measures",
      "notify community about water limits",
    ],
  },
  {
    id: "diego_disaster_response",
    label: "Diego: respuesta adaptativa",
    source: "Paper EVOLMD-MO, resumen sobre disaster response.",
    referenceText: "adaptive stream processing for disaster response",
    roles: [
      "disaster analyst monitoring live reports",
      "emergency coordinator tracking incidents",
      "data operator supporting response teams",
      "field responder reporting urgent events",
      "public safety analyst",
      "crisis operations specialist",
      "community alert coordinator",
      "response planner reviewing data streams",
    ],
    topics: [
      "adaptive disaster response",
      "live emergency stream processing",
      "rapid incident monitoring",
      "dynamic crisis data analysis",
      "synthetic disaster message generation",
      "emergency information triage",
      "real time response coordination",
      "disaster training data quality",
    ],
    actions: [
      "generate actionable alert text",
      "summarize changing emergency conditions",
      "report incident signals clearly",
      "support response team prioritization",
      "describe urgent public safety needs",
      "create concise disaster updates",
      "highlight relevant response context",
      "inform adaptive processing models",
    ],
  },
];

const SOLUTION_COST_ASSUMPTIONS = {
  evolmdMutationProbability: 0.05,
  mesapCallsPerSolution: 3,
  mesapRuntimeMultiplier: 1.18,
};

const workerRequests = new Map();
let nextWorkerRequestId = 1;
let stopRequested = false;
let solutionEvalStopRequested = false;
let latestEmbeddingVectors = null;
let latestCheckerRows = [];
let latestSolutionRows = [];
let psoDatabase = null;
let comparatorPollTimer = null;
let currentComparatorRunId = null;
let initialPopulationPollTimer = null;
let currentInitialPopulationRunId = null;

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
  simulatePsoSwitch: document.querySelector("#simulatePsoSwitch"),
  psoControls: document.querySelector("#psoControls"),
  psoSemanticOperator: document.querySelector("#psoSemanticOperator"),
  psoIndividualCount: document.querySelector("#psoIndividualCount"),
  psoChangeThreshold: document.querySelector("#psoChangeThreshold"),
  turbulenceMinSimilarity: document.querySelector("#turbulenceMinSimilarity"),
  turbulenceMaxSimilarity: document.querySelector("#turbulenceMaxSimilarity"),
  turbulenceControls: document.querySelectorAll("[data-turbulence-control]"),
  psoDbStatus: document.querySelector("#psoDbStatus"),
  componentPreset: document.querySelector("#componentPreset"),
  componentName: document.querySelector("#componentName"),
  componentDefinition: document.querySelector("#componentDefinition"),
  currentComponent: document.querySelector("#currentComponent"),
  targetComponent: document.querySelector("#targetComponent"),
  otherComponents: document.querySelector("#otherComponents"),
  referenceText: document.querySelector("#referenceText"),
  promptTemplate: document.querySelector("#promptTemplate"),
  baselineSimilarity: document.querySelector("#baselineSimilarity"),
  templateMetrics: document.querySelector(".metrics-grid[aria-label='Resumen de evaluación']"),
  validSuccessRate: document.querySelector("#validSuccessRate"),
  averageBestSimilarity: document.querySelector("#averageBestSimilarity"),
  averageImprovement: document.querySelector("#averageImprovement"),
  completedRuns: document.querySelector("#completedRuns"),
  validCandidates: document.querySelector("#validCandidates"),
  psoMetrics: document.querySelector("#psoMetrics"),
  psoSelectedComponents: document.querySelector("#psoSelectedComponents"),
  psoFailedChanges: document.querySelector("#psoFailedChanges"),
  psoFailureRate: document.querySelector("#psoFailureRate"),
  psoAppliedChanges: document.querySelector("#psoAppliedChanges"),
  psoAffectedIndividuals: document.querySelector("#psoAffectedIndividuals"),
  psoLlmCalls: document.querySelector("#psoLlmCalls"),
  psoAverageLlmTime: document.querySelector("#psoAverageLlmTime"),
  psoTotalLlmTime: document.querySelector("#psoTotalLlmTime"),
  psoValidCandidates: document.querySelector("#psoValidCandidates"),
  psoProgressFailures: document.querySelector("#psoProgressFailures"),
  psoExecutionErrors: document.querySelector("#psoExecutionErrors"),
  checkerStatusTone: document.querySelector("#checkerStatusTone"),
  checkerStatusTitle: document.querySelector("#checkerStatusTitle"),
  checkerStatusDetail: document.querySelector("#checkerStatusDetail"),
  checkerConnectionDot: document.querySelector("#checkerConnectionDot"),
  checkerConnectionText: document.querySelector("#checkerConnectionText"),
  renderedPromptPreview: document.querySelector("#renderedPromptPreview"),
  bestCandidateDetails: document.querySelector("#bestCandidateDetails"),
  runResultsTitle: document.querySelector("#runResultsTitle"),
  runResultsHead: document.querySelector("#runResultsHead"),
  runResultsBody: document.querySelector("#runResultsBody"),
  candidateResultsTitle: document.querySelector("#candidateResultsTitle"),
  candidateResultsHead: document.querySelector("#candidateResultsHead"),
  candidateResultsBody: document.querySelector("#candidateResultsBody"),
  candidateResultsTable: document.querySelector("#candidateResultsHead").closest("table"),
  clearCheckerResultsButton: document.querySelector("#clearCheckerResultsButton"),
  solutionLmEndpoint: document.querySelector("#solutionLmEndpoint"),
  solutionLmApiMode: document.querySelector("#solutionLmApiMode"),
  solutionLlmModelSelect: document.querySelector("#solutionLlmModelSelect"),
  solutionLlmModelManual: document.querySelector("#solutionLlmModelManual"),
  solutionCount: document.querySelector("#solutionCount"),
  solutionTemperature: document.querySelector("#solutionTemperature"),
  solutionTopP: document.querySelector("#solutionTopP"),
  solutionMaxTokens: document.querySelector("#solutionMaxTokens"),
  solutionRequestTimeout: document.querySelector("#solutionRequestTimeout"),
  solutionConcurrency: document.querySelector("#solutionConcurrency"),
  loadSolutionModelsButton: document.querySelector("#loadSolutionModelsButton"),
  previewSolutionsButton: document.querySelector("#previewSolutionsButton"),
  runSolutionsButton: document.querySelector("#runSolutionsButton"),
  stopSolutionsButton: document.querySelector("#stopSolutionsButton"),
  solutionReferencePreset: document.querySelector("#solutionReferencePreset"),
  solutionReferenceText: document.querySelector("#solutionReferenceText"),
  solutionReferenceSource: document.querySelector("#solutionReferenceSource"),
  solutionEmbeddingModel: document.querySelector("#solutionEmbeddingModel"),
  solutionDiversityBasis: document.querySelector("#solutionDiversityBasis"),
  solutionFidelityMin: document.querySelector("#solutionFidelityMin"),
  solutionCopyMax: document.querySelector("#solutionCopyMax"),
  solutionPromptTemplate: document.querySelector("#solutionPromptTemplate"),
  solutionLlmCalls: document.querySelector("#solutionLlmCalls"),
  solutionLlmTotalMs: document.querySelector("#solutionLlmTotalMs"),
  solutionFlowTotalMs: document.querySelector("#solutionFlowTotalMs"),
  solutionNonDominatedCount: document.querySelector("#solutionNonDominatedCount"),
  solutionEvolmdCost: document.querySelector("#solutionEvolmdCost"),
  solutionEvolmdCostDetail: document.querySelector("#solutionEvolmdCostDetail"),
  solutionMesapCost: document.querySelector("#solutionMesapCost"),
  solutionMesapCostDetail: document.querySelector("#solutionMesapCostDetail"),
  solutionStatusTone: document.querySelector("#solutionStatusTone"),
  solutionStatusTitle: document.querySelector("#solutionStatusTitle"),
  solutionStatusDetail: document.querySelector("#solutionStatusDetail"),
  solutionConnectionDot: document.querySelector("#solutionConnectionDot"),
  solutionConnectionText: document.querySelector("#solutionConnectionText"),
  solutionPromptPreview: document.querySelector("#solutionPromptPreview"),
  solutionObjectiveDetails: document.querySelector("#solutionObjectiveDetails"),
  solutionResultsBody: document.querySelector("#solutionResultsBody"),
  clearSolutionResultsButton: document.querySelector("#clearSolutionResultsButton"),
  initialLmApiMode: document.querySelector("#initialLmApiMode"),
  initialLmStudioBase: document.querySelector("#initialLmStudioBase"),
  initialPopulationLmModels: document.querySelector("#initialPopulationLmModels"),
  initialN: document.querySelector("#initialN"),
  initialTopK: document.querySelector("#initialTopK"),
  initialGenerationParallelism: document.querySelector("#initialGenerationParallelism"),
  initialTimeoutSeconds: document.querySelector("#initialTimeoutSeconds"),
  initialSeed: document.querySelector("#initialSeed"),
  initialEmbeddingModel: document.querySelector("#initialEmbeddingModel"),
  initialReferenceText: document.querySelector("#initialReferenceText"),
  initialDomain: document.querySelector("#initialDomain"),
  initialRolesMaxWords: document.querySelector("#initialRolesMaxWords"),
  initialTopicsMaxWords: document.querySelector("#initialTopicsMaxWords"),
  initialActionsMaxWords: document.querySelector("#initialActionsMaxWords"),
  initialGeneratedMinWords: document.querySelector("#initialGeneratedMinWords"),
  initialGeneratedMaxWords: document.querySelector("#initialGeneratedMaxWords"),
  initialStrategySummary: document.querySelector("#initialStrategySummary"),
  initialIntegrationDetails: document.querySelector("#initialIntegrationDetails"),
  initialPromptTemplate: document.querySelector("#initialPromptTemplate"),
  initialStageConfigs: document.querySelectorAll(".initial-stage-config"),
  loadInitialModelsButton: document.querySelector("#loadInitialModelsButton"),
  runInitialPopulationButton: document.querySelector("#runInitialPopulationButton"),
  cancelInitialPopulationButton: document.querySelector("#cancelInitialPopulationButton"),
  clearInitialPopulationButton: document.querySelector("#clearInitialPopulationButton"),
  initialPopulationConnectionDot: document.querySelector("#initialPopulationConnectionDot"),
  initialPopulationConnectionText: document.querySelector("#initialPopulationConnectionText"),
  initialRunStatus: document.querySelector("#initialRunStatus"),
  initialProgressPercent: document.querySelector("#initialProgressPercent"),
  initialProgressSummary: document.querySelector("#initialProgressSummary"),
  initialLlmCalls: document.querySelector("#initialLlmCalls"),
  initialLlmCallsDetail: document.querySelector("#initialLlmCallsDetail"),
  initialWallClock: document.querySelector("#initialWallClock"),
  initialSequentialWallClock: document.querySelector("#initialSequentialWallClock"),
  initialLlmTime: document.querySelector("#initialLlmTime"),
  initialEmbeddingTime: document.querySelector("#initialEmbeddingTime"),
  initialFinalIndividuals: document.querySelector("#initialFinalIndividuals"),
  initialNonDominated: document.querySelector("#initialNonDominated"),
  initialHypervolume: document.querySelector("#initialHypervolume"),
  initialSpread: document.querySelector("#initialSpread"),
  initialCallsPerIndividual: document.querySelector("#initialCallsPerIndividual"),
  initialRunId: document.querySelector("#initialRunId"),
  initialStatusTone: document.querySelector("#initialStatusTone"),
  initialStatusTitle: document.querySelector("#initialStatusTitle"),
  initialStatusDetail: document.querySelector("#initialStatusDetail"),
  initialProgressDetail: document.querySelector("#initialProgressDetail"),
  initialProgressBar: document.querySelector("#initialProgressBar"),
  initialProgressDetails: document.querySelector("#initialProgressDetails"),
  initialAnchorsContent: document.querySelector("#initialAnchorsContent"),
  initialPoolsContent: document.querySelector("#initialPoolsContent"),
  initialResultsBody: document.querySelector("#initialResultsBody"),
  initialArtifactDetails: document.querySelector("#initialArtifactDetails"),
  initialLogOutput: document.querySelector("#initialLogOutput"),
  comparatorReferenceText: document.querySelector("#comparatorReferenceText"),
  comparatorModel: document.querySelector("#comparatorModel"),
  comparatorTopK: document.querySelector("#comparatorTopK"),
  comparatorProposalParallelism: document.querySelector("#comparatorProposalParallelism"),
  comparatorTimeoutMinutes: document.querySelector("#comparatorTimeoutMinutes"),
  comparatorN: document.querySelector("#comparatorN"),
  comparatorGeneraciones: document.querySelector("#comparatorGeneraciones"),
  comparatorK: document.querySelector("#comparatorK"),
  comparatorProbCrossover: document.querySelector("#comparatorProbCrossover"),
  comparatorProbMutacion: document.querySelector("#comparatorProbMutacion"),
  runComparatorButton: document.querySelector("#runComparatorButton"),
  cancelComparatorButton: document.querySelector("#cancelComparatorButton"),
  clearComparatorButton: document.querySelector("#clearComparatorButton"),
  comparatorConnectionDot: document.querySelector("#comparatorConnectionDot"),
  comparatorConnectionText: document.querySelector("#comparatorConnectionText"),
  comparatorRunStatus: document.querySelector("#comparatorRunStatus"),
  comparatorProgressPercent: document.querySelector("#comparatorProgressPercent"),
  comparatorProgressSummary: document.querySelector("#comparatorProgressSummary"),
  comparatorLlmCalls: document.querySelector("#comparatorLlmCalls"),
  comparatorLlmCallsDetail: document.querySelector("#comparatorLlmCallsDetail"),
  comparatorWallClock: document.querySelector("#comparatorWallClock"),
  comparatorLlmTime: document.querySelector("#comparatorLlmTime"),
  comparatorCompletedProposals: document.querySelector("#comparatorCompletedProposals"),
  comparatorShownRows: document.querySelector("#comparatorShownRows"),
  comparatorRunId: document.querySelector("#comparatorRunId"),
  comparatorStatusTone: document.querySelector("#comparatorStatusTone"),
  comparatorStatusTitle: document.querySelector("#comparatorStatusTitle"),
  comparatorStatusDetail: document.querySelector("#comparatorStatusDetail"),
  comparatorProgressBar: document.querySelector("#comparatorProgressBar"),
  comparatorProgressDetail: document.querySelector("#comparatorProgressDetail"),
  comparatorProgressDetails: document.querySelector("#comparatorProgressDetails"),
  comparatorProposalCards: document.querySelector("#comparatorProposalCards"),
  comparatorResultsBody: document.querySelector("#comparatorResultsBody"),
  comparatorLogOutput: document.querySelector("#comparatorLogOutput"),
  comparatorIntegrationDetails: document.querySelector("#comparatorIntegrationDetails"),
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

const DEFAULT_PROMPT_TEMPLATES = {
  influence: dom.promptTemplate.value,
  turbulence: `TURBULENCE_N_PROMPT_QWEN = """
Generate exactly {num_candidates} very close rewrites of one prompt component.

Component type: {component_name}
Component definition: {component_definition}

Current component value: {current_component}

Context:
{other_components}

Task:
Rewrite Current with minimal wording changes.
Keep the same meaning and same component type.
Keep most words from Current in the same order.
Replace only 1 word if possible, maximum 2 words.
Do not use Context to add new information.

Rules:
- Return only {component_name} candidates.
- Each candidate must have 2-8 words.
- Do not copy Current exactly.
- Do not change the main meaning.
- Do not explain.
- No quotes.
- Exactly {num_candidates} numbered lines.

Format:
1) rewritten {component_name}
2) rewritten {component_name}
3) rewritten {component_name}
...
"""`,
};

const COMPARATOR_API = "/api/comparator";
const COMPARATOR_TERMINAL_STATUSES = new Set(["completed", "failed", "cancelled"]);
const INITIAL_POPULATION_API = "/api/initial-population";
const INITIAL_POPULATION_TERMINAL_STATUSES = new Set(["completed", "failed", "cancelled"]);

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

function formatDuration(milliseconds) {
  if (!Number.isFinite(milliseconds)) {
    return "--";
  }
  if (milliseconds < 1000) {
    return `${formatNumber(milliseconds, 0)} ms`;
  }
  return `${formatNumber(milliseconds / 1000, 2)} s`;
}

function readClampedNumber(input, label, min, max) {
  const rawValue = input.value.trim();
  const number = Number(rawValue);
  if (!rawValue) {
    throw new Error(`${label} debe ser un número válido.`);
  }
  if (!Number.isFinite(number)) {
    throw new Error(`${label} debe ser un número válido.`);
  }
  return Math.min(max, Math.max(min, number));
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

function selectedSolutionLlmModel() {
  return dom.solutionLlmModelSelect.value || dom.solutionLlmModelManual.value.trim();
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
    ["Cálculo", PIPELINE_CONFIG.batching],
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

function resetPsoMetrics() {
  dom.psoSelectedComponents.textContent = "--";
  dom.psoFailedChanges.textContent = "--";
  dom.psoFailureRate.textContent = "--";
  dom.psoAppliedChanges.textContent = "--";
  dom.psoAffectedIndividuals.textContent = "--";
  dom.psoLlmCalls.textContent = "--";
  dom.psoAverageLlmTime.textContent = "--";
  dom.psoTotalLlmTime.textContent = "--";
  dom.psoValidCandidates.textContent = "--";
  dom.psoProgressFailures.textContent = "--";
  dom.psoExecutionErrors.textContent = "--";
}

function selectedPsoOperator() {
  return Object.hasOwn(PSO_OPERATORS, dom.psoSemanticOperator.value)
    ? dom.psoSemanticOperator.value
    : "influence";
}

function psoOperatorLabel(operator) {
  return PSO_OPERATORS[operator]?.label || operator;
}

function movementOperatorLabel(movement) {
  if (movement.operator === "turbulence") {
    return PSO_OPERATORS.turbulence.statusLabel;
  }
  return movement.source === "pbest" ? "pbest" : "líder/gbest";
}

function updatePsoOperatorUi() {
  const showTurbulenceControls = dom.simulatePsoSwitch.checked && selectedPsoOperator() === "turbulence";
  dom.turbulenceControls.forEach((control) => {
    control.hidden = !showTurbulenceControls;
  });
  dom.turbulenceMinSimilarity.disabled = !showTurbulenceControls;
  dom.turbulenceMaxSimilarity.disabled = !showTurbulenceControls;
  dom.semanticMargin.disabled = showTurbulenceControls;
  dom.targetCopyThreshold.disabled = showTurbulenceControls;
}

function isDefaultPromptTemplate(template) {
  return Object.values(DEFAULT_PROMPT_TEMPLATES).some((defaultTemplate) => defaultTemplate.trim() === template.trim());
}

function applyPsoOperatorTemplate() {
  const operator = selectedPsoOperator();
  if (isDefaultPromptTemplate(dom.promptTemplate.value)) {
    dom.promptTemplate.value = DEFAULT_PROMPT_TEMPLATES[operator];
  }
  dom.renderedPromptPreview.textContent = renderPrompt();
  updatePsoOperatorUi();
}

function updateSimulationModeUi() {
  const isPsoMode = dom.simulatePsoSwitch.checked;
  dom.psoControls.hidden = !isPsoMode;
  updatePsoOperatorUi();
  dom.templateMetrics.classList.toggle("is-hidden", isPsoMode);
  dom.psoMetrics.classList.toggle("is-hidden", !isPsoMode);
  dom.runCount.disabled = isPsoMode;
  [
    dom.componentPreset,
    dom.componentName,
    dom.componentDefinition,
    dom.currentComponent,
    dom.targetComponent,
    dom.otherComponents,
  ].forEach((field) => {
    field.disabled = isPsoMode;
  });

  if (isPsoMode && dom.candidateCount.value === "4") {
    dom.candidateCount.value = "5";
  }

  resetCheckerMetrics();
  resetPsoMetrics();
  setResultsTableMode(isPsoMode ? "pso" : "standard");
  setStatus(
    dom.checkerStatusTone,
    dom.checkerStatusTitle,
    dom.checkerStatusDetail,
    isPsoMode ? "Modo PSO activo" : "Modo plantilla individual",
    isPsoMode
      ? "Se recorrerán individuos del JSON y se evaluarán movimientos por componente sorteada."
      : "Cada ejecución envía una solicitud independiente a LM Studio, sin historial compartido.",
  );
}

function setCheckerRunning(isRunning) {
  dom.runTemplateButton.disabled = isRunning;
  dom.stopTemplateButton.disabled = !isRunning;
  dom.loadModelsButton.disabled = isRunning;
  dom.previewPromptButton.disabled = isRunning;
  dom.psoSemanticOperator.disabled = isRunning;
  dom.psoIndividualCount.disabled = isRunning;
  dom.psoChangeThreshold.disabled = isRunning;
  if (isRunning) {
    dom.turbulenceMinSimilarity.disabled = true;
    dom.turbulenceMaxSimilarity.disabled = true;
  } else {
    updatePsoOperatorUi();
  }
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

function renderPromptWithVariables(variables, template = dom.promptTemplate.value) {
  return getTemplateBody(template).replace(/\{([A-Za-z0-9_]+)\}/g, (match, key) => (
    Object.hasOwn(variables, key) ? variables[key] : match
  ));
}

function renderPrompt() {
  return renderPromptWithVariables(checkerVariables());
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
  const candidates = [];

  cleaned.split(/\r?\n/).forEach((line) => {
    const candidate = normalizeCandidate(line);
    if (candidate) {
      candidates.push(candidate);
    }
  });

  return candidates.slice(0, expectedCount);
}

function validationReason(candidate, runCandidates, config) {
  const lower = candidate.toLocaleLowerCase();
  const current = config.current.toLocaleLowerCase();
  const target = typeof config.target === "string" ? config.target.toLocaleLowerCase() : "";
  const words = wordCount(candidate);

  if (words < config.minWords || words > config.maxWords) {
    return "invalid_length";
  }
  if (lower === current) {
    return "literal_copy_current";
  }
  if (target && lower === target) {
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
  const messages = config.systemPrompt
    ? [
        { role: "system", content: config.systemPrompt },
        { role: "user", content: prompt },
      ]
    : [{ role: "user", content: prompt }];
  const requestBody = isNativeApi
    ? {
        model: config.model,
        input: config.systemPrompt ? `${config.systemPrompt}\n\nUser instruction:\n${prompt}` : prompt,
        temperature: config.temperature,
        top_p: config.topP,
        max_output_tokens: config.maxTokens,
        stream: false,
        store: false,
      }
    : {
        model: config.model,
        messages,
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
  const contentToText = (content) => {
    if (typeof content === "string") {
      return content;
    }
    if (Array.isArray(content)) {
      return content
        .map((item) => (typeof item === "string" ? item : item?.text || item?.content || ""))
        .filter(Boolean)
        .join("\n");
    }
    return "";
  };

  if (apiMode === "native") {
    if (typeof payload.output_text === "string") {
      return payload.output_text.trim();
    }

    const output = Array.isArray(payload.output) ? payload.output : [];
    return output
      .filter((item) => item?.type === "message")
      .map((item) => contentToText(item.content))
      .filter(Boolean)
      .join("\n")
      .trim();
  }

  return contentToText(payload.choices?.[0]?.message?.content).trim();
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

async function fetchSolutionLmStudioModels() {
  const endpoint = normalizeEndpoint(dom.solutionLmEndpoint.value);
  dom.solutionLmEndpoint.value = endpoint;
  dom.loadSolutionModelsButton.disabled = true;
  dom.solutionConnectionDot.classList.add("is-busy");
  dom.solutionConnectionDot.classList.remove("is-error");
  dom.solutionConnectionText.textContent = "Consultando LM Studio";

  try {
    const response = await fetch(`${endpoint}/models`);
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.error?.message || `HTTP ${response.status}`);
    }

    const models = parseLmStudioModels(payload, dom.solutionLmApiMode.value);
    dom.solutionLlmModelSelect.replaceChildren(
      ...models.map((model) => {
        const option = document.createElement("option");
        option.value = model.id;
        option.textContent = model.label;
        return option;
      }),
    );

    if (models.length > 0) {
      const loaded = models.find((model) => model.loaded) || models[0];
      dom.solutionLlmModelSelect.value = loaded.id;
      dom.solutionLlmModelManual.value = loaded.id;
    }

    dom.solutionConnectionText.textContent = `${models.length} modelo(s) disponibles`;
  } catch (error) {
    dom.solutionConnectionDot.classList.add("is-error");
    dom.solutionConnectionText.textContent = "Error de conexión";
    setStatus(dom.solutionStatusTone, dom.solutionStatusTitle, dom.solutionStatusDetail, "No se pudo conectar a LM Studio", error.message, "error");
  } finally {
    dom.solutionConnectionDot.classList.remove("is-busy");
    dom.loadSolutionModelsButton.disabled = false;
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
  const simulatePso = dom.simulatePsoSwitch.checked;
  const psoOperator = dom.psoSemanticOperator.value;

  if (!endpoint) throw new Error("Define el endpoint de LM Studio.");
  if (!model) throw new Error("Selecciona o escribe el modelo LLM.");
  if (!Object.hasOwn(PSO_OPERATORS, psoOperator)) {
    throw new Error("Selecciona un operador semántico PSO válido.");
  }
  if (!simulatePso && (!variables.current_component || !variables.target_component)) {
    throw new Error("Current y Target son obligatorios.");
  }
  if (simulatePso && !variables.reference_text) {
    throw new Error("El texto de referencia fijo es obligatorio en modo PSO.");
  }

  const config = {
    endpoint,
    apiMode: dom.lmApiMode.value,
    model,
    simulatePso,
    embeddingModel: dom.checkerEmbeddingModel.value,
    runs: readClampedNumber(dom.runCount, "Ejecuciones independientes", 1, 200),
    expectedCandidates: readClampedNumber(dom.candidateCount, "Candidatos por ejecución", 1, 20),
    temperature: readClampedNumber(dom.temperature, "Temperatura", 0, 2),
    topP: readClampedNumber(dom.topP, "Top p", 0, 1),
    maxTokens: readClampedNumber(dom.maxTokens, "Máx. tokens", 8, 2048),
    timeoutMs: readClampedNumber(dom.requestTimeout, "Timeout por ejecución", 5, 600) * 1000,
    minWords: readClampedNumber(dom.minWords, "Mín. palabras", 1, 20),
    maxWords: readClampedNumber(dom.maxWords, "Máx. palabras", 1, 40),
    margin: readClampedNumber(dom.semanticMargin, "Margen eta", 0, 1),
    targetCopyThreshold: readClampedNumber(dom.targetCopyThreshold, "Tau copia target", 0, 1),
    concurrency: readClampedNumber(dom.concurrency, "Paralelismo", 1, 8),
    psoOperator,
    psoIndividualCount: readClampedNumber(dom.psoIndividualCount, "N individuos", 1, 200),
    psoChangeThreshold: readClampedNumber(dom.psoChangeThreshold, "Umbral sorteo cambio", 0, 1),
    turbulenceMinSimilarity: readClampedNumber(dom.turbulenceMinSimilarity, "Sim. turbulencia mínima", -1, 1),
    turbulenceMaxSimilarity: readClampedNumber(dom.turbulenceMaxSimilarity, "Sim. turbulencia máxima", -1, 1),
    current: variables.current_component,
    target: variables.target_component,
    referenceText: variables.reference_text,
    promptTemplate: dom.promptTemplate.value,
    prompt: renderPrompt(),
  };

  if (config.minWords > config.maxWords) {
    throw new Error("Mín. palabras no puede ser mayor que Máx. palabras.");
  }
  if (config.turbulenceMinSimilarity > config.turbulenceMaxSimilarity) {
    throw new Error("La similitud mínima de turbulencia no puede ser mayor que la máxima.");
  }

  return config;
}

function clearCheckerTables() {
  latestCheckerRows = [];
  dom.runResultsBody.innerHTML = '<tr><td colspan="6">Sin ejecuciones todavía.</td></tr>';
  dom.candidateResultsBody.innerHTML = '<tr><td colspan="6">Sin candidatos todavía.</td></tr>';
}

function setResultsTableMode(mode) {
  if (mode === "pso") {
    dom.runResultsTitle.textContent = "Movimientos PSO";
    dom.candidateResultsTable.classList.add("wide-table");
    dom.runResultsHead.innerHTML = `
      <tr>
        <th>#</th>
        <th>Individuo</th>
        <th>Componente</th>
        <th>Operador</th>
        <th>Movimiento aplicado</th>
        <th>Estado</th>
      </tr>
    `;
    dom.candidateResultsTitle.textContent = "Candidatos por movimiento PSO";
    dom.candidateResultsHead.innerHTML = `
      <tr>
        <th>Movimiento</th>
        <th>Individuo</th>
        <th>Componente</th>
        <th>Operador</th>
        <th>Current</th>
        <th>Referencia</th>
        <th>Candidato</th>
        <th>Similitud</th>
        <th>Delta / diversidad</th>
        <th>Validación</th>
        <th>Aplicado</th>
      </tr>
    `;
    dom.runResultsBody.innerHTML = '<tr><td colspan="6">Sin movimientos PSO todavía.</td></tr>';
    dom.candidateResultsBody.innerHTML = '<tr><td colspan="11">Sin candidatos PSO todavía.</td></tr>';
    return;
  }

  dom.runResultsTitle.textContent = "Resultados por ejecución";
  dom.candidateResultsTable.classList.remove("wide-table");
  dom.runResultsHead.innerHTML = `
    <tr>
      <th>#</th>
      <th>Mejor candidato válido</th>
      <th>Similitud</th>
      <th>Mejora</th>
      <th>Válidos</th>
      <th>Estado</th>
    </tr>
  `;
  dom.candidateResultsTitle.textContent = "Candidatos generados";
  dom.candidateResultsHead.innerHTML = `
    <tr>
      <th>Run</th>
      <th>Candidato</th>
      <th>Similitud con target</th>
      <th>Distancia</th>
      <th>Mejora</th>
      <th>Validación</th>
    </tr>
  `;
  dom.runResultsBody.innerHTML = '<tr><td colspan="6">Sin ejecuciones todavía.</td></tr>';
  dom.candidateResultsBody.innerHTML = '<tr><td colspan="6">Sin candidatos todavía.</td></tr>';
}

function appendCandidateRows(rows) {
  if (rows.length === 0) {
    return;
  }

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
      const status = run.error
        ? `<span class="invalid">${escapeHtml(run.error)}</span>`
        : run.status ? `<span class="invalid">${escapeHtml(run.status)}</span>` : '<span class="valid">ok</span>';
      tr.innerHTML = `
        <td>${run.runNumber}</td>
        <td>${best ? escapeHtml(best.candidate) : "--"}</td>
        <td>${best ? formatNumber(best.similarity) : "--"}</td>
        <td>${best ? formatSigned(best.improvement) : "--"}</td>
        <td>${run.validImprovementCount}/${run.candidateCount}</td>
        <td>${status}</td>
      `;
      return tr;
    }),
  );
}

function formatSigned(value) {
  const sign = value > 0 ? "+" : "";
  return `${sign}${formatNumber(value)}`;
}

function randomItem(items) {
  return items[Math.floor(Math.random() * items.length)] || null;
}

function escapeHtml(value) {
  return String(value ?? "")
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
        status: "sin candidatos parseables",
        error: null,
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
    const hasSemanticProgress = improvement > config.margin;
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
      status: null,
      error: null,
    },
    rows,
  };
}

function solutionReferencePreset() {
  return SOLUTION_REFERENCE_PRESETS.find((preset) => preset.id === dom.solutionReferencePreset.value)
    || SOLUTION_REFERENCE_PRESETS[0];
}

function populateSolutionReferencePresets() {
  dom.solutionReferencePreset.replaceChildren(
    ...SOLUTION_REFERENCE_PRESETS.map((preset) => new Option(preset.label, preset.id)),
  );
}

function applySolutionReferencePreset() {
  const preset = solutionReferencePreset();
  dom.solutionReferenceText.value = preset.referenceText;
  dom.solutionReferenceSource.textContent = preset.source;
  previewSolutions();
}

function resetSolutionMetrics() {
  dom.solutionLlmCalls.textContent = "--";
  dom.solutionLlmTotalMs.textContent = "--";
  dom.solutionFlowTotalMs.textContent = "--";
  dom.solutionNonDominatedCount.textContent = "--";
  dom.solutionEvolmdCost.textContent = "--";
  dom.solutionEvolmdCostDetail.textContent = "Estimación pendiente.";
  dom.solutionMesapCost.textContent = "--";
  dom.solutionMesapCostDetail.textContent = "Estimación pendiente.";
  dom.solutionObjectiveDetails.replaceChildren();
}

function clearSolutionResults() {
  latestSolutionRows = [];
  dom.solutionResultsBody.innerHTML = '<tr><td colspan="9">Sin soluciones evaluadas todavía.</td></tr>';
  resetSolutionMetrics();
}

function setSolutionRunning(isRunning) {
  dom.runSolutionsButton.disabled = isRunning;
  dom.stopSolutionsButton.disabled = !isRunning;
  dom.loadSolutionModelsButton.disabled = isRunning;
  dom.previewSolutionsButton.disabled = isRunning;
  dom.clearSolutionResultsButton.disabled = isRunning;
}

function solutionComponentList(solution) {
  return `role = ${solution.role}; topic = ${solution.topic}; action = ${solution.action}`;
}

function solutionVariables(solution, referenceText) {
  return {
    role: solution.role,
    topic: solution.topic,
    action: solution.action,
    component_list: solutionComponentList(solution),
    reference_text: referenceText,
  };
}

function renderSolutionPrompt(solution, referenceText, template) {
  return renderPromptWithVariables(solutionVariables(solution, referenceText), template);
}

function createMockSolutions(preset, count) {
  const solutions = [];
  for (let index = 0; index < count; index += 1) {
    const role = preset.roles[index % preset.roles.length];
    const topic = preset.topics[Math.floor(index / preset.roles.length) % preset.topics.length];
    const action = preset.actions[Math.floor(index / (preset.roles.length * preset.topics.length)) % preset.actions.length];
    solutions.push({
      id: `sol-${String(index + 1).padStart(3, "0")}`,
      number: index + 1,
      role,
      topic,
      action,
    });
  }
  return solutions;
}

function readSolutionConfig() {
  const endpoint = normalizeEndpoint(dom.solutionLmEndpoint.value);
  const model = selectedSolutionLlmModel();
  const referenceText = dom.solutionReferenceText.value.trim();
  const fidelityMin = readClampedNumber(dom.solutionFidelityMin, "Tau mín. fidelidad", -1, 1);
  const copyMax = readClampedNumber(dom.solutionCopyMax, "Tau máx. copia", -1, 1);

  if (!endpoint) throw new Error("Define el endpoint de LM Studio.");
  if (!model) throw new Error("Selecciona o escribe el modelo LLM.");
  if (!referenceText) throw new Error("Define el texto de referencia.");
  if (fidelityMin > copyMax) {
    throw new Error("Tau mín. fidelidad no puede ser mayor que Tau máx. copia.");
  }

  const solutionCount = Math.floor(readClampedNumber(dom.solutionCount, "Soluciones mock", 1, 200));
  const concurrency = Math.floor(readClampedNumber(dom.solutionConcurrency, "Paralelismo", 1, 8));

  return {
    endpoint,
    apiMode: dom.solutionLmApiMode.value,
    model,
    solutionCount,
    temperature: readClampedNumber(dom.solutionTemperature, "Temperatura", 0, 2),
    topP: readClampedNumber(dom.solutionTopP, "Top p", 0, 1),
    maxTokens: readClampedNumber(dom.solutionMaxTokens, "Máx. tokens", 8, 2048),
    timeoutMs: readClampedNumber(dom.solutionRequestTimeout, "Timeout por solución", 5, 600) * 1000,
    concurrency,
    embeddingModel: dom.solutionEmbeddingModel.value,
    diversityBasis: dom.solutionDiversityBasis.value,
    fidelityMin,
    copyMax,
    referenceText,
    referencePreset: solutionReferencePreset(),
    promptTemplate: dom.solutionPromptTemplate.value,
    systemPrompt: SOLUTION_GENERATION_SYSTEM_PROMPT,
  };
}

function previewSolutions() {
  try {
    const count = Math.min(200, Math.max(1, Number(dom.solutionCount.value) || 1));
    const firstSolution = createMockSolutions(solutionReferencePreset(), count)[0];
    dom.solutionPromptPreview.textContent = firstSolution
      ? renderSolutionPrompt(firstSolution, dom.solutionReferenceText.value.trim(), dom.solutionPromptTemplate.value)
      : "No hay soluciones para previsualizar.";
  } catch (error) {
    dom.solutionPromptPreview.textContent = error.message;
  }
}

function normalizeGeneratedText(rawOutput) {
  return rawOutput
    .replace(/```[A-Za-z]*\n?/g, "")
    .replace(/```/g, "")
    .replace(/^["'`]+|["'`]+$/g, "")
    .trim();
}

function objectiveVectorLabel(row) {
  if (!Number.isFinite(row.fidelity) || !Number.isFinite(row.diversity)) {
    return "--";
  }
  return `[${formatNumber(row.fidelity, 4)}, ${formatNumber(row.diversity, 4)}]`;
}

function solutionValidationLabel(row, config) {
  if (row.error) {
    return row.error;
  }
  if (!Number.isFinite(row.fidelity)) {
    return "pendiente F.O";
  }
  if (row.fidelity < config.fidelityMin) {
    return "baja_fidelidad";
  }
  if (row.fidelity > config.copyMax) {
    return "posible_copia";
  }
  return "ok";
}

function renderSolutionRows(rows, config = null) {
  if (rows.length === 0) {
    dom.solutionResultsBody.innerHTML = '<tr><td colspan="9">Sin soluciones evaluadas todavía.</td></tr>';
    return;
  }

  dom.solutionResultsBody.replaceChildren(
    ...rows.map((row) => {
      const tr = document.createElement("tr");
      const status = config ? solutionValidationLabel(row, config) : row.error || "pendiente";
      const isOk = status === "ok";
      tr.innerHTML = `
        <td>${row.number}</td>
        <td class="context-cell">${escapeHtml(row.role)}</td>
        <td class="context-cell">${escapeHtml(row.topic)}</td>
        <td class="context-cell">${escapeHtml(row.action)}</td>
        <td class="context-cell long-cell">${escapeHtml(row.prompt || "--")}</td>
        <td class="context-cell long-cell">${escapeHtml(row.generatedText || "--")}</td>
        <td>${objectiveVectorLabel(row)}</td>
        <td>${row.nonDominated ? '<span class="valid">sí</span>' : "--"}</td>
        <td><span class="${isOk ? "valid" : "invalid"}">${escapeHtml(status)}</span></td>
      `;
      return tr;
    }),
  );
}

function markNonDominated(rows) {
  rows.forEach((row) => {
    if (!Number.isFinite(row.fidelity) || !Number.isFinite(row.diversity)) {
      row.nonDominated = false;
      return;
    }

    row.nonDominated = !rows.some((other) => (
      other !== row
      && Number.isFinite(other.fidelity)
      && Number.isFinite(other.diversity)
      && other.fidelity >= row.fidelity
      && other.diversity >= row.diversity
      && (other.fidelity > row.fidelity || other.diversity > row.diversity)
    ));
  });
}

async function scoreSolutionObjectiveRows(rows, config) {
  const completedRows = rows.filter((row) => !row.error && row.generatedText);
  if (completedRows.length === 0) {
    markNonDominated(rows);
    return rows;
  }

  const generatedEmbeddings = await requestEmbeddings(
    config.embeddingModel,
    [config.referenceText, ...completedRows.map((row) => row.generatedText)],
  );
  const referenceEmbedding = generatedEmbeddings[0];
  const generatedVectors = generatedEmbeddings.slice(1);
  const diversityVectors = config.diversityBasis === "prompt"
    ? await requestEmbeddings(config.embeddingModel, completedRows.map((row) => row.prompt))
    : generatedVectors;

  completedRows.forEach((row, index) => {
    row.fidelity = cosineSimilarity(generatedVectors[index], referenceEmbedding);
    if (diversityVectors.length <= 1) {
      row.diversity = 0;
      return;
    }

    const totalDistance = diversityVectors.reduce((sum, vector, otherIndex) => {
      if (otherIndex === index) {
        return sum;
      }
      return sum + cosineDistance(cosineSimilarity(diversityVectors[index], vector));
    }, 0);
    row.diversity = totalDistance / (diversityVectors.length - 1);
  });

  markNonDominated(rows);
  return rows;
}

function estimateComparativeCosts(solutionCount, averageLlmMs) {
  const expectedMutations = Math.ceil(solutionCount * SOLUTION_COST_ASSUMPTIONS.evolmdMutationProbability);
  const evolmdCalls = (solutionCount * 2) + (expectedMutations * 2);
  const mesapCalls = solutionCount * SOLUTION_COST_ASSUMPTIONS.mesapCallsPerSolution;

  return {
    evolmdCalls,
    evolmdMs: Number.isFinite(averageLlmMs) ? evolmdCalls * averageLlmMs : NaN,
    mesapCalls,
    mesapMs: Number.isFinite(averageLlmMs)
      ? mesapCalls * averageLlmMs * SOLUTION_COST_ASSUMPTIONS.mesapRuntimeMultiplier
      : NaN,
  };
}

function summarizeSolutionResults(rows, config, llmCalls, flowElapsedMs) {
  const timedRows = rows.filter((row) => Number.isFinite(row.elapsedMs));
  const llmElapsedMs = timedRows.reduce((sum, row) => sum + row.elapsedMs, 0);
  const averageLlmMs = llmCalls > 0 ? llmElapsedMs / llmCalls : NaN;
  const completedRows = rows.filter((row) => !row.error && row.generatedText);
  const scoredRows = completedRows.filter((row) => Number.isFinite(row.fidelity));
  const nonDominatedRows = scoredRows.filter((row) => row.nonDominated);
  const bestFidelity = scoredRows.reduce((best, row) => (!best || row.fidelity > best.fidelity ? row : best), null);
  const bestDiversity = scoredRows.reduce((best, row) => (!best || row.diversity > best.diversity ? row : best), null);
  const estimates = estimateComparativeCosts(config.solutionCount, averageLlmMs);

  dom.solutionLlmCalls.textContent = `${llmCalls}/${config.solutionCount}`;
  dom.solutionLlmTotalMs.textContent = formatDuration(llmElapsedMs);
  dom.solutionFlowTotalMs.textContent = formatDuration(flowElapsedMs);
  dom.solutionNonDominatedCount.textContent = scoredRows.length ? `${nonDominatedRows.length}/${scoredRows.length}` : "--";
  dom.solutionEvolmdCost.textContent = `${estimates.evolmdCalls} llamadas`;
  dom.solutionEvolmdCostDetail.textContent = Number.isFinite(estimates.evolmdMs)
    ? `${formatDuration(estimates.evolmdMs)} estimados con Init/Data Agent y mutación Pm=0,05.`
    : "Sin promedio local para estimar ms.";
  dom.solutionMesapCost.textContent = `${estimates.mesapCalls} llamadas`;
  dom.solutionMesapCostDetail.textContent = Number.isFinite(estimates.mesapMs)
    ? `${formatDuration(estimates.mesapMs)} estimados con 3 llamadas por solución y sobrecosto 18%.`
    : "Sin promedio local para estimar ms.";

  renderDefinitionList(dom.solutionObjectiveDetails, [
    ["Modelo embedding", EMBEDDING_MODELS[config.embeddingModel].displayName],
    ["F1 fidelidad", "cos(texto generado, referencia)"],
    ["F2 diversidad", config.diversityBasis === "prompt" ? "distancia promedio entre prompts" : "distancia promedio entre textos generados"],
    ["Mayor fidelidad", bestFidelity ? `${bestFidelity.id}: ${formatNumber(bestFidelity.fidelity)}` : "--"],
    ["Mayor diversidad", bestDiversity ? `${bestDiversity.id}: ${formatNumber(bestDiversity.diversity)}` : "--"],
  ]);
}

async function runSolutionEvaluation() {
  let config;
  try {
    config = readSolutionConfig();
  } catch (error) {
    setStatus(dom.solutionStatusTone, dom.solutionStatusTitle, dom.solutionStatusDetail, "Configuración incompleta", error.message, "error");
    return;
  }

  solutionEvalStopRequested = false;
  clearSolutionResults();
  setSolutionRunning(true);

  const flowStarted = performance.now();
  const solutions = createMockSolutions(config.referencePreset, config.solutionCount);
  const rows = [];
  let nextSolutionIndex = 0;
  let llmCalls = 0;
  dom.solutionPromptPreview.textContent = renderSolutionPrompt(solutions[0], config.referenceText, config.promptTemplate);

  try {
    async function solutionWorker() {
      while (!solutionEvalStopRequested && nextSolutionIndex < solutions.length) {
        const solution = solutions[nextSolutionIndex];
        nextSolutionIndex += 1;
        const prompt = renderSolutionPrompt(solution, config.referenceText, config.promptTemplate);
        setStatus(
          dom.solutionStatusTone,
          dom.solutionStatusTitle,
          dom.solutionStatusDetail,
          "Generando textos",
          `Solución ${solution.number} de ${config.solutionCount}.`,
          "busy",
        );

        try {
          llmCalls += 1;
          const response = await callLmStudio(prompt, solution.number, config);
          const generatedText = normalizeGeneratedText(response.raw);
          if (!generatedText) {
            throw new Error("respuesta vacía");
          }
          rows.push({
            ...solution,
            prompt,
            generatedText,
            elapsedMs: response.elapsedMs,
            error: null,
          });
        } catch (error) {
          rows.push({
            ...solution,
            prompt,
            generatedText: "",
            elapsedMs: null,
            error: error.name === "AbortError" ? "timeout" : error.message,
          });
        }

        rows.sort((a, b) => a.number - b.number);
        latestSolutionRows = rows;
        renderSolutionRows(rows, config);
        summarizeSolutionResults(rows, config, llmCalls, performance.now() - flowStarted);
      }
    }

    await Promise.all(Array.from({ length: Math.min(config.concurrency, solutions.length) }, () => solutionWorker()));

    setStatus(
      dom.solutionStatusTone,
      dom.solutionStatusTitle,
      dom.solutionStatusDetail,
      "Calculando F.O",
      "Generando embeddings y frente de no dominancia.",
      "busy",
    );
    await scoreSolutionObjectiveRows(rows, config);
    rows.sort((a, b) => a.number - b.number);
    latestSolutionRows = rows;
    renderSolutionRows(rows, config);
    summarizeSolutionResults(rows, config, llmCalls, performance.now() - flowStarted);

    setStatus(
      dom.solutionStatusTone,
      dom.solutionStatusTitle,
      dom.solutionStatusDetail,
      solutionEvalStopRequested ? "Prueba detenida" : "Prueba completada",
      `${rows.length} solución(es) procesada(s).`,
    );
  } catch (error) {
    setStatus(dom.solutionStatusTone, dom.solutionStatusTitle, dom.solutionStatusDetail, "Error en prueba", error.message, "error");
  } finally {
    setSolutionRunning(false);
  }
}

function initialPopulationStatusLabel(status) {
  const labels = {
    queued: "en cola",
    running: "ejecutando",
    completed: "completado",
    failed: "fallido",
    cancelled: "cancelado",
  };
  return labels[status] || status || "--";
}

function formatOptionalNumber(value, digits = 6) {
  const number = Number(value);
  return Number.isFinite(number) ? formatNumber(number, digits) : "--";
}

function readInitialStageConfigs() {
  const stages = {};
  dom.initialStageConfigs.forEach((panel) => {
    const stageName = panel.dataset.initialStage;
    const stage = {};
    panel.querySelectorAll("[data-stage-field]").forEach((field) => {
      const key = field.dataset.stageField;
      if (["temperature", "topP"].includes(key)) {
        stage[key] = readClampedNumber(field, `${stageName}.${key}`, 0, key === "topP" ? 1 : 2);
      } else if (["topK", "maxTokens"].includes(key)) {
        stage[key] = Math.floor(readClampedNumber(field, `${stageName}.${key}`, key === "topK" ? 0 : 8, key === "topK" ? 500 : 4096));
      } else {
        const value = field.value.trim();
        if (!value) {
          throw new Error(`${stageName}.${key} no puede estar vacio.`);
        }
        stage[key] = value;
      }
    });
    stages[stageName] = stage;
  });
  return stages;
}

function readInitialPopulationConfig() {
  const referenceText = dom.initialReferenceText.value.trim();
  const domain = dom.initialDomain.value.trim();
  const baseUrl = dom.initialLmStudioBase.value.trim();
  const promptTemplate = dom.initialPromptTemplate.value.trim();

  if (!referenceText) throw new Error("Define el texto de referencia.");
  if (!domain) throw new Error("Define el dominio general.");
  if (!baseUrl) throw new Error("Define la Base URL de LM Studio.");
  if (!promptTemplate) throw new Error("Define la plantilla deterministica final.");

  const generatedMinWords = Math.floor(readClampedNumber(dom.initialGeneratedMinWords, "Min. palabras texto", 1, 500));
  const generatedMaxWords = Math.floor(readClampedNumber(dom.initialGeneratedMaxWords, "Max. palabras texto", 1, 500));
  if (generatedMinWords > generatedMaxWords) {
    throw new Error("Min. palabras texto no puede ser mayor que Max. palabras texto.");
  }

  return {
    strategyId: "hybrid-semantic-v6",
    referenceText,
    domain,
    n: Math.floor(readClampedNumber(dom.initialN, "N individuos", 1, 500)),
    topK: Math.floor(readClampedNumber(dom.initialTopK, "Top K tabla", 1, 500)),
    timeoutSeconds: Math.floor(readClampedNumber(dom.initialTimeoutSeconds, "Timeout corrida", 5, 3600)),
    generationParallelism: Math.floor(readClampedNumber(dom.initialGenerationParallelism, "Paralelismo generacion", 1, 16)),
    seed: Math.floor(readClampedNumber(dom.initialSeed, "Semilla", 0, 2147483647)),
    embeddingModel: dom.initialEmbeddingModel.value,
    lmStudio: {
      baseUrl,
      apiMode: dom.initialLmApiMode.value,
    },
    stages: readInitialStageConfigs(),
    promptTemplate,
    validation: {
      rolesMaxWords: Math.floor(readClampedNumber(dom.initialRolesMaxWords, "Max. palabras roles", 1, 20)),
      topicsMaxWords: Math.floor(readClampedNumber(dom.initialTopicsMaxWords, "Max. palabras topicos", 1, 30)),
      actionsMaxWords: Math.floor(readClampedNumber(dom.initialActionsMaxWords, "Max. palabras acciones", 1, 20)),
      generatedMinWords,
      generatedMaxWords,
    },
  };
}

async function requestInitialPopulationJson(path, options = {}) {
  const response = await fetch(`${INITIAL_POPULATION_API}${path}`, {
    headers: {
      "Content-Type": "application/json",
      ...(options.headers || {}),
    },
    ...options,
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.error || `HTTP ${response.status}`);
  }
  return payload;
}

async function loadInitialPopulationStrategies() {
  try {
    const payload = await requestInitialPopulationJson("/strategies");
    const strategies = payload.strategies || [];
    dom.initialStrategySummary.textContent = strategies
      .map((strategy) => `${strategy.displayName}: ${strategy.available ? "disponible" : "faltante"}`)
      .join("; ") || "Sin estrategias registradas.";
    renderDefinitionList(dom.initialIntegrationDetails, [
      ["Contrato", "{strategyId, displayName, rows, metrics, cost, outputDir, status}"],
      ["Estrategias", strategies.map((strategy) => strategy.strategyId).join(", ") || "--"],
      ["Salida", "runs/initial-population/<runId>/summary.json + artefactos por etapa"],
      ["Objetivos", "semantic_fidelity y semantic_diversity"],
      ["Costo", "llamadas LM Studio, tokens reportados, wall-clock y batches SBERT"],
    ]);
  } catch (error) {
    dom.initialStrategySummary.textContent = "No se pudo consultar el registro.";
    renderDefinitionList(dom.initialIntegrationDetails, [
      ["API", error.message],
    ]);
  }
}

async function loadInitialPopulationModels() {
  const params = new URLSearchParams({
    baseUrl: dom.initialLmStudioBase.value.trim(),
    apiMode: dom.initialLmApiMode.value,
  });
  dom.loadInitialModelsButton.disabled = true;
  try {
    const payload = await requestInitialPopulationJson(`/lm-studio/models?${params.toString()}`);
    dom.initialPopulationLmModels.replaceChildren(
      ...(payload.models || []).map((model) => {
        const option = document.createElement("option");
        option.value = model;
        return option;
      }),
    );
    dom.initialPopulationConnectionText.textContent = `${(payload.models || []).length} modelo(s)`;
    dom.initialPopulationConnectionDot.classList.remove("is-error", "is-busy");
    setStatus(dom.initialStatusTone, dom.initialStatusTitle, dom.initialStatusDetail, "Modelos cargados", `${(payload.models || []).length} modelo(s) disponibles desde LM Studio.`);
  } catch (error) {
    dom.initialPopulationConnectionText.textContent = "Error modelos";
    dom.initialPopulationConnectionDot.classList.add("is-error");
    setStatus(dom.initialStatusTone, dom.initialStatusTitle, dom.initialStatusDetail, "No se pudieron cargar modelos", error.message, "error");
  } finally {
    dom.loadInitialModelsButton.disabled = false;
  }
}

function setInitialPopulationRunning(isRunning, cancelRequested = false) {
  dom.runInitialPopulationButton.disabled = isRunning;
  dom.cancelInitialPopulationButton.disabled = !isRunning || !currentInitialPopulationRunId || cancelRequested;
  dom.cancelInitialPopulationButton.textContent = cancelRequested ? "Cancelando..." : "Cancelar";
  dom.clearInitialPopulationButton.disabled = isRunning;
  [
    dom.initialLmApiMode,
    dom.initialLmStudioBase,
    dom.initialN,
    dom.initialTopK,
    dom.initialGenerationParallelism,
    dom.initialTimeoutSeconds,
    dom.initialSeed,
    dom.initialEmbeddingModel,
    dom.initialReferenceText,
    dom.initialDomain,
    dom.initialRolesMaxWords,
    dom.initialTopicsMaxWords,
    dom.initialActionsMaxWords,
    dom.initialGeneratedMinWords,
    dom.initialGeneratedMaxWords,
    dom.initialPromptTemplate,
    ...Array.from(dom.initialStageConfigs).flatMap((panel) => Array.from(panel.querySelectorAll("[data-stage-field]"))),
  ].forEach((field) => {
    field.disabled = isRunning;
  });
}

function stopInitialPopulationPolling() {
  if (initialPopulationPollTimer) {
    window.clearInterval(initialPopulationPollTimer);
    initialPopulationPollTimer = null;
  }
}

function resetInitialPopulationUi() {
  stopInitialPopulationPolling();
  currentInitialPopulationRunId = null;
  dom.initialRunStatus.textContent = "--";
  dom.initialProgressPercent.textContent = "--";
  dom.initialProgressSummary.textContent = "Sin corrida activa.";
  dom.initialLlmCalls.textContent = "--";
  dom.initialLlmCallsDetail.textContent = "Total de llamadas a LM Studio.";
  dom.initialWallClock.textContent = "--";
  dom.initialSequentialWallClock.textContent = "--";
  dom.initialLlmTime.textContent = "--";
  dom.initialEmbeddingTime.textContent = "--";
  dom.initialFinalIndividuals.textContent = "--";
  dom.initialNonDominated.textContent = "--";
  dom.initialHypervolume.textContent = "--";
  dom.initialSpread.textContent = "--";
  dom.initialCallsPerIndividual.textContent = "--";
  dom.initialRunId.textContent = "--";
  dom.initialPopulationConnectionText.textContent = "Sin ejecucion";
  dom.initialPopulationConnectionDot.classList.remove("is-busy", "is-error");
  dom.initialResultsBody.innerHTML = '<tr><td colspan="11">Sin resultados todavia.</td></tr>';
  dom.initialLogOutput.textContent = "Sin logs todavia.";
  renderInitialProgress(null);
  renderInitialSemanticArtifacts(null);
  renderDefinitionList(dom.initialArtifactDetails, [
    ["Estado", "Sin corrida"],
    ["Salida", "--"],
    ["Error", "--"],
  ]);
  setInitialPopulationRunning(false);
  setStatus(
    dom.initialStatusTone,
    dom.initialStatusTitle,
    dom.initialStatusDetail,
    "Listo",
    "Ejecuta la estrategia hibrida v6 desde Python usando LM Studio local.",
  );
}

async function runInitialPopulation() {
  let config;
  try {
    config = readInitialPopulationConfig();
  } catch (error) {
    setStatus(dom.initialStatusTone, dom.initialStatusTitle, dom.initialStatusDetail, "Configuracion incompleta", error.message, "error");
    return;
  }

  stopInitialPopulationPolling();
  setInitialPopulationRunning(true);
  dom.initialResultsBody.innerHTML = '<tr><td colspan="11">Esperando resultados.</td></tr>';
  dom.initialLogOutput.textContent = "Iniciando corrida...";
  dom.initialPopulationConnectionDot.classList.add("is-busy");
  dom.initialPopulationConnectionDot.classList.remove("is-error");
  dom.initialPopulationConnectionText.textContent = "Ejecutando";
  setStatus(
    dom.initialStatusTone,
    dom.initialStatusTitle,
    dom.initialStatusDetail,
    "Iniciando estrategia",
    "El backend ejecutara el runner Python y la web consultara el progreso.",
    "busy",
  );

  try {
    const run = await requestInitialPopulationJson("/runs", {
      method: "POST",
      body: JSON.stringify(config),
    });
    currentInitialPopulationRunId = run.runId;
    setInitialPopulationRunning(true, Boolean(run.cancelRequested));
    renderInitialPopulationRun(run);
    initialPopulationPollTimer = window.setInterval(() => refreshInitialPopulationRun(currentInitialPopulationRunId), 2000);
    await refreshInitialPopulationRun(currentInitialPopulationRunId);
  } catch (error) {
    currentInitialPopulationRunId = null;
    dom.initialPopulationConnectionDot.classList.add("is-error");
    dom.initialPopulationConnectionText.textContent = "Error";
    setInitialPopulationRunning(false);
    setStatus(dom.initialStatusTone, dom.initialStatusTitle, dom.initialStatusDetail, "Error al iniciar", error.message, "error");
  }
}

async function refreshInitialPopulationRun(runId) {
  if (!runId) {
    return;
  }

  try {
    const run = await requestInitialPopulationJson(`/runs/${encodeURIComponent(runId)}`);
    renderInitialPopulationRun(run);
    if (INITIAL_POPULATION_TERMINAL_STATUSES.has(run.status)) {
      stopInitialPopulationPolling();
      currentInitialPopulationRunId = run.runId;
      setInitialPopulationRunning(false);
    } else {
      currentInitialPopulationRunId = run.runId;
      setInitialPopulationRunning(true, Boolean(run.cancelRequested));
    }
  } catch (error) {
    stopInitialPopulationPolling();
    setInitialPopulationRunning(false);
    dom.initialPopulationConnectionDot.classList.add("is-error");
    setStatus(dom.initialStatusTone, dom.initialStatusTitle, dom.initialStatusDetail, "Error al consultar corrida", error.message, "error");
  }
}

async function cancelInitialPopulationRun() {
  if (!currentInitialPopulationRunId) {
    return;
  }
  dom.cancelInitialPopulationButton.disabled = true;
  dom.cancelInitialPopulationButton.textContent = "Cancelando...";
  setStatus(dom.initialStatusTone, dom.initialStatusTitle, dom.initialStatusDetail, "Cancelando", "Se solicitara terminar el subproceso activo.", "busy");
  try {
    const run = await requestInitialPopulationJson(`/runs/${encodeURIComponent(currentInitialPopulationRunId)}/cancel`, {
      method: "POST",
      body: "{}",
    });
    renderInitialPopulationRun(run);
    setInitialPopulationRunning(!INITIAL_POPULATION_TERMINAL_STATUSES.has(run.status), Boolean(run.cancelRequested));
  } catch (error) {
    setInitialPopulationRunning(true, false);
    setStatus(dom.initialStatusTone, dom.initialStatusTitle, dom.initialStatusDetail, "No se pudo cancelar", error.message, "error");
  }
}

function renderInitialProgress(progress) {
  if (!progress) {
    dom.initialProgressPercent.textContent = "--";
    dom.initialProgressSummary.textContent = "Sin corrida activa.";
    dom.initialProgressDetail.textContent = "Sin ejecucion.";
    dom.initialProgressBar.style.width = "0%";
    renderDefinitionList(dom.initialProgressDetails, [
      ["Etapa", "--"],
      ["Tiempo transcurrido", "--"],
      ["Tiempo restante", "--"],
      ["Avance etapa", "--"],
    ]);
    return;
  }

  const percent = Math.max(0, Math.min(100, Number(progress.percent || 0)));
  dom.initialProgressPercent.textContent = `${percent}%`;
  dom.initialProgressSummary.textContent = progress.detail || "Ejecutando.";
  dom.initialProgressDetail.textContent = progress.detail || "Ejecutando.";
  dom.initialProgressBar.style.width = `${percent}%`;
  const stageCounter = progress.stageIndex && progress.stageTotal
    ? `${progress.stageIndex}/${progress.stageTotal}`
    : "--";
  const itemCounter = progress.total
    ? `${progress.completed ?? 0}/${progress.total}`
    : "--";
  renderDefinitionList(dom.initialProgressDetails, [
    ["Etapa", `${progress.stage || "--"} ${stageCounter}`],
    ["Tiempo transcurrido", progress.elapsedLabel || "--"],
    ["Tiempo restante estimado", progress.remainingLabel || "No disponible"],
    ["Avance etapa", itemCounter],
  ]);
}

function renderInitialCost(cost) {
  if (!cost) {
    dom.initialLlmCalls.textContent = "--";
    dom.initialLlmCallsDetail.textContent = "Total de llamadas a LM Studio.";
    dom.initialWallClock.textContent = "--";
    dom.initialSequentialWallClock.textContent = "--";
    dom.initialLlmTime.textContent = "--";
    dom.initialEmbeddingTime.textContent = "--";
    dom.initialCallsPerIndividual.textContent = "--";
    return;
  }

  dom.initialLlmCalls.textContent = String(cost.llmCalls ?? 0);
  dom.initialLlmCallsDetail.textContent = `${cost.llmSuccessfulCalls ?? 0} ok, ${cost.llmFailedCalls ?? 0} fallida(s), ${cost.totalTokens ?? 0} tokens reportados.`;
  dom.initialWallClock.textContent = cost.wallClockLabel || cost.processWallClockLabel || "--";
  dom.initialSequentialWallClock.textContent = cost.estimatedSequentialWallClockLabel || "--";
  dom.initialLlmTime.textContent = cost.llmClientWallClockLabel || "--";
  dom.initialEmbeddingTime.textContent = cost.embeddingWallClockLabel || "--";
  dom.initialCallsPerIndividual.textContent = Number.isFinite(Number(cost.llmCallsPerFinalIndividual))
    ? formatNumber(Number(cost.llmCallsPerFinalIndividual), 2)
    : "--";
}

function renderInitialPopulationRun(run) {
  const status = initialPopulationStatusLabel(run.status);
  const rows = run.rows || [];
  const metrics = run.metrics || {};

  dom.initialRunStatus.textContent = status;
  dom.initialRunId.textContent = run.runId || "--";
  dom.initialPopulationConnectionText.textContent = status;
  dom.initialPopulationConnectionDot.classList.toggle("is-busy", run.status === "queued" || run.status === "running");
  dom.initialPopulationConnectionDot.classList.toggle("is-error", run.status === "failed");
  dom.initialFinalIndividuals.textContent = String(metrics.completedRows ?? rows.length ?? 0);
  dom.initialNonDominated.textContent = metrics.completedRows ? `${metrics.nonDominatedRows ?? 0}/${metrics.completedRows}` : "--";
  dom.initialHypervolume.textContent = metrics.hypervolumeLabel || "No aplica";
  dom.initialSpread.textContent = metrics.spreadLabel || "No aplica";

  renderInitialProgress(run.progress || null);
  renderInitialCost(run.cost || null);
  renderInitialSemanticArtifacts(run.semanticArtifacts || null);
  renderInitialRows(rows);
  renderInitialLogs(run.logs || []);
  renderInitialArtifacts(run);

  const detail = run.error
    ? initialPopulationErrorLabel(run.error)
    : run.cancelRequested
      ? "Cancelacion solicitada; esperando cierre del subproceso."
      : run.progress?.detail || `${rows.length} individuo(s) visibles.`;
  setStatus(
    dom.initialStatusTone,
    dom.initialStatusTitle,
    dom.initialStatusDetail,
    `Estrategia ${status}`,
    detail,
    run.status === "running" || run.status === "queued" ? "busy" : run.status === "failed" ? "error" : "ready",
  );
}

function renderInitialSemanticArtifacts(artifacts) {
  const anchors = artifacts?.anchors || {};
  const pools = artifacts?.pools?.components || {};
  renderInitialAnchors(anchors);
  renderInitialPools(pools);
}

function renderInitialAnchors(anchors) {
  const entries = Object.entries(anchors || {}).filter(([, values]) => Array.isArray(values) && values.length);
  if (!entries.length) {
    dom.initialAnchorsContent.textContent = "Sin anclas todavia.";
    return;
  }

  dom.initialAnchorsContent.replaceChildren(
    ...entries.map(([name, values]) => {
      const group = document.createElement("div");
      group.className = "semantic-artifact-group";
      const title = document.createElement("h3");
      title.textContent = `${semanticArtifactLabel(name)} (${values.length})`;
      const list = document.createElement("div");
      list.className = "chip-list";
      values.forEach((value) => {
        const chip = document.createElement("span");
        chip.className = "semantic-chip";
        chip.textContent = value;
        list.append(chip);
      });
      group.append(title, list);
      return group;
    }),
  );
}

function renderInitialPools(pools) {
  const entries = ["roles", "topics", "actions"]
    .map((name) => [name, pools?.[name]])
    .filter(([, pool]) => pool && Array.isArray(pool.items) && pool.items.length);

  if (!entries.length) {
    dom.initialPoolsContent.textContent = "Sin pools todavia.";
    return;
  }

  dom.initialPoolsContent.replaceChildren(
    ...entries.map(([name, pool]) => {
      const group = document.createElement("div");
      group.className = "semantic-artifact-group";
      const title = document.createElement("h3");
      title.textContent = `${semanticArtifactLabel(name)} (${pool.items.length})`;
      const meta = document.createElement("p");
      meta.className = "artifact-meta";
      meta.textContent = `Solicitados: ${pool.requested ?? "--"}; descartados: ${pool.discarded ?? "--"}.`;
      const list = document.createElement("div");
      list.className = "chip-list";
      pool.items.forEach((value) => {
        const chip = document.createElement("span");
        chip.className = "semantic-chip";
        chip.textContent = value;
        list.append(chip);
      });
      group.append(title, meta, list);
      return group;
    }),
  );
}

function semanticArtifactLabel(name) {
  const labels = {
    entities: "Entidades",
    topics: "Topicos",
    actions: "Acciones",
    constraints: "Restricciones",
    roles: "Roles",
  };
  return labels[name] || name;
}

function initialPopulationErrorLabel(error) {
  if (!error) {
    return "--";
  }
  if (typeof error === "string") {
    return error;
  }
  const stage = error.stage ? `${error.stage}: ` : "";
  const details = error.details && typeof error.details === "object" ? error.details : {};
  const count = details.completed !== undefined && details.total !== undefined
    ? ` Completado: ${details.completed}/${details.total}.`
    : "";
  const generated = details.generatedTexts !== undefined
    ? ` Textos generados persistidos: ${details.generatedTexts}.`
    : "";
  const suggestion = details.suggestion ? ` ${details.suggestion}` : "";
  return `${stage}${error.message || JSON.stringify(error)}${count}${generated}${suggestion}`;
}

function renderInitialRows(rows) {
  if (!rows.length) {
    dom.initialResultsBody.innerHTML = '<tr><td colspan="11">Sin resultados todavia.</td></tr>';
    return;
  }

  const orderedRows = [...rows].sort((left, right) => {
    const leftNonDominated = left.nonDominated ? 0 : 1;
    const rightNonDominated = right.nonDominated ? 0 : 1;
    if (leftNonDominated !== rightNonDominated) {
      return leftNonDominated - rightNonDominated;
    }
    return Number(left.rank ?? left.selectionRank ?? Number.MAX_SAFE_INTEGER)
      - Number(right.rank ?? right.selectionRank ?? Number.MAX_SAFE_INTEGER);
  });

  dom.initialResultsBody.replaceChildren(
    ...orderedRows.map((row) => {
      const tr = document.createElement("tr");
      const isEvaluated = Array.isArray(row.objectiveVector) && row.objectiveVector.length > 1 && row.status === "ok";
      if (isEvaluated && !row.nonDominated) {
        tr.classList.add("is-dominated-row");
      }
      if (row.nonDominated) {
        tr.classList.add("is-nondominated-row");
      }
      const dominanceCell = row.nonDominated
        ? '<span class="valid">si</span>'
        : isEvaluated
          ? '<span class="invalid">no</span>'
          : "--";
      tr.innerHTML = `
        <td>${escapeHtml(String(row.rank ?? "--"))}</td>
        <td class="context-cell">${escapeHtml(row.role || "--")}</td>
        <td class="context-cell">${escapeHtml(row.topic || "--")}</td>
        <td class="context-cell">${escapeHtml(row.action || "--")}</td>
        <td class="context-cell long-cell"><div class="scroll-cell">${escapeHtml(row.prompt || "--")}</div></td>
        <td class="context-cell long-cell"><div class="scroll-cell">${escapeHtml(row.generatedText || "--")}</div></td>
        <td>${escapeHtml(formatOptionalNumber(row.fidelity, 6))}</td>
        <td>${escapeHtml(formatOptionalNumber(row.diversity, 6))}</td>
        <td>${escapeHtml(row.objectiveLabel || "--")}</td>
        <td>${dominanceCell}</td>
        <td><span class="${row.status === "ok" ? "valid" : "invalid"}">${escapeHtml(row.status || "--")}</span></td>
      `;
      return tr;
    }),
  );
}

function renderInitialLogs(logs) {
  if (!logs.length) {
    dom.initialLogOutput.textContent = "Sin logs todavia.";
    return;
  }
  dom.initialLogOutput.textContent = logs
    .slice(-60)
    .map((entry) => `[${entry.source || "runner"}] ${entry.message}`)
    .join("\n");
}

function renderInitialArtifacts(run) {
  const error = initialPopulationErrorLabel(run.error);
  const outputDir = run.strategy?.outputDir || run.runDir || "--";
  const cost = run.cost || {};
  renderDefinitionList(dom.initialArtifactDetails, [
    ["Estado", initialPopulationStatusLabel(run.status)],
    ["Salida", outputDir],
    ["Error", error === "--" ? "No aplica" : error],
    ["Llamadas / texto generado", Number.isFinite(Number(cost.llmCallsPerGeneratedText)) ? formatNumber(Number(cost.llmCallsPerGeneratedText), 2) : "--"],
    ["Wall-clock sin paralelismo", cost.estimatedSequentialWallClockLabel || "--"],
    ["Ahorro por paralelismo", cost.parallelismSavingsLabel || "--"],
    ["Embeddings", `${cost.embeddingBatches ?? 0} batch(es), ${cost.embeddingTexts ?? 0} texto(s)`],
    ["Tokens", `${cost.promptTokens ?? 0} prompt, ${cost.completionTokens ?? 0} completion`],
  ]);
}

function comparatorStatusLabel(status) {
  const labels = {
    queued: "en cola",
    running: "ejecutando",
    completed: "completado",
    failed: "fallido",
    cancelled: "cancelado",
  };
  return labels[status] || status || "--";
}

function comparatorStatusClass(status) {
  if (status === "completed" || status === "ok") {
    return "valid";
  }
  if (status === "queued" || status === "running") {
    return "in-progress";
  }
  return "invalid";
}

function setComparatorRunning(isRunning, cancelRequested = false) {
  dom.runComparatorButton.disabled = isRunning;
  dom.cancelComparatorButton.disabled = !isRunning || !currentComparatorRunId || cancelRequested;
  dom.cancelComparatorButton.textContent = cancelRequested ? "Cancelando..." : "Cancelar";
  dom.clearComparatorButton.disabled = isRunning;
  [
    dom.comparatorReferenceText,
    dom.comparatorModel,
    dom.comparatorTopK,
    dom.comparatorProposalParallelism,
    dom.comparatorTimeoutMinutes,
    dom.comparatorN,
    dom.comparatorGeneraciones,
    dom.comparatorK,
    dom.comparatorProbCrossover,
    dom.comparatorProbMutacion,
  ].forEach((field) => {
    field.disabled = isRunning;
  });
}

function stopComparatorPolling() {
  if (comparatorPollTimer) {
    window.clearInterval(comparatorPollTimer);
    comparatorPollTimer = null;
  }
}

function resetComparatorUi() {
  stopComparatorPolling();
  currentComparatorRunId = null;
  dom.comparatorRunStatus.textContent = "--";
  dom.comparatorProgressPercent.textContent = "--";
  dom.comparatorProgressSummary.textContent = "Sin corrida activa.";
  dom.comparatorLlmCalls.textContent = "--";
  dom.comparatorLlmCallsDetail.textContent = "Total de llamadas reales a Ollama.";
  dom.comparatorWallClock.textContent = "--";
  dom.comparatorLlmTime.textContent = "--";
  dom.comparatorCompletedProposals.textContent = "--";
  dom.comparatorShownRows.textContent = "--";
  dom.comparatorRunId.textContent = "--";
  dom.comparatorConnectionText.textContent = "Sin ejecucion";
  dom.comparatorConnectionDot.classList.remove("is-busy", "is-error");
  dom.comparatorResultsBody.innerHTML = '<tr><td colspan="8">Sin resultados todavia.</td></tr>';
  dom.comparatorProposalCards.innerHTML = `
    <article class="proposal-card">
      <strong>Sin corrida</strong>
      <p>Ejecuta el comparador para ver los resumenes.</p>
    </article>
  `;
  dom.comparatorLogOutput.textContent = "Sin logs todavia.";
  renderComparatorProgress(null);
  setComparatorRunning(false);
  setStatus(
    dom.comparatorStatusTone,
    dom.comparatorStatusTitle,
    dom.comparatorStatusDetail,
    "Listo",
    "Ejecuta EVOLMD y EVOLMD-MO desde sus clones Python usando Ollama local.",
  );
}

function readComparatorConfig() {
  const referenceText = dom.comparatorReferenceText.value.trim();
  const model = dom.comparatorModel.value.trim();
  if (!referenceText) {
    throw new Error("Define el texto de referencia.");
  }
  if (!model) {
    throw new Error("Define el modelo Ollama.");
  }

  return {
    referenceText,
    model,
    topK: Math.floor(readClampedNumber(dom.comparatorTopK, "Top K tabla", 1, 200)),
    n: Math.floor(readClampedNumber(dom.comparatorN, "N individuos", 1, 500)),
    generaciones: Math.floor(readClampedNumber(dom.comparatorGeneraciones, "Generaciones", 0, 500)),
    k: Math.floor(readClampedNumber(dom.comparatorK, "K torneo", 1, 100)),
    probCrossover: readClampedNumber(dom.comparatorProbCrossover, "Prob. crossover", 0, 1),
    probMutacion: readClampedNumber(dom.comparatorProbMutacion, "Prob. mutacion", 0, 1),
    proposalParallelism: Math.floor(readClampedNumber(dom.comparatorProposalParallelism, "Propuestas paralelas", 1, 8)),
    timeoutMinutes: Math.floor(readClampedNumber(dom.comparatorTimeoutMinutes, "Timeout por propuesta", 1, 1440)),
  };
}

async function requestComparatorJson(path, options = {}) {
  const response = await fetch(`${COMPARATOR_API}${path}`, {
    headers: {
      "Content-Type": "application/json",
      ...(options.headers || {}),
    },
    ...options,
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.error || `HTTP ${response.status}`);
  }
  return payload;
}

async function loadComparatorProposals() {
  try {
    const payload = await requestComparatorJson("/proposals");
    const proposalSummary = payload.proposals
      .map((proposal) => `${proposal.displayName}: ${proposal.available ? "clonado" : "faltante"} (${proposal.objectiveNames.join(", ")})`)
      .join("; ");
    renderDefinitionList(dom.comparatorIntegrationDetails, [
      ["Contrato", "{proposalId, displayName, rows, metrics, outputDir, status}"],
      ["Propuestas", proposalSummary],
      ["EVOLMD", "data_final_evaluada.json -> [fitness]"],
      ["EVOLMD post-hoc", "Vector diagnostico [fitness, semantic_diversity_posthoc]; HV/spread no alteran la seleccion."],
      ["EVOLMD-MO", "pareto_front.json -> [fidelity_sbert, diversity_individual]"],
      ["MO", "HV con referencia [0,0]; spread menor es mejor; para EVOLMD son diagnosticos post-hoc."],
    ]);
  } catch (error) {
    renderDefinitionList(dom.comparatorIntegrationDetails, [
      ["Contrato", "{proposalId, displayName, rows, metrics, outputDir, status}"],
      ["API", `No se pudo consultar el backend: ${error.message}`],
    ]);
  }
}

async function runComparator() {
  let config;
  try {
    config = readComparatorConfig();
  } catch (error) {
    setStatus(dom.comparatorStatusTone, dom.comparatorStatusTitle, dom.comparatorStatusDetail, "Configuracion incompleta", error.message, "error");
    return;
  }

  stopComparatorPolling();
  setComparatorRunning(true);
  dom.comparatorResultsBody.innerHTML = '<tr><td colspan="7">Esperando resultados.</td></tr>';
  dom.comparatorProposalCards.innerHTML = "";
  dom.comparatorLogOutput.textContent = "Iniciando corrida...";
  dom.comparatorConnectionDot.classList.add("is-busy");
  dom.comparatorConnectionDot.classList.remove("is-error");
  dom.comparatorConnectionText.textContent = "Ejecutando";
  setStatus(
    dom.comparatorStatusTone,
    dom.comparatorStatusTitle,
    dom.comparatorStatusDetail,
    "Iniciando comparador",
    "El backend ejecutara los clones Python y la web consultara el progreso.",
    "busy",
  );

  try {
    const run = await requestComparatorJson("/runs", {
      method: "POST",
      body: JSON.stringify(config),
    });
    currentComparatorRunId = run.runId;
    setComparatorRunning(true, Boolean(run.cancelRequested));
    renderComparatorRun(run);
    comparatorPollTimer = window.setInterval(() => refreshComparatorRun(currentComparatorRunId), 2000);
    await refreshComparatorRun(currentComparatorRunId);
  } catch (error) {
    currentComparatorRunId = null;
    dom.comparatorConnectionDot.classList.add("is-error");
    dom.comparatorConnectionText.textContent = "Error";
    setComparatorRunning(false);
    setStatus(dom.comparatorStatusTone, dom.comparatorStatusTitle, dom.comparatorStatusDetail, "Error al iniciar", error.message, "error");
  }
}

async function refreshComparatorRun(runId) {
  if (!runId) {
    return;
  }

  try {
    const run = await requestComparatorJson(`/runs/${encodeURIComponent(runId)}`);
    renderComparatorRun(run);
    if (COMPARATOR_TERMINAL_STATUSES.has(run.status)) {
      stopComparatorPolling();
      currentComparatorRunId = run.runId;
      setComparatorRunning(false);
    } else {
      currentComparatorRunId = run.runId;
      setComparatorRunning(true, Boolean(run.cancelRequested));
    }
  } catch (error) {
    stopComparatorPolling();
    setComparatorRunning(false);
    dom.comparatorConnectionDot.classList.add("is-error");
    setStatus(dom.comparatorStatusTone, dom.comparatorStatusTitle, dom.comparatorStatusDetail, "Error al consultar corrida", error.message, "error");
  }
}

async function cancelComparatorRun() {
  if (!currentComparatorRunId) {
    return;
  }
  dom.cancelComparatorButton.disabled = true;
  dom.cancelComparatorButton.textContent = "Cancelando...";
  setStatus(dom.comparatorStatusTone, dom.comparatorStatusTitle, dom.comparatorStatusDetail, "Cancelando", "Se solicitara terminar el proceso activo.", "busy");
  try {
    const run = await requestComparatorJson(`/runs/${encodeURIComponent(currentComparatorRunId)}/cancel`, {
      method: "POST",
      body: "{}",
    });
    renderComparatorRun(run);
    setComparatorRunning(!COMPARATOR_TERMINAL_STATUSES.has(run.status), Boolean(run.cancelRequested));
  } catch (error) {
    setComparatorRunning(true, false);
    setStatus(dom.comparatorStatusTone, dom.comparatorStatusTitle, dom.comparatorStatusDetail, "No se pudo cancelar", error.message, "error");
  }
}

function flattenComparatorRows(run) {
  return (run.proposals || []).flatMap((proposal) => (
    (proposal.rows || []).map((row) => ({
      ...row,
      proposalStatus: proposal.status,
      proposalError: proposal.error,
    }))
  ));
}

function comparatorProposalViews(run) {
  const finalById = new Map((run.proposals || []).map((proposal) => [proposal.proposalId, proposal]));
  const states = Object.values(run.proposalStates || {});
  const merged = states.map((state) => ({
    ...state,
    ...(finalById.get(state.proposalId) || {}),
    progressState: state,
  }));
  const knownIds = new Set(merged.map((proposal) => proposal.proposalId));
  for (const proposal of run.proposals || []) {
    if (!knownIds.has(proposal.proposalId)) {
      merged.push(proposal);
    }
  }
  return merged;
}

function renderComparatorProgress(progress) {
  if (!progress) {
    dom.comparatorProgressPercent.textContent = "--";
    dom.comparatorProgressSummary.textContent = "Sin corrida activa.";
    dom.comparatorProgressDetail.textContent = "Sin ejecucion.";
    dom.comparatorProgressBar.style.width = "0%";
    renderDefinitionList(dom.comparatorProgressDetails, [
      ["Tiempo transcurrido", "--"],
      ["Tiempo restante", "--"],
      ["Propuesta activa", "--"],
      ["Cola", "--"],
    ]);
    return;
  }

  const percent = Math.max(0, Math.min(100, Number(progress.percent || 0)));
  dom.comparatorProgressPercent.textContent = `${percent}%`;
  dom.comparatorProgressSummary.textContent = progress.detail || "Ejecutando.";
  dom.comparatorProgressDetail.textContent = progress.detail || "Ejecutando.";
  dom.comparatorProgressBar.style.width = `${percent}%`;
  renderDefinitionList(dom.comparatorProgressDetails, [
    ["Tiempo transcurrido", progress.elapsedLabel || "--"],
    ["Tiempo restante estimado", progress.remainingLabel || "No disponible"],
    ["Propuesta activa", progress.activeProposalName || "--"],
    ["Cola", `${progress.queuedProposals ?? 0}/${progress.totalProposals ?? 0}`],
  ]);
}

function renderComparatorCostSummary(costSummary) {
  if (!costSummary) {
    dom.comparatorLlmCalls.textContent = "--";
    dom.comparatorLlmCallsDetail.textContent = "Total de llamadas reales a Ollama.";
    dom.comparatorWallClock.textContent = "--";
    dom.comparatorLlmTime.textContent = "--";
    return;
  }

  const successful = costSummary.llmSuccessfulCalls ?? 0;
  const failed = costSummary.llmFailedCalls ?? 0;
  dom.comparatorLlmCalls.textContent = String(costSummary.llmCalls ?? 0);
  dom.comparatorLlmCallsDetail.textContent = `${successful} ok, ${failed} fallida(s), ${costSummary.totalTokens ?? 0} tokens reportados.`;
  dom.comparatorWallClock.textContent = costSummary.runWallClockLabel || "--";
  dom.comparatorLlmTime.textContent = costSummary.llmClientWallClockLabel || "--";
}

function renderComparatorRun(run) {
  const rows = flattenComparatorRows(run);
  const progress = run.progress || {};
  const completedProposals = progress.completedProposals ?? (run.proposals || []).filter((proposal) => proposal.status === "completed").length;
  const fallbackTotalProposals = Object.keys(run.proposalStates || {}).length || (run.proposals || []).length || 2;
  const totalProposals = progress.totalProposals ?? fallbackTotalProposals;
  const status = comparatorStatusLabel(run.status);

  dom.comparatorRunStatus.textContent = status;
  dom.comparatorCompletedProposals.textContent = `${completedProposals}/${totalProposals}`;
  dom.comparatorShownRows.textContent = String(rows.length);
  dom.comparatorRunId.textContent = run.runId || "--";
  dom.comparatorConnectionText.textContent = status;
  dom.comparatorConnectionDot.classList.toggle("is-busy", run.status === "queued" || run.status === "running");
  dom.comparatorConnectionDot.classList.toggle("is-error", run.status === "failed");

  renderComparatorProgress(run.progress || null);
  renderComparatorCostSummary(run.costSummary || null);
  renderComparatorCards(comparatorProposalViews(run));
  renderComparatorRows(rows);
  renderComparatorLogs(run.logs || []);

  const detail = run.error
    ? run.error
    : run.cancelRequested
      ? "Cancelacion solicitada; esperando cierre de procesos activos."
      : progress.detail || `${completedProposals} propuesta(s) completada(s), ${rows.length} fila(s) visibles.`;
  setStatus(
    dom.comparatorStatusTone,
    dom.comparatorStatusTitle,
    dom.comparatorStatusDetail,
    `Comparador ${status}`,
    detail,
    run.status === "running" || run.status === "queued" ? "busy" : run.status === "failed" ? "error" : "ready",
  );
}

function renderComparatorCards(proposals) {
  if (!proposals.length) {
    dom.comparatorProposalCards.innerHTML = `
      <article class="proposal-card">
        <strong>Esperando propuestas</strong>
        <p>El backend aun no ha devuelto resultados normalizados.</p>
      </article>
    `;
    return;
  }

  dom.comparatorProposalCards.replaceChildren(
    ...proposals.map((proposal) => {
      const metrics = proposal.metrics || {};
      const cost = proposal.cost || {};
      const hvLabel = metrics.postHocDiagnostic ? "HV post-hoc" : "HV";
      const spreadLabel = metrics.postHocDiagnostic ? "Spread post-hoc" : "Spread";
      const nonDominatedLabel = metrics.postHocDiagnostic ? "No dom. post-hoc" : "No dominadas";
      const progressState = proposal.progressState || proposal;
      const progressPercent = Math.round(Math.max(0, Math.min(1, Number(progressState.progress || 0))) * 100);
      const stageLabel = progressState.stageLabel || comparatorStatusLabel(proposal.status);
      const article = document.createElement("article");
      article.className = "proposal-card";
      article.innerHTML = `
        <strong>${escapeHtml(proposal.displayName || proposal.proposalId)}</strong>
        <p><span class="${comparatorStatusClass(proposal.status)}">${escapeHtml(comparatorStatusLabel(proposal.status))}</span>${proposal.error ? `: ${escapeHtml(proposal.error)}` : ""}</p>
        <div class="mini-progress" aria-label="Progreso ${escapeHtml(proposal.displayName || proposal.proposalId)}">
          <span style="width: ${progressPercent}%"></span>
        </div>
        <p>${escapeHtml(stageLabel)} (${progressPercent}%)</p>
        <dl>
          <dt>Filas</dt><dd>${escapeHtml(String(metrics.completedRows ?? 0))}/${escapeHtml(String(metrics.totalRows ?? 0))}</dd>
          <dt>Mejor F.O.</dt><dd>${escapeHtml(metrics.bestObjectiveLabel || "--")}</dd>
          <dt>${escapeHtml(nonDominatedLabel)}</dt><dd>${escapeHtml(String((metrics.postHocDiagnostic ? metrics.postHocNonDominatedRows : metrics.nonDominatedRows) ?? 0))}</dd>
          <dt>${escapeHtml(hvLabel)}</dt><dd>${escapeHtml(metrics.hypervolumeLabel || "No aplica")}</dd>
          <dt>${escapeHtml(spreadLabel)}</dt><dd>${escapeHtml(metrics.spreadLabel || "No aplica")}</dd>
          ${metrics.postHocDiagnostic ? `<dt>Vector post-hoc</dt><dd>${escapeHtml(metrics.bestDiagnosticObjectiveLabel || "--")}</dd>` : ""}
          <dt>Wall-clock</dt><dd>${escapeHtml(cost.processWallClockLabel || "--")}</dd>
          <dt>Llamadas LLM</dt><dd>${escapeHtml(String(cost.llmCalls ?? 0))}</dd>
          <dt>Tiempo LLM</dt><dd>${escapeHtml(cost.llmClientWallClockLabel || "--")}</dd>
          <dt>Prom. llamada</dt><dd>${escapeHtml(cost.llmAverageCallLabel || "No disponible")}</dd>
          <dt>Tokens</dt><dd>${escapeHtml(String(cost.totalTokens ?? 0))}</dd>
          <dt>Salida</dt><dd>${escapeHtml(metrics.outputDir || proposal.outputDir || "--")}</dd>
        </dl>
      `;
      return article;
    }),
  );
}

function renderComparatorRows(rows) {
  if (!rows.length) {
    dom.comparatorResultsBody.innerHTML = '<tr><td colspan="8">Sin resultados todavia.</td></tr>';
    return;
  }

  dom.comparatorResultsBody.replaceChildren(
    ...rows.map((row) => {
      const tr = document.createElement("tr");
      const status = row.status || row.proposalStatus || "--";
      const nonDominatedLabel = row.nonDominated
        ? '<span class="valid">si</span>'
        : row.postHocNonDominated
          ? '<span class="valid">si post-hoc</span>'
          : "--";
      tr.innerHTML = `
        <td>${escapeHtml(row.displayName || row.proposalId)}</td>
        <td>${escapeHtml(String(row.rank ?? "--"))}</td>
        <td class="context-cell long-cell">${escapeHtml(row.generatedText || "--")}</td>
        <td>${escapeHtml(row.objectiveLabel || "--")}</td>
        <td>${escapeHtml(row.diagnosticObjectiveLabel || "--")}</td>
        <td class="context-cell long-cell">${escapeHtml(row.prompt || "--")}</td>
        <td><span class="${comparatorStatusClass(status)}">${escapeHtml(status)}</span></td>
        <td>${nonDominatedLabel}</td>
      `;
      return tr;
    }),
  );
}

function renderComparatorLogs(logs) {
  if (!logs.length) {
    dom.comparatorLogOutput.textContent = "Sin logs todavia.";
    return;
  }
  dom.comparatorLogOutput.textContent = logs
    .slice(-40)
    .map((entry) => `[${entry.proposalId}] ${entry.message}`)
    .join("\n");
}

async function loadPsoDatabase() {
  if (psoDatabase) {
    return psoDatabase;
  }

  const response = await fetch("./data/pso-individuals.json");
  if (!response.ok) {
    throw new Error(`No se pudo cargar la BD PSO: HTTP ${response.status}`);
  }
  const payload = await response.json();
  if (!Array.isArray(payload.individuals) || payload.individuals.length < 1) {
    throw new Error("La BD PSO no contiene individuos.");
  }
  const invalidIndex = payload.individuals.findIndex((individual) => !isValidPsoIndividual(individual));
  if (invalidIndex >= 0) {
    throw new Error(`La BD PSO tiene un individuo incompleto en la posición ${invalidIndex + 1}.`);
  }

  psoDatabase = payload;
  dom.psoDbStatus.textContent = `BD PSO: ${payload.individuals.length} individuos cargados desde LLM/data/pso-individuals.json`;
  return psoDatabase;
}

function psoOtherComponents(position, componentKey) {
  return PSO_COMPONENTS
    .filter((component) => component.key !== componentKey)
    .map((component) => `${component.promptName} = ${position[component.key]}`)
    .join("; ");
}

function buildPsoPrompt(movement, config) {
  return renderPromptWithVariables({
    num_candidates: String(config.expectedCandidates),
    component_name: movement.component.promptName,
    component_definition: movement.component.definition,
    current_component: movement.current,
    target_component: movement.target || "",
    other_components: movement.otherComponents,
    reference_text: config.referenceText,
  }, config.promptTemplate);
}

function isCompletePsoVector(vector) {
  return PSO_COMPONENTS.every((component) => typeof vector?.[component.key] === "string" && vector[component.key].trim());
}

function isValidPsoIndividual(individual) {
  return typeof individual?.id === "string"
    && individual.id.trim()
    && isCompletePsoVector(individual)
    && isCompletePsoVector(individual.pbest)
    && isCompletePsoVector(individual.lider);
}

function createPsoMovementGroups(individuals, config) {
  const groups = [];
  let movementCount = 0;

  individuals.forEach((individual) => {
    const position = {
      rol: individual.rol,
      topico: individual.topico,
      accion: individual.accion,
    };
    const plans = [];

    PSO_COMPONENTS.forEach((component) => {
      const randomValue = Math.random();
      if (randomValue <= config.psoChangeThreshold) {
        return;
      }

      const operator = config.psoOperator;
      let targetSource = "turbulence";
      if (operator === "influence") {
        targetSource = Math.random() < 0.5 ? "pbest" : "lider";
      }
      movementCount += 1;
      plans.push({
        id: `mov-${String(movementCount).padStart(4, "0")}`,
        number: movementCount,
        individualId: individual.id,
        operator,
        component,
        componentKey: component.key,
        source: targetSource,
        randomValue,
        target: operator === "influence" ? individual[targetSource][component.key] : null,
      });
    });

    if (plans.length > 0) {
      groups.push({ individualId: individual.id, position, plans });
    }
  });

  return { groups, movementCount };
}

function buildPsoMovementFromPlan(plan, position) {
  return {
    ...plan,
    current: position[plan.componentKey],
    otherComponents: psoOtherComponents(position, plan.componentKey),
  };
}

async function scoreTurbulencePsoMovement(movement, candidates, config) {
  const embeddings = await requestEmbeddings(config.embeddingModel, [movement.current, ...candidates]);
  const currentEmbedding = embeddings[0];
  const validationConfig = {
    ...config,
    current: movement.current,
    target: null,
  };

  const rows = candidates.map((candidate, index) => {
    const similarity = cosineSimilarity(embeddings[index + 1], currentEmbedding);
    const diversity = cosineDistance(similarity);
    const baseReason = validationReason(candidate, candidates, validationConfig);
    const reason = baseReason !== "valid"
      ? baseReason
      : similarity < config.turbulenceMinSimilarity ? "too_distant_from_current"
      : similarity > config.turbulenceMaxSimilarity ? "too_close_to_current" : "valid";
    return {
      movementNumber: movement.number,
      individualId: movement.individualId,
      operator: movement.operator,
      componentName: movement.component.promptName,
      source: movement.source,
      current: movement.current,
      reference: movement.current,
      candidate,
      similarity,
      improvement: null,
      diversity,
      reason,
      isValidImprovement: reason === "valid",
      applied: false,
    };
  });

  const selectedCandidate = randomItem(rows.filter((row) => row.isValidImprovement));
  if (selectedCandidate) {
    selectedCandidate.applied = true;
  }

  return {
    baseline: null,
    rows,
    selectedCandidate,
    failureReason: selectedCandidate ? null : "sin candidato en rango de turbulencia",
  };
}

async function scorePsoMovement(movement, rawOutput, config) {
  const candidates = parseCandidates(rawOutput, config.expectedCandidates);
  if (candidates.length === 0) {
    return {
      baseline: null,
      rows: [],
      selectedCandidate: null,
      failureReason: "sin candidatos parseables",
    };
  }

  if (movement.operator === "turbulence") {
    return scoreTurbulencePsoMovement(movement, candidates, config);
  }

  const embeddings = await requestEmbeddings(config.embeddingModel, [movement.current, movement.target, ...candidates]);
  const currentEmbedding = embeddings[0];
  const targetEmbedding = embeddings[1];
  const baseline = cosineSimilarity(currentEmbedding, targetEmbedding);
  const validationConfig = {
    ...config,
    current: movement.current,
    target: movement.target,
  };

  const rows = candidates.map((candidate, index) => {
    const similarity = cosineSimilarity(embeddings[index + 2], targetEmbedding);
    const improvement = similarity - baseline;
    const baseReason = validationReason(candidate, candidates, validationConfig);
    const hasSemanticProgress = improvement > config.margin;
    const reason = baseReason !== "valid"
      ? baseReason
      : similarity >= config.targetCopyThreshold ? "semantic_copy_target"
      : hasSemanticProgress ? "valid" : "insufficient_semantic_progress";
    return {
      movementNumber: movement.number,
      individualId: movement.individualId,
      operator: movement.operator,
      componentName: movement.component.promptName,
      source: movement.source,
      current: movement.current,
      reference: movement.target,
      candidate,
      similarity,
      improvement,
      diversity: null,
      reason,
      isValidImprovement: reason === "valid",
      applied: false,
    };
  });

  const selectedCandidate = rows
    .filter((row) => row.isValidImprovement)
    .sort((a, b) => b.similarity - a.similarity)[0] || null;
  if (selectedCandidate) {
    selectedCandidate.applied = true;
  }

  return {
    baseline,
    rows,
    selectedCandidate,
    failureReason: selectedCandidate ? null : "sin candidato con progreso semántico",
  };
}

function renderPsoMovementRows(movementResults) {
  if (movementResults.length === 0) {
    dom.runResultsBody.innerHTML = '<tr><td colspan="6">Sin movimientos PSO todavía.</td></tr>';
    return;
  }

  dom.runResultsBody.replaceChildren(
    ...movementResults.map((result) => {
      const tr = document.createElement("tr");
      const movement = result.movement;
      const selected = result.selectedCandidate;
      const status = result.error
        ? `<span class="invalid">${escapeHtml(result.error)}</span>`
        : selected ? '<span class="valid">aplicado</span>' : `<span class="invalid">${escapeHtml(result.failureReason)}</span>`;
      tr.innerHTML = `
        <td>${movement.number}</td>
        <td>${escapeHtml(movement.individualId)}</td>
        <td>${escapeHtml(movement.component.promptName)}</td>
        <td>${escapeHtml(movementOperatorLabel(movement))}</td>
        <td>${selected ? escapeHtml(selected.candidate) : "--"}</td>
        <td>${status}</td>
      `;
      return tr;
    }),
  );
}

function appendPsoCandidateRows(rows) {
  if (rows.length === 0) {
    return;
  }

  if (latestCheckerRows.length === 0) {
    dom.candidateResultsBody.replaceChildren();
  }

  latestCheckerRows.push(...rows);
  rows.forEach((row) => {
    const tr = document.createElement("tr");
    const reference = row.reference ?? row.target ?? "--";
    const delta = row.operator === "turbulence" ? formatNumber(row.diversity) : formatSigned(row.improvement);
    tr.innerHTML = `
      <td>${row.movementNumber}</td>
      <td>${escapeHtml(row.individualId)}</td>
      <td>${escapeHtml(row.componentName)}</td>
      <td>${escapeHtml(movementOperatorLabel(row))}</td>
      <td class="context-cell">${escapeHtml(row.current)}</td>
      <td class="context-cell">${escapeHtml(reference)}</td>
      <td class="context-cell">${escapeHtml(row.candidate)}</td>
      <td>${formatNumber(row.similarity)}</td>
      <td>${delta}</td>
      <td><span class="${row.isValidImprovement ? "valid" : "invalid"}">${row.reason}</span></td>
      <td>${row.applied ? '<span class="valid">sí</span>' : "--"}</td>
    `;
    dom.candidateResultsBody.append(tr);
  });
}

function summarizePsoResults(movementResults, totalSelected = movementResults.length) {
  const selectedCount = totalSelected;
  const processedCount = movementResults.length;
  const applied = movementResults.filter((result) => result.selectedCandidate).length;
  const errored = movementResults.filter((result) => result.error).length;
  const failed = processedCount - applied;
  const candidateFailures = movementResults.filter((result) => !result.error && !result.selectedCandidate).length;
  const affectedIndividuals = new Set(
    movementResults
      .filter((result) => result.selectedCandidate)
      .map((result) => result.movement.individualId),
  ).size;
  const validCandidates = latestCheckerRows.filter((row) => row.reason === "valid").length;
  const timedCalls = movementResults.filter((result) => Number.isFinite(result.elapsedMs));
  const totalLlmElapsedMs = timedCalls.reduce((sum, result) => sum + result.elapsedMs, 0);
  const averageLlmElapsedMs = timedCalls.length ? totalLlmElapsedMs / timedCalls.length : NaN;
  const operator = movementResults[0]?.movement.operator || selectedPsoOperator();

  dom.psoSelectedComponents.textContent = String(selectedCount);
  dom.psoFailedChanges.textContent = String(failed);
  dom.psoFailureRate.textContent = processedCount ? `${formatNumber((failed / processedCount) * 100, 2)}%` : "--";
  dom.psoAppliedChanges.textContent = String(applied);
  dom.psoAffectedIndividuals.textContent = String(affectedIndividuals);
  dom.psoLlmCalls.textContent = String(processedCount);
  dom.psoAverageLlmTime.textContent = formatDuration(averageLlmElapsedMs);
  dom.psoTotalLlmTime.textContent = timedCalls.length ? formatDuration(totalLlmElapsedMs) : "--";
  dom.psoValidCandidates.textContent = String(validCandidates);
  dom.psoProgressFailures.textContent = String(candidateFailures);
  dom.psoExecutionErrors.textContent = String(errored);

  renderDefinitionList(dom.bestCandidateDetails, [
    ["Operador", psoOperatorLabel(operator)],
    ["Componentes sorteadas", String(selectedCount)],
    ["Cambios aplicados", String(applied)],
    ["No pudieron cambiar", String(failed)],
    ["Tiempo LLM total", timedCalls.length ? formatDuration(totalLlmElapsedMs) : "--"],
    ["Tiempo promedio / llamada", formatDuration(averageLlmElapsedMs)],
    ["Causa principal", candidateFailures > 0 ? "Sin candidato aceptable para uno o más movimientos." : "Sin fallos de candidato registrados."],
  ]);
}

async function runPsoSimulation(config) {
  const database = await loadPsoDatabase();
  const individuals = database.individuals.slice(0, config.psoIndividualCount);
  const { groups, movementCount } = createPsoMovementGroups(individuals, config);
  const movementResults = [];
  let nextGroupIndex = 0;

  resetPsoMetrics();
  setResultsTableMode("pso");
  dom.renderedPromptPreview.textContent = groups[0]?.plans[0]
    ? buildPsoPrompt(buildPsoMovementFromPlan(groups[0].plans[0], groups[0].position), config)
    : "Ninguna componente fue sorteada para cambiar con el umbral actual.";
  dom.psoSelectedComponents.textContent = String(movementCount);

  if (movementCount === 0) {
    summarizePsoResults([], 0);
    setStatus(dom.checkerStatusTone, dom.checkerStatusTitle, dom.checkerStatusDetail, "Simulación sin movimientos", "Ninguna componente superó el umbral de sorteo.");
    return;
  }

  async function groupWorker() {
    while (!stopRequested && nextGroupIndex < groups.length) {
      const group = groups[nextGroupIndex];
      nextGroupIndex += 1;

      for (const plan of group.plans) {
        if (stopRequested) {
          break;
        }

        const movement = buildPsoMovementFromPlan(plan, group.position);
        setStatus(
          dom.checkerStatusTone,
          dom.checkerStatusTitle,
          dom.checkerStatusDetail,
          "Simulando iteración PSO",
          `Movimiento ${movement.number} de ${movementCount}: ${movement.individualId}/${movement.component.promptName} con ${movementOperatorLabel(movement)}.`,
          "busy",
        );

        const requestStarted = performance.now();
        try {
          const prompt = buildPsoPrompt(movement, config);
          const response = await callLmStudio(prompt, movement.number, config);
          const scored = await scorePsoMovement(movement, response.raw, config);
          if (scored.selectedCandidate) {
            group.position[movement.componentKey] = scored.selectedCandidate.candidate;
          }

          movementResults.push({
            movement,
            baseline: scored.baseline,
            selectedCandidate: scored.selectedCandidate,
            failureReason: scored.failureReason,
            error: null,
            elapsedMs: response.elapsedMs,
          });
          appendPsoCandidateRows(scored.rows);
        } catch (error) {
          movementResults.push({
            movement,
            baseline: null,
            selectedCandidate: null,
            failureReason: "error de ejecución",
            error: error.name === "AbortError" ? "timeout" : error.message,
            elapsedMs: performance.now() - requestStarted,
          });
        }

        movementResults.sort((a, b) => a.movement.number - b.movement.number);
        renderPsoMovementRows(movementResults);
        summarizePsoResults(movementResults, movementCount);
      }
    }
  }

  await Promise.all(Array.from({ length: Math.min(config.concurrency, groups.length) }, () => groupWorker()));
  setStatus(
    dom.checkerStatusTone,
    dom.checkerStatusTitle,
    dom.checkerStatusDetail,
    stopRequested ? "Simulación detenida" : "Simulación PSO completada",
    `${movementResults.length} movimiento(s) procesado(s) de ${movementCount} sorteado(s).`,
  );
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
  resetPsoMetrics();
  dom.renderedPromptPreview.textContent = config.prompt;
  setCheckerRunning(true);
  setResultsTableMode(config.simulatePso ? "pso" : "standard");
  setStatus(
    dom.checkerStatusTone,
    dom.checkerStatusTitle,
    dom.checkerStatusDetail,
    config.simulatePso ? "Preparando simulación PSO" : "Calculando baseline",
    config.simulatePso ? "Cargando la BD de individuos y sorteando componentes." : "Generando embeddings de Current y Target.",
    "busy",
  );

  const runResults = [];
  let nextRun = 1;

  try {
    if (config.simulatePso) {
      await runPsoSimulation(config);
      return;
    }

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
  resetPsoMetrics();
  setResultsTableMode(dom.simulatePsoSwitch.checked ? "pso" : "standard");
});
dom.loadSolutionModelsButton.addEventListener("click", fetchSolutionLmStudioModels);
dom.previewSolutionsButton.addEventListener("click", previewSolutions);
dom.runSolutionsButton.addEventListener("click", runSolutionEvaluation);
dom.stopSolutionsButton.addEventListener("click", () => {
  solutionEvalStopRequested = true;
  setStatus(dom.solutionStatusTone, dom.solutionStatusTitle, dom.solutionStatusDetail, "Deteniendo", "Se terminarán las solicitudes ya iniciadas.", "busy");
});
dom.clearSolutionResultsButton.addEventListener("click", clearSolutionResults);
dom.solutionReferencePreset.addEventListener("change", applySolutionReferencePreset);
dom.solutionCount.addEventListener("change", previewSolutions);
dom.solutionPromptTemplate.addEventListener("input", previewSolutions);
dom.solutionReferenceText.addEventListener("input", previewSolutions);
dom.loadInitialModelsButton.addEventListener("click", loadInitialPopulationModels);
dom.runInitialPopulationButton.addEventListener("click", runInitialPopulation);
dom.cancelInitialPopulationButton.addEventListener("click", cancelInitialPopulationRun);
dom.clearInitialPopulationButton.addEventListener("click", resetInitialPopulationUi);
dom.runComparatorButton.addEventListener("click", runComparator);
dom.cancelComparatorButton.addEventListener("click", cancelComparatorRun);
dom.clearComparatorButton.addEventListener("click", resetComparatorUi);
dom.solutionLlmModelSelect.addEventListener("change", () => {
  if (dom.solutionLlmModelSelect.value) {
    dom.solutionLlmModelManual.value = dom.solutionLlmModelSelect.value;
  }
});
dom.solutionLmApiMode.addEventListener("change", () => {
  dom.solutionLmEndpoint.value = endpointForMode(dom.solutionLmApiMode.value);
  dom.solutionLlmModelSelect.replaceChildren(new Option("Cargar modelos desde LM Studio", ""));
  dom.solutionLlmModelManual.value = "meta-llama-3.1-8b-instruct";
  dom.solutionConnectionText.textContent = "Sin probar conexión";
  dom.solutionConnectionDot.classList.remove("is-error", "is-busy");
});
dom.simulatePsoSwitch.addEventListener("change", () => {
  updateSimulationModeUi();
  if (dom.simulatePsoSwitch.checked) {
    loadPsoDatabase().catch((error) => {
      dom.psoDbStatus.textContent = `BD PSO: ${error.message}`;
      setStatus(dom.checkerStatusTone, dom.checkerStatusTitle, dom.checkerStatusDetail, "Error al cargar BD PSO", error.message, "error");
    });
  }
});
dom.psoSemanticOperator.addEventListener("change", applyPsoOperatorTemplate);
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
populateSolutionReferencePresets();
applySolutionReferencePreset();
resetSolutionMetrics();
resetInitialPopulationUi();
loadInitialPopulationStrategies();
resetComparatorUi();
loadComparatorProposals();
dom.renderedPromptPreview.textContent = renderPrompt();
updateSimulationModeUi();
