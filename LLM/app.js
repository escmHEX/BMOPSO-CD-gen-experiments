import {
  COMPARATOR_RAW_OBJECTIVE_BOUNDS,
  comparatorBestCostProposalIds,
  comparatorCountByProposal,
  comparatorGlobalNonDominatedFront,
  comparatorHypervolumeArea,
  comparatorIsGloballyNonDominated,
  comparatorMetricExtremes,
  comparatorMetricMetadata,
  comparatorPointCoordinates,
  comparatorRawChartPoints,
} from "./comparator_chart_helpers.mjs";

const EMBEDDING_MODELS = {
  "all-MiniLM-L6-v2": {
    displayName: "all-MiniLM-L6-v2",
    family: "Sentence Transformers / SBERT",
    baseModel: "sentence-transformers/all-MiniLM-L6-v2",
    mtebAverage: "56,26",
    mtebRetrieval: "41,95",
    dimensions: "384",
    maxInput: "256 word pieces",
    approximateCost: "Bajo: modelo PyTorch cargado en backend Python",
    language: "Inglés",
    runtime: {
      task: "SentenceTransformer.encode",
      pooling: "Definido por sentence-transformers",
      normalize: "true",
      batching: "Embedding conjunto de ambos textos",
      execution: "Backend Python con sentence-transformers",
    },
  },
  "thenlper/gte-small": {
    displayName: "gte-small",
    family: "General Text Embeddings",
    baseModel: "thenlper/gte-small",
    mtebAverage: "61,36",
    mtebRetrieval: "49,46",
    dimensions: "384",
    maxInput: "512 tokens",
    approximateCost: "Bajo: modelo PyTorch cargado en backend Python",
    language: "Inglés",
    runtime: {
      task: "SentenceTransformer.encode",
      pooling: "Definido por sentence-transformers",
      normalize: "true",
      batching: "Embedding conjunto de ambos textos",
      execution: "Backend Python con sentence-transformers",
    },
  },
  "Xenova/all-MiniLM-L6-v2": {
    displayName: "all-MiniLM-L6-v2",
    family: "Sentence Transformers / SBERT",
    baseModel: "sentence-transformers/all-MiniLM-L6-v2",
    mtebAverage: "56,26",
    mtebRetrieval: "41,95",
    dimensions: "384",
    maxInput: "256 word pieces",
    approximateCost: "Alias heredado: se redirige al backend Python compartido",
    language: "Inglés",
    runtime: {
      task: "SentenceTransformer.encode",
      pooling: "Definido por sentence-transformers",
      normalize: "true",
      batching: "Embedding conjunto en backend",
      execution: "Backend Python con sentence-transformers",
    },
  },
  "Xenova/gte-small": {
    displayName: "gte-small",
    family: "General Text Embeddings",
    baseModel: "thenlper/gte-small",
    mtebAverage: "61,36",
    mtebRetrieval: "49,46",
    dimensions: "384",
    maxInput: "512 tokens",
    approximateCost: "Alias heredado: se redirige al backend Python compartido",
    language: "Inglés",
    runtime: {
      task: "SentenceTransformer.encode",
      pooling: "Definido por sentence-transformers",
      normalize: "true",
      batching: "Embedding conjunto en backend",
      execution: "Backend Python con sentence-transformers",
    },
  },
};

const PIPELINE_CONFIG = {
  task: "SentenceTransformer.encode",
  pooling: "Definido por sentence-transformers",
  normalize: "true",
  batching: "Embedding conjunto en backend",
  execution: "Backend Python con sentence-transformers",
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

const TURBULENCE_LLM_DEFAULT_MODEL = "Qwen3.5-2B";
const TURBULENCE_PPDB_DEFAULT_SOURCE_PATH = "data/external/ppdb/ppdb-2.0-s-all";
const TURBULENCE_PPDB_DEFAULT_INDEX_PATH = "data/turbulence/ppdb_index.json";
const OPERATOR_DEFAULT_TEMPERATURES = {
  influence: "0.45",
  turbulence: "0.35",
};
const OPERATOR_DEFAULT_MODELS = {
  influence: "meta-llama-3.1-8b-instruct",
  turbulence: TURBULENCE_LLM_DEFAULT_MODEL,
};

const TURBULENCE_SYSTEM_PROMPT = `You are a prompt-component rewriting module.

Task:
Generate very close rewrites of one semantic prompt component.
The rewrite must preserve the same meaning and the same component type.

General rules:
- Return only rewritten component candidates.
- Each candidate must have 2-8 words.
- Do not copy the current component exactly.
- Do not change the main meaning.
- Keep most words from the current component in the same order.
- Replace only 1 word if possible, maximum 2 words.
- Do not use context to add new information.
- Candidates must be distinct from each other.
- Do not explain.
- Do not use quotes.
- Do not add titles, labels, comments, or extra text.
- Return exactly the number of candidates requested by the user.
- Use numbered lines with this format:
1) rewritten candidate
2) rewritten candidate
3) rewritten candidate
...`;

const TURBULENCE_USER_PROMPT_TEMPLATE = `TURBULENCE_USER_PROMPT = """
Number of candidates: {num_candidates}

Component type: {component_name}
Component definition: {component_definition}
Current component value: {current_component}

Context:
{other_components}
"""`;

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

let stopRequested = false;
let solutionEvalStopRequested = false;
let latestEmbeddingVectors = null;
let latestCheckerRows = [];
let latestSolutionRows = [];
let psoDatabase = null;
let comparatorPollTimer = null;
let currentComparatorRunId = null;
let comparatorPollFailureCount = 0;
let latestComparatorRun = null;
let comparatorProposals = [];
let comparatorInstances = [];
let comparatorInstancesInitialized = false;
let comparatorInstanceModalState = null;
let comparatorCharts = [];
let comparatorChartSignature = "";
let comparatorChoiceInstances = [];
let comparatorChartFilterIds = new Set();
let comparatorChartFilterSignature = "";
let comparatorLogRunId = null;
let comparatorLogOffset = 0;
let comparatorLogLines = [];
let comparatorLogLoading = false;
let comparatorLogComplete = false;
let comparatorLogTerminalRefreshDone = false;
let comparatorLogLoadToken = 0;
let initialPopulationPollTimer = null;
let currentInitialPopulationRunId = null;
let initialComparisonPollTimer = null;
let currentInitialComparisonRunId = null;
let turbulenceComparisonPollTimer = null;
let currentTurbulenceComparisonRunId = null;
let turbulenceMetricPopover = null;
let referenceTextLibrary = [];
let lmStudioModelOptions = [];

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
  systemPrompt: document.querySelector("#systemPrompt"),
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
  initialReferencePreset: document.querySelector("#initialReferencePreset"),
  initialReferenceSaveLabel: document.querySelector("#initialReferenceSaveLabel"),
  saveInitialReferenceButton: document.querySelector("#saveInitialReferenceButton"),
  initialReferenceLibraryStatus: document.querySelector("#initialReferenceLibraryStatus"),
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
  initialComparisonReferencePreset: document.querySelector("#initialComparisonReferencePreset"),
  initialComparisonReferenceSaveLabel: document.querySelector("#initialComparisonReferenceSaveLabel"),
  saveInitialComparisonReferenceButton: document.querySelector("#saveInitialComparisonReferenceButton"),
  initialComparisonReferenceLibraryStatus: document.querySelector("#initialComparisonReferenceLibraryStatus"),
  initialComparisonReferenceText: document.querySelector("#initialComparisonReferenceText"),
  initialComparisonN: document.querySelector("#initialComparisonN"),
  initialComparisonTopK: document.querySelector("#initialComparisonTopK"),
  initialComparisonSeed: document.querySelector("#initialComparisonSeed"),
  initialComparisonRepetitions: document.querySelector("#initialComparisonRepetitions"),
  initialComparisonEmbeddingModel: document.querySelector("#initialComparisonEmbeddingModel"),
  compareHybridStrategy: document.querySelector("#compareHybridStrategy"),
  compareEvolmdStrategy: document.querySelector("#compareEvolmdStrategy"),
  compareEvolmdMoStrategy: document.querySelector("#compareEvolmdMoStrategy"),
  initialComparisonOllamaModel: document.querySelector("#initialComparisonOllamaModel"),
  initialComparisonBertModel: document.querySelector("#initialComparisonBertModel"),
  initialComparisonBaselineTimeout: document.querySelector("#initialComparisonBaselineTimeout"),
  initialComparisonTempPrompts: document.querySelector("#initialComparisonTempPrompts"),
  initialComparisonTempKeywords: document.querySelector("#initialComparisonTempKeywords"),
  initialComparisonTempGeneration: document.querySelector("#initialComparisonTempGeneration"),
  initialComparisonLmApiMode: document.querySelector("#initialComparisonLmApiMode"),
  initialComparisonLmStudioBase: document.querySelector("#initialComparisonLmStudioBase"),
  loadInitialComparisonModelsButton: document.querySelector("#loadInitialComparisonModelsButton"),
  initialComparisonGenerationParallelism: document.querySelector("#initialComparisonGenerationParallelism"),
  initialComparisonHybridTimeout: document.querySelector("#initialComparisonHybridTimeout"),
  initialComparisonDomain: document.querySelector("#initialComparisonDomain"),
  initialComparisonRolesMaxWords: document.querySelector("#initialComparisonRolesMaxWords"),
  initialComparisonTopicsMaxWords: document.querySelector("#initialComparisonTopicsMaxWords"),
  initialComparisonActionsMaxWords: document.querySelector("#initialComparisonActionsMaxWords"),
  initialComparisonGeneratedMinWords: document.querySelector("#initialComparisonGeneratedMinWords"),
  initialComparisonGeneratedMaxWords: document.querySelector("#initialComparisonGeneratedMaxWords"),
  initialComparisonPromptTemplate: document.querySelector("#initialComparisonPromptTemplate"),
  initialComparisonHybridStageConfigs: document.querySelector("#initialComparisonHybridStageConfigs"),
  runInitialComparisonButton: document.querySelector("#runInitialComparisonButton"),
  cancelInitialComparisonButton: document.querySelector("#cancelInitialComparisonButton"),
  clearInitialComparisonButton: document.querySelector("#clearInitialComparisonButton"),
  initialComparisonConnectionDot: document.querySelector("#initialComparisonConnectionDot"),
  initialComparisonConnectionText: document.querySelector("#initialComparisonConnectionText"),
  initialComparisonRunStatus: document.querySelector("#initialComparisonRunStatus"),
  initialComparisonProgressPercent: document.querySelector("#initialComparisonProgressPercent"),
  initialComparisonProgressSummary: document.querySelector("#initialComparisonProgressSummary"),
  initialComparisonLlmCalls: document.querySelector("#initialComparisonLlmCalls"),
  initialComparisonLlmCallsDetail: document.querySelector("#initialComparisonLlmCallsDetail"),
  initialComparisonWallClock: document.querySelector("#initialComparisonWallClock"),
  initialComparisonLlmTime: document.querySelector("#initialComparisonLlmTime"),
  initialComparisonEmbeddingTime: document.querySelector("#initialComparisonEmbeddingTime"),
  initialComparisonCompletedStrategies: document.querySelector("#initialComparisonCompletedStrategies"),
  initialComparisonRunId: document.querySelector("#initialComparisonRunId"),
  initialComparisonStatusTone: document.querySelector("#initialComparisonStatusTone"),
  initialComparisonStatusTitle: document.querySelector("#initialComparisonStatusTitle"),
  initialComparisonStatusDetail: document.querySelector("#initialComparisonStatusDetail"),
  initialComparisonProgressDetail: document.querySelector("#initialComparisonProgressDetail"),
  initialComparisonProgressBar: document.querySelector("#initialComparisonProgressBar"),
  initialComparisonProgressDetails: document.querySelector("#initialComparisonProgressDetails"),
  initialComparisonMetricsBody: document.querySelector("#initialComparisonMetricsBody"),
  initialComparisonStrategyDetails: document.querySelector("#initialComparisonStrategyDetails"),
  initialComparisonLogOutput: document.querySelector("#initialComparisonLogOutput"),
  initialComparisonIntegrationDetails: document.querySelector("#initialComparisonIntegrationDetails"),
  turbulenceStrategyLlm: document.querySelector("#turbulenceStrategyLlm"),
  turbulenceStrategyWordnet: document.querySelector("#turbulenceStrategyWordnet"),
  turbulenceStrategyDistilbert: document.querySelector("#turbulenceStrategyDistilbert"),
  turbulenceIndividualCount: document.querySelector("#turbulenceIndividualCount"),
  turbulenceSeed: document.querySelector("#turbulenceSeed"),
  turbulenceRepetitions: document.querySelector("#turbulenceRepetitions"),
  turbulenceChangeThreshold: document.querySelector("#turbulenceChangeThreshold"),
  turbulenceCandidateCount: document.querySelector("#turbulenceCandidateCount"),
  turbulenceComparisonMinSimilarity: document.querySelector("#turbulenceComparisonMinSimilarity"),
  turbulenceComparisonMaxSimilarity: document.querySelector("#turbulenceComparisonMaxSimilarity"),
  turbulenceMinWords: document.querySelector("#turbulenceMinWords"),
  turbulenceMaxWords: document.querySelector("#turbulenceMaxWords"),
  turbulenceEmbeddingModel: document.querySelector("#turbulenceEmbeddingModel"),
  turbulenceDistilbertModel: document.querySelector("#turbulenceDistilbertModel"),
  turbulenceFinalFidelityMin: document.querySelector("#turbulenceFinalFidelityMin"),
  turbulenceFinalFidelityMax: document.querySelector("#turbulenceFinalFidelityMax"),
  turbulenceReferenceText: document.querySelector("#turbulenceReferenceText"),
  turbulenceUsePpdb: document.querySelector("#turbulenceUsePpdb"),
  turbulencePpdbSourcePath: document.querySelector("#turbulencePpdbSourcePath"),
  turbulencePpdbIndexPath: document.querySelector("#turbulencePpdbIndexPath"),
  refreshTurbulencePpdbButton: document.querySelector("#refreshTurbulencePpdbButton"),
  prepareTurbulencePpdbButton: document.querySelector("#prepareTurbulencePpdbButton"),
  turbulencePpdbSetupStatus: document.querySelector("#turbulencePpdbSetupStatus"),
  turbulenceLmBaseUrl: document.querySelector("#turbulenceLmBaseUrl"),
  turbulenceLmApiMode: document.querySelector("#turbulenceLmApiMode"),
  turbulenceLlmModel: document.querySelector("#turbulenceLlmModel"),
  turbulenceOperatorParallelism: document.querySelector("#turbulenceOperatorParallelism"),
  turbulenceGenerationParallelism: document.querySelector("#turbulenceGenerationParallelism"),
  turbulenceLlmTemperature: document.querySelector("#turbulenceLlmTemperature"),
  turbulenceLlmTopP: document.querySelector("#turbulenceLlmTopP"),
  turbulenceLlmMaxTokens: document.querySelector("#turbulenceLlmMaxTokens"),
  turbulenceTimeoutSeconds: document.querySelector("#turbulenceTimeoutSeconds"),
  turbulenceLlmPromptTemplate: document.querySelector("#turbulenceLlmPromptTemplate"),
  turbulenceLlmSystemPrompt: document.querySelector("#turbulenceLlmSystemPrompt"),
  turbulenceFinalPromptTemplate: document.querySelector("#turbulenceFinalPromptTemplate"),
  loadTurbulenceModelsButton: document.querySelector("#loadTurbulenceModelsButton"),
  runTurbulenceComparisonButton: document.querySelector("#runTurbulenceComparisonButton"),
  cancelTurbulenceComparisonButton: document.querySelector("#cancelTurbulenceComparisonButton"),
  clearTurbulenceComparisonButton: document.querySelector("#clearTurbulenceComparisonButton"),
  turbulenceComparisonConnectionDot: document.querySelector("#turbulenceComparisonConnectionDot"),
  turbulenceComparisonConnectionText: document.querySelector("#turbulenceComparisonConnectionText"),
  turbulenceRunStatus: document.querySelector("#turbulenceRunStatus"),
  turbulenceRunStatusDetail: document.querySelector("#turbulenceRunStatusDetail"),
  turbulenceProgressPercent: document.querySelector("#turbulenceProgressPercent"),
  turbulenceProgressSummary: document.querySelector("#turbulenceProgressSummary"),
  turbulenceMovementCount: document.querySelector("#turbulenceMovementCount"),
  turbulenceRecommendation: document.querySelector("#turbulenceRecommendation"),
  turbulenceRecommendationDetail: document.querySelector("#turbulenceRecommendationDetail"),
  turbulenceBaselineFidelity: document.querySelector("#turbulenceBaselineFidelity"),
  turbulenceBaselineDiversity: document.querySelector("#turbulenceBaselineDiversity"),
  turbulencePpdbStatus: document.querySelector("#turbulencePpdbStatus"),
  turbulencePpdbDetail: document.querySelector("#turbulencePpdbDetail"),
  turbulenceRunId: document.querySelector("#turbulenceRunId"),
  turbulenceStatusTone: document.querySelector("#turbulenceStatusTone"),
  turbulenceStatusTitle: document.querySelector("#turbulenceStatusTitle"),
  turbulenceStatusDetail: document.querySelector("#turbulenceStatusDetail"),
  turbulenceStrategyMetricsBody: document.querySelector("#turbulenceStrategyMetricsBody"),
  turbulenceMovementRowsBody: document.querySelector("#turbulenceMovementRowsBody"),
  turbulenceLogOutput: document.querySelector("#turbulenceLogOutput"),
  turbulenceCandidateModal: document.querySelector("#turbulenceCandidateModal"),
  turbulenceCandidateModalTitle: document.querySelector("#turbulenceCandidateModalTitle"),
  turbulenceCandidateModalBody: document.querySelector("#turbulenceCandidateModalBody"),
  closeTurbulenceCandidateModalButton: document.querySelector("#closeTurbulenceCandidateModalButton"),
  comparatorReferencePreset: document.querySelector("#comparatorReferencePreset"),
  comparatorReferenceSaveLabel: document.querySelector("#comparatorReferenceSaveLabel"),
  saveComparatorReferenceButton: document.querySelector("#saveComparatorReferenceButton"),
  comparatorReferenceLibraryStatus: document.querySelector("#comparatorReferenceLibraryStatus"),
  comparatorReferenceText: document.querySelector("#comparatorReferenceText"),
  comparatorModel: document.querySelector("#comparatorModel"),
  comparatorTopK: document.querySelector("#comparatorTopK"),
  comparatorSeed: document.querySelector("#comparatorSeed"),
  comparatorRepetitions: document.querySelector("#comparatorRepetitions"),
  comparatorExecutionMode: document.querySelector("#comparatorExecutionMode"),
  comparatorExecutionModeNote: document.querySelector("#comparatorExecutionModeNote"),
  comparatorProposalParallelism: document.querySelector("#comparatorProposalParallelism"),
  comparatorTimeoutMinutes: document.querySelector("#comparatorTimeoutMinutes"),
  comparatorUpdateReposBeforeRun: document.querySelector("#comparatorUpdateReposBeforeRun"),
  comparatorProposalSelector: document.querySelector("#comparatorProposalSelector"),
  comparatorProposalConfigPanels: document.querySelector("#comparatorProposalConfigPanels"),
  comparatorInstanceModal: document.querySelector("#comparatorInstanceModal"),
  comparatorInstanceModalTitle: document.querySelector("#comparatorInstanceModalTitle"),
  comparatorInstanceModalSubtitle: document.querySelector("#comparatorInstanceModalSubtitle"),
  comparatorInstanceModalBody: document.querySelector("#comparatorInstanceModalBody"),
  saveComparatorInstanceModalButton: document.querySelector("#saveComparatorInstanceModalButton"),
  cancelComparatorInstanceModalButton: document.querySelector("#cancelComparatorInstanceModalButton"),
  comparatorN: document.querySelector("#comparatorN"),
  comparatorGeneraciones: document.querySelector("#comparatorGeneraciones"),
  runComparatorButton: document.querySelector("#runComparatorButton"),
  cancelComparatorButton: document.querySelector("#cancelComparatorButton"),
  recomputeComparatorMetricsButton: document.querySelector("#recomputeComparatorMetricsButton"),
  clearComparatorButton: document.querySelector("#clearComparatorButton"),
  comparatorResumeRunId: document.querySelector("#comparatorResumeRunId"),
  resumeComparatorButton: document.querySelector("#resumeComparatorButton"),
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
  comparatorChartProposalFilters: document.querySelector("#comparatorChartProposalFilters"),
  comparatorParetoCharts: document.querySelector("#comparatorParetoCharts"),
  comparatorCombinedParetoChart: document.querySelector("#comparatorCombinedParetoChart"),
  comparatorGlobalNonDominatedChart: document.querySelector("#comparatorGlobalNonDominatedChart"),
  comparatorHvChart: document.querySelector("#comparatorHvChart"),
  comparatorNonDominatedChart: document.querySelector("#comparatorNonDominatedChart"),
  comparatorSpreadChart: document.querySelector("#comparatorSpreadChart"),
  comparatorGlobalInertiaChart: document.querySelector("#comparatorGlobalInertiaChart"),
  comparatorGlobalEntropyChart: document.querySelector("#comparatorGlobalEntropyChart"),
  comparatorCostExplanation: document.querySelector("#comparatorCostExplanation"),
  comparatorCostTableHead: document.querySelector("#comparatorCostTableHead"),
  comparatorCostTableBody: document.querySelector("#comparatorCostTableBody"),
  comparatorCostTraceability: document.querySelector("#comparatorCostTraceability"),
  comparatorCostTraceabilityBody: document.querySelector("#comparatorCostTraceabilityBody"),
  comparatorLogOutput: document.querySelector("#comparatorLogOutput"),
  copyComparatorLogButton: document.querySelector("#copyComparatorLogButton"),
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
  turbulence: TURBULENCE_USER_PROMPT_TEMPLATE,
};

const DEFAULT_SYSTEM_PROMPT_TEMPLATES = {
  influence: "",
  turbulence: TURBULENCE_SYSTEM_PROMPT,
};

const COMPARATOR_API = "/api/comparator";
const COMPARATOR_TERMINAL_STATUSES = new Set(["completed", "failed", "cancelled"]);
const COMPARATOR_MAX_TRANSIENT_POLL_FAILURES = 5;
const COMPARATOR_LOG_CHUNK_LIMIT = 5000;
const COMPARATOR_LAST_RUN_ID_STORAGE_KEY = "comparator:lastRunId";
const INITIAL_POPULATION_API = "/api/initial-population";
const INITIAL_POPULATION_TERMINAL_STATUSES = new Set(["completed", "failed", "cancelled"]);
const INITIAL_POPULATION_COMPARISON_API = "/api/initial-population-comparison";
const INITIAL_POPULATION_COMPARISON_TERMINAL_STATUSES = new Set(["completed", "failed", "cancelled"]);
const REFERENCE_TEXTS_API = "/api/reference-texts";
const TURBULENCE_COMPARISON_API = "/api/turbulence-comparison";
const TURBULENCE_COMPARISON_TERMINAL_STATUSES = new Set(["completed", "failed", "cancelled"]);
const SBERT_API = "/api/sbert";
const LM_STUDIO_API = "/api/lm-studio";

const ABBREVIATION_TOOLTIPS = Object.freeze({
  "#": "Numero de fila o identificador ordinal.",
  "api": "Application Programming Interface.",
  "bd pso": "Base de datos de individuos PSO.",
  "distilbert": "Modelo DistilBERT usado por la estrategia correspondiente.",
  "evolmd post-hoc": "Metricas diagnosticas calculadas despues de la ejecucion nativa de EVOLMD.",
  "uniobjetivo post-hoc": "Metricas diagnosticas calculadas despues de ejecutar propuestas uniobjetivo como EVOLMD o MESAP.",
  "f.o": "Funcion objetivo.",
  "f.o.": "Funcion objetivo.",
  "f.o. comun": "Funcion objetivo comun usada para comparar estrategias.",
  "f.o. nativa": "Funcion objetivo nativa reportada por la estrategia.",
  "git": "Sistema de control de versiones Git.",
  "g": "Cantidad de generaciones o iteraciones configuradas.",
  "hv": "Hypervolume.",
  "hv comp.": "Hypervolume comparable en el espacio objetivo normalizado.",
  "hv comp. post-hoc": "Hypervolume comparable calculado post-hoc.",
  "hv por iteracion": "Hypervolume por iteracion.",
  "k": "Cantidad de repeticiones independientes.",
  "llm": "Large Language Model.",
  "llm actual": "Large Language Model actualmente configurado.",
  "llamadas llm": "Llamadas al Large Language Model.",
  "mejor comp.": "Mejor vector comparable normalizado.",
  "mejor f.o.": "Mejor funcion objetivo.",
  "mejor f.o. comun": "Mejor funcion objetivo comun.",
  "metricas web": "Metricas calculadas por el backend web despues de la ejecucion.",
  "mo comparable": "Comparacion multiobjetivo en el espacio objetivo normalizado comun.",
  "mteb average (56)": "Promedio MTEB sobre 56 tareas de evaluacion.",
  "n": "Tamano de poblacion o cantidad de individuos.",
  "n individuos": "Tamano de poblacion o cantidad de individuos.",
  "no dom. post-hoc": "Soluciones no dominadas calculadas post-hoc.",
  "no dominadas": "Soluciones no dominadas.",
  "no dominadas globales": "Soluciones no dominadas al comparar la union de propuestas.",
  "no dominadas nativas": "Soluciones no dominadas en el espacio nativo de la estrategia.",
  "ollama": "Servidor local Ollama usado para ejecutar el modelo LLM.",
  "ppdb": "Paraphrase Database.",
  "prom. llamada": "Tiempo promedio por llamada.",
  "prom. llm / individuo": "Tiempo promedio de LLM por individuo.",
  "pso": "Particle Swarm Optimization.",
  "rank": "Orden de la solucion dentro de la tabla o seleccion.",
  "run": "Identificador de ejecucion.",
  "run id": "Identificador de ejecucion.",
  "sbert": "Sentence-BERT.",
  "sin paralelismo estim.": "Estimacion sin ejecutar estrategias en paralelo.",
  "spread comp.": "Spread comparable en el espacio objetivo normalizado.",
  "spread comp. post-hoc": "Spread comparable calculado post-hoc.",
  "temp. prompts": "Temperatura usada para generar prompts.",
  "tiempo llm": "Tiempo acumulado asociado a llamadas LLM.",
  "tiempo llm cliente": "Suma de latencias cliente de llamadas LLM.",
  "tiempo llm total": "Tiempo total asociado a llamadas LLM.",
  "top p": "Parametro nucleus sampling top-p.",
  "total prop.": "Tiempo total de la propuesta.",
  "vector f.o": "Vector de funcion objetivo.",
  "vector f.o.": "Vector de funcion objetivo.",
  "wordnet": "Base lexical WordNet.",
  "wordnet+ppdb+sbert": "Estrategia que combina WordNet, PPDB y Sentence-BERT.",
});

const ABBREVIATION_TOKEN_TOOLTIPS = Object.freeze([
  { pattern: /\bAPI\b/u, title: "API = Application Programming Interface" },
  { pattern: /\bBD\b/u, title: "BD = Base de datos" },
  { pattern: /\bcomp\./iu, title: "comp. = comparable" },
  { pattern: /\bestim\./iu, title: "estim. = estimado" },
  { pattern: /\bF\.?\s*O\.?\b/u, title: "F.O. = funcion objetivo" },
  { pattern: /\bG\b/u, title: "G = generaciones o iteraciones" },
  { pattern: /\bGit\b/u, title: "Git = sistema de control de versiones" },
  { pattern: /\bHV\b/u, title: "HV = Hypervolume" },
  { pattern: /\bK\b/u, title: "K = repeticiones independientes" },
  { pattern: /\bLLM\b/u, title: "LLM = Large Language Model" },
  { pattern: /\bm(?:a|\u00e1)x\./iu, title: "max. = maximo" },
  { pattern: /\bm(?:i|\u00ed)n\./iu, title: "min. = minimo" },
  { pattern: /\bMO\b/u, title: "MO = multiobjetivo" },
  { pattern: /\bMTEB\b/u, title: "MTEB = Massive Text Embedding Benchmark" },
  { pattern: /\bN\b/u, title: "N = tamano de poblacion o cantidad de individuos" },
  { pattern: /\bpost-hoc\b/iu, title: "post-hoc = calculado despues de la ejecucion nativa" },
  { pattern: /\bPPDB\b/u, title: "PPDB = Paraphrase Database" },
  { pattern: /\bprom\./iu, title: "prom. = promedio" },
  { pattern: /\bprop\./iu, title: "prop. = propuesta" },
  { pattern: /\bPSO\b/u, title: "PSO = Particle Swarm Optimization" },
  { pattern: /\bSBERT\b/u, title: "SBERT = Sentence-BERT" },
]);

const ABBREVIATION_TOOLTIP_SELECTOR = [
  "dt",
  "th",
  ".eyebrow",
  ".panel-title h2",
  ".panel-title span",
  ".stat-card span",
  ".metric-card span",
  "label > span",
  "summary",
  "button.compact",
].join(", ");

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

function abbreviationTooltipFor(text) {
  const rawText = String(text || "");
  const key = rawText
    .replace(/\s+/gu, " ")
    .trim()
    .replace(/:$/u, "")
    .toLocaleLowerCase("es-CL");
  if (!key) return "";
  if (ABBREVIATION_TOOLTIPS[key]) return ABBREVIATION_TOOLTIPS[key];
  const withoutTrailingDot = key.replace(/\.$/u, "");
  if (ABBREVIATION_TOOLTIPS[withoutTrailingDot]) return ABBREVIATION_TOOLTIPS[withoutTrailingDot];
  const tokenTooltips = ABBREVIATION_TOKEN_TOOLTIPS
    .filter(({ pattern }) => pattern.test(rawText))
    .map(({ title }) => title);
  return [...new Set(tokenTooltips)].join("; ");
}

function applyAbbreviationTooltip(element, text = element?.textContent) {
  if (!(element instanceof Element)) return;
  if (element.title) {
    element.classList.add("has-abbreviation-tooltip");
    return;
  }
  const tooltip = abbreviationTooltipFor(text);
  if (!tooltip) return;
  element.title = tooltip;
  if (!element.hasAttribute("aria-label")) {
    element.setAttribute("aria-label", `${String(text || "").trim()}: ${tooltip}`);
  }
  element.classList.add("has-abbreviation-tooltip");
}

function decorateAbbreviationTooltips(root = document) {
  if (!(root instanceof Document || root instanceof Element)) return;
  const elements = [];
  if (root instanceof Element && root.matches(ABBREVIATION_TOOLTIP_SELECTOR)) {
    elements.push(root);
  }
  elements.push(...root.querySelectorAll(ABBREVIATION_TOOLTIP_SELECTOR));
  elements.forEach((element) => applyAbbreviationTooltip(element));
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
      applyAbbreviationTooltip(dt, term);
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
  return "http://127.0.0.1:1234";
}

function lmStudioBaseUrlFromEndpoint(endpoint) {
  const value = normalizeEndpoint(endpoint);
  if (!value || value.startsWith("/lmstudio")) {
    return "";
  }
  try {
    const url = new URL(value);
    url.pathname = url.pathname.replace(/\/(?:api\/v1|v1)$/u, "");
    url.search = "";
    url.hash = "";
    return url.toString().replace(/\/+$/u, "");
  } catch {
    return value.replace(/\/(?:api\/v1|v1)$/u, "");
  }
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
  const runtime = model.runtime || PIPELINE_CONFIG;
  dom.embeddingModelLabel.textContent = model.family;
  renderDefinitionList(dom.embeddingRuntimeDetails, [
    ["Pipeline", runtime.task],
    ["Pooling", runtime.pooling],
    ["Normalización", runtime.normalize],
    ["Cálculo", runtime.batching],
    ["Ejecución", runtime.execution],
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

async function requestEmbeddings(modelId, texts) {
  const response = await fetch(`${SBERT_API}/embeddings`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ model: modelId, texts }),
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.error || `Solicitud SBERT fallida (${response.status}).`);
  }
  return payload.embeddings;
}

async function requestSbertPair(model, textA, textB) {
  const response = await fetch(`${SBERT_API}/pair`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ model, textA, textB }),
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.error || `Solicitud SBERT fallida (${response.status}).`);
  }
  return payload;
}

async function requestLmStudioJson(path, payload, timeoutMs = 120000) {
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(`${LM_STUDIO_API}${path}`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      signal: controller.signal,
      body: JSON.stringify(payload),
    });
    const responsePayload = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(responsePayload.error || `Solicitud LM Studio fallida (${response.status}).`);
    }
    return responsePayload;
  } finally {
    window.clearTimeout(timeout);
  }
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
  setStatus(dom.embeddingStatusTone, dom.embeddingStatusTitle, dom.embeddingStatusDetail, "Calculando", "Preparando embeddings en el backend Python.", "busy");

  try {
    const selectedModel = selectedEmbeddingModel();
    const result = await requestSbertPair(selectedModel, textA, textB);
    latestEmbeddingVectors = result.embeddings;
    dom.embeddingSimilarity.textContent = formatNumber(result.similarity);
    dom.embeddingDistance.textContent = formatNumber(result.distance);
    dom.embeddingDimension.textContent = String(result.dimension);
    dom.embeddingPreviewA.textContent = vectorPreview(latestEmbeddingVectors[0]);
    dom.embeddingPreviewB.textContent = vectorPreview(latestEmbeddingVectors[1]);
    dom.copyEmbeddingAButton.disabled = false;
    dom.copyEmbeddingBButton.disabled = false;
    setStatus(
      dom.embeddingStatusTone,
      dom.embeddingStatusTitle,
      dom.embeddingStatusDetail,
      "Cálculo completado",
      `Similitud calculada con ${EMBEDDING_MODELS[selectedModel].displayName} en ${result.backend}.`,
    );
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

function isDefaultSystemPromptTemplate(template) {
  return Object.values(DEFAULT_SYSTEM_PROMPT_TEMPLATES).some((defaultTemplate) => defaultTemplate.trim() === template.trim());
}

function isDefaultOperatorTemperature(value) {
  return Object.values(OPERATOR_DEFAULT_TEMPERATURES).includes(String(value).trim());
}

function isDefaultOperatorModel(value) {
  return Object.values(OPERATOR_DEFAULT_MODELS).includes(String(value).trim());
}

function selectLoadedModelMatching(preferredModel) {
  const preferred = String(preferredModel || "").toLocaleLowerCase();
  if (!preferred || !dom.llmModelSelect.options.length) return "";
  const exact = Array.from(dom.llmModelSelect.options).find((option) => option.value.toLocaleLowerCase() === preferred);
  if (exact) return exact.value;
  const qwenMatch = preferred.includes("qwen")
    ? Array.from(dom.llmModelSelect.options).find((option) => {
        const value = option.value.toLocaleLowerCase();
        return value.includes("qwen") && (value.includes("2b") || value.includes("3.5"));
      })
    : null;
  return qwenMatch?.value || "";
}

function applyPsoOperatorTemplate() {
  const operator = selectedPsoOperator();
  if (isDefaultPromptTemplate(dom.promptTemplate.value)) {
    dom.promptTemplate.value = DEFAULT_PROMPT_TEMPLATES[operator];
  }
  if (isDefaultSystemPromptTemplate(dom.systemPrompt.value)) {
    dom.systemPrompt.value = DEFAULT_SYSTEM_PROMPT_TEMPLATES[operator];
  }
  if (isDefaultOperatorTemperature(dom.temperature.value)) {
    dom.temperature.value = OPERATOR_DEFAULT_TEMPERATURES[operator];
  }
  if (isDefaultOperatorModel(dom.llmModelManual.value)) {
    const preferredModel = OPERATOR_DEFAULT_MODELS[operator];
    const loadedModel = selectLoadedModelMatching(preferredModel);
    if (loadedModel) {
      dom.llmModelSelect.value = loadedModel;
      dom.llmModelManual.value = loadedModel;
    } else {
      dom.llmModelManual.value = preferredModel;
    }
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
  dom.systemPrompt.disabled = isRunning;
  dom.promptTemplate.disabled = isRunning;
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

function hasPromptMarkup(candidate) {
  return /```|\*\*|<\s*\/?\s*[A-Za-z][^>]*>|^\s*(?:system|user|assistant)\s*:|\b(?:based on|here'?s|breakdown|possible components?|current component value|potential next values?)\b/iu
    .test(candidate.trim());
}

function parseCandidates(rawOutput, expectedCount) {
  const cleaned = rawOutput
    .replace(/```[A-Za-z]*\n?/g, "")
    .replace(/```/g, "")
    .trim();
  const candidates = [];

  cleaned.split(/\r?\n/).forEach((line) => {
    if (!/^\s*\d+\s*[\).\]:-]\s*/u.test(line)) {
      return;
    }
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

  if (hasPromptMarkup(candidate)) {
    return "prompt_marker";
  }
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
  const started = performance.now();
  const payload = await requestLmStudioJson(
    "/chat",
    {
      baseUrl: lmStudioBaseUrlFromEndpoint(config.endpoint),
      apiMode: config.apiMode,
      model: config.model,
      systemPrompt: config.systemPrompt || "",
      userPrompt: prompt,
      temperature: config.temperature,
      topP: config.topP,
      maxTokens: config.maxTokens,
      timeoutSeconds: Math.max(1, Math.ceil(config.timeoutMs / 1000)),
    },
    config.timeoutMs + 5000,
  );

  return {
    raw: payload.text || "",
    elapsedMs: Number.isFinite(Number(payload.elapsedSeconds)) ? Number(payload.elapsedSeconds) * 1000 : performance.now() - started,
    usage: {
      prompt_tokens: payload.promptTokens ?? 0,
      completion_tokens: payload.completionTokens ?? 0,
      total_tokens: payload.totalTokens ?? 0,
    },
    runNumber,
  };
}

async function fetchLmStudioModels() {
  const endpoint = normalizeEndpoint(dom.lmEndpoint.value);
  dom.lmEndpoint.value = endpoint;
  dom.loadModelsButton.disabled = true;
  dom.checkerConnectionDot.classList.add("is-busy");
  dom.checkerConnectionDot.classList.remove("is-error");
  dom.checkerConnectionText.textContent = "Consultando LM Studio";

  try {
    const payload = await requestLmStudioJson("/models", {
      baseUrl: lmStudioBaseUrlFromEndpoint(endpoint),
      apiMode: dom.lmApiMode.value,
      timeoutSeconds: 30,
    }, 35000);
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
      const preferredModel = OPERATOR_DEFAULT_MODELS[selectedPsoOperator()];
      const preferredLoaded = selectLoadedModelMatching(preferredModel);
      const loaded = preferredLoaded
        ? models.find((model) => model.id === preferredLoaded)
        : models.find((model) => model.loaded) || models[0];
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
    const payload = await requestLmStudioJson("/models", {
      baseUrl: lmStudioBaseUrlFromEndpoint(endpoint),
      apiMode: dom.solutionLmApiMode.value,
      timeoutSeconds: 30,
    }, 35000);
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
  if (
    Array.isArray(payload.models)
    && payload.models.every((model) => model && typeof model.id === "string")
  ) {
    return payload.models;
  }

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
    systemPrompt: getTemplateBody(dom.systemPrompt.value),
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

function referenceTextControls() {
  return [
    {
      select: dom.initialReferencePreset,
      textarea: dom.initialReferenceText,
      labelInput: dom.initialReferenceSaveLabel,
      saveButton: dom.saveInitialReferenceButton,
      status: dom.initialReferenceLibraryStatus,
    },
    {
      select: dom.initialComparisonReferencePreset,
      textarea: dom.initialComparisonReferenceText,
      labelInput: dom.initialComparisonReferenceSaveLabel,
      saveButton: dom.saveInitialComparisonReferenceButton,
      status: dom.initialComparisonReferenceLibraryStatus,
    },
    {
      select: dom.comparatorReferencePreset,
      textarea: dom.comparatorReferenceText,
      labelInput: dom.comparatorReferenceSaveLabel,
      saveButton: dom.saveComparatorReferenceButton,
      status: dom.comparatorReferenceLibraryStatus,
    },
  ].filter((control) => control.select && control.textarea && control.labelInput && control.saveButton && control.status);
}

function referenceTextOptionLabel(item) {
  if (item.label) return item.label;
  const ref = item.ref ? ` - Ref. ${item.ref}` : "";
  return `${item.paper || "Texto guardado"}${ref}`;
}

function referenceTextSourceLabel(item) {
  const pieces = [item.paper, item.ref ? `Ref. ${item.ref}` : "", item.source].filter(Boolean);
  return pieces.join(" | ") || "Texto guardado";
}

function selectedReferenceTextItem(control) {
  return referenceTextLibrary.find((item) => item.id === control.select.value) || null;
}

function matchReferenceTextItem(text) {
  const normalized = text.trim();
  return referenceTextLibrary.find((item) => item.text.trim() === normalized) || null;
}

function setReferenceControlStatus(control, message, isError = false) {
  control.status.textContent = message;
  control.status.classList.toggle("invalid", isError);
  control.status.classList.toggle("valid", !isError && Boolean(message));
}

function populateReferenceTextControls() {
  referenceTextControls().forEach((control) => {
    const currentText = control.textarea.value.trim();
    control.select.replaceChildren(
      new Option("Seleccionar texto guardado", ""),
      ...referenceTextLibrary.map((item) => {
        const option = new Option(referenceTextOptionLabel(item), item.id);
        option.title = referenceTextSourceLabel(item);
        return option;
      }),
    );
    const match = matchReferenceTextItem(currentText);
    control.select.value = match ? match.id : "";
    setReferenceControlStatus(control, `${referenceTextLibrary.length} texto(s) disponibles.`);
  });
}

async function requestReferenceTextsJson(options = {}) {
  const response = await fetch(REFERENCE_TEXTS_API, {
    headers: {
      "Content-Type": "application/json",
      ...(options.headers || {}),
    },
    ...options,
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    const error = new Error(payload.error || `HTTP ${response.status}`);
    error.code = payload.code || "";
    throw error;
  }
  return payload;
}

async function loadReferenceTextLibrary() {
  try {
    const payload = await requestReferenceTextsJson();
    referenceTextLibrary = Array.isArray(payload.items) ? payload.items : [];
    populateReferenceTextControls();
  } catch (error) {
    referenceTextControls().forEach((control) => {
      control.select.replaceChildren(new Option("No se pudo cargar la BD", ""));
      setReferenceControlStatus(control, error.message, true);
    });
  }
}

function applyReferenceTextSelection(control) {
  const item = selectedReferenceTextItem(control);
  if (!item) return;
  control.textarea.value = item.text;
  setReferenceControlStatus(control, referenceTextSourceLabel(item));
  if (control.textarea === dom.initialComparisonReferenceText) {
    resetInitialComparisonUi();
  } else if (control.textarea === dom.comparatorReferenceText) {
    resetComparatorUi();
  }
}

function syncReferenceTextSelection(control) {
  const match = matchReferenceTextItem(control.textarea.value);
  control.select.value = match ? match.id : "";
  setReferenceControlStatus(
    control,
    match ? referenceTextSourceLabel(match) : `${referenceTextLibrary.length} texto(s) disponibles.`,
  );
}

async function saveReferenceTextFromControl(control) {
  const text = control.textarea.value.trim();
  if (!text) {
    setReferenceControlStatus(control, "No hay texto de referencia para guardar.", true);
    return;
  }
  control.saveButton.disabled = true;
  setReferenceControlStatus(control, "Guardando texto de referencia...");
  try {
    const payload = await requestReferenceTextsJson({
      method: "POST",
      body: JSON.stringify({
        label: control.labelInput.value.trim(),
        paper: "Usuario",
        source: "Guardado desde la web",
        text,
      }),
    });
    referenceTextLibrary = Array.isArray(payload.items) ? payload.items : referenceTextLibrary;
    populateReferenceTextControls();
    if (payload.item?.id) {
      control.select.value = payload.item.id;
      setReferenceControlStatus(control, `Guardado: ${referenceTextOptionLabel(payload.item)}`);
    }
    control.labelInput.value = "";
  } catch (error) {
    setReferenceControlStatus(control, error.message, true);
  } finally {
    control.saveButton.disabled = false;
  }
}

function setupReferenceTextLibraryControls() {
  referenceTextControls().forEach((control) => {
    control.select.addEventListener("change", () => applyReferenceTextSelection(control));
    control.textarea.addEventListener("input", () => syncReferenceTextSelection(control));
    control.saveButton.addEventListener("click", () => saveReferenceTextFromControl(control));
  });
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
  if (value === null || value === undefined || value === "") {
    return "--";
  }
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
    strategyId: "hybrid-semantic-v7",
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

function lmStudioModelSelects() {
  return Array.from(document.querySelectorAll("[data-lm-studio-model-select]"));
}

function updateLmStudioModelSelectOptions(models) {
  lmStudioModelOptions = [...new Set((models || []).map((model) => String(model).trim()).filter(Boolean))].sort();
  lmStudioModelSelects().forEach((select) => {
    const current = select.value;
    const preferred = select.dataset.preferredModel || "";
    const shouldKeepPreferred = preferred && (!current || current === OPERATOR_DEFAULT_MODELS.influence);
    const desired = shouldKeepPreferred ? preferred : current;
    if (!lmStudioModelOptions.length) {
      if (desired) {
        select.replaceChildren(new Option(desired, desired));
      }
      return;
    }
    const options = lmStudioModelOptions.map((model) => new Option(model, model));
    if (preferred && desired && !lmStudioModelOptions.includes(desired)) {
      options.unshift(new Option(`${desired} (no listado en LM Studio)`, desired));
    }
    select.replaceChildren(...options);
    select.value = desired && Array.from(select.options).some((option) => option.value === desired)
      ? desired
      : lmStudioModelOptions[0];
  });
}

async function loadLmStudioModelsForInitialConfig({ baseUrl, apiMode, button, statusElements }) {
  const params = new URLSearchParams({ baseUrl, apiMode });
  if (button) button.disabled = true;
  try {
    const payload = await requestInitialPopulationJson(`/lm-studio/models?${params.toString()}`);
    const models = payload.models || [];
    dom.initialPopulationLmModels.replaceChildren(
      ...models.map((model) => {
        const option = document.createElement("option");
        option.value = model;
        return option;
      }),
    );
    updateLmStudioModelSelectOptions(models);
    if (statusElements) {
      statusElements.connectionText.textContent = `${models.length} modelo(s)`;
      statusElements.connectionDot.classList.remove("is-error", "is-busy");
      setStatus(statusElements.tone, statusElements.title, statusElements.detail, "Modelos cargados", `${models.length} modelo(s) disponibles desde LM Studio.`);
    }
  } catch (error) {
    if (statusElements) {
      statusElements.connectionText.textContent = "Error modelos";
      statusElements.connectionDot.classList.add("is-error");
      setStatus(statusElements.tone, statusElements.title, statusElements.detail, "No se pudieron cargar modelos", error.message, "error");
    }
  } finally {
    if (button) button.disabled = false;
  }
}

async function loadInitialPopulationModels() {
  await loadLmStudioModelsForInitialConfig({
    baseUrl: dom.initialLmStudioBase.value.trim(),
    apiMode: dom.initialLmApiMode.value,
    button: dom.loadInitialModelsButton,
    statusElements: {
      connectionText: dom.initialPopulationConnectionText,
      connectionDot: dom.initialPopulationConnectionDot,
      tone: dom.initialStatusTone,
      title: dom.initialStatusTitle,
      detail: dom.initialStatusDetail,
    },
  });
}

async function loadInitialComparisonModels() {
  await loadLmStudioModelsForInitialConfig({
    baseUrl: dom.initialComparisonLmStudioBase.value.trim(),
    apiMode: dom.initialComparisonLmApiMode.value,
    button: dom.loadInitialComparisonModelsButton,
    statusElements: {
      connectionText: dom.initialComparisonConnectionText,
      connectionDot: dom.initialComparisonConnectionDot,
      tone: dom.initialComparisonStatusTone,
      title: dom.initialComparisonStatusTitle,
      detail: dom.initialComparisonStatusDetail,
    },
  });
}

async function loadTurbulenceModels() {
  await loadLmStudioModelsForInitialConfig({
    baseUrl: dom.turbulenceLmBaseUrl.value.trim(),
    apiMode: dom.turbulenceLmApiMode.value,
    button: dom.loadTurbulenceModelsButton,
    statusElements: {
      connectionText: dom.turbulenceComparisonConnectionText,
      connectionDot: dom.turbulenceComparisonConnectionDot,
      tone: dom.turbulenceStatusTone,
      title: dom.turbulenceStatusTitle,
      detail: dom.turbulenceStatusDetail,
    },
  });
}

function setInitialPopulationRunning(isRunning, cancelRequested = false) {
  dom.runInitialPopulationButton.disabled = isRunning;
  dom.cancelInitialPopulationButton.disabled = !isRunning || !currentInitialPopulationRunId || cancelRequested;
  dom.cancelInitialPopulationButton.textContent = cancelRequested ? "Cancelando..." : "Cancelar";
  dom.clearInitialPopulationButton.disabled = isRunning;
  dom.loadInitialModelsButton.disabled = isRunning;
  [
    dom.initialLmApiMode,
    dom.initialLmStudioBase,
    dom.initialN,
    dom.initialTopK,
    dom.initialGenerationParallelism,
    dom.initialTimeoutSeconds,
    dom.initialSeed,
    dom.initialEmbeddingModel,
    dom.initialReferencePreset,
    dom.initialReferenceSaveLabel,
    dom.saveInitialReferenceButton,
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
    "Ejecuta la estrategia hibrida v7 desde Python usando LM Studio local.",
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

function initialComparisonStageLabel(stageName) {
  const labels = {
    anchors: "Anclas",
    roles: "Pool roles",
    topics: "Pool topicos",
    actions: "Pool acciones",
    expansion: "Expansion",
    generation: "Generacion",
  };
  return labels[stageName] || stageName;
}

function initialComparisonStageDefaults() {
  const defaults = {};
  dom.initialStageConfigs.forEach((panel) => {
    const stageName = panel.dataset.initialStage;
    if (!stageName) return;
    defaults[stageName] = {};
    panel.querySelectorAll("[data-stage-field]").forEach((field) => {
      defaults[stageName][field.dataset.stageField] = field.value;
    });
  });
  return defaults;
}

function buildInitialComparisonStageEditors() {
  if (!dom.initialComparisonHybridStageConfigs || dom.initialComparisonHybridStageConfigs.children.length) {
    return;
  }
  const defaults = initialComparisonStageDefaults();
  dom.initialComparisonHybridStageConfigs.replaceChildren(
    ...Object.entries(defaults).map(([stageName, stage]) => {
      const article = document.createElement("article");
      article.className = "panel comparison-stage-config";
      article.dataset.comparisonStage = stageName;
      article.innerHTML = `
        <div class="panel-title">
          <h2>${escapeHtml(initialComparisonStageLabel(stageName))}</h2>
        </div>
        <div class="form-grid">
          <label>
            <span>Modelo</span>
            <select data-comparison-stage-field="model" data-lm-studio-model-select>
              <option value="${escapeHtml(stage.model || "")}">${escapeHtml(stage.model || "")}</option>
            </select>
          </label>
          <label>
            <span>Temperatura</span>
            <input data-comparison-stage-field="temperature" type="number" min="0" max="2" step="0.05" value="${escapeHtml(stage.temperature || "0.7")}">
          </label>
          <label>
            <span>Top p</span>
            <input data-comparison-stage-field="topP" type="number" min="0" max="1" step="0.01" value="${escapeHtml(stage.topP || "0.95")}">
          </label>
          <label>
            <span>Top k</span>
            <input data-comparison-stage-field="topK" type="number" min="0" max="500" value="${escapeHtml(stage.topK || "40")}">
          </label>
          <label>
            <span>Max. tokens</span>
            <input data-comparison-stage-field="maxTokens" type="number" min="8" max="4096" value="${escapeHtml(stage.maxTokens || "500")}">
          </label>
        </div>
        <label>
          <span>System prompt</span>
          <textarea data-comparison-stage-field="systemPrompt" class="code-textarea" rows="10">${escapeHtml(stage.systemPrompt || "")}</textarea>
        </label>
        <label>
          <span>User prompt</span>
          <textarea data-comparison-stage-field="userPrompt" class="code-textarea" rows="6">${escapeHtml(stage.userPrompt || "")}</textarea>
        </label>
      `;
      return article;
    }),
  );
  updateLmStudioModelSelectOptions(lmStudioModelOptions);
}

function readInitialComparisonStageConfigs() {
  const stages = {};
  document.querySelectorAll(".comparison-stage-config").forEach((panel) => {
    const stageName = panel.dataset.comparisonStage;
    const stage = {};
    panel.querySelectorAll("[data-comparison-stage-field]").forEach((field) => {
      const key = field.dataset.comparisonStageField;
      if (["temperature", "topP"].includes(key)) {
        stage[key] = readClampedNumber(field, `comparacion.${stageName}.${key}`, 0, key === "topP" ? 1 : 2);
      } else if (["topK", "maxTokens"].includes(key)) {
        stage[key] = Math.floor(readClampedNumber(field, `comparacion.${stageName}.${key}`, key === "topK" ? 0 : 8, key === "topK" ? 500 : 4096));
      } else {
        const value = field.value.trim();
        if (!value) {
          throw new Error(`comparacion.${stageName}.${key} no puede estar vacio.`);
        }
        stage[key] = value;
      }
    });
    stages[stageName] = stage;
  });
  return stages;
}

function selectedInitialComparisonStrategies() {
  return [
    dom.compareHybridStrategy,
    dom.compareEvolmdStrategy,
    dom.compareEvolmdMoStrategy,
  ].filter((checkbox) => checkbox.checked).map((checkbox) => checkbox.value);
}

function readInitialComparisonConfig() {
  const referenceText = dom.initialComparisonReferenceText.value.trim();
  const selectedStrategies = selectedInitialComparisonStrategies();
  const generatedMinWords = Math.floor(readClampedNumber(dom.initialComparisonGeneratedMinWords, "Min. palabras texto", 1, 500));
  const generatedMaxWords = Math.floor(readClampedNumber(dom.initialComparisonGeneratedMaxWords, "Max. palabras texto", 1, 500));
  const promptTemplate = dom.initialComparisonPromptTemplate.value.trim();
  const domain = dom.initialComparisonDomain.value.trim();
  const baseUrl = dom.initialComparisonLmStudioBase.value.trim();

  if (!referenceText) throw new Error("Define el texto de referencia.");
  if (!selectedStrategies.length) throw new Error("Selecciona al menos una estrategia.");
  if (!promptTemplate) throw new Error("Define la plantilla deterministica final.");
  if (!domain) throw new Error("Define el dominio general.");
  if (!baseUrl) throw new Error("Define la Base URL de LM Studio.");
  if (generatedMinWords > generatedMaxWords) {
    throw new Error("Min. palabras texto no puede ser mayor que Max. palabras texto.");
  }

  const n = Math.floor(readClampedNumber(dom.initialComparisonN, "N individuos", 1, 500));
  const topK = Math.floor(readClampedNumber(dom.initialComparisonTopK, "Top K tabla", 1, 500));
  const seed = Math.floor(readClampedNumber(dom.initialComparisonSeed, "Semilla", 0, 2147483647));
  const repetitionsK = Math.floor(readClampedNumber(dom.initialComparisonRepetitions, "K repeticiones", 1, 30));
  return {
    referenceText,
    selectedStrategies,
    n,
    topK,
    seed,
    repetitionsK,
    commonEmbeddingModel: dom.initialComparisonEmbeddingModel.value,
    hybrid: {
      strategyId: "hybrid-semantic-v7",
      referenceText,
      domain,
      n,
      topK,
      timeoutSeconds: Math.floor(readClampedNumber(dom.initialComparisonHybridTimeout, "Timeout estrategia", 5, 3600)),
      generationParallelism: Math.floor(readClampedNumber(dom.initialComparisonGenerationParallelism, "Paralelismo generacion", 1, 16)),
      seed,
      embeddingModel: dom.initialComparisonEmbeddingModel.value,
      lmStudio: {
        baseUrl,
        apiMode: dom.initialComparisonLmApiMode.value,
      },
      stages: readInitialComparisonStageConfigs(),
      promptTemplate,
      validation: {
        rolesMaxWords: Math.floor(readClampedNumber(dom.initialComparisonRolesMaxWords, "Max. palabras roles", 1, 20)),
        topicsMaxWords: Math.floor(readClampedNumber(dom.initialComparisonTopicsMaxWords, "Max. palabras topicos", 1, 30)),
        actionsMaxWords: Math.floor(readClampedNumber(dom.initialComparisonActionsMaxWords, "Max. palabras acciones", 1, 20)),
        generatedMinWords,
        generatedMaxWords,
      },
    },
    baselines: {
      model: dom.initialComparisonOllamaModel.value.trim(),
      bertModel: dom.initialComparisonBertModel.value.trim(),
      timeoutMinutes: Math.floor(readClampedNumber(dom.initialComparisonBaselineTimeout, "Timeout baseline", 1, 1440)),
      tempPrompts: readClampedNumber(dom.initialComparisonTempPrompts, "Temp. prompts", 0, 2),
      tempKeywords: readClampedNumber(dom.initialComparisonTempKeywords, "Temp. keywords", 0, 2),
      tempGeneration: readClampedNumber(dom.initialComparisonTempGeneration, "Temp. generacion", 0, 2),
    },
  };
}

async function requestInitialComparisonJson(path, options = {}) {
  const response = await fetch(`${INITIAL_POPULATION_COMPARISON_API}${path}`, {
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

async function loadInitialComparisonStrategies() {
  try {
    const payload = await requestInitialComparisonJson("/strategies");
    const strategies = payload.strategies || [];
    renderDefinitionList(dom.initialComparisonIntegrationDetails, [
      ["Contrato", "{strategyId, displayName, rows, metrics, nativeMetrics, cost, commonMetricCost, artifacts, status}"],
      ["Estrategias", strategies.map((strategy) => `${strategy.strategyId}: ${strategy.available ? "disponible" : "faltante"}`).join("; ") || "--"],
      ["Ejecucion", "Secuencial entre estrategias; paralelismo solo dentro de cada estrategia."],
      ["Metricas comunes", "[semantic_fidelity, semantic_diversity] post-hoc sobre textos finales N."],
      ["Salida", "runs/initial-population-comparison/<runId>/summary.json"],
    ]);
  } catch (error) {
    renderDefinitionList(dom.initialComparisonIntegrationDetails, [
      ["API", `No se pudo consultar el backend: ${error.message}`],
    ]);
  }
}

function setInitialComparisonRunning(isRunning, cancelRequested = false) {
  dom.runInitialComparisonButton.disabled = isRunning;
  dom.cancelInitialComparisonButton.disabled = !isRunning || !currentInitialComparisonRunId || cancelRequested;
  dom.cancelInitialComparisonButton.textContent = cancelRequested ? "Cancelando..." : "Cancelar";
  dom.clearInitialComparisonButton.disabled = isRunning;
  dom.loadInitialComparisonModelsButton.disabled = isRunning;
  dom.saveInitialComparisonReferenceButton.disabled = isRunning;
  document.querySelectorAll("#initialPopulationComparison input, #initialPopulationComparison select, #initialPopulationComparison textarea").forEach((field) => {
    field.disabled = isRunning;
  });
}

function stopInitialComparisonPolling() {
  if (initialComparisonPollTimer) {
    window.clearInterval(initialComparisonPollTimer);
    initialComparisonPollTimer = null;
  }
}

function resetInitialComparisonUi() {
  stopInitialComparisonPolling();
  currentInitialComparisonRunId = null;
  dom.initialComparisonRunStatus.textContent = "--";
  dom.initialComparisonProgressPercent.textContent = "--";
  dom.initialComparisonProgressSummary.textContent = "Sin corrida activa.";
  dom.initialComparisonLlmCalls.textContent = "--";
  dom.initialComparisonLlmCallsDetail.textContent = "LM Studio y Ollama, separados por estrategia.";
  dom.initialComparisonWallClock.textContent = "--";
  dom.initialComparisonLlmTime.textContent = "--";
  dom.initialComparisonEmbeddingTime.textContent = "--";
  dom.initialComparisonCompletedStrategies.textContent = "--";
  dom.initialComparisonRunId.textContent = "--";
  dom.initialComparisonConnectionText.textContent = "Sin ejecucion";
  dom.initialComparisonConnectionDot.classList.remove("is-busy", "is-error");
  dom.initialComparisonMetricsBody.innerHTML = '<tr><td colspan="13">Sin resultados todavia.</td></tr>';
  dom.initialComparisonStrategyDetails.innerHTML = `
    <details class="artifact-disclosure">
      <summary>Sin corrida</summary>
      <div class="semantic-artifact-content">Ejecuta una comparacion para ver textos, artefactos y metricas nativas.</div>
    </details>
  `;
  dom.initialComparisonLogOutput.textContent = "Sin logs todavia.";
  renderInitialComparisonProgress(null);
  setInitialComparisonRunning(false);
  setStatus(
    dom.initialComparisonStatusTone,
    dom.initialComparisonStatusTitle,
    dom.initialComparisonStatusDetail,
    "Listo",
    "Ejecuta estrategias de inicializacion en secuencia y calcula metricas comunes post-hoc.",
  );
}

async function runInitialComparison() {
  let config;
  try {
    config = readInitialComparisonConfig();
  } catch (error) {
    setStatus(dom.initialComparisonStatusTone, dom.initialComparisonStatusTitle, dom.initialComparisonStatusDetail, "Configuracion incompleta", error.message, "error");
    return;
  }

  stopInitialComparisonPolling();
  setInitialComparisonRunning(true);
  dom.initialComparisonMetricsBody.innerHTML = '<tr><td colspan="13">Esperando resultados.</td></tr>';
  dom.initialComparisonStrategyDetails.innerHTML = "";
  dom.initialComparisonLogOutput.textContent = "Iniciando corrida...";
  dom.initialComparisonConnectionDot.classList.add("is-busy");
  dom.initialComparisonConnectionDot.classList.remove("is-error");
  dom.initialComparisonConnectionText.textContent = "Ejecutando";
  setStatus(
    dom.initialComparisonStatusTone,
    dom.initialComparisonStatusTitle,
    dom.initialComparisonStatusDetail,
    "Iniciando comparacion",
    "El backend ejecutara las estrategias seleccionadas una por una.",
    "busy",
  );

  try {
    const run = await requestInitialComparisonJson("/runs", {
      method: "POST",
      body: JSON.stringify(config),
    });
    currentInitialComparisonRunId = run.runId;
    setInitialComparisonRunning(true, Boolean(run.cancelRequested));
    renderInitialComparisonRun(run);
    initialComparisonPollTimer = window.setInterval(() => refreshInitialComparisonRun(currentInitialComparisonRunId), 2000);
    await refreshInitialComparisonRun(currentInitialComparisonRunId);
  } catch (error) {
    currentInitialComparisonRunId = null;
    dom.initialComparisonConnectionDot.classList.add("is-error");
    dom.initialComparisonConnectionText.textContent = "Error";
    setInitialComparisonRunning(false);
    setStatus(dom.initialComparisonStatusTone, dom.initialComparisonStatusTitle, dom.initialComparisonStatusDetail, "Error al iniciar", error.message, "error");
  }
}

async function refreshInitialComparisonRun(runId) {
  if (!runId) return;
  try {
    const run = await requestInitialComparisonJson(`/runs/${encodeURIComponent(runId)}`);
    renderInitialComparisonRun(run);
    if (INITIAL_POPULATION_COMPARISON_TERMINAL_STATUSES.has(run.status)) {
      stopInitialComparisonPolling();
      currentInitialComparisonRunId = run.runId;
      setInitialComparisonRunning(false);
    } else {
      currentInitialComparisonRunId = run.runId;
      setInitialComparisonRunning(true, Boolean(run.cancelRequested));
    }
  } catch (error) {
    stopInitialComparisonPolling();
    setInitialComparisonRunning(false);
    dom.initialComparisonConnectionDot.classList.add("is-error");
    setStatus(dom.initialComparisonStatusTone, dom.initialComparisonStatusTitle, dom.initialComparisonStatusDetail, "Error al consultar corrida", error.message, "error");
  }
}

async function cancelInitialComparisonRun() {
  if (!currentInitialComparisonRunId) return;
  dom.cancelInitialComparisonButton.disabled = true;
  dom.cancelInitialComparisonButton.textContent = "Cancelando...";
  setStatus(dom.initialComparisonStatusTone, dom.initialComparisonStatusTitle, dom.initialComparisonStatusDetail, "Cancelando", "Se solicitara terminar el proceso activo.", "busy");
  try {
    const run = await requestInitialComparisonJson(`/runs/${encodeURIComponent(currentInitialComparisonRunId)}/cancel`, {
      method: "POST",
      body: "{}",
    });
    renderInitialComparisonRun(run);
    setInitialComparisonRunning(!INITIAL_POPULATION_COMPARISON_TERMINAL_STATUSES.has(run.status), Boolean(run.cancelRequested));
  } catch (error) {
    setInitialComparisonRunning(true, false);
    setStatus(dom.initialComparisonStatusTone, dom.initialComparisonStatusTitle, dom.initialComparisonStatusDetail, "No se pudo cancelar", error.message, "error");
  }
}

function initialComparisonStrategyViews(run) {
  const finalById = new Map((run.strategies || []).map((strategy) => [strategy.strategyId, strategy]));
  return Object.values(run.strategyStates || {}).map((state) => ({
    ...state,
    ...(finalById.get(state.strategyId) || {}),
    progressState: state,
  }));
}

function renderInitialComparisonProgress(progress, config = null) {
  if (!progress) {
    dom.initialComparisonProgressPercent.textContent = "--";
    dom.initialComparisonProgressSummary.textContent = "Sin corrida activa.";
    dom.initialComparisonProgressDetail.textContent = "Sin ejecucion.";
    dom.initialComparisonProgressBar.style.width = "0%";
    renderDefinitionList(dom.initialComparisonProgressDetails, [
      ["Tiempo transcurrido", "--"],
      ["Tiempo restante", "--"],
      ["Estrategia activa", "--"],
      ["Cola", "--"],
    ]);
    return;
  }
  const percent = Math.max(0, Math.min(100, Number(progress.percent || 0)));
  dom.initialComparisonProgressPercent.textContent = `${percent}%`;
  dom.initialComparisonProgressSummary.textContent = progress.detail || "Ejecutando.";
  dom.initialComparisonProgressDetail.textContent = progress.detail || "Ejecutando.";
  dom.initialComparisonProgressBar.style.width = `${percent}%`;
  renderDefinitionList(dom.initialComparisonProgressDetails, [
    ["Tiempo transcurrido", progress.elapsedLabel || "--"],
    ["Tiempo restante estimado", progress.remainingLabel || "No disponible"],
    ["Estrategia activa", progress.activeStrategyName || "--"],
    ["K repeticiones", config?.repetitionsK ?? 1],
    ["Cola", `${progress.queuedStrategies ?? 0}/${progress.totalStrategies ?? 0}`],
  ]);
}

function renderInitialComparisonCostSummary(costSummary) {
  if (!costSummary) {
    dom.initialComparisonLlmCalls.textContent = "--";
    dom.initialComparisonLlmCallsDetail.textContent = "LM Studio y Ollama, separados por estrategia.";
    dom.initialComparisonWallClock.textContent = "--";
    dom.initialComparisonLlmTime.textContent = "--";
    dom.initialComparisonEmbeddingTime.textContent = "--";
    return;
  }
  dom.initialComparisonLlmCalls.textContent = String(costSummary.llmCalls ?? 0);
  dom.initialComparisonLlmCallsDetail.textContent = `${costSummary.llmSuccessfulCalls ?? 0} ok, ${costSummary.llmFailedCalls ?? 0} fallida(s), ${costSummary.totalTokens ?? 0} tokens reportados.`;
  dom.initialComparisonWallClock.textContent = costSummary.runWallClockLabel || "--";
  dom.initialComparisonLlmTime.textContent = costSummary.llmClientWallClockLabel || "--";
  const native = costSummary.embeddingWallClockLabel || "0s";
  const common = costSummary.commonEmbeddingWallClockLabel || "0s";
  dom.initialComparisonEmbeddingTime.textContent = `${native} + ${common}`;
}

function renderInitialComparisonRun(run) {
  const status = comparatorStatusLabel(run.status);
  const progress = run.progress || {};
  const completed = progress.completedStrategies ?? (run.strategies || []).filter((strategy) => strategy.status === "completed").length;
  const total = progress.totalStrategies ?? Object.keys(run.strategyStates || {}).length;
  dom.initialComparisonRunStatus.textContent = status;
  dom.initialComparisonCompletedStrategies.textContent = `${completed}/${total}`;
  dom.initialComparisonRunId.textContent = run.runId || "--";
  dom.initialComparisonConnectionText.textContent = status;
  dom.initialComparisonConnectionDot.classList.toggle("is-busy", run.status === "queued" || run.status === "running");
  dom.initialComparisonConnectionDot.classList.toggle("is-error", run.status === "failed");

  const strategies = initialComparisonStrategyViews(run);
  renderInitialComparisonProgress(run.progress || null, run.config || null);
  renderInitialComparisonCostSummary(run.costSummary || null);
  renderInitialComparisonMetrics(strategies);
  renderInitialComparisonStrategyDetails(strategies);
  renderInitialComparisonLogs(run.logs || []);

  const detail = run.error
    ? run.error
    : run.cancelRequested
      ? "Cancelacion solicitada; esperando cierre del proceso activo."
      : progress.detail || `${completed} estrategia(s) completada(s).`;
  setStatus(
    dom.initialComparisonStatusTone,
    dom.initialComparisonStatusTitle,
    dom.initialComparisonStatusDetail,
    `Comparacion ${status}`,
    detail,
    run.status === "running" || run.status === "queued" ? "busy" : run.status === "failed" ? "error" : "ready",
  );
}

function finiteMetricValue(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function initialComparisonBestScalarIds(strategies, getter, direction = "max") {
  const values = strategies
    .filter((strategy) => strategy.status === "completed")
    .map((strategy) => ({ id: strategy.strategyId, value: finiteMetricValue(getter(strategy)) }))
    .filter((entry) => entry.id && entry.value !== null);
  if (!values.length) return new Set();
  const bestValue = direction === "min"
    ? Math.min(...values.map((entry) => entry.value))
    : Math.max(...values.map((entry) => entry.value));
  return new Set(values
    .filter((entry) => Math.abs(entry.value - bestValue) <= Number.EPSILON * 100)
    .map((entry) => entry.id));
}

function initialComparisonDominatesVector(left, right) {
  if (!Array.isArray(left) || !Array.isArray(right) || left.length !== right.length || !left.length) {
    return false;
  }
  const leftValues = left.map(finiteMetricValue);
  const rightValues = right.map(finiteMetricValue);
  if (leftValues.some((value) => value === null) || rightValues.some((value) => value === null)) {
    return false;
  }
  return leftValues.every((value, index) => value >= rightValues[index])
    && leftValues.some((value, index) => value > rightValues[index]);
}

function initialComparisonBestObjectiveIds(strategies) {
  const entries = strategies
    .filter((strategy) => strategy.status === "completed")
    .map((strategy) => ({ id: strategy.strategyId, vector: strategy.metrics?.bestObjectiveVector }))
    .filter((entry) => entry.id && Array.isArray(entry.vector) && entry.vector.length);
  return new Set(entries
    .filter((entry) => !entries.some((other) => other !== entry && initialComparisonDominatesVector(other.vector, entry.vector)))
    .map((entry) => entry.id));
}

function initialComparisonMetricCell(value, isBest) {
  const escaped = escapeHtml(value);
  return `<td>${isBest ? `<strong class="metric-best">${escaped}</strong>` : escaped}</td>`;
}

function initialComparisonMetricWinners(strategies) {
  return {
    individuals: initialComparisonBestScalarIds(strategies, (strategy) => strategy.metrics?.completedRows, "max"),
    nonDominated: initialComparisonBestScalarIds(strategies, (strategy) => strategy.metrics?.nonDominatedRows, "max"),
    objective: initialComparisonBestObjectiveIds(strategies),
    hypervolume: initialComparisonBestScalarIds(strategies, (strategy) => strategy.metrics?.hypervolume, "max"),
    spread: initialComparisonBestScalarIds(strategies, (strategy) => strategy.metrics?.spread, "min"),
    wallClock: initialComparisonBestScalarIds(strategies, (strategy) => strategy.cost?.processWallClockSeconds ?? strategy.cost?.wallClockSeconds, "min"),
    sequentialWallClock: initialComparisonBestScalarIds(strategies, (strategy) => strategy.cost?.estimatedSequentialWallClockSeconds, "min"),
    llmCalls: initialComparisonBestScalarIds(strategies, (strategy) => strategy.cost?.llmCalls, "min"),
    llmTime: initialComparisonBestScalarIds(strategies, (strategy) => strategy.cost?.llmClientWallClockSeconds, "min"),
    commonEmbeddings: initialComparisonBestScalarIds(strategies, (strategy) => strategy.commonMetricCost?.embeddingWallClockSeconds, "min"),
  };
}

function renderInitialComparisonMetrics(strategies) {
  if (!strategies.length) {
    dom.initialComparisonMetricsBody.innerHTML = '<tr><td colspan="13">Sin resultados todavia.</td></tr>';
    return;
  }
  const winners = initialComparisonMetricWinners(strategies);
  dom.initialComparisonMetricsBody.replaceChildren(
    ...strategies.map((strategy) => {
      const metrics = strategy.metrics || {};
      const cost = strategy.cost || {};
      const commonCost = strategy.commonMetricCost || {};
      const strategyId = strategy.strategyId;
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td>${escapeHtml(strategy.displayName || strategy.strategyId)}</td>
        <td><span class="${comparatorStatusClass(strategy.status)}">${escapeHtml(comparatorStatusLabel(strategy.status))}</span></td>
        <td>${escapeHtml(strategy.runtime || "--")}</td>
        ${initialComparisonMetricCell(`${metrics.completedRows ?? 0}/${metrics.totalRows ?? 0}`, winners.individuals.has(strategyId))}
        ${initialComparisonMetricCell(String(metrics.nonDominatedRows ?? 0), winners.nonDominated.has(strategyId))}
        ${initialComparisonMetricCell(metrics.bestObjectiveLabel || "--", winners.objective.has(strategyId))}
        ${initialComparisonMetricCell(metrics.hypervolumeLabel || "No aplica", winners.hypervolume.has(strategyId))}
        ${initialComparisonMetricCell(metrics.spreadLabel || "No aplica", winners.spread.has(strategyId))}
        ${initialComparisonMetricCell(cost.processWallClockLabel || cost.wallClockLabel || "--", winners.wallClock.has(strategyId))}
        ${initialComparisonMetricCell(cost.estimatedSequentialWallClockLabel || "--", winners.sequentialWallClock.has(strategyId))}
        ${initialComparisonMetricCell(String(cost.llmCalls ?? 0), winners.llmCalls.has(strategyId))}
        ${initialComparisonMetricCell(cost.llmClientWallClockLabel || "--", winners.llmTime.has(strategyId))}
        ${initialComparisonMetricCell(commonCost.embeddingWallClockLabel || "--", winners.commonEmbeddings.has(strategyId))}
      `;
      return tr;
    }),
  );
}

function renderInitialComparisonStrategyDetails(strategies) {
  if (!strategies.length) {
    dom.initialComparisonStrategyDetails.innerHTML = `
      <details class="artifact-disclosure">
        <summary>Sin corrida</summary>
        <div class="semantic-artifact-content">Ejecuta una comparacion para ver textos, artefactos y metricas nativas.</div>
      </details>
    `;
    return;
  }
  dom.initialComparisonStrategyDetails.replaceChildren(
    ...strategies.map((strategy) => {
      const details = document.createElement("details");
      details.className = "artifact-disclosure";
      const summary = document.createElement("summary");
      summary.textContent = `${strategy.displayName || strategy.strategyId} - ${comparatorStatusLabel(strategy.status)}`;
      const content = document.createElement("div");
      content.className = "semantic-artifact-content";
      content.append(
        initialComparisonDefinitionBlock("Metricas comunes", [
          ["F.O.", strategy.metrics?.bestObjectiveLabel || "--"],
          ["No dominadas", String(strategy.metrics?.nonDominatedRows ?? 0)],
          ["HV", strategy.metrics?.hypervolumeLabel || "No aplica"],
          ["Spread", strategy.metrics?.spreadLabel || "No aplica"],
        ]),
        initialComparisonDefinitionBlock("Metricas nativas", [
          ["Objetivos", (strategy.nativeMetrics?.objectiveNames || []).join(", ") || "--"],
          ["Mejor F.O.", strategy.nativeMetrics?.bestObjectiveLabel || "--"],
          ["No dominadas nativas", String(strategy.nativeMetrics?.nonDominatedRows ?? 0)],
        ]),
        initialComparisonDisclosure("Peculiaridades y artefactos", renderInitialComparisonArtifacts(strategy)),
        initialComparisonDisclosure(`Textos finales (${(strategy.rows || []).length})`, renderInitialComparisonRowsTable(strategy.rows || [])),
      );
      details.append(summary, content);
      return details;
    }),
  );
}

function initialComparisonDisclosure(title, contentNode) {
  const details = document.createElement("details");
  details.className = "artifact-disclosure nested-artifact-disclosure";
  const summary = document.createElement("summary");
  summary.textContent = title;
  details.append(summary, contentNode);
  return details;
}

function initialComparisonDefinitionBlock(title, entries) {
  const wrapper = document.createElement("div");
  wrapper.className = "semantic-artifact-group";
  const heading = document.createElement("h3");
  heading.textContent = title;
  const dl = document.createElement("dl");
  dl.className = "definition-grid compact-definition-grid";
  renderDefinitionList(dl, entries);
  wrapper.append(heading, dl);
  return wrapper;
}

function renderInitialComparisonArtifacts(strategy) {
  const wrapper = document.createElement("div");
  wrapper.className = "semantic-artifact-group";
  const heading = document.createElement("h3");
  heading.textContent = "Peculiaridades y artefactos";
  wrapper.append(heading);
  const artifacts = strategy.artifacts || {};
  if (strategy.strategyId === "hybrid-semantic-v7") {
    const semantic = artifacts.semanticArtifacts || {};
    wrapper.append(renderInitialComparisonChipGroup("Anclas", semantic.anchors || {}));
    wrapper.append(renderInitialComparisonPools(semantic.pools || {}));
  } else {
    const rolesTopics = artifacts.rolesTopics || {};
    wrapper.append(renderInitialComparisonChipGroup("Roles/topicos upstream", rolesTopics));
    const pre = document.createElement("pre");
    pre.className = "output-box strategy-artifact-json";
    pre.textContent = JSON.stringify(artifacts.upstreamPrompts || {}, null, 2);
    wrapper.append(pre);
  }
  return wrapper;
}

function renderInitialComparisonChipGroup(title, groups) {
  const wrapper = document.createElement("div");
  wrapper.className = "semantic-artifact-group";
  const heading = document.createElement("h3");
  heading.textContent = title;
  wrapper.append(heading);
  Object.entries(groups || {}).forEach(([groupName, values]) => {
    const listValues = Array.isArray(values) ? values : [];
    if (!listValues.length) return;
    const meta = document.createElement("p");
    meta.className = "artifact-meta";
    meta.textContent = `${semanticArtifactLabel(groupName)} (${listValues.length})`;
    const chips = document.createElement("div");
    chips.className = "chip-list";
    listValues.forEach((value) => {
      const chip = document.createElement("span");
      chip.className = "semantic-chip";
      chip.textContent = String(value);
      chips.append(chip);
    });
    wrapper.append(meta, chips);
  });
  return wrapper;
}

function renderInitialComparisonPools(pools) {
  const components = pools.components || {};
  const normalized = {};
  Object.entries(components).forEach(([name, payload]) => {
    normalized[name] = Array.isArray(payload?.items) ? payload.items : [];
  });
  return renderInitialComparisonChipGroup("Pools generados", normalized);
}

function renderInitialComparisonRowsTable(rows) {
  const wrapper = document.createElement("div");
  wrapper.className = "table-wrap comparison-population-scroll";
  const table = document.createElement("table");
  table.className = "wide-table initial-results-table comparison-results-table";
  table.innerHTML = `
    <thead>
      <tr>
        <th>Rank</th>
        <th>Rol</th>
        <th>Topico</th>
        <th>Accion</th>
        <th>Prompt</th>
        <th>Texto generado</th>
        <th>Fidelidad comun</th>
        <th>Diversidad comun</th>
        <th>F.O. comun</th>
        <th>F.O. nativa</th>
        <th>No dominada</th>
        <th>Estado</th>
      </tr>
    </thead>
  `;
  const tbody = document.createElement("tbody");
  if (!rows.length) {
    tbody.innerHTML = '<tr><td colspan="12">Sin resultados todavia.</td></tr>';
  } else {
    rows.forEach((row) => {
      const tr = document.createElement("tr");
      const evaluated = Array.isArray(row.commonObjectiveVector) && row.commonObjectiveVector.length > 1 && row.status === "ok";
      if (evaluated && !row.commonNonDominated) tr.classList.add("is-dominated-row");
      if (row.commonNonDominated) tr.classList.add("is-nondominated-row");
      tr.innerHTML = `
        <td>${escapeHtml(String(row.comparisonRank ?? row.rank ?? "--"))}</td>
        <td class="context-cell">${escapeHtml(row.role || "--")}</td>
        <td class="context-cell">${escapeHtml(row.topic || "--")}</td>
        <td class="context-cell">${escapeHtml(row.action || "--")}</td>
        <td class="context-cell long-cell"><div class="scroll-cell">${escapeHtml(row.prompt || "--")}</div></td>
        <td class="context-cell long-cell"><div class="scroll-cell">${escapeHtml(row.generatedText || "--")}</div></td>
        <td>${escapeHtml(formatOptionalNumber(row.commonFidelity, 6))}</td>
        <td>${escapeHtml(formatOptionalNumber(row.commonDiversity, 6))}</td>
        <td>${escapeHtml(row.commonObjectiveLabel || "--")}</td>
        <td>${escapeHtml(row.nativeObjectiveLabel || "--")}</td>
        <td>${row.commonNonDominated ? '<span class="valid">si</span>' : evaluated ? '<span class="invalid">no</span>' : "--"}</td>
        <td><span class="${row.status === "ok" ? "valid" : "invalid"}">${escapeHtml(row.status || "--")}</span></td>
      `;
      tbody.append(tr);
    });
  }
  table.append(tbody);
  wrapper.append(table);
  return wrapper;
}

function renderInitialComparisonLogs(logs) {
  if (!logs.length) {
    dom.initialComparisonLogOutput.textContent = "Sin logs todavia.";
    return;
  }
  dom.initialComparisonLogOutput.textContent = logs
    .slice(-60)
    .map((entry) => `[${entry.strategyId}] ${entry.message}`)
    .join("\n");
}

function turbulenceStatusLabel(status) {
  const labels = {
    queued: "en cola",
    running: "ejecutando",
    completed: "completado",
    failed: "fallido",
    cancelled: "cancelado",
  };
  return labels[status] || status || "--";
}

async function requestTurbulenceComparisonJson(path, options = {}) {
  const response = await fetch(`${TURBULENCE_COMPARISON_API}${path}`, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...(options.headers || {}),
    },
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.error || `Solicitud fallida (${response.status}).`);
  }
  return payload;
}

function readTurbulencePpdbPayload() {
  const sourcePath = dom.turbulencePpdbSourcePath.value.trim() || TURBULENCE_PPDB_DEFAULT_SOURCE_PATH;
  const indexPath = dom.turbulencePpdbIndexPath.value.trim() || TURBULENCE_PPDB_DEFAULT_INDEX_PATH;
  return {
    usePpdb: dom.turbulenceUsePpdb.checked,
    ppdbSourcePath: sourcePath,
    ppdbIndexPath: indexPath,
  };
}

function renderTurbulencePpdbStatus(ppdb) {
  if (!ppdb) {
    dom.turbulencePpdbStatus.textContent = "--";
    dom.turbulencePpdbDetail.textContent = "Índice compacto no revisado.";
    dom.turbulencePpdbSetupStatus.textContent = "Descarga PPDB desde Kaggle y deja el archivo en data/external/ppdb/ppdb-2.0-s-all.";
    return;
  }
  if (ppdb.enabled === false) {
    dom.turbulencePpdbStatus.textContent = "desactivado";
    dom.turbulencePpdbDetail.textContent = ppdb.message || "PPDB desactivado; la estrategia usará solo WordNet.";
    dom.turbulencePpdbSetupStatus.textContent = "PPDB no se consultará en la comparación mientras esté desactivado.";
    return;
  }

  const source = ppdb.source || {};
  const index = ppdb.index || {};
  dom.turbulencePpdbStatus.textContent = ppdb.available ? "disponible" : "no disponible";
  dom.turbulencePpdbDetail.textContent = ppdb.available
    ? `${ppdb.entries ?? index.entries ?? 0} clave(s) en ${ppdb.path || index.path || "índice local"}.`
    : (ppdb.message || index.message || "Índice compacto PPDB no preparado.");
  const sourceLabel = source.exists
    ? `Dataset local encontrado (${source.sizeLabel || "--"}): ${source.path || "--"}`
    : `Dataset local faltante: ${source.path || dom.turbulencePpdbSourcePath.value.trim() || "--"}`;
  const indexLabel = index.exists
    ? `Índice: ${index.available ? "disponible" : "inválido"} en ${index.path || ppdb.path || "--"}`
    : `Índice faltante: ${index.path || ppdb.path || dom.turbulencePpdbIndexPath.value.trim() || "--"}`;
  dom.turbulencePpdbSetupStatus.textContent = `${sourceLabel}. ${indexLabel}.`;
}

async function refreshTurbulencePpdbStatus() {
  try {
    const status = await requestTurbulenceComparisonJson("/ppdb/status", {
      method: "POST",
      body: JSON.stringify(readTurbulencePpdbPayload()),
    });
    renderTurbulencePpdbStatus(status);
  } catch (error) {
    dom.turbulencePpdbStatus.textContent = "error";
    dom.turbulencePpdbDetail.textContent = error.message;
    dom.turbulencePpdbSetupStatus.textContent = `No se pudo verificar PPDB: ${error.message}`;
  }
}

async function prepareTurbulencePpdb() {
  const originalText = dom.prepareTurbulencePpdbButton.textContent;
  dom.prepareTurbulencePpdbButton.disabled = true;
  dom.refreshTurbulencePpdbButton.disabled = true;
  dom.prepareTurbulencePpdbButton.textContent = "Preparando...";
  dom.turbulencePpdbSetupStatus.textContent = "Leyendo PPDB completo y generando índice compacto. Puede tardar varios minutos.";
  try {
    const payload = await requestTurbulenceComparisonJson("/ppdb/prepare", {
      method: "POST",
      body: JSON.stringify(readTurbulencePpdbPayload()),
    });
    renderTurbulencePpdbStatus(payload.ppdb);
  } catch (error) {
    dom.turbulencePpdbStatus.textContent = "error";
    dom.turbulencePpdbDetail.textContent = error.message;
    dom.turbulencePpdbSetupStatus.textContent = `No se pudo preparar PPDB: ${error.message}`;
  } finally {
    dom.prepareTurbulencePpdbButton.textContent = originalText;
    dom.prepareTurbulencePpdbButton.disabled = false;
    dom.refreshTurbulencePpdbButton.disabled = false;
  }
}

function readTurbulenceComparisonConfig() {
  const strategies = [];
  if (dom.turbulenceStrategyLlm.checked) strategies.push("llm");
  if (dom.turbulenceStrategyWordnet.checked) strategies.push("wordnet-ppdb-sbert");
  if (dom.turbulenceStrategyDistilbert.checked) strategies.push("distilbert-sbert");
  if (!strategies.length) {
    throw new Error("Selecciona al menos una estrategia.");
  }

  const minSimilarity = readClampedNumber(dom.turbulenceComparisonMinSimilarity, "Sim. minima", -1, 1);
  const maxSimilarity = readClampedNumber(dom.turbulenceComparisonMaxSimilarity, "Sim. maxima", -1, 1);
  if (minSimilarity > maxSimilarity) {
    throw new Error("La similitud minima no puede ser mayor que la maxima.");
  }
  const minWords = Math.floor(readClampedNumber(dom.turbulenceMinWords, "Min. palabras", 1, 40));
  const maxWords = Math.floor(readClampedNumber(dom.turbulenceMaxWords, "Max. palabras", 1, 80));
  if (minWords > maxWords) {
    throw new Error("El minimo de palabras no puede ser mayor que el maximo.");
  }
  const finalFidelityMin = readClampedNumber(dom.turbulenceFinalFidelityMin, "Fidelidad final minima", -1, 1);
  const finalFidelityMax = readClampedNumber(dom.turbulenceFinalFidelityMax, "Fidelidad final maxima", -1, 1);
  if (finalFidelityMin > finalFidelityMax) {
    throw new Error("La fidelidad final minima no puede ser mayor que la maxima.");
  }

  const requiredText = (input, label) => {
    const value = input.value.trim();
    if (!value) {
      throw new Error(`${label} no puede estar vacio.`);
    }
    return value;
  };

  return {
    strategies,
    individualCount: Math.floor(readClampedNumber(dom.turbulenceIndividualCount, "N individuos", 1, 200)),
    seed: Math.floor(readClampedNumber(dom.turbulenceSeed, "Semilla", 0, 2147483647)),
    repetitionsK: Math.floor(readClampedNumber(dom.turbulenceRepetitions, "K repeticiones", 1, 30)),
    kCandidates: Math.floor(readClampedNumber(dom.turbulenceCandidateCount, "K candidatos", 1, 30)),
    turbulenceMinSimilarity: minSimilarity,
    turbulenceMaxSimilarity: maxSimilarity,
    minWords,
    maxWords,
    embeddingModel: requiredText(dom.turbulenceEmbeddingModel, "Modelo SBERT"),
    distilbertModel: requiredText(dom.turbulenceDistilbertModel, "Modelo DistilBERT"),
    referenceText: requiredText(dom.turbulenceReferenceText, "Texto de referencia final"),
    finalFidelityMin,
    finalFidelityMax,
    usePpdb: dom.turbulenceUsePpdb.checked,
    ppdbSourcePath: dom.turbulenceUsePpdb.checked
      ? requiredText(dom.turbulencePpdbSourcePath, "Archivo PPDB completo")
      : (dom.turbulencePpdbSourcePath.value.trim() || TURBULENCE_PPDB_DEFAULT_SOURCE_PATH),
    ppdbIndexPath: dom.turbulenceUsePpdb.checked
      ? requiredText(dom.turbulencePpdbIndexPath, "Índice compacto PPDB")
      : (dom.turbulencePpdbIndexPath.value.trim() || TURBULENCE_PPDB_DEFAULT_INDEX_PATH),
    operatorParallelism: Math.floor(readClampedNumber(dom.turbulenceOperatorParallelism, "Paralelismo operador", 1, 8)),
    generationParallelism: Math.floor(readClampedNumber(dom.turbulenceGenerationParallelism, "Paralelismo generacion final", 1, 8)),
    lmStudio: {
      baseUrl: requiredText(dom.turbulenceLmBaseUrl, "Base URL LM Studio"),
      apiMode: dom.turbulenceLmApiMode.value,
      model: requiredText(dom.turbulenceLlmModel, "Modelo LLM"),
      temperature: readClampedNumber(dom.turbulenceLlmTemperature, "Temperatura", 0, 2),
      topP: readClampedNumber(dom.turbulenceLlmTopP, "Top p", 0, 1),
      maxTokens: Math.floor(readClampedNumber(dom.turbulenceLlmMaxTokens, "Max. tokens", 8, 4096)),
      timeoutSeconds: readClampedNumber(dom.turbulenceTimeoutSeconds, "Timeout", 5, 1200),
    },
    llmTurbulenceSystemPrompt: requiredText(dom.turbulenceLlmSystemPrompt, "System prompt turbulencia LLM"),
    llmTurbulencePromptTemplate: requiredText(dom.turbulenceLlmPromptTemplate, "Prompt turbulencia LLM"),
    finalPromptTemplate: requiredText(dom.turbulenceFinalPromptTemplate, "Prompt generacion final"),
  };
}

function setTurbulenceComparisonRunning(isRunning, cancelRequested = false) {
  dom.loadTurbulenceModelsButton.disabled = isRunning;
  dom.runTurbulenceComparisonButton.disabled = isRunning;
  dom.cancelTurbulenceComparisonButton.disabled = !isRunning || !currentTurbulenceComparisonRunId || cancelRequested;
  dom.cancelTurbulenceComparisonButton.textContent = cancelRequested ? "Cancelando..." : "Cancelar";
  dom.clearTurbulenceComparisonButton.disabled = isRunning;
  dom.refreshTurbulencePpdbButton.disabled = isRunning;
  dom.prepareTurbulencePpdbButton.disabled = isRunning;
  document
    .querySelectorAll("#turbulenceComparison input, #turbulenceComparison select, #turbulenceComparison textarea")
    .forEach((field) => {
      field.disabled = isRunning;
    });
}

function stopTurbulenceComparisonPolling() {
  if (turbulenceComparisonPollTimer) {
    window.clearInterval(turbulenceComparisonPollTimer);
    turbulenceComparisonPollTimer = null;
  }
}

function resetTurbulenceComparisonUi() {
  stopTurbulenceComparisonPolling();
  currentTurbulenceComparisonRunId = null;
  setTurbulenceComparisonRunning(false);
  dom.turbulenceRunStatus.textContent = "--";
  dom.turbulenceRunStatusDetail.textContent = "Sin corrida activa.";
  dom.turbulenceProgressPercent.textContent = "--";
  dom.turbulenceProgressSummary.textContent = "Sin ejecución.";
  dom.turbulenceMovementCount.textContent = "--";
  dom.turbulenceRecommendation.textContent = "--";
  dom.turbulenceRecommendationDetail.textContent = "Pendiente.";
  dom.turbulenceBaselineFidelity.textContent = "--";
  dom.turbulenceBaselineDiversity.textContent = "--";
  renderTurbulencePpdbStatus(null);
  dom.turbulenceRunId.textContent = "--";
  dom.turbulenceComparisonConnectionText.textContent = "Sin ejecución";
  dom.turbulenceComparisonConnectionDot.classList.remove("is-busy", "is-error");
  dom.turbulenceStrategyMetricsBody.innerHTML = '<tr><td colspan="10">Sin resultados todavía.</td></tr>';
  dom.turbulenceMovementRowsBody.innerHTML = '<tr><td colspan="8">Sin movimientos todavía.</td></tr>';
  dom.turbulenceLogOutput.textContent = "Sin logs todavía.";
  setStatus(
    dom.turbulenceStatusTone,
    dom.turbulenceStatusTitle,
    dom.turbulenceStatusDetail,
    "Listo",
    "Configura las estrategias y ejecuta una comparacion corta para validar el flujo.",
  );
}

async function runTurbulenceComparison() {
  try {
    const config = readTurbulenceComparisonConfig();
    stopTurbulenceComparisonPolling();
    currentTurbulenceComparisonRunId = null;
    setTurbulenceComparisonRunning(true);
    setStatus(
      dom.turbulenceStatusTone,
      dom.turbulenceStatusTitle,
      dom.turbulenceStatusDetail,
      "Enviando corrida",
      "Preparando la comparacion en backend.",
      "busy",
    );
    const run = await requestTurbulenceComparisonJson("/runs", {
      method: "POST",
      body: JSON.stringify(config),
    });
    currentTurbulenceComparisonRunId = run.runId;
    renderTurbulenceComparisonRun(run);
    turbulenceComparisonPollTimer = window.setInterval(
      () => refreshTurbulenceComparisonRun(currentTurbulenceComparisonRunId),
      2000,
    );
    await refreshTurbulenceComparisonRun(currentTurbulenceComparisonRunId);
  } catch (error) {
    currentTurbulenceComparisonRunId = null;
    stopTurbulenceComparisonPolling();
    setTurbulenceComparisonRunning(false);
    setStatus(dom.turbulenceStatusTone, dom.turbulenceStatusTitle, dom.turbulenceStatusDetail, "Error", error.message, "error");
    dom.turbulenceComparisonConnectionText.textContent = "Error";
    dom.turbulenceComparisonConnectionDot.classList.add("is-error");
  }
}

async function refreshTurbulenceComparisonRun(runId) {
  if (!runId) return;
  try {
    const run = await requestTurbulenceComparisonJson(`/runs/${encodeURIComponent(runId)}`);
    renderTurbulenceComparisonRun(run);
    if (TURBULENCE_COMPARISON_TERMINAL_STATUSES.has(run.status)) {
      stopTurbulenceComparisonPolling();
      currentTurbulenceComparisonRunId = run.runId;
      setTurbulenceComparisonRunning(false);
    } else {
      currentTurbulenceComparisonRunId = run.runId;
      setTurbulenceComparisonRunning(true, Boolean(run.cancelRequested));
    }
  } catch (error) {
    stopTurbulenceComparisonPolling();
    setTurbulenceComparisonRunning(false);
    setStatus(dom.turbulenceStatusTone, dom.turbulenceStatusTitle, dom.turbulenceStatusDetail, "Error consultando corrida", error.message, "error");
  }
}

async function cancelTurbulenceComparisonRun() {
  if (!currentTurbulenceComparisonRunId) return;
  setTurbulenceComparisonRunning(true, true);
  try {
    const run = await requestTurbulenceComparisonJson(`/runs/${encodeURIComponent(currentTurbulenceComparisonRunId)}/cancel`, {
      method: "POST",
      body: JSON.stringify({}),
    });
    renderTurbulenceComparisonRun(run);
    setTurbulenceComparisonRunning(!TURBULENCE_COMPARISON_TERMINAL_STATUSES.has(run.status), true);
  } catch (error) {
    setStatus(dom.turbulenceStatusTone, dom.turbulenceStatusTitle, dom.turbulenceStatusDetail, "Error cancelando corrida", error.message, "error");
    setTurbulenceComparisonRunning(true);
  }
}

function renderTurbulenceComparisonRun(run) {
  const progress = run.progress || {};
  const result = run.result || {};
  const baseline = result.baseline || {};
  const ppdb = result.ppdb || {};
  const repetitionsK = result.repetitionsK ?? run.config?.repetitionsK ?? 1;
  const percent = Number.isFinite(Number(progress.percent)) ? Math.round(Number(progress.percent)) : 0;
  const statusLabel = turbulenceStatusLabel(run.status);

  dom.turbulenceRunStatus.textContent = statusLabel;
  dom.turbulenceRunStatusDetail.textContent = progress.message || run.error || "Sin detalle.";
  dom.turbulenceProgressPercent.textContent = `${percent}%`;
  dom.turbulenceProgressSummary.textContent = `${progress.completed ?? 0}/${progress.total ?? 0} - ${progress.stage || "--"} | K ${repetitionsK}`;
  dom.turbulenceMovementCount.textContent = result.movementCount ?? "--";
  dom.turbulenceRunId.textContent = run.runId || "--";
  dom.turbulenceBaselineFidelity.textContent = formatOptionalNumber(baseline.averageFidelity, 6);
  dom.turbulenceBaselineDiversity.textContent = formatOptionalNumber(baseline.averageDiversity, 6);
  renderTurbulencePpdbStatus(ppdb);

  const recommendation = result.recommendation || {};
  dom.turbulenceRecommendation.textContent = recommendation.displayName || "--";
  dom.turbulenceRecommendationDetail.textContent = recommendation.message || "Pendiente.";

  const isRunning = run.status === "queued" || run.status === "running";
  dom.turbulenceComparisonConnectionText.textContent = statusLabel;
  dom.turbulenceComparisonConnectionDot.classList.toggle("is-busy", isRunning);
  dom.turbulenceComparisonConnectionDot.classList.toggle("is-error", run.status === "failed");
  setStatus(
    dom.turbulenceStatusTone,
    dom.turbulenceStatusTitle,
    dom.turbulenceStatusDetail,
    statusLabel,
    run.error || progress.message || "Sin detalle.",
    run.status === "failed" ? "error" : isRunning ? "busy" : "ready",
  );

  renderTurbulenceStrategyMetrics(result.strategies || [], run.config || {}, recommendation);
  renderTurbulenceMovementRows(result.strategies || []);
  renderTurbulenceLogs(run.logs || []);
}

function formatTurbulenceRate(value) {
  const number = Number(value);
  return Number.isFinite(number) ? `${formatNumber(number * 100, 2)}%` : "--";
}

function formatSignedOptional(value) {
  if (value === null || value === undefined || value === "") {
    return "--";
  }
  const number = Number(value);
  return Number.isFinite(number) ? formatSigned(number) : "--";
}

function turbulenceCostLabel(cost = {}) {
  const ppdbQueries = cost.ppdbQueries ?? 0;
  const ppdbAttempts = cost.ppdbLookupAttempts ?? ppdbQueries;
  const ppdbLabel = ppdbAttempts > ppdbQueries
    ? `PPDB ${ppdbQueries} (${ppdbAttempts} intento${ppdbAttempts === 1 ? "" : "s"} no disp.)`
    : `PPDB ${ppdbQueries}`;
  return [
    `LLM ${cost.llmCalls ?? 0}`,
    `WN ${cost.wordnetQueries ?? 0}`,
    ppdbLabel,
    `D-BERT ${cost.distilbertInferences ?? 0}`,
    `Emb ${cost.embeddingTexts ?? 0}`,
    `SBERT ${formatDuration(Number(cost.embeddingWallClockSeconds || 0) * 1000)}`,
  ].join(" | ");
}

function bestTurbulenceIndexes(strategies, getter, direction = "max") {
  const values = strategies
    .map((strategy, index) => ({ index, value: Number(getter(strategy)) }))
    .filter((entry) => Number.isFinite(entry.value));
  if (!values.length) {
    return new Set();
  }
  const target = direction === "min"
    ? Math.min(...values.map((entry) => entry.value))
    : Math.max(...values.map((entry) => entry.value));
  return new Set(values.filter((entry) => entry.value === target).map((entry) => entry.index));
}

function turbulenceBestClass(bestIndexes, index) {
  return bestIndexes.has(index) ? "metric-best" : "";
}

function turbulenceMetricLabel(key) {
  return {
    success: "Tasa de éxito",
    coverage: "Cobertura",
    similarity: "Similitud promedio del operador vs componente original",
    time: "Tiempo promedio",
    cost: "Costo operador",
    fidelity: "Fidelidad final semántica vs referencia",
    fidelityDelta: "Delta fidelidad vs baseline sin turbulencia",
    diversity: "Diversidad semántica entre textos finales",
    diversityDelta: "Delta diversidad vs baseline sin turbulencia",
  }[key] || key;
}

function turbulenceMetricValue(strategy, key, config = {}) {
  const operator = strategy.operatorMetrics || {};
  const finalMetrics = strategy.finalMetrics || {};
  const cost = operator.cost || {};
  const deltas = strategy.deltas || {};
  if (key === "success") return formatTurbulenceRate(operator.successRate);
  if (key === "coverage") return formatTurbulenceRate(operator.coverageRate);
  if (key === "similarity") return formatOptionalNumber(operator.averageSimilarity, 6);
  if (key === "time") return formatDuration(Number(operator.operatorAverageSeconds || 0) * 1000);
  if (key === "cost") return `${turbulenceCostLabel(cost)}; costo relativo ${formatOptionalNumber(strategy.relativeOperatorCost, 2)}`;
  if (key === "fidelity") return formatOptionalNumber(finalMetrics.averageFidelity, 6);
  if (key === "fidelityDelta") return formatSignedOptional(deltas.averageFidelityDelta);
  if (key === "diversity") return formatOptionalNumber(finalMetrics.averageDiversity, 6);
  if (key === "diversityDelta") return formatSignedOptional(deltas.averageDiversityDelta);
  return "--";
}

function turbulenceMetricCell(content, className, key) {
  const label = turbulenceMetricLabel(key);
  return `<td class="${className} metric-breakdown-cell" data-turbulence-breakdown="${key}" tabindex="0" role="button" aria-label="Ver desglose por K: ${escapeHtml(label)}">${content}</td>`;
}

function closeTurbulenceMetricPopover() {
  if (turbulenceMetricPopover) {
    turbulenceMetricPopover.hidden = true;
  }
}

function ensureTurbulenceMetricPopover() {
  if (turbulenceMetricPopover) return turbulenceMetricPopover;
  turbulenceMetricPopover = document.createElement("div");
  turbulenceMetricPopover.className = "metric-breakdown-popover";
  turbulenceMetricPopover.hidden = true;
  turbulenceMetricPopover.setAttribute("role", "tooltip");
  document.body.appendChild(turbulenceMetricPopover);
  document.addEventListener("click", (event) => {
    if (
      turbulenceMetricPopover
      && !turbulenceMetricPopover.hidden
      && !turbulenceMetricPopover.contains(event.target)
      && !(event.target instanceof Element && event.target.closest("[data-turbulence-breakdown]"))
    ) {
      closeTurbulenceMetricPopover();
    }
  });
  window.addEventListener("scroll", closeTurbulenceMetricPopover, true);
  window.addEventListener("resize", closeTurbulenceMetricPopover);
  return turbulenceMetricPopover;
}

function showTurbulenceMetricBreakdown(event, strategy, key, config) {
  event.stopPropagation();
  const popover = ensureTurbulenceMetricPopover();
  const repetitions = Array.isArray(strategy.repetitions) && strategy.repetitions.length
    ? strategy.repetitions
    : [strategy];
  const rows = repetitions.map((item, index) => ({
    k: item.repetitionIndex ?? index + 1,
    seed: item.repetitionSeed ?? "--",
    value: turbulenceMetricValue(item, key, config),
  }));
  popover.innerHTML = `
    <div class="metric-breakdown-title">${escapeHtml(strategy.displayName || strategy.strategyId || "Estrategia")}</div>
    <div class="metric-breakdown-subtitle">${escapeHtml(turbulenceMetricLabel(key))}</div>
    <p><strong>Agregado:</strong> ${escapeHtml(turbulenceMetricValue(strategy, key, config))}</p>
    <table>
      <thead>
        <tr><th>K</th><th>Semilla</th><th>Valor</th></tr>
      </thead>
      <tbody>
        ${rows.map((row) => `
          <tr>
            <td>${escapeHtml(String(row.k))}</td>
            <td>${escapeHtml(String(row.seed))}</td>
            <td>${escapeHtml(row.value)}</td>
          </tr>
        `).join("")}
      </tbody>
    </table>
  `;
  popover.hidden = false;
  const rect = event.currentTarget.getBoundingClientRect();
  const width = Math.max(160, Math.min(560, window.innerWidth - 24));
  popover.style.width = `${width}px`;
  const top = Math.min(window.innerHeight - popover.offsetHeight - 12, rect.bottom + 8);
  const left = Math.min(window.innerWidth - width - 12, Math.max(12, rect.left));
  popover.style.top = `${Math.max(12, top)}px`;
  popover.style.left = `${left}px`;
}

function addTurbulenceMetricBreakdowns(row, strategy, config) {
  row.querySelectorAll("[data-turbulence-breakdown]").forEach((cell) => {
    const key = cell.dataset.turbulenceBreakdown;
    cell.addEventListener("click", (event) => showTurbulenceMetricBreakdown(event, strategy, key, config));
    cell.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        showTurbulenceMetricBreakdown(event, strategy, key, config);
      }
    });
  });
}

function renderTurbulenceStrategyMetrics(strategies, config, recommendation) {
  if (!strategies.length) {
    dom.turbulenceStrategyMetricsBody.innerHTML = '<tr><td colspan="10">Sin resultados todavía.</td></tr>';
    return;
  }
  const best = {
    success: bestTurbulenceIndexes(strategies, (strategy) => strategy.operatorMetrics?.successRate, "max"),
    coverage: bestTurbulenceIndexes(strategies, (strategy) => strategy.operatorMetrics?.coverageRate, "max"),
    similarity: bestTurbulenceIndexes(strategies, (strategy) => strategy.operatorMetrics?.averageSimilarity, "max"),
    time: bestTurbulenceIndexes(strategies, (strategy) => strategy.operatorMetrics?.operatorAverageSeconds, "min"),
    cost: bestTurbulenceIndexes(strategies, (strategy) => strategy.relativeOperatorCost, "min"),
    fidelity: bestTurbulenceIndexes(strategies, (strategy) => strategy.finalMetrics?.averageFidelity, "max"),
    fidelityDelta: bestTurbulenceIndexes(strategies, (strategy) => strategy.deltas?.averageFidelityDelta, "max"),
    diversity: bestTurbulenceIndexes(strategies, (strategy) => strategy.finalMetrics?.averageDiversity, "max"),
    diversityDelta: bestTurbulenceIndexes(strategies, (strategy) => strategy.deltas?.averageDiversityDelta, "max"),
  };
  dom.turbulenceStrategyMetricsBody.replaceChildren(
    ...strategies.map((strategy, index) => {
      const operator = strategy.operatorMetrics || {};
      const finalMetrics = strategy.finalMetrics || {};
      const cost = operator.cost || {};
      const deltas = strategy.deltas || {};
      const warnings = Array.isArray(strategy.warnings) ? strategy.warnings : [];
      const isWinner = recommendation?.strategyId === strategy.strategyId;
      const tr = document.createElement("tr");
      if (isWinner) tr.classList.add("is-nondominated-row");
      tr.innerHTML = `
        <td>${escapeHtml(strategy.displayName || strategy.strategyId)}${warnings.length ? `<br><small>${escapeHtml(warnings[0])}</small>` : ""}</td>
        ${turbulenceMetricCell(formatTurbulenceRate(operator.successRate), turbulenceBestClass(best.success, index), "success")}
        ${turbulenceMetricCell(formatTurbulenceRate(operator.coverageRate), turbulenceBestClass(best.coverage, index), "coverage")}
        ${turbulenceMetricCell(formatOptionalNumber(operator.averageSimilarity, 6), turbulenceBestClass(best.similarity, index), "similarity")}
        ${turbulenceMetricCell(`${formatDuration(Number(operator.operatorAverageSeconds || 0) * 1000)}<br><small>wall ${formatDuration(Number(operator.operatorWallClockSeconds || 0) * 1000)}; par ${operator.operatorParallelism ?? 1}</small>`, turbulenceBestClass(best.time, index), "time")}
        ${turbulenceMetricCell(`${escapeHtml(turbulenceCostLabel(cost))}<br><small>costo relativo ${formatOptionalNumber(strategy.relativeOperatorCost, 2)}</small>`, turbulenceBestClass(best.cost, index), "cost")}
        ${turbulenceMetricCell(formatOptionalNumber(finalMetrics.averageFidelity, 6), turbulenceBestClass(best.fidelity, index), "fidelity")}
        ${turbulenceMetricCell(formatSignedOptional(deltas.averageFidelityDelta), turbulenceBestClass(best.fidelityDelta, index), "fidelityDelta")}
        ${turbulenceMetricCell(formatOptionalNumber(finalMetrics.averageDiversity, 6), turbulenceBestClass(best.diversity, index), "diversity")}
        ${turbulenceMetricCell(formatSignedOptional(deltas.averageDiversityDelta), turbulenceBestClass(best.diversityDelta, index), "diversityDelta")}
      `;
      addTurbulenceMetricBreakdowns(tr, strategy, config);
      return tr;
    }),
  );
}

function renderTurbulenceMovementRows(strategies) {
  const rows = strategies.flatMap((strategy, strategyIndex) =>
    (strategy.movementRows || []).map((row) => ({
      ...row,
      strategyName: strategy.displayName || strategy.strategyId,
      strategyIndex,
    })),
  ).sort((a, b) =>
    String(a.individualId || "").localeCompare(String(b.individualId || ""), undefined, { numeric: true, sensitivity: "base" })
    || Number(a.movementNumber || 0) - Number(b.movementNumber || 0)
    || a.strategyIndex - b.strategyIndex
  );
  if (!rows.length) {
    dom.turbulenceMovementRowsBody.innerHTML = '<tr><td colspan="8">Sin movimientos todavía.</td></tr>';
    return;
  }
  dom.turbulenceMovementRowsBody.replaceChildren(
    ...rows.slice(0, 160).map((row) => {
      const tr = document.createElement("tr");
      const statusText = row.error ? `${row.status || "error"}: ${row.error}` : (row.status || "--");
      const hasLlmDiagnostics = Boolean(row.diagnostics?.llm);
      tr.classList.add("turbulence-movement-row", `turbulence-strategy-row-${row.strategyIndex % 6}`);
      if (!row.success) {
        tr.classList.add("turbulence-row-invalid");
      }
      if (!row.success || hasLlmDiagnostics) {
        tr.classList.add("is-clickable-diagnostic");
        tr.tabIndex = 0;
        tr.setAttribute("role", "button");
        tr.setAttribute("aria-label", `Ver diagnóstico de ${row.strategyName} para ${row.individualId || "individuo"}`);
        tr.addEventListener("click", () => openTurbulenceCandidateModal(row));
        tr.addEventListener("keydown", (event) => {
          if (event.key === "Enter" || event.key === " ") {
            event.preventDefault();
            openTurbulenceCandidateModal(row);
          }
        });
      }
      tr.innerHTML = `
        <td><span class="strategy-badge">${escapeHtml(row.strategyName)}</span></td>
        <td>${escapeHtml(String(row.movementNumber ?? "--"))}</td>
        <td>${escapeHtml(row.individualId || "--")}</td>
        <td>${escapeHtml(row.componentName || "--")}</td>
        <td class="context-cell long-cell"><div class="scroll-cell">${escapeHtml(row.current || "--")}</div></td>
        <td class="context-cell long-cell"><div class="scroll-cell">${escapeHtml(row.selectedCandidate || "--")}</div></td>
        <td>${formatOptionalNumber(row.similarity, 6)}</td>
        <td><span class="${row.success ? "valid" : "invalid"}">${escapeHtml(statusText)}</span></td>
      `;
      return tr;
    }),
  );
}

function openTurbulenceCandidateModal(row) {
  const candidates = Array.isArray(row.candidates) ? row.candidates : [];
  const validCandidates = candidates.filter((candidate) => candidate.valid);
  const statusDescription = turbulenceReasonDescription(row.status);
  dom.turbulenceCandidateModalTitle.textContent = `${row.strategyName || "Estrategia"} - ${row.individualId || "individuo"}`;
  dom.turbulenceCandidateModalBody.innerHTML = `
    <dl class="diagnostic-summary">
      <div><dt>Movimiento</dt><dd>${escapeHtml(String(row.movementNumber ?? "--"))}</dd></div>
      <div><dt>Componente</dt><dd>${escapeHtml(row.componentName || "--")}</dd></div>
      <div><dt>Estado</dt><dd><span class="invalid">${escapeHtml(row.status || "--")}</span><small>${escapeHtml(statusDescription)}</small></dd></div>
      <div><dt>Cobertura</dt><dd>${row.coverage ? "sí" : "no"}</dd></div>
      <div><dt>Candidatos crudos</dt><dd>${escapeHtml(String(row.rawCandidateCount ?? 0))}</dd></div>
      <div><dt>Candidatos válidos</dt><dd>${escapeHtml(String(validCandidates.length))}</dd></div>
    </dl>
    <div class="diagnostic-block">
      <h3>Componente original</h3>
      <p>${escapeHtml(row.current || "--")}</p>
    </div>
    ${row.error ? `
      <div class="diagnostic-block diagnostic-error">
        <h3>Error</h3>
        <p>${escapeHtml(row.error)}</p>
      </div>
    ` : ""}
    ${turbulenceLlmDiagnosticsBlock(row.diagnostics?.llm)}
    ${turbulenceLocalDiagnosticsBlock(row.diagnostics?.local)}
    <div class="diagnostic-block">
      <h3>Candidatos generados internamente</h3>
      ${candidates.length ? turbulenceCandidateDiagnosticsTable(candidates) : '<p>No se generaron candidatos parseables antes del filtro semántico.</p>'}
    </div>
  `;
  dom.turbulenceCandidateModal.hidden = false;
  dom.closeTurbulenceCandidateModalButton.focus();
}

function turbulenceLlmDiagnosticsBlock(llm) {
  if (!llm) return "";
  return `
    <div class="diagnostic-block">
      <h3>Solicitud LLM usada</h3>
      <dl class="diagnostic-summary diagnostic-summary-compact">
        <div><dt>Modelo</dt><dd>${escapeHtml(llm.model || "--")}</dd></div>
        <div><dt>API</dt><dd>${escapeHtml(llm.apiMode || "--")}</dd></div>
        <div><dt>Temperatura</dt><dd>${formatOptionalNumber(llm.temperature, 2)}</dd></div>
        <div><dt>Top p</dt><dd>${formatOptionalNumber(llm.topP, 2)}</dd></div>
        <div><dt>Máx. tokens</dt><dd>${escapeHtml(String(llm.maxTokens ?? "--"))}</dd></div>
      </dl>
      ${turbulencePromptCopyBlock("system", "System prompt exacto", llm.systemPrompt || "")}
      ${turbulencePromptCopyBlock("user", "User prompt exacto", llm.userPrompt || "")}
    </div>
  `;
}

function turbulenceLocalDiagnosticsBlock(local) {
  if (!local) return "";
  const selected = local.selectedUnit;
  const unitLabel = selected
    ? `${selected.text || "--"} (${selected.pos || "--"}, lemma ${selected.lemma || "--"})`
    : "--";
  const wordnet = local.wordnet || {};
  const ppdb = local.ppdb || {};
  const distilbert = local.distilbert || {};
  const candidateStage = local.candidateStage || {};
  return `
    <div class="diagnostic-block">
      <h3>Diagnóstico operador local</h3>
      <dl class="diagnostic-summary diagnostic-summary-compact">
        <div><dt>Semilla unidad</dt><dd>${escapeHtml(String(local.unitSelectionSeed ?? "--"))}</dd></div>
        <div><dt>Unidad seleccionada</dt><dd>${escapeHtml(unitLabel)}</dd></div>
        <div><dt>Unidades generales</dt><dd>${escapeHtml(String((local.unitsGeneral || []).length))}</dd></div>
        <div><dt>Unidades elegibles</dt><dd>${escapeHtml(String((local.unitsEligible || []).length))}</dd></div>
        ${local.unitsCompatible ? `<div><dt>Compatibles DistilBERT</dt><dd>${escapeHtml(String(local.unitsCompatible.length))}</dd></div>` : ""}
        <div><dt>Reemplazos</dt><dd>${escapeHtml(String(candidateStage.replacementCount ?? "--"))}</dd></div>
        <div><dt>Variantes C_K</dt><dd>${escapeHtml(String(candidateStage.candidateCount ?? "--"))}</dd></div>
      </dl>
      ${wordnet.query ? `
        <p><strong>WordNet:</strong> query ${escapeHtml(wordnet.query || "--")} / POS ${escapeHtml(wordnet.pos || "--")} / synsets ${escapeHtml(String(wordnet.synsetGroupCount ?? 0))}.</p>
        <p><strong>Reemplazos WordNet:</strong> ${escapeHtml(shortList(wordnet.replacements || []))}</p>
      ` : ""}
      ${ppdb.enabled !== undefined ? `
        <p><strong>PPDB:</strong> ${ppdb.enabled ? "activado" : "desactivado"}; ${ppdb.available ? "índice disponible" : "índice no disponible"}; claves ${escapeHtml(shortList(ppdb.lookupKeys || []))}.</p>
        <p><strong>Reemplazos PPDB:</strong> ${escapeHtml(shortList(ppdb.replacements || []))}</p>
      ` : ""}
      ${distilbert.maskedText ? `
        <p><strong>DistilBERT:</strong> top_k ${escapeHtml(String(distilbert.topK ?? "--"))}; máscara <code>${escapeHtml(distilbert.maskedText)}</code>.</p>
        <p><strong>Predicciones usadas:</strong> ${escapeHtml(shortList(distilbert.replacements || []))}</p>
      ` : ""}
    </div>
  `;
}

function shortList(values, limit = 12) {
  if (!Array.isArray(values) || !values.length) return "--";
  const visible = values.slice(0, limit).map((value) => String(value));
  const suffix = values.length > limit ? `, ... +${values.length - limit}` : "";
  return `${visible.join(", ")}${suffix}`;
}

function turbulencePromptCopyBlock(key, title, text) {
  return `
    <div class="diagnostic-prompt-block">
      <div class="diagnostic-prompt-header">
        <h4>${escapeHtml(title)}</h4>
        <button class="secondary diagnostic-copy-button" type="button" data-copy-diagnostic-prompt="${escapeHtml(key)}">Copiar</button>
      </div>
      <textarea class="diagnostic-prompt-text" data-diagnostic-prompt="${escapeHtml(key)}" readonly>${escapeHtml(text)}</textarea>
    </div>
  `;
}

function turbulenceCandidateDiagnosticsTable(candidates) {
  const rows = candidates.map((candidate) => `
    <tr class="${candidate.valid ? "candidate-valid-row" : "candidate-invalid-row"}">
      <td class="context-cell long-cell"><div class="scroll-cell">${escapeHtml(candidate.candidate || "--")}</div></td>
      <td>${escapeHtml(candidate.source || "--")}</td>
      <td>${formatOptionalNumber(candidate.similarity, 6)}</td>
      <td>${formatOptionalNumber(candidate.diversity, 6)}</td>
      <td>${turbulenceReasonMarkup(candidate.reason, candidate.valid)}</td>
    </tr>
  `).join("");
  return `
    <div class="table-wrap">
      <table class="diagnostic-candidates-table">
        <thead>
          <tr>
            <th>Candidato</th>
            <th>Fuente</th>
            <th>Similitud</th>
            <th>Diversidad</th>
            <th>Razón</th>
          </tr>
        </thead>
        <tbody>${rows}</tbody>
      </table>
    </div>
  `;
}

function turbulenceReasonMarkup(reason, isValid) {
  const labelClass = isValid ? "valid" : "invalid";
  return `
    <span class="${labelClass} reason-code">${escapeHtml(reason || "--")}</span>
    <small class="reason-description">${escapeHtml(turbulenceReasonDescription(reason))}</small>
  `;
}

function turbulenceReasonDescription(reason) {
  const descriptions = {
    valid: "El candidato pasó los filtros de forma, longitud y rango semántico de turbulencia.",
    ok: "El operador aplicó un candidato válido.",
    no_valid_candidate: "La estrategia generó candidatos, pero ninguno pasó todos los filtros configurados.",
    no_units: "No hubo tokens lingüísticos modificables para el tipo de componente seleccionado.",
    no_compatible_units: "Hubo tokens modificables, pero ninguno se representa como un único token WordPiece para DistilBERT.",
    no_candidates: "La estrategia no pudo construir variantes válidas después de filtrar reemplazos básicos.",
    semantic_range_failed: "La estrategia construyó variantes, pero ninguna quedó dentro del rango semántico de turbulencia configurado.",
    error: "La estrategia falló antes de completar la evaluación del movimiento.",
    empty: "El candidato está vacío.",
    line_break: "El candidato trae saltos de línea; se esperaba una sola unidad textual.",
    not_numbered_line: "La línea no cumple el formato numerado exigido por la estrategia, por eso solo se muestra como diagnóstico y no cuenta como candidato.",
    prompt_marker: "El candidato contiene marcas de prompt, código o etiquetas en vez de solo el componente.",
    contains_prompt_markup: "El candidato contiene marcas de prompt, código o etiquetas en vez de solo el componente.",
    literal_copy_current: "El candidato no cambia el componente original.",
    too_short: "El candidato tiene menos palabras que el mínimo configurado.",
    too_long: "El candidato tiene más palabras que el máximo configurado.",
    invalid_length: "El candidato no cumple los límites de longitud configurados.",
    length_drift: "La cantidad de palabras se aleja demasiado del componente original; se considera un cambio de alcance excesivo.",
    too_distant_from_current: "La similitud semántica contra el componente original quedó bajo el mínimo de turbulencia; el cambio es demasiado grande.",
    too_close_to_current: "La similitud semántica contra el componente original quedó sobre el máximo de turbulencia; el cambio es demasiado leve.",
    semantic_copy_target: "El candidato queda demasiado cerca del texto objetivo y se considera copia semántica.",
    insufficient_semantic_progress: "El candidato no mejora lo suficiente respecto a la referencia semántica configurada.",
  };
  return descriptions[reason] || "Motivo técnico no documentado; revisar candidato, similitud y configuración de filtros.";
}

function closeTurbulenceCandidateModal() {
  dom.turbulenceCandidateModal.hidden = true;
  dom.turbulenceCandidateModalBody.replaceChildren();
}

async function copyTextToClipboard(text) {
  if (navigator.clipboard?.writeText) {
    await navigator.clipboard.writeText(text);
    return;
  }
  const textarea = document.createElement("textarea");
  textarea.value = text;
  textarea.setAttribute("readonly", "");
  textarea.style.position = "fixed";
  textarea.style.opacity = "0";
  document.body.append(textarea);
  textarea.select();
  document.execCommand("copy");
  textarea.remove();
}

function renderTurbulenceLogs(logs) {
  if (!logs.length) {
    dom.turbulenceLogOutput.textContent = "Sin logs todavía.";
    return;
  }
  dom.turbulenceLogOutput.textContent = logs
    .slice(-80)
    .map((entry) => `[${entry.time || "--"}] ${entry.message || ""}`)
    .join("\n");
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
  if (dom.recomputeComparatorMetricsButton && isRunning) {
    dom.recomputeComparatorMetricsButton.disabled = true;
  }
  dom.clearComparatorButton.disabled = isRunning;
  if (dom.resumeComparatorButton) {
    dom.resumeComparatorButton.disabled = isRunning;
  }
  [
    dom.comparatorReferencePreset,
    dom.comparatorReferenceSaveLabel,
    dom.saveComparatorReferenceButton,
    dom.comparatorReferenceText,
    dom.comparatorModel,
    dom.comparatorTopK,
    dom.comparatorSeed,
    dom.comparatorRepetitions,
    dom.comparatorExecutionMode,
    dom.comparatorProposalParallelism,
    dom.comparatorTimeoutMinutes,
    dom.comparatorUpdateReposBeforeRun,
    dom.comparatorN,
    dom.comparatorGeneraciones,
  ].forEach((field) => {
    if (field) field.disabled = isRunning;
  });
  dom.comparatorProposalSelector.querySelectorAll("input, select, button").forEach((field) => {
    const card = field.closest("[data-proposal-catalog-card]");
    const unavailable = card?.dataset.available === "0";
    field.disabled = isRunning || unavailable;
  });
  dom.comparatorProposalConfigPanels.querySelectorAll("input, select, textarea, button").forEach((field) => {
    field.disabled = isRunning;
  });
  comparatorChoiceInstances.forEach((choice) => {
    if (isRunning) {
      choice.disable();
    } else {
      choice.enable();
    }
  });
  syncComparatorExecutionModeControls(isRunning);
  if (!isRunning) {
    syncComparatorRecomputeButton(latestComparatorRun);
  }
}

function syncComparatorExecutionModeControls(isRunning = false) {
  const mode = dom.comparatorExecutionMode?.value || "fair_sequential";
  const fair = mode === "fair_sequential";
  if (fair && dom.comparatorProposalParallelism) {
    dom.comparatorProposalParallelism.value = "1";
  }
  if (dom.comparatorProposalParallelism) {
    dom.comparatorProposalParallelism.disabled = isRunning || fair;
  }
  if (dom.comparatorExecutionModeNote) {
    dom.comparatorExecutionModeNote.textContent = fair
      ? "Costos comparables: las propuestas se ejecutan una por una."
      : "Modo exploratorio: metricas visibles, pero costos no comparables por recursos compartidos.";
  }
}

function stopComparatorPolling() {
  if (comparatorPollTimer) {
    window.clearInterval(comparatorPollTimer);
    comparatorPollTimer = null;
  }
}

function loadStoredComparatorRunId() {
  try {
    return window.localStorage.getItem(COMPARATOR_LAST_RUN_ID_STORAGE_KEY) || "";
  } catch (_error) {
    return "";
  }
}

function storeComparatorRunId(runId) {
  if (!runId) return;
  if (dom.comparatorResumeRunId) {
    dom.comparatorResumeRunId.value = runId;
  }
  try {
    window.localStorage.setItem(COMPARATOR_LAST_RUN_ID_STORAGE_KEY, runId);
  } catch (_error) {
    // Local storage is optional; the visible input still supports manual reattach.
  }
}

function clearStoredComparatorRunId() {
  if (dom.comparatorResumeRunId) {
    dom.comparatorResumeRunId.value = "";
  }
  try {
    window.localStorage.removeItem(COMPARATOR_LAST_RUN_ID_STORAGE_KEY);
  } catch (_error) {
    // Nothing to clear when local storage is unavailable.
  }
}

function syncComparatorRecomputeButton(run = null) {
  if (!dom.recomputeComparatorMetricsButton) return;
  const status = run?.metricRecomputeStatus || {};
  const available = Boolean(status.available);
  const recommended = Boolean(status.recommended);
  dom.recomputeComparatorMetricsButton.disabled = !available;
  dom.recomputeComparatorMetricsButton.textContent = recommended
    ? "Recalcular metricas recomendado"
    : "Recalcular metricas";
  dom.recomputeComparatorMetricsButton.title = available
    ? "Reconstruye metricas y graficos desde los artefactos Python reales, sin reejecutar propuestas ni LLM."
    : "Disponible solo para corridas completadas.";
}

function setComparatorLogCopyButton(hasLogs) {
  if (!dom.copyComparatorLogButton) return;
  dom.copyComparatorLogButton.disabled = !hasLogs;
  dom.copyComparatorLogButton.textContent = "Copiar log";
}

async function copyComparatorLog() {
  const text = dom.comparatorLogOutput.textContent || "";
  if (!text.trim() || text === "Sin logs todavia.") {
    setComparatorLogCopyButton(false);
    return;
  }
  dom.copyComparatorLogButton.disabled = true;
  try {
    await copyTextToClipboard(text);
    dom.copyComparatorLogButton.textContent = "Copiado";
  } catch (error) {
    dom.copyComparatorLogButton.textContent = "Error";
  } finally {
    window.setTimeout(() => {
      const hasLogs = (dom.comparatorLogOutput.textContent || "").trim() !== ""
        && dom.comparatorLogOutput.textContent !== "Sin logs todavia.";
      setComparatorLogCopyButton(hasLogs);
    }, 1200);
  }
}

function resetComparatorUi(options = {}) {
  const clearStoredRunId = options.clearStoredRunId ?? true;
  stopComparatorPolling();
  currentComparatorRunId = null;
  comparatorPollFailureCount = 0;
  latestComparatorRun = null;
  if (clearStoredRunId) {
    clearStoredComparatorRunId();
  }
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
  dom.comparatorResultsBody.innerHTML = '<tr><td colspan="9">Sin resultados todavia.</td></tr>';
  dom.comparatorParetoCharts.replaceChildren();
  if (dom.comparatorChartProposalFilters) {
    dom.comparatorChartProposalFilters.innerHTML = '<span class="muted-text">Ejecuta una comparacion para activar filtros.</span>';
  }
  dom.comparatorCombinedParetoChart.innerHTML = "";
  dom.comparatorGlobalNonDominatedChart.innerHTML = "";
  dom.comparatorHvChart.innerHTML = "";
  dom.comparatorNonDominatedChart.innerHTML = "";
  dom.comparatorSpreadChart.innerHTML = "";
  dom.comparatorGlobalInertiaChart.innerHTML = "";
  dom.comparatorGlobalEntropyChart.innerHTML = "";
  renderComparatorCostDetails(null);
  comparatorChartSignature = "";
  comparatorChartFilterIds = new Set();
  comparatorChartFilterSignature = "";
  disposeComparatorCharts();
  dom.comparatorProposalCards.innerHTML = `
    <article class="proposal-card">
      <strong>Sin corrida</strong>
      <p>Ejecuta el comparador para ver los resumenes.</p>
    </article>
  `;
  resetComparatorLogLoader(null);
  dom.comparatorLogOutput.textContent = "Sin logs todavia.";
  setComparatorLogCopyButton(false);
  renderComparatorProgress(null);
  setComparatorRunning(false);
  syncComparatorRecomputeButton(null);
  setStatus(
    dom.comparatorStatusTone,
    dom.comparatorStatusTitle,
    dom.comparatorStatusDetail,
    "Listo",
    "Ejecuta las propuestas seleccionadas desde sus repos Python usando Ollama local.",
  );
}

function disposeComparatorCharts() {
  comparatorCharts.forEach((chart) => chart.dispose());
  comparatorCharts = [];
}

function disposeComparatorChoices() {
  comparatorChoiceInstances.forEach((choice) => choice.destroy());
  comparatorChoiceInstances = [];
}

function comparatorGitStatusText(proposal) {
  const snapshot = proposal.git?.snapshot || {};
  const expectedRemote = proposal.git?.expectedRemoteUrl ? `; esperado ${proposal.git.expectedRemoteUrl}` : "";
  if (!snapshot.isGit) {
    return snapshot.error ? `No Git: ${snapshot.error}${expectedRemote}` : `No Git${expectedRemote}`;
  }
  const dirty = snapshot.dirty ? `; ${snapshot.dirtyCount || 0} cambio(s) local(es)` : "";
  const remoteUrl = snapshot.remoteUrl ? `; ${snapshot.remoteUrl}` : "";
  return `Local ${snapshot.branch || "--"} @ ${snapshot.shortCommit || "--"}${dirty}${remoteUrl}${expectedRemote}`;
}

function comparatorEntityId(item) {
  return String(item?.instanceId || item?.proposalId || "");
}

function comparatorProposalById(proposalId) {
  return comparatorProposals.find((proposal) => proposal.proposalId === proposalId) || null;
}

function comparatorDefaultInstanceName(proposalId) {
  const proposal = comparatorProposalById(proposalId);
  const baseName = proposal?.displayName || proposalId;
  const count = comparatorInstances.filter((instance) => instance.proposalId === proposalId).length + 1;
  return `${baseName} - config ${count}`;
}

function comparatorNextInstanceId(proposalId) {
  let index = comparatorInstances.filter((instance) => instance.proposalId === proposalId).length + 1;
  let candidate = `${proposalId}-${index}`;
  const existing = new Set(comparatorInstances.map((instance) => instance.instanceId));
  while (existing.has(candidate)) {
    index += 1;
    candidate = `${proposalId}-${index}`;
  }
  return candidate;
}

function comparatorDefaultInstances(proposals) {
  return proposals
    .filter((proposal) => proposal.available)
    .map((proposal, index) => ({
      instanceId: proposal.proposalId,
      proposalId: proposal.proposalId,
      displayName: proposal.displayName,
      proposalConfig: { extraArgs: "", cliValues: {} },
      orderIndex: index,
    }));
}

function normalizeComparatorInstance(instance, index = 0) {
  const proposal = comparatorProposalById(instance.proposalId);
  const proposalId = proposal?.proposalId || instance.proposalId;
  return {
    instanceId: instance.instanceId || comparatorNextInstanceId(proposalId),
    proposalId,
    displayName: String(instance.displayName || comparatorDefaultInstanceName(proposalId)).trim(),
    proposalConfig: {
      extraArgs: "",
      ...(instance.proposalConfig || {}),
      cliValues: { ...((instance.proposalConfig || {}).cliValues || {}) },
    },
    orderIndex: index,
  };
}

function ensureComparatorInstances() {
  const availableIds = new Set(comparatorProposals.filter((proposal) => proposal.available).map((proposal) => proposal.proposalId));
  comparatorInstances = comparatorInstances
    .filter((instance) => availableIds.has(instance.proposalId))
    .map((instance, index) => normalizeComparatorInstance(instance, index));
  if (!comparatorInstancesInitialized && !comparatorInstances.length) {
    comparatorInstances = comparatorDefaultInstances(comparatorProposals);
  }
  comparatorInstancesInitialized = true;
}

function comparatorProposalInstanceCount(proposalId) {
  return comparatorInstances.filter((instance) => instance.proposalId === proposalId).length;
}

function syncComparatorProposalCatalogSelectionStates() {
  if (!dom.comparatorProposalSelector) return;
  dom.comparatorProposalSelector.querySelectorAll("[data-proposal-catalog-card]").forEach((card) => {
    const proposalId = card.dataset.proposalCatalogCard;
    const selectedCount = comparatorProposalInstanceCount(proposalId);
    card.dataset.selected = selectedCount ? "1" : "0";
    const selectionLabel = card.querySelector("[data-comparator-proposal-selection]");
    if (selectionLabel) {
      selectionLabel.textContent = selectedCount
        ? `Seleccionada: ${selectedCount} configuracion(es)`
        : "No seleccionada";
    }
    const addButton = card.querySelector("[data-add-comparator-instance]");
    if (addButton) {
      addButton.textContent = selectedCount ? "Agregar otra configuracion" : "Seleccionar propuesta";
    }
    const removeButton = card.querySelector("[data-remove-comparator-proposal]");
    if (removeButton) {
      removeButton.hidden = selectedCount === 0;
    }
  });
}

function renderComparatorProposalControls(proposals) {
  comparatorProposals = proposals || [];
  if (!dom.comparatorProposalSelector || !dom.comparatorProposalConfigPanels) return;
  disposeComparatorChoices();
  ensureComparatorInstances();
  dom.comparatorProposalSelector.replaceChildren(
    ...comparatorProposals.map((proposal) => {
      const selectedCount = comparatorProposalInstanceCount(proposal.proposalId);
      const card = document.createElement("article");
      card.className = "proposal-catalog-card";
      card.dataset.proposalCatalogCard = proposal.proposalId;
      card.dataset.available = proposal.available ? "1" : "0";
      card.dataset.selected = selectedCount ? "1" : "0";
      const missing = (proposal.dependencyStatus?.missing || []).join(", ");
      const availabilityDetail = proposal.available
        ? "Disponible"
        : missing
          ? `No disponible: faltan ${missing}`
          : "No disponible";
      const git = proposal.git || {};
      const gitStatus = comparatorGitStatusText(proposal);
      card.innerHTML = `
        <div class="proposal-catalog-head">
          <span class="proposal-catalog-status ${proposal.available ? "is-available" : "is-unavailable"}" aria-hidden="true"></span>
          <div>
            <strong>${escapeHtml(proposal.displayName)}</strong>
            <small>${escapeHtml(availabilityDetail)} - ${escapeHtml((proposal.objectiveNames || []).join(", "))}</small>
            <small data-comparator-proposal-selection>${selectedCount ? `Seleccionada: ${selectedCount} configuracion(es)` : "No seleccionada"}</small>
          </div>
        </div>
        <p>${escapeHtml(proposal.description || "")}</p>
        <div class="proposal-git-config compact">
          <div class="proposal-config-section-title">
            <strong>Git</strong>
            <small>${escapeHtml(gitStatus)}</small>
          </div>
          <div class="proposal-git-fields">
            <label>
              <span>Remote</span>
              <input type="text" value="${escapeHtml(git.remote || "origin")}" data-comparator-git-remote data-proposal-id="${escapeHtml(proposal.proposalId)}" autocomplete="off">
            </label>
            <label>
              <span>Branch</span>
              <input type="text" value="${escapeHtml(git.branch || "")}" data-comparator-git-branch data-proposal-id="${escapeHtml(proposal.proposalId)}" autocomplete="off">
            </label>
            <label>
              <span>Modo pull</span>
              <select data-comparator-git-pull-mode data-proposal-id="${escapeHtml(proposal.proposalId)}">
                <option value="ff-only" ${git.pullMode === "ff-only" ? "selected" : ""}>Fast-forward only</option>
              </select>
            </label>
          </div>
        </div>
        <div class="proposal-catalog-actions">
          <button type="button" class="secondary" data-add-comparator-instance="${escapeHtml(proposal.proposalId)}" ${proposal.available ? "" : "disabled"}>${selectedCount ? "Agregar otra configuracion" : "Seleccionar propuesta"}</button>
          <button type="button" class="danger" data-remove-comparator-proposal="${escapeHtml(proposal.proposalId)}" ${selectedCount ? "" : "hidden"} ${proposal.available ? "" : "disabled"}>Deseleccionar</button>
        </div>
      `;
      return card;
    }),
  );
  renderComparatorInstanceList();
}

function renderComparatorInstanceList() {
  if (!dom.comparatorProposalConfigPanels) return;
  if (!comparatorInstances.length) {
    dom.comparatorProposalConfigPanels.innerHTML = `
      <article class="proposal-config-card comparator-instance-empty">
        <strong>Sin instancias configuradas</strong>
        <p>Agrega una configuracion desde el catalogo de propuestas para ejecutar el comparador.</p>
      </article>
    `;
    return;
  }
  dom.comparatorProposalConfigPanels.replaceChildren(
    ...comparatorInstances.map((instance) => {
      const proposal = comparatorProposalById(instance.proposalId);
      const article = document.createElement("article");
      article.className = "proposal-config-card comparator-instance-card";
      article.dataset.comparatorInstanceCard = instance.instanceId;
      article.dataset.proposalId = instance.proposalId;
      article.innerHTML = `
        <div>
          <span class="eyebrow">${escapeHtml(proposal?.displayName || instance.proposalId)}</span>
          <h3>${escapeHtml(instance.displayName || instance.instanceId)}</h3>
          <p>${escapeHtml(comparatorInstanceConfigSummary(instance))}</p>
        </div>
        <div class="instance-card-actions">
          <button type="button" class="secondary" data-edit-comparator-instance="${escapeHtml(instance.instanceId)}">Editar</button>
          <button type="button" class="secondary" data-duplicate-comparator-instance="${escapeHtml(instance.instanceId)}">Duplicar</button>
          <button type="button" class="danger" data-delete-comparator-instance="${escapeHtml(instance.instanceId)}">Eliminar</button>
        </div>
      `;
      return article;
    }),
  );
}

function comparatorInstanceConfigSummary(instance) {
  const values = instance.proposalConfig?.cliValues || {};
  const keys = Object.keys(values);
  if (!keys.length && !instance.proposalConfig?.extraArgs) {
    return "Configuracion por defecto de la propuesta.";
  }
  const preview = keys.slice(0, 4).map((key) => `${key}=${comparatorPreviewValue(values[key])}`);
  const remaining = keys.length > preview.length ? `; +${keys.length - preview.length} cambio(s)` : "";
  return `${preview.join("; ")}${remaining}`;
}

function comparatorPreviewValue(value) {
  if (Array.isArray(value)) return `[${value.join(", ")}]`;
  if (value && typeof value === "object") return JSON.stringify(value);
  return String(value);
}

function enhanceComparatorSelects(container = dom.comparatorProposalConfigPanels) {
  if (!window.Choices) return;
  container.querySelectorAll("select.cli-select").forEach((select) => {
    comparatorChoiceInstances.push(new window.Choices(select, {
      allowHTML: false,
      searchEnabled: false,
      itemSelectText: "",
      shouldSort: false,
    }));
  });
}

function comparatorConfigurableOptions(proposal) {
  return (proposal.cliOptions || []).filter((option) => option.source !== "managed" && option.source !== "common");
}

function comparatorOptionKey(option) {
  return String(option.key || option.configPath || option.flag || "");
}

function comparatorOptionDescriptor(option) {
  return option.configPath ? `${option.flag} ${option.configPath}` : String(option.flag || comparatorOptionKey(option));
}

function comparatorOptionLabel(option) {
  if (option.label) return String(option.label);
  if (option.configPath) return String(option.configPath).split(".").at(-1).replace(/_/g, " ");
  return String(option.flag || "").replace(/^--/, "").replace(/-/g, " ");
}

function comparatorOptionInputId(proposal, option, suffix = "") {
  const raw = `${proposal.proposalId}-${comparatorOptionKey(option)}-${suffix}`;
  return `cli-${raw.replace(/[^A-Za-z0-9_-]+/g, "-")}`;
}

function comparatorOptionHelp(option) {
  const details = [];
  if (option.configPath) details.push(option.configPath);
  else if (option.flag) details.push(option.flag);
  if (option.default !== undefined) {
    const defaultText = typeof option.default === "string" ? option.default : JSON.stringify(option.default);
    details.push(`default: ${defaultText}`);
  }
  if (option.help) details.push(option.help);
  return details.join(" - ");
}

function renderComparatorCliFields(proposal, options) {
  if (proposal.kind !== "binary-mopso-cd") {
    return options.map((option) => renderComparatorCliOption(proposal, option));
  }
  const standalone = options.filter((option) => !option.configPath);
  const groupedOptions = options.filter((option) => option.configPath);
  const nodes = standalone.map((option) => renderComparatorCliOption(proposal, option));
  const groups = new Map();
  groupedOptions.forEach((option) => {
    const group = option.group || String(option.configPath || "").split(".")[0] || "config";
    if (!groups.has(group)) groups.set(group, []);
    groups.get(group).push(option);
  });
  Array.from(groups.entries())
    .sort(([left], [right]) => left.localeCompare(right))
    .forEach(([group, groupOptions]) => {
      const details = document.createElement("details");
      details.className = "proposal-config-group";
      details.innerHTML = `
        <summary>
          <span>
            <strong>${escapeHtml(group)}</strong>
            <small>${escapeHtml(groupOptions.length)} parametro(s) desde default.yaml</small>
          </span>
          <b aria-hidden="true"></b>
        </summary>
        <div class="proposal-config-group-body"></div>
      `;
      details.querySelector(".proposal-config-group-body").replaceChildren(
        ...groupOptions.map((option) => renderComparatorCliOption(proposal, option)),
      );
      nodes.push(details);
    });
  return nodes;
}

function renderComparatorCliOption(proposal, option) {
  const type = option.type || "string";
  if (type === "yaml") return renderComparatorYamlOption(proposal, option);
  if (type === "bool") return renderComparatorBoolOption(proposal, option);
  if (type === "multi_select") return renderComparatorMultiSelectOption(proposal, option);
  if (type === "repeatable_assignment_bool") return renderComparatorAssignmentBoolOption(proposal, option);
  if (type === "repeatable_assignment") return renderComparatorAssignmentOption(proposal, option);
  if (type === "repeatable") return renderComparatorRepeatableOption(proposal, option);
  return renderComparatorScalarOption(proposal, option);
}

function renderComparatorScalarOption(proposal, option) {
  const wrapper = document.createElement("label");
  wrapper.className = "cli-field";
  const inputId = comparatorOptionInputId(proposal, option);
  const optionKey = comparatorOptionKey(option);
  const isNumber = option.type === "int" || option.type === "float";
  const inputType = isNumber ? "number" : "text";
  const datalistId = option.choices?.length ? `${inputId}-choices` : "";
  const defaultValue = option.default === undefined ? "" : String(option.default);
  wrapper.innerHTML = `
    <span>${escapeHtml(comparatorOptionLabel(option))}</span>
    <input id="${escapeHtml(inputId)}" type="${inputType}" data-comparator-cli-value data-proposal-id="${escapeHtml(proposal.proposalId)}" data-cli-key="${escapeHtml(optionKey)}" data-cli-flag="${escapeHtml(option.flag)}" data-cli-type="${escapeHtml(option.type || "string")}" ${datalistId ? `list="${escapeHtml(datalistId)}"` : ""} ${option.min !== undefined ? `min="${escapeHtml(option.min)}"` : ""} ${option.max !== undefined ? `max="${escapeHtml(option.max)}"` : ""} ${option.step !== undefined ? `step="${escapeHtml(option.step)}"` : isNumber ? 'step="any"' : ""} placeholder="${escapeHtml(defaultValue)}">
    ${datalistId ? `<datalist id="${escapeHtml(datalistId)}">${option.choices.map((choice) => `<option value="${escapeHtml(choice)}"></option>`).join("")}</datalist>` : ""}
    <small>${escapeHtml(comparatorOptionHelp(option))}</small>
  `;
  return wrapper;
}

function renderComparatorYamlOption(proposal, option) {
  const wrapper = document.createElement("label");
  wrapper.className = "cli-field cli-yaml-field";
  const inputId = comparatorOptionInputId(proposal, option);
  const optionKey = comparatorOptionKey(option);
  const defaultValue = option.default === undefined ? "" : JSON.stringify(option.default);
  wrapper.innerHTML = `
    <span>${escapeHtml(comparatorOptionLabel(option))}</span>
    <textarea id="${escapeHtml(inputId)}" data-comparator-cli-value data-proposal-id="${escapeHtml(proposal.proposalId)}" data-cli-key="${escapeHtml(optionKey)}" data-cli-flag="${escapeHtml(option.flag)}" data-cli-type="yaml" placeholder="${escapeHtml(defaultValue)}"></textarea>
    <small>${escapeHtml(comparatorOptionHelp(option))}</small>
  `;
  return wrapper;
}

function renderComparatorBoolOption(proposal, option) {
  if (option.allowFalse) {
    return renderComparatorBoolSelectOption(proposal, option);
  }
  const label = document.createElement("label");
  label.className = "cli-field cli-field-checkbox";
  const optionKey = comparatorOptionKey(option);
  label.innerHTML = `
    <input type="checkbox" data-comparator-cli-value data-proposal-id="${escapeHtml(proposal.proposalId)}" data-cli-key="${escapeHtml(optionKey)}" data-cli-flag="${escapeHtml(option.flag)}" data-cli-type="bool">
    <span>
      <strong>${escapeHtml(comparatorOptionLabel(option))}</strong>
      <small>${escapeHtml(comparatorOptionHelp(option))}</small>
    </span>
  `;
  return label;
}

function renderComparatorBoolSelectOption(proposal, option) {
  const label = document.createElement("label");
  label.className = "cli-field";
  const optionKey = comparatorOptionKey(option);
  label.innerHTML = `
    <span>${escapeHtml(comparatorOptionLabel(option))}</span>
    <select class="cli-select" data-comparator-cli-value data-proposal-id="${escapeHtml(proposal.proposalId)}" data-cli-key="${escapeHtml(optionKey)}" data-cli-flag="${escapeHtml(option.flag)}" data-cli-type="bool">
      <option value="">Sin cambio</option>
      <option value="true">true</option>
      <option value="false">false</option>
    </select>
    <small>${escapeHtml(comparatorOptionHelp(option))}</small>
  `;
  return label;
}

function renderComparatorMultiSelectOption(proposal, option) {
  const fieldset = document.createElement("fieldset");
  fieldset.className = "cli-fieldset cli-choice-grid";
  fieldset.innerHTML = `<legend>${escapeHtml(comparatorOptionLabel(option))}</legend>`;
  const choices = Array.isArray(option.choices) ? option.choices : [];
  const optionKey = comparatorOptionKey(option);
  fieldset.append(
    ...choices.map((choice) => {
      const label = document.createElement("label");
      label.className = "cli-field-checkbox";
      label.innerHTML = `
        <input type="checkbox" value="${escapeHtml(choice)}" data-comparator-cli-multi data-proposal-id="${escapeHtml(proposal.proposalId)}" data-cli-key="${escapeHtml(optionKey)}" data-cli-flag="${escapeHtml(option.flag)}">
        <span>${escapeHtml(choice)}</span>
      `;
      return label;
    }),
  );
  const note = document.createElement("small");
  note.textContent = comparatorOptionHelp(option);
  fieldset.append(note);
  return fieldset;
}

function renderComparatorAssignmentBoolOption(proposal, option) {
  const fieldset = document.createElement("fieldset");
  fieldset.className = "cli-fieldset cli-assignment-fieldset";
  fieldset.innerHTML = `<legend>${escapeHtml(comparatorOptionLabel(option))}</legend>`;
  fieldset.append(
    ...comparatorOptionAssignments(option).map((assignment) => {
      const label = document.createElement("label");
      label.className = "cli-assignment-row";
      const optionKey = comparatorOptionKey(option);
      label.innerHTML = `
        <span>${escapeHtml(assignment.label || assignment.name)}</span>
        <select class="cli-select" data-comparator-cli-assignment data-assignment-kind="bool" data-proposal-id="${escapeHtml(proposal.proposalId)}" data-cli-key="${escapeHtml(optionKey)}" data-cli-flag="${escapeHtml(option.flag)}" data-assignment-name="${escapeHtml(assignment.name)}">
          <option value="">Sin cambio</option>
          <option value="true">Activar</option>
          <option value="false">Desactivar</option>
        </select>
      `;
      return label;
    }),
  );
  return fieldset;
}

function renderComparatorAssignmentOption(proposal, option) {
  const fieldset = document.createElement("fieldset");
  fieldset.className = "cli-fieldset cli-assignment-fieldset";
  fieldset.innerHTML = `<legend>${escapeHtml(comparatorOptionLabel(option))}</legend>`;
  fieldset.append(
    ...comparatorOptionAssignments(option).map((assignment) => {
      const label = document.createElement("label");
      label.className = "cli-assignment-row";
      const optionKey = comparatorOptionKey(option);
      label.innerHTML = `
        <span>${escapeHtml(assignment.label || assignment.name)}</span>
        <input type="text" data-comparator-cli-assignment data-assignment-kind="string" data-proposal-id="${escapeHtml(proposal.proposalId)}" data-cli-key="${escapeHtml(optionKey)}" data-cli-flag="${escapeHtml(option.flag)}" data-assignment-name="${escapeHtml(assignment.name)}" placeholder="Modelo opcional">
      `;
      return label;
    }),
  );
  return fieldset;
}

function renderComparatorRepeatableOption(proposal, option) {
  const fieldset = document.createElement("fieldset");
  fieldset.className = "cli-fieldset cli-repeat-fieldset";
  fieldset.innerHTML = `<legend>${escapeHtml(comparatorOptionLabel(option))}</legend>`;
  const optionKey = comparatorOptionKey(option);
  for (let index = 0; index < 3; index += 1) {
    const label = document.createElement("label");
    label.className = "cli-field";
    label.innerHTML = `
      <span>Valor ${index + 1}</span>
      <input type="text" data-comparator-cli-repeat data-proposal-id="${escapeHtml(proposal.proposalId)}" data-cli-key="${escapeHtml(optionKey)}" data-cli-flag="${escapeHtml(option.flag)}">
    `;
    fieldset.append(label);
  }
  return fieldset;
}

function comparatorOptionAssignments(option) {
  return Array.isArray(option.assignments) ? option.assignments : [];
}

function selectedComparatorProposalIds() {
  return Array.from(new Set(comparatorInstances.map((instance) => instance.proposalId).filter(Boolean)));
}

function comparatorProposalConfigs() {
  const configs = {};
  comparatorInstances.forEach((instance) => {
    configs[instance.proposalId] = configs[instance.proposalId] || instance.proposalConfig || { cliValues: {} };
  });
  return configs;
}

function comparatorProposalInstancesPayload() {
  return comparatorInstances.map((instance, index) => ({
    instanceId: instance.instanceId,
    proposalId: instance.proposalId,
    displayName: instance.displayName,
    proposalConfig: instance.proposalConfig || { extraArgs: "", cliValues: {} },
    orderIndex: index,
  }));
}

function collectComparatorCliValues(container) {
  const cliValues = {};
  container.querySelectorAll("[data-comparator-cli-value]").forEach((field) => {
    const key = field.dataset.cliKey || field.dataset.cliFlag;
    if (!key) return;
    const type = field.dataset.cliType || "string";
    let value;
    if (type === "bool" && field.tagName === "SELECT") {
      value = field.value;
    } else if (type === "bool") {
      value = field.checked;
    } else {
      value = field.value.trim();
    }
    const isBoolSelect = type === "bool" && field.tagName === "SELECT";
    const shouldInclude = isBoolSelect ? value !== "" : type === "bool" ? value : value !== "";
    if (shouldInclude) {
      cliValues[key] = type === "bool" && typeof value === "string" ? value === "true" : value;
    }
  });
  container.querySelectorAll("[data-comparator-cli-multi]").forEach((field) => {
    if (!field.checked) return;
    const key = field.dataset.cliKey || field.dataset.cliFlag;
    if (!key) return;
    const values = cliValues[key] || [];
    values.push(field.value);
    cliValues[key] = values;
  });
  container.querySelectorAll("[data-comparator-cli-repeat]").forEach((field) => {
    const key = field.dataset.cliKey || field.dataset.cliFlag;
    const value = field.value.trim();
    if (!key || !value) return;
    const values = cliValues[key] || [];
    values.push(value);
    cliValues[key] = values;
  });
  container.querySelectorAll("[data-comparator-cli-assignment]").forEach((field) => {
    const key = field.dataset.cliKey || field.dataset.cliFlag;
    const assignmentName = field.dataset.assignmentName;
    const value = field.value.trim();
    if (!key || !assignmentName || value === "") return;
    const values = cliValues[key] || {};
    values[assignmentName] = field.dataset.assignmentKind === "bool" ? value === "true" : value;
    cliValues[key] = values;
  });
  return cliValues;
}

function applyComparatorCliValues(container, cliValues = {}) {
  container.querySelectorAll("[data-comparator-cli-value]").forEach((field) => {
    const key = field.dataset.cliKey || field.dataset.cliFlag;
    if (!key || !(key in cliValues)) return;
    const value = cliValues[key];
    const type = field.dataset.cliType || "string";
    if (type === "bool" && field.tagName === "SELECT") {
      field.value = value === true ? "true" : value === false ? "false" : "";
    } else if (type === "bool") {
      field.checked = Boolean(value);
    } else {
      field.value = String(value);
    }
  });
  container.querySelectorAll("[data-comparator-cli-multi]").forEach((field) => {
    const key = field.dataset.cliKey || field.dataset.cliFlag;
    const values = Array.isArray(cliValues[key]) ? cliValues[key].map(String) : [];
    field.checked = values.includes(field.value);
  });
  const repeatPositions = new Map();
  container.querySelectorAll("[data-comparator-cli-repeat]").forEach((field) => {
    const key = field.dataset.cliKey || field.dataset.cliFlag;
    const values = Array.isArray(cliValues[key]) ? cliValues[key] : [];
    const position = repeatPositions.get(key) || 0;
    field.value = values[position] === undefined ? "" : String(values[position]);
    repeatPositions.set(key, position + 1);
  });
  container.querySelectorAll("[data-comparator-cli-assignment]").forEach((field) => {
    const key = field.dataset.cliKey || field.dataset.cliFlag;
    const assignmentName = field.dataset.assignmentName;
    const values = cliValues[key] && typeof cliValues[key] === "object" ? cliValues[key] : {};
    const value = values[assignmentName];
    if (value === undefined) {
      field.value = "";
    } else if (field.dataset.assignmentKind === "bool") {
      field.value = value ? "true" : "false";
    } else {
      field.value = String(value);
    }
  });
}

function openComparatorInstanceModal(proposalId, instanceId = null, duplicate = false) {
  const proposal = comparatorProposalById(proposalId);
  if (!proposal || !proposal.available) return;
  disposeComparatorChoices();
  const existing = instanceId ? comparatorInstances.find((instance) => instance.instanceId === instanceId) : null;
  const draft = existing
    ? normalizeComparatorInstance({
        ...existing,
        instanceId: duplicate ? comparatorNextInstanceId(proposalId) : existing.instanceId,
        displayName: duplicate ? `${existing.displayName} copia` : existing.displayName,
        proposalConfig: {
          extraArgs: "",
          cliValues: { ...((existing.proposalConfig || {}).cliValues || {}) },
        },
      })
    : normalizeComparatorInstance({
        instanceId: comparatorNextInstanceId(proposalId),
        proposalId,
        displayName: comparatorDefaultInstanceName(proposalId),
        proposalConfig: { extraArgs: "", cliValues: {} },
      });
  comparatorInstanceModalState = {
    mode: existing && !duplicate ? "edit" : "add",
    originalInstanceId: existing && !duplicate ? existing.instanceId : null,
    draft,
  };

  const configurableOptions = comparatorConfigurableOptions(proposal);
  const globalFlags = (proposal.cliOptions || [])
    .filter((option) => option.source === "common")
    .map((option) => comparatorOptionDescriptor(option))
    .join(", ");
  const managedFlags = (proposal.cliOptions || [])
    .filter((option) => option.source === "managed")
    .map((option) => comparatorOptionDescriptor(option))
    .join(", ");
  dom.comparatorInstanceModalTitle.textContent = existing && !duplicate
    ? `Editar ${draft.displayName}`
    : `Agregar ${proposal.displayName}`;
  dom.comparatorInstanceModalSubtitle.textContent = "Los parametros comunes N, G, semilla, K, referencia y modelo quedan fuera de esta configuracion.";
  dom.saveComparatorInstanceModalButton.hidden = false;
  dom.saveComparatorInstanceModalButton.disabled = false;
  dom.saveComparatorInstanceModalButton.textContent = "Guardar configuracion";
  dom.cancelComparatorInstanceModalButton.textContent = "Cancelar";
  dom.comparatorInstanceModalBody.innerHTML = `
    <label class="cli-field">
      <span>Nombre de instancia</span>
      <input type="text" data-comparator-instance-name value="${escapeHtml(draft.displayName)}" autocomplete="off">
      <small>Este nombre aparece en columnas, filtros, logs y graficos.</small>
    </label>
    <div class="proposal-config-runtime">
      <small>${escapeHtml(proposal.repositoryPath || "")}</small>
      <small>Python: ${escapeHtml(proposal.pythonExecutable || "--")}</small>
    </div>
    <div class="proposal-cli-fields" data-instance-cli-fields></div>
    <div class="proposal-config-footnotes">
      <small>Globales: ${escapeHtml(globalFlags || "ninguna")}</small>
      <small>Gestionadas por comparador: ${escapeHtml(managedFlags || "ninguna")}</small>
    </div>
  `;
  const fields = dom.comparatorInstanceModalBody.querySelector("[data-instance-cli-fields]");
  if (configurableOptions.length) {
    fields.replaceChildren(...renderComparatorCliFields(proposal, configurableOptions));
    applyComparatorCliValues(fields, draft.proposalConfig?.cliValues || {});
    enhanceComparatorSelects(fields);
  } else {
    fields.innerHTML = '<p class="muted-note">Esta propuesta no expone flags propios adicionales.</p>';
  }
  dom.comparatorInstanceModal.hidden = false;
  dom.comparatorInstanceModalBody.querySelector("[data-comparator-instance-name]")?.focus();
}

function openComparatorMessageModal(title, message) {
  disposeComparatorChoices();
  comparatorInstanceModalState = { mode: "message" };
  dom.comparatorInstanceModalTitle.textContent = title;
  dom.comparatorInstanceModalSubtitle.textContent = "";
  dom.comparatorInstanceModalBody.innerHTML = `<div class="diagnostic-block diagnostic-error"><p>${escapeHtml(message)}</p></div>`;
  dom.saveComparatorInstanceModalButton.hidden = true;
  dom.cancelComparatorInstanceModalButton.textContent = "Cerrar";
  dom.comparatorInstanceModal.hidden = false;
  dom.cancelComparatorInstanceModalButton.focus();
}

function closeComparatorInstanceModal() {
  comparatorInstanceModalState = null;
  disposeComparatorChoices();
  dom.comparatorInstanceModal.hidden = true;
  dom.comparatorInstanceModalBody.replaceChildren();
  dom.saveComparatorInstanceModalButton.hidden = false;
  dom.cancelComparatorInstanceModalButton.textContent = "Cancelar";
}

function saveComparatorInstanceModal() {
  if (!comparatorInstanceModalState || comparatorInstanceModalState.mode === "message") return;
  const nameInput = dom.comparatorInstanceModalBody.querySelector("[data-comparator-instance-name]");
  const displayName = String(nameInput?.value || "").trim();
  if (!displayName) {
    nameInput?.focus();
    return;
  }
  const fields = dom.comparatorInstanceModalBody.querySelector("[data-instance-cli-fields]");
  const draft = {
    ...comparatorInstanceModalState.draft,
    displayName,
    proposalConfig: {
      extraArgs: "",
      cliValues: fields ? collectComparatorCliValues(fields) : {},
    },
  };
  if (comparatorInstanceModalState.mode === "edit") {
    comparatorInstances = comparatorInstances.map((instance) =>
      instance.instanceId === comparatorInstanceModalState.originalInstanceId ? draft : instance,
    );
  } else {
    comparatorInstances = [...comparatorInstances, draft];
  }
  renderComparatorInstanceList();
  syncComparatorProposalCatalogSelectionStates();
  closeComparatorInstanceModal();
}

function deleteComparatorInstance(instanceId) {
  comparatorInstances = comparatorInstances.filter((instance) => instance.instanceId !== instanceId);
  renderComparatorInstanceList();
  syncComparatorProposalCatalogSelectionStates();
}

function removeComparatorProposalInstances(proposalId) {
  comparatorInstances = comparatorInstances.filter((instance) => instance.proposalId !== proposalId);
  renderComparatorInstanceList();
  syncComparatorProposalCatalogSelectionStates();
}

function findDuplicateComparatorInstances() {
  const seen = new Map();
  for (const instance of comparatorInstances) {
    const canonical = JSON.stringify(instance.proposalConfig || { extraArgs: "", cliValues: {} }, (_key, value) => {
      if (value && typeof value === "object" && !Array.isArray(value)) {
        return Object.keys(value).sort().reduce((acc, key) => {
          acc[key] = value[key];
          return acc;
        }, {});
      }
      return value;
    });
    const key = `${instance.proposalId}::${canonical}`;
    if (seen.has(key)) {
      return { first: seen.get(key), second: instance };
    }
    seen.set(key, instance);
  }
  return null;
}

function comparatorProposalGitConfigs() {
  const configs = {};
  comparatorProposals.forEach((proposal) => {
    const git = proposal.git || {};
    configs[proposal.proposalId] = {
      remote: git.remote || "origin",
      branch: git.branch || "",
      pullMode: git.pullMode || "ff-only",
      expectedRemoteUrl: git.expectedRemoteUrl || "",
    };
  });
  dom.comparatorProposalSelector.querySelectorAll("[data-comparator-git-remote]").forEach((field) => {
    const proposalId = field.dataset.proposalId;
    if (!proposalId) return;
    configs[proposalId] = configs[proposalId] || {};
    configs[proposalId].remote = field.value.trim();
  });
  dom.comparatorProposalSelector.querySelectorAll("[data-comparator-git-branch]").forEach((field) => {
    const proposalId = field.dataset.proposalId;
    if (!proposalId) return;
    configs[proposalId] = configs[proposalId] || {};
    configs[proposalId].branch = field.value.trim();
  });
  dom.comparatorProposalSelector.querySelectorAll("[data-comparator-git-pull-mode]").forEach((field) => {
    const proposalId = field.dataset.proposalId;
    if (!proposalId) return;
    configs[proposalId] = configs[proposalId] || {};
    configs[proposalId].pullMode = field.value;
  });
  return configs;
}

function readComparatorConfig() {
  const referenceText = dom.comparatorReferenceText.value.trim();
  const model = dom.comparatorModel.value.trim();
  const selectedProposalIds = selectedComparatorProposalIds();
  if (!referenceText) {
    throw new Error("Define el texto de referencia.");
  }
  if (!model) {
    throw new Error("Define el modelo Ollama.");
  }
  if (!selectedProposalIds.length) {
    throw new Error("Agrega al menos una instancia de propuesta.");
  }
  const duplicate = findDuplicateComparatorInstances();
  if (duplicate) {
    const message = `No se puede ejecutar porque estas instancias no difieren en configuracion: ${duplicate.first.displayName} y ${duplicate.second.displayName}.`;
    openComparatorMessageModal("Configuraciones duplicadas", message);
    const error = new Error(message);
    error.code = "duplicateProposalInstances";
    throw error;
  }

  return {
    referenceText,
    model,
    selectedProposalIds,
    proposalConfigs: comparatorProposalConfigs(),
    proposalInstances: comparatorProposalInstancesPayload(),
    executionMode: dom.comparatorExecutionMode?.value || "fair_sequential",
    updateRepositoriesBeforeRun: Boolean(dom.comparatorUpdateReposBeforeRun?.checked),
    proposalGitConfigs: comparatorProposalGitConfigs(),
    topK: Math.floor(readClampedNumber(dom.comparatorTopK, "Top K tabla", 1, 200)),
    seed: Math.floor(readClampedNumber(dom.comparatorSeed, "Semilla", 0, 2147483647)),
    repetitionsK: Math.floor(readClampedNumber(dom.comparatorRepetitions, "K repeticiones", 1, 30)),
    n: Math.floor(readClampedNumber(dom.comparatorN, "N individuos", 1, 500)),
    generaciones: Math.floor(readClampedNumber(dom.comparatorGeneraciones, "Generaciones", 0, 500)),
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
    if (dom.comparatorUpdateReposBeforeRun && payload.defaults?.updateRepositoriesBeforeRun !== undefined) {
      dom.comparatorUpdateReposBeforeRun.checked = Boolean(payload.defaults.updateRepositoriesBeforeRun);
    }
    if (dom.comparatorExecutionMode && payload.defaults?.executionMode) {
      dom.comparatorExecutionMode.value = payload.defaults.executionMode;
      syncComparatorExecutionModeControls(false);
    }
    renderComparatorProposalControls(payload.proposals || []);
    const proposalSummary = payload.proposals
      .map((proposal) => {
        const missing = (proposal.dependencyStatus?.missing || []).join(", ");
        const status = proposal.available ? "disponible" : `no disponible${missing ? `, faltan ${missing}` : ""}`;
        return `${proposal.displayName}: ${status} (${proposal.objectiveNames.join(", ")})`;
      })
      .join("; ");
    const gitSummary = payload.proposals
      .map((proposal) => `${proposal.displayName}: ${proposal.git?.remote || "origin"}/${proposal.git?.branch || "--"} (${proposal.git?.pullMode || "ff-only"})`)
      .join("; ");
    renderDefinitionList(dom.comparatorIntegrationDetails, [
      ["Contrato", "{instanceId, proposalId, displayName, rows, metrics, outputDir, status}"],
      ["Propuestas", proposalSummary],
      ["Git antes de ejecutar", `${payload.defaults?.updateRepositoriesBeforeRun ? "Activado" : "Desactivado"}; ${gitSummary}`],
      ["EVOLMD", "data_final_evaluada.json -> [fitness]"],
      ["MESAP", "population_final.json -> [fitness]"],
      ["Uniobjetivo post-hoc", "EVOLMD y MESAP usan vector diagnostico SBERT [fidelity_sbert_posthoc, semantic_diversity_posthoc]; HV/spread no alteran la seleccion nativa."],
      ["EVOLMD-MO", "pareto_front.json -> [fidelity_sbert, diversity_individual]"],
      ["Binary MOPSO-CD", "pareto_front.json -> [objectives.f1, objectives.f2]; seleccion desde final_selection_hybrid.json"],
      ["MO comparable", "Graficos, HV y spread usan [(f1 + 1) / 2, f2 / 2] con referencia [0,0]."],
    ]);
  } catch (error) {
    renderDefinitionList(dom.comparatorIntegrationDetails, [
      ["Contrato", "{instanceId, proposalId, displayName, rows, metrics, outputDir, status}"],
      ["API", `No se pudo consultar el backend: ${error.message}`],
    ]);
  }
}

async function runComparator() {
  let config;
  try {
    config = readComparatorConfig();
  } catch (error) {
    const title = error.code === "duplicateProposalInstances" ? "Configuraciones duplicadas" : "Configuracion incompleta";
    setStatus(dom.comparatorStatusTone, dom.comparatorStatusTitle, dom.comparatorStatusDetail, title, error.message, "error");
    return;
  }

  stopComparatorPolling();
  comparatorPollFailureCount = 0;
  comparatorChartSignature = "";
  comparatorChartFilterIds = new Set();
  comparatorChartFilterSignature = "";
  resetComparatorLogLoader(null);
  setComparatorRunning(true);
  dom.comparatorResultsBody.innerHTML = '<tr><td colspan="9">Esperando resultados.</td></tr>';
  dom.comparatorProposalCards.innerHTML = "";
  dom.comparatorLogOutput.textContent = "Iniciando corrida...";
  setComparatorLogCopyButton(false);
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
    storeComparatorRunId(run.runId);
    setComparatorRunning(true, Boolean(run.cancelRequested));
    renderComparatorRun(run);
    comparatorPollTimer = window.setInterval(() => refreshComparatorRun(currentComparatorRunId), 2000);
    await refreshComparatorRun(currentComparatorRunId);
  } catch (error) {
    currentComparatorRunId = null;
    dom.comparatorConnectionDot.classList.add("is-error");
    dom.comparatorConnectionText.textContent = "Error";
    setComparatorRunning(false);
    if (error.code === "duplicateProposalInstances") {
      openComparatorMessageModal("Configuraciones duplicadas", error.message);
    }
    setStatus(dom.comparatorStatusTone, dom.comparatorStatusTitle, dom.comparatorStatusDetail, "Error al iniciar", error.message, "error");
  }
}

async function refreshComparatorRun(runId) {
  if (!runId) {
    return null;
  }

  try {
    const run = await requestComparatorJson(`/runs/${encodeURIComponent(runId)}`);
    comparatorPollFailureCount = 0;
    renderComparatorRun(run);
    if (COMPARATOR_TERMINAL_STATUSES.has(run.status)) {
      stopComparatorPolling();
      currentComparatorRunId = run.runId;
      setComparatorRunning(false);
    } else {
      currentComparatorRunId = run.runId;
      setComparatorRunning(true, Boolean(run.cancelRequested));
    }
    return run;
  } catch (error) {
    comparatorPollFailureCount += 1;
    const shouldRetry = comparatorPollFailureCount < COMPARATOR_MAX_TRANSIENT_POLL_FAILURES;
    currentComparatorRunId = runId;
    dom.comparatorConnectionDot.classList.add("is-error");
    dom.comparatorConnectionText.textContent = shouldRetry ? "Reconectando" : "Error";
    if (shouldRetry) {
      setComparatorRunning(true, Boolean(latestComparatorRun?.cancelRequested));
      setStatus(
        dom.comparatorStatusTone,
        dom.comparatorStatusTitle,
        dom.comparatorStatusDetail,
        "Reconectando corrida",
        `No se pudo consultar el backend (${error.message}). Reintento ${comparatorPollFailureCount}/${COMPARATOR_MAX_TRANSIENT_POLL_FAILURES}; se conserva el ultimo estado recibido.`,
        "busy",
      );
      return null;
    }

    stopComparatorPolling();
    setComparatorRunning(false);
    setStatus(dom.comparatorStatusTone, dom.comparatorStatusTitle, dom.comparatorStatusDetail, "Error al consultar corrida", error.message, "error");
    return null;
  }
}

async function resumeComparatorRun() {
  const runId = (dom.comparatorResumeRunId?.value || loadStoredComparatorRunId()).trim();
  if (!runId) {
    setStatus(
      dom.comparatorStatusTone,
      dom.comparatorStatusTitle,
      dom.comparatorStatusDetail,
      "Run ID requerido",
      "Ingresa el identificador de la carpeta runs/comparator que quieres reanudar.",
      "error",
    );
    return;
  }

  stopComparatorPolling();
  comparatorPollFailureCount = 0;
  currentComparatorRunId = runId;
  storeComparatorRunId(runId);
  setComparatorRunning(true);
  dom.comparatorConnectionDot.classList.add("is-busy");
  dom.comparatorConnectionDot.classList.remove("is-error");
  dom.comparatorConnectionText.textContent = "Consultando";
  setStatus(
    dom.comparatorStatusTone,
    dom.comparatorStatusTitle,
    dom.comparatorStatusDetail,
    "Reanudando corrida",
    `Consultando ${runId}.`,
    "busy",
  );

  const run = await refreshComparatorRun(runId);
  if (!run) {
    return;
  }
  if (!COMPARATOR_TERMINAL_STATUSES.has(run.status)) {
    comparatorPollTimer = window.setInterval(() => refreshComparatorRun(currentComparatorRunId), 2000);
  }
}

async function recomputeComparatorMetrics() {
  const runId = (currentComparatorRunId || dom.comparatorResumeRunId?.value || loadStoredComparatorRunId()).trim();
  if (!runId) {
    setStatus(
      dom.comparatorStatusTone,
      dom.comparatorStatusTitle,
      dom.comparatorStatusDetail,
      "Run ID requerido",
      "Reanuda o ingresa una corrida completada antes de recalcular metricas.",
      "error",
    );
    return;
  }

  dom.recomputeComparatorMetricsButton.disabled = true;
  dom.recomputeComparatorMetricsButton.textContent = "Recalculando...";
  setStatus(
    dom.comparatorStatusTone,
    dom.comparatorStatusTitle,
    dom.comparatorStatusDetail,
    "Recalculando metricas",
    "El backend reconstruye metricas y graficos desde artefactos Python reales, sin reejecutar propuestas ni LLM.",
    "busy",
  );

  try {
    const run = await requestComparatorJson(`/runs/${encodeURIComponent(runId)}/recompute-metrics`, {
      method: "POST",
      body: "{}",
    });
    currentComparatorRunId = run.runId;
    renderComparatorRun(run);
    setComparatorRunning(false);
  } catch (error) {
    setComparatorRunning(false);
    setStatus(dom.comparatorStatusTone, dom.comparatorStatusTitle, dom.comparatorStatusDetail, "No se pudo recalcular", error.message, "error");
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
  const finalById = new Map((run.proposals || []).map((proposal) => [comparatorEntityId(proposal), proposal]));
  const states = Object.values(run.proposalStates || {});
  const merged = states.map((state) => ({
    ...state,
    ...(finalById.get(comparatorEntityId(state)) || {}),
    progressState: state,
  }));
  const knownIds = new Set(merged.map((proposal) => comparatorEntityId(proposal)));
  for (const proposal of run.proposals || []) {
    if (!knownIds.has(comparatorEntityId(proposal))) {
      merged.push(proposal);
    }
  }
  return merged;
}

function renderComparatorProgress(progress, config = null) {
  if (!progress) {
    dom.comparatorProgressPercent.textContent = "--";
    dom.comparatorProgressSummary.textContent = "Sin corrida activa.";
    dom.comparatorProgressDetail.textContent = "Sin ejecucion.";
    dom.comparatorProgressBar.style.width = "0%";
    renderDefinitionList(dom.comparatorProgressDetails, [
      ["Tiempo transcurrido", "--"],
      ["Tiempo restante", "--"],
      ["Base estimacion", "--"],
      ["Alcance ETA", "--"],
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
    ["Base estimacion", progress.etaBasisLabel || "No disponible"],
    ["Alcance ETA", progress.etaScopeLabel || "No disponible"],
    ["Propuesta activa", progress.activeProposalName || "--"],
    ["Modo ejecucion", config?.executionPolicy?.label || config?.executionMode || "--"],
    ["Paralelismo efectivo", config?.executionPolicy ? `${config.executionPolicy.effectiveParallelism} de ${config.executionPolicy.requestedParallelism} solicitado(s)` : "--"],
    ["K repeticiones", config?.repetitionsK ?? 1],
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
  const tokenLabel = costSummary.hasTokenReport ? `${costSummary.totalTokens ?? 0} tokens reportados` : "tokens no reportados";
  dom.comparatorLlmCalls.textContent = String(costSummary.llmCalls ?? 0);
  dom.comparatorLlmCallsDetail.textContent = `${successful} ok, ${failed} fallida(s), ${tokenLabel}.`;
  dom.comparatorWallClock.textContent = costSummary.runWallClockLabel || "--";
  dom.comparatorLlmTime.textContent = costSummary.llmClientWallClockLabel || "--";
}

function renderComparatorRun(run) {
  latestComparatorRun = run;
  storeComparatorRunId(run.runId);
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
  syncComparatorRecomputeButton(run);

  renderComparatorProgress(run.progress || null, run.config || null);
  renderComparatorCostSummary(run.costSummary || null);
  renderComparatorCards(comparatorProposalViews(run), run.config || null);
  renderComparatorRows(rows);
  renderComparatorCostDetails(run);
  renderComparatorCharts(run);
  renderComparatorLogs(run);

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
  decorateAbbreviationTooltips(document.getElementById("proposalComparator") || document);
}

function comparatorProposalParameterLabel(config) {
  if (!config) return "--";
  const n = config.n ?? "--";
  const generations = config.generaciones ?? config.iterations ?? "--";
  return `N = ${n}; G = ${generations}`;
}

function comparatorRepetitionProgressLabel(progressState, config) {
  const total = Math.max(1, Number(progressState?.totalRepetitions ?? config?.repetitionsK ?? 1));
  const completed = Math.max(0, Math.min(total, Number(progressState?.completedRepetitions ?? 0)));
  const current = progressState?.currentRepetitionIndex
    ? `; actual ${Math.max(1, Number(progressState.currentRepetitionIndex))}/${total}`
    : "";
  return {
    text: `Repeticion estocastica: ${completed}/${total}`,
    title: `Repeticiones estocasticas completadas sobre el total configurado${current}.`,
  };
}

function comparatorProposalSummaryTooltip(label, metrics = {}) {
  const key = String(label || "").trim().toLocaleLowerCase("es-CL");
  const diagnosticSuffix = metrics.postHocDiagnostic
    ? " En propuestas uniobjetivo se calcula como diagnostico post-hoc; no fue optimizado por el algoritmo."
    : "";
  if (key === "parametros") {
    return "Parametros comunes enviados por CLI: N es poblacion; G es generaciones o iteraciones.";
  }
  if (key === "soluciones finales") {
    return "Soluciones finales normalizadas correctamente sobre el total del archivo final; no corresponde a generaciones.";
  }
  if (key === "mejor f.o.") {
    return "Vector nativo de la solucion rank 1. En MO: no dominadas primero y luego mayor suma de objetivos nativos; en uniobjetivo: mayor fitness.";
  }
  if (key === "mejor comp.") {
    return "Vector comparable normalizado de la solucion con mayor suma f1_n + f2_n. Ambos objetivos se maximizan; no reemplaza el analisis Pareto.";
  }
  if (key.startsWith("no dom")) {
    return `Cantidad de soluciones no dominadas: ninguna otra solucion es igual o mejor en todos los objetivos y mejor en al menos uno.${diagnosticSuffix}`;
  }
  if (key.startsWith("hv")) {
    return `Hypervolume del frente no dominado en el espacio comparable normalizado con referencia [0,0]. Mayor es mejor.${diagnosticSuffix}`;
  }
  if (key.startsWith("spread")) {
    return `Uniformidad del frente no dominado en el espacio comparable normalizado. Menor es mejor; valores altos indican distancias mas irregulares.${diagnosticSuffix}`;
  }
  if (key === "vector post-hoc") {
    return "Vector diagnostico SBERT/diversidad calculado despues de ejecutar una propuesta uniobjetivo; se usa para comparar, no para decidir dentro del algoritmo.";
  }
  if (key === "algoritmo") {
    return "Tiempo wall-clock del proceso Python de la propuesta; no incluye metricas ni graficos del comparador.";
  }
  if (key === "total prop.") {
    return "Tiempo de la propuesta mas post-procesamiento externo del comparador; no altera el costo interno del algoritmo.";
  }
  if (key === "post") {
    return "Tiempo de seleccion o ranking externo posterior a la ejecucion; se reporta separado del algoritmo.";
  }
  if (key === "metricas web") {
    return "Tiempo de normalizacion y metricas calculadas por la web despues de ejecutar la propuesta.";
  }
  if (key === "llamadas llm") {
    return "Cantidad de llamadas registradas al modelo LLM durante la ejecucion de la propuesta.";
  }
  if (key === "tiempo llm") {
    return "Suma de latencias cliente de llamadas LLM reportadas por la propuesta o wrapper.";
  }
  if (key === "prom. llamada") {
    return "Tiempo LLM promedio por llamada registrada. Menor indica llamadas mas rapidas.";
  }
  if (key === "tokens") {
    return "Tokens reportados por la propuesta cuando existen; No reportado no se interpreta como cero.";
  }
  if (key === "git") {
    return "Rama y commit local usados para ejecutar la propuesta.";
  }
  if (key === "salida") {
    return "Directorio donde quedaron artefactos, resultados y logs de esa propuesta.";
  }
  return abbreviationTooltipFor(label);
}

function comparatorProposalSummaryTerm(label, metrics = {}) {
  const tooltip = comparatorProposalSummaryTooltip(label, metrics);
  const title = tooltip ? ` title="${escapeHtml(tooltip)}"` : "";
  return `<dt${title}>${escapeHtml(label)}</dt>`;
}

function renderComparatorCards(proposals, config = null) {
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
      const hvLabel = metrics.postHocDiagnostic ? "HV comp. post-hoc" : "HV comp.";
      const spreadLabel = metrics.postHocDiagnostic ? "Spread comp. post-hoc" : "Spread comp.";
      const nonDominatedLabel = metrics.postHocDiagnostic ? "No dom. post-hoc" : "No dominadas";
      const tokenValue = comparatorCostHasTokenReport(cost) ? String(cost.totalTokens ?? 0) : "No reportado";
      const progressState = proposal.progressState || proposal;
      const progressPercent = Math.round(Math.max(0, Math.min(1, Number(progressState.progress || 0))) * 100);
      const stageLabel = progressState.stageLabel || comparatorStatusLabel(proposal.status);
      const repetitionProgress = comparatorRepetitionProgressLabel(progressState, config);
      const git = proposal.gitRevision || {};
      const gitLabel = git.shortCommit
        ? `${git.configuredBranch || git.branch || "--"} @ ${git.shortCommit}${git.dirty ? " (local dirty)" : ""}`
        : "--";
      const parameterLabel = comparatorProposalParameterLabel(config);
      const term = (label) => comparatorProposalSummaryTerm(label, metrics);
      const article = document.createElement("article");
      article.className = "proposal-card";
      article.innerHTML = `
        <strong>${escapeHtml(proposal.displayName || proposal.proposalId)}</strong>
        <p><span class="${comparatorStatusClass(proposal.status)}">${escapeHtml(comparatorStatusLabel(proposal.status))}</span>${proposal.error ? `: ${escapeHtml(proposal.error)}` : ""}</p>
        <div class="mini-progress" aria-label="Progreso ${escapeHtml(proposal.displayName || proposal.proposalId)}">
          <span style="width: ${progressPercent}%"></span>
        </div>
        <p>${escapeHtml(stageLabel)} (${progressPercent}%)</p>
        <p title="${escapeHtml(repetitionProgress.title)}">${escapeHtml(repetitionProgress.text)}</p>
        <dl>
          ${term("Parametros")}<dd title="N: tamano de poblacion o cantidad de individuos; G: generaciones o iteraciones configuradas.">${escapeHtml(parameterLabel)}</dd>
          ${term("Soluciones finales")}<dd>${escapeHtml(String(metrics.completedRows ?? 0))}/${escapeHtml(String(metrics.totalRows ?? 0))} <small title="OK/total: soluciones finales normalizadas correctamente sobre el total del archivo final; no corresponde a G.">OK/total; no es G</small></dd>
          ${term("Mejor F.O.")}<dd>${escapeHtml(metrics.bestObjectiveLabel || "--")}</dd>
          ${term("Mejor comp.")}<dd>${escapeHtml(metrics.bestComparableObjectiveLabel || "--")}</dd>
          ${term(nonDominatedLabel)}<dd>${escapeHtml(String((metrics.postHocDiagnostic ? metrics.postHocNonDominatedRows : metrics.nonDominatedRows) ?? 0))}</dd>
          ${term(hvLabel)}<dd>${escapeHtml(metrics.hypervolumeLabel || "No aplica")}</dd>
          ${term(spreadLabel)}<dd>${escapeHtml(metrics.spreadLabel || "No aplica")}</dd>
          ${metrics.postHocDiagnostic ? `${term("Vector post-hoc")}<dd>${escapeHtml(metrics.bestDiagnosticObjectiveLabel || "--")}</dd>` : ""}
          ${term("Algoritmo")}<dd>${escapeHtml(cost.processWallClockLabel || "--")}</dd>
          ${term("Total prop.")}<dd>${escapeHtml(cost.proposalTotalWallClockLabel || cost.processWallClockLabel || "--")}</dd>
          ${term("Post")}<dd>${escapeHtml(cost.postProcessingWallClockLabel || "0s")}</dd>
          ${term("Metricas web")}<dd>${escapeHtml(cost.metricExtractionLabel || "0s")}</dd>
          ${term("Llamadas LLM")}<dd>${escapeHtml(String(cost.llmCalls ?? 0))}</dd>
          ${term("Tiempo LLM")}<dd>${escapeHtml(cost.llmClientWallClockLabel || "--")}</dd>
          ${term("Prom. llamada")}<dd>${escapeHtml(cost.llmAverageCallLabel || "No disponible")}</dd>
          ${term("Tokens")}<dd>${escapeHtml(tokenValue)}</dd>
          ${term("Git")}<dd>${escapeHtml(gitLabel)}</dd>
          ${term("Salida")}<dd>${escapeHtml(metrics.outputDir || proposal.outputDir || "--")}</dd>
        </dl>
      `;
      return article;
    }),
  );
}

function renderComparatorCostDetails(run) {
  if (!run) {
    renderComparatorCostExplanation(null);
    renderComparatorCostTable([], null);
    renderComparatorCostTraceability([]);
    return;
  }
  const proposals = run.proposals || [];
  renderComparatorCostExplanation(run);
  renderComparatorCostTable(proposals, run.config?.executionPolicy || {});
  renderComparatorCostTraceability(proposals);
}

function renderComparatorCostExplanation(run) {
  if (!dom.comparatorCostExplanation) return;
  dom.comparatorCostExplanation.classList.toggle("is-warning", Boolean(run) && !run.config?.executionPolicy?.costsComparable);
  if (!run) {
    dom.comparatorCostExplanation.innerHTML = "<p>Sin corrida activa.</p>";
    return;
  }
  const policy = run.config?.executionPolicy || {};
  const comparable = Boolean(policy.costsComparable);
  const requested = policy.requestedParallelism ?? run.config?.proposalParallelism ?? "--";
  const effective = policy.effectiveParallelism ?? run.config?.effectiveProposalParallelism ?? "--";
  const runCost = run.costSummary || {};
  const comparisonDetail = comparable
    ? "las propuestas se ejecutan una por una."
    : "las propuestas comparten Ollama/CPU/GPU; los valores se muestran solo como diagnostico.";
  dom.comparatorCostExplanation.innerHTML = `
    <p><strong>${escapeHtml(comparable ? "Costos comparables" : "Costos no comparables")}</strong>: ${escapeHtml(comparisonDetail)}</p>
    <p>Modo: ${escapeHtml(policy.label || run.config?.executionMode || "--")}; paralelismo efectivo ${escapeHtml(String(effective))} de ${escapeHtml(String(requested))} solicitado(s); wall-clock total de corrida ${escapeHtml(runCost.runWallClockLabel || "--")}.</p>
    <p>Menor es mejor en las metricas de costo. Algoritmo es el runtime interno reportado por cada propuesta; Proceso Python es el wall-clock del proceso; Total propuesta suma proceso y post-proceso del comparador. Post-procesamiento, extraccion de metricas y preparacion visual no se suman al costo real del algoritmo.</p>
    <p>Tokens y duracion Ollama solo se comparan cuando todas las propuestas completadas reportan esa metrica con fuente real; si no, se muestra No reportado y no se destaca ganador.</p>
  `;
}

function comparatorCostMetricDefinitions() {
  const secondsMetric = (key, labelKey) => ({
    value: (_proposal, cost) => cost[key],
    format: (_proposal, cost) => cost[labelKey] || "--",
    isReported: (_proposal, cost) => Number.isFinite(Number(cost[key])),
  });
  return [
    {
      id: "algorithm",
      label: "Algoritmo",
      detail: "Runtime interno reportado por la propuesta.",
      ...secondsMetric("algorithmRuntimeSeconds", "algorithmRuntimeLabel"),
    },
    {
      id: "process",
      label: "Proceso Python",
      detail: "Wall-clock del proceso Python ejecutado por el backend.",
      ...secondsMetric("processWallClockSeconds", "processWallClockLabel"),
    },
    {
      id: "proposalTotal",
      label: "Total propuesta",
      detail: "Proceso Python mas post-procesamiento del comparador.",
      ...secondsMetric("proposalTotalWallClockSeconds", "proposalTotalWallClockLabel"),
    },
    {
      id: "post",
      label: "Post-procesamiento",
      detail: "Seleccion o ranking externo del comparador; no carga al algoritmo.",
      ...secondsMetric("postProcessingWallClockSeconds", "postProcessingWallClockLabel"),
    },
    {
      id: "metrics",
      label: "Extraccion metricas",
      detail: "Normalizacion y metricas web posteriores; no carga al algoritmo.",
      ...secondsMetric("metricExtractionSeconds", "metricExtractionLabel"),
    },
    {
      id: "plots",
      label: "Preparacion visual",
      detail: "Armado de datos para tablas y graficos; no carga al algoritmo.",
      ...secondsMetric("plotPreparationSeconds", "plotPreparationLabel"),
    },
    {
      id: "llmCalls",
      label: "Llamadas LLM",
      detail: "Total de llamadas reales registradas hacia el LLM.",
      value: (_proposal, cost) => cost.llmCalls,
      format: (_proposal, cost) => String(cost.llmCalls ?? 0),
      isReported: (_proposal, cost) => Number.isFinite(Number(cost.llmCalls)),
    },
    {
      id: "llmTime",
      label: "Tiempo LLM cliente",
      detail: "Suma de latencias cliente de llamadas LLM.",
      ...secondsMetric("llmClientWallClockSeconds", "llmClientWallClockLabel"),
    },
    {
      id: "llmAverage",
      label: "Promedio por llamada",
      detail: "Tiempo LLM cliente dividido por cantidad de llamadas.",
      ...secondsMetric("llmAverageCallSeconds", "llmAverageCallLabel"),
    },
    {
      id: "llmFailed",
      label: "Llamadas fallidas",
      detail: "Errores registrados durante llamadas LLM.",
      value: (_proposal, cost) => cost.llmFailedCalls,
      format: (_proposal, cost) => String(cost.llmFailedCalls ?? 0),
      isReported: (_proposal, cost) => Number.isFinite(Number(cost.llmFailedCalls)),
    },
    {
      id: "tokens",
      label: "Tokens reportados",
      detail: "Suma de prompt y completion tokens cuando la propuesta los reporta.",
      value: (_proposal, cost) => cost.totalTokens,
      format: (_proposal, cost) => String(cost.totalTokens ?? 0),
      isReported: (_proposal, cost) => comparatorCostHasTokenReport(cost),
      requireAllReported: true,
    },
    {
      id: "ollamaDuration",
      label: "Duracion Ollama reportada",
      detail: "Duracion total informada por Ollama cuando esta disponible.",
      value: (_proposal, cost) => cost.ollamaTotalDurationSeconds,
      format: (_proposal, cost) => cost.ollamaTotalDurationLabel || "--",
      isReported: (_proposal, cost) => comparatorCostHasOllamaDuration(cost),
      requireAllReported: true,
    },
  ];
}

function comparatorCostHasTokenReport(cost = {}) {
  return Boolean(cost.hasTokenReport)
    || Number(cost.totalTokens) > 0
    || Number(cost.promptEvalCount) > 0
    || Number(cost.evalCount) > 0;
}

function comparatorCostHasOllamaDuration(cost = {}) {
  return Boolean(cost.hasOllamaDurationReport) || Number(cost.ollamaTotalDurationSeconds) > 0;
}

function renderComparatorCostTable(proposals, policy) {
  if (!dom.comparatorCostTableHead || !dom.comparatorCostTableBody) return;
  dom.comparatorCostTableHead.innerHTML = `
    <tr>
      <th>Metrica</th>
      <th>Detalle</th>
      ${proposals.map((proposal) => `
        <th>
          <span>${escapeHtml(proposal.displayName || proposal.proposalId)}</span>
          <small class="${comparatorStatusClass(proposal.status)}">${escapeHtml(comparatorStatusLabel(proposal.status))}</small>
        </th>
      `).join("")}
    </tr>
  `;
  if (!proposals.length) {
    dom.comparatorCostTableBody.innerHTML = '<tr><td colspan="2">Sin resultados todavia.</td></tr>';
    return;
  }
  const metrics = comparatorCostMetricDefinitions();
  const costsComparable = Boolean(policy?.costsComparable);
  dom.comparatorCostTableBody.replaceChildren(
    ...metrics.map((metric) => {
      const winners = comparatorBestCostProposalIds(proposals, metric, { costsComparable });
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <th scope="row">${escapeHtml(metric.label)}</th>
        <td>${escapeHtml(metric.detail)}</td>
        ${proposals.map((proposal) => comparatorCostMetricCell(proposal, metric, winners)).join("")}
      `;
      return tr;
    }),
  );
  decorateAbbreviationTooltips(document.getElementById("comparatorCostsTab") || dom.comparatorCostTableBody);
}

function comparatorCostMetricCell(proposal, metric, winners) {
  const cost = proposal.cost || {};
  const rawValue = metric.value(proposal, cost);
  const value = rawValue === null || rawValue === undefined || rawValue === "" ? NaN : Number(rawValue);
  const reported = Number.isFinite(value) && (!metric.isReported || metric.isReported(proposal, cost));
  const best = proposal.status === "completed" && reported && winners.has(comparatorEntityId(proposal));
  const label = reported ? metric.format(proposal, cost) : "No reportado";
  const className = best ? "metric-best comparator-cost-best" : "";
  return `<td class="${className}">${best ? `<strong>${escapeHtml(label)}</strong>` : escapeHtml(label)}</td>`;
}

function renderComparatorCostTraceability(proposals) {
  if (!dom.comparatorCostTraceabilityBody) return;
  if (!proposals.length) {
    dom.comparatorCostTraceabilityBody.innerHTML = "Sin corrida activa.";
    if (dom.comparatorCostTraceability) dom.comparatorCostTraceability.open = false;
    return;
  }
  dom.comparatorCostTraceabilityBody.innerHTML = proposals.map((proposal) => {
    const update = proposal.repositoryUpdate || {};
    const revision = proposal.gitRevision || {};
    const metrics = proposal.metrics || {};
    return `
      <div class="semantic-artifact-group">
        <h3>${escapeHtml(proposal.displayName || proposal.proposalId)}</h3>
        <dl class="definition-grid compact-definition-grid">
          <div><dt>Comando</dt><dd>${escapeHtml(proposal.command || "--")}</dd></div>
          <div><dt>Git</dt><dd>${escapeHtml(`${update.status || revision.status || "--"}; ${update.remote || revision.remote || "--"}/${update.branch || revision.configuredBranch || revision.branch || "--"}; commit ${revision.shortCommit || "--"}; ${update.message || ""}`)}</dd></div>
          <div><dt>Salida</dt><dd>${escapeHtml(metrics.outputDir || proposal.outputDir || "--")}</dd></div>
        </dl>
      </div>
    `;
  }).join("");
}

function renderComparatorCharts(run) {
  if (!window.echarts) {
    dom.comparatorParetoCharts.innerHTML = '<article class="panel"><p>ECharts no esta disponible.</p></article>';
    return;
  }
  const signature = comparatorChartsSignature(run);
  if (signature === comparatorChartSignature) {
    return;
  }
  comparatorChartSignature = signature;
  disposeComparatorCharts();
  const proposals = (run.proposals || []).filter((proposal) => proposal.status === "completed");
  renderComparatorChartFilters(proposals);
  const filteredProposals = comparatorFilteredProposals(proposals);
  renderComparatorParetoCharts(filteredProposals);
  renderComparatorCombinedSelectedChart(filteredProposals);
  renderComparatorGlobalNonDominatedChart(filteredProposals);
  renderComparatorMetricLine(dom.comparatorHvChart, filteredProposals, "hypervolume", "HV por iteracion");
  renderComparatorMetricLine(dom.comparatorNonDominatedChart, filteredProposals, "nonDominatedRows", "Soluciones no dominadas");
  renderComparatorMetricLine(dom.comparatorSpreadChart, filteredProposals, "spread", "Spread por iteracion");
  renderComparatorMetricLine(dom.comparatorGlobalInertiaChart, filteredProposals, "globalInertia", "Inercia global por iteracion");
  renderComparatorMetricLine(dom.comparatorGlobalEntropyChart, filteredProposals, "globalEntropy", "Entropia global por iteracion");
}

function comparatorChartsSignature(run) {
  return JSON.stringify({
    runId: run.runId || "",
    filters: Array.from(comparatorChartFilterIds).sort(),
    proposals: (run.proposals || []).map((proposal) => ({
      instanceId: comparatorEntityId(proposal),
      proposalId: proposal.proposalId,
      status: proposal.status,
      pareto: ((proposal.charts || {}).pareto || []).length,
      selected: ((proposal.charts || {}).selected || []).length,
      nonDominated: ((proposal.charts || {}).nonDominated || []).length,
      series: (proposal.series || []).length,
      seriesValues: (proposal.series || []).map((point) => [
        point.generation,
        point.hypervolume,
        point.nonDominatedRows,
        point.spread,
        point.globalInertia,
        point.globalEntropy,
      ]),
      rows: (proposal.rows || []).length,
    })),
  });
}

function comparatorFilterSignature(proposals) {
  return proposals.map((proposal) => `${comparatorEntityId(proposal)}:${proposal.displayName || comparatorEntityId(proposal)}`).join("|");
}

function renderComparatorChartFilters(proposals) {
  if (!dom.comparatorChartProposalFilters) return;
  const signature = comparatorFilterSignature(proposals);
  if (!proposals.length) {
    comparatorChartFilterSignature = signature;
    comparatorChartFilterIds = new Set();
    dom.comparatorChartProposalFilters.innerHTML = '<span class="muted-text">Sin propuestas completadas para filtrar.</span>';
    return;
  }
  if (signature === comparatorChartFilterSignature) {
    return;
  }
  comparatorChartFilterSignature = signature;
  comparatorChartFilterIds = new Set(proposals.map((proposal) => comparatorEntityId(proposal)));
  dom.comparatorChartProposalFilters.replaceChildren(
    ...proposals.map((proposal, index) => {
      const entityId = comparatorEntityId(proposal);
      const label = document.createElement("label");
      label.className = "chart-filter-option";
      label.innerHTML = `
        <input type="checkbox" data-comparator-chart-filter="${escapeHtml(entityId)}" checked>
        <span class="chart-filter-swatch" style="background: ${chartPalette(index)}"></span>
        <span>${escapeHtml(proposal.displayName || proposal.proposalId)}</span>
      `;
      return label;
    }),
  );
}

function comparatorFilteredProposals(proposals) {
  if (!proposals.length) return [];
  if (!comparatorChartFilterIds.size) return [];
  return proposals.filter((proposal) => comparatorChartFilterIds.has(comparatorEntityId(proposal)));
}

function onComparatorChartFilterChange(event) {
  const input = event.target.closest("[data-comparator-chart-filter]");
  if (!input || !dom.comparatorChartProposalFilters?.contains(input)) return;
  const proposalId = input.dataset.comparatorChartFilter;
  if (input.checked) {
    comparatorChartFilterIds.add(proposalId);
  } else {
    comparatorChartFilterIds.delete(proposalId);
  }
  comparatorChartSignature = "";
  if (latestComparatorRun) {
    renderComparatorCharts(latestComparatorRun);
  }
}

function renderComparatorParetoCharts(proposals) {
  if (!proposals.length) {
    dom.comparatorParetoCharts.innerHTML = '<article class="panel"><p class="muted-text">Sin propuestas seleccionadas para mostrar frentes de Pareto.</p></article>';
    return;
  }
  dom.comparatorParetoCharts.replaceChildren(
    ...proposals.map((proposal) => {
      const article = document.createElement("article");
      article.className = "panel";
      article.innerHTML = `
        <div class="panel-title">
          <h2>${escapeHtml(proposal.displayName)}</h2>
          <span>Frente de Pareto</span>
        </div>
        <div class="pareto-chart-stack">
          <section class="pareto-chart-pane">
            <h3>Normalizado</h3>
            <div class="chart-surface" data-normalized-chart></div>
          </section>
          <section class="pareto-chart-pane">
            <h3>No normalizado</h3>
            <div class="chart-surface" data-raw-chart></div>
          </section>
        </div>
      `;
      const normalizedChartNode = article.querySelector("[data-normalized-chart]");
      const rawChartNode = article.querySelector("[data-raw-chart]");
      window.queueMicrotask(() => {
        const normalizedChart = window.echarts.init(normalizedChartNode);
        const rawChart = window.echarts.init(rawChartNode);
        comparatorCharts.push(normalizedChart, rawChart);
        normalizedChart.setOption(paretoChartOption("Frente comparable normalizado", proposal.charts || {}, proposal.metrics || {}));
        rawChart.setOption(rawParetoChartOption("Frente semantico no normalizado", proposal.charts || {}));
      });
      return article;
    }),
  );
}

function renderComparatorCombinedSelectedChart(proposals) {
  const comparisonPool = comparatorAllNonDominatedPoints(proposals);
  const selectedSeries = proposals.map((proposal, index) => {
    const color = chartPalette(index);
    return {
      name: proposal.displayName,
      type: "scatter",
      symbolSize: 14,
      label: { show: false },
      data: comparatorProposalChartPoints(proposal, "selected").map((point) =>
        comparatorChartPointData(point, color, {
          globallyNonDominated: comparatorIsGloballyNonDominated(point, comparisonPool),
        }),
      ),
      itemStyle: { color },
    };
  });
  const chart = window.echarts.init(dom.comparatorCombinedParetoChart);
  comparatorCharts.push(chart);
  chart.setOption(
    baseScatterOption("Top 5 combinado", selectedSeries, {
      description: "Ejes normalizados comparables. Mayor fidelidad y diversidad es mejor. Borde rojo: no dominada frente a la union de propuestas.",
    }),
  );
}

function renderComparatorGlobalNonDominatedChart(proposals) {
  const globalFront = comparatorGlobalNonDominatedPoints(proposals);
  const countsByProposal = comparatorCountByProposal(globalFront);
  const series = proposals.map((proposal, index) => {
    const color = chartPalette(index);
    const entityId = comparatorEntityId(proposal);
    const points = globalFront.filter((point) => (point.instanceId || point.proposalId) === entityId);
    return {
      name: `${proposal.displayName} (${countsByProposal.get(entityId) || 0})`,
      type: "scatter",
      symbolSize: 13,
      label: { show: false },
      data: points.map((point) => comparatorChartPointData(point, color)),
      itemStyle: { color },
    };
  });
  const chart = window.echarts.init(dom.comparatorGlobalNonDominatedChart);
  comparatorCharts.push(chart);
  chart.setOption(
    baseScatterOption("No dominadas globales", series, {
      description: `${globalFront.length} soluciones globalmente no dominadas en ejes normalizados comparables. Mayor fidelidad y diversidad es mejor.`,
    }),
  );
}

function renderComparatorMetricLine(container, proposals, metricKey, title) {
  const metadata = comparatorMetricMetadata(metricKey);
  const series = proposals.map((proposal, index) => ({
    name: proposal.displayName,
    type: "line",
    connectNulls: false,
    showSymbol: false,
    data: (proposal.series || [])
      .filter((point) => point[metricKey] !== null && point[metricKey] !== undefined)
      .map((point) => [point.generation, point[metricKey]]),
    itemStyle: { color: chartPalette(index) },
  }));
  const metricValues = series.flatMap((item) => item.data.map((point) => point[1])).filter(Number.isFinite);
  const referenceTarget = series.find((item) => item.data.length > 0);
  if (referenceTarget && metricValues.length > 0) {
    referenceTarget.markLine = comparatorMetricReferenceLines(metricValues, metadata.higherIsBetter);
  }
  const chart = window.echarts.init(container);
  comparatorCharts.push(chart);
  chart.setOption({
    title: { text: title, subtext: metadata.description, left: 8, top: 6, textStyle: { fontSize: 13 }, subtextStyle: { fontSize: 11, color: "#64748b" } },
    tooltip: safeChartTooltip("axis", comparatorLineTooltipFormatter),
    legend: { top: 60, type: "scroll" },
    grid: { left: 52, right: 22, top: 106, bottom: 70, containLabel: true },
    toolbox: { feature: { saveAsImage: {}, dataZoom: {} }, right: 8, top: 38 },
    dataZoom: [{ type: "inside" }, { type: "slider", height: 18, bottom: 10 }],
    xAxis: { type: "value", name: "Iteracion", nameLocation: "middle", nameGap: 34, minInterval: 1 },
    yAxis: { type: "value", scale: true },
    graphic: comparatorEmptyChartGraphic(series, "Sin datos para las propuestas filtradas."),
    series,
  });
}

function paretoChartOption(title, charts, metrics = {}) {
  const allPoints = (charts.pareto || []).map((point) => ({
    value: [point.x, point.y],
    labelText: point.label,
    prompt: point.prompt,
    rank: point.rank,
    nativeObjectiveVector: point.nativeObjectiveVector,
    comparableObjectiveVector: point.comparableObjectiveVector,
    coordinateSpace: point.coordinateSpace,
  }));
  const selectedPoints = (charts.selected || []).map((point) => ({
    value: [point.x, point.y],
    labelText: point.label,
    prompt: point.prompt,
    rank: point.rank,
    nativeObjectiveVector: point.nativeObjectiveVector,
    comparableObjectiveVector: point.comparableObjectiveVector,
    coordinateSpace: point.coordinateSpace,
  }));
  const hvAreaSeries = comparatorHypervolumeAreaSeries(charts.nonDominated || [], metrics.hypervolumeLabel || "");
  return baseScatterOption(title, [
    ...hvAreaSeries,
    { name: "Individuos", type: "scatter", symbolSize: 8, data: allPoints, label: { show: false }, itemStyle: { color: "#60a5fa", opacity: 0.72 } },
    { name: "Seleccionadas", type: "scatter", symbol: "diamond", symbolSize: 15, data: selectedPoints, label: { show: false }, itemStyle: { color: "#b42318" } },
  ], {
    description: "Ejes normalizados comparables. Area sombreada: HV dominado respecto a [0, 0].",
  });
}

function rawParetoChartOption(title, charts) {
  const allPoints = comparatorRawChartPoints(charts.pareto || []).map((point) => ({
    value: point.value,
    labelText: point.label,
    prompt: point.prompt,
    rank: point.rank,
    nativeObjectiveVector: point.nativeObjectiveVector,
    comparableObjectiveVector: point.comparableObjectiveVector,
    coordinateSpace: point.coordinateSpace,
  }));
  const selectedPoints = comparatorRawChartPoints(charts.selected || []).map((point) => ({
    value: point.value,
    labelText: point.label,
    prompt: point.prompt,
    rank: point.rank,
    nativeObjectiveVector: point.nativeObjectiveVector,
    comparableObjectiveVector: point.comparableObjectiveVector,
    coordinateSpace: point.coordinateSpace,
  }));
  return baseScatterOption(title, [
    { name: "Individuos", type: "scatter", symbolSize: 8, data: allPoints, label: { show: false }, itemStyle: { color: "#60a5fa", opacity: 0.72 } },
    { name: "Seleccionadas", type: "scatter", symbol: "diamond", symbolSize: 15, data: selectedPoints, label: { show: false }, itemStyle: { color: "#b42318" } },
  ], {
    description: "Ejes semanticos raw comunes: fidelidad [-1, 1], diversidad [0, 2]. HV no se calcula en este espacio.",
    tooltipFormatter: comparatorRawScatterTooltipFormatter,
    xAxisName: "Fidelidad semantica raw",
    yAxisName: "Diversidad semantica raw",
    xAxisMin: COMPARATOR_RAW_OBJECTIVE_BOUNDS.xMin,
    xAxisMax: COMPARATOR_RAW_OBJECTIVE_BOUNDS.xMax,
    yAxisMin: COMPARATOR_RAW_OBJECTIVE_BOUNDS.yMin,
    yAxisMax: COMPARATOR_RAW_OBJECTIVE_BOUNDS.yMax,
    idealPoint: [COMPARATOR_RAW_OBJECTIVE_BOUNDS.xMax, COMPARATOR_RAW_OBJECTIVE_BOUNDS.yMax],
  });
}

function comparatorHypervolumeAreaSeries(points, hypervolumeLabel) {
  if (!hypervolumeLabel || hypervolumeLabel === "No aplica") return [];
  const area = comparatorHypervolumeArea(points);
  if (!area) return [];
  return [
    {
      name: "Area HV",
      type: "line",
      data: area.lineData,
      showSymbol: false,
      silent: true,
      tooltip: { show: false },
      lineStyle: { color: "#2563eb", width: 1.2, opacity: 0.5 },
      areaStyle: { color: "rgba(37, 99, 235, 0.16)" },
      z: 0,
      markPoint: {
        silent: true,
        symbol: "rect",
        symbolSize: [92, 26],
        itemStyle: {
          color: "rgba(255, 255, 255, 0.9)",
          borderColor: "#93c5fd",
          borderWidth: 1,
        },
        label: {
          show: true,
          formatter: `HV = ${hypervolumeLabel}`,
          color: "#1e3a8a",
          fontWeight: 700,
          fontSize: 11,
        },
        data: [{ coord: area.labelPosition }],
      },
    },
  ];
}

function baseScatterOption(title, series, options = {}) {
  const idealSeries = comparatorIdealSeries(series, options.idealPoint);
  const renderedSeries = idealSeries ? [...series, idealSeries] : series;
  return {
    title: {
      text: title,
      subtext: options.description || "Mayor fidelidad y diversidad es mejor.",
      left: 8,
      top: 6,
      textStyle: { fontSize: 13 },
      subtextStyle: { fontSize: 11, color: "#64748b" },
    },
    tooltip: safeChartTooltip("item", options.tooltipFormatter || comparatorScatterTooltipFormatter),
    legend: { top: 62, type: "scroll" },
    grid: { left: 58, right: 24, top: 112, bottom: 78, containLabel: true },
    toolbox: { feature: { saveAsImage: {}, dataZoom: {} }, right: 8, top: 38 },
    dataZoom: [{ type: "inside" }, { type: "slider", height: 18, bottom: 12 }],
    xAxis: {
      type: "value",
      name: options.xAxisName || "Fidelidad normalizada",
      nameLocation: "middle",
      nameGap: 42,
      scale: true,
      min: options.xAxisMin,
      max: options.xAxisMax,
    },
    yAxis: {
      type: "value",
      name: options.yAxisName || "Diversidad normalizada",
      nameLocation: "middle",
      nameGap: 44,
      scale: true,
      min: options.yAxisMin,
      max: options.yAxisMax,
    },
    graphic: comparatorEmptyChartGraphic(renderedSeries, "Sin datos para las propuestas filtradas."),
    series: renderedSeries,
  };
}

function comparatorProposalChartPoints(proposal, chartKey) {
  return (((proposal.charts || {})[chartKey]) || []).map((point) => ({
    ...point,
    instanceId: point.instanceId || comparatorEntityId(proposal),
    proposalId: point.proposalId || proposal.proposalId,
    displayName: point.displayName || proposal.displayName,
    baseDisplayName: point.baseDisplayName || proposal.baseDisplayName,
  }));
}

function comparatorAllNonDominatedPoints(proposals) {
  return proposals.flatMap((proposal) => comparatorProposalChartPoints(proposal, "nonDominated"));
}

function comparatorGlobalNonDominatedPoints(proposals) {
  return comparatorGlobalNonDominatedFront(comparatorAllNonDominatedPoints(proposals));
}

function comparatorChartPointData(point, color, options = {}) {
  const borderColor = options.globallyNonDominated ? "#dc2626" : color;
  const borderWidth = options.globallyNonDominated ? 3 : 0;
  return {
    value: [point.x, point.y],
    labelText: point.label,
    prompt: point.prompt,
    rank: point.rank,
    instanceId: point.instanceId,
    proposalId: point.proposalId,
    displayName: point.displayName,
    baseDisplayName: point.baseDisplayName,
    sourceIndex: point.sourceIndex,
    repetitionIndex: point.repetitionIndex,
    nativeObjectiveVector: point.nativeObjectiveVector,
    comparableObjectiveVector: point.comparableObjectiveVector,
    coordinateSpace: point.coordinateSpace,
    globallyNonDominated: Boolean(options.globallyNonDominated),
    itemStyle: { color, borderColor, borderWidth },
  };
}

function comparatorIdealSeries(series, fixedPoint = null) {
  const points = series.flatMap((item) => item.data || []);
  const coordinates = points.map(comparatorPointCoordinates).filter(Boolean);
  if (coordinates.length === 0) return null;
  const idealX = fixedPoint ? fixedPoint[0] : Math.max(1, ...coordinates.map((point) => point.x));
  const idealY = fixedPoint ? fixedPoint[1] : Math.max(1, ...coordinates.map((point) => point.y));
  return {
    name: "Punto ideal",
    type: "scatter",
    symbol: "star",
    symbolSize: 20,
    silent: false,
    data: [{ value: [idealX, idealY], labelText: "Referencia ideal visual", rank: "--" }],
    itemStyle: { color: "#f59e0b", borderColor: "#92400e", borderWidth: 1.5 },
    z: 5,
  };
}

function comparatorMetricReferenceLines(values, higherIsBetter) {
  const extremes = comparatorMetricExtremes(values, higherIsBetter);
  if (!extremes) return null;
  return {
    symbol: "none",
    silent: true,
    data: [
      {
        name: "Mejor",
        yAxis: extremes.bestValue,
        lineStyle: { color: "#16a34a", type: "dashed", width: 2 },
        label: { color: "#166534", formatter: "Mejor: {c}" },
      },
      {
        name: "Peor",
        yAxis: extremes.worstValue,
        lineStyle: { color: "#dc2626", type: "dashed", width: 2 },
        label: { color: "#991b1b", formatter: "Peor: {c}" },
      },
    ],
  };
}

function comparatorEmptyChartGraphic(series, message) {
  const hasData = series.some((item) => (item.data || []).length > 0);
  if (hasData) return [];
  return [
    {
      type: "text",
      left: "center",
      top: "middle",
      silent: true,
      style: {
        text: message,
        fill: "#64748b",
        fontSize: 13,
        fontWeight: 600,
      },
    },
  ];
}

function safeChartTooltip(trigger, formatter) {
  return {
    trigger,
    confine: true,
    appendToBody: false,
    borderColor: "#d7e1ea",
    extraCssText: [
      "max-width: 360px",
      "white-space: normal",
      "overflow-wrap: anywhere",
      "word-break: break-word",
      "line-height: 1.35",
      "box-shadow: 0 12px 32px rgba(15, 35, 55, 0.18)",
    ].join(";"),
    formatter,
  };
}

function comparatorScatterTooltipFormatter(params) {
  const data = params.data || {};
  const value = params.value || [];
  const frontNote = data.globallyNonDominated ? "<br><strong>Globalmente no dominada</strong>" : "";
  const native = Array.isArray(data.nativeObjectiveVector) && data.nativeObjectiveVector.length >= 2
    ? `<br>Vector nativo: [${formatOptionalNumber(data.nativeObjectiveVector[0], 6)}, ${formatOptionalNumber(data.nativeObjectiveVector[1], 6)}]`
    : "";
  return `
    <strong>${escapeHtml(params.seriesName)}</strong><br>
    F1 norm.: ${formatOptionalNumber(value[0], 6)}<br>
    F2 norm.: ${formatOptionalNumber(value[1], 6)}${native}<br>
    Rank: ${escapeHtml(String(data.rank ?? "--"))}${frontNote}<br>
    ${escapeHtml(String(data.labelText || "")).slice(0, 300)}
  `;
}

function comparatorRawScatterTooltipFormatter(params) {
  const data = params.data || {};
  const value = params.value || [];
  const normalized = Array.isArray(data.comparableObjectiveVector) && data.comparableObjectiveVector.length >= 2
    ? `<br>Vector normalizado: [${formatOptionalNumber(data.comparableObjectiveVector[0], 6)}, ${formatOptionalNumber(data.comparableObjectiveVector[1], 6)}]`
    : "";
  return `
    <strong>${escapeHtml(params.seriesName)}</strong><br>
    Fidelidad raw: ${formatOptionalNumber(value[0], 6)}<br>
    Diversidad raw: ${formatOptionalNumber(value[1], 6)}${normalized}<br>
    Rank: ${escapeHtml(String(data.rank ?? "--"))}<br>
    ${escapeHtml(String(data.labelText || "")).slice(0, 300)}
  `;
}

function comparatorLineTooltipFormatter(params) {
  const items = Array.isArray(params) ? params : [params];
  return items.map((item) => {
    const value = item.value || [];
    return `${item.marker || ""}${escapeHtml(item.seriesName)}: ${formatOptionalNumber(value[1], 6)}<br><small>Iteracion ${escapeHtml(String(value[0] ?? "--"))}</small>`;
  }).join("<br>");
}

function chartPalette(index) {
  return ["#2458b8", "#0f766e", "#b42318", "#7c3aed", "#ca8a04"][index % 5];
}

function activateComparatorTab(tabName) {
  document.querySelectorAll(".comparator-tab").forEach((button) => {
    const isActive = button.dataset.comparatorTab === tabName;
    button.classList.toggle("is-active", isActive);
    button.setAttribute("aria-selected", String(isActive));
  });
  document.querySelectorAll(".comparator-tab-panel").forEach((panel) => {
    const expectedId = `comparator${tabName[0].toUpperCase()}${tabName.slice(1)}Tab`;
    panel.classList.toggle("is-active", panel.id === expectedId);
  });
  window.setTimeout(() => comparatorCharts.forEach((chart) => chart.resize()), 0);
}

function renderComparatorRows(rows) {
  if (!rows.length) {
    dom.comparatorResultsBody.innerHTML = '<tr><td colspan="9">Sin resultados todavia.</td></tr>';
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
      const referenceText = row.referenceText
        || latestComparatorRun?.config?.referenceText
        || latestComparatorRun?.referenceText
        || "--";
      tr.innerHTML = `
        <td>${escapeHtml(row.displayName || row.proposalId)}</td>
        <td>${escapeHtml(String(row.rank ?? "--"))}</td>
        <td class="context-cell long-cell">${escapeHtml(referenceText)}</td>
        <td class="context-cell long-cell">${escapeHtml(row.generatedText || "--")}</td>
        <td>${escapeHtml(row.diagnosticObjectiveLabel || row.objectiveLabel || "--")}</td>
        <td>${escapeHtml(row.comparableObjectiveLabel || "--")}</td>
        <td class="context-cell long-cell">${escapeHtml(row.prompt || "--")}</td>
        <td><span class="${comparatorStatusClass(status)}">${escapeHtml(status)}</span></td>
        <td>${nonDominatedLabel}</td>
      `;
      return tr;
    }),
  );
}

function formatComparatorLogEntry(entry) {
  return `[${entry?.proposalId || "system"}] ${entry?.message || ""}`;
}

function renderComparatorLogLines() {
  if (!comparatorLogLines.length) {
    dom.comparatorLogOutput.textContent = comparatorLogLoading ? "Cargando log completo..." : "Sin logs todavia.";
    setComparatorLogCopyButton(false);
    return;
  }
  dom.comparatorLogOutput.textContent = comparatorLogLines.join("\n");
  setComparatorLogCopyButton(true);
}

function resetComparatorLogLoader(runId = null) {
  comparatorLogRunId = runId;
  comparatorLogOffset = 0;
  comparatorLogLines = [];
  comparatorLogLoading = false;
  comparatorLogComplete = false;
  comparatorLogTerminalRefreshDone = false;
  comparatorLogLoadToken += 1;
}

function renderComparatorLogs(run) {
  if (!run?.runId) {
    resetComparatorLogLoader(null);
    renderComparatorLogLines();
    return;
  }

  if (comparatorLogRunId !== run.runId) {
    resetComparatorLogLoader(run.runId);
    const summaryTail = (run.logs || []).map(formatComparatorLogEntry);
    if (summaryTail.length) {
      comparatorLogLines = [
        "Cargando log completo desde el backend...",
        "",
        ...summaryTail,
      ];
      setComparatorLogCopyButton(true);
    }
    renderComparatorLogLines();
  }

  if (comparatorLogComplete && !COMPARATOR_TERMINAL_STATUSES.has(run.status)) {
    comparatorLogComplete = false;
  }
  if (comparatorLogComplete && COMPARATOR_TERMINAL_STATUSES.has(run.status) && !comparatorLogTerminalRefreshDone) {
    comparatorLogComplete = false;
    comparatorLogTerminalRefreshDone = true;
  }

  if (!comparatorLogLoading && !comparatorLogComplete) {
    loadComparatorLogChunks(run.runId);
  }
}

async function loadComparatorLogChunks(runId) {
  if (!runId || comparatorLogLoading) return;
  comparatorLogLoading = true;
  const token = ++comparatorLogLoadToken;
  try {
    while (token === comparatorLogLoadToken && comparatorLogRunId === runId) {
      const payload = await requestComparatorJson(
        `/runs/${encodeURIComponent(runId)}/logs?offset=${encodeURIComponent(String(comparatorLogOffset))}&limit=${COMPARATOR_LOG_CHUNK_LIMIT}`,
      );
      if (token !== comparatorLogLoadToken || comparatorLogRunId !== runId) return;
      const chunk = Array.isArray(payload.logs) ? payload.logs : [];
      if (comparatorLogOffset === 0) {
        comparatorLogLines = [];
      }
      if (chunk.length) {
        comparatorLogLines.push(...chunk.map(formatComparatorLogEntry));
        comparatorLogOffset = Number(payload.nextOffset ?? comparatorLogOffset + chunk.length);
        renderComparatorLogLines();
      } else {
        comparatorLogOffset = Number(payload.nextOffset ?? comparatorLogOffset);
      }
      comparatorLogComplete = !payload.hasMore;
      if (comparatorLogComplete) {
        renderComparatorLogLines();
        return;
      }
      await new Promise((resolve) => window.setTimeout(resolve, 0));
    }
  } catch (error) {
    const fallbackLogs = latestComparatorRun?.runId === runId ? latestComparatorRun.logs || [] : [];
    if (!comparatorLogLines.length && fallbackLogs.length) {
      comparatorLogLines = fallbackLogs.map(formatComparatorLogEntry);
      renderComparatorLogLines();
    }
    comparatorLogComplete = true;
  } finally {
    comparatorLogLoading = false;
  }
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

window.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && !dom.turbulenceCandidateModal.hidden) {
    closeTurbulenceCandidateModal();
  }
  if (event.key === "Escape" && dom.comparatorInstanceModal && !dom.comparatorInstanceModal.hidden) {
    closeComparatorInstanceModal();
  }
  if (event.key === "Escape") {
    closeTurbulenceMetricPopover();
  }
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
dom.loadInitialComparisonModelsButton.addEventListener("click", loadInitialComparisonModels);
dom.runInitialComparisonButton.addEventListener("click", runInitialComparison);
dom.cancelInitialComparisonButton.addEventListener("click", cancelInitialComparisonRun);
dom.clearInitialComparisonButton.addEventListener("click", resetInitialComparisonUi);
dom.loadTurbulenceModelsButton.addEventListener("click", loadTurbulenceModels);
dom.turbulenceUsePpdb.addEventListener("change", refreshTurbulencePpdbStatus);
dom.refreshTurbulencePpdbButton.addEventListener("click", refreshTurbulencePpdbStatus);
dom.prepareTurbulencePpdbButton.addEventListener("click", prepareTurbulencePpdb);
dom.runTurbulenceComparisonButton.addEventListener("click", runTurbulenceComparison);
dom.cancelTurbulenceComparisonButton.addEventListener("click", cancelTurbulenceComparisonRun);
dom.clearTurbulenceComparisonButton.addEventListener("click", resetTurbulenceComparisonUi);
dom.closeTurbulenceCandidateModalButton.addEventListener("click", closeTurbulenceCandidateModal);
dom.turbulenceCandidateModal.addEventListener("click", (event) => {
  if (event.target === dom.turbulenceCandidateModal) {
    closeTurbulenceCandidateModal();
  }
});
dom.turbulenceCandidateModalBody.addEventListener("click", async (event) => {
  if (!(event.target instanceof Element)) return;
  const button = event.target.closest("[data-copy-diagnostic-prompt]");
  if (!button) return;
  const key = button.dataset.copyDiagnosticPrompt;
  const textarea = dom.turbulenceCandidateModalBody.querySelector(`[data-diagnostic-prompt="${CSS.escape(key)}"]`);
  if (!textarea) return;
  const originalText = button.textContent;
  try {
    await copyTextToClipboard(textarea.value);
    button.textContent = "Copiado";
  } catch (error) {
    button.textContent = "No copiado";
  } finally {
    window.setTimeout(() => {
      button.textContent = originalText;
    }, 1400);
  }
});
dom.runComparatorButton.addEventListener("click", runComparator);
dom.cancelComparatorButton.addEventListener("click", cancelComparatorRun);
dom.recomputeComparatorMetricsButton?.addEventListener("click", recomputeComparatorMetrics);
dom.clearComparatorButton.addEventListener("click", resetComparatorUi);
dom.resumeComparatorButton?.addEventListener("click", resumeComparatorRun);
dom.copyComparatorLogButton.addEventListener("click", copyComparatorLog);
dom.comparatorChartProposalFilters?.addEventListener("change", onComparatorChartFilterChange);
dom.comparatorExecutionMode.addEventListener("change", () => syncComparatorExecutionModeControls(false));
dom.comparatorProposalSelector?.addEventListener("click", (event) => {
  if (!(event.target instanceof Element)) return;
  const removeButton = event.target.closest("[data-remove-comparator-proposal]");
  if (removeButton) {
    removeComparatorProposalInstances(removeButton.dataset.removeComparatorProposal);
    return;
  }
  const button = event.target.closest("[data-add-comparator-instance]");
  if (!button) return;
  openComparatorInstanceModal(button.dataset.addComparatorInstance);
});
dom.comparatorProposalConfigPanels?.addEventListener("click", (event) => {
  if (!(event.target instanceof Element)) return;
  const editButton = event.target.closest("[data-edit-comparator-instance]");
  if (editButton) {
    const instance = comparatorInstances.find((item) => item.instanceId === editButton.dataset.editComparatorInstance);
    if (instance) openComparatorInstanceModal(instance.proposalId, instance.instanceId);
    return;
  }
  const duplicateButton = event.target.closest("[data-duplicate-comparator-instance]");
  if (duplicateButton) {
    const instance = comparatorInstances.find((item) => item.instanceId === duplicateButton.dataset.duplicateComparatorInstance);
    if (instance) openComparatorInstanceModal(instance.proposalId, instance.instanceId, true);
    return;
  }
  const deleteButton = event.target.closest("[data-delete-comparator-instance]");
  if (deleteButton) {
    deleteComparatorInstance(deleteButton.dataset.deleteComparatorInstance);
  }
});
dom.saveComparatorInstanceModalButton?.addEventListener("click", saveComparatorInstanceModal);
dom.cancelComparatorInstanceModalButton?.addEventListener("click", closeComparatorInstanceModal);
dom.comparatorInstanceModal?.addEventListener("click", (event) => {
  if (event.target === dom.comparatorInstanceModal) {
    closeComparatorInstanceModal();
  }
});
document.querySelectorAll(".comparator-tab").forEach((button) => {
  button.addEventListener("click", () => activateComparatorTab(button.dataset.comparatorTab));
});
window.addEventListener("resize", () => comparatorCharts.forEach((chart) => chart.resize()));
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
dom.turbulenceLmApiMode.addEventListener("change", () => {
  dom.turbulenceLlmModel.replaceChildren(new Option(TURBULENCE_LLM_DEFAULT_MODEL, TURBULENCE_LLM_DEFAULT_MODEL));
  dom.turbulenceComparisonConnectionText.textContent = "Sin ejecución";
  dom.turbulenceComparisonConnectionDot.classList.remove("is-error", "is-busy");
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
  dom.llmModelManual.value = OPERATOR_DEFAULT_MODELS[selectedPsoOperator()];
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
setupReferenceTextLibraryControls();
loadReferenceTextLibrary();
populateSolutionReferencePresets();
applySolutionReferencePreset();
resetSolutionMetrics();
resetInitialPopulationUi();
loadInitialPopulationStrategies();
buildInitialComparisonStageEditors();
resetInitialComparisonUi();
loadInitialComparisonStrategies();
resetTurbulenceComparisonUi();
refreshTurbulencePpdbStatus();
resetComparatorUi({ clearStoredRunId: false });
if (dom.comparatorResumeRunId) {
  dom.comparatorResumeRunId.value = loadStoredComparatorRunId();
}
loadComparatorProposals();
dom.renderedPromptPreview.textContent = renderPrompt();
updateSimulationModeUi();
decorateAbbreviationTooltips(document);
