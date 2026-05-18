const worker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });

const form = document.querySelector("#calculatorForm");
const modelLabel = document.querySelector("#modelLabel");
const modelSelect = document.querySelector("#modelSelect");
const textA = document.querySelector("#textA");
const textB = document.querySelector("#textB");
const calculateButton = document.querySelector("#calculateButton");
const clearButton = document.querySelector("#clearButton");
const statusTone = document.querySelector("#statusTone");
const statusTitle = document.querySelector("#statusTitle");
const statusDetail = document.querySelector("#statusDetail");
const similarityValue = document.querySelector("#similarityValue");
const distanceValue = document.querySelector("#distanceValue");
const dimensionValue = document.querySelector("#dimensionValue");
const normAValue = document.querySelector("#normAValue");
const normBValue = document.querySelector("#normBValue");
const previewA = document.querySelector("#previewA");
const previewB = document.querySelector("#previewB");
const fullA = document.querySelector("#fullA");
const fullB = document.querySelector("#fullB");
const copyAButton = document.querySelector("#copyAButton");
const copyBButton = document.querySelector("#copyBButton");
const runtimeConfig = document.querySelector("#runtimeConfig");
const modelDetails = document.querySelector("#modelDetails");

const VECTOR_PREVIEW_SIZE = 16;
const MODEL_OPTIONS = {
  "Xenova/all-MiniLM-L6-v2": {
    displayName: "all-MiniLM-L6-v2",
    family: "Sentence Transformers / SBERT",
    baseModel: "sentence-transformers/all-MiniLM-L6-v2",
    mtebAverage: "56,26",
    mtebRetrieval: "41,95",
    dimensions: "384",
    maxInput: "256 word pieces por defecto en el model card",
    approximateCost: "Muy bajo: ONNX quantized aprox. 24,5 MB",
    language: "Inglés",
    notes: "Modelo base rápido para similitud semántica general.",
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
    notes: "Mejor calidad promedio que all-MiniLM-L6-v2 y compatible con Transformers.js.",
  },
};

const PIPELINE_CONFIG = {
  task: "feature-extraction",
  pooling: "mean",
  normalize: "true",
  dtype: "default de Transformers.js para navegador",
  execution: "Web Worker en el navegador",
};

let activeRequestId = 0;
let latestEmbeddings = null;

function selectedModelId() {
  return modelSelect.value;
}

function selectedModel() {
  return MODEL_OPTIONS[selectedModelId()];
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

function renderModelInfo() {
  const model = selectedModel();
  modelLabel.textContent = model.family;

  renderDefinitionList(runtimeConfig, [
    ["Pipeline", PIPELINE_CONFIG.task],
    ["Pooling", PIPELINE_CONFIG.pooling],
    ["Normalización", PIPELINE_CONFIG.normalize],
    ["Precisión", PIPELINE_CONFIG.dtype],
    ["Ejecución", PIPELINE_CONFIG.execution],
  ]);

  renderDefinitionList(modelDetails, [
    ["Modelo", `${model.displayName} (${selectedModelId()})`],
    ["Base", model.baseModel],
    ["Idioma", model.language],
    ["MTEB Average (56)", model.mtebAverage],
    ["MTEB Retrieval (15)", model.mtebRetrieval],
    ["Dimensión", model.dimensions],
    ["Entrada máxima", model.maxInput],
    ["Costo aproximado", model.approximateCost],
    ["Nota", model.notes],
  ]);
}

function setStatus(title, detail, state = "ready") {
  statusTitle.textContent = title;
  statusDetail.textContent = detail;
  statusTone.classList.toggle("is-busy", state === "busy");
  statusTone.classList.toggle("is-error", state === "error");
}

function setBusy(isBusy) {
  calculateButton.disabled = isBusy;
  calculateButton.textContent = isBusy ? "Calculando..." : "Calcular";
}

function resetResults() {
  latestEmbeddings = null;
  similarityValue.textContent = "--";
  distanceValue.textContent = "--";
  dimensionValue.textContent = "--";
  normAValue.textContent = "Norma --";
  normBValue.textContent = "Norma --";
  previewA.textContent = "Ejecuta el cálculo para ver el vector.";
  previewB.textContent = "Ejecuta el cálculo para ver el vector.";
  fullA.textContent = "";
  fullB.textContent = "";
  copyAButton.disabled = true;
  copyBButton.disabled = true;
}

function formatNumber(value, digits = 6) {
  return new Intl.NumberFormat("es-CL", {
    maximumFractionDigits: digits,
    minimumFractionDigits: digits,
  }).format(value);
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
  const rawSimilarity = dotProduct(vectorA, vectorB) / denominator;
  return Math.min(1, Math.max(-1, rawSimilarity));
}

function vectorPreview(vector) {
  return JSON.stringify(vector.slice(0, VECTOR_PREVIEW_SIZE), null, 2);
}

function fullVector(vector) {
  return JSON.stringify(vector, null, 2);
}

function renderResult(embeddings) {
  const [embeddingA, embeddingB] = embeddings;
  const similarity = cosineSimilarity(embeddingA, embeddingB);
  const distance = 1 - similarity;
  const normA = vectorNorm(embeddingA);
  const normB = vectorNorm(embeddingB);

  latestEmbeddings = embeddings;
  similarityValue.textContent = formatNumber(similarity);
  distanceValue.textContent = formatNumber(distance);
  dimensionValue.textContent = String(embeddingA.length);
  normAValue.textContent = `Norma ${formatNumber(normA, 4)}`;
  normBValue.textContent = `Norma ${formatNumber(normB, 4)}`;
  previewA.textContent = `${vectorPreview(embeddingA)}\n\nPrimeros ${VECTOR_PREVIEW_SIZE} valores de ${embeddingA.length}.`;
  previewB.textContent = `${vectorPreview(embeddingB)}\n\nPrimeros ${VECTOR_PREVIEW_SIZE} valores de ${embeddingB.length}.`;
  fullA.textContent = fullVector(embeddingA);
  fullB.textContent = fullVector(embeddingB);
  copyAButton.disabled = false;
  copyBButton.disabled = false;
}

function formatProgress(progress) {
  if (!progress || typeof progress !== "object") {
    return "Preparando recursos del modelo.";
  }

  const status = progress.status ? String(progress.status) : "progreso";
  const file = progress.file ? `: ${progress.file}` : "";
  const percentage = typeof progress.progress === "number"
    ? ` (${Math.round(progress.progress)}%)`
    : "";

  return `${status}${file}${percentage}`;
}

function validateInputs() {
  const firstText = textA.value.trim();
  const secondText = textB.value.trim();

  if (!firstText || !secondText) {
    throw new Error("Ingresa ambos textos antes de calcular.");
  }

  return [firstText, secondText];
}

async function copyVector(index, button) {
  if (!latestEmbeddings) {
    return;
  }

  await navigator.clipboard.writeText(fullVector(latestEmbeddings[index]));
  const previousText = button.textContent;
  button.textContent = "Copiado";
  window.setTimeout(() => {
    button.textContent = previousText;
  }, 1200);
}

form.addEventListener("submit", (event) => {
  event.preventDefault();

  let texts;
  try {
    texts = validateInputs();
  } catch (error) {
    setStatus("Falta información", error.message, "error");
    return;
  }

  activeRequestId += 1;
  const model = selectedModel();
  const modelId = selectedModelId();
  resetResults();
  setBusy(true);
  setStatus("Cargando modelo", `Preparando ${model.displayName}.`, "busy");
  worker.postMessage({ id: activeRequestId, modelId, texts });
});

clearButton.addEventListener("click", () => {
  textA.value = "";
  textB.value = "";
  resetResults();
  setStatus("Listo para calcular", "Ingresa dos textos para generar sus embeddings.", "ready");
  textA.focus();
});

modelSelect.addEventListener("change", () => {
  renderModelInfo();
  resetResults();
  setStatus("Modelo cambiado", `Seleccionaste ${selectedModel().displayName}. Ejecuta el cálculo nuevamente.`, "ready");
});

copyAButton.addEventListener("click", () => {
  copyVector(0, copyAButton).catch((error) => {
    setStatus("No se pudo copiar", error.message, "error");
  });
});

copyBButton.addEventListener("click", () => {
  copyVector(1, copyBButton).catch((error) => {
    setStatus("No se pudo copiar", error.message, "error");
  });
});

worker.addEventListener("message", (event) => {
  const message = event.data;
  if (message.id && message.id !== activeRequestId) {
    return;
  }

  if (message.type === "progress") {
    setStatus("Descargando o preparando modelo", formatProgress(message.progress), "busy");
    return;
  }

  if (message.type === "result") {
    renderResult(message.embeddings);
    setBusy(false);
    setStatus("Cálculo completado", `Embeddings generados con ${MODEL_OPTIONS[message.modelId].displayName}.`, "ready");
    return;
  }

  if (message.type === "error") {
    setBusy(false);
    setStatus("Error al calcular", message.message, "error");
  }
});

worker.addEventListener("error", (event) => {
  setBusy(false);
  setStatus("Error del worker", event.message, "error");
});

renderModelInfo();
