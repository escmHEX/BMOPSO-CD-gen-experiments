import { pipeline } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0";

const MODEL_CONFIGS = {
  "Xenova/all-MiniLM-L6-v2": { pooling: "mean", normalize: true },
  "Xenova/gte-small": { pooling: "mean", normalize: true },
};

const extractorPromises = new Map();
const embeddingCache = new Map();

function cacheKey(modelId, text) {
  return `${modelId}\n${text}`;
}

function getModelConfig(modelId) {
  const config = MODEL_CONFIGS[modelId];
  if (!config) {
    throw new Error(`Modelo de embedding no soportado: ${modelId}`);
  }
  return config;
}

function getExtractor(modelId, requestId) {
  getModelConfig(modelId);

  if (!extractorPromises.has(modelId)) {
    extractorPromises.set(modelId, pipeline("feature-extraction", modelId, {
      progress_callback: (progress) => {
        self.postMessage({ type: "progress", id: requestId, progress });
      },
    }));
  }

  return extractorPromises.get(modelId);
}

function tensorToVector(values) {
  return Array.isArray(values[0]) ? values[0] : values;
}

async function embedSingleText(modelId, text, requestId) {
  const key = cacheKey(modelId, text);
  if (embeddingCache.has(key)) {
    return embeddingCache.get(key);
  }

  const config = getModelConfig(modelId);
  const extractor = await getExtractor(modelId, requestId);
  const output = await extractor(text, config);
  const vector = tensorToVector(output.tolist());

  if (typeof output.dispose === "function") {
    output.dispose();
  }

  embeddingCache.set(key, vector);
  return vector;
}

async function embedTexts(modelId, texts, requestId) {
  getModelConfig(modelId);
  const requestCache = new Map();
  const vectors = [];

  for (const text of texts) {
    if (!requestCache.has(text)) {
      requestCache.set(text, await embedSingleText(modelId, text, requestId));
    }
    vectors.push(requestCache.get(text));
  }

  return vectors;
}

self.addEventListener("message", async (event) => {
  const { id, modelId, texts } = event.data;

  try {
    const embeddings = await embedTexts(modelId, texts, id);
    self.postMessage({ type: "result", id, modelId, embeddings });
  } catch (error) {
    self.postMessage({
      type: "error",
      id,
      message: error instanceof Error ? error.message : String(error),
    });
  }
});
