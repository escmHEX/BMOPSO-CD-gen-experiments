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

async function embedTexts(modelId, texts, requestId) {
  const config = getModelConfig(modelId);
  const vectors = new Array(texts.length);
  const missingTexts = [];
  const missingIndexes = [];
  const seenMissing = new Map();

  texts.forEach((text, index) => {
    const key = cacheKey(modelId, text);
    if (embeddingCache.has(key)) {
      vectors[index] = embeddingCache.get(key);
      return;
    }

    if (seenMissing.has(text)) {
      missingIndexes.push({ originalIndex: index, missingIndex: seenMissing.get(text) });
      return;
    }

    seenMissing.set(text, missingTexts.length);
    missingIndexes.push({ originalIndex: index, missingIndex: missingTexts.length });
    missingTexts.push(text);
  });

  if (missingTexts.length > 0) {
    const extractor = await getExtractor(modelId, requestId);
    const output = await extractor(missingTexts, config);
    const computed = output.tolist();

    if (typeof output.dispose === "function") {
      output.dispose();
    }

    missingTexts.forEach((text, index) => {
      embeddingCache.set(cacheKey(modelId, text), computed[index]);
    });

    missingIndexes.forEach(({ originalIndex, missingIndex }) => {
      vectors[originalIndex] = computed[missingIndex];
    });
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
