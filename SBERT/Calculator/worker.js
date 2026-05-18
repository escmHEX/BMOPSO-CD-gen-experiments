import { pipeline } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0";

const SUPPORTED_MODELS = new Set([
  "Xenova/all-MiniLM-L6-v2",
  "Xenova/gte-small",
]);

const extractorPromises = new Map();

function getExtractor(modelId, requestId) {
  if (!SUPPORTED_MODELS.has(modelId)) {
    throw new Error(`Modelo no soportado: ${modelId}`);
  }

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

async function embedText(modelId, text, requestId) {
  const extractor = await getExtractor(modelId, requestId);
  const output = await extractor(text, { pooling: "mean", normalize: true });
  const vector = tensorToVector(output.tolist());

  if (typeof output.dispose === "function") {
    output.dispose();
  }

  return vector;
}

self.addEventListener("message", async (event) => {
  const { id, modelId, texts } = event.data;

  try {
    const embeddings = [];
    for (const text of texts) {
      embeddings.push(await embedText(modelId, text, id));
    }

    self.postMessage({
      type: "result",
      id,
      modelId,
      embeddings,
    });
  } catch (error) {
    self.postMessage({
      type: "error",
      id,
      message: error instanceof Error ? error.message : String(error),
    });
  }
});
