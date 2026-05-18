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

self.addEventListener("message", async (event) => {
  const { id, modelId, texts } = event.data;

  try {
    const extractor = await getExtractor(modelId, id);
    const output = await extractor(texts, { pooling: "mean", normalize: true });
    const embeddings = output.tolist();

    if (typeof output.dispose === "function") {
      output.dispose();
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
