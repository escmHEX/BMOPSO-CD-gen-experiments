import { pipeline } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0";

const MODEL_ID = "Xenova/all-MiniLM-L6-v2";

let extractorPromise;

function getExtractor(requestId) {
  if (!extractorPromise) {
    extractorPromise = pipeline("feature-extraction", MODEL_ID, {
      progress_callback: (progress) => {
        self.postMessage({ type: "progress", id: requestId, progress });
      },
    });
  }

  return extractorPromise;
}

self.addEventListener("message", async (event) => {
  const { id, texts } = event.data;

  try {
    const extractor = await getExtractor(id);
    const output = await extractor(texts, { pooling: "mean", normalize: true });
    const embeddings = output.tolist();

    if (typeof output.dispose === "function") {
      output.dispose();
    }

    self.postMessage({
      type: "result",
      id,
      modelId: MODEL_ID,
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
