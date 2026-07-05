const GPT55_STANDARD_SHORT_INPUT_USD_PER_1M = 5;
const GPT55_STANDARD_SHORT_OUTPUT_USD_PER_1M = 30;
const HAIKU45_INPUT_USD_PER_1M = 1;
const HAIKU45_OUTPUT_USD_PER_1M = 5;
const TOKENS_PER_MILLION = 1_000_000;

function nonNegativeFiniteNumber(value) {
  const number = Number(value);
  return Number.isFinite(number) && number > 0 ? number : 0;
}

function hasSplitTokenReport(cost = {}) {
  return Boolean(cost.hasTokenReport)
    || Number(cost.promptEvalCount) > 0
    || Number(cost.evalCount) > 0;
}

function trimUsdDecimals(text, minimumDecimals = 2) {
  const [integerPart, decimalPart = ""] = text.split(".");
  if (!decimalPart) return `${integerPart}.${"0".repeat(minimumDecimals)}`;
  let trimmed = decimalPart.replace(/0+$/, "");
  if (trimmed.length < minimumDecimals) {
    trimmed = trimmed.padEnd(minimumDecimals, "0");
  }
  return `${integerPart}.${trimmed}`;
}

function comparatorCommercialExecutionCost(cost = {}, inputUsdPer1M, outputUsdPer1M) {
  if (!hasSplitTokenReport(cost)) return null;
  const inputTokens = nonNegativeFiniteNumber(cost.promptEvalCount);
  const outputTokens = nonNegativeFiniteNumber(cost.evalCount);
  const value = (
    (inputTokens * inputUsdPer1M)
    + (outputTokens * outputUsdPer1M)
  ) / TOKENS_PER_MILLION;
  return Number(value.toFixed(12));
}

export function comparatorGpt55CommercialExecutionCost(cost = {}) {
  return comparatorCommercialExecutionCost(
    cost,
    GPT55_STANDARD_SHORT_INPUT_USD_PER_1M,
    GPT55_STANDARD_SHORT_OUTPUT_USD_PER_1M,
  );
}

export function comparatorHaiku45CommercialExecutionCost(cost = {}) {
  return comparatorCommercialExecutionCost(
    cost,
    HAIKU45_INPUT_USD_PER_1M,
    HAIKU45_OUTPUT_USD_PER_1M,
  );
}

function formatComparatorCommercialExecutionCost(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "No reportado";
  if (number === 0) return "USD $0.00";
  if (Math.abs(number) < 0.01) {
    return `USD $${trimUsdDecimals(number.toFixed(6))}`;
  }
  if (Math.abs(number) < 1) {
    return `USD $${trimUsdDecimals(number.toFixed(4))}`;
  }
  return `USD $${number.toFixed(2)}`;
}

export const formatComparatorGpt55CommercialExecutionCost = formatComparatorCommercialExecutionCost;
export const formatComparatorHaiku45CommercialExecutionCost = formatComparatorCommercialExecutionCost;
