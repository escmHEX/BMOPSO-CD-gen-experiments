import assert from "node:assert/strict";
import { test } from "node:test";

import {
  comparatorGpt55CommercialExecutionCost,
  comparatorHaiku45CommercialExecutionCost,
  formatComparatorGpt55CommercialExecutionCost,
  formatComparatorHaiku45CommercialExecutionCost,
} from "../LLM/comparator_cost_helpers.mjs";

test("gpt-5.5 commercial cost uses standard short-context input and output prices", () => {
  const cost = comparatorGpt55CommercialExecutionCost({
    hasTokenReport: true,
    promptEvalCount: 100_000,
    evalCount: 10_000,
  });

  assert.equal(cost, 0.8);
});

test("gpt-5.5 commercial cost keeps short-context prices for aggregated proposal tokens", () => {
  const cost = comparatorGpt55CommercialExecutionCost({
    hasTokenReport: true,
    promptEvalCount: 272_001,
    evalCount: 1_000,
  });

  assert.equal(cost, 1.390005);
});

test("gpt-5.5 commercial cost is unavailable without reported tokens", () => {
  const cost = comparatorGpt55CommercialExecutionCost({
    promptEvalCount: 0,
    evalCount: 0,
  });

  assert.equal(cost, null);
});

test("gpt-5.5 commercial cost keeps zero when tokens were reported", () => {
  const cost = comparatorGpt55CommercialExecutionCost({
    hasTokenReport: true,
    promptEvalCount: 0,
    evalCount: 0,
  });

  assert.equal(cost, 0);
  assert.equal(formatComparatorGpt55CommercialExecutionCost(cost), "USD $0.00");
});

test("gpt-5.5 commercial cost format keeps small non-zero costs visible", () => {
  const cost = comparatorGpt55CommercialExecutionCost({
    hasTokenReport: true,
    promptEvalCount: 10,
    evalCount: 2,
  });

  assert.equal(cost, 0.00011);
  assert.equal(formatComparatorGpt55CommercialExecutionCost(cost), "USD $0.00011");
});

test("haiku 4.5 commercial cost uses input and output prices", () => {
  const cost = comparatorHaiku45CommercialExecutionCost({
    hasTokenReport: true,
    promptEvalCount: 100_000,
    evalCount: 10_000,
  });

  assert.equal(cost, 0.15);
});

test("haiku 4.5 commercial cost format keeps small non-zero costs visible", () => {
  const cost = comparatorHaiku45CommercialExecutionCost({
    hasTokenReport: true,
    promptEvalCount: 10,
    evalCount: 2,
  });

  assert.equal(cost, 0.00002);
  assert.equal(formatComparatorHaiku45CommercialExecutionCost(cost), "USD $0.00002");
});
