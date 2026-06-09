import assert from "node:assert/strict";
import { test } from "node:test";

import {
  COMPARATOR_RAW_OBJECTIVE_BOUNDS,
  comparatorBestCostProposalIds,
  comparatorCountByProposal,
  comparatorGlobalNonDominatedFront,
  comparatorHypervolumeArea,
  comparatorIsGloballyNonDominated,
  comparatorMetricExtremes,
  comparatorMetricMetadata,
  comparatorRawChartPoints,
} from "../LLM/comparator_chart_helpers.mjs";

test("global non-dominated front is computed from all proposal points", () => {
  const points = [
    { proposalId: "evolmd", x: 0.7, y: 0.4, label: "dominated" },
    { proposalId: "evolmd-mo", x: 0.8, y: 0.5, label: "front-a" },
    { proposalId: "binary-mopso-cd", x: 0.6, y: 0.8, label: "front-b" },
    { proposalId: "binary-mopso-cd", x: 0.8, y: 0.5, label: "duplicate-front-a" },
  ];

  const front = comparatorGlobalNonDominatedFront(points);

  assert.deepEqual(front.map((point) => point.label), ["front-a", "front-b", "duplicate-front-a"]);
});

test("selected points are marked against the global non-dominated pool", () => {
  const pool = [
    { proposalId: "evolmd-mo", x: 0.8, y: 0.5 },
    { proposalId: "binary-mopso-cd", x: 0.6, y: 0.8 },
  ];

  assert.equal(comparatorIsGloballyNonDominated({ x: 0.7, y: 0.4 }, pool), false);
  assert.equal(comparatorIsGloballyNonDominated({ x: 0.7, y: 0.7 }, pool), true);
});

test("global front counts are grouped by proposal", () => {
  const counts = comparatorCountByProposal([
    { proposalId: "evolmd" },
    { proposalId: "binary-mopso-cd" },
    { proposalId: "binary-mopso-cd" },
  ]);

  assert.equal(counts.get("evolmd"), 1);
  assert.equal(counts.get("binary-mopso-cd"), 2);
});

test("raw chart points use native semantic objectives", () => {
  const points = comparatorRawChartPoints([
    {
      label: "row",
      value: [0.95, 0.2],
      nativeObjectiveVector: [0.9, 1.4],
      comparableObjectiveVector: [0.95, 0.7],
    },
  ]);

  assert.deepEqual(points[0].value, [0.9, 1.4]);
  assert.equal(points[0].x, 0.9);
  assert.equal(points[0].y, 1.4);
  assert.equal(points[0].coordinateSpace, "semantic_raw");
  assert.deepEqual(points[0].comparableObjectiveVector, [0.95, 0.7]);
});

test("raw chart points skip missing or invalid native vectors", () => {
  const points = comparatorRawChartPoints([
    { nativeObjectiveVector: [0.9] },
    { nativeObjectiveVector: ["bad", 0.4] },
    { value: [0.4, 0.5] },
    { nativeObjectiveVector: [-0.2, 1.1] },
  ]);

  assert.equal(points.length, 1);
  assert.deepEqual(points[0].value, [-0.2, 1.1]);
});

test("raw semantic chart bounds are fixed for cross-proposal comparison", () => {
  assert.deepEqual(COMPARATOR_RAW_OBJECTIVE_BOUNDS, {
    xMin: -1,
    xMax: 1,
    yMin: 0,
    yMax: 2,
  });
});

test("hypervolume area uses stepped front from reference origin", () => {
  const area = comparatorHypervolumeArea([
    { x: 0.5, y: 0.8 },
    { x: 0.9, y: 0.4 },
  ]);

  assert.deepEqual(area.lineData, [
    [0, 0.8],
    [0.5, 0.8],
    [0.5, 0.4],
    [0.9, 0.4],
    [0.9, 0],
  ]);
  assert.equal(Number(area.area.toFixed(6)), 0.56);
});

test("hypervolume area ignores dominated points", () => {
  const withDominated = comparatorHypervolumeArea([
    { x: 0.5, y: 0.8 },
    { x: 0.9, y: 0.4 },
    { x: 0.4, y: 0.3 },
  ]);
  const withoutDominated = comparatorHypervolumeArea([
    { x: 0.5, y: 0.8 },
    { x: 0.9, y: 0.4 },
  ]);

  assert.deepEqual(withDominated.lineData, withoutDominated.lineData);
  assert.equal(withDominated.area, withoutDominated.area);
});

test("hypervolume area collapses equal fidelity with maximum diversity", () => {
  const area = comparatorHypervolumeArea([
    { x: 0.5, y: 0.4 },
    { x: 0.5, y: 0.8 },
    { x: 0.9, y: 0.4 },
  ]);

  assert.deepEqual(area.lineData, [
    [0, 0.8],
    [0.5, 0.8],
    [0.5, 0.4],
    [0.9, 0.4],
    [0.9, 0],
  ]);
});

test("hypervolume area returns null without valid points", () => {
  assert.equal(comparatorHypervolumeArea([]), null);
  assert.equal(comparatorHypervolumeArea([{ x: "bad", y: 0.4 }]), null);
});

test("metric extremes respect best direction", () => {
  assert.deepEqual(comparatorMetricExtremes([0.2, 0.8, 0.4], true), {
    bestValue: 0.8,
    worstValue: 0.2,
  });
  assert.deepEqual(comparatorMetricExtremes([0.2, 0.8, 0.4], false), {
    bestValue: 0.2,
    worstValue: 0.8,
  });
});

test("diagnostic iteration metrics use higher-is-better metadata", () => {
  assert.deepEqual(comparatorMetricMetadata("globalInertia"), {
    description: "Mayor inercia indica mayor dispersion global de embeddings.",
    higherIsBetter: true,
  });
  assert.deepEqual(comparatorMetricMetadata("globalEntropy"), {
    description: "Mayor entropia indica mayor variedad conceptual global.",
    higherIsBetter: true,
  });
});

test("cost winners choose the lowest completed reported value", () => {
  const proposals = [
    { proposalId: "evolmd", status: "completed", cost: { llmCalls: 10 } },
    { proposalId: "evolmd-mo", status: "completed", cost: { llmCalls: 6 } },
    { proposalId: "binary", status: "failed", cost: { llmCalls: 1 } },
  ];
  const winners = comparatorBestCostProposalIds(proposals, {
    value: (_proposal, cost) => cost.llmCalls,
  }, { costsComparable: true });

  assert.deepEqual([...winners], ["evolmd-mo"]);
});

test("cost winners are disabled in exploratory mode", () => {
  const winners = comparatorBestCostProposalIds([
    { proposalId: "a", status: "completed", cost: { seconds: 1 } },
    { proposalId: "b", status: "completed", cost: { seconds: 2 } },
  ], {
    value: (_proposal, cost) => cost.seconds,
  }, { costsComparable: false });

  assert.equal(winners.size, 0);
});

test("cost winners do not treat missing reports as zero", () => {
  const winners = comparatorBestCostProposalIds([
    { proposalId: "a", status: "completed", cost: { totalTokens: 120 } },
    { proposalId: "b", status: "completed", cost: { totalTokens: 0 } },
  ], {
    value: (_proposal, cost) => cost.totalTokens,
    isReported: (_proposal, cost) => Number(cost.totalTokens) > 0,
    requireAllReported: true,
  }, { costsComparable: true });

  assert.equal(winners.size, 0);
});
