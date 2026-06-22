import assert from "node:assert/strict";
import { test } from "node:test";

import {
  comparatorBenchmarkProposals,
  comparatorBestCostProposalIds,
  comparatorBestMetricProposalIds,
  comparatorCountByProposal,
  comparatorGlobalNonDominatedFront,
  comparatorHypervolumeArea,
  comparatorIsBinaryProposal,
  comparatorIsGloballyNonDominated,
  comparatorMetricCellClassName,
  comparatorMetricExtremes,
  comparatorMetricMetadata,
  comparatorProposalColor,
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

test("global front counts are grouped by instance when available", () => {
  const counts = comparatorCountByProposal([
    { proposalId: "binary-mopso-cd", instanceId: "binary-a" },
    { proposalId: "binary-mopso-cd", instanceId: "binary-b" },
    { proposalId: "binary-mopso-cd", instanceId: "binary-b" },
  ]);

  assert.equal(counts.get("binary-a"), 1);
  assert.equal(counts.get("binary-b"), 2);
  assert.equal(counts.get("binary-mopso-cd"), undefined);
});

test("benchmark table orders Binary MOPSO-CD first", () => {
  const ordered = comparatorBenchmarkProposals([
    { proposalId: "mesap", displayName: "MESAP" },
    { proposalId: "evolmd-mo", displayName: "EVOLMD-MO" },
    { proposalId: "binary-mopso-cd", displayName: "Binary MOPSO-CD" },
    { proposalId: "evolmd", displayName: "EVOLMD" },
  ]);

  assert.equal(ordered[0].proposalId, "binary-mopso-cd");
  assert.deepEqual(ordered.slice(1).map((proposal) => proposal.displayName), ["EVOLMD", "EVOLMD-MO", "MESAP"]);
});

test("benchmark metric winner cell uses strong visual classes", () => {
  assert.equal(
    comparatorMetricCellClassName({ primaryColumn: true, best: true }),
    "is-primary-proposal metric-best comparator-cost-best comparator-metric-best-cell",
  );
  assert.equal(comparatorMetricCellClassName({ primaryColumn: false, best: true }), "metric-best comparator-cost-best comparator-metric-best-cell");
  assert.equal(comparatorMetricCellClassName({ primaryColumn: true, best: false }), "is-primary-proposal");
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
  assert.deepEqual(comparatorMetricMetadata("contribution"), {
    description: "Mayor Contribution indica mayor aporte al frente combinado P*.",
    higherIsBetter: true,
  });
  assert.deepEqual(comparatorMetricMetadata("extent"), {
    description: "Mayor Extent indica mayor cobertura del frente comparable.",
    higherIsBetter: true,
  });
  assert.deepEqual(comparatorMetricMetadata("unaryEntropy"), {
    description: "Mayor Unary Entropy indica mejor distribucion del frente comparable.",
    higherIsBetter: true,
  });
  assert.deepEqual(comparatorMetricMetadata("globalInertia"), {
    description: "Mayor K-means inertia indica mayor dispersion global de embeddings.",
    higherIsBetter: true,
  });
  assert.deepEqual(comparatorMetricMetadata("globalEntropy"), {
    description: "Mayor entity entropy indica mayor variedad conceptual o semantica.",
    higherIsBetter: true,
  });
});

test("binary proposal always uses configured green chart color", () => {
  assert.equal(comparatorIsBinaryProposal({ proposalId: "binary-mopso-cd" }), true);
  assert.equal(comparatorIsBinaryProposal({ instanceId: "binary-mopso-cd:2", proposalId: "binary-mopso-cd" }), true);
  assert.equal(comparatorProposalColor({ proposalId: "binary-mopso-cd" }, 3), "#2A8C00");
  assert.equal(comparatorProposalColor({ instanceId: "binary-mopso-cd:2", proposalId: "binary-mopso-cd" }, 4), "#2A8C00");
  assert.notEqual(comparatorProposalColor({ proposalId: "evolmd-mo" }, 0), "#2A8C00");
});

test("quality winners do not depend on cost comparability", () => {
  const winners = comparatorBestMetricProposalIds([
    { proposalId: "binary-mopso-cd", status: "completed", metrics: { hypervolume: 0.8 } },
    { proposalId: "evolmd", status: "completed", metrics: { hypervolume: 0.7 } },
  ], {
    kind: "quality",
    direction: "max",
    value: (proposal) => proposal.metrics.hypervolume,
  }, { costsComparable: false });

  assert.deepEqual([...winners], ["binary-mopso-cd"]);
});

test("cost winners remain disabled when costs are not comparable", () => {
  const winners = comparatorBestMetricProposalIds([
    { proposalId: "binary-mopso-cd", status: "completed", cost: { promptEvalCount: 120, hasTokenReport: true } },
    { proposalId: "evolmd", status: "completed", cost: { promptEvalCount: 80, hasTokenReport: true } },
  ], {
    kind: "cost",
    direction: "min",
    value: (_proposal, cost) => cost.promptEvalCount,
    isReported: (_proposal, cost) => cost.hasTokenReport,
  }, { costsComparable: false });

  assert.equal(winners.size, 0);
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

test("cost winners use instance ids when present", () => {
  const proposals = [
    { proposalId: "binary-mopso-cd", instanceId: "binary-a", status: "completed", cost: { seconds: 8 } },
    { proposalId: "binary-mopso-cd", instanceId: "binary-b", status: "completed", cost: { seconds: 3 } },
  ];
  const winners = comparatorBestCostProposalIds(proposals, {
    value: (_proposal, cost) => cost.seconds,
  }, { costsComparable: true });

  assert.deepEqual([...winners], ["binary-b"]);
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
