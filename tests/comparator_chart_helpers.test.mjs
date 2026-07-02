import assert from "node:assert/strict";
import { test } from "node:test";

import {
  comparatorBmopsoInternalAnalyses,
  comparatorBenchmarkProposals,
  comparatorApplyColumnOrder,
  comparatorBestCostProposalIds,
  comparatorBestMetricProposalIds,
  comparatorHasBmopsoInternalAnalysis,
  comparatorCanRecontinueRun,
  comparatorCountByProposal,
  comparatorGlobalNonDominatedFront,
  comparatorFrontDiagnostics,
  comparatorHypervolumeArea,
  comparatorKMeansInertia,
  comparatorChartAxisWindow,
  comparatorEntityEntropy,
  comparatorExpandedAxisWindow,
  comparatorIsBinaryProposal,
  comparatorIsGloballyNonDominated,
  comparatorMetricCellClassName,
  comparatorMetricDeltaLabel,
  comparatorMetricDeltaPercent,
  comparatorMoveColumnId,
  comparatorMetricMeanStdDevLabel,
  comparatorMetricExtremes,
  comparatorMetricMetadata,
  comparatorMetricReferenceLinePatch,
  comparatorActivePointCount,
  comparatorLimitSeriesToIteration,
  comparatorPartitionPointsByExclusion,
  comparatorPointChartRepetitionOptions,
  comparatorPointChartViewProposal,
  comparatorPointInteractionKey,
  comparatorProposalChartStyleAssignments,
  comparatorProposalColor,
  comparatorSeriesIterationExtent,
  comparatorUnaryEntropy,
  comparatorVisibleFrontPointCount,
  comparatorVisibleFrontChartPoints,
} from "../LLM/comparator_chart_helpers.mjs";

function rgbDistance(left, right) {
  const leftRgb = hexToRgb(left);
  const rightRgb = hexToRgb(right);
  return Math.sqrt(
    ((leftRgb.r - rightRgb.r) ** 2)
    + ((leftRgb.g - rightRgb.g) ** 2)
    + ((leftRgb.b - rightRgb.b) ** 2),
  );
}

function hexToRgb(color) {
  const value = String(color || "").replace("#", "");
  return {
    r: Number.parseInt(value.slice(0, 2), 16),
    g: Number.parseInt(value.slice(2, 4), 16),
    b: Number.parseInt(value.slice(4, 6), 16),
  };
}

test("chart axis window expands the default data scale by the requested zoom factor", () => {
  const window = comparatorChartAxisWindow([
    { value: [2, 10] },
    { value: [4, 20] },
  ], "x", { paddingRatio: 0.1, zoomFactor: 100 });

  assert.deepEqual(window, {
    defaultMin: 1.8,
    defaultMax: 4.2,
    zoomMin: -117,
    zoomMax: 123,
  });
});

test("chart axis window gives a non-zero default range for a single point", () => {
  const window = comparatorChartAxisWindow([{ x: 5, y: 0.5 }], "y", {
    paddingRatio: 0.1,
    zoomFactor: 100,
    minSpan: 0.2,
  });

  assert.equal(window.defaultMin, 0.4);
  assert.equal(window.defaultMax, 0.6);
  assert.equal(window.zoomMax - window.zoomMin, 20);
});

test("chart axis window ignores non-finite values", () => {
  const window = comparatorChartAxisWindow([
    { value: [Number.NaN, 1] },
    { value: [3, 2] },
    { value: [Number.POSITIVE_INFINITY, 3] },
  ], "x", { paddingRatio: 0, zoomFactor: 10 });

  assert.deepEqual(window, {
    defaultMin: 2.5,
    defaultMax: 3.5,
    zoomMin: -2,
    zoomMax: 8,
  });
});

test("expanded chart axis window uses explicit default bounds", () => {
  const window = comparatorExpandedAxisWindow(10, 12, { zoomFactor: 100 });

  assert.deepEqual(window, {
    defaultMin: 10,
    defaultMax: 12,
    zoomMin: -89,
    zoomMax: 111,
  });
});

test("metric reference line patch hides and restores the target series markLine", () => {
  const referenceLines = {
    symbol: "none",
    silent: true,
    data: [{ name: "Mejor", yAxis: 0.8 }, { name: "Peor", yAxis: 0.2 }],
  };

  assert.deepEqual(comparatorMetricReferenceLinePatch("hv:evolmd", referenceLines, true), [{
    id: "hv:evolmd",
    markLine: {
      ...referenceLines,
      data: [],
    },
  }]);
  assert.deepEqual(comparatorMetricReferenceLinePatch("hv:evolmd", referenceLines, false), [{
    id: "hv:evolmd",
    markLine: referenceLines,
  }]);
});

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

test("benchmark column order applies saved ids and appends new proposals in base order", () => {
  const proposals = comparatorBenchmarkProposals([
    { proposalId: "mesap", displayName: "MESAP" },
    { proposalId: "binary-mopso-cd", instanceId: "binary-a", displayName: "Binary A" },
    { proposalId: "binary-mopso-cd", instanceId: "binary-b", displayName: "Binary B" },
    { proposalId: "evolmd", displayName: "EVOLMD" },
  ]);

  const ordered = comparatorApplyColumnOrder(proposals, ["binary-b", "missing", "mesap"]);

  assert.deepEqual(ordered.map((proposal) => proposal.instanceId || proposal.proposalId), [
    "binary-b",
    "mesap",
    "binary-a",
    "evolmd",
  ]);
});

test("benchmark column move reorders dragged id before target id", () => {
  assert.deepEqual(
    comparatorMoveColumnId(["binary-a", "binary-b", "mesap"], "mesap", "binary-a"),
    ["mesap", "binary-a", "binary-b"],
  );
  assert.deepEqual(
    comparatorMoveColumnId(["binary-a", "binary-b", "mesap"], "binary-a", "mesap"),
    ["binary-b", "binary-a", "mesap"],
  );
  assert.deepEqual(
    comparatorMoveColumnId(["binary-a", "binary-b", "mesap"], "binary-a", "mesap", "after"),
    ["binary-b", "mesap", "binary-a"],
  );
});

test("benchmark metric winner cell uses strong visual classes", () => {
  assert.equal(
    comparatorMetricCellClassName({ primaryColumn: true, best: true }),
    "is-primary-proposal metric-best comparator-cost-best comparator-metric-best-cell",
  );
  assert.equal(comparatorMetricCellClassName({ primaryColumn: false, best: true }), "metric-best comparator-cost-best comparator-metric-best-cell");
  assert.equal(comparatorMetricCellClassName({ primaryColumn: true, best: false }), "is-primary-proposal");
});

test("metric delta percent treats higher values as better for max metrics", () => {
  assert.equal(comparatorMetricDeltaPercent(0.5, 0.6, "max"), 20);
  assert.equal(comparatorMetricDeltaPercent(0.5, 0.4, "max"), -20);
});

test("metric delta percent treats lower values as better for min metrics", () => {
  assert.equal(comparatorMetricDeltaPercent(100, 80, "min"), 20);
  assert.equal(comparatorMetricDeltaPercent(100, 125, "min"), -25);
});

test("metric delta percent returns null for invalid or zero baselines", () => {
  assert.equal(comparatorMetricDeltaPercent(0, 1, "max"), null);
  assert.equal(comparatorMetricDeltaPercent(Number.NaN, 1, "max"), null);
  assert.equal(comparatorMetricDeltaPercent(1, Number.POSITIVE_INFINITY, "max"), null);
});

test("metric delta label keeps explicit improvement signs", () => {
  assert.equal(comparatorMetricDeltaLabel(20), "+20%");
  assert.equal(comparatorMetricDeltaLabel(-8.4), "-8%");
  assert.equal(comparatorMetricDeltaLabel(0), "0%");
});

test("metric mean standard deviation label appends explicit SD when available", () => {
  assert.equal(comparatorMetricMeanStdDevLabel("0.300000", "0.141421"), "0.300000 ± 0.141421");
  assert.equal(comparatorMetricMeanStdDevLabel("0.300000", "No aplica"), "0.300000");
  assert.equal(comparatorMetricMeanStdDevLabel("0.300000", ""), "0.300000");
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

test("front chart points include only non-dominated individuals and selected front points", () => {
  const charts = {
    pareto: [
      {
        x: 0.8,
        y: 0.5,
        instanceId: "evolmd-mo-1",
        proposalId: "evolmd-mo",
        sourceIndex: 1,
        rank: 2,
        label: "front point",
        prompt: "front prompt",
      },
      {
        x: 0.4,
        y: 0.2,
        instanceId: "evolmd-mo-1",
        proposalId: "evolmd-mo",
        sourceIndex: 2,
        rank: 12,
        label: "dominated point",
        prompt: "dominated prompt",
      },
    ],
    nonDominated: [
      {
        x: 0.8,
        y: 0.5,
        instanceId: "evolmd-mo-1",
        proposalId: "evolmd-mo",
        sourceIndex: 1,
        rank: 2,
        label: "front point",
        prompt: "front prompt",
      },
    ],
    selected: [
      {
        x: 0.8,
        y: 0.5,
        instanceId: "evolmd-mo-1",
        proposalId: "evolmd-mo",
        sourceIndex: null,
        rank: 1,
        label: "front point",
        prompt: "front prompt",
      },
      {
        x: 0.4,
        y: 0.2,
        instanceId: "evolmd-mo-1",
        proposalId: "evolmd-mo",
        sourceIndex: null,
        rank: 2,
        label: "dominated point",
        prompt: "dominated prompt",
      },
    ],
  };

  const points = comparatorVisibleFrontChartPoints(charts, "pareto-points");

  assert.deepEqual(points.individuals.map((point) => point.label), ["front point"]);
  assert.deepEqual(points.selected.map((point) => point.label), ["front point"]);
});

test("binary front chart points preserve native pareto individuals", () => {
  const charts = {
    pareto: [
      {
        x: 0.8,
        y: 0.5,
        instanceId: "binary-mopso-cd-2",
        proposalId: "binary-mopso-cd",
        sourceIndex: 1,
        rank: 1,
        label: "binary native front",
        prompt: "binary front prompt",
      },
      {
        x: 0.7,
        y: 0.4,
        instanceId: "binary-mopso-cd-2",
        proposalId: "binary-mopso-cd",
        sourceIndex: 2,
        rank: 2,
        label: "binary native point",
        prompt: "binary native prompt",
      },
    ],
    nonDominated: [
      {
        x: 0.8,
        y: 0.5,
        instanceId: "binary-mopso-cd-2",
        proposalId: "binary-mopso-cd",
        sourceIndex: 1,
        rank: 1,
        label: "binary native front",
        prompt: "binary front prompt",
      },
    ],
    selected: [
      {
        x: 0.8,
        y: 0.5,
        instanceId: "binary-mopso-cd-2",
        proposalId: "binary-mopso-cd",
        sourceIndex: null,
        rank: 1,
        label: "binary native front",
        prompt: "binary front prompt",
      },
      {
        x: 0.7,
        y: 0.4,
        instanceId: "binary-mopso-cd-2",
        proposalId: "binary-mopso-cd",
        sourceIndex: null,
        rank: 2,
        label: "binary native point",
        prompt: "binary native prompt",
      },
    ],
  };

  const points = comparatorVisibleFrontChartPoints(charts, "pareto-points");

  assert.deepEqual(points.individuals.map((point) => point.label), ["binary native front", "binary native point"]);
  assert.deepEqual(points.selected.map((point) => point.label), ["binary native front", "binary native point"]);
});

test("point chart view proposal selects a single repetition without merging K", () => {
  const proposal = {
    proposalId: "binary-mopso-cd",
    instanceId: "binary-a",
    displayName: "Binary A",
    charts: {
      pareto: [{ label: "aggregated top k" }],
      nonDominated: [{ label: "aggregated top k" }],
      selected: [],
    },
    pointChartRepetitions: [
      {
        repetitionIndex: 1,
        repetitionSeed: 42,
        metrics: { hypervolumeLabel: "0.110000" },
        charts: {
          pareto: [{ label: "rep 1 a" }, { label: "rep 1 b" }],
          nonDominated: [{ label: "rep 1 a" }],
          selected: [],
        },
      },
      {
        repetitionIndex: 2,
        repetitionSeed: 43,
        metrics: { hypervolumeLabel: "0.220000" },
        charts: {
          pareto: [{ label: "rep 2 a" }, { label: "rep 2 b" }, { label: "rep 2 c" }],
          nonDominated: [{ label: "rep 2 a" }],
          selected: [],
        },
      },
    ],
  };

  const selected = comparatorPointChartViewProposal(proposal, 2);

  assert.equal(selected.pointChartRepetition.repetitionIndex, 2);
  assert.equal(selected.pointChartUnavailable, false);
  assert.deepEqual(selected.charts.pareto.map((point) => point.label), ["rep 2 a", "rep 2 b", "rep 2 c"]);
  assert.equal(selected.metrics.hypervolumeLabel, "0.220000");
});

test("visible front point count follows selected repetition chart semantics", () => {
  const proposal = {
    proposalId: "binary-mopso-cd",
    instanceId: "binary-a",
    pointChartRepetitions: [
      {
        repetitionIndex: 1,
        charts: {
          pareto: [{ label: "rep 1 native", proposalId: "binary-mopso-cd" }],
          nonDominated: [],
          selected: [],
        },
        embeddingFrontRows: [{ text: "rep 1 legacy" }],
      },
      {
        repetitionIndex: 2,
        charts: {
          pareto: [
            { label: "rep 2 native a", proposalId: "binary-mopso-cd" },
            { label: "rep 2 native b", proposalId: "binary-mopso-cd" },
            { label: "rep 2 native c", proposalId: "binary-mopso-cd" },
          ],
          nonDominated: [{ label: "rep 2 comparable", proposalId: "binary-mopso-cd" }],
          selected: [],
        },
        embeddingFrontRows: [{ text: "rep 2 legacy" }],
      },
    ],
  };

  assert.equal(comparatorVisibleFrontPointCount(proposal, 2), 3);
});

test("point chart view proposal marks missing repetitions unavailable", () => {
  const proposal = {
    proposalId: "evolmd-mo",
    instanceId: "evolmd-mo-a",
    displayName: "EVOLMD-MO A",
    pointChartRepetitions: [
      {
        repetitionIndex: 1,
        charts: { pareto: [{ label: "rep 1 only" }], nonDominated: [{ label: "rep 1 only" }], selected: [] },
      },
    ],
  };

  const selected = comparatorPointChartViewProposal(proposal, 2);

  assert.equal(selected.pointChartUnavailable, true);
  assert.deepEqual(selected.charts.pareto, []);
  assert.equal(selected.pointChartRepetition.repetitionIndex, 2);
});

test("point chart repetition options use the union of completed repetition indexes", () => {
  const options = comparatorPointChartRepetitionOptions([
    {
      pointChartRepetitions: [
        { repetitionIndex: 1 },
        { repetitionIndex: 3 },
      ],
    },
    {
      pointChartRepetitions: [
        { repetitionIndex: 2 },
      ],
    },
  ]);

  assert.deepEqual(options.map((item) => item.repetitionIndex), [1, 2, 3]);
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

test("front diagnostics recalculate from active comparable points only", () => {
  const points = [
    { value: [0.2, 0.2], semanticEmbedding: [0, 0], entityTerms: ["alpha"], entityTokenCount: 4 },
    { value: [0.8, 0.8], semanticEmbedding: [10, 0], entityTerms: ["beta"], entityTokenCount: 4 },
    { value: [0.9, 0.1], semanticEmbedding: [0, 10], entityTerms: ["beta"], entityTokenCount: 4 },
    { value: [0.1, 0.9], semanticEmbedding: [10, 10], entityTerms: ["gamma"], entityTokenCount: 4 },
    { value: [0.5, 0.5], semanticEmbedding: [5, 5], entityTerms: ["gamma"], entityTokenCount: 4 },
    { value: [0.6, 0.4], semanticEmbedding: [5, 6], entityTerms: ["gamma"], entityTokenCount: 4 },
  ];

  const full = comparatorFrontDiagnostics(points);
  const withoutFirst = comparatorFrontDiagnostics(points.slice(1));

  assert.equal(Number(full.hypervolume.toFixed(6)), 0.66);
  assert.equal(Number(withoutFirst.hypervolume.toFixed(6)), 0.66);
  assert.notEqual(full.unaryEntropy, withoutFirst.unaryEntropy);
  assert.notEqual(full.globalInertia, withoutFirst.globalInertia);
  assert.notEqual(full.globalEntropy, withoutFirst.globalEntropy);
});

test("front unary entropy matches the comparator grid definition", () => {
  assert.equal(comparatorUnaryEntropy([{ value: [0.2, 0.8] }]), 0);
  assert.equal(
    comparatorUnaryEntropy([
      { value: [0.1, 0.1] },
      { value: [0.9, 0.9] },
    ]),
    1,
  );
});

test("front k-means inertia uses semantic embeddings and normalizes by point count", () => {
  const points = [
    { semanticEmbedding: [0, 0] },
    { semanticEmbedding: [0, 2] },
    { semanticEmbedding: [10, 10] },
    { semanticEmbedding: [10, 12] },
    { semanticEmbedding: [20, 20] },
    { semanticEmbedding: [20, 22] },
  ];

  assert.equal(comparatorKMeansInertia(points, 3), 1);
  assert.equal(comparatorKMeansInertia([{ value: [0, 0] }]), null);
});

test("front entity entropy aggregates per-point semantic terms", () => {
  const score = comparatorEntityEntropy([
    { entityTerms: ["alpha"], entityTokenCount: 3 },
    { entityTerms: ["beta"], entityTokenCount: 3 },
    { entityTerms: ["beta"], entityTokenCount: 3 },
  ]);

  assert.equal(Number(score.toFixed(6)), Number((0.9182958340544896 / Math.log2(9)).toFixed(6)));
  assert.equal(comparatorEntityEntropy([{ entityTerms: [], entityTokenCount: 3 }]), 0);
  assert.equal(comparatorEntityEntropy([{ value: [0, 0] }]), null);
});

test("interactive point keys are stable for cloned chart points", () => {
  const point = {
    value: [0.72, 0.44],
    instanceId: "binary-a",
    proposalId: "binary-mopso-cd",
    sourceIndex: 3,
    repetitionIndex: 1,
    rank: 2,
    labelText: "candidate text",
  };

  assert.equal(
    comparatorPointInteractionKey({ ...point }, 99, "Individuos"),
    comparatorPointInteractionKey({ ...point }, 0, "Individuos"),
  );
  assert.notEqual(
    comparatorPointInteractionKey(point, 0, "Individuos"),
    comparatorPointInteractionKey(point, 0, "Seleccionadas"),
  );
});

test("interactive point partition excludes points without mutating source arrays", () => {
  const points = [
    { value: [0.4, 0.8], sourceIndex: 1, labelText: "active" },
    { value: [0.7, 0.5], sourceIndex: 2, labelText: "excluded" },
    { value: [0.9, 0.3], sourceIndex: 3, labelText: "active-b" },
  ];
  const excludedKey = comparatorPointInteractionKey(points[1], 1, "Individuos");
  const partition = comparatorPartitionPointsByExclusion(points, new Set([excludedKey]), "Individuos");

  assert.deepEqual(partition.active.map((entry) => entry.point.labelText), ["active", "active-b"]);
  assert.deepEqual(partition.inactive.map((entry) => entry.point.labelText), ["excluded"]);
  assert.equal(comparatorActivePointCount(points, new Set([excludedKey]), "Individuos"), 2);
  assert.deepEqual(points.map((point) => point.labelText), ["active", "excluded", "active-b"]);
});

test("iteration extent and limiter use finite generations without mutating source series", () => {
  const series = [
    { name: "A", data: [[0, 0.05], [1, 0.1], [2, 0.3], [3, 0.4]] },
    { name: "B", data: [[1, 0.2], [4, 0.6], [Number.NaN, 0.8]] },
  ];
  const limited = comparatorLimitSeriesToIteration(series, 2);

  assert.deepEqual(comparatorSeriesIterationExtent(series), { min: 0, max: 4 });
  assert.deepEqual(limited.map((item) => item.data), [[[0, 0.05], [1, 0.1], [2, 0.3]], [[1, 0.2]]]);
  assert.deepEqual(series[0].data, [[0, 0.05], [1, 0.1], [2, 0.3], [3, 0.4]]);
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

test("non-binary proposals use the first twenty high-contrast colors without repeats", () => {
  const colors = Array.from({ length: 20 }, (_unused, index) =>
    comparatorProposalColor({ proposalId: `baseline-${index}` }, index),
  );
  const reservedColors = new Set(["#000000", "#FFFFFF", "#2A8C00"]);

  assert.equal(new Set(colors).size, 20);
  colors.forEach((color) => {
    assert.match(color, /^#[0-9A-F]{6}$/);
    assert.equal(reservedColors.has(color), false);
  });
});

test("non-binary proposal palette keeps adjacent colors visually separated", () => {
  const colors = Array.from({ length: 20 }, (_unused, index) =>
    comparatorProposalColor({ proposalId: `baseline-${index}` }, index),
  );
  const minimumDistance = colors.reduce((minimum, color, index) => {
    const distances = colors.slice(index + 1).map((otherColor) => rgbDistance(color, otherColor));
    return Math.min(minimum, ...distances);
  }, Number.POSITIVE_INFINITY);

  assert.ok(minimumDistance >= 45, `minimum distance was ${minimumDistance}`);
});

test("chart style assignments keep a single binary instance green", () => {
  const proposals = [{
    instanceId: "binary-mopso-cd:qwen",
    proposalId: "binary-mopso-cd",
    proposalConfig: {
      extraArgs: "",
      cliValues: {
        "router.task_models.synthetic_text_generation": "qwen3:4b-instruct-2507-q4_K_M",
      },
    },
  }];

  const styles = comparatorProposalChartStyleAssignments(proposals);
  const binaryStyle = styles.get("binary-mopso-cd:qwen");

  assert.equal(binaryStyle.color, "#2A8C00");
  assert.equal(binaryStyle.emphasized, true);
});

test("chart style assignments reserve green only for the unmodified binary instance when several binary instances are compared", () => {
  const proposals = [
    {
      instanceId: "binary-mopso-cd:default",
      proposalId: "binary-mopso-cd",
      proposalConfig: { extraArgs: "", cliValues: {} },
    },
    {
      instanceId: "binary-mopso-cd:qwen",
      proposalId: "binary-mopso-cd",
      proposalConfig: {
        extraArgs: "",
        cliValues: {
          "router.task_models.synthetic_text_generation": "qwen3:4b-instruct-2507-q4_K_M",
        },
      },
    },
    {
      instanceId: "evolmd-mo",
      proposalId: "evolmd-mo",
    },
  ];

  const styles = comparatorProposalChartStyleAssignments(proposals);

  assert.equal(styles.get("binary-mopso-cd:default").color, "#2A8C00");
  assert.equal(styles.get("binary-mopso-cd:default").emphasized, true);
  assert.notEqual(styles.get("binary-mopso-cd:qwen").color, "#2A8C00");
  assert.equal(styles.get("binary-mopso-cd:qwen").emphasized, false);
  assert.notEqual(styles.get("evolmd-mo").color, "#2A8C00");
});

test("chart style assignments do not reserve green when several binary instances are all modified", () => {
  const proposals = [
    {
      instanceId: "binary-mopso-cd:qwen",
      proposalId: "binary-mopso-cd",
      proposalConfig: {
        extraArgs: "",
        cliValues: {
          "router.task_models.synthetic_text_generation": "qwen3:4b-instruct-2507-q4_K_M",
        },
      },
    },
    {
      instanceId: "binary-mopso-cd:mopso",
      proposalId: "binary-mopso-cd",
      proposalConfig: {
        extraArgs: "",
        cliValues: {
          "mopso.leader_tournament_size": "8",
        },
      },
    },
  ];

  const styles = comparatorProposalChartStyleAssignments(proposals);

  assert.notEqual(styles.get("binary-mopso-cd:qwen").color, "#2A8C00");
  assert.notEqual(styles.get("binary-mopso-cd:mopso").color, "#2A8C00");
  assert.equal(styles.get("binary-mopso-cd:qwen").emphasized, false);
  assert.equal(styles.get("binary-mopso-cd:mopso").emphasized, false);
});

test("chart style assignments keep non-binary palette stable when a green binary instance is present", () => {
  const defaultBinary = {
    instanceId: "binary-mopso-cd:default",
    proposalId: "binary-mopso-cd",
    proposalConfig: { extraArgs: "", cliValues: {} },
  };
  const baseline = { instanceId: "evolmd-mo", proposalId: "evolmd-mo" };

  const withoutBinary = comparatorProposalChartStyleAssignments([baseline]);
  const withBinary = comparatorProposalChartStyleAssignments([defaultBinary, baseline]);

  assert.equal(withBinary.get("binary-mopso-cd:default").color, "#2A8C00");
  assert.equal(withBinary.get("evolmd-mo").color, withoutBinary.get("evolmd-mo").color);
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

test("BMOPSO internal analysis availability is driven by payload", () => {
  assert.equal(comparatorHasBmopsoInternalAnalysis([
    { proposalId: "binary-mopso-cd", instanceId: "binary-a", status: "completed" },
  ]), false);

  assert.equal(comparatorHasBmopsoInternalAnalysis([
    {
      proposalId: "binary-mopso-cd",
      instanceId: "binary-a",
      status: "completed",
      internalBmopsoAnalysis: { available: true, series: [{ generation: 1, hypervolume: 0.2 }] },
    },
  ]), true);
});

test("BMOPSO internal analyses preserve multiple instances", () => {
  const analyses = comparatorBmopsoInternalAnalyses([
    {
      proposalId: "binary-mopso-cd",
      instanceId: "binary-a",
      displayName: "Binary A",
      internalBmopsoAnalysis: { available: true, metrics: { hypervolumeLabel: "0.310000" } },
    },
    {
      proposalId: "evolmd-mo",
      instanceId: "evolmd-mo",
      displayName: "EVOLMD-MO",
    },
    {
      proposalId: "binary-mopso-cd",
      instanceId: "binary-b",
      displayName: "Binary B",
      internalBmopsoAnalysis: { available: true, metrics: { hypervolumeLabel: "0.470000" } },
    },
  ]);

  assert.deepEqual(analyses.map((item) => item.instanceId), ["binary-a", "binary-b"]);
  assert.deepEqual(analyses.map((item) => item.displayName), ["Binary A", "Binary B"]);
  assert.deepEqual(analyses.map((item) => item.analysis.metrics.hypervolumeLabel), ["0.310000", "0.470000"]);
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

test("recontinue availability is blocked only for missing or completed runs", () => {
  assert.equal(comparatorCanRecontinueRun(null), false);
  assert.equal(comparatorCanRecontinueRun({ runId: "run-1", status: "completed" }), false);
  assert.equal(comparatorCanRecontinueRun({ runId: "run-1", status: "running" }), true);
  assert.equal(comparatorCanRecontinueRun({ runId: "run-1", status: "queued" }), true);
  assert.equal(comparatorCanRecontinueRun({ runId: "run-1", status: "failed" }), true);
  assert.equal(comparatorCanRecontinueRun({ runId: "run-1", status: "cancelled" }), true);
});
