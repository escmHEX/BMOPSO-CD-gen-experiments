export function comparatorPointCoordinates(point) {
  const value = point?.value || [];
  const x = Number(point?.x ?? value[0]);
  const y = Number(point?.y ?? value[1]);
  if (!Number.isFinite(x) || !Number.isFinite(y)) return null;
  return { x, y };
}

function finiteAxisValue(point, axis) {
  const coordinates = comparatorPointCoordinates(point);
  if (!coordinates) return null;
  const value = axis === "y" ? coordinates.y : coordinates.x;
  return Number.isFinite(value) ? value : null;
}

function clampNumber(value, minimum, maximum) {
  return Math.max(minimum, Math.min(maximum, value));
}

function cleanAxisNumber(value) {
  return Number(Number(value).toPrecision(12));
}

function stableIdentityValue(value) {
  if (value === null || value === undefined) return "";
  const normalized = String(value).trim();
  return normalized.length ? normalized : "";
}

export function comparatorPointInteractionKey(point, fallbackIndex = 0, namespace = "") {
  const coordinates = comparatorPointCoordinates(point);
  const coordinatePart = coordinates
    ? `${cleanAxisNumber(coordinates.x)},${cleanAxisNumber(coordinates.y)}`
    : "no-coordinates";
  const identityParts = [
    namespace,
    point?.instanceId,
    point?.proposalId,
    point?.sourceIndex,
    point?.repetitionIndex,
    point?.rank,
    point?.labelText ?? point?.label,
    point?.prompt,
  ].map(stableIdentityValue).filter(Boolean);

  if (identityParts.length > (namespace ? 1 : 0)) {
    return [...identityParts, coordinatePart].join("|");
  }
  return [stableIdentityValue(namespace), coordinatePart, Math.max(0, Number(fallbackIndex) || 0)].join("|");
}

export function comparatorFrontMembershipKey(point, namespace = "") {
  const coordinates = comparatorPointCoordinates(point);
  const coordinatePart = coordinates
    ? `${cleanAxisNumber(coordinates.x)},${cleanAxisNumber(coordinates.y)}`
    : "no-coordinates";
  return [
    namespace,
    point?.instanceId,
    point?.proposalId,
    point?.labelText ?? point?.label,
    point?.prompt,
    coordinatePart,
  ].map(stableIdentityValue).filter(Boolean).join("|");
}

function comparatorChartProposal(charts = {}) {
  for (const points of [charts.pareto, charts.nonDominated, charts.selected]) {
    if (!Array.isArray(points)) continue;
    const point = points.find(Boolean);
    if (point) {
      return {
        proposalId: point.proposalId,
        instanceId: point.instanceId,
      };
    }
  }
  return {};
}

export function comparatorVisibleFrontChartPoints(charts = {}, namespace = "") {
  const individuals = comparatorIsBinaryProposal(comparatorChartProposal(charts))
    ? charts.pareto || []
    : charts.nonDominated || [];
  const frontKeys = new Set(individuals.map((point) => comparatorFrontMembershipKey(point, namespace)));
  const selected = (charts.selected || []).filter((point) =>
    frontKeys.has(comparatorFrontMembershipKey(point, namespace)),
  );
  return { individuals, selected };
}

export function comparatorSelectedFrontPointsForIndividuals(individuals = [], selected = [], namespace = "") {
  const individualsByMembership = new Map();
  (individuals || []).forEach((point) => {
    const key = comparatorFrontMembershipKey(point, namespace);
    if (key && !individualsByMembership.has(key)) {
      individualsByMembership.set(key, point);
    }
  });

  return (selected || []).map((point) => {
    const match = individualsByMembership.get(comparatorFrontMembershipKey(point, namespace));
    if (!match) return point;
    return {
      ...point,
      instanceId: point.instanceId ?? match.instanceId,
      proposalId: point.proposalId ?? match.proposalId,
      sourceIndex: point.sourceIndex ?? match.sourceIndex,
      repetitionIndex: point.repetitionIndex ?? match.repetitionIndex,
      rank: point.rank ?? match.rank,
      label: point.label ?? match.label,
      prompt: point.prompt ?? match.prompt,
      displayName: point.displayName ?? match.displayName,
      baseDisplayName: point.baseDisplayName ?? match.baseDisplayName,
    };
  });
}

function finiteRepetitionIndex(value) {
  const numeric = Number(value);
  return Number.isInteger(numeric) && numeric > 0 ? numeric : null;
}

function normalChartPayload(charts) {
  return charts && typeof charts === "object"
    ? charts
    : { pareto: [], nonDominated: [], selected: [], series: [] };
}

function normalMetricsPayload(metrics) {
  return metrics && typeof metrics === "object" ? metrics : {};
}

function normalPointChartRepetition(item = {}, fallbackIndex = null) {
  const repetitionIndex = finiteRepetitionIndex(item.repetitionIndex) ?? finiteRepetitionIndex(fallbackIndex);
  const payload = {
    repetitionIndex,
    repetitionSeed: item.repetitionSeed ?? null,
    charts: normalChartPayload(item.charts),
    metrics: normalMetricsPayload(item.metrics),
    embeddingFrontRows: Array.isArray(item.embeddingFrontRows) ? item.embeddingFrontRows : [],
  };
  if (item.internalBmopsoAnalysis && typeof item.internalBmopsoAnalysis === "object") {
    payload.internalBmopsoAnalysis = item.internalBmopsoAnalysis;
  }
  return payload;
}

export function comparatorPointChartRepetitions(proposal = {}) {
  if (Array.isArray(proposal.pointChartRepetitions) && proposal.pointChartRepetitions.length) {
    return proposal.pointChartRepetitions
      .filter((item) => item && typeof item === "object")
      .map((item) => normalPointChartRepetition(item));
  }

  if (Array.isArray(proposal.repetitions) && proposal.repetitions.length) {
    const repetitions = proposal.repetitions
      .filter((item) => item && typeof item === "object" && item.status === "completed")
      .map((item) => normalPointChartRepetition(item));
    if (repetitions.length) return repetitions;
  }

  return [normalPointChartRepetition({
    repetitionIndex: null,
    repetitionSeed: null,
    charts: proposal.charts,
    metrics: proposal.metrics,
    embeddingFrontRows: proposal.embeddingFrontRows,
    internalBmopsoAnalysis: proposal.internalBmopsoAnalysis,
  })];
}

export function comparatorPointChartRepetitionOptions(proposals = []) {
  const indexes = new Set();
  (proposals || []).forEach((proposal) => {
    comparatorPointChartRepetitions(proposal).forEach((item) => {
      const repetitionIndex = finiteRepetitionIndex(item.repetitionIndex);
      if (repetitionIndex !== null) indexes.add(repetitionIndex);
    });
  });
  return [...indexes]
    .sort((left, right) => left - right)
    .map((repetitionIndex) => ({ repetitionIndex, label: `Rep ${repetitionIndex}` }));
}

export function comparatorPointChartViewProposal(proposal = {}, selectedRepetitionIndex = null) {
  const requestedIndex = finiteRepetitionIndex(selectedRepetitionIndex);
  const repetitions = comparatorPointChartRepetitions(proposal);
  const selected = requestedIndex === null
    ? repetitions[0]
    : repetitions.find((item) => finiteRepetitionIndex(item.repetitionIndex) === requestedIndex);

  if (!selected) {
    return {
      ...proposal,
      charts: { pareto: [], nonDominated: [], selected: [], series: [] },
      metrics: normalMetricsPayload(proposal.metrics),
      embeddingFrontRows: [],
      internalBmopsoAnalysis: null,
      pointChartRepetition: {
        repetitionIndex: requestedIndex,
        repetitionSeed: null,
      },
      pointChartUnavailable: true,
    };
  }

  return {
    ...proposal,
    charts: selected.charts,
    metrics: Object.keys(selected.metrics).length ? selected.metrics : normalMetricsPayload(proposal.metrics),
    embeddingFrontRows: selected.embeddingFrontRows,
    internalBmopsoAnalysis: selected.internalBmopsoAnalysis || null,
    pointChartRepetition: selected,
    pointChartUnavailable: false,
  };
}

export function comparatorVisibleFrontPointCount(proposal = {}, selectedRepetitionIndex = null, namespace = "pareto-points") {
  const viewProposal = comparatorPointChartViewProposal(proposal, selectedRepetitionIndex);
  return comparatorVisibleFrontChartPoints(viewProposal.charts, namespace).individuals.length;
}

export function comparatorCanRecontinueRun(run) {
  return Boolean(run?.runId && run.status !== "completed");
}

export function comparatorPartitionPointsByExclusion(points = [], excludedKeys = new Set(), namespace = "") {
  const excluded = excludedKeys instanceof Set ? excludedKeys : new Set(excludedKeys || []);
  return (points || []).reduce((partition, point, index) => {
    const key = comparatorPointInteractionKey(point, index, namespace);
    const entry = { point, key, index };
    if (excluded.has(key)) {
      partition.inactive.push(entry);
    } else {
      partition.active.push(entry);
    }
    return partition;
  }, { active: [], inactive: [] });
}

export function comparatorActivePointCount(points = [], excludedKeys = new Set(), namespace = "") {
  return comparatorPartitionPointsByExclusion(points, excludedKeys, namespace).active.length;
}

function finiteIterationValue(point) {
  const value = Array.isArray(point) ? point[0] : point?.generation ?? point?.value?.[0];
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : null;
}

export function comparatorSeriesIterationExtent(series = []) {
  const values = (series || [])
    .flatMap((item) => item?.data || [])
    .map(finiteIterationValue)
    .filter((value) => value !== null);
  if (!values.length) return null;
  return {
    min: Math.min(...values),
    max: Math.max(...values),
  };
}

export function comparatorLimitSeriesToIteration(series = [], iterationLimit = null) {
  const limit = Number(iterationLimit);
  const hasLimit = Number.isFinite(limit);
  return (series || []).map((item) => ({
    ...item,
    data: (item?.data || []).filter((point) => {
      const iteration = finiteIterationValue(point);
      return iteration !== null && (!hasLimit || iteration <= limit);
    }),
  }));
}

const COMPARATOR_BINARY_COLOR = "#2A8C00";
const COMPARATOR_NON_BINARY_PALETTE = Object.freeze([
  "#1F77B4",
  "#D62728",
  "#9467BD",
  "#FF7F0E",
  "#17BECF",
  "#F200F2",
  "#F2DA61",
  "#601773",
  "#4B18F2",
  "#8C7E38",
  "#F26183",
  "#731F00",
  "#BF0093",
  "#E961F2",
  "#003673",
  "#61F2F2",
  "#F2DA00",
  "#6196F2",
  "#0041F2",
  "#A218F2",
]);

export function comparatorExpandedAxisWindow(defaultMin, defaultMax, options = {}) {
  const rawMin = Number(defaultMin);
  const rawMax = Number(defaultMax);
  if (!Number.isFinite(rawMin) || !Number.isFinite(rawMax)) return null;

  const zoomFactor = Math.max(1, Number(options.zoomFactor) || 100);
  const minSpan = Math.max(0, Number(options.minSpan) || 1);
  let visibleMin = Math.min(rawMin, rawMax);
  let visibleMax = Math.max(rawMin, rawMax);

  if (visibleMin === visibleMax) {
    const center = visibleMin;
    const span = minSpan || Math.max(1, Math.abs(center) * 0.1);
    visibleMin = center - (span / 2);
    visibleMax = center + (span / 2);
  }

  const visibleSpan = visibleMax - visibleMin;
  const center = (visibleMin + visibleMax) / 2;
  const zoomSpan = visibleSpan * zoomFactor;
  const zoomMin = center - (zoomSpan / 2);
  const zoomMax = center + (zoomSpan / 2);

  return {
    defaultMin: cleanAxisNumber(visibleMin),
    defaultMax: cleanAxisNumber(visibleMax),
    zoomMin: cleanAxisNumber(zoomMin),
    zoomMax: cleanAxisNumber(zoomMax),
  };
}

export function comparatorChartAxisWindow(points, axis = "x", options = {}) {
  const values = (points || [])
    .map((point) => finiteAxisValue(point, axis))
    .filter((value) => value !== null);
  if (!values.length) return null;

  const dataMin = Math.min(...values);
  const dataMax = Math.max(...values);
  if (dataMin === dataMax) {
    return comparatorExpandedAxisWindow(dataMin, dataMax, options);
  }

  const paddingRatio = Math.max(0, Number(options.paddingRatio) || 0);
  const padding = (dataMax - dataMin) * paddingRatio;
  return comparatorExpandedAxisWindow(dataMin - padding, dataMax + padding, options);
}

export function comparatorMetricReferenceLinePatch(seriesId, referenceLines, hidden) {
  if (!seriesId || !referenceLines) return [];
  return [{
    id: String(seriesId),
    markLine: hidden ? { ...referenceLines, data: [] } : referenceLines,
  }];
}

export function comparatorProposalEntityId(proposal) {
  return proposal?.instanceId || proposal?.proposalId || "";
}

export function comparatorIsBinaryProposal(proposal) {
  const proposalId = String(proposal?.proposalId || proposal?.baseProposalId || proposal || "");
  const instanceId = String(proposal?.instanceId || "");
  return proposalId === "binary-mopso-cd" || instanceId === "binary-mopso-cd" || instanceId.startsWith("binary-mopso-cd:");
}

function comparatorPaletteColor(index = 0) {
  const colorIndex = Math.abs(Math.trunc(Number(index) || 0)) % COMPARATOR_NON_BINARY_PALETTE.length;
  return COMPARATOR_NON_BINARY_PALETTE[colorIndex];
}

export function comparatorProposalColor(proposal, index = 0) {
  if (comparatorIsBinaryProposal(proposal)) return COMPARATOR_BINARY_COLOR;
  return comparatorPaletteColor(index);
}

function comparatorProposalConfigModified(proposal) {
  const config = proposal?.proposalConfig || {};
  if (String(config.extraArgs || "").trim()) return true;
  const cliValues = config.cliValues && typeof config.cliValues === "object" ? config.cliValues : {};
  return Object.keys(cliValues).length > 0;
}

function comparatorReservedBinaryEntityId(proposals) {
  const binaryProposals = proposals.filter(comparatorIsBinaryProposal);
  if (binaryProposals.length === 0) return "";
  if (binaryProposals.length === 1) return comparatorProposalEntityId(binaryProposals[0]);
  const unmodified = binaryProposals.find((proposal) => !comparatorProposalConfigModified(proposal));
  return unmodified ? comparatorProposalEntityId(unmodified) : "";
}

export function comparatorProposalChartStyleAssignments(proposals = []) {
  const list = Array.isArray(proposals) ? proposals : [];
  const reservedBinaryEntityId = comparatorReservedBinaryEntityId(list);
  const styles = new Map();
  let paletteIndex = 0;

  list.forEach((proposal) => {
    const entityId = comparatorProposalEntityId(proposal);
    if (!entityId) return;
    const isBinary = comparatorIsBinaryProposal(proposal);
    const emphasized = Boolean(isBinary && entityId === reservedBinaryEntityId);
    const color = emphasized ? COMPARATOR_BINARY_COLOR : comparatorPaletteColor(paletteIndex++);
    styles.set(entityId, {
      color,
      emphasized,
      isBinary,
      lineWidth: emphasized ? 3 : 2,
      symbolSize: emphasized ? 11 : 9,
      pointOpacity: emphasized ? 0.86 : 0.68,
    });
  });

  return styles;
}

export function comparatorBenchmarkProposals(proposals = []) {
  return [...proposals].sort((left, right) => {
    const leftIsBinary = left?.proposalId === "binary-mopso-cd" ? 0 : 1;
    const rightIsBinary = right?.proposalId === "binary-mopso-cd" ? 0 : 1;
    if (leftIsBinary !== rightIsBinary) return leftIsBinary - rightIsBinary;
    return String(left?.displayName || left?.proposalId || "").localeCompare(
      String(right?.displayName || right?.proposalId || ""),
      "es",
      { sensitivity: "base" },
    );
  });
}

export function comparatorApplyColumnOrder(proposals = [], orderedIds = []) {
  const base = Array.isArray(proposals) ? proposals : [];
  const byId = new Map(
    base
      .map((proposal) => [stableIdentityValue(proposal?.instanceId || proposal?.proposalId), proposal])
      .filter(([id]) => id),
  );
  const ordered = [];
  const used = new Set();
  for (const rawId of Array.isArray(orderedIds) ? orderedIds : []) {
    const id = stableIdentityValue(rawId);
    const proposal = byId.get(id);
    if (!proposal || used.has(id)) continue;
    ordered.push(proposal);
    used.add(id);
  }
  for (const proposal of base) {
    const id = stableIdentityValue(proposal?.instanceId || proposal?.proposalId);
    if (!id || used.has(id)) continue;
    ordered.push(proposal);
  }
  return ordered;
}

export function comparatorMoveColumnId(orderedIds = [], draggedId = "", targetId = "", placement = "before") {
  const dragged = stableIdentityValue(draggedId);
  const target = stableIdentityValue(targetId);
  if (!dragged || !target || dragged === target) return [...orderedIds];
  const next = (Array.isArray(orderedIds) ? orderedIds : [])
    .map(stableIdentityValue)
    .filter(Boolean);
  const fromIndex = next.indexOf(dragged);
  const targetIndex = next.indexOf(target);
  if (fromIndex === -1 || targetIndex === -1) return next;
  next.splice(fromIndex, 1);
  const adjustedTargetIndex = next.indexOf(target);
  const insertIndex = placement === "after" ? adjustedTargetIndex + 1 : adjustedTargetIndex;
  next.splice(insertIndex, 0, dragged);
  return next;
}

export function comparatorBmopsoInternalAnalyses(proposals = []) {
  const list = Array.isArray(proposals) ? proposals : [];
  return list.flatMap((proposal) => {
    const analysis = proposal?.internalBmopsoAnalysis;
    if (!analysis?.available) return [];
    return [{
      instanceId: comparatorProposalEntityId(proposal) || analysis.instanceId || "",
      proposalId: proposal?.proposalId || analysis.proposalId || "",
      displayName: proposal?.displayName || analysis.displayName || proposal?.proposalId || "",
      analysis,
    }];
  });
}

export function comparatorHasBmopsoInternalAnalysis(proposals = []) {
  return comparatorBmopsoInternalAnalyses(proposals).length > 0;
}

export function comparatorDominates(candidate, point) {
  const candidateCoordinates = comparatorPointCoordinates(candidate);
  const pointCoordinates = comparatorPointCoordinates(point);
  if (!candidateCoordinates || !pointCoordinates) return false;
  const atLeastEqual = candidateCoordinates.x >= pointCoordinates.x && candidateCoordinates.y >= pointCoordinates.y;
  const strictlyBetter = candidateCoordinates.x > pointCoordinates.x || candidateCoordinates.y > pointCoordinates.y;
  return atLeastEqual && strictlyBetter;
}

export function comparatorGlobalNonDominatedFront(points) {
  return points.filter((point, index) =>
    !points.some((candidate, candidateIndex) => candidateIndex !== index && comparatorDominates(candidate, point)),
  );
}

export function comparatorIsGloballyNonDominated(point, comparisonPool) {
  return comparisonPool.length > 0 && !comparisonPool.some((candidate) => comparatorDominates(candidate, point));
}

export function comparatorCountByProposal(points) {
  return points.reduce((counts, point) => {
    const id = point.instanceId || point.proposalId;
    counts.set(id, (counts.get(id) || 0) + 1);
    return counts;
  }, new Map());
}

export function comparatorHypervolumeArea(points) {
  const frontCoordinates = comparatorGlobalNonDominatedFront(points)
    .map(comparatorPointCoordinates)
    .filter(Boolean)
    .map((point) => ({
      x: Math.max(0, point.x),
      y: Math.max(0, point.y),
    }));
  if (!frontCoordinates.length) return null;

  const collapsed = new Map();
  frontCoordinates.forEach((point) => {
    collapsed.set(point.x, Math.max(collapsed.get(point.x) ?? 0, point.y));
  });

  const ordered = [...collapsed.entries()]
    .map(([x, y]) => ({ x, y }))
    .sort((a, b) => a.x - b.x);
  if (!ordered.length) return null;

  const lineData = [[0, ordered[0].y]];
  let previousX = 0;
  let area = 0;
  let weightedX = 0;
  let weightedY = 0;

  ordered.forEach((point, index) => {
    if (index > 0) {
      lineData.push([previousX, point.y]);
    }
    lineData.push([point.x, point.y]);

    const width = Math.max(0, point.x - previousX);
    const rectangleArea = width * point.y;
    if (rectangleArea > 0) {
      area += rectangleArea;
      weightedX += rectangleArea * ((previousX + point.x) / 2);
      weightedY += rectangleArea * (point.y / 2);
    }
    previousX = point.x;
  });

  lineData.push([previousX, 0]);

  const maxY = Math.max(...ordered.map((point) => point.y));
  const labelPosition = area > 0.02
    ? [weightedX / area, weightedY / area]
    : [
        Math.max(0.06, previousX * 0.5),
        Math.max(0.06, maxY * 0.55),
      ];

  return { lineData, labelPosition, area };
}

export function comparatorExtent(points = []) {
  const coordinates = (points || [])
    .map(comparatorPointCoordinates)
    .filter(Boolean)
    .map((point) => [point.x, point.y]);
  if (coordinates.length <= 1) return 0;

  const width = Math.min(...coordinates.map((point) => point.length));
  if (width <= 0) return 0;

  let totalRange = 0;
  for (let index = 0; index < width; index += 1) {
    const values = coordinates.map((point) => clampNumber(Number(point[index]) || 0, 0, 1));
    totalRange += Math.max(...values) - Math.min(...values);
  }
  return cleanAxisNumber(Math.sqrt(totalRange));
}

export function comparatorUnaryEntropy(points = [], mu = 5) {
  const coordinates = (points || [])
    .map(comparatorPointCoordinates)
    .filter(Boolean)
    .map((point) => [point.x, point.y]);
  if (coordinates.length <= 1) return 0;

  const gridSize = Math.trunc(Number(mu) || 0);
  const width = 2;
  if (gridSize <= 1) return 0;

  const cells = new Map();
  coordinates.forEach((point) => {
    const cell = point.map((value) =>
      Math.min(gridSize, Math.floor(gridSize * clampNumber(Number(value) || 0, 0, 1)) + 1),
    ).join("|");
    cells.set(cell, (cells.get(cell) || 0) + 1);
  });

  const denominatorBase = Math.min(coordinates.length, gridSize ** width);
  if (denominatorBase <= 1) return 0;

  let entropy = 0;
  cells.forEach((count) => {
    const probability = count / coordinates.length;
    if (probability > 0) entropy -= probability * Math.log(probability);
  });
  return clampNumber(entropy / Math.log(denominatorBase), 0, 1);
}

function semanticEmbedding(point) {
  const raw = Array.isArray(point?.semanticEmbedding)
    ? point.semanticEmbedding
    : point?.frontDiagnostics?.embedding;
  const values = Array.isArray(raw)
    ? raw.map(Number).filter(Number.isFinite)
    : [];
  return values.length ? values : null;
}

function squaredDistance(left, right, width) {
  let total = 0;
  for (let index = 0; index < width; index += 1) {
    total += (left[index] - right[index]) ** 2;
  }
  return total;
}

function lexicographicEmbeddingCompare(left, right) {
  const width = Math.min(left.length, right.length);
  for (let index = 0; index < width; index += 1) {
    if (left[index] !== right[index]) return left[index] - right[index];
  }
  return left.length - right.length;
}

function initialKMeansCenters(embeddings, clusters, width) {
  const ordered = [...embeddings].sort(lexicographicEmbeddingCompare);
  const centers = [ordered[0].slice(0, width)];
  while (centers.length < clusters) {
    let best = null;
    ordered.forEach((embedding) => {
      const candidate = embedding.slice(0, width);
      const distance = Math.min(...centers.map((center) => squaredDistance(candidate, center, width)));
      if (
        !best
        || distance > best.distance
        || (distance === best.distance && lexicographicEmbeddingCompare(candidate, best.embedding) < 0)
      ) {
        best = { embedding: candidate, distance };
      }
    });
    centers.push(best.embedding);
  }
  return centers;
}

export function comparatorKMeansInertia(points = [], clusterCount = 5) {
  const embeddings = (points || []).map(semanticEmbedding).filter(Boolean);
  if (!embeddings.length) return null;
  if (embeddings.length <= 1) return 0;

  const width = Math.min(...embeddings.map((embedding) => embedding.length));
  if (width <= 0) return null;

  const clusters = Math.min(
    embeddings.length,
    Math.max(1, Math.trunc(Number(clusterCount) || 5)),
  );
  if (clusters >= embeddings.length) return 0;

  let centers = initialKMeansCenters(embeddings, clusters, width);
  let assignments = new Array(embeddings.length).fill(-1);
  for (let iteration = 0; iteration < 100; iteration += 1) {
    let changed = false;
    const nextAssignments = embeddings.map((embedding) => {
      const vector = embedding.slice(0, width);
      let bestCluster = 0;
      let bestDistance = squaredDistance(vector, centers[0], width);
      for (let cluster = 1; cluster < centers.length; cluster += 1) {
        const distance = squaredDistance(vector, centers[cluster], width);
        if (distance < bestDistance) {
          bestCluster = cluster;
          bestDistance = distance;
        }
      }
      return bestCluster;
    });

    nextAssignments.forEach((assignment, index) => {
      if (assignment !== assignments[index]) changed = true;
    });
    assignments = nextAssignments;

    const sums = Array.from({ length: clusters }, () => new Array(width).fill(0));
    const counts = new Array(clusters).fill(0);
    embeddings.forEach((embedding, index) => {
      const cluster = assignments[index];
      counts[cluster] += 1;
      for (let dimension = 0; dimension < width; dimension += 1) {
        sums[cluster][dimension] += embedding[dimension];
      }
    });

    centers = centers.map((center, cluster) => {
      if (!counts[cluster]) return center;
      return sums[cluster].map((sum) => sum / counts[cluster]);
    });
    if (!changed) break;
  }

  const total = embeddings.reduce((sum, embedding, index) =>
    sum + squaredDistance(embedding, centers[assignments[index]], width), 0);
  return cleanAxisNumber(total / embeddings.length);
}

export function comparatorEntityEntropy(points = []) {
  const diagnostics = (points || []).map((point) => {
    const terms = Array.isArray(point?.entityTerms)
      ? point.entityTerms
      : point?.frontDiagnostics?.entityTerms;
    const tokenCount = Number(point?.entityTokenCount ?? point?.frontDiagnostics?.entityTokenCount);
    return { terms, tokenCount, complete: Array.isArray(terms) && Number.isFinite(tokenCount) };
  });
  if (!diagnostics.length || diagnostics.some((point) => !point.complete)) return null;

  let totalTokens = 0;
  const counts = new Map();

  diagnostics.forEach(({ terms, tokenCount }) => {
    if (Number.isFinite(tokenCount) && tokenCount > 0) totalTokens += tokenCount;
    terms.forEach((term) => {
      const key = String(term || "").trim().toLowerCase();
      if (key) counts.set(key, (counts.get(key) || 0) + 1);
    });
  });

  if (!counts.size || totalTokens <= 1) return 0;

  const totalTerms = [...counts.values()].reduce((sum, count) => sum + count, 0);
  let entropy = 0;
  counts.forEach((count) => {
    const probability = count / totalTerms;
    if (probability > 0) entropy -= probability * Math.log2(probability);
  });
  return cleanAxisNumber(entropy / Math.log2(totalTokens));
}

function pointHasSemanticEmbedding(point) {
  return Boolean(semanticEmbedding(point));
}

function pointHasEntityDiagnostics(point) {
  const terms = Array.isArray(point?.entityTerms)
    ? point.entityTerms
    : point?.frontDiagnostics?.entityTerms;
  const tokenCount = Number(point?.entityTokenCount ?? point?.frontDiagnostics?.entityTokenCount);
  return Array.isArray(terms) && Number.isFinite(tokenCount);
}

export function comparatorAverageFinite(values = []) {
  const finiteValues = (values || [])
    .filter((value) => value !== null && value !== undefined && value !== "")
    .map(Number)
    .filter(Number.isFinite);
  if (!finiteValues.length) return null;
  return cleanAxisNumber(finiteValues.reduce((sum, value) => sum + value, 0) / finiteValues.length);
}

export function comparatorAverageWithReplacement(values = [], replacementIndex = -1, replacementValue = null) {
  const targetIndex = Math.trunc(Number(replacementIndex));
  const replaced = (values || []).map((value, index) => (
    index === targetIndex ? replacementValue : value
  ));
  return comparatorAverageFinite(replaced);
}

function normalPointsByProposalEntries(pointsByProposal = {}) {
  if (pointsByProposal instanceof Map) {
    return [...pointsByProposal.entries()];
  }
  if (pointsByProposal && typeof pointsByProposal === "object") {
    return Object.entries(pointsByProposal);
  }
  return [];
}

export function comparatorContributionByProposal(pointsByProposal = {}) {
  const proposalPoints = new Map(
    normalPointsByProposalEntries(pointsByProposal).map(([proposalId, points]) => [
      stableIdentityValue(proposalId),
      new Set((Array.isArray(points) ? points : [])
        .map(comparatorPointCoordinates)
        .filter(Boolean)
        .map((point) => `${cleanAxisNumber(point.x)},${cleanAxisNumber(point.y)}`)),
    ]).filter(([proposalId]) => proposalId),
  );
  const allPoints = [...new Set([...proposalPoints.values()].flatMap((points) => [...points]))]
    .map((key) => {
      const [x, y] = key.split(",").map(Number);
      return { key, x, y };
    })
    .filter((point) => Number.isFinite(point.x) && Number.isFinite(point.y));
  const frontKeys = new Set(comparatorGlobalNonDominatedFront(allPoints).map((point) => point.key));
  const contributions = new Map([...proposalPoints.keys()].map((proposalId) => [proposalId, 0]));
  if (!frontKeys.size) return contributions;

  frontKeys.forEach((pointKey) => {
    const producers = [...proposalPoints.entries()]
      .filter(([_proposalId, points]) => points.has(pointKey))
      .map(([proposalId]) => proposalId);
    if (!producers.length) return;
    const credit = 1 / producers.length;
    producers.forEach((proposalId) => {
      contributions.set(proposalId, (contributions.get(proposalId) || 0) + (credit / frontKeys.size));
    });
  });
  contributions.forEach((value, proposalId) => {
    contributions.set(proposalId, cleanAxisNumber(value));
  });
  return contributions;
}

export function comparatorFrontDiagnostics(points = [], options = {}) {
  const activePoints = points || [];
  const hasCompleteEmbeddings = activePoints.length > 0 && activePoints.every(pointHasSemanticEmbedding);
  const hasCompleteEntityDiagnostics = activePoints.length > 0 && activePoints.every(pointHasEntityDiagnostics);
  return {
    nonDominatedRows: activePoints.length,
    hypervolume: comparatorHypervolumeArea(activePoints)?.area ?? null,
    extent: activePoints.length ? comparatorExtent(activePoints) : null,
    unaryEntropy: activePoints.length ? comparatorUnaryEntropy(activePoints, options.unaryEntropyGridSize ?? 5) : null,
    globalInertia: hasCompleteEmbeddings
      ? comparatorKMeansInertia(activePoints, options.kMeansClusters ?? 5)
      : null,
    globalEntropy: hasCompleteEntityDiagnostics ? comparatorEntityEntropy(activePoints) : null,
  };
}

export function comparatorMetricMetadata(metricKey) {
  if (metricKey === "nonDominatedRows") {
    return { description: "Mayor cantidad de soluciones Pareto disponibles es mejor.", higherIsBetter: true };
  }
  if (metricKey === "contribution") {
    return { description: "Mayor Contribution indica mayor aporte al frente combinado P*.", higherIsBetter: true };
  }
  if (metricKey === "extent") {
    return { description: "Mayor Extent indica mayor cobertura del frente comparable.", higherIsBetter: true };
  }
  if (metricKey === "unaryEntropy") {
    return { description: "Mayor Unary Entropy indica mejor distribucion del frente comparable.", higherIsBetter: true };
  }
  if (metricKey === "globalInertia") {
    return { description: "Mayor K-means inertia indica mayor dispersion global de embeddings.", higherIsBetter: true };
  }
  if (metricKey === "globalEntropy") {
    return { description: "Mayor entity entropy indica mayor variedad conceptual o semantica.", higherIsBetter: true };
  }
  return { description: "Mayor HV es mejor: mas area dominada respecto a [0,0].", higherIsBetter: true };
}

export function comparatorMetricExtremes(values, higherIsBetter) {
  const finiteValues = values.filter(Number.isFinite);
  if (finiteValues.length === 0) return null;
  const minimum = Math.min(...finiteValues);
  const maximum = Math.max(...finiteValues);
  return {
    bestValue: higherIsBetter ? maximum : minimum,
    worstValue: higherIsBetter ? minimum : maximum,
  };
}

export function comparatorBestMetricProposalIds(proposals, metric, options = {}) {
  if (metric.kind === "cost" && !options.costsComparable) return new Set();
  const completed = proposals.filter((proposal) => proposal?.status === "completed");
  const entries = completed
    .map((proposal) => {
      const cost = proposal.cost || {};
      const rawValue = metric.value(proposal, cost);
      const value = rawValue === null || rawValue === undefined || rawValue === "" ? NaN : Number(rawValue);
      const reported = metric.isReported ? metric.isReported(proposal, cost) : true;
      return { id: comparatorProposalEntityId(proposal), value, reported };
    })
    .filter((entry) => entry.id && entry.reported && Number.isFinite(entry.value));

  if (!entries.length) return new Set();
  if (metric.requireAllReported && entries.length !== completed.length) return new Set();

  const target = metric.direction === "max"
    ? Math.max(...entries.map((entry) => entry.value))
    : Math.min(...entries.map((entry) => entry.value));
  return new Set(entries.filter((entry) => entry.value === target).map((entry) => entry.id));
}

export function comparatorBestCostProposalIds(proposals, metric, options = {}) {
  return comparatorBestMetricProposalIds(proposals, { ...metric, kind: "cost" }, options);
}

export function comparatorMetricCellClassName({ primaryColumn = false, best = false } = {}) {
  return [
    primaryColumn ? "is-primary-proposal" : "",
    best ? "metric-best comparator-cost-best comparator-metric-best-cell" : "",
  ].filter(Boolean).join(" ");
}

export function comparatorMetricDeltaPercent(primaryValue, comparisonValue, direction = "max") {
  const baseline = Number(primaryValue);
  const current = Number(comparisonValue);
  if (!Number.isFinite(baseline) || !Number.isFinite(current) || baseline === 0) return null;

  const denominator = Math.abs(baseline);
  const delta = direction === "min"
    ? ((baseline - current) / denominator) * 100
    : ((current - baseline) / denominator) * 100;
  const cleanDelta = cleanAxisNumber(delta);
  return Object.is(cleanDelta, -0) ? 0 : cleanDelta;
}

export function comparatorMetricDeltaLabel(deltaPercent) {
  const value = Number(deltaPercent);
  if (!Number.isFinite(value)) return "";
  const rounded = Math.round(Math.abs(value));
  if (rounded === 0) return "0%";
  return `${value > 0 ? "+" : "-"}${rounded}%`;
}

export function comparatorMetricMeanStdDevLabel(meanLabel, stdDevLabel) {
  const mean = String(meanLabel ?? "").trim();
  const stdDev = String(stdDevLabel ?? "").trim();
  if (!mean || !stdDev || stdDev === "--" || stdDev === "No aplica") return mean;
  return `${mean} ± ${stdDev}`;
}
