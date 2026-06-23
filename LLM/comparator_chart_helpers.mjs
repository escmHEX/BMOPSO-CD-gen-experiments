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

function cleanAxisNumber(value) {
  return Number(Number(value).toPrecision(12));
}

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

export function comparatorProposalColor(proposal, index = 0) {
  if (comparatorIsBinaryProposal(proposal)) return "#2A8C00";
  const palette = ["#2458b8", "#0f766e", "#b42318", "#7c3aed", "#ca8a04"];
  return palette[index % palette.length];
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
