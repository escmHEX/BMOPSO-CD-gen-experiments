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

export const COMPARATOR_TRANSPARENT_BACKGROUND = "rgba(0, 0, 0, 0)";

export function comparatorFormatAxisTick(value, decimals = 2) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return String(value ?? "");
  const precision = Math.max(0, Math.min(6, Number(decimals) || 0));
  const rounded = Number(numeric.toFixed(precision));
  return Object.is(rounded, -0) ? "0" : String(rounded);
}

export function comparatorChartAxisTickFormatter(value) {
  return comparatorFormatAxisTick(value);
}

function normalizeComparableText(value) {
  return String(value || "")
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .trim()
    .toLowerCase();
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

function finiteGenerationIndex(value) {
  const numeric = Number(value);
  return Number.isInteger(numeric) && numeric >= 0 ? numeric : null;
}

export function comparatorInternalBmopsoFrontOptions(analysis = {}) {
  const baseAnalysis = analysis && typeof analysis === "object" ? analysis : {};
  const finalAnalysis = {
    ...baseAnalysis,
    charts: normalChartPayload(baseAnalysis.charts),
    metrics: normalMetricsPayload(baseAnalysis.metrics),
  };
  const options = [{
    key: "final",
    label: "Final",
    generation: null,
    analysis: finalAnalysis,
  }];
  const iterationFronts = Array.isArray(baseAnalysis.iterationFronts) ? baseAnalysis.iterationFronts : [];
  iterationFronts
    .filter((front) => front && typeof front === "object" && finiteGenerationIndex(front.generation) !== null)
    .sort((left, right) => finiteGenerationIndex(left.generation) - finiteGenerationIndex(right.generation))
    .forEach((front) => {
      const generation = finiteGenerationIndex(front.generation);
      const charts = normalChartPayload(front.charts);
      options.push({
        key: `generation:${generation}`,
        label: `Iteración ${generation}`,
        generation,
        analysis: {
          ...baseAnalysis,
          source: front.source || baseAnalysis.source,
          metrics: normalMetricsPayload(front.metrics),
          charts: {
            ...charts,
            selected: [],
          },
          iterationFronts: [],
        },
      });
    });
  return options;
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

export function comparatorVisibleFrontCounts(charts = {}, excludedKeys = new Set(), namespace = "pareto-points") {
  const visibleFront = comparatorVisibleFrontChartPoints(charts, namespace);
  const selectedFrontPoints = comparatorSelectedFrontPointsForIndividuals(
    visibleFront.individuals,
    visibleFront.selected,
    namespace,
  );
  return {
    front: comparatorPartitionPointsByExclusion(visibleFront.individuals, excludedKeys, namespace).active.length,
    selected: comparatorPartitionPointsByExclusion(selectedFrontPoints, excludedKeys, namespace).active.length,
  };
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

export function comparatorSelectedFrontPointsWithEditable(
  baseSelectedPoints = [],
  frontPoints = [],
  selectedKeys = new Set(),
  excludedKeys = new Set(),
  namespace = "",
) {
  const selected = Array.isArray(baseSelectedPoints) ? [...baseSelectedPoints] : [];
  const editableKeys = selectedKeys instanceof Set ? selectedKeys : new Set(selectedKeys || []);
  if (!editableKeys.size) return selected;
  const excluded = excludedKeys instanceof Set ? excludedKeys : new Set(excludedKeys || []);
  const selectedMembership = new Set(
    selected
      .map((point) => comparatorFrontMembershipKey(point, namespace))
      .filter(Boolean),
  );
  (Array.isArray(frontPoints) ? frontPoints : []).forEach((point, index) => {
    const interactionKey = point?.pointInteractionKey || comparatorPointInteractionKey(point, index, namespace);
    if (!editableKeys.has(interactionKey) || excluded.has(interactionKey)) return;
    const membershipKey = comparatorFrontMembershipKey(point, namespace);
    if (membershipKey && selectedMembership.has(membershipKey)) return;
    selected.push(point);
    if (membershipKey) {
      selectedMembership.add(membershipKey);
    }
  });
  return selected;
}

export function comparatorClearPointExclusions(excludedKeys) {
  if (!(excludedKeys instanceof Set) || excludedKeys.size === 0) return false;
  excludedKeys.clear();
  return true;
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

export function comparatorIterationAxisWindow(series = []) {
  const extent = comparatorSeriesIterationExtent(series);
  if (!extent) return null;
  let min = Math.floor(extent.min);
  let max = Math.ceil(extent.max);
  if (min === max) {
    if (min > 0) {
      min -= 1;
    } else {
      max += 1;
    }
  }
  return {
    defaultMin: min,
    defaultMax: max,
    zoomMin: min,
    zoomMax: max,
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

export function comparatorSeriesWithFinalMetricReplacement(series = [], metricKey = "", replacementValue = null) {
  if (replacementValue === null || replacementValue === undefined || replacementValue === "") return series;
  const adjustedValue = Number(replacementValue);
  if (!metricKey || !Number.isFinite(adjustedValue)) return series;

  let replacementIndex = -1;
  let replacementGeneration = -Infinity;
  (series || []).forEach((point, index) => {
    const generation = finiteIterationValue(point);
    const rawMetricValue = point?.[metricKey];
    const metricValue = rawMetricValue === null || rawMetricValue === undefined || rawMetricValue === ""
      ? NaN
      : Number(rawMetricValue);
    if (generation === null || !Number.isFinite(metricValue)) return;
    if (generation >= replacementGeneration) {
      replacementGeneration = generation;
      replacementIndex = index;
    }
  });
  if (replacementIndex === -1) return series;

  return series.map((point, index) => (
    index === replacementIndex ? { ...point, [metricKey]: adjustedValue } : point
  ));
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

const COMPARATOR_DEFAULT_CHART_LABELS = Object.freeze({
  pareto: Object.freeze({
    title: "Frente de Pareto",
    xAxis: "Fidelidad normalizada",
    yAxis: "Diversidad normalizada",
  }),
  combinedSelected: Object.freeze({
    title: "Soluciones seleccionadas combinadas",
    xAxis: "Fidelidad normalizada",
    yAxis: "Diversidad normalizada",
  }),
  contributionScatter: Object.freeze({
    title: "Contribución al frente combinado",
    xAxis: "Fidelidad normalizada",
    yAxis: "Diversidad normalizada",
  }),
  hypervolume: Object.freeze({
    title: "Hipervolumen",
    xAxis: "Iteración",
    yAxis: "Hipervolumen",
  }),
  contribution: Object.freeze({
    title: "Contribución",
    xAxis: "Iteración",
    yAxis: "Contribución",
  }),
  extent: Object.freeze({
    title: "Extensión del frente",
    xAxis: "Iteración",
    yAxis: "Extensión",
  }),
  unaryEntropy: Object.freeze({
    title: "Entropía unaria",
    xAxis: "Iteración",
    yAxis: "Entropía unaria",
  }),
  globalInertia: Object.freeze({
    title: "Inercia global de embeddings",
    xAxis: "Iteración",
    yAxis: "Inercia global",
  }),
  globalEntropy: Object.freeze({
    title: "Entropía global de entidades",
    xAxis: "Iteración",
    yAxis: "Entropía global",
  }),
  embeddingOverlay: Object.freeze({
    title: "Proyección combinada de diversidad",
    xAxis: "Dimensión proyectada 1",
    yAxis: "Dimensión proyectada 2",
  }),
  embeddingProjection: Object.freeze({
    title: "Proyección de diversidad",
    xAxis: "Dimensión proyectada 1",
    yAxis: "Dimensión proyectada 2",
  }),
  internalBmopsoPareto: Object.freeze({
    title: "Frente interno de Binary MOPSO-CD",
    xAxis: "Objetivo nativo normalizado 1",
    yAxis: "Objetivo nativo normalizado 2",
  }),
  internalBmopsoHypervolume: Object.freeze({
    title: "Hipervolumen interno de Binary MOPSO-CD",
    xAxis: "Iteración",
    yAxis: "Hipervolumen interno",
  }),
});

const COMPARATOR_LEGACY_CHART_LABELS = Object.freeze({
  contributionScatter: Object.freeze({
    title: "Contribucion al frente combinado",
  }),
  hypervolume: Object.freeze({
    xAxis: "Iteracion",
  }),
  contribution: Object.freeze({
    title: "Contribucion",
    xAxis: "Iteracion",
    yAxis: "Contribucion",
  }),
  extent: Object.freeze({
    title: "Extension del frente",
    xAxis: "Iteracion",
    yAxis: "Extension",
  }),
  unaryEntropy: Object.freeze({
    title: "Entropia unaria",
    xAxis: "Iteracion",
    yAxis: "Entropia unaria",
  }),
  globalInertia: Object.freeze({
    xAxis: "Iteracion",
  }),
  globalEntropy: Object.freeze({
    title: "Entropia global de entidades",
    xAxis: "Iteracion",
    yAxis: "Entropia global",
  }),
  embeddingOverlay: Object.freeze({
    title: "Proyeccion combinada de diversidad",
    xAxis: "Dimension proyectada 1",
    yAxis: "Dimension proyectada 2",
  }),
  embeddingProjection: Object.freeze({
    title: "Proyeccion de diversidad",
    xAxis: "Dimension proyectada 1",
    yAxis: "Dimension proyectada 2",
  }),
  internalBmopsoHypervolume: Object.freeze({
    xAxis: "Iteracion",
  }),
});

export const COMPARATOR_SELECTED_STAR_SYMBOL = "path://M12,2L14.9,8.6L22,9.2L16.7,13.8L18.3,20.8L12,17.1L5.7,20.8L7.3,13.8L2,9.2L9.1,8.6Z";

function chartLabelValue(value, fallback) {
  const text = String(value ?? "").trim();
  return text || fallback;
}

function chartLabelOverrideValue(value, fallback, legacyValue) {
  const text = chartLabelValue(value, fallback);
  return legacyValue !== undefined && text === legacyValue ? fallback : text;
}

export function comparatorDefaultChartLabels() {
  return Object.fromEntries(
    Object.entries(COMPARATOR_DEFAULT_CHART_LABELS).map(([key, value]) => [key, { ...value }]),
  );
}

export function comparatorMergeChartLabels(overrides = {}, defaults = comparatorDefaultChartLabels()) {
  const source = overrides && typeof overrides === "object" ? overrides : {};
  return Object.fromEntries(
    Object.entries(defaults).map(([key, defaultValue]) => {
      const override = source[key] && typeof source[key] === "object" ? source[key] : {};
      const legacy = COMPARATOR_LEGACY_CHART_LABELS[key] || {};
      return [key, {
        title: chartLabelOverrideValue(override.title, defaultValue.title, legacy.title),
        xAxis: chartLabelOverrideValue(override.xAxis, defaultValue.xAxis, legacy.xAxis),
        yAxis: chartLabelOverrideValue(override.yAxis, defaultValue.yAxis, legacy.yAxis),
      }];
    }),
  );
}

function comparatorSeriesLegendColor(series) {
  return series?.itemStyle?.borderColor
    || series?.itemStyle?.color
    || series?.lineStyle?.color
    || series?.areaStyle?.color
    || "#64748b";
}

function comparatorLegendIcon(series) {
  if (series?.symbol === "star" || series?.symbol === COMPARATOR_SELECTED_STAR_SYMBOL) {
    return COMPARATOR_SELECTED_STAR_SYMBOL;
  }
  if (series?.type === "line") return "path://M0,5L28,5L28,7L0,7Z";
  return "circle";
}

export function comparatorPublicationLegendEntries(series = [], selected = {}) {
  const seen = new Set();
  return (series || []).map((item) => {
    if (item?.showInLegend === false) return null;
    const name = String(item?.name || "").trim();
    if (!name || seen.has(name) || selected[name] === false) return null;
    seen.add(name);
    return {
      name,
      color: comparatorSeriesLegendColor(item),
      icon: comparatorLegendIcon(item),
      textColor: "#111111",
    };
  }).filter(Boolean);
}

const COMPARATOR_PUBLICATION_LEGEND_BASE_WIDTH = 420;
const COMPARATOR_PUBLICATION_LEGEND_GAP = 20;
const COMPARATOR_PUBLICATION_LEGEND_RIGHT = 8;
const COMPARATOR_PUBLICATION_LEGEND_HORIZONTAL_PADDING = 32;
const COMPARATOR_PUBLICATION_LEGEND_SYMBOL_SPACE = 42;
const COMPARATOR_PUBLICATION_LEGEND_WIDTH_SAFETY = 8;
const COMPARATOR_PUBLICATION_LEGEND_RENDER_SAFETY = 12;

function estimatedPublicationLegendTextWidth(text, measureText = null) {
  if (typeof measureText === "function") {
    const measured = Number(measureText(String(text || "")));
    if (Number.isFinite(measured) && measured > 0) return measured;
  }
  const chars = Array.from(String(text || ""));
  const weightedWidth = chars.reduce((width, char) => {
    if (/\s/.test(char)) return width + 4;
    if (/[ilI1|.,:;]/.test(char)) return width + 3.8;
    if (/[mwMW@%]/.test(char)) return width + 10;
    if (/[A-ZÁÉÍÓÚÑ]/.test(char)) return width + 8;
    if (/[0-9]/.test(char)) return width + 7.1;
    if (/[-_=+()[\]{}<>/\\]/.test(char)) return width + 5.2;
    return width + 6.8;
  }, 0);
  return Math.max(weightedWidth, chars.length * 6.6);
}

export function comparatorPublicationLegendLayout(series = [], selected = {}, options = {}) {
  const entries = comparatorPublicationLegendEntries(series, selected);
  const measureText = typeof options.measureText === "function" ? options.measureText : null;
  const longestTextWidth = entries.reduce((max, entry) => (
    Math.max(max, estimatedPublicationLegendTextWidth(entry.name, measureText))
  ), 0);
  const width = entries.length
    ? Math.max(
        228,
        Math.ceil(
          longestTextWidth
          + COMPARATOR_PUBLICATION_LEGEND_SYMBOL_SPACE
          + COMPARATOR_PUBLICATION_LEGEND_WIDTH_SAFETY,
        ),
      )
    : 0;
  const boxWidth = entries.length ? width + COMPARATOR_PUBLICATION_LEGEND_HORIZONTAL_PADDING : 0;
  const reservedWidth = entries.length ? boxWidth + COMPARATOR_PUBLICATION_LEGEND_RENDER_SAFETY : 0;
  const legendRight = COMPARATOR_PUBLICATION_LEGEND_RIGHT;
  const legendGap = COMPARATOR_PUBLICATION_LEGEND_GAP;
  return {
    entries,
    width,
    boxWidth,
    reservedWidth,
    legendGap,
    legendRight,
    gridRight: entries.length ? reservedWidth + legendRight + legendGap : 24,
  };
}

export function comparatorPublicationExportWidth(series = [], selected = {}, baseWidth = 1280, options = {}) {
  const width = Number(baseWidth);
  const safeBaseWidth = Number.isFinite(width) && width > 0 ? width : 1280;
  const layout = comparatorPublicationLegendLayout(series, selected, options);
  return safeBaseWidth + Math.max(0, layout.reservedWidth - COMPARATOR_PUBLICATION_LEGEND_BASE_WIDTH);
}

function cloneComparatorChartOptionValue(value) {
  if (Array.isArray(value)) return value.map((item) => cloneComparatorChartOptionValue(item));
  if (!value || typeof value !== "object") return value;
  return Object.fromEntries(
    Object.entries(value).map(([key, entryValue]) => [key, cloneComparatorChartOptionValue(entryValue)]),
  );
}

function asComponentArray(component) {
  if (component === undefined || component === null) return [];
  return Array.isArray(component) ? component : [component];
}

function normalizeLegendName(entry) {
  return String(typeof entry === "string" ? entry : entry?.name || "").trim();
}

function stripTitleSubtext(title) {
  const clean = (entry) => {
    if (!entry || typeof entry !== "object") return entry;
    const next = { ...entry };
    delete next.subtext;
    delete next.subtextStyle;
    return next;
  };
  return Array.isArray(title) ? title.map(clean) : clean(title);
}

function numericLayoutValue(value, fallback = 0) {
  if (Number.isFinite(Number(value))) return Number(value);
  return fallback;
}

function firstComponent(component) {
  return asComponentArray(component)[0] || null;
}

function positionExportTitle(title, grid, exportWidth) {
  if (!Number.isFinite(Number(exportWidth)) || Number(exportWidth) <= 0) return title;
  const gridEntry = firstComponent(grid);
  if (!gridEntry || typeof gridEntry !== "object") return title;
  const left = numericLayoutValue(gridEntry.left, 0);
  const right = numericLayoutValue(gridEntry.right, 0);
  const top = numericLayoutValue(gridEntry.top, 72);
  const center = (left + (Number(exportWidth) - right)) / 2;
  const titleTop = Math.max(8, top - 50);
  const position = (entry) => {
    if (!entry || typeof entry !== "object") return entry;
    return {
      ...entry,
      left: center,
      top: titleTop,
      textAlign: "center",
    };
  };
  return Array.isArray(title) ? title.map((entry, index) => (index === 0 ? position(entry) : entry)) : position(title);
}

function positionExportLegend(legend, grid, exportWidth) {
  const width = Number(exportWidth);
  if (!Number.isFinite(width) || width <= 0) return legend;
  const gridEntry = firstComponent(grid);
  if (!gridEntry || typeof gridEntry !== "object") return legend;
  const gridRight = Number(gridEntry.right);
  if (!Number.isFinite(gridRight)) return legend;
  const legendGap = Number(firstComponent(legend)?.comparatorLegendGap);
  const left = width - gridRight + (Number.isFinite(legendGap) ? legendGap : COMPARATOR_PUBLICATION_LEGEND_GAP);
  const position = (entry) => {
    if (!entry || typeof entry !== "object") return entry;
    const next = {
      ...entry,
      left,
    };
    delete next.right;
    delete next.comparatorLegendGap;
    return next;
  };
  return Array.isArray(legend) ? legend.map(position) : position(legend);
}

function isWhiteBackground(value) {
  const normalized = String(value || "").replace(/\s+/g, "").toLowerCase();
  return normalized === "#fff"
    || normalized === "#ffffff"
    || normalized === "white"
    || normalized === "rgb(255,255,255)"
    || normalized === "rgba(255,255,255,1)";
}

function cleanExportGrid(grid) {
  const clean = (entry) => {
    if (!entry || typeof entry !== "object") return entry;
    const next = { ...entry };
    if (isWhiteBackground(next.backgroundColor)) {
      next.backgroundColor = COMPARATOR_TRANSPARENT_BACKGROUND;
    }
    return next;
  };
  return Array.isArray(grid) ? grid.map(clean) : clean(grid);
}

function cleanExportLegend(legend, series) {
  const selected = asComponentArray(legend)[0]?.selected || {};
  const visibleNames = new Set(
    (series || [])
      .map((item) => String(item?.name || "").trim())
      .filter((name) => name && selected[name] !== false),
  );
  const clean = (entry) => {
    if (!entry || typeof entry !== "object") return entry;
    const next = { ...entry };
    if (isWhiteBackground(next.backgroundColor)) {
      next.backgroundColor = COMPARATOR_TRANSPARENT_BACKGROUND;
    }
    if (Array.isArray(next.data)) {
      next.data = next.data.filter((item) => visibleNames.has(normalizeLegendName(item)));
    }
    next.show = next.data ? next.data.length > 0 : visibleNames.size > 0;
    return next;
  };
  return Array.isArray(legend) ? legend.map(clean) : clean(legend);
}

function normalizeDataZoomAxisIndexes(value) {
  if (value === undefined || value === null) return [];
  return (Array.isArray(value) ? value : [value])
    .map((entry) => Number(entry))
    .filter((entry) => Number.isInteger(entry) && entry >= 0);
}

function dataZoomWindowForAxis(dataZoom, axisIndexKey, axisIndex) {
  return (dataZoom || []).find((zoom) => {
    if (!zoom || typeof zoom !== "object") return false;
    const indexes = normalizeDataZoomAxisIndexes(zoom[axisIndexKey]);
    if (!indexes.includes(axisIndex)) return false;
    return Number.isFinite(Number(zoom.startValue)) && Number.isFinite(Number(zoom.endValue));
  }) || null;
}

function outwardRoundedAxisBound(value, direction) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return value;
  const scaled = numeric * 100;
  const rounded = direction === "min" ? Math.floor(scaled) / 100 : Math.ceil(scaled) / 100;
  return Object.is(rounded, -0) ? 0 : rounded;
}

function roundedExportAxisInterval(min, max) {
  const low = Number(min);
  const high = Number(max);
  if (!Number.isFinite(low) || !Number.isFinite(high) || high <= low) return null;
  const rawInterval = (high - low) / 5;
  if (!Number.isFinite(rawInterval) || rawInterval <= 0) return null;
  const magnitude = 10 ** Math.floor(Math.log10(rawInterval));
  const normalized = rawInterval / magnitude;
  const niceStep = normalized <= 1 ? 1 : normalized <= 2 ? 2 : normalized <= 5 ? 5 : 10;
  const interval = niceStep * magnitude;
  const rounded = interval >= 1
    ? Math.ceil(interval)
    : Math.ceil(interval * 100) / 100;
  return rounded > 0 ? rounded : null;
}

function isIterationAxis(entry) {
  return normalizeComparableText(entry?.name) === "iteracion";
}

function stepDecimals(step) {
  const text = String(step);
  if (!text.includes(".")) return 0;
  return text.split(".")[1].length;
}

function cleanStepNumber(value, decimals = 2) {
  const precision = Math.max(0, Math.min(6, Number(decimals) || 0));
  const rounded = Number(Number(value).toFixed(precision));
  return Object.is(rounded, -0) ? 0 : rounded;
}

function ceilNiceStep(rawStep, decimals = 2) {
  const numeric = Number(rawStep);
  const minimum = 10 ** -Math.max(0, Math.min(6, Number(decimals) || 0));
  if (!Number.isFinite(numeric) || numeric <= minimum) return minimum;
  const magnitude = 10 ** Math.floor(Math.log10(numeric));
  for (const multiplier of [1, 2, 5, 10]) {
    const candidate = cleanStepNumber(multiplier * magnitude, Math.max(decimals, stepDecimals(multiplier * magnitude)));
    if (candidate >= numeric && candidate >= minimum) return cleanStepNumber(candidate, decimals);
  }
  return cleanStepNumber(10 * magnitude, decimals);
}

function floorToStep(value, step) {
  return cleanStepNumber(Math.floor((Number(value) + Number.EPSILON) / step) * step);
}

function ceilToStep(value, step) {
  return cleanStepNumber(Math.ceil((Number(value) - Number.EPSILON) / step) * step);
}

export function comparatorRegularAxisScale(min, max, options = {}) {
  const low = Number(min);
  const high = Number(max);
  if (!Number.isFinite(low) || !Number.isFinite(high)) return null;
  const decimals = Math.max(0, Math.min(2, Number(options.decimals ?? 2)));
  const targetIntervals = Math.max(1, Math.trunc(Number(options.targetIntervals) || 5));
  const minIntervals = Math.max(1, Math.trunc(Number(options.minIntervals) || 5));
  const dataMin = Math.min(low, high);
  const dataMax = Math.max(low, high);
  const rawRange = dataMax - dataMin;
  const minimumStep = 10 ** -decimals;
  const interval = ceilNiceStep(Math.max(rawRange / targetIntervals, minimumStep), decimals);
  const minimumSpan = interval * minIntervals;
  let scaleMin = floorToStep(dataMin, interval);
  let scaleMax = ceilToStep(dataMax, interval);

  if (scaleMax <= scaleMin) {
    const center = cleanStepNumber((dataMin + dataMax) / 2, decimals);
    scaleMin = floorToStep(center - (minimumSpan / 2), interval);
    scaleMax = cleanStepNumber(scaleMin + minimumSpan, decimals);
  }

  if ((scaleMax - scaleMin) < minimumSpan) {
    const targetMaxFromMin = cleanStepNumber(scaleMin + minimumSpan, decimals);
    if (targetMaxFromMin >= dataMax) {
      scaleMax = targetMaxFromMin;
    } else {
      scaleMax = ceilToStep(dataMax, interval);
      scaleMin = cleanStepNumber(scaleMax - minimumSpan, decimals);
    }
  }

  while (scaleMin > dataMin) scaleMin = cleanStepNumber(scaleMin - interval, decimals);
  while (scaleMax < dataMax) scaleMax = cleanStepNumber(scaleMax + interval, decimals);

  const intervalCount = Math.max(1, Math.round((scaleMax - scaleMin) / interval));
  const ticks = Array.from({ length: intervalCount + 1 }, (_, index) =>
    cleanStepNumber(scaleMin + (interval * index), decimals));
  const labels = ticks.map((tick) => comparatorFormatAxisTick(tick, decimals));
  if (new Set(labels).size !== labels.length) {
    const expandedMax = cleanStepNumber(scaleMin + (minimumStep * minIntervals), decimals);
    return comparatorRegularAxisScale(scaleMin, expandedMax, { decimals, targetIntervals, minIntervals });
  }

  return {
    min: scaleMin,
    max: scaleMax,
    interval,
    splitNumber: intervalCount,
    ticks,
  };
}

export function comparatorRegularAxisScaleForPoints(points = [], axis = "y", options = {}) {
  const values = (points || [])
    .map((point) => finiteAxisValue(point, axis))
    .filter((value) => value !== null);
  if (!values.length) return null;
  return comparatorRegularAxisScale(Math.min(...values), Math.max(...values), options);
}

function roundedIterationAxisMax(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric) || numeric <= 0) return 10;
  return Math.max(10, Math.ceil(numeric / 10) * 10);
}

function cleanExportAxis(axis, dataZoom, axisIndexKey) {
  const clean = (entry, axisIndex) => {
    if (!entry || typeof entry !== "object") return entry;
    const next = { ...entry };
    const zoom = dataZoomWindowForAxis(dataZoom, axisIndexKey, axisIndex);
    if (zoom) {
      const startValue = Number(zoom.startValue);
      const endValue = Number(zoom.endValue);
      next.min = outwardRoundedAxisBound(Math.min(startValue, endValue), "min");
      next.max = outwardRoundedAxisBound(Math.max(startValue, endValue), "max");
    }
    if (axisIndexKey === "xAxisIndex" && isIterationAxis(next)) {
      next.min = 0;
      next.max = roundedIterationAxisMax(next.max);
      next.interval = 10;
      next.minInterval = 10;
    } else if (axisIndexKey === "yAxisIndex") {
      const scale = comparatorRegularAxisScale(next.min, next.max, { minIntervals: 3 });
      if (scale) {
        next.min = scale.min;
        next.max = scale.max;
        next.interval = scale.interval;
        next.splitNumber = scale.splitNumber;
      }
    }
    next.axisLabel = {
      ...(next.axisLabel || {}),
      formatter: comparatorChartAxisTickFormatter,
    };
    const interval = axisIndexKey === "xAxisIndex" && isIterationAxis(next)
      ? null
      : roundedExportAxisInterval(next.min, next.max);
    if (interval !== null && next.interval === undefined) {
      next.interval = interval;
    }
    return next;
  };
  return Array.isArray(axis) ? axis.map((entry, index) => clean(entry, index)) : clean(axis, 0);
}

export function comparatorChartExportOption(option = {}, exportLayout = {}) {
  const next = cloneComparatorChartOptionValue(option || {});
  const dataZoom = asComponentArray(next.dataZoom);
  next.backgroundColor = COMPARATOR_TRANSPARENT_BACKGROUND;
  next.animation = false;
  next.animationDuration = 0;
  next.animationDurationUpdate = 0;
  next.stateAnimation = { duration: 0 };
  next.title = stripTitleSubtext(next.title);
  next.xAxis = cleanExportAxis(next.xAxis, dataZoom, "xAxisIndex");
  next.yAxis = cleanExportAxis(next.yAxis, dataZoom, "yAxisIndex");
  delete next.toolbox;
  delete next.dataZoom;
  delete next.brush;
  next.graphic = [];
  next.series = asComponentArray(next.series)
    .filter((series) => !series?.comparatorExportExclude)
    .map((series) => {
      if (!series || typeof series !== "object") return series;
      const cleanSeries = { ...series };
      delete cleanSeries.markLine;
      delete cleanSeries.comparatorExportExclude;
      cleanSeries.animation = false;
      cleanSeries.animationDuration = 0;
      cleanSeries.animationDurationUpdate = 0;
      cleanSeries.clip = true;
      return cleanSeries;
    });
  next.legend = cleanExportLegend(next.legend, next.series);
  next.grid = cleanExportGrid(next.grid);
  next.legend = positionExportLegend(next.legend, next.grid, exportLayout.exportWidth);
  next.title = positionExportTitle(next.title, next.grid, exportLayout.exportWidth);
  return next;
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

export function comparatorMetricReferenceLines(values, higherIsBetter) {
  const extremes = comparatorMetricExtremes(values, higherIsBetter);
  if (!extremes) return null;
  return {
    symbol: "none",
    silent: true,
    data: [
      {
        name: "Mejor",
        yAxis: extremes.bestValue,
        lineStyle: { color: "#16a34a", type: "dashed", width: 2 },
        label: {
          color: "#166534",
          formatter: "Mejor: {c}",
          position: "insideEndTop",
          distance: 4,
        },
      },
      {
        name: "Peor",
        yAxis: extremes.worstValue,
        lineStyle: { color: "#dc2626", type: "dashed", width: 2 },
        label: {
          color: "#991b1b",
          formatter: "Peor: {c}",
          position: "insideEndBottom",
          distance: 4,
        },
      },
    ],
  };
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
  return { description: "Mayor hipervolumen es mejor: más área dominada respecto a [0,0].", higherIsBetter: true };
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
