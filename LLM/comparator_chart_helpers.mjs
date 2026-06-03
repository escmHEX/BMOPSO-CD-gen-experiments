export function comparatorPointCoordinates(point) {
  const value = point?.value || [];
  const x = Number(point?.x ?? value[0]);
  const y = Number(point?.y ?? value[1]);
  if (!Number.isFinite(x) || !Number.isFinite(y)) return null;
  return { x, y };
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
    counts.set(point.proposalId, (counts.get(point.proposalId) || 0) + 1);
    return counts;
  }, new Map());
}

export function comparatorMetricMetadata(metricKey) {
  if (metricKey === "spread") {
    return { description: "Menor spread es mejor.", higherIsBetter: false };
  }
  if (metricKey === "nonDominatedRows") {
    return { description: "Mayor cantidad de soluciones no dominadas es mejor.", higherIsBetter: true };
  }
  return { description: "Mayor HV es mejor.", higherIsBetter: true };
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
