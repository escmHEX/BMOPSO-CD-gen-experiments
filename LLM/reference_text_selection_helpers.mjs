const NUMERIC_SORT_KEYS = new Set([
  "selectionRank",
  "wordCount",
  "clusterDisplayIndex",
  "clusterSize",
  "scoreLocal",
  "globalRepresentativity",
  "mmrScore",
  "minSemanticDistanceToSelected",
  "originalIndex",
]);

export const DEFAULT_REFERENCE_TEXT_SELECTION_SORT = {
  key: "__default",
  direction: "asc",
};

function finiteSortNumber(value, direction) {
  const number = Number(value);
  if (Number.isFinite(number)) return number;
  return direction === "desc" ? Number.NEGATIVE_INFINITY : Number.POSITIVE_INFINITY;
}

function compareText(left, right) {
  return String(left ?? "").localeCompare(String(right ?? ""), undefined, {
    numeric: true,
    sensitivity: "base",
  });
}

function compareDefaultRepresentatives(left, right) {
  const leftRank = Number(left.representative.selectionRank);
  const rightRank = Number(right.representative.selectionRank);
  const leftSelected = Number.isFinite(leftRank);
  const rightSelected = Number.isFinite(rightRank);
  if (leftSelected !== rightSelected) return leftSelected ? -1 : 1;
  if (leftSelected && rightSelected && leftRank !== rightRank) return leftRank - rightRank;
  if (!leftSelected && !rightSelected) {
    const qualityDiff =
      finiteSortNumber(right.representative.globalRepresentativity, "asc") -
      finiteSortNumber(left.representative.globalRepresentativity, "asc");
    if (qualityDiff !== 0) return qualityDiff;
  }
  return left.index - right.index;
}

export function nextReferenceTextSelectionSort(currentSort, key) {
  const current = currentSort || DEFAULT_REFERENCE_TEXT_SELECTION_SORT;
  if (current.key === key) {
    return {
      key,
      direction: current.direction === "asc" ? "desc" : "asc",
    };
  }
  return { key, direction: "asc" };
}

export function sortReferenceTextSelectionRepresentatives(
  representatives,
  sort = DEFAULT_REFERENCE_TEXT_SELECTION_SORT,
) {
  const sortState = sort || DEFAULT_REFERENCE_TEXT_SELECTION_SORT;
  const key = sortState.key || DEFAULT_REFERENCE_TEXT_SELECTION_SORT.key;
  const direction = sortState.direction === "desc" ? "desc" : "asc";
  const multiplier = direction === "desc" ? -1 : 1;
  return [...(representatives || [])]
    .map((representative, index) => ({ representative, index }))
    .sort((left, right) => {
      if (key === DEFAULT_REFERENCE_TEXT_SELECTION_SORT.key) {
        return compareDefaultRepresentatives(left, right);
      }
      let comparison;
      if (NUMERIC_SORT_KEYS.has(key)) {
        comparison =
          finiteSortNumber(left.representative[key], direction) -
          finiteSortNumber(right.representative[key], direction);
      } else {
        comparison = compareText(left.representative[key], right.representative[key]);
      }
      if (comparison !== 0) return comparison * multiplier;
      return left.index - right.index;
    })
    .map((item) => item.representative);
}
