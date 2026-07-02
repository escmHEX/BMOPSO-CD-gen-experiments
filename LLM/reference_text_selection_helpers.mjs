const NUMERIC_SORT_KEYS = new Set([
  "score",
  "semanticDistance",
  "lengthDistance",
  "wordCount",
  "originalIndex",
]);

export const DEFAULT_REFERENCE_TEXT_SELECTION_SORT = {
  key: "score",
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

export function sortReferenceTextSelectionCandidates(candidates, sort = DEFAULT_REFERENCE_TEXT_SELECTION_SORT) {
  const sortState = sort || DEFAULT_REFERENCE_TEXT_SELECTION_SORT;
  const direction = sortState.direction === "desc" ? "desc" : "asc";
  const multiplier = direction === "desc" ? -1 : 1;
  const key = sortState.key || DEFAULT_REFERENCE_TEXT_SELECTION_SORT.key;
  return [...(candidates || [])]
    .map((candidate, index) => ({ candidate, index }))
    .sort((left, right) => {
      let comparison;
      if (NUMERIC_SORT_KEYS.has(key)) {
        comparison = finiteSortNumber(left.candidate[key], direction) - finiteSortNumber(right.candidate[key], direction);
      } else {
        comparison = compareText(left.candidate[key], right.candidate[key]);
      }
      if (comparison !== 0) return comparison * multiplier;
      return left.index - right.index;
    })
    .map((item) => item.candidate);
}
