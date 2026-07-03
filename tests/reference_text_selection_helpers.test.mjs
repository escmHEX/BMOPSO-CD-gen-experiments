import assert from "node:assert/strict";
import { test } from "node:test";

import {
  nextReferenceTextSelectionSort,
  sortReferenceTextSelectionRepresentatives,
} from "../LLM/reference_text_selection_helpers.mjs";

test("sortReferenceTextSelectionRepresentatives shows selected rows first by rank by default", () => {
  const rows = [
    { tweetId: "unselected-best", globalRepresentativity: 0.99 },
    { tweetId: "selected-second", selectionRank: 2, globalRepresentativity: 0.2 },
    { tweetId: "selected-first", selectionRank: 1, globalRepresentativity: 0.1 },
    { tweetId: "unselected-next", globalRepresentativity: 0.5 },
  ];

  const sorted = sortReferenceTextSelectionRepresentatives(rows);

  assert.deepEqual(sorted.map((row) => row.tweetId), [
    "selected-first",
    "selected-second",
    "unselected-best",
    "unselected-next",
  ]);
  assert.deepEqual(rows.map((row) => row.tweetId), [
    "unselected-best",
    "selected-second",
    "selected-first",
    "unselected-next",
  ]);
});

test("sortReferenceTextSelectionRepresentatives sorts numeric columns", () => {
  const rows = [
    { tweetId: "small", clusterSize: 2 },
    { tweetId: "large", clusterSize: 8 },
  ];

  const sorted = sortReferenceTextSelectionRepresentatives(rows, { key: "clusterSize", direction: "desc" });

  assert.deepEqual(sorted.map((row) => row.tweetId), ["large", "small"]);
});

test("sortReferenceTextSelectionRepresentatives sorts text columns case-insensitively", () => {
  const rows = [
    { tweetId: "b", text: "zulu" },
    { tweetId: "a", text: "Alpha" },
  ];

  const sorted = sortReferenceTextSelectionRepresentatives(rows, { key: "text", direction: "asc" });

  assert.deepEqual(sorted.map((row) => row.tweetId), ["a", "b"]);
});

test("nextReferenceTextSelectionSort switches columns to ascending and toggles same column", () => {
  assert.deepEqual(
    nextReferenceTextSelectionSort({ key: "globalRepresentativity", direction: "desc" }, "globalRepresentativity"),
    { key: "globalRepresentativity", direction: "asc" },
  );
  assert.deepEqual(
    nextReferenceTextSelectionSort({ key: "globalRepresentativity", direction: "asc" }, "wordCount"),
    { key: "wordCount", direction: "asc" },
  );
});
