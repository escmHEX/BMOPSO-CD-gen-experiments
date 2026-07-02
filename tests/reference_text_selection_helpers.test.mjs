import assert from "node:assert/strict";
import { test } from "node:test";

import {
  nextReferenceTextSelectionSort,
  sortReferenceTextSelectionCandidates,
} from "../LLM/reference_text_selection_helpers.mjs";

test("sortReferenceTextSelectionCandidates orders by score ascending by default", () => {
  const rows = [
    { tweetId: "b", score: 0.4, originalIndex: 2 },
    { tweetId: "a", score: 0.1, originalIndex: 1 },
    { tweetId: "c", score: 0.4, originalIndex: 3 },
  ];

  const sorted = sortReferenceTextSelectionCandidates(rows);

  assert.deepEqual(sorted.map((row) => row.tweetId), ["a", "b", "c"]);
  assert.deepEqual(rows.map((row) => row.tweetId), ["b", "a", "c"]);
});

test("sortReferenceTextSelectionCandidates toggles numeric columns descending", () => {
  const rows = [
    { tweetId: "short", wordCount: 3 },
    { tweetId: "long", wordCount: 12 },
  ];

  const sorted = sortReferenceTextSelectionCandidates(rows, { key: "wordCount", direction: "desc" });

  assert.deepEqual(sorted.map((row) => row.tweetId), ["long", "short"]);
});

test("sortReferenceTextSelectionCandidates sorts text columns case-insensitively", () => {
  const rows = [
    { tweetId: "b", text: "zulu" },
    { tweetId: "a", text: "Alpha" },
  ];

  const sorted = sortReferenceTextSelectionCandidates(rows, { key: "text", direction: "asc" });

  assert.deepEqual(sorted.map((row) => row.tweetId), ["a", "b"]);
});

test("nextReferenceTextSelectionSort switches columns to ascending and toggles same column", () => {
  assert.deepEqual(
    nextReferenceTextSelectionSort({ key: "score", direction: "asc" }, "score"),
    { key: "score", direction: "desc" },
  );
  assert.deepEqual(
    nextReferenceTextSelectionSort({ key: "score", direction: "desc" }, "wordCount"),
    { key: "wordCount", direction: "asc" },
  );
});
