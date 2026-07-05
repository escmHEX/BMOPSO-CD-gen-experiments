import assert from "node:assert/strict";
import { test } from "node:test";

import {
  comparatorHistoricalInstancesFromRun,
  comparatorSameInitialPopulationForBmopsoPayload,
  comparatorSameInitialPopulationGeneratorCandidates,
} from "../LLM/comparator_instance_helpers.mjs";

test("historical comparator instances prefer config proposalInstances over proposal results", () => {
  const run = {
    config: {
      proposalInstances: [
        {
          instanceId: "binary-mopso-cd-4",
          proposalId: "binary-mopso-cd",
          displayName: "Binary MOPSO-CD - config 4",
          baseDisplayName: "Binary MOPSO-CD",
          orderIndex: 3,
          runtimeConfig: {
            n: 16,
            repetitionsK: 3,
          },
          proposalConfig: {
            extraArgs: "",
            cliValues: {
              "router.task_models.synthetic_text_generation": "qwen3:4b-instruct-2507-q4_K_M",
            },
          },
        },
        {
          instanceId: "mesap-1",
          proposalId: "mesap",
          displayName: "MESAP - config 1",
          baseDisplayName: "MESAP",
          proposalConfig: {
            extraArgs: "",
            cliValues: {},
          },
        },
      ],
    },
    proposals: [
      {
        instanceId: "binary-mopso-cd",
        proposalId: "binary-mopso-cd",
        displayName: "Binary MOPSO-CD",
        proposalConfig: {
          extraArgs: "",
          cliValues: {},
        },
      },
    ],
  };

  const instances = comparatorHistoricalInstancesFromRun(run);

  assert.equal(instances.length, 2);
  assert.deepEqual(instances.map((instance) => instance.instanceId), ["binary-mopso-cd-4", "mesap-1"]);
  assert.equal(
    instances[0].proposalConfig.cliValues["router.task_models.synthetic_text_generation"],
    "qwen3:4b-instruct-2507-q4_K_M",
  );
  assert.deepEqual(instances[0].runtimeConfig, { n: 16, repetitionsK: 3 });
  assert.equal(instances[0].orderIndex, 3);
});

test("historical comparator instances keep empty runtime config as defaults", () => {
  const run = {
    config: {
      proposalInstances: [
        {
          instanceId: "evolmd-a",
          proposalId: "evolmd",
          displayName: "EVOLMD A",
          proposalConfig: {
            extraArgs: "",
            cliValues: {},
          },
        },
      ],
    },
  };

  const instances = comparatorHistoricalInstancesFromRun(run);

  assert.deepEqual(instances[0].runtimeConfig, {});
});

test("historical comparator instances preserve multiple instances for the same proposal", () => {
  const run = {
    config: {
      proposalInstances: [
        {
          instanceId: "binary-mopso-cd-5",
          proposalId: "binary-mopso-cd",
          displayName: "Binary MOPSO-CD - config 5",
          proposalConfig: {
            extraArgs: "",
            cliValues: {
              "router.task_thinking.semantic_anchor_extraction": true,
              "router.task_models.semantic_anchor_extraction": "qwen3:4b-instruct-2507-q4_K_M",
            },
          },
        },
        {
          instanceId: "binary-mopso-cd-7",
          proposalId: "binary-mopso-cd",
          displayName: "Binary MOPSO-CD - config 7",
          proposalConfig: {
            extraArgs: "",
            cliValues: {
              "mopso.leader_tournament_size": "8",
              "mopso.p_anchor_max": "0.5",
            },
          },
        },
      ],
    },
  };

  const instances = comparatorHistoricalInstancesFromRun(run);

  assert.deepEqual(instances.map((instance) => instance.instanceId), ["binary-mopso-cd-5", "binary-mopso-cd-7"]);
  assert.equal(instances[0].proposalId, "binary-mopso-cd");
  assert.equal(instances[1].proposalId, "binary-mopso-cd");
  assert.equal(instances[0].proposalConfig.cliValues["router.task_thinking.semantic_anchor_extraction"], true);
  assert.equal(instances[1].proposalConfig.cliValues["mopso.leader_tournament_size"], "8");
});

test("historical comparator instances fall back to proposal results for legacy runs", () => {
  const run = {
    config: {},
    proposals: [
      {
        instanceId: "binary-a",
        proposalId: "binary-mopso-cd",
        displayName: "Binary A",
        baseDisplayName: "Binary MOPSO-CD",
        proposalConfig: {
          extraArgs: "--set custom.flag=1",
          cliValues: {
            "selection.k": "4",
          },
        },
      },
      {
        proposalId: "mesap",
        displayName: "MESAP",
      },
    ],
  };

  const instances = comparatorHistoricalInstancesFromRun(run);

  assert.deepEqual(instances.map((instance) => instance.instanceId), ["binary-a", "mesap"]);
  assert.equal(instances[0].proposalConfig.extraArgs, "--set custom.flag=1");
  assert.equal(instances[0].proposalConfig.cliValues["selection.k"], "4");
  assert.deepEqual(instances[1].proposalConfig, { extraArgs: "", cliValues: {} });
});

test("historical comparator instances apply persistent labels by instance id", () => {
  const run = {
    instanceLabels: {
      "binary-a": "Binary MOPSO-CD - tesis",
    },
    config: {
      proposalInstances: [
        {
          instanceId: "binary-a",
          proposalId: "binary-mopso-cd",
          displayName: "Binary MOPSO-CD - old",
          baseDisplayName: "Binary MOPSO-CD",
        },
      ],
    },
  };

  const instances = comparatorHistoricalInstancesFromRun(run);

  assert.equal(instances[0].displayName, "Binary MOPSO-CD - tesis");
  assert.equal(instances[0].baseDisplayName, "Binary MOPSO-CD");
});

test("same initial population generator candidates are limited to Binary instances", () => {
  const instances = [
    { instanceId: "binary-a", proposalId: "binary-mopso-cd", displayName: "Binary A" },
    { instanceId: "mesap-a", proposalId: "mesap", displayName: "MESAP A" },
    { instanceId: "binary-b", proposalId: "binary-mopso-cd", displayName: "Binary B" },
  ];

  const candidates = comparatorSameInitialPopulationGeneratorCandidates(instances);

  assert.deepEqual(candidates.map((candidate) => candidate.instanceId), ["binary-a", "binary-b"]);
});

test("same initial population payload reassigns invalid generator to first Binary instance", () => {
  const instances = [
    { instanceId: "mesap-a", proposalId: "mesap", displayName: "MESAP A" },
    { instanceId: "binary-b", proposalId: "binary-mopso-cd", displayName: "Binary B" },
  ];

  const payload = comparatorSameInitialPopulationForBmopsoPayload(
    { enabled: true, generatorInstanceId: "deleted-binary" },
    instances,
  );

  assert.deepEqual(payload, {
    enabled: true,
    generatorInstanceId: "binary-b",
    scope: "per_repetition",
  });
});

test("same initial population payload stays disabled by default", () => {
  const payload = comparatorSameInitialPopulationForBmopsoPayload(null, [
    { instanceId: "binary-a", proposalId: "binary-mopso-cd" },
  ]);

  assert.deepEqual(payload, {
    enabled: false,
    generatorInstanceId: null,
    scope: "per_repetition",
  });
});
