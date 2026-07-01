function objectValue(value) {
  return value && typeof value === "object" && !Array.isArray(value) ? value : {};
}

function stringValue(value) {
  return String(value ?? "").trim();
}

function normalizeProposalConfig(value) {
  const config = objectValue(value);
  return {
    extraArgs: String(config.extraArgs ?? ""),
    cliValues: { ...objectValue(config.cliValues) },
  };
}

function normalizeHistoricalInstance(value, index) {
  const source = objectValue(value);
  const proposalId = stringValue(source.proposalId || source.instanceId);
  if (!proposalId) return null;
  const instanceId = stringValue(source.instanceId || proposalId);
  const displayName = stringValue(source.displayName || instanceId);
  return {
    instanceId,
    proposalId,
    displayName,
    baseDisplayName: stringValue(source.baseDisplayName || displayName),
    proposalConfig: normalizeProposalConfig(source.proposalConfig),
    orderIndex: Number.isFinite(source.orderIndex) ? source.orderIndex : index,
  };
}

const SAME_INITIAL_POPULATION_SCOPE = "per_repetition";
const BINARY_PROPOSAL_ID = "binary-mopso-cd";

export function comparatorSameInitialPopulationGeneratorCandidates(instances = []) {
  return (Array.isArray(instances) ? instances : [])
    .map((item) => objectValue(item))
    .filter((item) => stringValue(item.proposalId) === BINARY_PROPOSAL_ID)
    .map((item) => ({
      instanceId: stringValue(item.instanceId),
      proposalId: stringValue(item.proposalId),
      displayName: stringValue(item.displayName || item.instanceId),
    }))
    .filter((item) => item.instanceId);
}

export function comparatorSameInitialPopulationForBmopsoPayload(value, instances = []) {
  const config = objectValue(value);
  const enabled = Boolean(config.enabled);
  const candidates = comparatorSameInitialPopulationGeneratorCandidates(instances);
  if (!enabled || !candidates.length) {
    return {
      enabled: false,
      generatorInstanceId: null,
      scope: SAME_INITIAL_POPULATION_SCOPE,
    };
  }
  const requestedGeneratorId = stringValue(config.generatorInstanceId);
  const selected = candidates.find((candidate) => candidate.instanceId === requestedGeneratorId) || candidates[0];
  return {
    enabled: true,
    generatorInstanceId: selected.instanceId,
    scope: SAME_INITIAL_POPULATION_SCOPE,
  };
}

export function comparatorHistoricalInstancesFromRun(run) {
  const config = objectValue(run?.config);
  const configuredInstances = Array.isArray(config.proposalInstances) ? config.proposalInstances : [];
  const source = configuredInstances.length
    ? configuredInstances
    : (Array.isArray(run?.proposals) ? run.proposals : []);
  return source
    .map((item, index) => normalizeHistoricalInstance(item, index))
    .filter(Boolean);
}
