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
