const CREDENTIAL_REF_PATTERN = /^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/;
const ZHIYAN_AGENT_IMAGE = "mirrors.tencent.com/zhiyan-agent/agent-integration:0.0.32";
const ZHIYAN_AGENT_IMAGE_PATTERN = /^mirrors\.tencent\.com\/zhiyan-agent\/agent-integration(?::[A-Za-z0-9][A-Za-z0-9._-]{0,127}|@sha256:[0-9a-f]{64})$/;
const ZHIYAN_LOG_NAME_PATTERN = /^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/;


export function emptyProviderConfigDraft(providerType = "kubernetes") {
  return {
    provider_config_ref: "",
    provider_type: providerType,
    namespace: "",
    status: "disabled",
    auth_type: "in_cluster",
    api_url: "",
    credential_ref: "",
    credential_configured: null,
    credential_token: "",
    credential_ca_certificate: "",
    tls_strict: true,
    manager_resource_request_ratio: "1",
    base_url: "",
    api_account: "",
    cluster_name: "",
    project_name: "",
    zhiyan_log_enabled: false,
    zhiyan_log_name: "",
    zhiyan_agent_image: ZHIYAN_AGENT_IMAGE,
    zhiyan_log_volume_mb: "4096",
  };
}

export function providerConfigToDraft(record) {
  const config = record?.config || {};
  const providerType = record?.provider_type || config.provider_type || "kubernetes";
  return {
    ...emptyProviderConfigDraft(providerType),
    provider_config_ref: record?.provider_config_ref || "",
    status: record?.status || "disabled",
    namespace: config.namespace || "",
    auth_type: config.auth_type || "in_cluster",
    api_url: config.api_url || "",
    credential_ref: config.credential_ref || "",
    credential_configured: record?.credential_configured ?? null,
    credential_token: "",
    credential_ca_certificate: "",
    tls_strict: config.tls_strict !== false,
    manager_resource_request_ratio: String(config.manager_resource_request_ratio ?? 1),
    base_url: config.base_url || "",
    api_account: config.api_account || "",
    cluster_name: config.cluster_name || "",
    project_name: config.project_name || "",
    zhiyan_log_enabled: Boolean(config.zhiyan_log),
    zhiyan_log_name: config.zhiyan_log?.log_name || "",
    zhiyan_agent_image: config.zhiyan_log?.agent_image || ZHIYAN_AGENT_IMAGE,
    zhiyan_log_volume_mb: String((config.zhiyan_log?.volume_size_bytes ?? 4 * 1024 ** 3) / 1024 ** 2),
  };
}

export function providerConfigToPayload(draft) {
  const providerConfigRef = requiredText(draft?.provider_config_ref, "provider_config_ref");
  const providerType = draft?.provider_type;
  const namespace = requiredText(draft?.namespace, "namespace");
  const status = draft?.status === "enabled" ? "enabled" : "disabled";
  let config;
  let credential;

  if (providerType === "kubernetes") {
    const authType = draft?.auth_type === "secret" ? "secret" : "in_cluster";
    let apiUrl = "";
    let credentialRef = null;
    if (authType === "secret") {
      apiUrl = requiredText(draft?.api_url, "api_url");
      credentialRef = credentialRefValue(draft?.credential_ref);
      const token = optionalText(draft?.credential_token);
      const caCertificate = optionalText(draft?.credential_ca_certificate);
      if (token || caCertificate) {
        credential = {
          provider_type: "kubernetes",
          token: requiredText(token, "credential_token"),
          ca_certificate: requiredText(caCertificate, "credential_ca_certificate"),
        };
      }
    }
    const ratio = Number(String(draft?.manager_resource_request_ratio ?? "1").trim());
    if (!Number.isFinite(ratio) || ratio <= 0 || ratio > 1) {
      throw new Error("manager_resource_request_ratio must be greater than 0 and at most 1");
    }
    config = {
      provider_type: "kubernetes",
      namespace,
      auth_type: authType,
      api_url: apiUrl,
      credential_ref: credentialRef,
      tls_strict: draft?.tls_strict !== false,
      manager_resource_request_ratio: ratio,
    };
  } else if (providerType === "tkex") {
    config = {
      provider_type: "tkex",
      namespace,
      base_url: requiredText(draft?.base_url, "base_url"),
      api_account: requiredText(draft?.api_account, "api_account"),
      credential_ref: credentialRefValue(draft?.credential_ref),
      cluster_name: requiredText(draft?.cluster_name, "cluster_name"),
      project_name: requiredText(draft?.project_name, "project_name"),
    };
    if (draft?.zhiyan_log_enabled) {
      const logName = requiredText(draft?.zhiyan_log_name, "zhiyan_log_name");
      if (!ZHIYAN_LOG_NAME_PATTERN.test(logName)) {
        throw new Error("zhiyan_log_name must contain only letters, numbers, dots, underscores, or hyphens");
      }
      const agentImage = requiredText(draft?.zhiyan_agent_image, "zhiyan_agent_image");
      if (!ZHIYAN_AGENT_IMAGE_PATTERN.test(agentImage)) {
        throw new Error("zhiyan_agent_image must use the official Zhiyan agent repository");
      }
      const volumeMb = Number(String(draft?.zhiyan_log_volume_mb ?? "").trim());
      const volumeSizeBytes = volumeMb * 1024 ** 2;
      if (!Number.isSafeInteger(volumeMb) || volumeMb < 1 || !Number.isSafeInteger(volumeSizeBytes)) {
        throw new Error("zhiyan_log_volume_mb must be a positive safe integer");
      }
      config.zhiyan_log = {
        log_name: logName,
        agent_image: agentImage,
        volume_size_bytes: volumeSizeBytes,
      };
    }
    const token = optionalText(draft?.credential_token);
    if (token) credential = { provider_type: "tkex", token };
  } else {
    throw new Error("provider_type must be kubernetes or tkex");
  }

  const payload = { provider_config_ref: providerConfigRef, status, config };
  if (credential) payload.credential = credential;
  return payload;
}

function requiredText(value, field) {
  const normalized = String(value ?? "").trim();
  if (!normalized) throw new Error(`${field} is required`);
  return normalized;
}

function optionalText(value) {
  return String(value ?? "").trim();
}

function credentialRefValue(value) {
  const normalized = requiredText(value, "credential_ref");
  if (!CREDENTIAL_REF_PATTERN.test(normalized)) {
    throw new Error("credential_ref must be a logical name, not a path");
  }
  return normalized;
}
