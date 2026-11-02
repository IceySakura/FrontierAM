import assert from "node:assert/strict";
import test from "node:test";

import {
  emptyProviderConfigDraft,
  providerConfigToDraft,
  providerConfigToPayload,
} from "./providerConfig.js";


test("builds in-cluster Kubernetes provider config without credential paths", () => {
  const draft = {
    ...emptyProviderConfigDraft("kubernetes"),
    provider_config_ref: "kubernetes/devcloud",
    namespace: "nexus-online-service",
    status: "enabled",
  };
  assert.deepEqual(providerConfigToPayload(draft), {
    provider_config_ref: "kubernetes/devcloud",
    status: "enabled",
    config: {
      provider_type: "kubernetes",
      namespace: "nexus-online-service",
      auth_type: "in_cluster",
      api_url: "",
      credential_ref: null,
      tls_strict: true,
      manager_resource_request_ratio: 1,
    },
  });
});

test("builds secret-backed Kubernetes and TKEx provider configs", () => {
  assert.deepEqual(providerConfigToPayload({
    ...emptyProviderConfigDraft("kubernetes"),
    provider_config_ref: "kubernetes/remote",
    namespace: "workload",
    auth_type: "secret",
    api_url: "https://kubernetes.example",
    credential_ref: "kube-remote",
    manager_resource_request_ratio: "0.5",
  }).config, {
    provider_type: "kubernetes",
    namespace: "workload",
    auth_type: "secret",
    api_url: "https://kubernetes.example",
    credential_ref: "kube-remote",
    tls_strict: true,
    manager_resource_request_ratio: 0.5,
  });

  assert.deepEqual(providerConfigToPayload({
    ...emptyProviderConfigDraft("tkex"),
    provider_config_ref: "tkex/shanghai-load",
    namespace: "production",
    base_url: "https://tkex.example",
    api_account: "nexus",
    credential_ref: "tkex-shanghai-load",
    cluster_name: "cluster-shanghai",
    project_name: "nexus",
    status: "enabled",
  }).config, {
    provider_type: "tkex",
    namespace: "production",
    base_url: "https://tkex.example",
    api_account: "nexus",
    credential_ref: "tkex-shanghai-load",
    cluster_name: "cluster-shanghai",
    project_name: "nexus",
  });
});

test("round trips persisted records into editable drafts", () => {
  const draft = providerConfigToDraft({
    provider_config_ref: "kubernetes/remote",
    provider_type: "kubernetes",
    status: "disabled",
    config: {
      provider_type: "kubernetes",
      namespace: "workload",
      auth_type: "secret",
      api_url: "https://kubernetes.example",
      credential_ref: "kube-remote",
      tls_strict: false,
      manager_resource_request_ratio: 0.25,
    },
    credential_configured: true,
  });
  assert.equal(draft.provider_config_ref, "kubernetes/remote");
  assert.equal(draft.auth_type, "secret");
  assert.equal(draft.tls_strict, false);
  assert.equal(draft.manager_resource_request_ratio, "0.25");
  assert.equal(draft.credential_configured, true);
  assert.equal(draft.credential_token, "");
  assert.equal(draft.credential_ca_certificate, "");
});

test("round trips TKEx Zhiyan log settings", () => {
  const payload = providerConfigToPayload({
    ...emptyProviderConfigDraft("tkex"),
    provider_config_ref: "tkex/shanghai-load",
    namespace: "production",
    base_url: "https://tkex.example",
    api_account: "nexus",
    credential_ref: "tkex-shanghai-load",
    cluster_name: "cluster-shanghai",
    project_name: "nexus",
    zhiyan_log_enabled: true,
    zhiyan_log_name: "dev_nexus_dynamic_manager_log",
    zhiyan_agent_image: "mirrors.tencent.com/zhiyan-agent/agent-integration:0.0.12",
    zhiyan_log_volume_mb: "4096",
  });

  assert.deepEqual(payload.config.zhiyan_log, {
    log_name: "dev_nexus_dynamic_manager_log",
    agent_image: "mirrors.tencent.com/zhiyan-agent/agent-integration:0.0.12",
    volume_size_bytes: 4 * 1024 ** 3,
  });
  const draft = providerConfigToDraft({
    provider_config_ref: "tkex/shanghai-load",
    provider_type: "tkex",
    status: "enabled",
    config: payload.config,
  });
  assert.equal(draft.zhiyan_log_enabled, true);
  assert.equal(draft.zhiyan_log_name, "dev_nexus_dynamic_manager_log");
  assert.equal(draft.zhiyan_agent_image, "mirrors.tencent.com/zhiyan-agent/agent-integration:0.0.12");
  assert.equal(draft.zhiyan_log_volume_mb, "4096");
});

test("uses the Zhiyan agent version validated for TKEx dynamic managers", () => {
  const draft = emptyProviderConfigDraft("tkex");

  assert.equal(draft.zhiyan_agent_image, "mirrors.tencent.com/zhiyan-agent/agent-integration:0.0.32");
});

test("omits disabled TKEx Zhiyan log settings", () => {
  const payload = providerConfigToPayload({
    ...emptyProviderConfigDraft("tkex"),
    provider_config_ref: "tkex/shanghai-load",
    namespace: "production",
    base_url: "https://tkex.example",
    api_account: "nexus",
    credential_ref: "tkex-shanghai-load",
    cluster_name: "cluster-shanghai",
    project_name: "nexus",
    zhiyan_log_enabled: false,
    zhiyan_log_name: "stale_name",
  });

  assert.equal(Object.hasOwn(payload.config, "zhiyan_log"), false);
});

test("rejects invalid TKEx Zhiyan log settings", () => {
  const draft = {
    ...emptyProviderConfigDraft("tkex"),
    provider_config_ref: "tkex/shanghai-load",
    namespace: "production",
    base_url: "https://tkex.example",
    api_account: "nexus",
    credential_ref: "tkex-shanghai-load",
    cluster_name: "cluster-shanghai",
    project_name: "nexus",
    zhiyan_log_enabled: true,
    zhiyan_log_name: "dev_nexus_dynamic_manager_log",
  };

  assert.throws(
    () => providerConfigToPayload({ ...draft, zhiyan_log_name: "invalid log name" }),
    /zhiyan_log_name/,
  );
  assert.throws(
    () => providerConfigToPayload({ ...draft, zhiyan_agent_image: "registry.example/agent:latest" }),
    /official Zhiyan agent repository/,
  );
  for (const volume of ["", "0", "1.5", "9007199254740992"]) {
    assert.throws(
      () => providerConfigToPayload({ ...draft, zhiyan_log_volume_mb: volume }),
      /zhiyan_log_volume_mb/,
    );
  }
});

test("adds typed provider credentials only when an administrator enters new values", () => {
  const tkex = providerConfigToPayload({
    ...emptyProviderConfigDraft("tkex"),
    provider_config_ref: "tkex/shanghai-load",
    namespace: "production",
    base_url: "https://tkex.example",
    api_account: "nexus",
    credential_ref: "tkex-shanghai-load",
    credential_token: "tkex-token",
    cluster_name: "cluster-shanghai",
    project_name: "nexus",
  });
  assert.deepEqual(tkex.credential, {
    provider_type: "tkex",
    token: "tkex-token",
  });

  const kubernetes = providerConfigToPayload({
    ...emptyProviderConfigDraft("kubernetes"),
    provider_config_ref: "kubernetes/remote",
    namespace: "workload",
    auth_type: "secret",
    api_url: "https://kubernetes.example",
    credential_ref: "kube-remote",
    credential_token: "kube-token",
    credential_ca_certificate: "kube-ca",
  });
  assert.deepEqual(kubernetes.credential, {
    provider_type: "kubernetes",
    token: "kube-token",
    ca_certificate: "kube-ca",
  });

  const unchanged = providerConfigToPayload({
    ...providerConfigToDraft({
      provider_config_ref: "tkex/shanghai-load",
      provider_type: "tkex",
      status: "enabled",
      credential_configured: true,
      config: tkex.config,
    }),
  });
  assert.equal(Object.hasOwn(unchanged, "credential"), false);
});

test("requires both Kubernetes token and CA when rotating a credential", () => {
  assert.throws(
    () => providerConfigToPayload({
      ...emptyProviderConfigDraft("kubernetes"),
      provider_config_ref: "kubernetes/remote",
      namespace: "workload",
      auth_type: "secret",
      api_url: "https://kubernetes.example",
      credential_ref: "kube-remote",
      credential_token: "kube-token",
    }),
    /credential_ca_certificate/,
  );
});

test("rejects incomplete configs and arbitrary credential paths", () => {
  assert.throws(
    () => providerConfigToPayload({
      ...emptyProviderConfigDraft("kubernetes"),
      provider_config_ref: "kubernetes/remote",
      namespace: "workload",
      auth_type: "secret",
      api_url: "https://kubernetes.example",
      credential_ref: "/var/run/secrets/token",
    }),
    /credential_ref/,
  );
  assert.throws(
    () => providerConfigToPayload({
      ...emptyProviderConfigDraft("tkex"),
      provider_config_ref: "tkex/missing",
      namespace: "workload",
    }),
    /base_url/,
  );
});
