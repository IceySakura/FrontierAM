{{- define "nexus-distributed.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "nexus-distributed.otelEnv" -}}
{{- $root := index . 0 -}}
{{- $serviceName := index . 1 -}}
{{- $processRole := index . 2 -}}
{{- if $root.Values.otel.enabled }}
- name: NEXUS_OTEL_ENABLED
  value: "true"
- name: NEXUS_OTEL_PROCESS_ROLE
  value: {{ $processRole | quote }}
- name: OTEL_SERVICE_NAME
  value: {{ $serviceName | quote }}
- name: OTEL_TRACES_EXPORTER
  value: {{ $root.Values.otel.tracesExporter | quote }}
- name: OTEL_EXPORTER_OTLP_ENDPOINT
  value: {{ required "otel.endpoint is required when otel.enabled=true" $root.Values.otel.endpoint | quote }}
- name: OTEL_EXPORTER_OTLP_PROTOCOL
  value: {{ $root.Values.otel.protocol | quote }}
- name: OTEL_TRACES_SAMPLER
  value: {{ $root.Values.otel.tracesSampler | quote }}
- name: OTEL_PROPAGATORS
  value: tracecontext
- name: OTEL_METRICS_EXPORTER
  value: none
- name: OTEL_LOGS_EXPORTER
  value: none
{{- if $root.Values.otel.existingSecret }}
- name: OTEL_RESOURCE_ATTRIBUTES
  valueFrom:
    secretKeyRef:
      name: {{ $root.Values.otel.existingSecret }}
      key: {{ $root.Values.otel.resourceAttributesKey }}
- name: OTEL_EXPORTER_OTLP_HEADERS
  valueFrom:
    secretKeyRef:
      name: {{ $root.Values.otel.existingSecret }}
      key: {{ $root.Values.otel.exporterHeadersKey }}
{{- end }}
{{- end }}
{{- end -}}

{{- define "nexus-distributed.fullname" -}}
{{- if .Values.fullnameOverride -}}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- .Release.Name | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}

{{- define "nexus-distributed.componentName" -}}
{{- $root := index . 0 -}}
{{- $component := index . 1 -}}
{{- printf "%s-%s" (include "nexus-distributed.fullname" $root) $component | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "nexus-distributed.secretName" -}}
{{- if .Values.secrets.existingSecret -}}
{{- .Values.secrets.existingSecret -}}
{{- else -}}
{{- include "nexus-distributed.componentName" (list . "secrets") -}}
{{- end -}}
{{- end -}}

{{- define "nexus-distributed.mysqlSecretName" -}}
{{- if .Values.mysql.auth.existingSecret -}}
{{- .Values.mysql.auth.existingSecret -}}
{{- else -}}
{{- include "nexus-distributed.componentName" (list . "mysql-auth") -}}
{{- end -}}
{{- end -}}

{{- define "nexus-distributed.controlImage" -}}
{{- if .Values.controlImage.digest -}}
{{- printf "%s@%s" .Values.controlImage.repository .Values.controlImage.digest -}}
{{- else -}}
{{- printf "%s:%s" .Values.controlImage.repository (.Values.controlImage.tag | default .Chart.AppVersion) -}}
{{- end -}}
{{- end -}}

{{- define "nexus-distributed.workloadImage" -}}
{{- if .Values.workloadImage.digest -}}
{{- printf "%s@%s" .Values.workloadImage.repository .Values.workloadImage.digest -}}
{{- else -}}
{{- printf "%s:%s" .Values.workloadImage.repository (.Values.workloadImage.tag | default .Chart.AppVersion) -}}
{{- end -}}
{{- end -}}

{{- define "nexus-distributed.builderImage" -}}
{{- if .Values.builderImage.digest -}}
{{- printf "%s@%s" .Values.builderImage.repository .Values.builderImage.digest -}}
{{- else -}}
{{- printf "%s:%s" .Values.builderImage.repository (.Values.builderImage.tag | default .Chart.AppVersion) -}}
{{- end -}}
{{- end -}}

{{- define "nexus-distributed.labels" -}}
helm.sh/chart: {{ .Chart.Name }}-{{ .Chart.Version | replace "+" "_" }}
app.kubernetes.io/name: {{ include "nexus-distributed.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end -}}

{{- define "nexus-distributed.selectorLabels" -}}
app.kubernetes.io/name: {{ include "nexus-distributed.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}

{{- define "nexus-distributed.databaseUrl" -}}
{{- if .Values.database.url -}}
{{- .Values.database.url -}}
{{- else -}}
{{- printf "mysql+asyncmy://root:%s@%s:%v/%s?charset=utf8mb4" (required "mysql.auth.rootPassword is required when database.url is empty" .Values.mysql.auth.rootPassword | urlquery) (include "nexus-distributed.componentName" (list . "mysql")) .Values.mysql.service.port .Values.database.name -}}
{{- end -}}
{{- end -}}

{{- define "nexus-distributed.processEnv" -}}
- name: PYTHONUNBUFFERED
  value: "1"
- name: NEXUS_DIST_CONFIG_STORE_REQUIRED
  value: {{ ternary "1" "0" .Values.configStore.required | quote }}
{{- end -}}

{{- define "nexus-distributed.databaseEnv" -}}
- name: NEXUS_DIST_DATABASE_URL_FILE
  value: /var/run/secrets/nexus/database/url
{{- end -}}

{{- define "nexus-distributed.databaseSecretVolumeMount" -}}
- name: nexus-secrets
  mountPath: /var/run/secrets/nexus
  readOnly: true
{{- end -}}

{{- define "nexus-distributed.databaseSecretVolume" -}}
- name: nexus-secrets
  projected:
    sources:
      - secret:
          name: {{ include "nexus-distributed.secretName" . }}
          items:
            - key: {{ .Values.secrets.databaseUrlKey }}
              path: database/url
{{- end -}}
