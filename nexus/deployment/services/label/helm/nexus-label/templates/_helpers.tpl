{{- define "nexus-label.name" -}}
{{- .Chart.Name | trunc 63 | trimSuffix "-" }}
{{- end }}

{{- /*
资源名固定为 chart 名，不带 release 名前缀：不同环境已由 namespace 隔离，
再叠加 release 名只会让 kubectl 命令随环境变化（nexus-label-dev /
nexus-label-integration），排查时容易记错。
*/ -}}
{{- define "nexus-label.fullname" -}}
{{- .Chart.Name | trunc 63 | trimSuffix "-" }}
{{- end }}

{{- define "nexus-label.labels" -}}
helm.sh/chart: {{ printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" }}
{{ include "nexus-label.selectorLabels" . }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{- define "nexus-label.selectorLabels" -}}
app.kubernetes.io/name: {{ include "nexus-label.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}
