{{/*
Expand the name of the chart.
*/}}
{{- define "memguard.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "memguard.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart label value.
*/}}
{{- define "memguard.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "memguard.labels" -}}
helm.sh/chart: {{ include "memguard.chart" . }}
{{ include "memguard.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "memguard.selectorLabels" -}}
app.kubernetes.io/name: {{ include "memguard.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Service account name
*/}}
{{- define "memguard.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "memguard.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Target namespace for the existing-deployment patch job.
Defaults to .Release.Namespace when patchExistingDeployment.targetNamespace is empty.
*/}}
{{- define "memguard.patchTargetNamespace" -}}
{{- if .Values.patchExistingDeployment.targetNamespace }}
{{- .Values.patchExistingDeployment.targetNamespace }}
{{- else }}
{{- .Release.Namespace }}
{{- end }}
{{- end }}
