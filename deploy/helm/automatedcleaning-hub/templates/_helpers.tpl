{{- define "hub.fullname" -}}{{ .Release.Name | trunc 63 | trimSuffix "-" }}{{- end -}}
{{- define "hub.labels" -}}
app.kubernetes.io/name: automatedcleaning-hub
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end -}}
{{- define "hub.selector" -}}
app.kubernetes.io/name: automatedcleaning-hub
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}
{{- define "hub.serviceAccount" -}}{{ default (include "hub.fullname" .) .Values.serviceAccount.name }}{{- end -}}
{{- define "hub.dbSecret" -}}{{ default (printf "%s-db" (include "hub.fullname" .)) .Values.database.existingSecret }}{{- end -}}
