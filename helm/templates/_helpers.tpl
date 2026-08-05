{{- define "rain-predictor.name" -}}
{{ .Chart.Name }}
{{- end }}

{{- define "rain-predictor.fullname" -}}
{{ .Release.Name }}-{{ .Chart.Name }}
{{- end }}