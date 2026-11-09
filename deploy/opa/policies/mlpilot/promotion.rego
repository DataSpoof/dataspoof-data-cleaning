# Promotion rules for the MLPilot Hub, checked on every stage change in addition to
# the workspace's own policy (min/max metrics, require_signed) and four-eyes approval.
#
# input: {"workspace": {"name", "policy", "require_approval"},
#         "model": {"name", "version", "stage", "metrics", "signed", "task", "description", "stage_history"},
#         "to_stage", "user": {"id", "email", "is_service"}}
package mlpilot.promotion

cfg := data.mlpilot.config

late_stages := {"staging", "production"}

# dev -> qa -> staging -> production: nothing skips QA.
deny contains "the model must pass QA (be moved to 'qa') before staging or production" if {
	cfg.require_qa
	input.to_stage in late_stages
	not "qa" in input.model.stage_history
}

# Production needs a written description (the model card summary).
deny contains "a model description (model card summary) is required for production" if {
	input.to_stage == "production"
	not has_description
}

has_description if {
	is_string(input.model.description)
	trim_space(input.model.description) != ""
}

# Organisation-wide metric floors for production (workspace policies can be stricter).
deny contains msg if {
	input.to_stage == "production"
	some metric, minimum in cfg.min_metrics
	value := input.model.metrics[metric]
	value < minimum
	msg := sprintf("%s must be at least %v for production (got %v)", [metric, minimum, value])
}

deny contains msg if {
	input.to_stage == "production"
	some metric, _ in cfg.min_metrics
	not is_number(object.get(input.model.metrics, metric, null))
	msg := sprintf("%s was not recorded, and production needs it", [metric])
}

# Fairness: the four-fifths rule, when the model was evaluated for it.
deny contains msg if {
	input.to_stage == "production"
	di := input.model.metrics.disparate_impact
	di < cfg.min_disparate_impact
	msg := sprintf("disparate impact %.2f is below %.2f (four-fifths rule)", [di, cfg.min_disparate_impact])
}

# Signed models only in production, when the organisation requires it.
deny contains "production models must be signed (mlpilot.security.sign_model)" if {
	cfg.require_signed_production
	input.to_stage == "production"
	not input.model.signed
}

# Service accounts can move models up to QA, not further.
deny contains "service accounts can promote at most to 'qa'" if {
	input.user.is_service
	input.to_stage in late_stages
}
