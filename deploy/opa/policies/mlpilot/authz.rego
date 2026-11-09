# Access rules for the MLPilot Hub, evaluated after the Hub's own role check.
# They can only take access away (deny-overrides): the Hub has already verified the role.
#
# input: {"user": {"id", "email", "subject", "is_admin", "is_service"}, "workspace", "role",
#         "required_role", "request": {"method", "path"}}
package mlpilot.authz

cfg := data.mlpilot.config

default allow := false

allow if count(reasons) == 0

# Service accounts (CI pipelines, schedulers) may train, register and score, but never approve.
reasons contains "service accounts cannot approve promotions or administer workspaces" if {
	input.user.is_service
	input.required_role in {"approver", "admin"}
}

# Change freeze: a frozen workspace is read-only for everyone but platform admins.
reasons contains msg if {
	input.workspace in cfg.frozen_workspaces
	input.request.method != "GET"
	not input.user.is_admin
	msg := sprintf("workspace %q is in a change freeze (read-only)", [input.workspace])
}

# Restricted workspaces: only identities from the listed e-mail domains.
reasons contains msg if {
	domains := cfg.restricted_workspaces[input.workspace]
	not email_in(domains)
	not input.user.is_admin
	msg := sprintf("workspace %q is restricted to %v identities", [input.workspace, domains])
}

email_in(domains) if {
	some d in domains
	endswith(lower(input.user.email), concat("", ["@", lower(d)]))
}
