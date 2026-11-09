# opa test deploy/opa/policies -v
package automatedcleaning_test

import data.automatedcleaning.authz
import data.automatedcleaning.promotion

base_cfg := {
	"require_qa": true, "require_signed_production": false, "min_metrics": {"roc_auc": 0.7},
	"min_disparate_impact": 0.8, "frozen_workspaces": ["finance"],
	"restricted_workspaces": {"risk": ["bank.example"]},
}

person := {"id": "u1", "email": "ana@bank.example", "is_admin": false, "is_service": false}

robot := {"id": "u2", "email": "ci@bank.example", "is_admin": false, "is_service": true}

# ── authz ──────────────────────────────────────────────────────────────────
test_normal_request_allowed if {
	authz.allow with input as {
		"user": person, "workspace": "growth", "role": "approver",
		"required_role": "approver", "request": {"method": "POST", "path": "/api/promotions/1/approve"},
	}
		with data.automatedcleaning.config as base_cfg
}

test_service_account_cannot_approve if {
	not authz.allow with input as {
		"user": robot, "workspace": "growth", "role": "approver",
		"required_role": "approver", "request": {"method": "POST", "path": "/x"},
	}
		with data.automatedcleaning.config as base_cfg
	authz.reasons["service accounts cannot approve promotions or administer workspaces"] with input as {
		"user": robot, "workspace": "growth", "role": "approver", "required_role": "approver",
		"request": {"method": "POST", "path": "/x"},
	}
		with data.automatedcleaning.config as base_cfg
}

test_service_account_can_register if {
	authz.allow with input as {
		"user": robot, "workspace": "growth", "role": "data_scientist",
		"required_role": "data_scientist", "request": {"method": "POST", "path": "/x"},
	}
		with data.automatedcleaning.config as base_cfg
}

test_change_freeze_blocks_writes_not_reads if {
	not authz.allow with input as {
		"user": person, "workspace": "finance", "role": "admin",
		"required_role": "data_scientist", "request": {"method": "POST", "path": "/x"},
	}
		with data.automatedcleaning.config as base_cfg
	authz.allow with input as {
		"user": person, "workspace": "finance", "role": "viewer",
		"required_role": "viewer", "request": {"method": "GET", "path": "/x"},
	}
		with data.automatedcleaning.config as base_cfg
}

test_restricted_workspace_by_email_domain if {
	authz.allow with input as {
		"user": person, "workspace": "risk", "role": "viewer",
		"required_role": "viewer", "request": {"method": "GET", "path": "/x"},
	}
		with data.automatedcleaning.config as base_cfg
	not authz.allow with input as {
		"user": object.union(person, {"email": "eve@gmail.com"}), "workspace": "risk",
		"role": "viewer", "required_role": "viewer", "request": {"method": "GET", "path": "/x"},
	}
		with data.automatedcleaning.config as base_cfg
}

# ── promotion ──────────────────────────────────────────────────────────────
good_model := {
	"name": "churn", "version": 3, "stage": "staging", "metrics": {"roc_auc": 0.84, "disparate_impact": 0.93},
	"signed": true, "description": "Weekly churn propensity", "stage_history": ["dev", "qa", "staging"],
}

test_good_model_can_go_to_production if {
	count(promotion.deny) == 0 with input as {"model": good_model, "to_stage": "production", "user": person}
		with data.automatedcleaning.config as base_cfg
}

test_qa_cannot_be_skipped if {
	msgs := promotion.deny with input as {
		"model": object.union(good_model, {"stage_history": ["dev"]}),
		"to_stage": "staging", "user": person,
	}
		with data.automatedcleaning.config as base_cfg
	msgs == {"the model must pass QA (be moved to 'qa') before staging or production"}
}

test_metric_floor_description_fairness_and_signing if {
	bad := object.union(good_model, {
		"metrics": {"roc_auc": 0.6, "disparate_impact": 0.7}, "description": " ",
		"signed": false,
	})
	msgs := promotion.deny with input as {"model": bad, "to_stage": "production", "user": person}
		with data.automatedcleaning.config as object.union(base_cfg, {"require_signed_production": true})
	msgs == {
		"roc_auc must be at least 0.7 for production (got 0.6)",
		"a model description (model card summary) is required for production",
		"disparate impact 0.70 is below 0.80 (four-fifths rule)",
		"production models must be signed (automatedcleaning.security.sign_model)",
	}
}

test_missing_metric_fails_the_floor if {
	msgs := promotion.deny with input as {
		"model": object.union(object.remove(good_model, ["metrics"]), {"metrics": {}}), "to_stage": "production",
		"user": person,
	}
		with data.automatedcleaning.config as base_cfg
	msgs == {"roc_auc was not recorded, and production needs it"}
}

test_service_account_only_up_to_qa if {
	count(promotion.deny) == 0 with input as {
		"model": object.union(good_model, {"stage_history": ["dev"]}),
		"to_stage": "qa", "user": robot,
	}
		with data.automatedcleaning.config as base_cfg
	promotion.deny["service accounts can promote at most to 'qa'"] with input as {
		"model": good_model,
		"to_stage": "production", "user": robot,
	}
		with data.automatedcleaning.config as base_cfg
}
