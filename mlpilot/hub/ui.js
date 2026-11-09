"use strict";
// MLPilot Hub UI. Everything rendered is escaped; the token lives in sessionStorage only.
const $ = (s) => document.querySelector(s);
const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c]));
const fmt = (v) => typeof v === "number" ? (Math.abs(v) < 1e4 ? +v.toFixed(4) : v.toExponential(3)) : esc(v);
let token = sessionStorage.getItem("ach_token") || "";
let me = null, ws = null, tab = "models";

function toast(text, err) {
  const m = $("#msg"); m.textContent = text; m.className = err ? "err" : ""; m.style.display = "block";
  clearTimeout(toast.t); toast.t = setTimeout(() => (m.style.display = "none"), 5000);
}

async function api(method, path, body) {
  const res = await fetch(path, {method, headers: {"Authorization": "Bearer " + token,
    ...(body ? {"Content-Type": "application/json"} : {})}, body: body ? JSON.stringify(body) : undefined});
  if (res.status === 401) { signOut(); throw new Error("Session expired"); }
  const text = await res.text();
  const data = text ? JSON.parse(text) : null;
  if (!res.ok) {
    const d = data && data.detail;
    throw new Error(typeof d === "string" ? d : (d && d.message ? d.message + ": " + (d.failures || []).join("; ") : res.statusText));
  }
  return data;
}

function signOut() { sessionStorage.removeItem("ach_token"); token = ""; $("#app").hidden = true; $("#login").hidden = false; }

async function start() {
  if (!token) { $("#login").hidden = false; return; }
  try { me = await api("GET", "/api/me"); } catch (e) { return; }
  $("#login").hidden = true; $("#app").hidden = false;
  $("#who").textContent = (me.email || me.name || me.subject) + (me.is_admin ? " · platform admin" : "");
  const names = Object.keys(me.workspaces).sort();
  $("#ws").innerHTML = names.map((n) => `<option>${esc(n)}</option>`).join("") || "<option disabled>No workspaces</option>";
  ws = names[0] || null;
  render();
}

function table(cols, rows, empty) {
  if (!rows.length) return `<p class="muted">${esc(empty || "Nothing yet.")}</p>`;
  return `<div class="tbl"><table><thead><tr>${cols.map((c) => `<th>${esc(c[0])}</th>`).join("")}</tr></thead><tbody>` +
    rows.map((r) => `<tr>${cols.map((c) => `<td>${c[1](r)}</td>`).join("")}</tr>`).join("") + "</tbody></table></div>";
}
const chip = (s) => `<span class="chip ${esc(s)}">${esc(s)}</span>`;
const badge = (cls, text) => `<span class="chip ${esc(cls)}">${esc(text)}</span>`;
const metrics = (m) => Object.entries(m || {}).slice(0, 4).map(([k, v]) => `${esc(k)} <b>${fmt(v)}</b>`).join("<br>");

const views = {
  async models() {
    const models = await api("GET", `/api/workspaces/${ws}/models`);
    let html = `<div class="card"><h2>Models</h2>` + table([["Model", (m) => `<a href="#" data-model="${esc(m.name)}">${esc(m.name)}</a>`],
      ["Task", (m) => esc(m.task)], ["Versions", (m) => m.versions], ["Production", (m) => m.production ? "v" + m.production : "–"],
      ["Staging", (m) => m.staging ? "v" + m.staging : "–"]], models, "No models registered yet. Use HubClient.register(...).") + "</div>";
    html += `<div id="versions"></div>`;
    $("#view").innerHTML = html;
    document.querySelectorAll("[data-model]").forEach((a) => a.onclick = (e) => { e.preventDefault(); showVersions(a.dataset.model); });
    if (models.length) showVersions(models[0].name);
  },
  async approvals() {
    const reqs = await api("GET", `/api/workspaces/${ws}/promotions?status=pending`);
    $("#view").innerHTML = `<div class="card"><h2>Pending production approvals</h2><p class="muted">Four-eyes rule: an approver other than the requester must decide.</p>` +
      table([["Model", (r) => `${esc(r.model)} v${r.version}`], ["Reason", (r) => esc(r.reason)], ["Evidence", (r) => metrics(r.evidence.metrics)],
        ["Requested", (r) => esc(r.created_at.slice(0, 16).replace("T", " "))],
        ["", (r) => `<div class="row"><button data-approve="${esc(r.id)}">Approve</button><button class="danger" data-reject="${esc(r.id)}">Reject</button></div>`]],
        reqs, "No pending requests.") + "</div>";
    document.querySelectorAll("[data-approve],[data-reject]").forEach((b) => b.onclick = async () => {
      const ok = !!b.dataset.approve, id = b.dataset.approve || b.dataset.reject;
      const note = prompt(ok ? "Approval note (optional)" : "Why is it rejected?") ;
      if (note === null) return;
      try { await api("POST", `/api/promotions/${id}/${ok ? "approve" : "reject"}`, {note}); toast(ok ? "Approved and deployed to production" : "Rejected"); render(); }
      catch (e) { toast(e.message, true); }
    });
  },
  async runs() {
    const runs = await api("GET", `/api/workspaces/${ws}/runs?limit=200`);
    $("#view").innerHTML = `<div class="card"><h2>Runs</h2>` + table([["Experiment", (r) => esc(r.experiment)], ["Run", (r) => esc(r.name)],
      ["Status", (r) => badge(r.status === "failed" ? "bad" : r.status === "finished" ? "production" : "staging", r.status)], ["Metrics", (r) => metrics(r.metrics)],
      ["Started", (r) => esc(r.started_at.slice(0, 16).replace("T", " "))]], runs, "No runs yet.") + "</div>";
  },
  async monitoring() {
    const models = await api("GET", `/api/workspaces/${ws}/models`);
    let html = "";
    for (const m of models) {
      const rows = await api("GET", `/api/workspaces/${ws}/models/${encodeURIComponent(m.name)}/monitoring?limit=20`);
      html += `<div class="card"><h2>${esc(m.name)}</h2>` + table([["Checked", (r) => esc(r.checked_at.slice(0, 16).replace("T", " "))],
        ["Rows", (r) => r.rows], ["Drift share", (r) => r.drift_share == null ? "–" : (r.drift_share * 100).toFixed(0) + "%"],
        ["Metric", (r) => r.metric ? `${esc(r.metric)} ${fmt(r.metric_value)}` : "–"],
        ["Status", (r) => r.retrain_recommended ? badge("bad", "retrain") : (r.alerts.length ? badge("pending", "warnings") : badge("production", "healthy"))]],
        rows, "No monitoring results reported.") + "</div>";
    }
    $("#view").innerHTML = html || `<div class="card muted">No models.</div>`;
  },
  async members() {
    const rows = await api("GET", `/api/workspaces/${ws}/members`);
    const admin = me.workspaces[ws] === "admin";
    $("#view").innerHTML = `<div class="card"><h2>Members</h2>` + table([["User", (r) => esc(r.user)], ["Role", (r) => chip(r.role)]], rows) +
      (admin ? `<div class="row" style="margin-top:12px"><input id="mUser" placeholder="email or user id"><select id="mRole">
      <option>viewer</option><option selected>data_scientist</option><option>approver</option><option>admin</option></select>
      <button id="mAdd">Set role</button></div>` : "") + "</div>";
    if (admin) $("#mAdd").onclick = async () => {
      try { await api("PUT", `/api/workspaces/${ws}/members`, {user: $("#mUser").value.trim(), role: $("#mRole").value}); toast("Saved"); render(); }
      catch (e) { toast(e.message, true); }
    };
  },
  async audit() {
    let rows, verify = null;
    try { rows = await api("GET", `/api/audit?workspace_name=${encodeURIComponent(ws)}&limit=300`); }
    catch (e) { $("#view").innerHTML = `<div class="card muted">${esc(e.message)}</div>`; return; }
    if (me.is_admin) verify = await api("GET", "/api/audit/verify");
    $("#view").innerHTML = `<div class="card"><h2>Audit log</h2>` +
      (verify ? `<p>${verify.valid ? badge("production", "chain intact") : badge("bad", "TAMPERED at #" + verify.broken_at)} <span class="muted">${verify.entries} entries verified</span></p>` : "") +
      table([["#", (r) => r.id], ["When", (r) => esc(r.ts.slice(0, 19).replace("T", " "))], ["Who", (r) => esc(r.actor)], ["Action", (r) => `<code>${esc(r.action)}</code>`],
        ["Target", (r) => esc(r.target)], ["Details", (r) => esc(JSON.stringify(r.details)).slice(0, 160)]], rows) + "</div>";
  },
};

async function showVersions(name) {
  const rows = await api("GET", `/api/workspaces/${ws}/models/${encodeURIComponent(name)}/versions`);
  const role = me.workspaces[ws];
  $("#versions").innerHTML = `<div class="card"><h2>${esc(name)} — versions</h2>` + table([["Version", (v) => "v" + v.version],
    ["Stage", (v) => chip(v.stage)], ["Metrics", (v) => metrics(v.metrics)], ["Signed", (v) => v.signed ? "✓" : "–"],
    ["Created", (v) => esc(v.created_at.slice(0, 16).replace("T", " "))],
    ["", (v) => role === "viewer" ? "" : `<div class="row">` +
      (v.stage !== "staging" && v.stage !== "production" ? `<button class="ghost" data-stage="staging" data-v="${v.version}">→ staging</button>` : "") +
      (v.stage !== "production" ? `<button data-stage="production" data-v="${v.version}">→ production</button>` : "") + `</div>`]],
    rows.slice().reverse()) + "</div>";
  document.querySelectorAll("[data-stage]").forEach((b) => b.onclick = async () => {
    const reason = prompt(`Reason for moving v${b.dataset.v} to ${b.dataset.stage}`);
    if (reason === null) return;
    try {
      const out = await api("POST", `/api/workspaces/${ws}/models/${encodeURIComponent(name)}/versions/${b.dataset.v}/transitions`, {stage: b.dataset.stage, reason});
      toast(out.status === "pending_approval" ? "Sent for approval" : "Done"); showVersions(name);
    } catch (e) { toast(e.message, true); }
  });
}

async function render() {
  document.querySelectorAll("#tabs button").forEach((b) => b.classList.toggle("on", b.dataset.tab === tab));
  if (!ws) { $("#view").innerHTML = `<div class="card muted">You're not a member of any workspace yet.</div>`; return; }
  $("#view").innerHTML = `<div class="card muted">Loading…</div>`;
  try { await views[tab](); } catch (e) { $("#view").innerHTML = `<div class="card">${esc(e.message)}</div>`; }
}

$("#signin").onclick = () => { token = $("#token").value.trim(); sessionStorage.setItem("ach_token", token); start(); };
$("#token").addEventListener("keydown", (e) => { if (e.key === "Enter") $("#signin").click(); });
$("#signout").onclick = signOut;
$("#ws").onchange = (e) => { ws = e.target.value; render(); };
document.querySelectorAll("#tabs button").forEach((b) => b.onclick = () => { tab = b.dataset.tab; render(); });
start();
