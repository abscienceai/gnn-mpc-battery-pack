#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import hashlib
import json
import py_compile
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
errors = []
passes = []

def ok(msg): passes.append(msg)
def fail(msg): errors.append(msg)

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def require(rel):
    p = ROOT / rel
    if not p.exists():
        fail(f"missing required file: {rel}")
        return None
    return p

def git_lines(*args):
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True, stderr=subprocess.STDOUT
    ).splitlines()

# Branch
try:
    branch = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=ROOT, text=True
    ).strip()
    if branch == "publication-v2":
        ok("publication-v2 branch confirmed")
    else:
        fail(f"unexpected branch: {branch!r}")
except Exception as exc:
    fail(f"unable to determine branch: {exc}")

required = [
    "README.md","REPRODUCIBILITY.md","RESULTS_MANIFEST.md","CITATION.cff",
    "LICENSE","requirements.txt","data/README.md",
    "src/graph_battery_pack.py","src/ecm_fitting.py","src/train_gnn.py",
    "src/safe_fast_charge_optimizer.py","src/submission_revision_controllers.py",
    "src/run_canonical_lfp_n30.py","src/analyze_canonical_lfp_n30.py",
    "src/run_thermal_initial_condition_control.py",
    "src/run_hust_timestamp_replay_v2.py",
    "src/analyze_hust_timestamp_replay_v2.py",
    "src/run_controller_timing_validation.py",
    "src/smoke_action_gnn_v2.py","src/smoke_integrated_cccv.py",
    "src/smoke_matched_physics_cem.py","src/smoke_true_cccv.py",
    "src/generate_submission_figures_v2.py","src/refine_submission_figures_v2.py",
    "results/ecm/ecm_params_20260630_202031.parquet",
    "results/models/pack_gnn_action_v2_20261003_232518.pt",
    "results/canonical_lfp_n30/canonical_lfp_n30_raw.jsonl",
    "results/canonical_lfp_n30/canonical_lfp_n30_summary.csv",
    "results/canonical_lfp_n30/canonical_lfp_n30_summary.json",
    "results/canonical_lfp_n30/statistics/descriptive_statistics.csv",
    "results/canonical_lfp_n30/statistics/paired_statistics.csv",
    "results/canonical_lfp_n30/statistics/safety_statistics.csv",
    "results/diagnostics/thermal_initial_condition_n30/thermal_initial_condition_raw.jsonl",
    "results/diagnostics/thermal_initial_condition_n30/thermal_initial_condition_summary.json",
    "results/diagnostics/thermal_initial_condition_n30/thermal_initial_condition_statistics.csv",
    "results/diagnostics/hust_timestamp_replay_v2_n30/hust_timestamp_replay_v2_raw.jsonl",
    "results/diagnostics/hust_timestamp_replay_v2_n30/hust_timestamp_replay_v2_summary.json",
    "results/diagnostics/hust_timestamp_replay_v2_n30/hust_timestamp_replay_v2_statistics.csv",
    "results/diagnostics/hust_timestamp_replay_v2_n30/hust_timestamp_replay_v2_safety.csv",
    "results/diagnostics/hust_timestamp_replay_v2_n30/hust_timestamp_replay_v2_provenance.json",
    "results/diagnostics/controller_timing_validation/controller_timing_raw.csv",
    "results/diagnostics/controller_timing_validation/controller_timing_summary.json",
    "provenance/submission_evidence_freeze_20261004.json",
    "provenance/pack_gnn_action_v2_canonical.json",
    "provenance/pack_gnn_action_v2_heldout_action_test.json",
    "provenance/pack_gnn_action_v2_ranking_fidelity.json",
    "provenance/thermal_initial_condition_n30.json",
    "provenance/controller_timing_validation.json",
    "figures/submission_v2/fig1_controller_architecture.png",
    "figures/submission_v2/fig2_canonical_paired_tradeoff.pdf",
    "figures/submission_v2/fig2_canonical_paired_tradeoff.png",
    "figures/submission_v2/fig3_thermal_initial_condition.pdf",
    "figures/submission_v2/fig3_thermal_initial_condition.png",
    "figures/submission_v2/fig4_hust_paired_tradeoff.pdf",
    "figures/submission_v2/fig4_hust_paired_tradeoff.png",
    "figures/submission_v2/fig5_surrogate_ranking_fidelity.pdf",
    "figures/submission_v2/fig5_surrogate_ranking_fidelity.png",
    "figures/submission_v2/figS1_action_conditioning.pdf",
    "figures/submission_v2/figS1_action_conditioning.png",
    "figures/submission_v2/figure_manifest.json",
    "figures/submission_v2/latex_figure_snippets.tex",
]
for rel in required:
    require(rel)
if not any(x.startswith("missing required file:") for x in errors):
    ok("all required final publication files exist")

expected_hashes = {
    "results/ecm/ecm_params_20260630_202031.parquet":
        "5580ff78d2518c645588d5bdd58ace96507e14bb2622dd6e26cf080397d94988",
    "results/models/pack_gnn_action_v2_20261003_232518.pt":
        "2f3acc96e77504be0a060f5bee9bb2263b62df08c8fcd2afca353005c02e0b5f",
    "figures/submission_v2/fig1_controller_architecture.png":
        "2c3ec64095a1ef34cc855aa4fd1a9ebb624a19955bdd7b628561035d795586aa",
}
for rel, expected in expected_hashes.items():
    p = ROOT / rel
    if p.exists():
        got = sha256(p)
        if got == expected:
            ok(f"hash verified: {rel}")
        else:
            fail(f"hash mismatch: {rel} expected {expected} got {got}")

fig1_pdf = ROOT / "figures/submission_v2/fig1_controller_architecture.pdf"
if fig1_pdf.exists():
    fail("obsolete Figure 1 PDF still exists")
else:
    ok("Figure 1 is PNG-only")

snippet = ROOT / "figures/submission_v2/latex_figure_snippets.tex"
if snippet.exists():
    t = snippet.read_text(encoding="utf-8")
    if "fig1_controller_architecture.png" not in t:
        fail("LaTeX snippets do not reference final Figure 1 PNG")
    if "fig1_controller_architecture.pdf" in t:
        fail("LaTeX snippets still reference obsolete Figure 1 PDF")

manifest_path = ROOT / "figures/submission_v2/figure_manifest.json"
if manifest_path.exists():
    try:
        m = json.loads(manifest_path.read_text(encoding="utf-8"))
        out = m.get("outputs", {})
        k = "figures/submission_v2/fig1_controller_architecture.png"
        if out.get(k) == expected_hashes[k]:
            ok("Figure 1 manifest hash verified")
        else:
            fail("figure manifest contains wrong Figure 1 PNG hash")
        if "figures/submission_v2/fig1_controller_architecture.pdf" in out:
            fail("figure manifest still contains obsolete Figure 1 PDF")
    except Exception as exc:
        fail(f"unable to parse figure manifest: {exc}")

server_hits = []
for p in sorted((ROOT / "src").glob("*.py")):
    txt = p.read_text(encoding="utf-8")
    if "/home/msoylu/" in txt:
        server_hits.append(str(p.relative_to(ROOT)))
if server_hits:
    fail("server-specific paths remain: " + ", ".join(server_hits))
else:
    ok("source tree contains no /home/msoylu paths")

refine = ROOT / "src/refine_submission_figures_v2.py"
if refine.exists() and "OUT_DIR" in refine.read_text(encoding="utf-8"):
    fail("refine script contains stray OUT_DIR")
else:
    ok("refine Figure 1 path is portable")

syntax_errors = []
for p in sorted((ROOT / "src").glob("*.py")):
    try:
        py_compile.compile(str(p), doraise=True)
    except Exception as exc:
        syntax_errors.append(f"{p.relative_to(ROOT)}: {exc}")
if syntax_errors:
    fail("syntax errors: " + " | ".join(syntax_errors))
else:
    ok("all final Python source files compile")

try:
    tracked = git_lines("ls-files")
except Exception as exc:
    tracked = []
    fail(f"unable to list tracked files: {exc}")

forbidden = [
    "battery_revision_2026","canonical_lfp_n30_pre_matched_physics",
    "figure_contact_sheet","canonical_NMC","canonical_LCO","cross_chem",
    "hardware_validation_obsfix","simplempc_matched_n30","pack_mlp_fair",
    "node_only_gnn_","fig_cross_chemistry","fig_online_adaptation",
    "fig_failure_cases","fig_scaling",
]
bad = sorted({p for p in tracked if any(f in p for f in forbidden)})
if bad:
    fail("obsolete/wrong-project files still tracked: " + ", ".join(bad))
else:
    ok("obsolete/wrong-project evidence is not tracked")

canonical = ROOT / "results/canonical_lfp_n30/canonical_lfp_n30_summary.json"
if canonical.exists():
    try:
        c = json.loads(canonical.read_text(encoding="utf-8"))["summary"]
        checks = [
            ("GraphOptimizer","charging_time_min","mean",21.6333333333,0.02),
            ("Physics-CEM","charging_time_min","mean",25.0333333333,0.02),
            ("GraphOptimizer","final_SOC_sigma_pct","mean",0.366281,0.01),
            ("Physics-CEM","final_SOC_sigma_pct","mean",0.414353,0.01),
            ("GraphOptimizer","final_T_gradient_C","mean",0.155344,0.01),
            ("Physics-CEM","final_T_gradient_C","mean",0.115964,0.01),
        ]
        local_fail = False
        for ctrl, metric, leaf, target, tol in checks:
            got = float(c[ctrl][metric][leaf])
            if abs(got-target) > tol:
                fail(f"canonical mismatch {ctrl} {metric}: {got}")
                local_fail = True
        if not local_fail:
            ok("canonical N=30 headline metrics match frozen results")
    except Exception as exc:
        fail(f"unable to validate canonical summary: {exc}")

hust = ROOT / "src/run_hust_timestamp_replay_v2.py"
if hust.exists():
    t = hust.read_text(encoding="utf-8").lower()
    if "not claimed as external plant validation" in t and "physical interpolation every 60 s" in t:
        ok("HUST protocol and claim boundary are present")
    else:
        fail("HUST protocol/claim boundary text missing")

timing = ROOT / "src/run_controller_timing_validation.py"
if timing.exists():
    t = timing.read_text(encoding="utf-8").lower()
    if "not hardware-in-the-loop validation" in t:
        ok("software timing non-HIL boundary is present")
    else:
        fail("timing non-HIL boundary text missing")

readme = ROOT / "README.md"
if readme.exists():
    t = readme.read_text(encoding="utf-8").lower()
    required_phrases = [
        "matr-informed","external measured-protocol replay",
        "not external plant validation","not hil",
        "one-step model-based safety","no voltage-prediction head",
    ]
    missing = [p for p in required_phrases if p not in t]
    if missing:
        fail("README missing claim-boundary text: " + ", ".join(missing))
    else:
        ok("README final claim boundaries present")

    legacy = [
        "table 10","table 11",
        "cross-chemistry safety is chemistry-dependent","34.6",
    ]
    hits = [p for p in legacy if p in t]
    if hits:
        fail("README obsolete headline content remains: " + ", ".join(hits))
    else:
        ok("README obsolete cross-chemistry headlines removed")

data_readme = ROOT / "data/README.md"
if data_readme.exists():
    t = data_readme.read_text(encoding="utf-8").lower()
    if "final manuscript uses matr and hust only" in t:
        ok("data README final MATR + HUST scope confirmed")
    else:
        fail("data README does not state final MATR + HUST scope")

freeze = ROOT / "provenance/submission_evidence_freeze_20261004.json"
if freeze.exists():
    try:
        f = json.loads(freeze.read_text(encoding="utf-8"))
        expected = "a28143ea3bb60081214ae9aa91e9051b1c202e72"
        if f.get("git_head_final") == expected:
            ok("historical frozen computational-evidence commit preserved")
        else:
            fail("historical evidence manifest git_head_final changed")
    except Exception as exc:
        fail(f"unable to parse historical evidence manifest: {exc}")

if "finalize_publication_docs.sh" in tracked:
    fail("temporary finalize_publication_docs.sh helper is tracked")
else:
    ok("temporary documentation helper is not tracked")

print("="*78)
print("FINAL PUBLICATION REPOSITORY AUDIT")
print("="*78)
for item in passes:
    print("PASS:", item)
if errors:
    print()
    for item in errors:
        print("FAIL:", item)
    print()
    print(f"AUDIT RESULT: FAIL ({len(errors)} issue(s))")
    sys.exit(1)
print()
print("AUDIT RESULT: PASS")
