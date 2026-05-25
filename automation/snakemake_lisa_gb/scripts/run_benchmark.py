#!/usr/bin/env python3
import argparse, ast, json, os, re, subprocess, time
from pathlib import Path


def _f(p, t=float):
    m = re.findall(p, t, re.MULTILINE)
    return (t and m and t and t) and t


def find_num(pattern, text):
    m = re.findall(pattern, text, re.MULTILINE)
    return float(m[-1]) if m else None


def find_list(pattern, text):
    m = re.findall(pattern, text, re.MULTILINE)
    if not m:
        return None
    try:
        return ast.literal_eval(m[-1])
    except Exception:
        return None


p = argparse.ArgumentParser()
p.add_argument("--python", required=True)
p.add_argument("--repo-root", required=True)
p.add_argument("--problem-factory", required=True)
p.add_argument("--config-json", required=True)
p.add_argument("--algo", required=True)
p.add_argument("--seed", type=int, required=True)
p.add_argument("--n-live", type=int, required=True)
p.add_argument("--tol", type=float, required=True)
p.add_argument("--num-inner-steps", type=int, required=True)
p.add_argument("--ggns-step-size", type=float, required=True)
p.add_argument("--ggns-num-inner-steps", type=int, required=True)
p.add_argument("--initial-num-steps", type=int, required=True)
p.add_argument("--refinement-num-steps", type=int, required=True)
p.add_argument("--max-batches", type=int, required=True)
p.add_argument("--ess-min", type=float, default=0)
p.add_argument("--bad-logz-min", type=float, default=-1e99)
p.add_argument("--local-mode-f0-sigma-max", type=float, default=1e-7)
p.add_argument("--skip-plots", action="store_true")
p.add_argument("--out-json", required=True)
a = p.parse_args()

cmd = [
    a.python, "-m", "jax_samplers.cli", "--algo", a.algo, "--problem", "factory",
    "--problem-factory", a.problem_factory, "--n-live", str(a.n_live), "--tol", str(a.tol),
    "--num-inner-steps", str(a.num_inner_steps), "--seed", str(a.seed),
    "--initial-num-steps", str(a.initial_num_steps), "--refinement-num-steps", str(a.refinement_num_steps),
    "--max-batches", str(a.max_batches), "--ggns-step-size", str(a.ggns_step_size),
    "--ggns-num-inner-steps", str(a.ggns_num_inner_steps),
]
if a.skip_plots:
    cmd.append("--skip-plots")

env = os.environ.copy()
env["JAX_SAMPLERS_CONFIG"] = str(Path(a.config_json).resolve())
env["PYTHONPATH"] = f"{a.repo_root}/src:" + env.get("PYTHONPATH", "")

t0 = time.time()
cp = subprocess.run(cmd, capture_output=True, text=True, env=env)
runtime = time.time() - t0
text = (cp.stdout or "") + "\n" + (cp.stderr or "")

diags = find_list(r"^\[post\] diagnostics\s*=\s*(\{.*\})$", text)

logz = find_num(r"logZ\s*=\s*([-+0-9.eE]+)", text)
logz_std = find_num(r"logZ\s*=\s*[-+0-9.eE]+±([-+0-9.eE]+)", text)
ess = find_num(r"ESS\s*=\s*([-+0-9.eE]+)", text)
best_logl = find_num(r"best logL\s*=\s*([-+0-9.eE]+)", text)
if diags and isinstance(diags, dict):
    logz = diags.get("logZ", logz)
    logz_std = diags.get("logZ_std", logz_std)
    ess = diags.get("ESS", ess)
    best_logl = diags.get("best_logL", best_logl)

f0_mean = find_num(r"f0_mean['\"]?\s*[:=]\s*([-+0-9.eE]+)", text)
f0_std = find_num(r"f0_std['\"]?\s*[:=]\s*([-+0-9.eE]+)", text)
samples_shape = find_list(r"^\[post\] samples\.shape\s*=\s*(\(.*\))$", text)
weights_shape = find_list(r"^\[post\] weights\.shape\s*=\s*(\(.*\))$", text)
pk_vals = find_list(r"pK_vals['\"]?\s*[:=]\s*(\[[^\]]*\])", text)
pk_probs = find_list(r"pK_probs['\"]?\s*[:=]\s*(\[[^\]]*\])", text)
if diags and isinstance(diags, dict):
    pk_vals = diags.get("pK_vals", pk_vals)
    pk_probs = diags.get("pK_probs", pk_probs)

status = "pass" if cp.returncode == 0 else "crash"
labels = [status]
if cp.returncode == 0:
    if ess is not None and ess < a.ess_min:
        labels.append("low_ESS")
    if logz is None or logz <= a.bad_logz_min:
        labels.append("bad_logZ")
    if f0_std is not None and f0_std > a.local_mode_f0_sigma_max:
        labels.append("local_mode_suspected")

if a.algo in {"ggns", "dynamic_ggns"} and status == "crash":
    labels.append("experimental")

out = {
    "algo": a.algo, "seed": a.seed, "n_live": a.n_live, "tol": a.tol,
    "num_inner_steps": a.num_inner_steps, "runtime_seconds": runtime,
    "return_code": cp.returncode, "status": status, "status_labels": labels,
    "logZ": logz, "logZ_std": logz_std, "ESS": ess, "best_logL": best_logl,
    "pK_vals": pk_vals, "pK_probs": pk_probs, "f0_mean": f0_mean, "f0_std": f0_std,
    "samples_shape": samples_shape, "weights_shape": weights_shape,
    "ggns_step_size": a.ggns_step_size, "ggns_num_inner_steps": a.ggns_num_inner_steps,
    "initial_num_steps": a.initial_num_steps, "refinement_num_steps": a.refinement_num_steps,
    "max_batches": a.max_batches,
}

out_path = Path(a.out_json)
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
