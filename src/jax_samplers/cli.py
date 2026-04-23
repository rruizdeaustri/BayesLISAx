# cli.py
from __future__ import annotations

import os, sys, json, inspect, argparse
import numpy as np


# ------------------------------------------------------------
# Bootstrap parse: grab --config / --dtype BEFORE importing JAX
# ------------------------------------------------------------
_boot = argparse.ArgumentParser(add_help=False)
_boot.add_argument("--config", type=str, default="")
_boot.add_argument("--dtype", type=str, default="")
_boot_args, _ = _boot.parse_known_args()

if _boot_args.config:
    os.environ["JAX_SAMPLERS_CONFIG"] = _boot_args.config
elif os.environ.get("LISA_CFG") and not os.environ.get("JAX_SAMPLERS_CONFIG"):
    os.environ["JAX_SAMPLERS_CONFIG"] = os.environ["LISA_CFG"]

if _boot_args.dtype:
    os.environ["JAX_SAMPLERS_DTYPE"] = _boot_args.dtype

# ── Precision must be set BEFORE importing jax or anything that imports jax ──
from .core.precision import REQUESTED_DTYPE, DTYPE, CDTYPE, EPS, to_real, to_cplx, jnp_real, jnp_cplx

import jax
import jax.numpy as jnp
import jax.random as jr

# Safe non-interactive matplotlib on headless nodes
import matplotlib
if not os.environ.get("DISPLAY") and not os.environ.get("MPLBACKEND"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import corner

# Problems / registry
from .problems.gaussmix import GaussianMixtureMeans
from .core.generic_problem import GenericProblem
from .core.problem import Problem
from .adapters.transdim_product_space import TransdimProductSpaceProblem
from .registry import get_sampler

from .core.rj import RJGenericConfig
from .core.utils import (
    _labels,
    _resolve_schema,
    _parse_aliases,
    _parse_truth,
    infer_K_and_extras,
    labels_lisa_safe,
    sanitize_labels_for_matplotlib
)

# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def _load_obj(entrypoint: str):
    mod, _, name = entrypoint.partition(":")
    if not mod or not name:
        raise ValueError(f"Invalid entry point '{entrypoint}'. Use 'pkg.module:Name'.")
    return getattr(__import__(mod, fromlist=[name]), name)

def _load_json_maybe(path: str) -> dict:
    if not path:
        return {}
    try:
        if os.path.exists(path):
            with open(path, "r") as fh:
                return json.load(fh)
    except Exception:
        pass
    return {}

def _get_cfg_path(args) -> str:
    # args.config is already parsed by main parser; env is set by bootstrap
    if getattr(args, "config", ""):
        return args.config
    return os.environ.get("JAX_SAMPLERS_CONFIG", "")


def _extract_prior_bounds(cfg_json: dict, problem) -> tuple:
    """
    Try to extract lower/upper bound lists from various sources, in order of preference:
      1. cfg_json["priors"]["uniform"] — per-parameter dicts, repeated Kmax times if model.Kmax > 1
      2. cfg_json["prior_box"] — simpler single-source format
      3. problem.lower / problem.upper attributes
    Returns (lower_or_None, upper_or_None) as Python lists of floats.
    """
    lower = None
    upper = None

    # 1. Try priors.uniform
    priors_uniform = cfg_json.get("priors", {}).get("uniform", None)
    if isinstance(priors_uniform, dict) and priors_uniform:
        Kmax = int(cfg_json.get("model", {}).get("Kmax", 1))
        lo_list = []
        hi_list = []
        for _name, bounds in priors_uniform.items():
            if isinstance(bounds, (list, tuple)) and len(bounds) == 2:
                lo_list.append(float(bounds[0]))
                hi_list.append(float(bounds[1]))
        if lo_list:
            # Repeat Kmax times if multi-source
            lower = lo_list * Kmax
            upper = hi_list * Kmax
            return lower, upper

    # 2. Try prior_box
    prior_box = cfg_json.get("prior_box", None)
    if isinstance(prior_box, dict) and prior_box:
        lo_list = []
        hi_list = []
        for _name, bounds in prior_box.items():
            if isinstance(bounds, (list, tuple)) and len(bounds) == 2:
                lo_list.append(float(bounds[0]))
                hi_list.append(float(bounds[1]))
        if lo_list:
            lower = lo_list
            upper = hi_list
            return lower, upper

    # 3. Try problem attributes
    lower = getattr(problem, "lower", None)
    upper = getattr(problem, "upper", None)
    if lower is not None:
        lower = [float(x) for x in lower]
    if upper is not None:
        upper = [float(x) for x in upper]

    return lower, upper

def _build_parser():
    p = argparse.ArgumentParser(description="jax_samplers CLI")

    # Debug / JAX
    p.add_argument("--disable-jit", action="store_true", help="Disable JAX JIT (debug only)")

    # Core
    p.add_argument("--algo", type=str, default="ns")
    p.add_argument("--config", type=str, default="", help="Path to JSON config file.")
    p.add_argument("--dtype", type=str, default="", help="Override dtype: float32 or float64.")

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n", type=int, default=400)

    # Problem selection
    p.add_argument("--problem", choices=["gmm2", "fixed", "transdim", "factory"], default="gmm2")
    p.add_argument("--problem-factory", type=str, default="",
                   help="Entry point 'module:make' that returns a Problem instance (use with --problem factory).")

    # ---------- Plot / schema ----------
    p.add_argument("--schema", choices=["generic", "fermi"], default="generic")
    p.add_argument("--fields", type=str, default="f0, fdot, iota, psi, lam, beta")
    p.add_argument(
        "--sample-repr",
        choices=["auto", "gated_raw", "fixed"],
        default="auto",
        help="How to interpret posterior samples before plotting."
    )
    p.add_argument(
        "--truth",
        type=str,
        default="",
        help="Optional whitespace-separated truth values (must match plotted dimension)."
    )
    p.add_argument("--start-col", type=int, default=0)
    p.add_argument("--sort-by", type=str, default="")
    p.add_argument("--layout", choices=["component-major", "field-major"], default="component-major")
    p.add_argument("--extras", type=int, default=None, help="If set, fix number of extra (global) params at end.")

    p.add_argument("--label-style", choices=["plain", "latex", "template"], default="latex")
    p.add_argument("--label-template", type=str, default="{name}_{k}")
    p.add_argument(
        "--field-aliases",
        type=str,
        default=r"f0:f_0,fdot:\dot{f},iota:\iota,psi:\psi,lam:\lambda,beta:\beta,p:p",
        help="Comma-separated key:value aliases, e.g. fdot:\\dot{f},lam:\\lambda",
    )
    p.add_argument("--save", type=str, default="")
    p.add_argument("--no-show", action="store_true")
    p.add_argument("--skip-plots", action="store_true",
                   help="Run inference and exit before generating any plots.")
    p.add_argument("--plain-labels", action="store_true")
    p.add_argument("--no-titles", action="store_true")
    p.add_argument("--no-contours", action="store_true")

    # ---------- FIXED problem hooks ----------
    p.add_argument("--dim", type=int, default=None)
    p.add_argument("--logprior", type=str, default="")
    p.add_argument("--loglikelihood", type=str, default="")
    p.add_argument("--sample-prior", type=str, default="")

    # ---------- PRODUCT-SPACE transdim ----------
    p.add_argument("--family", type=str, default="")
    p.add_argument("--family-kwargs", type=str, default="")

    # ---------- GATED transdim plotting ----------
    # In gates mode, K is derived from p_k gates inside a fixed-dimensional sample vector.
    p.add_argument("--gates", action="store_true",
                   help="Interpret fixed-dim samples as gated transdim: infer K_eff from p_k gates.")
    p.add_argument("--Kmax", type=int, default=None,
                   help="Kmax used for gates (required for --gates unless readable from config).")
    p.add_argument("--dim-per", type=int, default=7,
                   help="Per-component block size. For LISA GB with p gate, usually 7 (f0,fdot,iota,psi,lam,beta,p).")
    p.add_argument("--p-index", type=int, default=-1,
                   help="Index of gate p inside each per-component block. Default -1 (last).")
    p.add_argument("--p-active-min", type=float, default=None,
                   help="Gate threshold for activity. If omitted, tries config priors.p_active_min, else 1e-3.")

    # ---------- NS (BlackJAX) ----------
    p.add_argument("--n-live", type=int, default=500)
    p.add_argument("--num-delete-ratio", type=float, default=0.3)
    p.add_argument("--num-inner-steps", type=int, default=0, help="Slice steps per live point; 0 -> auto (3*dim)")
    p.add_argument("--tol", type=float, default=3.0)
    p.add_argument("--max-batch", type=int, default=0, help="(optional) chunk size for batched likelihood")
    # Hamiltonian NS specific
    p.add_argument("--ham-dt-ini", type=float, default=0.3, help="Initial step size for Hamiltonian NS")
    p.add_argument("--ham-min-reflections", type=int, default=2, help="Min reflections for Hamiltonian NS")
    p.add_argument("--ham-max-reflections", type=int, default=10, help="Max reflections for Hamiltonian NS")
    p.add_argument("--ham-sigma-vel", type=float, default=0.0, help="Velocity sigma for Hamiltonian NS")
    p.add_argument("--ham-max-steps", type=int, default=150, help="Max steps for Hamiltonian NS")
    p.add_argument("--ham-lower", type=str, default="", help="Comma-separated lower bounds for Hamiltonian NS, e.g. '0.0,-1e-13,...'")
    p.add_argument("--ham-upper", type=str, default="", help="Comma-separated upper bounds for Hamiltonian NS")

    # ---------- JAXNS ----------
    p.add_argument("--s", type=int, default=10)
    p.add_argument("--max-samples", type=float, default=None)

    # IMPORTANT: argparse bools should be flags
    p.add_argument("--grad-guided", action="store_true")
    p.add_argument("--no-parameter-estimation", action="store_true")
    p.add_argument("--difficult-model", action="store_true")
    p.add_argument("--box-low", type=float, default=-10.0)
    p.add_argument("--box-high", type=float, default=10.0)

    # ---------- SMC (NUTS shared) ----------
    p.add_argument("--n-particles", type=int, default=1000)
    p.add_argument("--nuts-step-size", type=float, default=1e-3)
    p.add_argument("--nuts-inv-mass-diag", type=float, default=1.0)
    p.add_argument("--target-ess", type=float, default=0.5)
    p.add_argument("--num-mcmc-steps", type=int, default=5)
    p.add_argument("--max-tree-depth", type=int, default=None)

    # ---------- SMC-HMC ----------
    p.add_argument("--hmc-step-size", type=float, default=1e-3)
    p.add_argument("--hmc-n-leapfrog", type=int, default=10)
    p.add_argument("--hmc-inv-mass-diag", type=float, default=1.0)
    p.add_argument("--hmc-target-ess", type=float, default=0.5)
    p.add_argument("--hmc-num-mcmc-steps", type=int, default=5)
    p.add_argument("--hmc-resampler", choices=["systematic", "multinomial", "stratified"], default="systematic")
    p.add_argument("--hmc-verbose", action="store_true", help="Print SMC-HMC diagnostics.")

    # ---------- NumPyro NUTS ----------
    p.add_argument("--numpyro-warmup", type=int, default=1000)
    p.add_argument("--numpyro-samples", type=int, default=2000)
    p.add_argument("--numpyro-chains", type=int, default=1)
    p.add_argument("--numpyro-target-accept", type=float, default=0.8)
    p.add_argument("--numpyro-dense-mass", action="store_true")
    p.add_argument("--numpyro-max-tree-depth", type=int, default=10)
    p.add_argument("--numpyro-no-progress", action="store_true")
    p.add_argument("--verbose", action="store_true")

    # ---------- RJ ----------
    p.add_argument("--rj-steps", type=int, default=20000)
    p.add_argument("--rj-sigma-rw", type=float, default=0.2)
    p.add_argument("--rj-sigma-split", type=float, default=0.6)
    p.add_argument("--rj-p-birth", type=float, default=0.3)

    # ---------- RJ generic ----------
    p.add_argument("--rj-p-up", type=float, default=0.3)
    p.add_argument("--rjgen-sigma-rw", type=float, default=0.2)
    p.add_argument("--proposal", type=str, default="")
    p.add_argument("--proposal-kwargs", type=str, default="")

    return p


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    args = _build_parser().parse_args()

    if args.disable_jit:
        jax.config.update("jax_disable_jit", True)

    cfg_path = _get_cfg_path(args)
    cfg_json = _load_json_maybe(cfg_path)

    # ---------- Build Problem ----------
    is_product_space_transdim = False  # (K stored in column 0)
    family = None

    if args.problem == "gmm2":
        problem = GaussianMixtureMeans(n=args.n, key=jr.PRNGKey(args.seed))

    elif args.problem == "fixed":
        need = ["dim", "logprior", "loglikelihood"]
        if args.algo != "jaxns":
            need.append("sample_prior")
        missing = [n for n in need if not getattr(args, n.replace("-", "_"))]
        if missing:
            raise SystemExit("--problem fixed requires: " + ", ".join(f"--{m}" for m in missing))
        problem = GenericProblem.from_entrypoints(
            dim=args.dim,
            logprior_ep=args.logprior,
            loglikelihood_ep=args.loglikelihood,
            sample_prior_ep=(args.sample_prior or None),
        )

    elif args.problem == "factory":
        if not args.problem_factory:
            raise SystemExit("--problem factory requires --problem-factory 'module:make'")
        Maker = _load_obj(args.problem_factory)
        try:
            sig = inspect.signature(Maker)
            problem = Maker(args) if any(p.name == "args" for p in sig.parameters.values()) else Maker()
        except (ValueError, TypeError):
            try:
                problem = Maker(args)
            except TypeError:
                problem = Maker()

    else:  # product-space transdim (RJ / product-space NS wrappers etc.)
        if not args.family:
            raise SystemExit("For --problem transdim you must pass --family 'module:Class'.")
        Family = _load_obj(args.family)
        fam_kwargs = json.loads(args.family_kwargs) if args.family_kwargs else {}
        family = Family(**fam_kwargs)
        problem = TransdimProductSpaceProblem(family)
        is_product_space_transdim = True

    # Auto inner steps for BlackJAX-NS
    if args.algo in ("ns", "dynamic_nss", "ns_hamiltonian") and (args.num_inner_steps is None or args.num_inner_steps <= 0):
        args.num_inner_steps = 3 * problem.dim

    # ---------- Build sampler config ----------
    if args.algo == "ns":
        from .samplers.blackjax_ns import NSConfig
        cfg = NSConfig(
            n_live=args.n_live,
            num_delete_ratio=args.num_delete_ratio,
            num_inner_steps=args.num_inner_steps,
            tol=args.tol,
        )

    elif args.algo == "dynamic_nss":
        from .samplers.blackjax_ns import NSConfig
        cfg = NSConfig(
            n_live=args.n_live,
            num_delete_ratio=args.num_delete_ratio,
            num_inner_steps=args.num_inner_steps,
            tol=args.tol,
        )

    elif args.algo == "ns_hamiltonian":
        from .samplers.blackjax_ns import NSConfig

        # Try to extract lower/upper from JSON config priors
        lower, upper = _extract_prior_bounds(cfg_json, problem)

        # CLI overrides take precedence
        if args.ham_lower:
            lower = [float(x) for x in args.ham_lower.split(",")]
        if args.ham_upper:
            upper = [float(x) for x in args.ham_upper.split(",")]

        cfg = NSConfig(
            n_live=args.n_live,
            num_delete_ratio=args.num_delete_ratio,
            num_inner_steps=args.num_inner_steps,
            tol=args.tol,
            dt_ini=args.ham_dt_ini,
            min_reflections=args.ham_min_reflections,
            max_reflections=args.ham_max_reflections,
            sigma_vel=args.ham_sigma_vel,
            ham_max_steps=args.ham_max_steps,
            lower=lower,
            upper=upper,
        )

    elif args.algo == "jaxns":
        from .samplers.jaxns_unified import JAXNSConfig
        cfg = JAXNSConfig(
            num_live_points=args.n_live,
            s=args.s,
            max_samples=args.max_samples,
            gradient_guided=args.grad_guided,
            parameter_estimation=(not args.no_parameter_estimation),
            box_low=args.box_low,
            box_high=args.box_high,
        )

    elif args.algo == "smc-nuts":
        from .samplers.smc_nuts import SMCConfig
        cfg = SMCConfig(
            n_particles=args.n_particles,
            step_size=args.nuts_step_size,
            inv_mass_diag=args.nuts_inv_mass_diag,
            target_ess=args.target_ess,
            num_mcmc_steps=args.num_mcmc_steps
        )

    elif args.algo == "smc-hmc":
        from .samplers.smc_hmc import HMCSMCConfig
        cfg = HMCSMCConfig(
            n_particles=args.n_particles,
            step_size=args.hmc_step_size,
            inv_mass_diag=args.hmc_inv_mass_diag,
            n_leapfrog=args.hmc_n_leapfrog,
            target_ess=args.hmc_target_ess,
            num_mcmc_steps=args.hmc_num_mcmc_steps,
            resampler=args.hmc_resampler,
            verbose=args.hmc_verbose
        )

    elif args.algo == "numpyro-nuts":
        try:
            from .samplers.numpyro_nuts import NumPyroNUTSConfig
        except Exception:
            raise SystemExit("NumPyro backend requested but NumPyro isn't available.")
        cfg = NumPyroNUTSConfig(
            num_warmup=args.numpyro_warmup,
            num_samples=args.numpyro_samples,
            num_chains=args.numpyro_chains,
            target_accept_prob=args.numpyro_target_accept,
            dense_mass=args.numpyro_dense_mass,
            max_tree_depth=args.numpyro_max_tree_depth,
            progress_bar=(not args.numpyro_no_progress),
            verbose=args.verbose
        )

    elif args.algo == "rj-generic":
        cfg = RJGenericConfig(
            steps=args.rj_steps,
            p_up=args.rj_p_up,
            sigma_rw=args.rjgen_sigma_rw,
        )

    else:
        raise ValueError(f"Unknown --algo '{args.algo}'")

    # ---------- Construct sampler ----------
    SamplerCls = get_sampler(args.algo)

    if args.algo == "rj-generic":
        if args.problem != "transdim":
            raise SystemExit("rj-generic requires --problem transdim (a ModelFamily).")
        if not args.proposal:
            raise SystemExit("rj-generic requires --proposal 'module:Class'.")
        Proposal = _load_obj(args.proposal)
        prop_kwargs = json.loads(args.proposal_kwargs) if args.proposal_kwargs else {}
        from .samplers.rj_generic import RJGeneric
        sampler = RJGeneric(problem=None, cfg=cfg, family=family, proposal=Proposal(**prop_kwargs)).init(jr.PRNGKey(args.seed))
    else:
        sampler = SamplerCls(problem, cfg).init(jr.PRNGKey(args.seed))

    # ---------- Run ----------
    res = sampler.run(jr.PRNGKey(args.seed + 1))

    # ---------- Extract samples ----------
    samples = np.asarray(res.samples)
    weights = None if getattr(res, "weights", None) is None else np.asarray(res.weights)

    if args.skip_plots:
        print("[post] --skip-plots requested; skipping all plotting.")
        print("[post] samples.shape =", samples.shape)
        if weights is not None:
            print("[post] weights.shape =", weights.shape)
        diagnostics = getattr(res, "diagnostics", None)
        if diagnostics:
            print("[post] diagnostics =", diagnostics)
        return

    # ---------- Plot / postprocess ----------
    fields, start_col, sort_by, transforms = _resolve_schema(args)
    aliases = _parse_aliases(args.field_aliases)
    truth = _parse_truth(args.truth) if args.truth else None

    # -----------------------------------------------------------------
    # A) PRODUCT-SPACE transdim: K is stored in column 0
    # -----------------------------------------------------------------
    if is_product_space_transdim:
        # This matches your existing behavior, but fixed to be robust.
        # (We assume: samples[:,0] is K and the rest are padded params.)
        # Prefer reading K_min/K_max from problem if present.
        K_min = int(getattr(problem, "K_min", 1))
        K_max = int(getattr(problem, "K_max", np.max(samples[:, 0]).astype(int)))

        K_vals = samples[:, 0].astype(int)
        K_vals = np.clip(K_vals, K_min, K_max)

        if weights is None:
            counts = np.bincount(K_vals, minlength=K_max + 1)[K_min:]
            pmf = counts / (counts.sum() + 1e-300)
        else:
            bins = np.arange(K_min, K_max + 2) - 0.5
            w = weights / (weights.sum() + 1e-300)
            pmf, _ = np.histogram(K_vals, bins=bins, weights=w)
            pmf = pmf / (pmf.sum() + 1e-300)

        xticks = np.arange(K_min, K_max + 1)
        print("p(K|y):", {int(k): float(p) for k, p in zip(xticks, pmf)})

        plt.figure()
        plt.bar(xticks, pmf, align="center")
        plt.xticks(xticks)
        plt.xlabel("K")
        plt.ylabel("p(K | y)")
        plt.title(f"{args.algo} – posterior over K (product-space)")
        if args.save:
            plt.savefig(args.save, dpi=150, bbox_inches="tight")
        elif not args.no_show:
            try:
                plt.show()
            except Exception:
                plt.savefig("posterior.png", dpi=150, bbox_inches="tight")
        return

    # -----------------------------------------------------------------
    # B) GATED transdim: K derived from gate p_k inside fixed vector
    # -----------------------------------------------------------------
    if args.gates:
        # Determine Kmax
        Kmax = args.Kmax
        if Kmax is None:
            # try config: {"settings":{"Kmax":...}} or {"Kmax":...}
            Kmax = (
                cfg_json.get("settings", {}).get("Kmax", None)
                or cfg_json.get("Kmax", None)
            )
        if Kmax is None:
            raise SystemExit("--gates requires --Kmax (or config with settings.Kmax).")
        Kmax = int(Kmax)

        per = int(args.dim_per)
        D_expected = Kmax * per
        if samples.shape[1] != D_expected:
            raise RuntimeError(
                f"--gates expects D = Kmax*dim_per = {D_expected}, got D={samples.shape[1]}.\n"
                f"  Kmax={Kmax}, dim_per={per}."
            )

        th = samples.reshape(samples.shape[0], Kmax, per)
        p_idx = args.p_index if args.p_index >= 0 else (per + args.p_index)

        if not (0 <= p_idx < per):
            raise ValueError(f"Invalid --p-index={args.p_index} for dim_per={per} (resolved p_idx={p_idx}).")

        gates = th[:, :, p_idx]

        p_active_min = args.p_active_min
        if p_active_min is None:
            # Try config: {"priors":{"p_active_min":...}} else default
            p_active_min = float(cfg_json.get("priors", {}).get("p_active_min", 1e-3))
        #g_u = gates
        #p = 1/(1+np.exp(-g_u))   # sigmoid
        #K_eff = (p > p_active_min).sum(axis=1).astype(int)

        K_eff = (gates > p_active_min).sum(axis=1).astype(int)

        # Posterior pmf for K_eff in [0..Kmax]
        if weights is None:
            counts = np.bincount(K_eff, minlength=Kmax + 1)
            pmf = counts / (counts.sum() + 1e-300)
        else:
            bins = np.arange(-0.5, Kmax + 1.5, 1.0)
            w = weights / (weights.sum() + 1e-300)
            pmf, _ = np.histogram(K_eff, bins=bins, weights=w)
            pmf = pmf / (pmf.sum() + 1e-300)

        xticks = np.arange(0, Kmax + 1)
        print("p(K_eff|y):", {int(k): float(p) for k, p in zip(xticks, pmf)})

        plt.figure()
        plt.bar(xticks, pmf, align="center")
        plt.xticks(xticks)
        plt.xlabel("K_eff")
        plt.ylabel("p(K_eff | y)")
        plt.title(f"{args.algo} – posterior over K_eff (gates)")
        if args.save:
            plt.savefig(args.save, dpi=150, bbox_inches="tight")
        elif not args.no_show:
            try:
                plt.show()
            except Exception:
                plt.savefig("posterior.png", dpi=150, bbox_inches="tight")

        # --- Corner plot for physical params in gated mode ---
        # indices for the physical params you want (exclude p)
        phys_idx = [0, 1, 2, 3, 4, 5]  # f0, fdot, iota, psi, lam, beta
        X = th[:, :, phys_idx].reshape(th.shape[0], Kmax * len(phys_idx))

        labels = labels_lisa_safe(
            fields_list=["f0", "fdot", "iota", "psi", "lam", "beta"],
            K=Kmax,
            aliases=aliases,
            style=getattr(args, "label_style", "plain"),
            k_start=1,
        )
        
        labels = sanitize_labels_for_matplotlib(labels)

        fig = corner.corner(
            X,
            weights=weights,   # will be None if you used posterior_points() → fine
            labels=labels,
            show_titles=(not args.no_titles),
            plot_contours=(not args.no_contours),
            title_fmt=".3e",
            truths=truth,
        )
        fig.suptitle(f"{args.algo} – gated posterior (Kmax={Kmax})")

        if args.save:
            fig.savefig(args.save, dpi=150, bbox_inches="tight")
        elif not args.no_show:
            plt.show()
                
        return

    # -----------------------------------------------------------------
    # C) FIXED-dim plotting (corner), with generic gated-LISA handling
    # -----------------------------------------------------------------
    fields_list = [t.strip() for t in args.fields.split(",") if t.strip()]
    n_phys = len(fields_list)
    D = samples.shape[1]

    print("[plot debug] samples.shape =", samples.shape)
    print("[plot debug] fields_list =", fields_list, " len =", len(fields_list))
    print("[plot debug] start_col =", args.start_col, " extras_arg =", args.extras)

    # Try to read plotting metadata from config
    plot_cfg = cfg_json.get("plot", {})
    sample_repr = plot_cfg.get("sample_repr", "auto")
    Kmax = plot_cfg.get("Kmax", None)
    dim_per = plot_cfg.get("dim_per", None)
    p_index = plot_cfg.get("p_index", -1)

    # Auto-detect common gated case
    if sample_repr == "auto":
        if Kmax is not None and dim_per is not None and D == int(Kmax) * int(dim_per):
            sample_repr = "gated_raw"
        else:
            sample_repr = "fixed"

    if sample_repr == "gated_raw":
        if Kmax is None:
            raise ValueError("For gated_raw plotting you need plot.Kmax in the JSON.")
        if dim_per is None:
            raise ValueError("For gated_raw plotting you need plot.dim_per in the JSON.")

        K = int(Kmax)
        dim_per = int(dim_per)
        p_idx = int(p_index)
        if p_idx < 0:
            p_idx = dim_per + p_idx

        if not (0 <= p_idx < dim_per):
            raise ValueError(f"Invalid p_index={p_index} for dim_per={dim_per}")
            
        if D != K * dim_per:
            raise ValueError(
                f"gated_raw expects D = Kmax * dim_per = {K * dim_per}, got D={D}"
            )

        th = samples.reshape(samples.shape[0], K, dim_per)

        # remove the gate column, keep the physical ones
        phys_idx = [i for i in range(dim_per) if i != p_idx]
        if len(phys_idx) != n_phys:
            raise ValueError(
                f"After removing p_index={p_idx}, got {len(phys_idx)} physical dims per source, "
                f"but fields_list has {n_phys} fields: {fields_list}"
            )

        X = th[:, :, phys_idx].reshape(samples.shape[0], K * n_phys)

        print("[plot debug] detected gated raw posterior")
        print("[plot debug] Kmax =", K)
        print("[plot debug] dim_per =", dim_per, " p_idx =", p_idx)
        print("[plot debug] X.shape =", X.shape)

    else:
        extras_arg = None if args.extras is None else int(args.extras)
        K, extras = infer_K_and_extras(
            samples,
            fields_list,
            start_col=int(args.start_col),
            extras=extras_arg
        )

        print("[plot debug] inferred K =", K, " extras =", extras)

        X = samples[:, int(args.start_col) : int(args.start_col) + K * n_phys]
        print("[plot debug] X.shape =", X.shape)
        print("[plot debug] plotted K =", K)

    # default to plain for maximum robustness unless user explicitly requests math
    label_style = getattr(args, "label_style", "plain")
    if label_style not in ("plain", "math"):
        label_style = "plain"

    labels = labels_lisa_safe(
        fields_list=fields_list,
        K=K,
        aliases=aliases,
        style=label_style,
        k_start=1
    )
    labels = sanitize_labels_for_matplotlib(labels)

    if truth is not None and truth.shape[0] != X.shape[1]:
        raise ValueError(f"--truth has {truth.shape[0]} values but plotted dimension is {X.shape[1]}.")

    fig = corner.corner(
        X,
        weights=weights,
        labels=labels,
        show_titles=(not args.no_titles),
        plot_contours=(not args.no_contours),
        title_fmt=".3e",
        truths=truth
    )
    fig.suptitle(f"{args.algo} – per-component posterior (K={K}, repr={sample_repr})")

    if args.save:
        fig.savefig(args.save, dpi=150, bbox_inches="tight")
    elif not args.no_show:
        try:
            plt.show()
        except Exception:
            fig.savefig("posterior.png", dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    main()
