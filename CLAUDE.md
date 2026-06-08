# CLAUDE.md — MetricsQNM.jl

> Authoritative instruction set for this repository. Read it top to bottom at the
> start of every session and after any context compression — the rules drift out of
> working memory faster than you think, which is why they are numbered. House style
> shared with the sibling projects `../su2-fft`, `../almost-idempotent-channels`,
> `../BennettVM.jl` (ground-truth-before-code, red-green TDD, cross-checks-before-code).

## What this is

A Julia implementation of the **METRICS** spectral method for black-hole quasinormal
modes (QNMs): reproduce **Table I of arxiv:2312.08435** (Kerr/GR) and **Tables II–IV of
arxiv:2406.11986** (scalar-Gauss-Bonnet, sGB), then exceed them. Pipeline: symbolic
linearization of the field equations → a bespoke `SparsePoly` CAS (denominator-tracking) →
spectral (Chebyshev×associated-Legendre) Galerkin assembly of D̃(ω) → rectangular QEP /
Newton eigenvalue solve. **M = 1**; table values are ωM (dimensionless).

## THE MANDATE — a complete, Mathematica-free Julia pipeline that beats the original

This is the reboot's reason to exist (see `REBOOT.md`). The production code runs with
**zero Mathematica/wolframscript dependency at runtime**; wolframscript is used **only at
development time to generate golden masters**. The pipeline must be **faster than the
original and at least as accurate**, reproduce the papers' tables under a real regression
harness, and then be pushed beyond (more digits / spin / modes). The one Mathematica-sourced
piece (the sGB background H_i) is being staged out: Phase-1 export-to-Julia-data; Phase-2
recompute natively in Julia. **Do not reintroduce `.nb` parsing into the runtime path.**

If a result resists reproduction or a stage cannot be validated against an oracle, that is a
finding to escalate (a stop condition), not a corner to paper over. File it in `FINDINGS.md`;
do not fake it.

## Read order

For any task, in order. **Refuse to add code or math content before steps 1–3.**

0. `HANDOFF.md` — five-minute orientation: current state, what's built, what's next.
1. This file (`CLAUDE.md`).
2. `REBOOT.md` — the staged reboot plan, decisions, and phase gates.
3. **`papers/2406.11986_src/main.tex`** (sGB) and `papers/source/main.tex` (GR) — the
   verbatim arXiv source. **Canonical ground truth.** When notes, memory, a HANDOFF, or a
   subagent disagree with it, the `.tex` wins.
4. `FINDINGS.md` — the living log of paper subtleties, known traps, the "tests that can't
   fail," and open escalations. Skim before touching a new region; **append to it** when you
   find a new trap, and cite `FINDINGS.md §X` from the source comment where it bites.
5. `ALGORITHM.md` (once it exists) — the canonical math+code narrative for the module you
   touch, every formula cited to a `.tex` line.

If a claim in code is not anchored to a `.tex` line number, it is undocumented; fix that
before changing the code.

## The Laws (unconditional)

**Law 1 — Ground truth before code.** Every implementation decision cites the `.tex` line it
implements, with a verbatim copy of the equation/statement in a source comment:

```julia
# main.tex:976-978 — "we divide the equations by them to simplify them and improve their
#   numerical stability": after clearing to the per-equation LCD Σ^p Δ^q (1-χ²)^s, divide
#   the whole equation k through by its common prefactor. (FINDINGS §prefactor-division)
```

Not your memory of the paper. Not what "should" work. The paper is on disk — open it.

**Law 2 — Reuse before reinvention.** Do not reinvent the proven core. The GR pipeline
reproduces 219/220 digits of Table I; the SparsePoly CAS, the spectral operators, the Leaver
oracle, and TensorGR.jl (`../TensorGR.jl`, **read-only**) are deliberate choices. Wrap and
extend; do not "modernize" the working path without a measured improvement.

**Law 3 — Validate, don't trust.** Every stage is checked against an independent oracle
(golden masters, Leaver, exact special cases) with the check committed to `runtests.jl`. The
old code rotted into a *non-converging* state precisely because **nothing asserted
correctness**. A computed number is wrong until an oracle says otherwise.

**Law 4 — Audition, don't presume.** No solver or assembly route is presumed fit. Solver
routes (QEP survey, σ_min Newton, full Newton on (v,ω), eigenvalue perturbation), clearing
strategies, and N-selection are **candidates auditioned via red-green TDD against correctness
AND performance**, then selected on evidence. When no candidate dominates, add performance
dimensions (wall time, cond(J), accuracy, N-convergence rate, basin robustness) and keep the
Pareto frontier, dispatchable, with regime boundaries recorded.

## The Rules (numbered, non-negotiable)

0. **Laws 1–4 apply.** Always.

1. **Cite everything.** Every math routine cites (a) the `.tex` line and (b) the verbatim
   equation. This is how a routine survives a rewrite.

2. **Skepticism.** Be skeptical of your own assumptions, subagent output, previous-session
   claims, the existing tests, and the paper itself (LaTeX typos and off-by-one constants
   happen). LLM summaries of GR/QNM derivations are especially untrustworthy — re-read the
   source. When a test passes, ask whether it actually exercised the property you think.

3. **All bugs are deep.** No bandaids. A `cleanup!` tolerance that drops "small" terms,
   a constant that makes N=8 pass but garbage at N=20, a solver that converges with a Leaver
   seed but to 2% with a cold one — each is a future incident with a long fuse. Find the root
   cause. (This project has *already* been bitten by every one of these.)

4. **Fail fast, fail loud.** `@assert` invariants; never silently return. If a basis is too
   small to resolve d_max, cond(J) explodes, a denominator didn't cancel, or a coefficient
   magnitude is implausible — abort with a clear message at the call site. Corrupted output is
   worse than a crash. **Never** use a *relative* cleanup tolerance against a polynomial whose
   LCD-clearing inflated its max coefficient (that is exactly the 1.5% H4 bug — FINDINGS).

5. **"Runs without errors" is NOT a passing test.** Every test asserts an invariant against a
   known value or bound: GR ω matches Leaver to ≥10 digits; H_i matches the wolframscript
   golden master; the combined system at ζ=0 reproduces GR ω₀ to machine precision; ω₁ matches
   paper Table II; an a→0 limit reproduces Regge–Wheeler/Zerilli. A script that prints numbers
   and `@test`s nothing is a diagnostic, not a test — keep it in `examples/`, not `test/`.

6. **Cross-checks > unit tests, and written BEFORE the code.** The strongest oracles here:
   - **GR ω vs Leaver** (`leaver_qnm`) — machine-precision anchor for D̃⁰.
   - **Julia H_i vs wolframscript golden master** at sample (r,χ,a).
   - **combined D̃⁰+ζD̃¹ at ζ=0 ⇒ GR ω₀** to machine precision (gauge sanity for D̃¹).
   - **sGB ω₁ vs paper Table II** (hardcoded expected values, nlm=022 first).
   - **a→0 ⇒ Schwarzschild** Regge–Wheeler/Zerilli closed forms.
   Unit tests catch typos; cross-checks catch algorithmic errors. Add the cross-check first.

7. **RED-GREEN TDD — two valid shapes.**
   - *Spec-from-scratch:* RED → GREEN → refactor. Write the failing assertion (usually a bound
     or a cross-check) first; watch it fail; write the minimum code to pass.
   - *Port-and-verify:* for a routine realizing a paper result, implement it, capture its
     invariants as tests, **mutation-prove** the tests catch regressions (perturb the impl,
     confirm RED, restore), and cross-validate against an independent oracle.
   The discipline is "a test has caught a real regression," not "the test was committed first."

8. **Get feedback fast.** Run the relevant test after every non-trivial change; check every
   ~50 LOC, don't code blind for 500. A capped `julia --project -e` probe is fine (Rule 13).

9. **Tiered workflow + reviewer-gating.** Scale effort to change size:
   - *Trivial* (≤5 LOC; typo/comment): direct fix, reviewer-exempt.
   - *Small* (≤30 LOC; one function on existing abstractions): write the test, write the code,
     one reviewer subagent before declaring done.
   - *Core* (new algorithm/assembly/solver route, >30 LOC, math routine): research the `.tex`
     + `FINDINGS.md` first; write the cross-check before the code; hostile reviewer subagent
     always. For contested choices (e.g. clearing strategy), run 2–3 research subagents first.

10. **~200 LOC per source file.** When a module approaches it, split. (The 1722-line
    `symbolic_pipeline.jl` is the cautionary tale this reboot is dismantling.)

11. **Literate programming.** Each source file opens with a docstring: which `.tex` result it
    realizes, the method, which defensive checks guard which degeneracy. Update `ALGORITHM.md`
    when you touch the algorithm; the docs are the contract.

12. **No emojis, no marketing.** Read like SQLite / TigerBeetle docs. Concrete numbers always:
    "GR |Δω|=9e-14 at a=0.3,N=8", not "very accurate". No "robust", "blazing fast".

13. **Memory & process safety (WSL2 — this host has an OOM history; see incident log below).**
    Cap every Julia run: `ulimit -v ~24-28e6` (KB), `--threads=2`, `OPENBLAS_NUM_THREADS=2`,
    small N first. Never launch an unbounded N-sweep. Builds are ~6 min/N at N≈16–18 — budget
    for it. **Do not delegate heavy Julia/wolframscript runs to autonomous subagents** (they
    retry on failure and spawn runaway processes); keep workflow agents read-only/no-exec and
    run heavy jobs yourself, capped, single-shot.

14. **No GitHub CI.** Quality gates run locally (user directive across all their projects).
    Do not create `.github/workflows/`.

## Cross-check ladder (weakest → strongest)

1. Internal sanity (norms, Hermiticity-where-expected, finite values, plausible magnitudes).
2. GR ω vs `leaver_qnm` to ≥10 digits (Table I) at small N.
3. Julia H_i vs wolframscript golden master (`goldens/Hi_samples_golden.json`) to ~1e-10.
4. Combined system at ζ=0 reproduces GR ω₀ to machine precision (D̃¹ didn't corrupt D̃⁰).
5. sGB ω₁ vs paper Table II to its stated accuracy; and stable under N→N+2.

## Domain hallucination-risk callouts (mistakes that look right but aren't)

Each cites where it bites. When you catch yourself about to do one, stop and re-check.

- **Numerical `a` is fine — but ONLY after per-equation denominator clearing.** Substituting
  `a` *before* clearing bakes the Kerr metric's r-structure into coefficients (~24 powers of
  r) and inflates d_max → the spectral basis can't resolve it → cond(J) ∝ N⁵. Clear first,
  evaluate `a` per order. (FINDINGS; main.tex 1237–1248) The old "keep `a` symbolic / 6-var
  CAS" dogma was a misreading — the paper uses numerical a^k.
- **The QEP solver is seed-dependent.** `solve_qep_svd` is a Ritz/shift-invert projection at
  ω₀, **not** "all eigenvalues, no initial guess." Cold seed (0.5−0.1i) → ~2% at a=0.3;
  Leaver-seeded → 9e-14. Seed from `leaver_qnm` (a legitimate physical prior).
- **σ_min-Newton-on-ω-alone finds spurious basins at high d_max** (diverges to ω≈120−3171i for
  sGB). Use full Newton on (v,ω) or eigenvalue perturbation (Eq. 111). But note: full Newton
  on the matched system **also currently fails** (ω₁≈|10| vs O(0.1)) — see the prefactor gap.
- **`cleanup!` must use ABSOLUTE tolerance, not relative.** LCD clearing inflates a numerator's
  max coefficient (e.g. to 5.9e13); a relative tol then drops legitimate 0.01–0.06 terms → a
  1.5% error that looks like a deep bug. (FINDINGS §H4-cleanup)
- **The derivative convention is z-space, not a missing chain rule.** α are r-derivative orders
  from h-parsing; the chain-rule factors are baked into K_z by `_r_to_z` (coeff=(2rp)^δ·G).
  This is correct and gives 219/220 digits — NOT a bug. But `_leg_d3` (β=3, new for sGB) is
  comment-derived and unverified: audit it first if sGB angular results look wrong.
- **The combined D̃⁰+ζD̃¹ must be in ONE gauge** (same per-equation clearing) or the
  perturbation is garbage. The **missing paper step 2 (divide out the common prefactor)** is
  the prime suspect for the broken ω₁.
- **iu-trick:** complex ω is reconstructed from evaluations at iu∈{0,1,−1}, assuming total
  degree ≤2 in (ω_re, ω_im, iu). Higher structure fails *silently*. Assert it.
- **Σ→0, Δ→0 near the horizon; spectra are perturbation-sensitive near extremal a.** Don't
  trust naive conditioning at high spin.
- **SpinWeightedSpheroidalHarmonics.jl returns λ, not A_slm** — convert before feeding the
  Leaver angular eigenvalue. (memory: feedback_swsh_convention)

## Probe/sweep hygiene — incidents from the 2026-06 reboot session (do not repeat)

- **An autonomous workflow "empirical" subagent spawned runaway Julia** (`--threads=8`,
  rebuilding matrices) and *relaunched it on every failure*, nearly OOMing WSL. Lesson: never
  let an agent run heavy Julia; agents are read-only/no-exec, the orchestrator runs capped
  jobs single-shot.
- **`pkill -f <pattern>` self-matches the grepping shell → exit 144**, and can miss/мiskill.
  Prefer `pkill -x`, or kill explicit PIDs, and don't trust the exit code of pkill pipelines.
- **`ulimit -v 28e6` (≈28 GB) makes the OS kill a runaway Julia instead of WSL.** Use it on
  every heavy run. 62 GB RAM but treat 10 GB free as the abort floor (`free -h`).
- **wolframscript v1.13 segfaults on kernel shutdown AFTER producing output** — `Export`
  results to disk, capture stdout, and do NOT treat a nonzero exit code as failure.
- **The sGB background notebook path** `reference/2406.11986_source/Supplementary_materials.nb`
  is **not tracked on main** (only `papers/2406.11986_src/`); the runtime `.nb` dep is being
  removed entirely (Phase-1 data export).

## Paper line map (verify against the live `.tex` when you cite — line numbers are approximate)

`papers/2406.11986_src/main.tex` (sGB):
- ~962 clear all denominators per equation; **976–978 divide out common prefactor** (step 2,
  the missing piece).
- ~1219 / ~197 Newton-Raphson for the ζ⁰ (GR) solve.
- 1237–1248, 1244 H_i as a spin power series, numerical a^k per order.
- ~1066–1133, Eq. 111 (~1129) eigenvalue perturbation x⁽¹⁾ = −J⁺·[D̃⁽¹⁾(ω₀)·v₀].
- ~1302 scan N=1..25; ~239 / 125–130 fit ω⁽¹⁾(a); Tables II–IV the sGB targets.

## "Done" checklist for a math-routine change

- [ ] `julia --project -e 'using Pkg; Pkg.test()'` (capped) — all asserts green.
- [ ] GR Table I spot-check still matches Leaver to ≥10 digits at small N.
- [ ] New/changed routine cites its `.tex` line + verbatim equation in a source comment.
- [ ] A cross-check (oracle), not just a unit test, guards the change; mutation-proved.
- [ ] `ALGORITHM.md`/`FINDINGS.md`/`HANDOFF.md` updated as appropriate.

## Stop conditions (escalate to the user)

- A paper result you cannot reproduce after honest effort — file the specific obstruction.
- A bound/accuracy that degrades with N where it should converge (you've likely hit an
  ill-posed gauge/d_max, as the current sGB solver does).
- A `.tex` statement that looks like a typo or contradicts another — flag, do not silently "fix".
- A precision/conditioning wall the spectral method cannot pass at feasible N.

## File map (reboot target — see REBOOT.md)

```
core/  kerr, perturbation_ansatz, spectral, leaver         (proven; keep)
cas/   sparse_poly                                         (keep; demarcate; revisit N_VARS)
gr/    linearize, symbolic_pipeline(GR), assembly, rectangular_qep   (keep; slim)
sgb/   sgb_background(data-loader), sgb_equations, sgb_assembly, sgb_perturbation
goldens/  exported H_i data + wolframscript reference values + the .wls exporters
test/  runtests.jl (cross-checks); examples/ for diagnostic scripts
```

Persistent tracking: this branch uses beads (`bd ready`, `bd show`). Ephemeral in-session
state may use the harness task tools. The harness auto-`memory/` is separate. Don't keep
markdown TODO lists.
