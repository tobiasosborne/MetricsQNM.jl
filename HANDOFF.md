# Handoff — MetricsQNM.jl (updated 2026-06-08)

Five-minute orientation for a fresh agent. Read `CLAUDE.md` next, then `REBOOT.md`,
`FINDINGS.md`, and the paper source `papers/2406.11986_src/main.tex` before writing code.

## Project in one sentence
Compute Kerr (GR) and scalar-Gauss-Bonnet (sGB) quasinormal-mode frequencies via the METRICS
spectral method, reproducing Tables I (arxiv:2312.08435) and II–IV (arxiv:2406.11986) — then
exceeding them — as a **complete, Mathematica-free Julia pipeline** (M = 1; ωM dimensionless).

## Current state (2026-06 reboot in progress)
- **Branch `main` is canonical.** It is an unrelated clean-reboot history (no common ancestor
  with `master`/`svd-compression-solver`, the old dead-end lineage kept for archaeology).
- **GR core: works.** 219/220 digits of Table I via `build_system_bespoke` → `solve_qep_svd`.
  Caveat: `solve_qep_svd` is a *seed-dependent* Ritz/shift-invert solve (Leaver-seeded → 9e-14;
  cold seed → ~2%), not a seed-free global eigensolver (FINDINGS §qep-seed).
- **sGB extension: BROKEN, not merely unfinished.** Its headline algorithm (per-equation
  matched clearing + full Newton on (v,ω), `build_matched_sgb_system` + `solve_qep_full_newton`)
  **does not converge.** Empirical (a=0.3, ζ=0.01, Leaver-seeded, ran this session): N=16 and
  N=18 hit max-iter with residual stuck at ~1e-8 (never → 1e-12), ω drifts off the fundamental,
  ω₁≈|9.5|→|10.7| vs paper Table II O(0.1), and it gets *worse* with N (refuting the
  "N<d_max=16 aliasing" hypothesis). Builds are slow (~340–400 s/N at N=16–18).
- **Why it's broken (prime suspects):** the paper's **step 2 "divide out each equation's common
  Σ^p Δ^q (1−χ²)^s prefactor"** (main.tex 976–978) is **not implemented** → wrong-gauge D̃¹;
  and only **Source 1 of 4** sGB sources is implemented (system is 10×6; paper is 11×7) so a
  converged ω₁ would still be physically incomplete. See FINDINGS §prefactor-division, §sgb-sources.

## What this session did (2026-06)
- Made `main` canonical (switched local checkout off the old `svd-compression-solver` snapshot).
- Audited `main`; ran the full-Newton blocker test (the BROKEN result above; the prior HANDOFF
  had it as "IN PROGRESS — USE THIS", which is now known to be wrong).
- Found the old `CLAUDE.md` dogma ("keep `a` symbolic / extend CAS to 6 vars; Newton forbidden;
  QEP needs no seed") **contradicts the paper and our own findings** — it drove the over-built
  1722-line `symbolic_pipeline.jl`. **Rewrote `CLAUDE.md`** to the sibling-project house style
  (ground-truth-before-code, red-green TDD, cross-checks-before-code, fail-loud, domain
  hallucination callouts, probe/sweep incident log).
- Wrote **`REBOOT.md`** (staged plan + phase gates) and seeded **`FINDINGS.md`** (12 traps).

## Decisions (user, 2026-06)
- **H_i background = stage it:** Phase-1 export from notebook via wolframscript → committed
  Julia data (`goldens/`) → Mathematica-free at runtime; Phase-2 recompute natively in Julia.
- **"Outperform" = speed + reproduce Tables I & II first** (with a real regression harness),
  then exceed. wolframscript is dev-time-only (golden masters).

## Next steps (REBOOT.md Phase 0 → 1)
1. Establish baseline GREEN (capped: package loads + GR ω₀ vs Leaver at small N).
2. Apply the verified dead-code deletion (plan ready: ~450 lines from `symbolic_pipeline.jl`,
   ~270 from `sgb_linearize.jl` incl. the 4 `UndefVarError` landmines; 4 dead test files).
3. wolframscript export H_i → `goldens/*.json` → pure-Julia loader (drop `.nb` runtime dep) →
   **first golden-master gate** (H_i Julia vs Wolfram; FINDINGS §notebook-path, §wolframscript-segfault).
4. **Implement the paper's step-2 prefactor division**; re-run the N-sweep; gate on combined
   system at ζ=0 ⇒ GR ω₀ (machine precision) and ω₁ → O(0.1) stable in N.

## Key signatures (current, on main)
```julia
ωL = leaver_qnm(a; s=-2, l=2, m=2, n=0)                 # reference oracle
sys, nf = build_system_bespoke(a, N, m)                  # GR D̃ (proven)
e = solve_qep_svd(sys; ω₀=ωL, refine=1)                 # seed-dependent QEP
sys_gr, sys_corr, _, matched = build_matched_sgb_system(a, N, m; max_a_order=2)  # sGB (broken: needs step-2)
res = solve_qep_full_newton(sys_full, ωL, v0; parity=:polar, tol=1e-12)          # does NOT converge yet
```

## Memory / process safety (WSL2, OOM history — see CLAUDE.md Rule 13)
Cap every Julia run: `( ulimit -v 28000000; OPENBLAS_NUM_THREADS=2 julia --project=. --threads=2
--heap-size-hint=10G script.jl )`, small N first, never an unbounded sweep. Don't delegate heavy
Julia to autonomous agents (they retry → runaway). wolframscript segfaults on exit *after*
output (Export to disk, ignore exit code). `pkill -f` self-matches → exit 144 (use `pkill -x`).

## Per-equation sGB clearing targets (a=0.3, reference; from the matched-clearing diagnostic)
eq1,4,7,9,10: Σ⁶Δ²(1−χ²)¹ (+8 r-deg over GR); eq2: Σ⁵Δ³(1−χ²)¹ (+8); eq3,6: Σ⁵Δ²(1−χ²)² (+6);
eq5,8: Σ⁵Δ²(1−χ²)¹ (+6). Per-a-order d_max: a⁰=16 … a⁹=7 (no global inflation). These feed the
step-2 prefactor division.
