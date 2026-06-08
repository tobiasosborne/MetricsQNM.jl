# FINDINGS.md — MetricsQNM.jl

Living log of paper subtleties, load-bearing traps, the "tests that can't fail," and open
escalations. **Skim before touching a new region; append when you find a new trap; cite
`FINDINGS.md §<slug>` from the source comment where it bites.** (House convention from
`../almost-idempotent-channels/paper/FINDINGS.md`.) Line numbers marked "verify" must be
confirmed against the live `.tex` when cited in code.

---

## §prefactor-division — the missing paper step 2 (OPEN, prime suspect)
The paper clears all denominators per equation, **then divides each equation through by its
common Σ^p Δ^q (1−χ²)^s prefactor** "to improve numerical stability" (main.tex 976–978,
verify). The code clears (per-equation matched clearing in `build_matched_sgb_system`) but
does **not** do the prefactor division. **Symptom:** full Newton on the combined D̃⁰+ζD̃¹
does not converge — at a=0.3, ζ=0.01, seeded at Leaver, N=16/18 plateau at residual ~1e-8,
ω drifts off, ω₁≈|10| (N=16) → |10.7| (N=18) vs paper Table II O(0.1), and it gets *worse*
with N (refuting the "N<d_max aliasing" hope). Implement step 2 and re-test (REBOOT Phase 1).

## §gauge-consistency — D̃⁰ and D̃¹ must share one clearing
Eigenvalue perturbation (Eq. 111) and full Newton both require D̃⁰ and D̃¹ in the **same
gauge** (same per-equation clearing). Independent clearing gave pinv residual 0.989 and
ω₁≈1.3e8 (test_independent_clearing, FAILED). Cross-check: the combined system at ζ=0 MUST
reproduce GR ω₀ to machine precision — if not, D̃¹ assembly corrupted the gauge.

## §numerical-a — fine, but only AFTER clearing
Substituting `a` as Float64 *before* extracting/clearing coefficients bakes the Kerr metric's
r-structure (~15–24 powers of r from Σ,Δ) into c_{k,d,p}, inflating d_max so the degree-N
basis can't resolve it (cond(J) ∝ N⁵). The paper keeps `a` as a power series and evaluates it
numerically *per order* (main.tex 1237–1248, verify). The earlier "keep `a` symbolic / extend
CAS to 6 vars" approach chased a property the paper never used; it is not required. Clear
per-equation, then evaluate `a` per order.

## §qep-seed — the QEP solver is seed-dependent (NOT "all eigenvalues, no seed")
`solve_qep_svd` (rectangular_qep.jl) is a Ritz/shift-invert projection at ω₀. Empirically
(a=0.3): Leaver-seeded → |Δω|=9.14e-14; cold seed (0.5−0.1i) → nearest eigenvalue 1.9e-2 off.
The "219/220 digits" GR result is real but obtained with a 2-sig-fig Leaver seed (a legitimate
physical prior). Document the seed; do not claim seed-free global eigensolving.

## §sigmin-newton-spurious — σ_min Newton on ω alone fails at high d_max
With sGB d_max≈16, σ_min-minimization has no eigenvector info and converges to spurious
basins (ω≈120−3171i). Use full Newton on (v,ω) or eigenvalue perturbation. (HANDOFF; FAILED
ablations test_matched_clearing, test_full_newton.)

## §H4-cleanup — cleanup! must be ABSOLUTE tolerance, not relative
LCD clearing inflated H4's numerator max coefficient to 5.94e13; `cleanup!(...; relative=true,
tol=1e-15)` then thresholded at 0.059 and dropped 8 legitimate terms (coeffs 0.012–0.058),
causing a 1.5% error that masqueraded as a deep bug. Fix: absolute `tol=1e-15` for sGB cleanup.
Trap pattern: any relative tolerance against a denominator-cleared polynomial.

## §deriv-convention — z-space operators are correct (not a missing chain rule)
The h-term α indices are r-derivative orders, but assembly applies z-derivative operators.
This is NOT a missing chain rule: `_r_to_z` bakes the chain-rule factors into the z-space
coefficients (coeff = (2rp)^δ · G_val). It is correct and yields 219/220 GR digits. CAVEAT:
`_leg_d3` (third χ-derivative, β=3, new for sGB) is comment-derived and independently
unverified — audit it against a golden master if sGB angular results look wrong.

## §iu-trick — complex ω via iu∈{0,1,−1}, degree ≤2 assumed
Complex frequency dependence is reconstructed from evaluations at iu∈{0,1,−1}, assuming total
degree ≤2 in (ω_re, ω_im, iu). Higher structure fails silently — assert the degree.

## §swsh-lambda — SpinWeightedSpheroidalHarmonics returns λ, not A_slm
Convert λ → A_slm before feeding the angular separation constant to the Leaver solver.
(memory: feedback_swsh_convention)

## §sgb-sources — only 1 of 4 sGB sources implemented (physics incompleteness)
`sgb_linearize.jl` (compute_sgb_correction_equations) implements only Source 1 (O(ζ) Ricci
from the H background). Missing: Source 2 (sGB modification tensor A_μ^ν), Source 3 (scalar
stress-energy T_μ^ν), Source 4 (horizon Ω_H¹, κ¹ corrections), plus the 7th unknown h₇
(scalar perturbation) and the 11th equation (scalar wave). System is 10×6; paper is 11×7. A
converged ω₁ from the current code would still be physically wrong — this is REBOOT Phase 3.

## §notebook-path — runtime .nb dependency (being removed)
`sgb_background.jl` reads `reference/2406.11986_source/Supplementary_materials.nb`, a path NOT
tracked on main (only `papers/2406.11986_src/` is) → fresh checkout cannot build sGB. Phase-1
exports H_i to committed Julia data (`goldens/`) and removes `.nb` parsing from the runtime.

## §wolframscript-segfault — kernel segfaults on shutdown AFTER output
wolframscript v1.13 on this Linux box prints/Exports correctly, then segfaults on kernel
shutdown. Always `Export` to disk and capture stdout; never treat the nonzero exit code as
failure.

---
## Open escalations
- Confirm exact main.tex line numbers above (currently from session audit, not re-verified).
- Decide whether the reboot keeps SparsePoly N_VARS=6 (symbolic `a`) or reverts to 5
  (numerical `a` per order makes 6 likely unnecessary; measure before deciding).
