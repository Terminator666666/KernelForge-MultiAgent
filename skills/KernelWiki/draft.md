# Refresh Round Draft — Tool-Version Correction + Upstream Repo Refresh

> Status: **DRAFT**, not yet a RLCR plan. Discuss with Codex before promoting via `/humanize:gen-plan`.
> Repo today (`2026-04-27`): 545 markdown files / 0 validation errors / quality-gate snapshot dated `2026-04-17`.
> Previous Phase-3 draft (data-depth / artifact-bundle work) has been merged — see `plan-phase3.md`. This draft tracks the **next** refresh, not Phase 3.

## 1. Why now

Two drift problems are bleeding accuracy:

1. **Version-locked claims are now wrong.** The wiki was authored against Triton 3.5 (no native tcgen05/TMEM path), CUTLASS 3.x/4.x at the time of writing, etc. Triton 3.6 ships native `tcgen05.mma` + TMEM (gluon front-end + `warp_specialize` + tcgen05 lowering), and CUTLASS has moved to 4.5-dev (one PR in the cutlass ledger is literally titled "Update blackwell tutorial to be compatible with 4.5-dev version"). Several wiki pages still teach the older world.
2. **Upstream repos have moved.** Candidate ledgers were searched on `2026-04-17`; today is `2026-04-27` (10 days of new merges). DeepGEMM in particular is referenced as a kernel case study but has **no tracked-repo slot**: it lives only as a blog (`blog-deepgemm`) plus a single artifact bundle pinned at SHA `7f2a703…` from `2026-04-17`. New DeepGEMM commits (sparse-MLA path, MoE layout work, NVFP4 explorations) cannot be ingested through the current schema.

These two problems share infrastructure (validator, ledgers, artifact-bundle refresh), so they belong in one refresh round.

## 2. Concrete drift inventory

### 2.1 Outdated claims (file:line)

| Where | Claim | Status | Fix |
|---|---|---|---|
| `wiki/languages/triton-blackwell.md:11` | `blackwell_relevance: "no tcgen05/TMEM access"` (frontmatter) | Stale: Triton 3.6 ships tcgen05 codegen + TMEM accumulator | Rewrite frontmatter blurb; tag as `version-sensitive` |
| `wiki/languages/triton-blackwell.md:16` | "Triton-generated kernels typically achieve 30-50% of hand-optimized CuTe-DSL" | Pre-3.6 number; new gluon-tcgen05 path closes much of the gap, ratio depends on the kernel shape | Replace with a dated table: pre-3.6 vs 3.6+ |
| `wiki/languages/triton-blackwell.md:18-26` | `FlashInfer-Bench` table for GPT-5 / Gemini 2.5 Pro / Claude Opus 4.1 | Undated, no Triton version stamp | Stamp Triton version + benchmark date; promote to a per-version subsection |
| `wiki/languages/triton-blackwell.md:73-78` | "Limitations on Blackwell" lists "No direct tcgen05 access … compiler generates wgmma" and "No TMEM: accumulators stay in registers" | Both false in Triton 3.6 | Replace with "Pre-3.6 limitations" + "3.6 path" |
| `data/inclusion-policy.yaml:5-7` | Comment justifies Triton lane scope by citing the no-tcgen05/TMEM claim | Premise gone, scope still narrower than CuTe DSL but for different reasons (compiler maturity, not access) | Rewrite the rationale and reconsider sub-scopes |
| `references/primer.md:86` | "Triton: On Blackwell: no direct tcgen05/TMEM exposure; useful for prototypes." | Stale | Rewrite |
| `references/examples.md:107` | "Triton limitations on SM100 (no tcgen05/TMEM direct access)" | Stale | Rewrite |
| `wiki/languages/cute-dsl.md:15,94` | "CUTLASS 4.x" | Drifting (4.5-dev now landing) | Pin a version range and re-verify FA4 perf claim against current build |
| `wiki/hardware/clc.md:160` | "CUTLASS 3.x for SM100 provides CLC support" | Inconsistent with other pages saying 4.x | Reconcile with current CUTLASS minor version |
| `wiki/kernels/nsa.md:179,191`, `wiki/kernels/gated-delta-net.md:251`, `wiki/kernels/grouped-gemm.md:187` | Generic "Triton has CPU launch overhead / use CUDA graphs" framing | Still partly true but premise weaker post-3.6 | Audit each one and add a "CUDA-graph vs gluon" decision note |
| `plan-phase2.md`, `plan-phase3.md` | Multiple references to the no-tcgen05 premise | Historical record — leave as written, but add a top-of-file "superseded by refresh round" pointer |

### 2.2 Repo / source freshness

Existing tracked repos (ledger `searched_at`: `2026-04-17` for all 5):

| Repo | Captured PR pages | Newest captured `date` | Ledger include-set | Gap (included, not captured)¹ |
|---|---|---|---|---|
| NVIDIA/cutlass | 32 | 2026-03-25 | 34 | 2 |
| sgl-project/sglang | 105 | 2026-04-14 | 684 | 628 |
| vllm-project/vllm | 126 | 2026-04-16 | 908 | 827 |
| flashinfer-ai/flashinfer | 126 | 2026-01-22 | 642 | 516 |
| pytorch/pytorch | 71 | 2026-04-15 | 80 | 35 |

¹ Gap is an upper bound: `generate-pr-pages.py` re-runs `is_kernel_related` against PR file diffs and silently skips PRs whose actual file changes don't match the kernel allowlist, so the realized "missing pages" count is smaller. Need to instrument `generate-pr-pages.py` to emit a `data/pr-page-skipped.yaml` that records the second-pass filter decisions before we can size this exactly. **FlashInfer's date gap (Jan→Apr) is the clearest signal** — the include-set spans through `2026-04-16` but no PR-pages exist past `2026-01-22`.

Repos that are **referenced in the wiki but not tracked**:

| Upstream repo | Where it is mentioned | Current ingestion path | Refresh problem |
|---|---|---|---|
| `deepseek-ai/DeepGEMM` | `wiki/kernels/deepgemm.md`, `blog-deepgemm`, artifact bundle pinned at `7f2a703` (`2026-04-17`) | Single blog summary + 2 verbatim `.cuh` files | No candidate ledger; no PR pages; can't ingest commits between bundle SHAs without manual re-upload. Wiki cannot answer "what changed in DeepGEMM since April?" |
| `deepseek-ai/FlashMLA` | `wiki/kernels/flashmla.md`, `wiki/kernels/sparse-mla.md`, `blog-flashmla` | Blog summary + 2 code blocks extracted from blog markdown (`upstream_sha: none`) | Same gap; sparse-MLA evolution (V3.2 line) hard to track |
| `Dao-AILab/flash-attention` (FA-4) | `wiki/kernels/flash-attention-4.md`, `blog-flash-attention-4`, `doc-flash-attention-4` | Blog + Tri Dao writeup; no upstream commit pin | Cannot follow FA-4 patches |
| `triton-lang/triton` | `wiki/languages/triton-blackwell.md`, `lang-triton` | None — wiki claims about Triton are not anchored to a Triton release tag | The bug we are fixing in Stream A originates here |

### 2.3 Tooling gaps that surface during refresh

- `scripts/generate-pr-pages.py` does not record **why** a ledger-included PR was skipped at PR-page generation time. Needed before Stream B can claim coverage.
- `scripts/fetch_pr_diff.py` consumes `data/core-prs.yaml` (52 entries) but artifact bundles for the long-tail PR pages are not refreshed when their merge SHA changes upstream (e.g., reverted PRs). `scripts/verify_verbatim.py` exists but only re-checks SHA-256 against pinned blobs, not against upstream HEAD.
- No equivalent of `candidate ledger` exists for **DSL-version sensitivity**. We need a "version-sensitive claims registry" so that whenever Triton/CuTe/CUTLASS releases a new minor, the validator flags pages that need re-verification.

## 3. Goal (Definition of Done)

After this round:

1. Every claim of the form "{tool} {version} cannot do {feature}" is annotated with the tool's version range AND validated against the latest stable release. Triton 3.6's tcgen05 + TMEM path is documented as a first-class lane; pre-3.6 limitations are kept as a "Historical context" subsection.
2. `data/inclusion-policy.yaml` for Triton is rewritten on a current premise — capture criteria refer to "kernels lowering through tcgen05" or "kernels that beat CUDA-graph CuTe-DSL on shape X" rather than "Triton has no tcgen05 path".
3. DeepGEMM, FlashMLA, and FA-4 are first-class tracked upstreams: each has a candidate ledger + a `sources/prs/<repo>/` slot, and their wiki kernel pages cite specific upstream commits (not just a blog snapshot).
4. The five existing ledgers are re-searched up to today's date; PR-page coverage is regenerated; the second-pass skip reasons are written to a queryable file.
5. `scripts/validate.py` enforces a new `version_sensitive: <tool>@<version-range>` field on any wiki page that makes a version-dependent claim, and a stale-pin warning fires if the recorded version is older than N days from a known-good release.
6. The 5 existing artifact bundles for tcgen05/Triton-related kernels are re-verified against current upstream HEAD and either re-pinned or annotated with an `upstream_drift: <#commits>` line.

## 4. Plan outline (for Codex to challenge)

### Stream A — Tool-version correction (1–2 RLCR rounds)

Round A1: schema + validator for version-sensitive claims.
- Add optional `version_sensitive` block to wiki frontmatter:
  ```yaml
  version_sensitive:
    - tool: triton
      claim_valid_for: ">=3.5,<3.6"
      last_verified_release: "3.5.1"
      last_verified_at: 2026-01-15
  ```
- Validator rule: any page whose body contains a regex like `\b(no|cannot|does not).+(tcgen05|TMEM|wgmma|UMMA)\b` must carry a `version_sensitive` block, OR explicitly opt out via `version_sensitive: not-applicable` with a one-line reason.
- New `scripts/check_version_freshness.py`: looks up the latest published version of each tracked tool (Triton, CUTLASS, CUDA), compares against `last_verified_release`, and prints stale rows.

Round A2: rewrite `wiki/languages/triton-blackwell.md` and downstream references.
- Restructure into "3.6+ path", "pre-3.6 history", "When to still prefer CuTe DSL".
- Update `references/primer.md`, `references/examples.md`, `data/inclusion-policy.yaml` comment block.
- Verify the FlashInfer-Bench table is reproducible against a current FlashInfer-Bench snapshot; if not, mark the table as "as of 2025-XX, Triton 3.5" and add a "needs re-run on 3.6" placeholder.
- Update CUTLASS minor-version stamps (`3.x` vs `4.x` vs `4.5-dev`) on a version-sensitive registry; do not pretend a single number applies across the wiki.

### Stream B — Upstream repo refresh (3–4 RLCR rounds)

Round B1: schema + tooling for new tracked repos.
- Add `deepseek-ai/DeepGEMM`, `deepseek-ai/FlashMLA`, `Dao-AILab/flash-attention` to the tracked-repo list (`SKILL.md`, `README.md`, `references/primer.md` repo table, candidate ledger directory).
- Generalize `scripts/generate-pr-pages.py` and `scripts/fetch_pr_diff.py` to handle these three repos (the file-path allowlists need to know about `deep_gemm/`, `csrc/sm100/`, etc.).
- Define `searched_at` cadence: ledgers must be searched within 14 days of the validator run, otherwise it warns.

Round B2: refresh existing five ledgers.
- Re-run candidate-ledger generation for cutlass/sglang/vllm/flashinfer/pytorch up to today's HEAD.
- Instrument `generate-pr-pages.py` to record second-pass skip reasons in `data/pr-page-skipped.yaml`. Backfill skip reasons for the existing 105/126/126/71 captured pages so the gap numbers are auditable.
- Close the FlashInfer Jan→Apr capture gap (clearest miss in the inventory).

Round B3: ingest DeepGEMM / FlashMLA / FA-4.
- For each: write the candidate ledger, generate PR pages for last 12 months of merges, refresh the artifact bundles to a current SHA.
- Update the wiki kernel pages (`kernel-deepgemm`, `kernel-flashmla`, `kernel-flash-attention-4`, `kernel-sparse-mla`) to cite the new PR IDs alongside the blog summaries.

Round B4: provenance audit.
- Run `scripts/verify_verbatim.py` over every artifact bundle; for any bundle whose `upstream_sha` is more than 60 days behind upstream HEAD, emit a `bundles-needing-refresh.md` report and either re-pin or add `upstream_drift_commits: <N>`.
- Drop `version_sensitive` blocks on wiki pages that depend on each refreshed bundle.

### Stream C — Validation & sign-off (1 round)

- `validate.py` clean.
- `query.py "Triton tcgen05"` returns the rewritten lang-triton page in the top 3.
- `query.py --repo deepgemm` returns ≥ 5 PR pages.
- Ad-hoc smoke: ask the skill "what changed in DeepGEMM since the April snapshot?" and verify the answer cites at least one PR page newer than `2026-04-17`.
- Codex sign-off on (a) Triton-page rewrite accuracy and (b) DeepGEMM/FlashMLA/FA-4 ingestion scope.

## 5. Open questions for Codex

1. **Version-sensitive registry placement**: per-page frontmatter (proposed) vs. central `data/version-claims.yaml`? Per-page is more local but harder to audit; central is easier to script but drifts from the page that owns the claim.
2. **Triton scope**: now that 3.6 has tcgen05, should the Triton inclusion-policy lane widen toward parity with the CuTe DSL lane, or stay narrower because Triton's optimizer is still less mature for CTA-cluster kernels? (I lean toward "widen, but with a `triton-tcgen05-mature: false` skip-criterion until we can benchmark.")
3. **New tracked repos**: DeepGEMM is unambiguous. FlashMLA may overlap heavily with sglang's vendored copy — do we double-track or reference-only? FA-4 raises the same question vs. vllm's vendored fork.
4. **Ledger re-search cost**: each repo's `gh search prs` query costs O(N) API calls; refreshing all 5 ledgers + 3 new ones in one round may exceed the 5000/hour secondary rate limit. Stream B may need to be split across two RLCR rounds with checkpointing.
5. **FlashInfer-Bench freshness**: is the Triton-3.6 number even publicly available yet? If not, do we strip the table and replace with a single sentence pointing to the live leaderboard, rather than ship pre-3.6 numbers next to a "3.6 has tcgen05" claim?
6. **Backwards links from `plan-phase2.md` / `plan-phase3.md`**: those plans are historical record. Do we add a "superseded by refresh round" header pointer to them, or leave them untouched and let the validator's version-sensitive checker handle drift?
7. **CuTe DSL CUTLASS-version stamps**: I propose pinning to a specific CUTLASS minor (e.g. `4.5.x`) in each page that claims "CUTLASS 4.x supports …". Is that worth the maintenance cost vs. a single global version variable?
8. **Newer tool-version drift not yet flagged**: `cutile` (CUDA 13.1), `tilelang`, `tilus`, `JAX-Pallas` Blackwell support — should we proactively run the same audit on these, or wait until we have a concrete user complaint?

## 6. Risks

- **Triton 3.6 capability claims**: I have not verified the exact 3.6 release notes inside this session. Before promoting the Triton page rewrite, we must read the 3.6 changelog and one tutorial to confirm the tcgen05/TMEM path is genuinely user-visible (not gluon-only / not dev-branch-only).
- **Refresh round vs. existing artifact-stability guarantee**: Phase 3 enforces verbatim SHA-256 on artifact bundles. Re-pinning bundles in Round B4 will break those checksums by design — the validator must understand the difference between "drift" and "tampering".
- **Ledger churn**: re-searching the 5 existing ledgers will reshuffle decisions on edge-case PRs (e.g. PRs we previously deferred may now be includable, and vice versa). Need a `decision_history` field on each ledger entry so we don't silently flip past decisions.
- **Scope creep into Phase 4**: this draft is deliberately scoped to "drift fix", not "new wiki content". Resist the urge to add new wiki pages (e.g. a Triton-3.6-specific kernel case study) inside this round.

## 7. Request for Codex

Please push back on:

1. **Stream A's version-sensitive schema** — is the frontmatter approach right, or should this be a sidecar file?
2. **Triton 3.6 evidence** — what would you cite to confirm the tcgen05 path is real and stable, beyond a release note? Is there a Triton-tutorial PR or a concrete model-serving repo (vLLM, SGLang, FlashInfer) that already lowers through it?
3. **Whether DeepGEMM warrants its own ledger** vs. being treated as a special-case artifact bundle that the wiki refreshes manually each quarter. (My take: ledger; DeepGEMM commits are too frequent.)
4. **Round count** — 1+1+3+1 = 6 RLCR rounds feels heavy. Can Stream B's three rounds be folded into two by deferring artifact-bundle re-pinning to a follow-up?
5. **Anything I missed in the drift inventory** — particularly outdated claims about CUDA toolkit versions (we mention CUDA 12.8, 13.1, 13.x in different pages with no central source of truth), or about Hopper/Blackwell driver requirements.
