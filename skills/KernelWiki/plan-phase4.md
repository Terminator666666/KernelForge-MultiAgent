# KernelWiki Refresh Round — Tool-Version Correction + Upstream Repo Refresh

## Goal Description

Eliminate two classes of accuracy drift in the KernelWiki without compromising the Phase 3 byte-stable provenance contract:

1. **Tool-version drift** — wiki claims authored against Triton 3.5 (and earlier CUTLASS minors) that are now factually wrong on Triton 3.6+ and CUTLASS 4.5-dev. The most acute case: `wiki/languages/triton-blackwell.md`, `references/primer.md`, `references/examples.md`, and the YAML scalar `data/inclusion-policy.yaml::triton.description` all repeat a "Triton has no tcgen05/TMEM access" claim that no longer holds.
2. **Upstream repo drift** — three case-study upstreams (`deepseek-ai/DeepGEMM`, `deepseek-ai/FlashMLA`, `Dao-AILab/flash-attention`) are referenced as kernel pages but exist only as blog summaries plus single frozen artifact bundles, so the wiki cannot ingest new commits. The five existing tracked-repo ledgers (`cutlass`, `sglang`, `vllm`, `flashinfer`, `pytorch`) were searched on `2026-04-17`; FlashInfer's PR-page coverage in particular stops at `2026-01-22` even though its include-set spans through `2026-04-16`. PR-page generation hardcodes `captured_at: "2026-04-17"` (`scripts/generate-pr-pages.py:242`), so any refresh must fix that before coverage metrics are meaningful.

The round produces (a) corrected tool/DSL claims anchored to specific tool releases and on-repo evidence (DEC-1 hybrid registry: per-page `version_sensitive: <id>` pointer + central `data/version-claims.yaml`), (b) refreshed candidate ledgers and PR pages for the five existing tracked repos (driven by a new `scripts/refresh_candidate_ledger.py` plus checked-in `data/refresh-cutoff.yaml` and `data/refresh-search-results.yaml`), (c) integration of DeepGEMM (full-ledger), FlashMLA (case-study), and FA-4 (case-study) per DEC-2, (d) infrastructure that prevents the next drift cycle (hybrid version-claim registry, PR-page skip audit, validator extensions for ledger shape and CUTLASS pinning per DEC-4), and (e) a deterministic, offline freshness check (`scripts/check_version_freshness.py`, advisory by default with `--strict` opt-in).

## Acceptance Criteria

Following TDD philosophy, each criterion lists positive and negative tests for deterministic verification. All tests are runnable from the repo root unless otherwise noted.

- AC-1: Triton 3.6 page rewrite is evidence-anchored, with the evidence checked in **before** the page rewrite.
  - AC-1.1: Triton 3.6 evidence corpus exists (prerequisite for AC-1.2).
    - Positive Tests:
      - `data/triton-3.6-evidence.md` exists, lists supported Blackwell lowering pathways (e.g., `tl.dot`, gluon, `warp_specialize`), names specific Triton release(s), and cites at least one `official-doc` source-id and at least one `upstream-code` source-id. Each cited source-id resolves via `python3 scripts/get_page.py <source-id>`.
      - At least one new `sources/docs/*.md` page summarizes the Triton 3.6 release notes (`source_category: official-doc`).
      - At least one new `sources/prs/<repo>/PR-<N>.md` page (from an existing tracked repo: cutlass / sglang / vllm / flashinfer / pytorch) with `source_category: upstream-code` demonstrates a kernel that lowers through the Triton 3.6 Blackwell path. (DEC-3 chose to defer `triton-lang/triton` tracking; therefore the upstream-code anchor cannot come from triton-lang itself this round.)
    - Negative Tests:
      - Validator fails when `data/triton-3.6-evidence.md` cites a `source_id` that does not exist as a `sources/**/*.md` frontmatter `id`.
      - Validator fails when the evidence memo contains no entries with `source_category: upstream-code`.
  - AC-1.2: `wiki/languages/triton-blackwell.md` is rewritten with verified evidence.
    - Positive Tests:
      - The page frontmatter has `confidence: verified` and an `evidence_basis:` list with at least one entry whose `evidence_type: official-doc` and at least one whose `evidence_type: upstream-code`. Each entry's `source_id` matches a source-id from `data/triton-3.6-evidence.md`.
      - The page body contains a clearly-marked "Pre-3.6 historical context" subsection that preserves the old "no tcgen05/TMEM" framing AND a "Triton 3.6+ Blackwell path" subsection that enumerates the supported lowering pathways (per the evidence memo, not invented).
      - The page frontmatter carries a `version_sensitive: <id>` pointer that resolves to an entry in `data/version-claims.yaml` (per DEC-1 hybrid). The registry entry's `last_verified_release` matches an entry in `data/tool-versions.yaml`.
      - `wiki-language` page schema in `data/schemas.yaml` is extended to allow `evidence_basis` and `confidence` (already optional; `evidence_basis` newly added) so the validator accepts the rewritten page.
    - Negative Tests:
      - Validator fails when the page declares `confidence: verified` without an `evidence_basis` list containing both an `official-doc` entry and an `upstream-code` entry whose `source_id`s exist.
      - Validator fails when the page's `version_sensitive: <id>` pointer does not resolve to a `data/version-claims.yaml` entry, or when that registry entry names a Triton release with no corresponding entry in `data/tool-versions.yaml`.
      - `python3 scripts/grep_wiki.py "no tcgen05" --only wiki` returns zero hits outside a line within the explicitly-marked "Pre-3.6 historical context" subsection.

- AC-2: Version-sensitive claims are inventoried and validated across the agreed surfaces, by hybrid-registry lookup (not by YAML-comment scanning).
  - Positive Tests:
    - For every in-scope file (see Allowed Choices), every passage matching the registry's enumerated claim signatures has BOTH a per-page `version_sensitive: <id>` pointer (in frontmatter for wiki/reference pages; via `applies_to:` listing in the registry for the inclusion-policy scalars) AND a corresponding `data/version-claims.yaml` entry naming `id`, `tool`, `claim_valid_for` (semver range), `last_verified_release`, `last_verified_at`, `applies_to` (list of file paths or YAML JSON-pointers), and at least one supporting `source_id`.
    - The validator surface in this round is exactly: `wiki/**/*.md`, `references/primer.md`, `references/examples.md`, and the YAML scalar fields `data/inclusion-policy.yaml::cute-dsl.description` and `data/inclusion-policy.yaml::triton.description`. The validator parses inclusion-policy as YAML data and checks the `description:` scalar text — it does NOT scan YAML comments.
    - `python3 scripts/validate.py` exits 0 against the converged repo.
  - Negative Tests:
    - The validator fails when a flagged claim-bearing file in scope lacks a `version_sensitive: <id>` pointer (per-page side of the hybrid registry).
    - The validator fails when a per-page pointer's `<id>` does not resolve to a `data/version-claims.yaml` entry (forward direction of bidirectional check).
    - The validator fails when a registry entry's `applies_to` lists a path/pointer that does not exist or that lacks the corresponding per-page pointer (reverse direction of bidirectional check).
    - The validator fails when a registry entry references a `source_id` that does not exist as a `sources/**/*.md` frontmatter id.
    - The validator fails when the YAML scalar `data/inclusion-policy.yaml::triton.description` contains the literal substring "no direct tcgen05/TMEM access" (case-insensitive).

- AC-3: Candidate-ledger schema is normalized.
  - Positive Tests:
    - Every `candidates/*.yaml` (existing five plus any newly added under DEC-2) contains the same top-level fields: `repo`, `searched_at`, `keywords_used`, `total_candidates`, `included`, `excluded`, `deferred`, `prs`. Each entry under `prs[*]` contains `number`, `title`, `date`, `decision`, `reason`.
    - `python3 scripts/validate.py` includes a ledger-shape check that enforces this schema and verifies row-count vs summary-count consistency.
  - Negative Tests:
    - The validator fails on `candidates/flashinfer.yaml` until its top-level fields match the other four ledgers.
    - The validator fails when `total_candidates != included + excluded + deferred` for any ledger.

- AC-4: PR-page generation no longer hardcodes `captured_at` and the generator records its OWN skip reasons deterministically.
  - Positive Tests:
    - `scripts/generate-pr-pages.py` derives `captured_at` from the current run date (or an explicit `--captured-at YYYY-MM-DD` flag) and writes that exact date into every newly generated `sources/prs/<repo>/PR-<N>.md`. The validator parses the `captured_at` frontmatter field (not a literal-string check) and verifies it falls on or after `data/refresh-cutoff.yaml::cutoff_date`.
    - For every ledger row with `decision: include` that does not produce a PR page within `generate-pr-pages.py`'s control flow, the generator appends a row to `data/pr-page-skipped.yaml` with `pr_id`, `repo`, `pr_number`, `stage` (one of `pre-fetch`, `is-kernel-related`), `reason`, and `recorded_at`. (Empty-bundle / file-allowlist filtering happens inside `scripts/fetch_pr_diff.py` and is OUT OF SCOPE for AC-4 — those PRs do still produce a `sources/prs/.../PR-N.md` page.)
    - `data/pr-page-skipped.yaml` is byte-stable across re-runs: re-running the generator on identical inputs produces the same file (sort order: by `repo`, then `pr_number`).
    - `python3 scripts/validate.py` includes a check that every ledger `include` row appears either as a generated PR page or as a row in `pr-page-skipped.yaml`.
  - Negative Tests:
    - The validator fails when a freshly-regenerated PR page's parsed `captured_at` field is older than `data/refresh-cutoff.yaml::cutoff_date`.
    - The validator fails when an `include` ledger row appears in neither `sources/prs/.../PR-N.md` nor `data/pr-page-skipped.yaml`.

- AC-5: Existing five ledgers are re-searched up to a refresh cutoff, with deterministic checked-in evidence.
  - Positive Tests:
    - `data/refresh-cutoff.yaml` exists with a `cutoff_date` field set to the round's refresh date. (Mandatory artifact; not a user decision.)
    - `data/refresh-search-results.yaml` exists, listing per-repo: `repo_slug`, `searched_at`, `cutoff_date_used`, `pr_numbers_seen` (sorted list of PR numbers returned by the search query within the window), `last_pr_date_seen`. The file is regenerated by `scripts/refresh_candidate_ledger.py` and is byte-stable for identical inputs.
    - Each of `candidates/{cutlass,sglang,vllm,flashinfer,pytorch}.yaml` has `searched_at == data/refresh-cutoff.yaml::cutoff_date`.
    - For each repo, the ledger's `prs` array is a superset of the previous-round ledger plus any new rows whose `number` appears in `data/refresh-search-results.yaml::<repo_slug>.pr_numbers_seen` (no rows lost).
    - The FlashInfer Jan→Apr coverage gap is closed deterministically: every `prs[*]` row in `candidates/flashinfer.yaml` with `decision: include` AND `date > 2026-01-22` either has a corresponding `sources/prs/flashinfer/PR-<N>.md` or a row in `data/pr-page-skipped.yaml`.
  - Negative Tests:
    - Validator fails if any of the five ledgers' `searched_at` is not exactly equal to `data/refresh-cutoff.yaml::cutoff_date`.
    - Validator fails if the per-repo PR-number set in `data/refresh-search-results.yaml` is not a strict subset of the ledger's `prs[*].number` set after refresh.
    - Validator fails if any FlashInfer `include` row with `date > 2026-01-22` appears in neither generated PR page nor skip audit.

- AC-6: New tracked upstreams are integrated into the discovery surface, with each repo using only existing source page types. Per DEC-2: DeepGEMM full-ledger; FlashMLA case-study; FA-4 case-study.
  - Positive Tests:
    - **DeepGEMM (full-ledger)**: `candidates/deepgemm.yaml` exists (passing AC-3 schema), at least one `sources/prs/deepgemm/PR-<N>.md` page is generated for a merge in the last 12 months, and `wiki/kernels/deepgemm.md::sources` lists at least one of those new PR source-ids that was not present in the pre-refresh git revision.
    - **FlashMLA (case-study)**: `sources/blogs/flashmla.md::retrieved_at` is updated to today's date; `artifacts/blogs/flashmla/code/PROVENANCE.yaml::upstream_sha` is updated to a current `deepseek-ai/FlashMLA` commit; `wiki/kernels/flashmla.md::sources` and `wiki/kernels/sparse-mla.md::sources` reference the refreshed `blog-flashmla` source-id; no new source page type is introduced.
    - **FA-4 (case-study)**: `sources/blogs/flash-attention-4.md::retrieved_at` is updated to today's date; `artifacts/blogs/flash-attention-4/code/PROVENANCE.yaml::upstream_sha` is updated to a current `Dao-AILab/flash-attention` commit; `wiki/kernels/flash-attention-4.md::sources` references the refreshed `blog-flash-attention-4` source-id; no new source page type is introduced.
    - Smoke queries (each documented in the plan; each must return ≥ 1 row against the converged repo):
      - DeepGEMM: `python3 scripts/query.py --repo deepgemm`
      - FlashMLA: `python3 scripts/query.py --tag mla --type kernel` followed by `python3 scripts/get_page.py kernel-flashmla --follow-sources` (the latter must surface the refreshed `blog-flashmla` content).
      - FA-4: `python3 scripts/query.py --tag flash-attention --type kernel` followed by `python3 scripts/get_page.py kernel-flash-attention-4 --follow-sources` (the latter must surface the refreshed `blog-flash-attention-4` content).
  - Negative Tests:
    - Validator fails if `references/primer.md` repository table lists DeepGEMM but no `sources/prs/deepgemm/PR-*.md` exists.
    - Validator fails if `wiki/kernels/deepgemm.md::sources` does not contain at least one source-id that was not present in the pre-refresh git revision (verified via the round's commit message listing the new source-ids).
    - Validator fails if FlashMLA or FA-4 case-study refresh did not bump `retrieved_at` and `upstream_sha` (both must change relative to pre-refresh git revision).
    - Validator fails if any path of the form `sources/upstreams/**/*.md` exists (this round explicitly does not introduce that source type).

- AC-7: ~~Artifact-bundle integrity and upstream HEAD drift are separated by SCHEMA, not by motive.~~ **OUT OF SCOPE this round** per DEC-5. Recorded as deferred intent for the next refresh round. The existing `verify_verbatim.py` semantics (pinned-SHA-256 against pinned upstream ref) remain unchanged. DeepGEMM's bundle is re-pinned to a current `upstream_sha` via task17 using the existing schema; no `upstream_drift_*` fields are introduced.
  - Deferred-intent note (for the next round): introduce three optional PROVENANCE fields (`upstream_head_sha`, `upstream_drift_commits`, `upstream_drift_checked_at`) with an explicit allowlist, plus `scripts/check_bundle_drift.py` and a static-analysis grep test that confirms the script never writes `upstream_sha`.

- AC-8: Freshness checking is deterministic and offline.
  - Positive Tests:
    - `scripts/check_version_freshness.py` reads only checked-in inputs (`data/tool-versions.yaml`, the version-claim registry, ledger `searched_at` fields, `data/refresh-cutoff.yaml`) and exits with a structured report (warning rows + advisory exit code 0 by default).
    - The script imports neither `urllib`, `requests`, nor invokes `gh` / `git fetch` / network commands. A static-analysis test (grep) confirms this.
    - The freshness check is advisory-only by default; `--strict` converts warnings to failures for opt-in CI.
  - Negative Tests:
    - Static-analysis test fails (and CI rejects the change) if `scripts/check_version_freshness.py` adds any of `urllib`, `requests`, `subprocess.*gh`, `subprocess.*git fetch`, or similar network calls.
    - The check fails when invoked with `--strict` and any tool version entry is older than the configured staleness threshold.

- AC-9: Historical plan documents have an explicit supersession mechanism. Per DEC-7: per-file header.
  - Positive Tests:
    - `plan-phase2.md` and `plan-phase3.md` each begin (within their first three lines, before any other content) with a single standardized line of the form `> Superseded by [plan-phase4.md](plan-phase4.md).` (or equivalent markdown). The header line is the only edit allowed to those files this round.
    - The freshness checker (`scripts/check_version_freshness.py`) detects the supersession line by parsing it and suppresses freshness warnings for those files.
  - Negative Tests:
    - Validator fails if `plan-phase2.md` or `plan-phase3.md` lacks the supersession header line in its first three lines.
    - Validator fails if a `references/supersession.md` file is created (DEC-7 chose the per-file mechanism; the index file must not exist).
    - Validator fails if any line of `plan-phase2.md` or `plan-phase3.md` body content (i.e., past the header line) is changed relative to the pre-refresh git revision.

- AC-10: Auto-generated indices are regenerated and consistent with discoverability fixtures.
  - Positive Tests:
    - `python3 scripts/generate-indices.py` exits 0 and produces no diff against the committed `queries/*.md` after refresh.
    - Repository coverage tables in `references/primer.md` report PR counts that match `find sources/prs/<repo>/ -name "PR-*.md" | wc -l` for each repo (verified by validator), and the table now includes a row for `deepseek-ai/DeepGEMM`.
    - The three fixture queries documented in AC-6 (DeepGEMM `--repo deepgemm`, FlashMLA `--tag mla --type kernel` + `get_page --follow-sources`, FA-4 `--tag flash-attention --type kernel` + `get_page --follow-sources`) each return ≥ 1 row against the converged repo.
  - Negative Tests:
    - Validator fails when committed `queries/*.md` does not match `generate-indices.py` output.
    - Validator fails when `references/primer.md`'s repo table PR count disagrees with the actual file count, or when the table omits the DeepGEMM row.
    - Acceptance fails when any of the three AC-6 fixture queries returns zero rows.

- AC-11: Inclusion-policy rationale is rewritten on a current premise (validated by parsed YAML, not comment scanning).
  - Positive Tests:
    - The YAML scalar `data/inclusion-policy.yaml::triton.description` no longer contains the substring "no direct tcgen05/TMEM access" (case-insensitive). Its replacement text either widens the lane (with revised capture criteria reflected in `triton.capture_criteria`) or names a current, factually accurate reason for keeping it narrow (e.g., compiler maturity for cluster-scope kernels, downstream-evidence threshold). The chosen rationale matches the evidence in `data/triton-3.6-evidence.md`.
    - Re-running `python3 scripts/compute_core_prs.py` reproduces `data/core-prs.yaml` byte-for-byte against the existing inputs (no accidental shift in the deterministic checksum from a malformed lane edit).
  - Negative Tests:
    - Validator fails when the parsed `triton.description` YAML scalar still contains the obsolete phrase.
    - Validator fails when `data/core-prs.yaml::checksum_sha256` no longer matches the regenerated content.

## Path Boundaries

> Upper and lower bounds collapsed to the same point after Phase 6 user decisions (DEC-1..5, DEC-7). Both bounds describe the same implementation; the wider/narrower options are recorded in `## Pending User Decisions` for audit.

### Upper Bound (Maximum Acceptable Scope) — converged

The implementation lands AC-1 through AC-6 and AC-8 through AC-11 (AC-7 deferred per DEC-5) with these chosen values:
- DEC-1 hybrid registry: every claim-bearing page in scope carries a `version_sensitive: <id>` pointer in frontmatter; full claim metadata lives in `data/version-claims.yaml`. The validator enforces bidirectional consistency.
- DEC-2 mixed: DeepGEMM full-ledger; FlashMLA case-study; FA-4 case-study. No new source page types.
- DEC-3: `triton-lang/triton` is NOT tracked. Triton 3.6 evidence anchored via release-notes doc page + downstream-repo PR.
- DEC-4 mixed: `confidence: verified` pages cite stable CUTLASS only; lower-confidence pages may cite `4.5-dev` with a `version_sensitive` block pinning a dev SHA.
- DEC-5: AC-7 deferred. `scripts/check_bundle_drift.py` not shipped; no `upstream_drift_*` PROVENANCE fields.
- DEC-7: Per-file "Superseded by" headers on `plan-phase2.md` and `plan-phase3.md`. No `references/supersession.md` index.
- Validator surface: `wiki/**/*.md`, `references/primer.md`, `references/examples.md`, and parsed YAML scalars `data/inclusion-policy.yaml::{cute-dsl,triton}.description`. `README.md`, `SKILL.md`, `index.md`, `plan-phase*.md` body, and `draft.md` remain out of validator scope (freshness checker treats them as advisory).
- Skip-audit `data/pr-page-skipped.yaml` is byte-stable across re-runs and `validate.py`-enforced.
- `data/refresh-cutoff.yaml` and `data/refresh-search-results.yaml` are mandatory artifacts produced by `scripts/refresh_candidate_ledger.py` (task14).

### Lower Bound (Minimum Acceptable Scope) — same as Upper Bound

After user decisions, the lower bound matches the upper bound. The earlier wider/narrower options remain documented in the Pending User Decisions section for audit but no longer represent a live choice for this round. Any further reduction would constitute a re-opened decision (DEC-N rollback) rather than a path inside this plan.

### Allowed Choices

After Phase 6 user decisions, most "Can use … or …" clauses collapse to a single chosen path. The list below records the post-decision constraints; the Pending User Decisions section retains the original options for audit.

- Must use: hybrid `version_sensitive` registry per DEC-1 — per-page frontmatter `version_sensitive: <id>` pointer **and** central `data/version-claims.yaml` registry. Both surfaces required.
- Must use: DEC-2 mixed ingestion modes — DeepGEMM full-ledger; FlashMLA case-study; FA-4 case-study. No new source page types.
- Must use: a checked-in `data/tool-versions.yaml` snapshot for freshness checks (deterministic, offline).
- Can use: extending `scripts/validate.py` directly OR adding the new `scripts/check_version_freshness.py` invoked by `validate.py`. Implementer's choice; both produce the same enforcement contract.
- Can use: any kernel-allowlist update needed for `deep_gemm/include/**` (or other DeepGEMM-canonical paths) in `scripts/fetch_pr_diff.py` and `scripts/generate-pr-pages.py`.
- Validator surface scope (AC-2 / AC-9 / AC-11): exactly `wiki/**/*.md`, `references/primer.md`, `references/examples.md`, and parsed YAML scalars `data/inclusion-policy.yaml::{cute-dsl,triton}.description`. `README.md`, `SKILL.md`, `index.md`, `plan-phase*.md` body, `draft.md`, and any `*.md` outside the listed paths are explicitly OUT OF SCOPE for validator-enforced version-claim checking in this round.
- Must use (DEC-4 mixed): `confidence: verified` wiki pages cite stable CUTLASS releases only; `confidence: source-reported` and `confidence: experimental` pages may cite `4.5-dev` only when accompanied by a `version_sensitive` block whose registry entry pins a specific dev SHA.
- Must use (DEC-7): per-file "Superseded by" headers on `plan-phase2.md` and `plan-phase3.md`. No `references/supersession.md` index.
- Cannot use: regex-based version-claim auto-detection as the **sole** trigger for AC-2 (must be backed by an explicit hybrid-registry entry per claim).
- Cannot use: live network queries inside `scripts/validate.py` or `scripts/check_version_freshness.py` (enforced by static-analysis grep test in CI).
- Cannot use: overwriting `upstream_sha` in any `PROVENANCE.yaml`. AC-7 (drift schema fields + drift checker) is deferred per DEC-5; DeepGEMM bundle re-pin uses the existing schema only.
- Cannot use: editing `plan-phase2.md` or `plan-phase3.md` body text. The single supersession header line at the top is the only allowed edit.
- Cannot use: claiming Triton 3.6 native `tcgen05`/TMEM support without both an `official-doc` source and an `upstream-code` source committed under `sources/**/*.md` AND cited in `data/triton-3.6-evidence.md`.
- Cannot use: introducing a `sources/upstreams/` directory or any new `source-*` page type in this round.
- Cannot use: tracking `triton-lang/triton` as a candidate-ledger upstream this round (DEC-3 deferred; downstream-repo PR is the upstream-code anchor for Triton 3.6 evidence).

## Feasibility Hints and Suggestions

> Reference only. These describe one viable path, not a prescriptive design.

### Conceptual Approach

Three streams executed in this order:

```
Stream A (Corpus Normalization)
  A1. Fix hardcoded captured_at in generate-pr-pages.py (task1)
  A2. Normalize candidates/flashinfer.yaml shape to match the other four ledgers (task2)
  A3. Add data/tool-versions.yaml snapshot (Triton, CUTLASS, CUDA) (task3)
  A4. Define hybrid version_sensitive schema (per-page block + central data/version-claims.yaml) per DEC-1 (task4)
  A5. Add scripts/check_version_freshness.py (offline, advisory) + static-analysis grep test (task6)
  A6. Extend scripts/validate.py with ledger-shape, version-claim, skip-audit, evidence_basis, hybrid-registry consistency, and DEC-4 CUTLASS-pinning checks (task5)
  A7. Add data/pr-page-skipped.yaml audit emitter to generate-pr-pages.py (task7)
  A8. Insert per-file "Superseded by plan-phase4.md" header on plan-phase2.md and plan-phase3.md (task8)

Stream B (Content Refresh)
  B1. Codex-consult (analyze): "what is the exact Triton 3.6 lowering surface for tcgen05/TMEM?"
       Deliverable: data/triton-3.6-evidence.md memo (task9).
  B2. Add sources/docs/triton-3.6-blackwell.md (release-notes doc) + at least one downstream-repo PR source page exercising the path (task10).
  B3. Rewrite wiki/languages/triton-blackwell.md per AC-1.2 with confidence: verified + evidence_basis (task11).
  B4. Update references/primer.md, references/examples.md, and reconcile CUTLASS stamps per DEC-4 mixed policy (task12).
  B5. Rewrite the YAML scalar data/inclusion-policy.yaml::triton.description per AC-11 (task13).

Stream C (Upstream Refresh)
  C1. Add scripts/refresh_candidate_ledger.py + write data/refresh-cutoff.yaml + data/refresh-search-results.yaml (task14).
  C2. Re-run ledger refresh for existing 5 repos; commit updated yaml (task15).
  C3. Generate PR pages; record skips; close FlashInfer Jan->Apr gap (task16).
  C4. DeepGEMM full-ledger ingestion: candidates/deepgemm.yaml + sources/prs/deepgemm/PR-*.md + bundle re-pin (task17).
  C5. FlashMLA case-study refresh: blog summary + bundle re-pin (task18).
  C6. FA-4 case-study refresh: blog summary + bundle re-pin (task19).
  C7. Regenerate queries/*.md and update references/primer.md repo counts (task22).

Stream D (Validation & Sign-off)
  D1. python3 scripts/validate.py == 0 errors (task23).
  D2. Smoke queries pass (per AC-6, AC-1, AC-11).
  D3. Final Codex consult (analyze) for sign-off on Triton page accuracy and ingestion scope (task24).
```

### Relevant References

- `scripts/generate-pr-pages.py:242` — hardcoded `"captured_at": "2026-04-17"` to be parameterized.
- `candidates/flashinfer.yaml:1-2` — top-level shape lacks `searched_at`/`keywords_used` (backfilled at file end).
- `scripts/validate.py:797-829` — current validation scope (sources, wiki, artifact bundles).
- `scripts/verify_verbatim.py:154,162,287` — pinned-SHA verification logic; do not modify default semantics.
- `scripts/compute_core_prs.py:168-348` — only CuTe-DSL and Triton policy lanes; new tracked repos may need a new lane or stay outside core capture.
- `data/inclusion-policy.yaml::triton.description` — Triton-lane scalar carrying the obsolete rationale (`#`-prefixed YAML comments above the lane are NOT in validator scope).
- `wiki/languages/triton-blackwell.md:11,16,18-26,73-78` — concentrated outdated Triton claims.
- `wiki/hardware/clc.md:160` vs `wiki/languages/cute-dsl.md:15,94` — disagreeing CUTLASS minor stamps.
- `artifacts/kernels/deepgemm/full/PROVENANCE.yaml` — DeepGEMM bundle currently pinned at `7f2a703…` (`2026-04-17`).

## Dependencies and Sequence

### Milestones

1. Stream A (Corpus Normalization)
   - Phase A.1: Bug fixes — `generate-pr-pages.py` captured_at + `flashinfer.yaml` shape (task1, task2).
   - Phase A.2: Schema additions — `data/tool-versions.yaml`, hybrid `version_sensitive` schema, `wiki-language` schema extension, per-file supersession headers (task3, task4, task8).
   - Phase A.3: Validator extension + freshness checker — ledger-shape, version-claim presence, skip-audit consistency, hybrid-registry bidirectional consistency, DEC-4 CUTLASS pinning rule, offline freshness check (task5, task6, task7).
2. Stream B (Content Refresh) — depends on Stream A completing the schema decisions.
   - Phase B.1: Triton 3.6 evidence gathering (analyze task9).
   - Phase B.2: Source-page additions — release-notes doc + downstream-repo upstream-code page (task10).
   - Phase B.3: Wiki rewrites — Triton page, references, inclusion-policy `triton.description`, CUTLASS pin reconciliation (task11, task12, task13).
3. Stream C (Upstream Refresh) — runs in parallel with Stream B once Stream A's PR-page tooling is fixed.
   - Phase C.1: Ledger refresh tooling + cutoff/search-results artifacts (task14, task15).
   - Phase C.2: PR-page generation + FlashInfer gap closure (task16).
   - Phase C.3: New upstream ingestion — DeepGEMM full-ledger, FlashMLA case-study, FA-4 case-study (task17, task18, task19).
   - Phase C.4: Index regeneration (task22).
4. Stream D (Validation & Sign-off) — depends on Streams B and C completing.
   - Phase D.1: `validate.py` 0-errors + smoke queries (task23).
   - Phase D.2: Codex sign-off on Triton page accuracy and ingestion scope (task24).

Stream A's phases are strictly sequential (A.1 → A.2 → A.3). Streams B and C run concurrently after A. Stream D is the gating final stream.

## Task Breakdown

| Task ID | Description | Target AC | Tag | Depends On |
|---------|-------------|-----------|-----|------------|
| task1 | Replace hardcoded `captured_at` in `scripts/generate-pr-pages.py:242` with run-date default plus `--captured-at YYYY-MM-DD` override; add a unit-style smoke check that the output frontmatter `captured_at` is parsed as a valid ISO date and equals the override when provided. | AC-4 | coding | - |
| task2 | Normalize `candidates/flashinfer.yaml` to the same top-level schema as the other four ledgers (`repo`, `searched_at`, `keywords_used`, `total_candidates`, `included`, `excluded`, `deferred`, `prs`); preserve all existing rows; commit a small fixture `tests/fixtures/ledger-shape-{ok,bad}.yaml` for AC-3 validation tests. | AC-3 | coding | - |
| task3 | Add `data/tool-versions.yaml` snapshot listing Triton, CUTLASS, and CUDA Toolkit releases of record (release name, release date, release_notes_url, evidence_source_ids). Releases-of-record values come from upstream release notes captured via Codex consult; no live querying. | AC-2, AC-8 | coding | - |
| task4 | Define and document, in `data/schemas.yaml`, (a) the `version_sensitive` schema per DEC-1 hybrid mode — both a per-page frontmatter block (with required `id`, optional inline metadata) AND a central `data/version-claims.yaml` registry; per-page pointers reference registry entries by `id` and the validator must verify bidirectional consistency. (b) extend `wiki-language` page schema to allow `evidence_basis` as an optional field so AC-1.2 can declare verified evidence on `lang-triton`. This task only changes the schema spec — the validator code that enforces it lives in task5. | AC-1, AC-2 | coding | task3 |
| task5 | Extend `scripts/validate.py` to enforce the schemas defined in task4: (a) ledger-shape check including row-count vs summary-count consistency (AC-3), (b) version-claim presence check parsing `data/inclusion-policy.yaml` as YAML and inspecting `cute-dsl.description` and `triton.description` scalars (AC-2, AC-11), (c) PR-page skip-audit consistency check (AC-4), (d) explicit `evidence_basis` requirement enforcement when `confidence: verified` (AC-1.2), (e) hybrid-registry bidirectional consistency check (every `version_sensitive: <id>` pointer in a wiki/reference page resolves to a `data/version-claims.yaml` entry, and every registry entry's `applies_to` paths exist as page files), (f) DEC-4 CUTLASS pinning rule: any wiki page that mentions `4.5-dev` (case-insensitive substring on the parsed body, excluding code fences) must have `confidence: source-reported` or `confidence: experimental` AND a `version_sensitive` block whose registry entry pins to a specific dev SHA. | AC-1.2, AC-2, AC-3, AC-4, AC-11 | coding | task1, task2, task4 |
| task6 | Add `scripts/check_version_freshness.py` (offline, advisory-only by default; `--strict` flag for blocking mode); add a static-analysis test in CI that grep-rejects `urllib`, `requests`, `subprocess.*gh`, `subprocess.*git fetch` from the script. | AC-8 | coding | task3 |
| task7 | Add `data/pr-page-skipped.yaml` audit emitter to `scripts/generate-pr-pages.py`; sort key is `(repo, pr_number)`; only the two stages owned by this script are recorded (`pre-fetch`, `is-kernel-related`); document regen determinism. | AC-4 | coding | task1 |
| task8 | Implement DEC-7 per-file supersession headers: insert a single standardized "Superseded by [`plan-phase4.md`](plan-phase4.md)" header line at the top of `plan-phase2.md` and `plan-phase3.md` (the only edit allowed to those files this round). Ensure the freshness checker (task6) honors the supersession marker. The negative test must also verify that `references/supersession.md` does NOT exist. | AC-9 | coding | task6 |
| task9 | Codex consultation: enumerate the exact Triton 3.6 Blackwell lowering surface (`tl.dot` vs gluon vs `warp_specialize`), specific releases, supporting upstream-code commit IDs, and any caveats (cluster-scope, layout constraints, perf shape ranges). Deliverable: a checked-in `data/triton-3.6-evidence.md` memo listing each pathway with a `source_id` reference. | AC-1.1 | analyze | - |
| task10 | Add `sources/docs/triton-3.6-blackwell.md` (release-notes summary, `source_category: official-doc`) AND at least one downstream-repo PR source page that exercises the new Triton 3.6 lowering path (DEC-3 chose to defer triton-lang/triton tracking, so the upstream-code anchor MUST come from an existing tracked repo such as flashinfer/sglang/vllm/pytorch). Both source-ids are referenced from `data/triton-3.6-evidence.md`. | AC-1.1 | coding | task9 |
| task11 | Rewrite `wiki/languages/triton-blackwell.md` per AC-1.2: set `confidence: verified`, populate `evidence_basis` with one `official-doc` and one `upstream-code` entry, add a `version_sensitive` reference per DEC-1, restructure the body into "Triton 3.6+ Blackwell path" + clearly-marked "Pre-3.6 historical context" subsections. | AC-1.2 | coding | task4, task10 |
| task12 | Update `references/primer.md` Triton row, `references/examples.md` example 7 wording, and audit + reconcile CUTLASS minor-version stamps per the DEC-4 mixed policy across wiki pages (`wiki/hardware/clc.md`, `wiki/languages/cute-dsl.md`, `wiki/techniques/persistent-kernels.md`, `wiki/techniques/warp-specialization.md`, `wiki/techniques/epilogue-fusion.md`, `wiki/hardware/tmem.md`). For each touched page: (i) if `confidence: verified`, ensure all CUTLASS mentions are stable releases (4.5.x or earlier stable), (ii) if `confidence: source-reported` or `experimental` and the page mentions `4.5-dev`, attach a `version_sensitive` block pointing to a registry entry that pins a specific dev-branch SHA, (iii) any page not yet ready for either rule is added to `data/version-claims-todo.md` with a one-line deferral reason. | AC-2 | coding | task11 |
| task13 | Rewrite the Triton-lane `description:` YAML scalar in `data/inclusion-policy.yaml` per AC-11; verify `python3 scripts/compute_core_prs.py` still produces a byte-stable `data/core-prs.yaml` (`checksum_sha256` unchanged for unchanged inputs). | AC-11 | coding | task11 |
| task14 | Add `scripts/refresh_candidate_ledger.py` that (a) accepts a list of repo slugs, (b) queries `gh search prs` for each per the keywords already documented in each ledger's `keywords_used`, (c) writes `data/refresh-cutoff.yaml` with `cutoff_date`, and (d) writes `data/refresh-search-results.yaml` with per-repo `pr_numbers_seen` sorted and `last_pr_date_seen`. The script's outputs must be byte-stable for identical query inputs (i.e., results sorted; deterministic ordering documented). | AC-5 | coding | task2 |
| task15 | Run `scripts/refresh_candidate_ledger.py` for the existing five repos; commit updated `candidates/{cutlass,sglang,vllm,flashinfer,pytorch}.yaml` rows and updated `searched_at` matching `data/refresh-cutoff.yaml::cutoff_date`. | AC-5 | coding | task14 |
| task16 | Generate new PR pages for the refreshed ledgers via `scripts/generate-pr-pages.py`; record skips into `data/pr-page-skipped.yaml`; specifically close the FlashInfer Jan→Apr coverage gap by ensuring every FlashInfer `include` row with `date > 2026-01-22` is either generated or appears in the skip audit. | AC-4, AC-5 | coding | task1, task5, task7, task15 |
| task17 | DeepGEMM ingestion (DEC-2 chose **full-ledger mode**): write `candidates/deepgemm.yaml` using the same schema as the other five ledgers; update kernel-allowlist globs in `scripts/fetch_pr_diff.py` to include `deep_gemm/include/**` and any other DeepGEMM-canonical kernel paths; generate `sources/prs/deepgemm/PR-*.md` pages for the latest 12 months of merged PRs; refresh `artifacts/kernels/deepgemm/full/PROVENANCE.yaml::upstream_sha` to the current DeepGEMM HEAD commit using existing schema (no `upstream_drift_*` fields); update `wiki/kernels/deepgemm.md::sources` to include at least one new PR source-id from the new ledger. | AC-6 | coding | task5, task15 |
| task18 | FlashMLA ingestion (DEC-2 chose **case-study mode**): refresh `sources/blogs/flashmla.md::retrieved_at` to today's date; refresh `artifacts/blogs/flashmla/code/PROVENANCE.yaml::upstream_sha` (and any byte-stable per-file SHA-256 entries it tracks) to a current `deepseek-ai/FlashMLA` commit; update `wiki/kernels/flashmla.md::sources` and `wiki/kernels/sparse-mla.md::sources` to cite the refreshed source-id. No new ledger or PR pages. No new source page type. | AC-6 | coding | task17 |
| task19 | FA-4 ingestion (DEC-2 chose **case-study mode**): refresh `sources/blogs/flash-attention-4.md::retrieved_at` to today's date; refresh `artifacts/blogs/flash-attention-4/code/PROVENANCE.yaml::upstream_sha` to a current `Dao-AILab/flash-attention` commit; update `wiki/kernels/flash-attention-4.md::sources` to cite the refreshed source-id. No new ledger or PR pages. No new source page type. | AC-6 | coding | task17 |
| task20 | ~~`triton-lang/triton` tracked upstream.~~ **REMOVED** per DEC-3 = defer. Triton 3.6 evidence is anchored via task10's release-notes doc + downstream-repo PR. Recorded as deferred intent for the next refresh round. | n/a | n/a | n/a |
| task21 | ~~`scripts/check_bundle_drift.py` + drift schema fields.~~ **REMOVED** per DEC-5 = defer. AC-7 is out of scope this round; deferred-intent note sits under AC-7 above for the next refresh. | n/a | n/a | n/a |
| task22 | Regenerate `queries/*.md` via `scripts/generate-indices.py`; update repo coverage tables in `references/primer.md` for the DeepGEMM addition. | AC-10 | coding | task12, task16, task17, task18, task19 |
| task23 | Run `python3 scripts/validate.py` to 0 errors; resolve any drift; capture the run output, the deterministic fixture queries, and the static-analysis grep results for the deliberation log. | AC-1..AC-6, AC-8..AC-11 | coding | task6, task8, task12, task13, task16, task18, task19, task22 |
| task24 | Final Codex sign-off consultation: read the rewritten Triton page + ingestion-scope changes + the validator-extension diff, and confirm Triton 3.6 claims are accurately scoped to the cited evidence and ingestion mode choices match DEC-2. | AC-1, AC-6 | analyze | task23 |

## Claude-Codex Deliberation

### Agreements
- The wiki claims about Triton lacking tcgen05/TMEM access are factually wrong on Triton 3.6 and must be rewritten with explicit evidence anchoring.
- DeepGEMM, FlashMLA, and FA-4 currently have no native ingestion slot beyond a frozen blog summary, and at minimum DeepGEMM warrants first-class tracking.
- Existing five ledgers' `searched_at` is stale relative to the refresh cutoff and must be re-searched.
- `scripts/generate-pr-pages.py` hardcoding `captured_at: "2026-04-17"` is a real defect and any refresh that does not fix it produces meaningless coverage metrics.
- Artifact bundle integrity (pinned SHA-256 vs pinned upstream ref) and upstream HEAD drift are different invariants and must be tracked in distinct fields; `verify_verbatim.py` must not be modified to compare against HEAD by default.
- A regex-based version-claim auto-detector is insufficient on its own; an explicit registry (per-page or central) is required.
- Freshness checking must be deterministic and offline; live network queries belong outside `validate.py`. Enforced by static-analysis grep test.
- Reverted upstream PRs are new commits, not mutated history; bundle provenance for a merged PR's `upstream_sha` is not allowed to drift.
- **README.md and historical plan-phase\*.md are out of scope for validator-enforced version-claim checking this round** (formerly DEC-6; both reviewers agree).
- **`data/refresh-cutoff.yaml` is a mandatory artifact, not a user decision** (formerly DEC-8; both reviewers agree).
- AC-2/AC-11 must validate parsed YAML scalar fields (e.g., `triton.description`), never YAML comments — `yaml.safe_load` discards comments.
- The Triton 3.6 evidence corpus must be checked in BEFORE the Triton page rewrite consumes it (AC-1.1 strictly precedes AC-1.2).
- DEC-2 case-study mode reuses the existing `source-blog` page type. No `sources/upstreams/` directory is introduced this round.

### Resolved Disagreements

- **"10-day drift" framing** (Round 0/1): Codex flagged that `cutlass.yaml` allegedly contained PR rows dated *after* `2026-04-27`, undermining the timestamp model. Verification showed the latest `cutlass` ledger date is `2026-03-25` — no future-dated rows exist. Conclusion: Codex's specific concern was a false alarm, but the broader point that "ledger `searched_at` vs PR `date` vs run `captured_at` are three distinct timestamps and must be reconciled" is preserved as AC-3/AC-4/AC-5.
- **Validator scope expansion** (Round 1): Claude initially proposed extending validation to `references/*.md`, `README.md`, and historical plans. Codex pointed out that `validate.py` only currently covers `sources/`/`wiki/`/artifact bundles. Resolution: validator surface extends to `wiki/**`, `references/primer.md`, `references/examples.md`, and the parsed YAML scalar fields of `data/inclusion-policy.yaml` only. `README.md`, `SKILL.md`, `index.md`, historical plans, and `draft.md` are out of scope. Encoded in AC-2, AC-9, and Allowed Choices.
- **Bundle drift detection** (Round 1): Claude proposed adding `upstream_drift_commits` to existing PROVENANCE files. Codex flagged that this conflates two invariants and that the validator can only inspect final state, not motive. Resolution: drift metadata lives in distinct schema-enforced fields (`upstream_head_sha`, `upstream_drift_commits`, `upstream_drift_checked_at`); the bundle PROVENANCE schema explicitly forbids `upstream_drift_*` fields without a paired non-null `upstream_sha`; `scripts/check_bundle_drift.py` is verified by static analysis (grep) never to write `upstream_sha`. Encoded in AC-7.
- **Triton lane widening** (Round 1): Codex argued widening the Triton lane prematurely is risky if the 3.6 lowering surface is narrow (gluon-only). Resolution: AC-1.2 requires the rewrite to enumerate exact lowering surfaces (per the checked-in evidence memo), not a blanket "first-class lane" claim; the lane policy in `inclusion-policy.yaml` is rewritten on a current premise but does not mandate widening (encoded in AC-11). The widen-or-narrow decision is documented in DEC-3-related discussion and reflected in `data/triton-3.6-evidence.md`.
- **Regex auto-detector vs registry** (Round 1): Codex objected to relying on a regex trigger to detect version-sensitive claims. Resolution: AC-2 requires explicit registry entries; regex is at most a *helper* for inventory-time discovery, not the validator's source of truth.
- **AC-4 stage taxonomy** (Round 2): Codex flagged that `bundle-empty` is a `fetch_pr_diff.py` outcome, not a `generate-pr-pages.py` outcome. Resolution: AC-4 records only the two stages owned by `generate-pr-pages.py`: `pre-fetch` and `is-kernel-related`. Empty-bundle handling stays in `fetch_pr_diff.py` and is intentionally out of AC-4's scope.
- **AC-1 evidence chicken-and-egg** (Round 2): Codex noted that "validator rejects" was not yet grounded because `wiki-language` had no `evidence_basis` field. Resolution: AC-1 split into AC-1.1 (evidence corpus exists) and AC-1.2 (page rewrite consumes it); task5 extends the `wiki-language` schema to allow `evidence_basis`; the page is required to have `confidence: verified` once rewritten.
- **AC-2 / AC-11 YAML-comment validation** (Round 2): Codex noted that `yaml.safe_load` drops comments, so validating the inclusion-policy lane *comment* is not implementable. Resolution: validation operates on the YAML scalar `description:` field (a real data value), not on `#`-prefixed comment lines.
- **AC-5 deterministic upstream truth** (Round 2): Codex flagged "if upstream had merges in that window" as non-deterministic. Resolution: AC-5 anchors to checked-in `data/refresh-cutoff.yaml` and `data/refresh-search-results.yaml` produced by a new `scripts/refresh_candidate_ledger.py` (task14). The validator compares ledger state against checked-in artifacts only.
- **AC-6 case-study source type** (Round 2): Codex flagged that `sources/upstreams/<repo>.md` is not a recognized source type. Resolution: case-study mode reuses the existing `source-blog` type — refresh `sources/blogs/<slug>.md::retrieved_at` and the artifact bundle's `PROVENANCE.yaml::upstream_sha`; AC-6 explicitly forbids the `sources/upstreams/` directory.
- **AC-11 self-tests phrasing** (Round 2): Codex flagged that there is no existing self-test contract for `compute_core_prs.py` / `fetch_pr_diff.py`. Resolution: AC-11's positive test re-runs `compute_core_prs.py` and verifies `data/core-prs.yaml`'s existing `checksum_sha256` is reproduced byte-for-byte — that is the de-facto self-test today and is sufficient.
- **DAG repairs** (Round 2): task22 (queries regen) now depends on task12 (references update); task23 (final validate) now depends on task8/12/13/22; task14 added a real ledger-refresh script; task17/18/19 explicitly branch on DEC-2 mode (resolved per-repo in Phase 6: full / case-study / case-study).
- **Lower-bound honesty** (Round 2): Lower bound now includes AC-2 (Triton-only surface) so no false Triton claim survives the round, plus AC-9 supersession so the freshness checker has clean baseline. The phrase "data/upstream-coverage.yaml" was removed (it had no task or schema support).
- **DEC-6 / DEC-8 status** (Round 2): Both moved out of pending — DEC-6 (README/historical scope) is a confirmed agreement; DEC-8 (`data/refresh-cutoff.yaml`) is a mandatory artifact captured in AC-5.

### Round 2 Additional Resolutions
- **Upper Bound comment-scanning leakage** (Round 2): Codex flagged that the Upper Bound text still mentioned validating "inclusion-policy.yaml comments". Resolution: Upper Bound now says parsed YAML scalar fields `cute-dsl.description` and `triton.description` only, matching AC-2/AC-11.
- **Stream C case-study leakage** (Round 2): Codex flagged that Stream C still wrote `case-study mode: refreshed artifact bundle + sources/upstreams/<repo>.md`, contradicting AC-6. Resolution: Stream C now reads `refreshed sources/blogs/<slug>.md::retrieved_at + re-pinned artifact bundle (no new source page type)`.
- **task22 conditional DAG hole** (Round 2 → Phase 6 finalization): Codex flagged that task22 (queries regen) had no conditional dep on task20 (`triton-lang/triton` tracking under DEC-3). Round 2 resolution added "conditionally task20" to task22 and explicit conditional deps to task23. After DEC-3 = defer in Phase 6, task20 was removed from the active task list, so task22 and task23 no longer carry the conditional-task20 dep (it's resolved by removal, not by execution).
- **task4 / task5 ownership drift** (Round 2): Schema definition (task4) and validator enforcement (task5) are now explicitly separated; task4 only changes `data/schemas.yaml`, task5 implements the validator code that enforces it.
- **Terminology consistency** (Round 2): "Triton lane comment" replaced with "`data/inclusion-policy.yaml::triton.description`" in Feasibility References and Stream B.

### Convergence Status
- Final Status: **`converged`**. Round 1 closed 10 REQUIRED_CHANGES; Round 2 closed all 3 remaining REQUIRED_CHANGES and applied all 3 OPTIONAL_IMPROVEMENTS. All 6 user-facing decisions (DEC-1, DEC-2, DEC-3, DEC-4, DEC-5, DEC-7) were resolved during gen-plan Phase 6 with the user's chosen values recorded in the Pending User Decisions section. The plan is ready to proceed to RLCR loop execution starting at task1 (Stream A.1).

## Pending User Decisions

All decisions below have been resolved by the user during gen-plan Phase 6. They are retained as decision records so that downstream RLCR rounds can audit the chosen path.

- DEC-1: Version-sensitive registry placement.
  - Claude Position: Per-page `version_sensitive` frontmatter block. Locality of ownership; the page that makes the claim carries the metadata; easy to grep and edit.
  - Codex Position: Central `data/version-claims.yaml` registry with page-local pointers. Better auditability, supports cross-page consistency checks, one place to diff for the next refresh round.
  - Tradeoff Summary: Per-page is more discoverable when editing one page; central is more auditable across the wiki. Hybrid (registry + per-page pointer) gives both.
  - Decision Status: **Hybrid** — each claim-bearing page carries a `version_sensitive: <id>` pointer in frontmatter; the full claim metadata (`tool`, `claim_valid_for`, `last_verified_release`, `last_verified_at`, `source_id`s) lives in `data/version-claims.yaml`. Both task4 (schema) and task5 (validator) implement both surfaces. The validator must verify that every per-page pointer resolves to a registry entry and vice versa.

- DEC-2: New tracked upstream ingestion mode, decided per repo.
  - Claude Position: DeepGEMM full-ledger (high commit cadence, primary kernel reference); FlashMLA case-study (heavy overlap with sglang/vllm vendored copies); FA-4 case-study (heavy overlap with vllm vendored fork).
  - Codex Position: Treat all three as case-study upstreams first; promote any one to full-ledger only after a follow-up round demonstrates need.
  - Tradeoff Summary: Full-ledger gives historical PR coverage at the cost of per-round ledger maintenance. Case-study gives a single commit-pinned bundle plus a refreshed `source-blog` summary, lower cost but no intermediate-commit discoverability.
  - Decision Status: **Mixed per-repo**:
    - DeepGEMM → full-ledger mode (task17 implements `candidates/deepgemm.yaml` + `sources/prs/deepgemm/PR-*.md`).
    - FlashMLA → case-study mode (task18 refreshes `sources/blogs/flashmla.md::retrieved_at` + bundle).
    - FA-4 → case-study mode (task19 refreshes `sources/blogs/flash-attention-4.md::retrieved_at` + bundle).

- DEC-3: Track `triton-lang/triton` as a first-class upstream this round?
  - Claude Position: Defer. Anchor Triton 3.6 claims to release notes (`source-doc`) and a downstream-repo PR that exercises the new path (existing tracked repos).
  - Codex Position (Round 1): Lean toward adding now so Triton claims always cite a Triton commit-or-tag. (Round 2 softened to "default to defer is reasonable".)
  - Tradeoff Summary: Tracking triton-lang costs a new ledger + per-PR generator updates plus likely new tag-vocabulary entries. Deferring leaves Triton claims anchored to release-note text plus downstream-PR evidence.
  - Decision Status: **Defer** — `triton-lang/triton` is NOT added as a tracked upstream this round. task20 is removed from the active task list (recorded as deferred intent). task22's "conditionally task20" dependency is dropped. task10 must produce both a release-notes `sources/docs/triton-3.6-blackwell.md` page AND at least one downstream-repo PR source page that exercises the new lowering path.

- DEC-4: CUTLASS pinning — stable releases only, or `4.5-dev` acceptable?
  - Claude Position: Stable-only for `verified` confidence pages; `4.5-dev` permitted on `experimental` or `source-reported` confidence pages when accompanied by an explicit `version_sensitive` annotation pointing at a specific dev branch SHA.
  - Codex Position: Stable-only across the board; dev-branch references should be cited in `sources/blogs/*` pages rather than appearing in wiki claims.
  - Tradeoff Summary: Stable-only is more rigorous but loses access to features only available in dev. Mixed policy is more permissive but harder to audit.
  - Decision Status: **Mixed** — `wiki/**/*.md` pages with `confidence: verified` cite stable CUTLASS releases only; pages with `confidence: source-reported` or `confidence: experimental` may cite `4.5-dev` only when accompanied by a `version_sensitive` block pointing at a specific dev-branch SHA. The validator (task5) enforces this rule by cross-checking each CUTLASS-version mention against the page's `confidence` and `version_sensitive` fields.

- DEC-5: Ship `scripts/check_bundle_drift.py` this round, or defer?
  - Claude Position: Ship a minimal optional drift checker (CI-advisory).
  - Codex Position: Defer — pinned-SHA integrity already exists; AC-7 only matters if this ships.
  - Tradeoff Summary: Shipping adds one script + schema additions + a static-analysis test. Deferring keeps this round tightly scoped.
  - Decision Status: **Defer** — AC-7 is OUT OF SCOPE for this round. task21 is removed from the active task list. The new `upstream_drift_*` PROVENANCE fields are NOT added this round; bundles continue to use the existing schema (with re-pinned `upstream_sha` for DeepGEMM in task17). The original AC-7-prep sub-clause that would have added a PROVENANCE allowlist enforcement to task5 is dropped; task5's six sub-clauses (a)–(f) cover ledger shape, version-claim presence, skip-audit consistency, evidence_basis enforcement, hybrid-registry bidirectional consistency, and DEC-4 CUTLASS pinning rule (no AC-7 prep).

- DEC-7: Historical plan supersession mechanism.
  - Claude Position: A single standardized "Superseded by …" header line at the top of each historical plan.
  - Codex Position: A separate `references/supersession.md` index; preserves historical plans completely untouched.
  - Tradeoff Summary: Per-file header is more discoverable when reading the plan; index is more discoverable when scanning the wiki.
  - Decision Status: **Per-file header** — task8 implements a single standardized "Superseded by `plan-phase4.md`" header line at the top of each of `plan-phase2.md` and `plan-phase3.md`. No `references/supersession.md` index is created. AC-9's positive test checks for the header line; the negative test checks that `references/supersession.md` does NOT exist (DEC-7 chose exactly one mechanism).

## Implementation Notes

### Code Style Requirements

- Implementation code and comments must NOT contain plan-specific terminology such as "AC-", "Milestone", "Step", "Phase", or similar workflow markers — these terms exist in this plan document only, not in the resulting codebase.
- Use descriptive, domain-appropriate naming: e.g., `version_sensitive_claims.yaml`, `pr_page_skipped.yaml`, `bundle_drift_report.md`.
- Validator additions must follow the existing structure of `scripts/validate.py` (per-page-type checkers, accumulated error list, single-pass walk).
- New scripts must use the existing `scripts/_wiki_root.py` resolver so they work both inside the repo and when installed under `~/.claude/skills/`.
- Keep error messages cite-able: every validator error must include a path and a one-line reason.

### Testing & Verification

- After every coding task, run `python3 scripts/validate.py`. Stream A's tasks may temporarily fail validation (validator extensions are added before content fixes); document expected interim failures in commit messages.
- For Stream B and C tasks, run `python3 scripts/grep_wiki.py` on the affected pages to verify both positive and negative AC checks.
- For task22 / task23, run `python3 scripts/query.py` for each of the AC-6 smoke queries (DeepGEMM `--repo deepgemm`; FlashMLA `--tag mla --type kernel`; FA-4 `--tag flash-attention --type kernel`), and follow the case-study queries with `python3 scripts/get_page.py <kernel-id> --follow-sources`. Capture output for the deliberation log.
- For AC-1.2's negative test, run `python3 scripts/grep_wiki.py "no tcgen05" --only wiki` and verify zero hits outside the explicitly-marked "Pre-3.6 historical context" subsection of `wiki/languages/triton-blackwell.md`.
- For AC-8 / AC-11 negative tests, run a static-analysis grep:
  - `grep -E "(import urllib|import requests|subprocess.*gh |subprocess.*git fetch)" scripts/check_version_freshness.py` must return zero matches.
  - `grep -i "no direct tcgen05/TMEM access" data/inclusion-policy.yaml` must return zero matches against the parsed YAML scalars (re-checked via `python3 -c "import yaml; ..."`).

## Output File Convention

This plan was produced from `draft.md` via `/humanize:gen-plan` and is the main output file. No translated language variant is generated this round (`alternative_plan_language` is empty). The original draft is preserved verbatim below the plan body for reference.

--- Original Design Draft Start ---

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

--- Original Design Draft End ---
