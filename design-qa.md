# Product Design QA

- Source visual truth: `/mnt/c/Users/T/.codex/generated_images/019fa26f-688c-7353-a674-6a1ba2cd7d6e/call_neZqW7Va1nxwO4l9vcZb76JL.png`
- Implementation screenshot: `/mnt/c/Users/T/.codex/visualizations/2026/07/27/019fa26f-688c-7353-a674-6a1ba2cd7d6e/llmgen-results-skills-only.png`
- Full catalog screenshot: `/mnt/c/Users/T/.codex/visualizations/2026/07/27/019fa26f-688c-7353-a674-6a1ba2cd7d6e/llmgen-catalog-implemented.png`
- Viewport: 1440 × 1024
- State: completed single-query Greedy Autoregressive route; 10 Skill candidates,
  each with inline Code tokens; first candidate selected with inline detail
- Full-view comparison:
  `/mnt/c/Users/T/.codex/visualizations/2026/07/27/019fa26f-688c-7353-a674-6a1ba2cd7d6e/design-qa-skills-only-comparison.png`

## Findings

No actionable P0, P1, or P2 issues remain.

- Fonts and typography: the implementation preserves the reference hierarchy
  using the existing system Chinese sans-serif stack and a single monospace stack
  for Code, scores, counters, and IDs. Labels remain legible at desktop, tablet,
  and mobile widths.
- Spacing and layout rhythm: the three-pane workbench, compact top bar, numbered
  sections, continuous surfaces, and restrained dividers match the selected
  direction. The center pane is now exclusively a ranked Skill list; Query and
  standalone Code regions were removed, and each Skill carries its own Code tokens.
- Colors and visual tokens: warm ivory surfaces, near-black text, vermilion
  selection/action states, muted gray metadata, and green scores/status remain
  consistent with the reference. Contrast is sufficient in inspected states.
- Image quality and asset fidelity: the selected visual contains no photographic
  or illustrative assets. The ambiguous LG badge was replaced with a dedicated
  input-to-branches Skill Router mark and matching browser favicon.
- Copy and content: every visible method label now says
  `Greedy Autoregressive`; Beam Search, batch input, Top K controls, model metrics,
  candidate Code tokens, and Skill details remain available.
- Interaction and accessibility: keyboard focus styles, semantic buttons, tabs,
  labels, loading/empty/result states, reduced-motion handling, and Escape/keyboard
  routing behavior are present. Browser checks found no console errors or page-level
  horizontal overflow.
- Full catalog: “查看全部” opens the complete candidate space. A 1,000-Skill
  fixture rendered all 1,000 rows, built 16 first-level Code branches and 64 leaf
  nodes, searched “天气查询” to one exact result, filtered `<T1_00>` to 63 Skills,
  and opened the selected Skill detail.
- Responsive checks: 1024 × 768 and 390 × 844 captures have no page-level
  horizontal overflow; the mobile workbench collapses to a single column.

## Comparison history

1. The first candidate-focused implementation retained Query and a compact
   standalone Code block in the result pane.
2. The revised requirement removed both blocks and the raw JSON disclosure.
   Code tokens moved into each candidate row beside its domain metadata.
3. Browser verification confirms 10 candidates, two tokens per candidate,
   selected-detail linkage, no stale result blocks, no console errors, and no
   page-level horizontal overflow.

## Open questions

None.

## Implementation checklist

- [x] Candidate Skills are the primary center-pane output.
- [x] Route results contain no duplicated Query or standalone Code section.
- [x] Every candidate Skill displays its own hierarchical Code tokens.
- [x] Greedy copy is renamed to Greedy Autoregressive.
- [x] Full-page candidate directory loads the unique complete candidate set.
- [x] Hierarchical Code tree filters the same candidate collection.
- [x] Candidate and directory selections reveal original Skill details.
- [x] Desktop, tablet, mobile, console, and automated tests pass.

## Follow-up polish

- P3: if the full directory is routinely used with substantially more than
  1,000 Skills, row virtualization could reduce DOM work without changing the UI.

final result: passed

---

# Training Console Product Design QA — 2026-07-28

- Source visual truth:
  `docs/training-console/selected-design.png` (portable repository copy);
  original generation:
  `/mnt/c/Users/T/.codex/generated_images/019fa26f-688c-7353-a674-6a1ba2cd7d6e/call_ZPRBNV5UOjy2IfabR4qrGwWw.png`
- Implemented desktop capture:
  `/mnt/c/Users/T/.codex/visualizations/2026/07/27/019fa26f-688c-7353-a674-6a1ba2cd7d6e/llmgen-training-console-implemented-v3.png`
- Full-view comparison:
  `/mnt/c/Users/T/.codex/visualizations/2026/07/27/019fa26f-688c-7353-a674-6a1ba2cd7d6e/llmgen-training-console-comparison-v3.png`
- Focused center-workspace comparison:
  `/mnt/c/Users/T/.codex/visualizations/2026/07/27/019fa26f-688c-7353-a674-6a1ba2cd7d6e/llmgen-training-console-focus-comparison-v3.png`
- 1250px regression capture:
  `/mnt/c/Users/T/.codex/visualizations/2026/07/27/019fa26f-688c-7353-a674-6a1ba2cd7d6e/llmgen-training-console-1250.png`
- Mobile capture:
  `/mnt/c/Users/T/.codex/visualizations/2026/07/27/019fa26f-688c-7353-a674-6a1ba2cd7d6e/llmgen-training-console-mobile.png`
- Mutable-profile capture:
  `/mnt/c/Users/T/.codex/visualizations/2026/07/27/019fa26f-688c-7353-a674-6a1ba2cd7d6e/llmgen-training-console-mutable-config.png`
- Desktop viewport and state: 1440 × 1024, `clawhub-full-4gpu` v6,
  Retrieval selected, 4-GPU ZeRO-3 contract, persisted detached run completed.

## Findings

No actionable P0, P1, or P2 issues remain.

- Visual hierarchy: the implementation preserves the selected direction's compact
  top bar, three-pane developer workbench, warm ivory surfaces, vermilion action
  states, restrained dividers, monospace parameter metadata, and dense form rhythm.
- Configuration management: the left rail exposes multiple profile families and
  stable version slots. Every loaded version is editable; saving overwrites the
  same `vN.json` and increments the visible revision.
- Training workflow: all nine configuration views are reachable, including
  separate Memorization, Alignment, and Retrieval views. The selected stage,
  validation state, value provenance, overrides, defaults, and advanced fields
  remain visible without changing training code.
- Independence contract: the right rail shows the exact CLI boundary, resolved
  resource and output contracts, immutable snapshot path, and persisted run state.
  It offers no stop, kill, or pause control that could make the Web service own the
  training lifecycle. Runner PID, training PID, exit code, actual state-root log
  path, checkpoint, and progress are rendered from persisted metadata.
- Mutation journey: a live `mutable-live-qa` profile was created as `v1 · r1`,
  updated in place to `v1 · r2`, and retained exactly one profile file. A stale
  `r1` update returned HTTP 400 while the stored `r2` value remained unchanged.
  Chrome then edited Retrieval Epochs and saved the same slot as `v1 · r3`.
- Interaction coverage: profile/version loading, stage navigation, phase switch,
  “仅看覆盖项”, default comparison, contract/snapshot tabs, save, submit, and
  periodic run refresh were exercised.
- Responsive behavior: 1321 × 900 keeps three columns; 1250 × 900 and
  1100 × 900 move the run contract below the two-column workspace; 390 × 844 uses
  one continuous column. None has page-level horizontal overflow, and all nine
  stages and primary actions remain present.
- Browser health: desktop, tablet, mobile, and mutable-profile checks reported zero
  console errors, zero failed network requests, and no page-level overflow.
- Automated checks: the complete repository suite passed (194 tests), including
  mutable profile revisions, immutable run snapshots, validation and injection
  rejection, API snapshots,
  bounded log reads, credential filtering, package completeness, and a real
  Web-process SIGKILL survival test.

## Comparison history

1. The first browser capture established the complete three-pane layout and
   interaction states. The source and implementation were combined into one
   side-by-side image before judging fidelity.
2. The end-to-end browser journey created v5 and verified that saved configuration
   identity, provenance, and a read-only detached run remain coherent.
3. Completion audit found a 1241–1275px overflow gap and missing Runner PID /
   exit-code rows. The breakpoint moved to 1320px, the fields were added, and the
   misleading artifact `$RUN_DIR/logs` path was replaced by the real persisted
   state-root log path.
4. The final desktop capture plus 1321/1250/1100/390 browser checks found no
   remaining P0, P1, or P2 mismatch, console error, network failure, or horizontal
   overflow.
5. The final independence audit closed DNS-rebinding, unbounded-log, proxy
   redaction, prefixed-environment-secret, and credential-bearing URL edges.
   Loopback Host/Origin requests, a 1 MiB log-tail cap, 16 KiB line cap, and
   structured URL validation now have regression coverage.
6. Profile persistence changed from append-only versions to mutable version slots.
   Revision-based optimistic concurrency and immutable per-run snapshots preserve
   conflict safety and training reproducibility.

## Intentional differences

- The header reports `1 / 4` available GPUs because it reads the QA host's actual
  `nvidia-smi` result; the contract still correctly displays the configured four
  devices. The source mock's `4 / 4` is illustrative.
- Paths, profile versions, timestamps, PIDs, and progress are real persisted QA
  values rather than copied mock text.

## Follow-up polish

- P3: on phone widths the full developer console is intentionally a long vertical
  document. A future compact mode could collapse the profile library and pipeline,
  but no control is clipped or inaccessible in the current implementation.

final result: passed
