# Product Design QA

- Source visual truth: `/mnt/c/Users/T/.codex/generated_images/019fa26f-688c-7353-a674-6a1ba2cd7d6e/call_neZqW7Va1nxwO4l9vcZb76JL.png`
- Implementation screenshot: `/mnt/c/Users/T/.codex/visualizations/2026/07/27/019fa26f-688c-7353-a674-6a1ba2cd7d6e/llmgen-router-implemented.png`
- Full catalog screenshot: `/mnt/c/Users/T/.codex/visualizations/2026/07/27/019fa26f-688c-7353-a674-6a1ba2cd7d6e/llmgen-catalog-implemented.png`
- Viewport: 1440 × 1024
- State: completed single-query Greedy Autoregressive route; 10 Skill candidates,
  4 Code paths, first candidate selected with inline detail
- Full-view comparison:
  `/mnt/c/Users/T/.codex/visualizations/2026/07/27/019fa26f-688c-7353-a674-6a1ba2cd7d6e/design-qa-comparison.png`
- Focused center comparison:
  `/mnt/c/Users/T/.codex/visualizations/2026/07/27/019fa26f-688c-7353-a674-6a1ba2cd7d6e/design-qa-focus-center.png`

## Findings

No actionable P0, P1, or P2 issues remain.

- Fonts and typography: the implementation preserves the reference hierarchy
  using the existing system Chinese sans-serif stack and a single monospace stack
  for Code, scores, counters, and IDs. Labels remain legible at desktop, tablet,
  and mobile widths.
- Spacing and layout rhythm: the three-pane workbench, compact top bar, numbered
  sections, continuous surfaces, and restrained dividers match the selected
  direction. The requested product change is intentional: decoded Skill candidates
  now dominate the center pane, while Code paths occupy a compact two-row strip.
- Colors and visual tokens: warm ivory surfaces, near-black text, vermilion
  selection/action states, muted gray metadata, and green scores/status remain
  consistent with the reference. Contrast is sufficient in inspected states.
- Image quality and asset fidelity: the selected visual contains no photographic
  or illustrative assets. The existing LG text mark is preserved; the interface
  does not replace target imagery or icons with placeholder/CSS artwork.
- Copy and content: every visible method label now says
  `Greedy Autoregressive`; Beam Search, batch input, Top K controls, model metrics,
  candidate details, and raw-output access remain available.
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

1. Initial implementation placed the candidate list first but left the compact
   Code section below the first viewport (P2 discoverability).
2. The Code section was moved directly below the Query summary, compressed into
   a two-column strip, and its raw output moved into disclosure. The candidate list
   still owns the majority of the center pane.
3. Post-fix capture and focused comparison show the requested hierarchy with no
   remaining P0/P1/P2 mismatch.

## Open questions

None.

## Implementation checklist

- [x] Candidate Skills are the primary center-pane output.
- [x] Code paths remain visible in a compact supporting region.
- [x] Greedy copy is renamed to Greedy Autoregressive.
- [x] Full-page candidate directory loads the unique complete candidate set.
- [x] Hierarchical Code tree filters the same candidate collection.
- [x] Candidate and directory selections reveal original Skill details.
- [x] Desktop, tablet, mobile, console, and automated tests pass.

## Follow-up polish

- P3: if the full directory is routinely used with substantially more than
  1,000 Skills, row virtualization could reduce DOM work without changing the UI.

final result: passed
