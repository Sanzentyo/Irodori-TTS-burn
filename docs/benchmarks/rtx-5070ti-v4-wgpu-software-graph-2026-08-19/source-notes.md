# Source and chart notes

## Chart map

- Section: paired latency evidence.
- Question: whether process-local software graph replay reduces equal-boundary codec latency consistently across fresh processes.
- Family/type: comparison, grouped signed bar.
- Grain: one median delta per fresh process and boundary; five processes, ten paired blocks per process.
- Fields: `session`, `boundary`, `delta_ms`, `blocks`.
- Takeaway supported: all process-level median device and readback deltas are negative; magnitude varies under laptop clock and power noise.
- Palette: hard two-root cap (`blue-orange`) plus a neutral dashed zero reference; boundary is also named in the legend and exact table.
- Delivery: canonical `artifact.json` packaged to `report.html` by the Data Analytics portable builder.

No additional chart is included. Five paired session estimates are enough for a signed comparison but too sparse for a distribution or trend claim; exact latency and VRAM values are retained in the report table. The report does not pool older campaigns.

## Evidence locations

- Fresh external artifact: `benchmark-artifacts/irodori-v4-wgpu-software-graph-packed-20260819-attempt1`.
- Source pin: `4d8b573453e1530d8009847c1b5b71c71ec3f627`.
- The artifact's `SHA256SUMS` covers its copied binary, environment pins, raw logs, NVML CSV, and derived summary.
- `vulkaninfo` was not installed. The failure is retained; the actual WGPU Vulkan adapter identity is recorded in every raw session log.

The original exclusive-page campaign remains historical context only. Its
timings are not pooled with the packed-arena estimator. The old and new NVML
peak medians are compared descriptively because their fixture, process shape,
sampling interval, and adapter are pinned identically, but this is not a paired
cross-campaign estimate.

## Technical report structure mapping

- Technical summary: `Decision` and the headline metric strip.
- Key finding and visual evidence: `Packed reuse improves latency without the original arena overhead`, the grouped signed bar, and the exact session table.
- Scope and metric definitions: chart subtitle, card descriptions, source query filters, and the visible runtime/API contract.
- Methodology: source query metadata plus the reproduction section; ABBA/BAAB blocks and independent NVML controls are kept distinct.
- Limitations and robustness: correctness, portability, fixed-shape, and cross-campaign caveats remain adjacent to the affected claims.
- Recommended next steps: capture the final consumer transform, bound shape residency, and add non-Vulkan smoke coverage.
- Further question: whether a multi-shape service should retain one graph, use an LRU, or rebuild on shape transition. This is folded into the limitations/next-work section because there is not yet comparative evidence for a separate section.

## QA intent

The visible report leads with the adoption decision, preserves device/readback boundary symmetry, separates the normal-only VRAM controls, and labels portability as source-compatible rather than cross-platform validated. The graph remains opt-in because latency improves at a measurable memory cost.

The portable builder passed artifact validation, package generation, exact
payload equality, semantic fallback checks, and structural verification on
2026-08-19. No compatible installed Chromium headless shell was available, so
enhanced-reader desktop/narrow viewport and source-dialog interaction QA remain
`structural_only`; the self-contained semantic chart table remains available.
