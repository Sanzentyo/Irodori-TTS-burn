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

- Fresh external artifact: `benchmark-artifacts/irodori-v4-wgpu-software-graph-20260819-attempt2`.
- Source pin: `7159372055f78e037370f9092cc128f38604c2ea`.
- The artifact's `SHA256SUMS` covers its copied binary, environment pins, raw logs, NVML CSV, and derived summary.
- `vulkaninfo` was not installed. The failure is retained; the actual WGPU Vulkan adapter identity is recorded in every raw session log.

## QA intent

The visible report leads with the adoption decision, preserves device/readback boundary symmetry, separates the normal-only VRAM controls, and labels portability as source-compatible rather than cross-platform validated. The graph remains opt-in because latency improves at a measurable memory cost.
