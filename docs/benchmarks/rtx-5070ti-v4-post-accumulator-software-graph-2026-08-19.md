# RTX 5070 Ti Laptop: accumulator store後のsoftware graph再測定（2026-08-19）

## 結論

process-local `CapturedCodecDecode`の効果は、最新のCubeK accumulator pointwise storeと
重ねても維持された。5 fresh processすべてでpaired device差の中央値が負で、software graphの
session device中央値の中央値は`13.759362 ms`だった。最新PyTorch CUDA F16の同一device boundary
`13.391 ms`との差は約`0.368 ms`、`2.75%`である。

固定shape low-latency sessionではsoftware graphを引き続き推奨する。ただし本測定はlatencyの
積み上がり確認であり、VRAMは同時採取していない。VRAMについて旧campaignの131 MiB差を今回の
fresh値として流用しない。

## fresh paired結果

各processは5 warmup + 10 ABBA/BAAB block、automatic retry 0。差はsoftware graph - normal graph。

| session | graph device median | normal device median | paired block delta median |
|---:|---:|---:|---:|
| 1 | 13.735510 ms | 14.193196 ms | -0.404388 ms |
| 2 | 13.703844 ms | 14.108930 ms | -0.490372 ms |
| 3 | 13.759362 ms | 14.187267 ms | -0.445525 ms |
| 4 | 13.917934 ms | 14.259383 ms | -0.459371 ms |
| 5 | 13.857963 ms | 14.193071 ms | -0.443730 ms |

全candidate/control sampleはwaveform SHA-256
`04daa96513fe33c680bc0ca475b2182936074a4578312a76f3dfab821f49cc38`でbitwise一致し、
WGPU uncaptured errorは全processで0だった。

## pinとartifact

- source base: `70c860e`
- profiler binary SHA-256: `a6d3335aa539f85ad3e5ae35700dfcadf65047e84f4df005070b1271186e03a7`
- fixture SHA-256: `08663e52df6c63eea7ebf5ae2ba35778bdb5bc7d20cd06e430e855a86174229e`
- codec SHA-256: `b14a25da5d68bf779d8c44a40706ddc7c4819cb60489b2a80a6408ca03c514fb`
- fresh artifact: `/home/sanzentyo/benchmark-artifacts/irodori-v4-post-accumulator-software-graph-20260819-attempt1`

旧software-graph campaignのsampleはpoolしていない。process-local pipeline/graphであり、
cross-process cacheの代替ではない。source設計はWGPU共通だが、実測はNVIDIA/Vulkanのみである。
