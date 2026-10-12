# RTX 5070 Ti Laptop: decoder内NHWC residual state screen（2026-08-19）

## 結論

decoder block内のshortcut rawと次unit向けSnake activationをともにNHWCで保持し、pointwiseの
residual readとraw storeをchannel-contiguousにする構造をscreenした。C768はdirect pointwise
kernelのrelease contract外なのでproduction経路を維持し、C384/C192/C96だけを候補にした。

同一binary、3 fresh process、各5 warmup + 10 measuredの50-frame比較では、候補の
device-complete中央値の中央値が`17.272 ms`、production controlが`15.159 ms`で、候補は
`13.94%`遅かった。readback-completeも`18.116 ms`対`15.881 ms`で`14.07%`遅い。
全sampleのwaveform SHA-256はbitwise一致し、WGPU uncaptured errorは0だったが、性能理由で
productionには採用しない。

## 変更したdataflow

productionは各unitのraw shortcutをNCL、k7/Snakeからpointwiseへ渡すactivationをNHWCで保持する。

```text
raw shortcut NCL
Snake/k7 activation NHWC
  -> pointwise + bias + NCL residual
  -> raw NCL + next activation NHWC
```

候補はblock入口と出口以外のshortcutもNHWCにした。

```text
block entry raw NCL
  -> res0: raw NHWC + next activation NHWC
  -> res1: raw NHWC + next activation NHWC
  -> res2: next-block activation NCL または final raw NCL
```

これにより一threadが扱う4 channelのresidual accessとraw storeは連続になる。一方、現在の
pointwise workgroupは隣接する`local_id.x` laneを隣接timeへ割り当てる。NCLではそのlane間accessが
連続だが、NHWCではchannel数分離れるため、subgroup全体ではかえってcoalescingを失う。raw writeの
総量も減らない。実測はこのworkgroup mappingとNCLの組合せが有利であることを示した。

したがって次にNHWC stateを再検討するなら、layoutだけを替えるのではなく、channelを隣接laneへ
割り当てるNHWC専用pointwise mappingと一体で設計する必要がある。より直接的には、raw shortcut
そのもののwrite/readを消すproducer-consumer fusionを優先する。

## 契約とfail-closed

`PreparedNhwcResidualPair`がraw/activatedの二つのNHWC tensorを一つの型状態として保持する。
launcherはinput、residual、raw output、activated outputのlayoutをkernel IDとshader templateへ含める。
候補routeはC384/C192/C96、contiguous layout、F16/F32 dtype、same device、packed pointwise weight、
resource limitのいずれかが外れた場合に測定中のsilent fallbackをせず、configuration errorを返す。

この厳格化により、初期実装がC768でproductionへfallbackしていたことをscreen前に検出した。
profile CLIでは`--profile-repeats 0`を正式にwhole-decode-only条件として扱う。

## fresh session

| session | NHWC state device ms | production device ms | NHWC state readback ms | production readback ms |
|---:|---:|---:|---:|---:|
| 1 | 17.204 | 15.138 | 18.116 | 15.881 |
| 2 | 17.272 | 15.178 | 18.084 | 15.728 |
| 3 | 17.274 | 15.159 | 18.126 | 16.019 |
| median | **17.272** | **15.159** | **18.116** | **15.881** |

- candidate/control waveform SHA-256:
  `113ba560546d82a3112332ac67b3cea5d5b83b407109d3df3817e5b82b609e05`
- profiler binary SHA-256:
  `391abe24f11ac1e2d719ed19c41eebe80227f3343b0e8ec87024a7b866d7ce97`
- dirty patch SHA-256 at build:
  `31ee6beb8c7653b42fde60a08c0102fceef77971118034c2c162d8a4aef06fe0`

## portability

layoutをkernel templateの型付きpolicyとして表現し、shaderはWGSL F32/F16で同じdataflowを持つ。
Vulkan固有API、GPU名、device別tile値は使っていないためsource設計はMetal/DX12にも移植可能である。
ただしcompileと性能を実測したのはNVIDIA/Vulkanだけであり、他backendの性能は主張しない。

## artifact

- source base: `87401807cea05bca916841318903519776857c1d`
- `/home/sanzentyo/benchmark-artifacts/irodori-v4-nhwc-residual-state-20260819-attempt1`
- F16 oracle SHA-256:
  `08663e52df6c63eea7ebf5ae2ba35778bdb5bc7d20cd06e430e855a86174229e`
- codec SHA-256:
  `b14a25da5d68bf779d8c44a40706ddc7c4819cb60489b2a80a6408ca03c514fb`
- GPU: NVIDIA GeForce RTX 5070 Ti Laptop GPU、driver `595.71.05`、Vulkan adapter 0、
  CUDA/NVML index 0、PCI `00000000:01:00.0`、VRAM `12,227 MiB`

旧`/tmp` artifactや別campaignの測定値は今回の集計へpoolしていない。
