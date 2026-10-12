# RTX 5070 Ti Laptop: F16 codec structural optimization (2026-08-13)

## 結論

F16 codec decoderの12本のdilated k=7 convolutionを、個別shape向けの手書き
packed-residue kernelからBurn/CubeCLのimplicit-GEMMへ移した。これはmaterialized im2colを
作らず、WGPU backendのCMMMA対応を利用する一般的なconvolution algorithmである。F32は既存の
packed-residue経路を維持する。

50 latent frames（2.0秒音声）の5 fresh processでは、codec device-completeの
session中央値の中央値が24.836 msから19.324 msへ22.2%短縮した。4 Euler evaluationsのRFと
codecを同一境界で足した値は56.333/58.014 msから50.378/51.646 msへ短縮した。PyTorch F16の
66.892/67.375 msに対し、device-completeで24.7%、readback-completeで23.3%高速である。

restored-cacheのNVML peakは3,069 MiBで、直前実装の3,093 MiBから増えていない。fresh
environmentの最初のautotune/compileは6,124 MiBまで使うため、service readiness前に行う。
このfresh peakはsteady/request peakへ混ぜない。

## pinと条件

- branch: `codex/v4-wgsl-fusion`
- measurement start HEAD: `3af128d9a5532e613bf4a48b71c0c887815c7af5`
- implementation commit: `489190e8539981d757a5cf967decc8b627977419`
- GPU: NVIDIA GeForce RTX 5070 Ti Laptop GPU、12,227 MiB
- driver: 595.71.05
- WGPU adapter: Vulkan discrete adapter 0
- CUDA/NVML index: 0
- PCI bus ID: `00000000:01:00.0`
- Burn: `=0.22.0-pre.2`
- CubeCL: `=0.11.0-pre.2`
- backend dispatch: WGPUのみ
- model revision: `e4aaac4df355ff560dcd35e0dae272c3a759317b`
- model SHA-256: `5863c986345d9f6d20b7d8748fee1af02079c5161cf0c9e52557da0a0c378593`
- codec revision: `47376ee24834d7a05a48ebabfe3cde29b3c5e214`
- converted codec SHA-256: `b14a25da5d68bf779d8c44a40706ddc7c4819cb60489b2a80a6408ca03c514fb`
- F16 oracle SHA-256: `08663e52df6c63eea7ebf5ae2ba35778bdb5bc7d20cd06e430e855a86174229e`
- precision: native F16 storage/compute policy、WGSLのreduction/GEMM accumulatorはF32
- PyTorch比較: TF32 off、autocast off、4 Euler evaluations、forward batches `[2,2,1,1]`、
  effective rows 6、12 layers、48 block calls

artifactは
`/home/sanzentyo/benchmark-artifacts/irodori-v4-f16-structural-opt-20260813-attempt1`
にあり、raw log、NVML CSV、環境、binary hash、fresh CubeCL environment、`summary.json`、
`SHA256SUMS`を含む。以前の`/tmp` artifactや旧campaignの時間値はpoolしていない。固定入力は
SHAを再検証して使用した。

## 計測境界

- device-complete: pre-start device syncからstageのdevice completionまで。CPU result取得を含まない。
- readback-complete: device-completeにowned contiguous F32 CPU result取得と後続syncを加える。
- first-shape: fresh processで、そのshapeのprocess-local pipeline生成がまだ残る最初のrequest。
- cache-warm: 同じprocess内でshape pipeline生成後のrequest。
- fresh-autotune: 新しいCubeCL environmentで候補選択から実施。
- restored-autotune: 同campaignのenvironmentを別processで復元。

stage profileにはCubeCL `ComputeClient::profile`を使用した。adapterがdevice timestampを提供する場合は
GPU timestamp、提供しない場合は明示的なsynchronized system-clock fallbackとしてsourceを記録する。
従来の各stage後CPU syncは順位確認には使えても加算可能なGPU時間ではなかった。新計器では旧codec
24.86 ms中、k=7の12本が16.41 ms（約66%）、pointwiseが4.08 ms、ConvTransposeが1.57 msだった。

## 実装

`CodecK7Algorithm`をADTとして追加した。

- `AccuracyApproved`: production既定。F16はimplicit-GEMM、F32はpacked-residue。
- `PackedResidue`: differential profile用。
- `CubeClImplicitGemm`: differential profile用。

通常の`decode_wgsl`は`AccuracyApproved`だけを使い、callerがpaired `Option`で経路を作る必要はない。
明示variantは`profile` featureの診断APIからのみ実行する。unsupported convolution setupは既存経路へ
fail-closed fallbackする。

implicit-GEMMは`ConvStrategy::ImplicitGemm`を用い、入力/weightを展開したim2col Tensorとして
materializeしない。biasを同じconvolutionへ渡し、後段Snakeは既存F16 WGSLを使う。F16経路では
不要になったpacked-residue weight duplicateを準備しない。F32のsource、cache、kernel selection、
出力hashは変更していない。

## 2秒条件の結果

| runtime | RF device/readback ms | codec device/readback ms | 合計 device/readback ms |
|---|---:|---:|---:|
| PyTorch CUDA F16 | 53.501 / 53.532 | 13.391 / 13.843 | 66.892 / 67.375 |
| Rust WGPU F16、変更前 | 31.497 / 32.063 | 24.836 / 25.951 | 56.333 / 58.014 |
| Rust WGPU F16、implicit-GEMM | 31.054 / 31.480 | 19.324 / 20.166 | **50.378 / 51.646** |

RustはPyTorchよりdevice-completeで24.7%、readback-completeで23.3%高速である。同じprecision、
request、RF schedule/CFG意味論だが、同じoperator graphではない。

5 processのsession中央値は次の通り。repeat 1のcompile/autotuneはsteady poolから除外した。

| session | RF device/readback ms | codec device/readback ms | NVML peak MiB |
|---:|---:|---:|---:|
| 1 fresh environment | 31.130 / 31.480 | 19.089 / 20.133 | 6,124 |
| 2 restored | 31.702 / 32.284 | 19.374 / 20.166 | 3,069 |
| 3 restored | 30.887 / 31.390 | 19.324 / 20.181 | 3,069 |
| 4 restored | 31.054 / 31.567 | 19.339 / 19.989 | 3,069 |
| 5 restored | 30.784 / 31.473 | 19.323 / 20.267 | 3,069 |

## accuracy

50-frame codec-onlyのPyTorch F16 oracle比較は次の通り。

| route | SNR dB | max abs | RMSE | cosine |
|---|---:|---:|---:|---:|
| 変更前 packed-residue | 56.3272 | 4.1504e-3 | 2.0783e-4 | 0.999998842855 |
| implicit-GEMM | 56.1892 | 3.6621e-3 | 2.1116e-4 | 0.999998807197 |

implicit-GEMMは3 warmup、5 profile、5 fresh processの各repeatで決定的だった。codec-only hashは
`fc4f2d1fa4537b186f70814b862c446283b6ff4a49c974f3a7296dd2c210c72e`。

full RF+codec waveformはSNR 31.3412 dB、max abs 5.6190e-2、cosine 0.999633110449で、hashは
`18dafe954f800b793d3682067268ab3e944e99daabccf160b73cadef976a5635`。latentは従来と同じhash
`aaa97505a73ee8b5c9816ecf62b6f1dd4cae60388e0e5167b36359ef2b1f449d`、SNR 43.2198 dB。
差はcodec accumulation order由来で、F16 hard gateを通る。85 dBはF32 numerical-reproducibility
targetとして維持し、F16音声quality gateへ流用していない。

F32 regressionはSNR 113.1972 dB、max abs 5.2601e-6、hash
`dcf32ebeb57f1213e59748c96604a04b89904ed12095c8b7e061d63e7cec1516`で従来経路を維持した。

### 六つの長さ

以前のF32 oracleはSHAを再検証した入力としてのみ使い、F16 codecを新しく実行した。これはF16
full-pipeline oracleの代用ではなく、codecの長さ汎化gateである。全長でNaN/Infなし、WGPU uncaptured
errorなし、複合F16 gateを通過した。

| seconds / frames | warm device ms | SNR dB | max abs |
|---|---:|---:|---:|
| 1.8 / 45 | 18.520 | 58.2006 | 1.5684e-3 |
| 4.48 / 112 | 54.144 | 56.8212 | 1.4814e-3 |
| 10.2 / 255 | 129.136 | 55.9492 | 1.5710e-3 |
| 13.32 / 333 | 183.511 | 56.3132 | 2.2731e-3 |
| 19.56 / 489 | 265.599 | 58.4893 | 2.2116e-3 |
| 27.4 / 685 | 390.114 | 59.1379 | 2.1756e-3 |

45-frameではwarmup 3回後にも最初の2 measured requestが約44--50 ms、その後17--19 msとなった。
GPU workの遅延compileがwarmup callのreturnより後まで残るためである。readiness warmupは単にN回enqueue
せず、各manifest shapeを実行後にdevice syncし、real validationを1回行う必要がある。489 framesの
既知accuracy問題を性能PASSで隠しておらず、このcodec gateは通ったがfull RF+codecの489-frame F16
oracle campaignは未完了として残す。

## memoryとwarmup

restored service条件では3,069 MiBで変更前3,093 MiBを24 MiB下回った。F16で不要な
packed-residue weight cacheを作らず、implicit-GEMMもmaterialized im2colを持たないため、steady VRAMを
増やしていない。ただしallocatorのreservation粒度が大きく、isolated codecのNVMLではcache削除前後が
同じ1.2 GiB bucketになる場合がある。

fresh CubeCL environmentはRF autotuneと12 shapeのcodec pipeline compileを含み6,124 MiB、RF first
23.38秒、codec first 11.54秒だった。restored processではこれらがsteadyへ落ちる。ComputePipeline objectは
process-localなので、persistent environmentだけではゼロにならない。long-lived session、manifest-driven
DryRun、real validation後のreadinessが本命である。

named environment、autotune metadata、DryRun、implicit-GEMMはWGPU/CubeCL共通でVulkan/Metal/DX12に
実装可能。ただしF16 capability、CMMMA availability、driver pipeline cache、性能選択はadapter別に検証する。
Vulkan固有`wgpu::PipelineCache`はportable production contractにせず補助最適化とする。cache directoryは
OS user cache convention下の`Irodori-TTS-burn/cubecl`、CLIでは`--cubecl-cache-dir`、環境変数では
`IRODORI_TTS_BURN_CACHE_DIR`で上書きする。

## crate ergonomics

production callerは従来通り`prepare_decoder_for_wgsl`と`decode_wgsl`を呼ぶだけでよい。dtype-awareな
`AccuracyApproved` policyがF16/F32を選ぶ。診断CLIは実行precisionとfixture native precisionを別newtypeで
受け取り、F16 executionを固定F32 oracleへ比較できる。stage receiptはdurationだけでなくtimestamp sourceを
持つため、device timeと同期host timeを誤ってpoolできない。

残る改善は、precision/profile/capability receiptを`RuntimeBuilder<Cold> -> Runtime<Warmed>`に保持し、
manifest外shapeを型付き`Unwarmed` slow pathまたは明示拒否へ送ることである。

## 次の本質的な候補

1. custom Fusion providerでimplicit-GEMMのbias後Snakeを同じepilogueへ入れ、12 dispatchと中間Tensorを削る。
2. pointwise projection + residual + 次unitのSnakeをFusion providerへ移し、現在別管理のlayout policyを
   backend bridgeへ集約する。
3. 六つのshape/B1/B2を`DryRun`し、real validation + device syncまでを`Runtime<Warmed>`の構築条件にする。
4. accuracy-approved autotune receiptにkernel/config、dtype、adapter identity、fixture hash、accuracy metricsを
   保存し、最速候補だけをcacheしない。
5. 489/685 framesを含むfull RF+codec F16 oracle、voice design/clone、all-resident/phase batchを再測定する。
6. 同一長・同一CFG topologyのtensor micro-batchを、single request latencyとは別campaignで評価する。

## 再開手順

1. branchとこのreport記載のimplementation commitを確認する。
2. artifactの`SHA256SUMS`を検証し、model/codec/fixture SHAを再確認する。
3. 新しいcampaign directoryと新しいCubeCL environmentを作る。旧時間値をpoolしない。
4. 45/112/255/333/489/685、B1/B2をDryRunし、各shapeをreal execution + device syncで検証する。
5. restored-cacheの別processを最低5回実行し、repeat 1とsteady repeatsを分離する。
6. full F16 oracleで489 accuracyを通してから、最初のSnake epilogue Fusion候補を実装する。
