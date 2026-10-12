# RTX 5070 Ti Laptop: F16 codec layout fusion and stem policy (2026-08-13)

## 結論

50 latent frames（2.0秒、96,000 samples）のF16 codecについて、productionの
device-completeを19.172 msから15.873 msへ、readback-completeを19.753 msから16.723 msへ
短縮した。5 fresh processそれぞれ10 measured requestのsession medianを取り、そのmedianを採用値とした。
短縮率は17.2% / 15.3%である。

PyTorch CUDA F16の同一境界は13.391 / 13.843 msなので、今回は上回っていない。Rust WGPUは
device-completeで18.5%、readback-completeで20.8%遅い。operatorの意味と精度は同じだが、backendの
operator graphは同一ではない。

採用した変更はshape別の定数選択ではなく、次の三つのlayout/lifetime改善である。

1. CubeCL k7 convolutionが生成するcontiguous NHWCへSnakeを直接適用し、不要な中間layout copyを除く。
2. residual unit間のprepared activationをNHWC residentに保ち、pointwise projectionがNHWCを直接読む。
   raw residualだけは型状態どおりNCLに保つ。
3. decoder stemはF16だけBurn/CubeCL tuned convolutionを使う。F32は従来のdirect WGSLを維持する。

## fresh campaignとpin

- branch: `codex/v4-wgsl-fusion`
- campaign start HEAD: `e3889577e26b3a34f97301040dbdf115cf980188`
- adopted implementation commit: `112234feedf5a6aeb59378eaf3d382c6917b99f3`
- GPU: NVIDIA GeForce RTX 5070 Ti Laptop GPU、12,227 MiB
- driver: 595.71.05
- WGPU adapter: Vulkan discrete adapter 0
- CUDA/NVML index: 0
- PCI bus ID: `00000000:01:00.0`
- campaign開始時のavailable VRAM: 11,774 MiB
- Rust: 1.95.0 (`59807616e`)、Cargo 1.95.0
- Burn/Burn-CubeCL: `=0.22.0-pre.2`
- CubeCL: `=0.11.0-pre.2`
- production backend dispatch: WGPUのみ
- Irodori model revision: `e4aaac4df355ff560dcd35e0dae272c3a759317b`
- Irodori model SHA-256: `5863c986345d9f6d20b7d8748fee1af02079c5161cf0c9e52557da0a0c378593`
- codec revision: `47376ee24834d7a05a48ebabfe3cde29b3c5e214`
- codec input SHA-256: `db120339c5ee7eca1912cdf29bc612b947a0808e69c3cebfb4936b45a762c1d5`
- converted codec SHA-256: `b14a25da5d68bf779d8c44a40706ddc7c4819cb60489b2a80a6408ca03c514fb`
- F16 oracle SHA-256: `08663e52df6c63eea7ebf5ae2ba35778bdb5bc7d20cd06e430e855a86174229e`
- final profiler binary SHA-256: `b1945179bf0e4499a5daacaf78c27678699f4012347fd7ab5f891fa5f35c78ad`

fresh outputは
`/home/sanzentyo/benchmark-artifacts/irodori-v4-f16-codec-layout-fusion-20260813-attempt1`
である。このcampaignの比較には、このdirectoryで新しく取得したsession logだけを使った。過去の
`/tmp` artifactや旧campaignの時間値はpoolしていない。固定fixture、converted codec、PyTorch境界値は
SHAでpinした既存oracle/referenceであり、新しいRust sessionの集計には混ぜていない。

## 境界とprotocol

- device-complete: pre-start device syncからcodecのdevice completionまで。
- readback-complete: device-completeにowned contiguous F32 CPU waveform取得までを加える。
- workload: 50 latent frames、96,000 samples、native F16 storage、F32 accumulator。
- warmup/measured: 各fresh processで3 warmup + 10 measured。
- aggregation: 各process内medianを計算し、その5 session medianのmedianを採用。
- accuracy: max abs、mean abs、RMSE、SNR、cosine、NaN/Inf。
- determinism: 各sessionの全measured waveform SHA-256一致。

PyTorch F16との比較はCPU readbackを両runtimeに含めた。readbackを片方だけ含む値は比較に使っていない。

## 最終結果

| runtime / condition | device-complete ms | readback-complete ms | Rust比 |
|---|---:|---:|---:|
| PyTorch CUDA F16 | **13.391** | **13.843** | 1.000x |
| Rust WGPU F16 final | 15.873 | 16.723 | 1.185x / 1.208x |
| campaign開始時Rust WGPU F16 | 19.172 | 19.753 | 1.432x / 1.427x |

finalの5 session medianは次である。

| session | device ms | readback ms |
|---:|---:|---:|
| 1 | 15.789 | 16.588 |
| 2 | 15.873 | 16.723 |
| 3 | 15.895 | 16.890 |
| 4 | 15.885 | 16.727 |
| 5 | 15.818 | 16.723 |

全sessionのwaveform hashは
`d2ed183a5bc64e6447b1e6eb466813e8724a903ad95db8da7b742d7cbe985c06`で一致した。
F16 oracleに対する50-frame waveformはmax abs `3.173828125e-3`、RMSE
`2.222655167e-4`、SNR `55.743860 dB`、cosine `0.999998681025`、NaN/Inf 0だった。

NVMLはfinal codec processを100 ms間隔で34 samples記録した。peak usedは1,210 MiB、minimum freeは
10,565 MiBだった。これはcodec-only processの値であり、all-resident RF+codecのpeakへ読み替えては
ならない。

## 長さaccuracy

F16 executionを独立したpinned F32 oracleへ比較し、全長でduration-derived latent shapeをそのままdecodeした。
旧accuracy campaignの時間値は使わず、今回のbinaryで新しく実行した結果だけを記録する。

| audio相当 | frames | samples | SNR dB | max abs | device ms |
|---:|---:|---:|---:|---:|---:|
| 1.80 s | 45 | 86,400 | 57.273 | 1.894e-3 | 15.009 |
| 4.48 s | 112 | 215,040 | 56.436 | 1.726e-3 | 37.617 |
| 10.20 s | 255 | 489,600 | 55.769 | 3.196e-3 | 84.690 |
| 13.32 s | 333 | 639,360 | 55.992 | 4.829e-3 | 113.648 |
| 19.56 s | 489 | 938,880 | 58.209 | 2.145e-3 | 162.964 |
| 27.40 s | 685 | 1,315,200 | 59.091 | 2.105e-3 | 236.341 |

489-frameをaccuracyなしで性能PASSにはしていない。今回の489-frameはfinite、deterministicで複合accuracy
gateを通った。F32 regressionもSNR 113.1972 dB、max abs `5.2601e-6`、cosine
`0.999999999998`、uncaptured WGPU error 0で通った。

## 採用・不採用候補

### 採用: output-layout fusion

CubeCL convolutionは物理NHWCを生成する。従来はNCHWへmaterializeしてからSnakeを実行していた。
Snakeと32x32 tiled transposeを一つのkernelへまとめた段階で、19.172 msから17.759 msへ短縮した。
さらにactivationをNHWCのまま次のpointwiseへ渡すtype-stateを導入し、不要な往復を除いた。

`PreparedActivation::{Ncl, Nhwc, ResiduePacked}`と`PointwiseActivation::{Ncl, Nhwc}`が物理layoutを
明示する。fallback時だけ`Nhwc -> Ncl`をmaterializeするため、invalidなlayoutを通常の`Tensor<3>`だけで
暗黙に渡さない。

### 採用: F16 tuned stem

F16 stemの手書きshaderはscalar F32 accumulatorで0.84 msだった。Burn/CubeCL tuned convolutionは
0.317 msで、同じF16 storage/F32 accumulationのaccuracy gateを通った。productionはdtypeでpolicyを
選び、F16のみtuned route、F32は既存direct routeとした。

### 不採用

- pointwise NHWC shared-memory flush: output coalescingよりoccupancy低下が大きく、16.37 msだった。
- prepared k7 weight duplicate: 5 fresh sessionで16.556 ms対unprepared 16.499 ms、かつ約32 MiB増加。
- CubeK sync-strided: 17.289 ms。
- CubeK async-cyclic: 38.649 ms。
- CubeK async-strided: 38.974 ms。

失敗条件はretryでproduction条件へ選び直さず、各raw logへ保存した。

## crate ergonomics

production callerにshape別algorithm knobは追加していない。診断用`CodecAlgorithmPlan`は`profile` feature内で
`stem`、`k7`、`pointwise`を必須fieldとして持ち、paired `Option`を避ける。productionは
`AccuracyApproved`だけを使う。

pointwise launcherはinput/output layoutをenumで保持し、kernel cache IDにも含める。SourceTemplateの
index式と物理shape/stride preflightは同じenumから生成される。unsupported dtype、shape、device limit、
layoutはdispatch前にfail closedし、portable pathへ戻る。

## なぜここで大幅短縮候補が枯渇したか

残る支配項は12本のk7 convolutionで、現行は概念的に次の二dispatchである。

```text
CubeK implicit-GEMM convolution + bias -> F16 NHWC intermediate
F32 Snake activation                  -> F16 NHWC output
```

本質的な次候補はCubeKのmatrix accumulator drainへSnakeをepilogueとして入れ、中間write/readと12 dispatchを
同時に消すことである。しかしCubeK 0.3.0-pre.2の公開convolution launcherはconcrete `TensorOutput`を固定し、
callerが`GlobalWriter`/epilogueを差し替えるhookを公開していない。実現には依存crateへ汎用custom-epilogue
抽象化を追加するか、Burn custom Fusion/backend extension境界を先に作る必要がある。Irodori shapeだけを
複製した巨大shaderはad hocなので採用しない。

既存のgeneric algorithm family、weight preparation、layout fusion、stem policy、pointwise algorithmは今回と
直前campaignで実測済みである。このため、依存crateのAPI設計を変えずに試せる非ad-hoc候補はここで
枯渇したと判断した。

## 次回の優先順位と再開手順

1. 新campaignを作り、旧時間値をpoolせず50-frame production baselineを再取得する。
2. CubeKへ`GlobalWriter`を包む汎用unary epilogue interfaceを提案する。bias適用後のF32 accumulatorへ
   parameterized Snakeを適用でき、通常writerへゼロコストfallbackできる設計にする。
3. `K7BiasSnake`をbackend extension ADTとして定義し、portable `conv + Snake`とWGPU fused writerを同じ
   operation contractにする。
4. 50-frameで2%以上短縮した場合だけ、45/112/255/333/489/685、5 fresh session、NVMLへ昇格する。
5. 489-frame accuracy、F32 regression、WGPU error 0を必須gateにする。
6. custom epilogue後もPyTorch 13.391 / 13.843 msを上回らなければ、pointwiseにも同じwriter抽象化を適用する。
