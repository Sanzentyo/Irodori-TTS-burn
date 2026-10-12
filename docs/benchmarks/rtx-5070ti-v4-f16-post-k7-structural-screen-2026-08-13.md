# RTX 5070 Ti Laptop: F16 codec post-k7 structural screen (2026-08-13)

## 結論

50 latent frames（2.0秒音声）のF16 codecについて、前cycleで採用したCubeCL
implicit-GEMM `SimpleSyncCyclic`の次に、pointwise convolutionのalgorithm置換と、k=7
convolutionの残るCubeCL algorithm familyを同一境界でscreenした。全候補はaccuracy gateを
通ったが、device-completeで現行より7.2%から94.4%遅かった。したがってproduction経路は変更せず、
F16 k=7は`ConvStrategy::ImplicitGemm`のautotune/inferred `SimpleSyncCyclic`、pointwiseは既存の
packed matmul + fused residual/Snake finalizerを維持する。

この結果から、次の本質的な短縮はalgorithm enumをshapeごとに固定することではない。現在のraw
`CubeTensor` launcher境界をbackend bridgeへ集約し、k=7 convolutionとSnake、またはpointwise
projectionとresidual/Snakeを一つのFusion operation/epilogueとして表現して、中間Tensorとdispatchを
同時に消す必要がある。現状のmodel codeではcustom WGSLを実行するためにBurn Fusion graphから
primitiveを取り出しており、その境界を残したままcustom Fusion providerを登録しても二つのoperationを
一つのsegmentとして認識できない。

## fresh campaignとpin

- branch: `codex/v4-wgsl-fusion`
- measurement start HEAD: `4e14338b27a8bcd514cb2684a00253a46dc0bfee`
- diagnostic comparison commit: `78fb7cb`
- GPU: NVIDIA GeForce RTX 5070 Ti Laptop GPU、12,227 MiB
- driver: 595.71.05
- WGPU adapter: Vulkan discrete adapter 0
- CUDA/NVML index: 0
- PCI bus ID: `00000000:01:00.0`
- Burn: `=0.22.0-pre.2`
- CubeCL: `=0.11.0-pre.2`
- backend dispatch: WGPUのみ
- model revision: `e4aaac4df355ff560dcd35e0dae272c3a759317b`
- codec revision: `47376ee24834d7a05a48ebabfe3cde29b3c5e214`
- converted codec SHA-256:
  `b14a25da5d68bf779d8c44a40706ddc7c4819cb60489b2a80a6408ca03c514fb`
- F16 oracle SHA-256:
  `08663e52df6c63eea7ebf5ae2ba35778bdb5bc7d20cd06e430e855a86174229e`
- profiler binary SHA-256:
  `a3c717ef2ab8e63910b5eb2e58fe2cec23dd54efd5d42e74d25c78859464c08e`

fresh outputは
`/home/sanzentyo/benchmark-artifacts/irodori-v4-f16-post-k7-fusion-20260813-attempt1`
に置いた。raw log、環境、入力/binary hash、`summary.json`、`SHA256SUMS`を含む。以前の`/tmp`
artifactや旧campaignの時間値はpoolしていない。固定fixtureとconverted codecはSHAを再検証した。

## 境界とprotocol

- device-complete: pre-start device syncからcodecのdevice completionまで。
- readback-complete: device-completeにowned contiguous F32 CPU waveform取得までを加える。
- accuracy: 同じPyTorch F16 codec oracleに対するSNR、max abs、RMSE、cosine、NaN/Inf。
- determinism: 3 warmupと全measured repeatのwaveform SHA-256一致。
- workload: 50 latent frames、96,000 samples、native F16 storage、F32 accumulator。

各candidateは独立logとして保存し、OOMやaccuracy failureをretryで別条件へ選び直していない。
今回はどの候補も速度screenで不採用になったため、5 fresh session、六つの長さ、NVML比較へは
昇格させなかった。production codeとmemory lifetimeを変更していないため、前cycleで確定した
restored all-pipeline NVML 3,069 MiBも更新値として再利用せず、単にproduction不変のreferenceとする。

## 結果

| algorithm plan | device ms | readback ms | device差 | SNR dB | hash |
|---|---:|---:|---:|---:|---|
| production: k7 sync-cyclic + packed pointwise | **19.0147** | **19.7407** | baseline | 56.1892 | `fc4f2d1f…` |
| pointwise CubeCL implicit-GEMM | 21.3241 | 22.5291 | +12.1% | 56.4487 | `672ddbb1…` |
| k7 sync-strided | 20.3845 | 21.2745 | +7.2% | 56.1892 | `fc4f2d1f…` |
| k7 async-cyclic | 36.9638 | 37.9176 | +94.4% | 56.3272 | `eef3a021…` |
| k7 async-strided | 36.8132 | 37.5278 | +93.6% | 56.3272 | `eef3a021…` |

baselineと各candidateのdevice/readbackは、それぞれ同じprocessの10 measured requestの中央値である。
stage profilerはGPU device timestampを使用した。pointwise implicit-GEMMは長いC192/C96 projectionで
既存packed pathより遅く、dispatchを減らさずlayout conversionを増やすため総時間も悪化した。
sync-stridedは正確かつ決定的だったが、同じoperator familyのsync-cyclicを上回らなかった。async系は
このadapter/shapeで約2倍になった。

最終sourceでproduction defaultを再実行し、F16は19.4252 / 20.3384 ms、hash
`fc4f2d1fa4537b186f70814b862c446283b6ff4a49c974f3a7296dd2c210c72e`、SNR 56.1892 dB、
uncaptured WGPU error 0だった。campaign baselineとの差は通常のprocess変動内で、candidateの採用はない。
F32 regressionは35.1083 / 36.1392 ms、SNR 113.1972 dB、max abs 5.2601e-6、hash
`dcf32ebeb57f1213e59748c96604a04b89904ed12095c8b7e061d63e7cec1516`、uncaptured error 0だった。

## 実装範囲とcrate ergonomics

reject済み候補をproduction APIへ露出させないため、比較基盤は`profile` feature内に閉じた。
`CodecAlgorithmPlan { k7, pointwise }`は一回のdifferential runの二つのpolicyを必須fieldとして保持し、
paired `Option`を使わない。productionの`decode_wgsl`は従来通りaccuracy-approved policyだけを使う。
明示candidateは`profile_codec_decoder`からdevice timestamp条件でのみ選べ、unsupported setupは既存経路へ
fail-closed fallbackする。通常の`inference,codec` buildにはcubekの低水準比較依存やcandidate APIが入らない。

この比較基盤を残す理由は、今後CubeCL/Burnを更新した際に同じalgorithm familyを再評価できること、
そしてcustom epilogue実装前後を同じ境界で比較できることにある。runtime callerへshape-specific knobを
増やすためではない。

## なぜ直ちにcustom Fusion providerへしなかったか

現在のk=7経路は概念的に次である。

```text
Burn Tensor
  -> raw CubeTensorへ変換
  -> CubeCL implicit-GEMM convolution
  -> Burn Tensorへ戻す
  -> raw CubeTensorへ変換
  -> handwritten Snake WGSL
```

Burn custom Fusion providerはFusion graph上で認識できるoperation segmentを置換する。一方、上のraw
launcher呼出はgraph segmentをそこで切るため、providerを登録するだけではconvolutionとSnakeを一つに
できない。先に低水準handle変換とfallbackを`backend_bridge`へ隔離し、次のいずれかを実装する必要がある。

1. k=7 conv + bias + Snakeを一つのbackend extension operationとして定義し、WGPU implementationで
   cubek matmul/convolutionへF32-accumulatorのSnake epilogue writerを渡す。
2. portable graphのconv + Snake patternをcustom Fusion providerが認識し、Fusion handleから既存bufferを
   直接bindingする。unsupported shape/deviceはunfused reference implementationへ戻す。
3. cubek側へ汎用custom epilogue interfaceを追加し、Irodori固有kernel forkではなく再利用可能な境界にする。

最初の候補は1である。operationの意味、dtype、layout、fallbackを一つの型に閉じられ、巨大なattention
全体のようなmonolithic shaderを作らずに12個の中間Tensor/dispatchを削れる。精度gateを通してから
Fusion pattern recognitionへ拡張する。

## 次の最適化優先順位

1. `backend_bridge::codec::K7BiasSnake`をADTとして作り、portable fallbackとWGPU fused epilogueを同じ
   operation contractにする。shape、dilation、dtypeのunsupported caseは明示fallbackする。
2. cubek implicit-GEMMへF32 accumulatorの`x + alpha.recip() * sin(alpha*x)^2` epilogueを追加し、
   中間F16 Tensorのwrite/readとSnake dispatchを消す。
3. 同じ境界で50 framesをscreenし、2%以上短縮した場合だけ六つの長さ、5 fresh session、NVML、full
   RF+codecへ昇格する。
4. pointwiseはprojection + bias + residual + next Snakeを一つのprovider候補にし、implicit-GEMMへの
   単純置換は再試行しない。
5. backend bridgeができた後にBurn custom Fusion providerへ接続し、provider hit/fallback/compile countを
   warmup manifestへ記録する。

## 再開手順

1. branch HEADと本reportのcampaign start HEADを確認する。
2. artifactで`sha256sum -c SHA256SUMS`を実行し、fixture/codec/binary pinを確認する。
3. 新しいcampaign directoryとCubeCL environmentを作り、旧時間値をpoolしない。
4. `profile_codec_decoder`のproduction planで50-frame baselineを取り直す。
5. `K7BiasSnake` portable fallbackのhashをproductionと一致させる。
6. WGPU fused epilogueを実装し、device/readback、F16複合accuracy gate、F32 85 dB gateを通す。
7. 50-frameで2%以上短縮した候補だけを六長、5 fresh process、NVML campaignへ進める。
