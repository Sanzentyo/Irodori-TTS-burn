# RTX 5070 Ti Laptop: CubeK custom epilogueによるk7 + Snake融合（2026-08-13）

## 結論

> 監査追補: この初回reportのAPI契約、edge mask、dispatch表現、数値dataflowは後続の
> [hardening report](rtx-5070ti-v4-cubek-epilogue-hardening-2026-08-13.md)で訂正した。
> 性能・accuracyの初回campaign記録として残すが、現在の実装説明には後続reportを用いること。

CubeK 0.3.0-pre.2のglobal output writerへ、runtime parameterと絶対出力座標を受け取る
汎用custom epilogue境界を追加した。IrodoriのF16 codecでは、この境界にSnakeを実装し、
12本の`k=7 convolution + bias`と直後のSnakeを各1 dispatchへ融合した。

50 latent frames（2.0秒、96,000 samples）の最終binaryによる5 fresh process中央値は、
device-complete `15.642 ms`、readback-complete `16.352 ms`だった。直前productionの
`15.873 / 16.723 ms`から1.5% / 2.2%短縮した。PyTorch CUDA F16の同一境界
`13.391 / 13.843 ms`にはまだ届かず、WGPUは16.8% / 18.1%遅い。

## 実装した抽象化

局所shape専用WGSLは追加していない。exact pinした`cubek-matmul`と`cubek-convolution`
を`vendor/`へ置き、Cargoの`[patch.crates-io]`で二crateを同時に固定した。変更点は次である。

1. `GlobalWriterFamily<RC>` / `GlobalWriter<RC>`がruntime configと論理出力originを受け取る。
2. `GlobalEpilogue<RC>`は、F16/F32等のscalar、絶対`(m, n)`座標、runtime configから
   store直前の値を返す純粋なtraitである。
3. `EpiloguePlaneWriterFamily<E>`は既存PlaneWriterと同じstage drainを行い、各scalarだけを
   `E::apply`へ通す。standard PlaneWriter/UnitWriterはruntime情報を無視するため演算を変えない。
4. `SimpleAlgorithm<..., GW>`と`SimpleConv<..., GW>`でwriter familyを差し替えられる。
   sync/async、loading strategy、tile選択とepilogueを直交させた。
5. convolution runtime argsの`epilogue_param`は`ComptimeOption`である。standard routeでは
   `None`なので、追加storage bindingも分岐も生成しない。
6. `SnakeEpilogue`はchannel座標でF32 alphaを読み、F32で`sin`と除算を実行して、最後だけ
   output dtypeへcastする。F16 alphaをF32へ展開した小さなcacheはmodel load時に一度だけ作る。

物理dataflowは次になった。

```text
before:
CubeK implicit-GEMM + bias -> F16 NHWC intermediate (global write)
standalone Snake           -> F16 NHWC output       (global read/write)

after（この初回実装の正確なcast順序）:
CubeK F32 accumulator -> F16 cast -> F32-promoted Snake -> F16 NHWC output
```

epilogueはstage accumulatorのdrain時に適用される。したがってconvolutionのload、tile、FMA、
bias順序は変えず、中間Tensorのwrite/readと12 dispatchだけを除いた。

## portability

この実装はCubeCL IRの`#[cube]` traitと既存WGPU runtimeだけを使う。WGSL文字列、NVIDIA API、
subgroup幅、Vulkan-only pipeline cacheには依存しないため、WGPUが動くVulkan、Metal、DX12で
同じsource contractを使える。実測はVulkan/NVIDIAだけなので、Metal/DX12での性能とshader
acceptanceは別途確認が必要である。

F32 epilogue parameterは意図的である。F16 checkpointのalphaを一度F32へcastし、standalone
F16 Snakeと同じ「F16値をF32へpromoteして演算」する意味を保つ。BF16は今回のscope外である。

## fresh campaignと境界

- branch: `codex/v4-wgsl-fusion`
- campaign start HEAD: `cf587c60c2d738466224276dc522a361597d18fb`
- implementation commit: `258f7f1e142493abf943f42ee221308d89202647`
- GPU: NVIDIA GeForce RTX 5070 Ti Laptop GPU、12,227 MiB
- driver: 595.71.05
- WGPU adapter: Vulkan discrete adapter 0
- CUDA/NVML index: 0
- PCI bus ID: `00000000:01:00.0`
- Burn/Burn-CubeCL: `=0.22.0-pre.2`
- CubeCL: `=0.11.0-pre.2`
- CubeK matmul/convolution: `=0.3.0-pre.2` + repository-local patch
- execution storage: F16
- accumulator/Snake arithmetic: F32
- oracle SHA-256: `08663e52df6c63eea7ebf5ae2ba35778bdb5bc7d20cd06e430e855a86174229e`
- converted codec SHA-256: `b14a25da5d68bf779d8c44a40706ddc7c4819cb60489b2a80a6408ca03c514fb`
- final profiler binary SHA-256: `816877dd4b4efa39ab04dd776f3ac2768beb9cdf31a3ac7c5dd02a6fe80c5f76`

fresh outputは
`/home/sanzentyo/benchmark-artifacts/irodori-v4-cubek-epilogue-20260813-attempt1`
である。旧`/tmp` artifactや旧sessionの時間値を新session集計へpoolしていない。比較対象の
直前productionとPyTorch値は、pin済みの前campaignにある独立した境界値である。

- device-complete: pre-start device syncからcodec device completionまで。
- readback-complete: device-completeにowned contiguous F32 CPU waveform取得までを加える。
- fresh session: 5 process、各5 warmup + 10 measured。
- 集計: process内medianの5値を取り、そのmedianを採用。

## 50-frame性能とaccuracy

| runtime / condition | device-complete ms | readback-complete ms |
|---|---:|---:|
| PyTorch CUDA F16 | **13.391** | **13.843** |
| Rust WGPU F16 custom epilogue | 15.642 | 16.352 |
| Rust WGPU F16 previous production | 15.873 | 16.723 |

fresh session mediansは次である。

| session | device ms | readback ms |
|---:|---:|---:|
| 1 | 15.739 | 16.540 |
| 2 | 15.526 | 16.191 |
| 3 | 15.694 | 16.352 |
| 4 | 15.471 | 16.180 |
| 5 | 15.642 | 16.368 |

全measured waveformはhash
`1607e73627b74f23f1267022471a7cd609c043aa106810f3bb00b249a3bdad55`
で一致した。F16 oracleに対しmax abs `3.417968750e-3`、RMSE `2.139069166e-4`、
SNR `56.076805 dB`、cosine `0.999998775886`、NaN/Inf 0、uncaptured WGPU error 0だった。

NVMLは100 ms間隔38 samplesでpeak used 1,186 MiB、minimum free 10,589 MiBだった。前campaignの
codec-only peak 1,210 MiBに対し増加していない。24 MiB差にはsampling/allocator変動も含まれるため、
全量を中間Tensor削除の効果とは断定しない。

## 長さaccuracyとF32回帰

各長さを独立したpinned F32 oracleへ比較し、489 framesをaccuracyなしで性能PASSにしていない。

| audio相当 | frames | SNR dB | max abs | finite / WGPU error |
|---:|---:|---:|---:|---:|
| 1.80 s | 45 | 57.565 | 2.057e-3 | pass / 0 |
| 4.48 s | 112 | 56.937 | 1.888e-3 | pass / 0 |
| 10.20 s | 255 | 56.782 | 3.344e-3 | pass / 0 |
| 13.32 s | 333 | 57.251 | 1.959e-3 | pass / 0 |
| 19.56 s | 489 | 58.900 | 1.685e-3 | pass / 0 |
| 27.40 s | 685 | 59.883 | 1.856e-3 | pass / 0 |

F32 production regressionはSNR `113.197200 dB`、max abs `5.260109901e-6`、cosine
`0.999999999998`、uncaptured WGPU error 0で、従来値を維持した。F32 productionは既存の
specialized WGSL routeを使うため、custom epilogueを選ばない。

## warmupと残課題

custom writerは新しいpipeline identityを作る。2 warmupだけのscreeningでは初期compile/tuneの残りが
measured領域へ入り、最初の三回が48.7、27.2、14.1 msと収束した。このため正式sessionは5 warmupに
増やした。これはprocess-local pipeline生成であり、CubeCL environment bundleだけでComputePipelineを
process間復元できることを意味しない。productionではlong-lived sessionのreadiness前にmanifest shapeを
このwriter variant込みでwarmupする。

次の候補は、同じepilogue境界をpointwise residual/gateへ適用して中間writeを減らすことである。ただし
現時点でもPyTorchとの差はdevice 2.25 ms、readback 2.51 ms残る。shape専用巨大shaderへ戻らず、同じ
generic writer、Burn Fusion provider、またはlayout lifetime削減で埋める。
