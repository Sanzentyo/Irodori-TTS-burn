# RTX 5070 Ti Laptop: CubeK accumulator store transform（2026-08-19）

## 結論

F16 codecの8本のintra-block pointwise境界を、CubeKの汎用CMMA coreと
accumulator-domain multi-output store transformで実行するproduction routeへ変更した。
projection、bias、residual加算、raw NCL出力、次のSnake済みNHWC出力を1 dispatchで行う。

50-frame codecの5 fresh process中央値はdevice-complete `14.067 ms`、
readback-complete `14.984 ms`だった。直前採用値`15.585 / 16.486 ms`に対して
`9.74% / 9.11%`短縮した。PyTorch CUDA F16の独立した同一境界campaign値
`13.391 / 13.843 ms`との差はdevice `5.05%`、readback `8.24%`まで縮まった。
PyTorch値は今回のsessionへpoolせず、過去のpin済みcampaignとの比較値としてのみ扱う。

六長さ、489-frame accuracy、F32 production、VRAMを全て通したため、profile-only screenから
F16 productionへ昇格した。

## 本質的な変更

手書きpointwise shaderのtile数をGPU固有に調整したのではない。vendored CubeKへ次の汎用境界を
追加した。

- `AccumulatorGlobalStoreTransform<RC>`はstage accumulatorと絶対論理座標を受け取る。
- transformはprimary outputを返しながら、型付きruntime bindingへauxiliary outputを書ける。
- writerはinterior tileのbranch-free fast pathとedge lane maskを持ち、範囲外laneではtransformを
  呼ばない。
- `SimpleAccumulatorTransformConv<LL, LR, E>`により、parameterなし通常convolutionと
  auxiliary binding必須convolutionを型で分離した。
- F16 residual、alpha、raw outputはdispatch前にdtype、contiguous、buffer長、device client、
  address widthを検査する。

Irodori側の`F16ResidualSnakeStore`は次の順序を持つ。

```text
CubeK F32 projection accumulator
  + bias
  + F16 shortcutをF32へpromoteした値
  -> raw F16 NCL auxiliary store
  -> unrounded F32 raw値でSnake
  -> activated F16 NHWC primary store
```

primary NHWCは次のk=7 implicit-GEMMが直接消費し、raw NCLは次ResidualUnitのidentity shortcutに
なる。k=1 weightはcheckpoint-native OIK `[O,I,1]`をlogical OKI `[O,1,I]`へpermuteしても
物理stride `[I,1,1]`のcontiguous viewである。このrouteにはrequest-time weight layout copyも
persistent duplicateもない。

従来のdirect pointwise+residual+Snake pairも既に1 dispatchだった。今回の短縮はdispatch数を
さらに減らしたためではなく、同じmulti-output意味論をscalar/tiled手書きprojectionからCubeKの
CMMA coreへ移し、中間projectionをmaterializeしない汎用store transformと組み合わせた効果である。

## same-process paired evidence

同一model、同一binary、同一standalone block-boundary graphで、候補と既存packed pointwise controlを
ABBA/BAAB順に交互実行した。5 fresh process、各5 warmup + 10 block、各route 20 sampleである。

| session | candidate-control block delta median ms | improved blocks |
|---:|---:|---:|
| 1 | -0.551 | 8/10 |
| 2 | -0.517 | 8/10 |
| 3 | -0.505 | 7/10 |
| 4 | -0.546 | 9/10 |
| 5 | -0.527 | 10/10 |
| median | **-0.527** | **42/50** |

候補hashは`04daa965...cc38`、controlは`113ba560...9e05`で、経路間bitwise同一ではない。
各経路内は全repeatで決定的だった。候補はaccumulator-domain residual加算のため丸め位置が異なるが、
F16 oracle SNRはcontrol `56.074 dB`から候補`56.623 dB`へ改善した。

## production fresh sessions

- measured commit: `9228827b5156ee5dd46f16149d41828ef39001e5`
- binary SHA-256: `60c893b3933835b899324448024900c46a74b24d996794703eb6dd5961128bc8`
- session: 5 fresh process、各5 warmup + 10 measured
- boundary: pre-start device syncからdevice completion、またはowned contiguous F32 CPU waveformまで
- CubeCL cache: campaign固有の新規directory。旧`/tmp`、旧campaignのcache・時間値は不使用

| session | device-complete ms | readback-complete ms |
|---:|---:|---:|
| 1 | 13.966 | 14.984 |
| 2 | 14.001 | 14.916 |
| 3 | 14.082 | 14.906 |
| 4 | 14.132 | 15.404 |
| 5 | 14.067 | 15.162 |
| median | **14.067** | **14.984** |

全50 waveformはhash `04daa96513fe33c680bc0ca475b2182936074a4578312a76f3dfab821f49cc38`、
SNR `56.622776 dB`、max abs `3.41796875e-3`、cosine `0.999998916470`、NaN/Inf 0、
uncaptured WGPU error 0だった。

## 長さaccuracyとF32回帰

各F16 runは独立したpinned F32 oracleと比較した。489 framesをaccuracyなしで性能PASSにはしていない。

| audio相当 | frames | SNR dB | max abs | WGPU error |
|---:|---:|---:|---:|---:|
| 1.80 s | 45 | 57.600 | 2.128e-3 | 0 |
| 4.48 s | 112 | 57.502 | 1.668e-3 | 0 |
| 10.20 s | 255 | 57.486 | 1.620e-3 | 0 |
| 13.32 s | 333 | 57.707 | 1.959e-3 | 0 |
| 19.56 s | 489 | 59.554 | 1.865e-3 | 0 |
| 27.40 s | 685 | 60.381 | 1.672e-3 | 0 |

F32 50-frame productionはSNR `113.197200 dB`、max abs `5.260109901e-6`、cosine
`0.999999999998`、WGPU error 0である。F32はdtype preflightで新routeを選ばず、従来経路を維持する。

## 環境・VRAM・artifact

- GPU: NVIDIA GeForce RTX 5070 Ti Laptop GPU、12,227 MiB
- driver: `595.71.05`
- WGPU: Vulkan discrete adapter 0
- CUDA/NVML index: 0
- PCI bus ID: `00000000:01:00.0`
- campaign開始前available: 11,774 MiB
- NVML: 100 ms間隔488 samples、peak used `1,186 MiB`、minimum free `10,589 MiB`
- converted codec SHA-256: `b14a25da5d68bf779d8c44a40706ddc7c4819cb60489b2a80a6408ca03c514fb`
- F16 fixture SHA-256: `08663e52df6c63eea7ebf5ae2ba35778bdb5bc7d20cd06e430e855a86174229e`

fresh outputは
`/home/sanzentyo/benchmark-artifacts/irodori-v4-cubek-pointwise-store-20260819-attempt2`
である。raw session、paired session、length、F32、NVML、environment、summary、SHA256SUMSを含む。
`vulkaninfo` binaryはこのhostに無かったため、その事実をenvironment logへfail-closedで記録した。

## portabilityと次の候補

transformはCubeCL IR、typed view、generic writer familyだけで実装し、NVIDIA API、subgroup幅、Vulkan
pipeline cache、手書きWGSL文字列へ依存しない。source設計はVulkan、Metal、DX12で共有可能だが、
今回の実行検証はVulkan/NVIDIAのみである。

次の構造候補は、残る各block最後の4本のpointwiseについて、Snakeなしの
`AccumulatorResidualStore`を使いCMMA projection+bias+residualを直接NCLへstoreすること、その後に
wm-head/ConvTranspose境界のallocation lifetimeを再確認することである。shape別tile調整やparameter sweepは
これらを試し切った後、別branchの自動tunerとして扱う。

## verification

- vendored `cubek-convolution` unit test: 1 passed
- `cargo test --lib --features inference,codec,profile`: 513 passed、0 failed、20 ignored
- `cargo clippy --lib --features inference,codec,profile -- -D warnings`: pass
- `cargo fmt --all`: pass
- 5 fresh production sessions、5 fresh paired sessions、六長さ、F32 regression: pass

