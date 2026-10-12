# RTX 5070 Ti Laptop: CubeK post-cast epilogue監査修正（2026-08-13）

## 結論

custom k7+Snake routeの固定モデル上のaccuracyは維持しつつ、汎用APIとして問題だった
境界tileのparameter OOB、optionalな生binding、数値契約の曖昧さを修正した。最終50-frame
productionの5 fresh process中央値はdevice-complete `15.585 ms`、readback-complete
`16.486 ms`である。直前採用値`15.642 / 16.352 ms`との差はそれぞれ`-0.37% / +0.82%`で、
性能を維持した。PyTorch CUDA F16の独立campaign値`13.391 / 13.843 ms`には、同一境界で
なお`16.38% / 19.10%`届かない。

監査で提案されたsingle-storage OKI canonicalizationも実装して同一processで比較したが、
request-time repackよりdevice中央値が`0.165 ms`遅かった。そのためproductionには採用せず、
再現用のprofile候補として残した。「k7 operator全体が常に1 dispatch」という表現は撤回する。
現productionではconvolution+Snake本体は1 dispatchだが、weight layout copyが前置され得る。

## correctness hardening

### writer-level edge mask

旧writerは全laneへepilogueを適用してから`write_checked`していた。新writerは出力sliceの有効
shapeを保持し、interior tileはuniformなfull-tile fast path、edge tileはlaneごとの論理座標を
先に検査する。したがって、捨てるlaneは`SnakeEpilogue::apply`へ入らず、channel parameterを
一切readしない。座標変換も手計算を廃止し、storeと同じ`TiledLayout::to_source_pos`へ統一した。

GPU直接テストでは`N={1,15,17,95,97}`、`M={1,15,17,65}`を実行した。`M=65`は後続partitionの
非ゼロoriginを含む。全shapeでreadback完了、finite、WGPU errorなしだった。

### typed launch contract

通常routeは`SimpleConv<...>`、parameter必須routeは
`SimplePostCastEpilogueConv<..., E>`へ分離し、`Routine::PostCastEpilogue`でlaunch APIを拘束した。
custom writerを通常launcherから引数なしで起動する状態は型で表現できない。

Snakeはprivate fieldを持つ`F32EpilogueParameters<R>`だけを受け取り、dispatch前に次を検査する。

- non-quantized F32であること
- shape/strideが連続であること
- byte offsetがF32 alignmentを満たし、backing buffer長が足りること
- parameterの要素数が`out_channels`以上であること
- launch先と同じ`ComputeClient`であること
- parameter bindingもaddress-width算定に含めること

vendor crateの直接contract testも復元した。source provenance、crate archive SHA、license、
再生成方法は`vendor/UPSTREAM.md`に固定した。

### 数値契約とderived cache

実装はaccumulator-domain epilogueではない。正確な順序は次である。

```text
F32 accumulator
  -> output global dtype (F16)へcast
  -> F16値をF32へpromote
  -> F32 Snake
  -> F16 store
```

trait名を`PostCastGlobalEpilogue`へ変更し、将来のaccumulator-domain epilogueと区別した。この順序は
従来standalone F16 Snakeの丸め意味に合わせるためであり、演算バグではない。

`#[module(skip)] alpha_epilogue_f32`はrecord/device traversal後に陳腐化し得るため、全ResidualUnitの
明示的`prepare_for_wgsl`でlearned alphaから必ず再構築する。将来の完全な
`PreparedDecoderPlan<Device,DType>`分離までは、このfail-closedな再prepareを契約とする。

## single-storage実験

元OIKをprepare時に一度だけpitched OKIへcopyし、logical OIKは同一storageのstride viewとして
保持する候補を実装した。duplicate方式と違い、約32 MiBの二重weightは持たない。同一binary、
同一process、5 warmup後、ABBA/BAAB 5 block（各route 10 sample）の結果は次だった。

| route | device-complete median | readback-complete median | waveform |
|---|---:|---:|---|
| single-storage OKI | 15.757 ms | 16.615 ms | `113ba560...` |
| request-time repack | **15.592 ms** | **16.574 ms** | `113ba560...` |

bitwise同一だがsingle-storageはdeviceで1.06%遅い。pitched allocationを使ってもこの結果だったため、
高速化維持という採用条件を満たさない。候補は`implicit-gemm-single-storage`と
`--paired-single-storage`で再測定できるが、productionはrequest-time repackを維持する。

## fresh campaign

- output: `/home/sanzentyo/benchmark-artifacts/irodori-v4-cubek-hardening-20260813-attempt1`
- source start: `07d56b9e86bafd1de7edd540642030c1bc67371a`
- measured implementation commit: `e4e8c36ede2f97ac21ea15e22e5299a56f077af6`
- final measured binary SHA-256: `3bdd17a831aaacacca263a1b1268f1dec2de1747d8eb2d187b3245c25cc36b0a`
- GPU: NVIDIA GeForce RTX 5070 Ti Laptop GPU、12,227 MiB
- driver: `595.71.05`
- WGPU: Vulkan discrete adapter 0
- CUDA/NVML index: 0
- PCI bus ID: `00000000:01:00.0`
- codec SHA-256: `b14a25da5d68bf779d8c44a40706ddc7c4819cb60489b2a80a6408ca03c514fb`
- F16 oracle SHA-256: `08663e52df6c63eea7ebf5ae2ba35778bdb5bc7d20cd06e430e855a86174229e`
- F32 oracle SHA-256: `5ea1fcddac1160780dfb53377ecf8fed935fc6f0bab2e2e55464a06868637094`
- boundary: pre-start syncからdevice completion、またはowned contiguous F32 CPU waveformまで
- session: 5 fresh process、各5 warmup + 10 measured、process medianのmedian

session medianは次である。

| session | device ms | readback ms |
|---:|---:|---:|
| 1 | 15.775 | 16.598 |
| 2 | 15.585 | 16.542 |
| 3 | 15.510 | 16.229 |
| 4 | 15.579 | 16.371 |
| 5 | 15.613 | 16.486 |
| median | **15.585** | **16.486** |

waveformは全repeatでhash `113ba560546d82a3112332ac67b3cea5d5b83b407109d3df3817e5b82b609e05`、
SNR `56.074 dB`、max abs `3.418e-3`、cosine `0.999998775055`だった。旧reportとはhashが
異なるが、最初の差分位置は未局在化であり、edge-mask変更だけを原因とは断定しない。oracle gateは
同等に通過する。

最終binaryに対するNVML 100 ms samplingは215点、peak used `1,186 MiB`、minimum free
`10,589 MiB`だった。VRAM回帰は観測されなかった。

## 長さ・F32回帰

45/112/255/333/489/685 framesをそれぞれpinned F32 oracleへ再比較し、全件pass、WGPU error 0だった。
SNRは順に`57.576 / 56.937 / 56.773 / 57.252 / 58.900 / 59.883 dB`である。489 framesもaccuracyを
通している。F32回帰はSNR `113.197 dB`、max abs `5.260e-6`、cosine `0.999999999998`だった。

## verification

- `cargo test --lib --features inference,codec`: 507 passed、0 failed、17 ignored
- partial-tile Vulkan GPU test: 1 passed
- vendored CubeK typed-contract test: 1 passed
- `cargo clippy --all-targets --features inference,codec,cli,profile -- -D warnings`: pass
- `cargo fmt --all`およびvendored sourceのrustfmt: pass
- `uvx ruff check scripts`: pass

## portabilityと残課題

mask、typed spec、parameter validationはCubeCL IR/host Rustで書かれており、NVIDIA固有APIや
WGSL文字列には依存しない。Vulkan、Metal、DX12で共有できるsource設計だが、実行確認済みなのは
Vulkan/NVIDIAだけである。Metal/DX12のcompile/accuracy smokeが終わるまではupstream-readyとは
呼ばない。

次の優先順位は、weight repackを消すためにstorage layoutだけを変えるのではなく、実測で遅くなった
原因をGPU timestampとallocator pitch/vectorization receiptで分解すること、その後にtile/vector
Snake epilogue、custom epilogue込みのshape別CubeK探索、fixed-k7 halo loaderを順に評価することである。
`GlobalStoreTransform`によるdual outputはその後とし、巨大なIrodori専用monolithic shaderは優先しない。
