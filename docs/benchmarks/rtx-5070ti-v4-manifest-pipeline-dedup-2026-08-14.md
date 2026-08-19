# RTX 5070 Ti Laptop: v4 manifest codec pipeline dedup（2026-08-14）

## 結論

`WarmupManifest::v4_service()`の8 caseをRFとcodecへそのまま各8回渡していたうち、codec側の
112-frame TextOnly / Designed / PreparedCloneは同じ入力geometry
`(batch=1, latent_frames=112, latent_dim=32)`であり、同一decoder pipeline集合を3回辿っていた。
codec DryRunを実runtime geometryでdedupし、RFは8 caseを維持したままcodecを8 caseから
6 unique shapeへ減らした。serviceが受理するmanifestとreal validation caseは変更していない。

5 fresh-process A/BではDryRun wall中央値が`29.626 s`から`29.570 s`へ`0.056 s`
（`0.19%`）短くなった。pair差の中央値は`-0.129 s`だが、5組中1組は`+0.604 s`であり、
wall効果は測定ノイズ域である。初回real request中央値は`0.196 s`から`0.198 s`で実質不変だった。
発行回数は確実に減り、readinessを弱めない小さい変更なので採用するが、startup短縮の主要施策とは
数えない。

## 重複key

CubeCL/WGPUのprocess-local pipeline cacheは`KernelId`へ`ExecutionMode`を加えたidentityで再利用する。
codec decodeはvoice topologyを入力に取らず、同じresident decoderへ同じtensor geometryを渡すため、
以下の3 manifest entryが同じcodec kernel/pipeline列を生成する。

| manifest entry | codec geometry key | 判断 |
|---|---|---|
| 112 / TextOnly | `(1, 112, 32)` | 保持 |
| 112 / Designed | `(1, 112, 32)` | 重複のためcodec DryRunを省略 |
| 112 / PreparedClone | `(1, 112, 32)` | 重複のためcodec DryRunを省略 |

45 / 255 / 333 / 489 / 685 frameはそれぞれ異なるcodec geometryである。RF側はlatent長または
conditioning topologyが異なり、後続caseに固有pipelineがないことを保証できないため、8 caseすべてを
保守的に維持した。部分的に共通するRF `KernelId`は既存のCubeCL pipeline cacheが再利用する。

結果としてcompile-only traversalはRF 8 + codec 8の16回から、RF 8 + codec 6の14回へ減る。
`WarmupReport`には`dry_run_rf_cases`、`dry_run_codec_shapes`、
`dry_run_codec_duplicates_skipped`を追加し、この判断をruntime reportから監査可能にした。

## A/B条件

- source base: `96037a2`
- GPU: NVIDIA GeForce RTX 5070 Ti Laptop GPU、12,227 MiB
- runtime: Vulkan WGPU、strict FP32、TF32 off、autocast off
- model/codec: all-resident、RF/codec checkpoint parallel load
- profile: RF `fixed112_packed_only`、codec `fixed112_packed_only`
- workload: 112-frame unconditioned requestを3件
- DryRun: RFは両variantとも3件、codecはbaseline 3件 / dedup 1件
- CubeCL: 同じapproved fixed112 autotune bundleをimportし、processごとに新規cache directory
- aggregation: baseline / dedup各5 fresh process、交互実行、自動retry 0

このfixed112 A/Bはmanifest中の112-frame codec重複2件と同じ削減数を再現するscreenである。
六長すべてを同時に走らせたwallの主張には拡張しない。

## 結果

| metric | baseline | dedup | 差 |
|---|---:|---:|---:|
| DryRun wall median | 29.626 s | 29.570 s | -0.056 s / -0.19% |
| DryRun wall range | 29.353–30.765 s | 29.215–30.637 s | — |
| first request median | 0.196 s | 0.198 s | +0.002 s / +0.79% |
| first RF median | 0.120 s | 0.121 s | +0.001 s |
| first codec median | 0.075 s | 0.075 s | +0.001 s |

各process内の3出力は同一hashだった。fresh process間では既知hash
`5c22e03be6864d320a7881939b318d0d066b06af3005942457a7dc7e1e43c8b9`と、前のparallel warmup
screenでも観測済みのalternate
`ae6510616fb18a0acb2bd27ff62df52fa4235f9802d815553b64add4cb8f40b2`が出た。dedupによる新規hashは
観測していないが、alternateのoracle accuracyは今回も算出していないため、cross-process bitwise
determinismや新しいaccuracy承認の根拠にはしない。

## artifact

- `/home/sanzentyo/benchmark-artifacts/irodori-v4-manifest-dedup-N9b9f5Li`
- 保存binary: `bench_v4_residency-baseline` / `bench_v4_residency-dedup`
- raw result: `baseline-{1..5}.json` / `dedup-{1..5}.json`
- `SHA256SUMS`: `d806b1fb73c650e30fd6acab71b4f2f5d734d41cea843bfb294a8b911012c6fe`
