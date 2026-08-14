# RTX 5070 Ti Laptop: multi-row採用後の候補screen（2026-08-14）

## 結論

k7 geometry-selected multi-rowを採用した後、50 latent framesのF16 codecで残る
k7/pointwise候補をscreenした。いずれもaccuracy gateとfinite/WGPU error 0を通したが、
productionより速いと判断できる候補はなかった。production sourceは
`multi_rows = M >= N && N >= 384`、CubeK CMMA、direct pointwise T64/O96/K32を維持する。

- source HEAD: `b23e830`
- GPU: NVIDIA GeForce RTX 5070 Ti Laptop GPU、12,227 MiB
- driver: `595.71.05`
- execution: Vulkan WGPU、F16 storage、F32 accumulator/Snake
- codec SHA-256: `b14a25da5d68bf779d8c44a40706ddc7c4819cb60489b2a80a6408ca03c514fb`
- F16 oracle SHA-256: `08663e52df6c63eea7ebf5ae2ba35778bdb5bc7d20cd06e430e855a86174229e`

比較はGPU device timestampを主に使った。process間のclock/power driftが見えた候補は
A-B-Aでproductionを前後に置き、候補値を線形補間したcontrolとも比較した。2%未満の局所差や
host wallだけの改善は採用根拠にしていない。

## k7 geometry follow-up

### selector境界

50-frameでforced multi-rowをblock別に確認した。block 0の`M=600, N=768`は
`0.913 ms`でproduction single-rowの`0.895 ms`より遅かった。block 2/3もそれぞれ
約`0.149 / 0.150 ms`遅く、block 1だけが小幅に速かった。したがって、採用済みの
`M >= N && N >= 384`を緩めない。

artifact:
`/home/sanzentyo/benchmark-artifacts/irodori-v4-k7-geometry-threshold-20260814-attempt1`

### multi-row N partition

N partition 2/8を探索した。N=2は別processの112-frame screenでは良く見えたが、同一binaryの
paired ABBAではselected k7のblock内差がmedian `+0.0395 ms`、勝率4/10、
device-completeが`+0.7417 ms`、勝率2/10だった。N=8も50-frame block 1で
約`1.250 ms`とproduction N=4の約`1.247 ms`を上回らなかった。双方不採用とした。

artifact:
`/home/sanzentyo/benchmark-artifacts/irodori-v4-k7-multi-rows-tiling-20260814-attempt1`

### tile matmul family

`SimpleArgs.tile_matmul=Mma`を20 stage repeatでscreenした。CMMA controlのk7合計
約`7.20 ms`に対しMMAはmedian `11.053 ms`、device-completeも`19.486 ms`だった。
waveformは決定的でSNR `55.757 dB`だが、hashは`8b755ae8…`へ変わる。大幅に遅いため不採用。

artifact:
`/home/sanzentyo/benchmark-artifacts/irodori-v4-k7-tile-kind-20260814-attempt1`

### Snake reciprocal cache

model準備時に`[alpha, 1/(alpha+epsilon)]`をF32で作り、CubeK post-cast Snakeの除算を
乗算へ置換した。20 repeatのk7 stage合計はproduction前`7.1981 ms`、候補
`7.2640 ms`、production後`7.3301 ms`だった。候補は前後controlの線形補間
`7.2641 ms`と0.0001 ms未満で一致し、速度効果はなかった。全stageでは補間controlより
約`+0.014 ms`だった。waveformはproductionとbitwise一致したが不採用とした。

artifact:
`/home/sanzentyo/benchmark-artifacts/irodori-v4-k7-snake-reciprocal-20260814-attempt1`

### CubeK global loader family

fixed-k7専用halo loaderへ進む前に、CMMA、multi-row selector、Snake epilogueを固定し、
既存のglobal loaderだけを片側ずつ変更した。各候補は5 warmup + 10 measured + 20 stage repeatを
完走し、productionとbitwise一致、accuracy gate通過、WGPU error 0だった。候補をすべて戻した
同時刻controlのk7合計medianは`7.287 ms`、全stageは`14.119 ms`だった。

| LHS input loader | RHS weight loader | k7 median ms | control差 | 判断 |
|---|---|---:|---:|---|
| strided | cyclic | 7.506 | +3.0% | 不採用 |
| tilewise | cyclic | 7.401 | +1.6% | 不採用 |
| cyclic | tilewise | 7.437 | +2.1% | 不採用 |
| cyclic | strided | 8.271 | +13.5% | 不採用 |

汎用loaderの組合せ変更では短縮せず、専用loaderを作るならk=7のhalo重複除去を実際に行う
必要がある。

artifacts:

- `/home/sanzentyo/benchmark-artifacts/irodori-v4-k7-lhs-strided-20260814-attempt1`
- `/home/sanzentyo/benchmark-artifacts/irodori-v4-k7-lhs-tilewise-20260814-attempt1`
- `/home/sanzentyo/benchmark-artifacts/irodori-v4-k7-rhs-tilewise-20260814-attempt1`
- `/home/sanzentyo/benchmark-artifacts/irodori-v4-k7-rhs-strided-20260814-attempt1`
- `/home/sanzentyo/benchmark-artifacts/irodori-v4-k7-loader-control-20260814-attempt1`

### 標準CubeK loader経路

custom post-cast Snake epilogueを使わない既存profile経路の未記録3条件も、それぞれ
5 warmup + 10 measured + 20 stage repeatでscreenした。

| 経路 | device median ms | k7 median ms | 全stage median ms | 判断 |
|---|---:|---:|---:|---|
| production control | 15.458 | **7.287** | **14.119** | production |
| sync-strided | 17.216 | 9.364 | 16.079 | 不採用 |
| async cyclic | 38.468 | 31.098 | 38.919 | 不採用 |
| async-strided | 38.920 | 30.949 | 38.791 | 不採用 |

全経路はaccuracy gateとWGPU error 0を通した。sync-stridedはhash `d2ed183a…`、async二経路は
`e7c98882…`で決定的だったが、productionとはbitwise不一致であり、速度も大幅に遅い。

artifacts:

- `/home/sanzentyo/benchmark-artifacts/irodori-v4-k7-standard-sync-strided-20260814-attempt1`
- `/home/sanzentyo/benchmark-artifacts/irodori-v4-k7-standard-async-20260814-attempt1`
- `/home/sanzentyo/benchmark-artifacts/irodori-v4-k7-standard-async-strided-20260814-attempt1`

## pointwise follow-up

### C768 direct route

既存direct T64/O96/K32をC768へ拡張した。block 0のpointwise三段合計は
production packed path `0.202 ms`に対しdirect候補`0.465 ms`で、2倍以上遅かった。
候補hashは`c640407c…`へ変わったが、SNR `56.333 dB`、max abs `3.90625e-3`で
accuracy gate自体は通った。速度とbitwise性の両方から不採用。

artifact:
`/home/sanzentyo/benchmark-artifacts/irodori-v4-pointwise-c768-20260814-attempt1`

### exact output-channel guard除去

C384/C192/C96はすべてO96で割り切れるため、F16 direct shaderのoutput-channel guardを
除去した。20 repeatのpointwise合計はcontrol `4.441 ms`に対し候補`4.464 ms`、
全stageも`13.954 ms`対`14.052 ms`だった。非対象k7も同時に遅くなっておりprocess driftを
含むが、少なくとも短縮は観測できない。bitwise一致のまま不採用とした。

artifact:
`/home/sanzentyo/benchmark-artifacts/irodori-v4-pointwise-bounds-20260814-attempt1`

### reduction tile K16/K24/K48

direct pointwiseのK tileを20 repeatずつscreenした。最も近いK32 controlと各候補の
pointwise stage合計は次の通り。

| K tile | pointwise median ms | 判断 |
|---:|---:|---|
| 32 | **4.461** | production |
| 16 | 4.465 | 同等、全体採用差なし |
| 24 | 4.560 | 遅い |
| 48 | 4.851 | 遅い |

K16はblock 2/3を約0.9–1.2%短縮した一方、block 1を約3%悪化させた。shape selectorを
追加しても50-frame codec全体の上限は約0.035 ms未満で、2% gateに届かない。
全候補はproductionとbitwise一致し、WGPU error 0だったため、K32を維持する。

artifacts:

- `/home/sanzentyo/benchmark-artifacts/irodori-v4-pointwise-k16-20260814-attempt1`
- `/home/sanzentyo/benchmark-artifacts/irodori-v4-pointwise-k-tile-20260814-attempt1`

### bias/alphaのvec4 loadとvec4 Snake

各threadが連続4 channelを所有するdirect F16 shaderで、bias/alphaを`vec4<f16>`として読み、
residual加算とSnakeを`vec4<f32>`で評価した。最初のbuildはresidualの型変換不足により
warmup 1でvalidation errorとなり、修正後に5 warmup + 10 measured + 20 stage repeatを完走した。
pointwise合計のmedianはK32 control `4.461 ms`に対し候補`4.548 ms`（約2.0%悪化）、
全stageも`14.045 ms`から`14.123 ms`へ悪化した。accuracy gateはSNR `56.063 dB`、
max abs `3.41796875e-3`で通ったが、演算順序が変わりhashは`7067af0f…`になったため不採用。

artifact:
`/home/sanzentyo/benchmark-artifacts/irodori-v4-pointwise-vec-params-20260814-attempt1`

演算順序の影響を切り分けるため、bias/alphaのbinding/loadだけを`vec4<f16>`にし、加算とSnakeは
従来のscalar順序を保つ候補もA-B-Aで測った。前control、候補、後controlのpointwise合計medianは
`4.488 / 4.433 / 4.436 ms`で、候補時刻の線形補間controlは`4.462 ms`。差は`-0.029 ms`
（0.65%）に留まった。全stageも候補`14.025 ms`に対し補間control`14.033 ms`で、差は
`-0.008 ms`（0.06%）だった。productionとbitwise一致したが2% gate未満のため不採用。

artifact:
`/home/sanzentyo/benchmark-artifacts/irodori-v4-pointwise-vec-param-loads-20260814-attempt1`

## 次の候補

小さなselector、guard、既存algorithm/loader familyは今回で打ち止めとする。次の有力候補は、
汎用loader置換ではなく実際にhalo重複を除くCubeK fixed-k7 LHS loader、または
pointwise projection/residual/next-Snakeを
backend operationとしてまとめる構造変更である。どちらも50-frameで2%以上短縮した場合だけ
六長・5 fresh process・NVML campaignへ昇格する。

source auditでは、RTXで選ばれるmulti-row blueprintのLHS stageは`M=128, K=32`で、現行im2colの
K順はkernel-majorである。channelsが384以上なので各K stageは単一kernel plane内に収まり、
既存loader typeの交換だけではkernel plane間のhalo重複を再利用できない。fixed-k7候補は
channel-major K viewと対応するRHS packを組にするか、kernel planeをまたぐLHS cacheを持つ必要がある。
前者をprofile-only routeとして小さくscreenするのが次の実装順序となる。
interior stageの座標列挙では、4096個の展開LHS要素に対する重複しないsource要素はdilation
1/3/9で平均約`721 / 775 / 934`個、global readの理論上限は`5.68x / 5.29x / 4.39x`削減となる。
これはshared writeとCMMA量を変えないload上限であり、latency短縮率の予測値ではない。

pointwise側は現WGSLがprojection、residual、次Snake、raw/activatedのdual outputまで既に1 dispatchへ
融合している。backend operation化でさらに進めるには、現行のpure one-output post-cast epilogueではなく、
追加output bindingを所有できるCubeK `GlobalWriter` contractが必要であり、fixed-k7 loaderより変更面が広い。

## 最終production確認

全候補を戻してT64/O96/K32とCMMAを再ビルドし、5 warmup + 10 measured + 5 stage repeatを
完走した。device-complete medianは`15.266 ms`、readback-complete medianは`16.055 ms`。
waveform hashは採用済みrouteと同じ`113ba560546d82a3112332ac67b3cea5d5b83b407109d3df3817e5b82b609e05`、
SNR `56.074 dB`、max abs `3.41796875e-3`、WGPU error 0だった。

final artifact:
`/home/sanzentyo/benchmark-artifacts/irodori-v4-post-multi-row-final-20260814-attempt1`
（profiler binary SHA-256: `4bce73b82e76f335e7da9b0829979423fd602d8ce35fcf65d0b915210da5eb73`）

verification:

- `cargo test --lib --features inference,codec`: 508 passed、0 failed、17 ignored
- partial-tile Vulkan GPU test: 1 passed
- k7 route bitwise Vulkan GPU test: 1 passed
- `cargo clippy --all-targets --features inference,codec,cli,profile -- -D warnings`: pass
- `cargo fmt --all -- --check`: pass
- `uvx ruff check scripts`: pass
- 拡張`cargo test --all-targets --features inference,codec,cli,profile`はlib 511 passed / 0 failed /
  18 ignored、CLI bin 30 passedまで完走した。その後bench targetが未配置の
  `target/dacvae_weights.safetensors`を要求して停止したため、fixture欠如としてコード失敗と分離する。

その後のloader-family screenをすべて戻した同時刻production controlも、5 warmup + 10 measured +
20 stage repeatを完走した。device-complete medianは`15.458 ms`、readback-complete medianは
`16.334 ms`、k7合計medianは`7.287 ms`、全stageは`14.119 ms`。waveform hash、accuracy、
WGPU error 0はいずれも上記final確認と一致した。保存binary SHA-256は
`d240606b86ba8ad28704d863deed49cec00ad2b3083df95d548fd3dea0284fcd`。

最後に同じproduction binaryを5 fresh process、各5 warmup + 10 measuredで再確認した。

| session | device median ms | readback median ms |
|---:|---:|---:|
| 1 | 15.400 | 16.284 |
| 2 | 15.182 | 16.027 |
| 3 | 16.334 | 19.145 |
| 4 | 15.323 | 16.247 |
| 5 | 15.388 | 16.345 |
| median of medians | **15.388** | **16.284** |

session 3はhost/readback側を含む外れ値を持つため、sampleを他processへpoolせず事前定義どおり
session medianのmedianを採用した。全50 measured waveformは上記production hashと一致し、全processで
WGPU error 0だった。artifactは
`/home/sanzentyo/benchmark-artifacts/irodori-v4-post-multi-row-fresh-final-20260814-attempt1`。

本レポートが参照する20 artifact directoryはすべて存在し、各`SHA256SUMS`を再検証済み。
