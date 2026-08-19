# RTX 5070 Ti Laptop: k7 weight copyとone-dispatch評価（2026-08-14）

## 結論

現productionの12本のk7 operatorは、`weight layout copy`と
`CubeK convolution + bias + Snake`の2 dispatchである。copyだけをGPU timestampで分離すると、
50 latent framesの1 requestあたり中央値約`0.177 ms`、合計約31.4 MiBだった。

copyを消す二つの汎用経路を実装したが、どちらもproductionより遅かった。

1. exact prepared OKIを直接bindingする経路
2. source OIKのstride viewをCubeKが直接読む経路

prepared OKIは現行repack結果とshape、stride、dtype、waveform hashが一致する。それでも同一model
ABBAでは遅く、copy量の大きいweightだけをpreparedにするhybridも5 fresh sessionすべてで負けた。
source OIK直読はRHS vector sizeが低下し、50-frame device-completeが`20.784 ms`まで悪化した。

したがって、単純な「dispatch数最小」をproduction目標にしない。現在のcopyは約`0.177 ms`を使う
一方、直後のCubeKが必要とするvectorized OKIを作る役割を持ち、end-to-endでは残した方が速い。
productionはrequest-time repackを維持する。one-dispatch経路はprofile-onlyの検証候補として残す。

## fresh campaign

- output: `/home/sanzentyo/benchmark-artifacts/irodori-v4-k7-one-dispatch-20260814-attempt1`
- source start: `ca74d0be54f18435e0b2d94200904c520a956878`
- implementation commit: `bd36066f2dfb69c7842af98da99dcb0d208e6f57`
- GPU: NVIDIA GeForce RTX 5070 Ti Laptop GPU、12,227 MiB
- driver: `595.71.05`
- WGPU: Vulkan discrete adapter 0
- CUDA/NVML index: 0
- PCI bus ID: `00000000:01:00.0`
- execution: F16 storage、F32 accumulator/Snake
- codec SHA-256: `b14a25da5d68bf779d8c44a40706ddc7c4819cb60489b2a80a6408ca03c514fb`
- oracle SHA-256: `08663e52df6c63eea7ebf5ae2ba35778bdb5bc7d20cd06e430e855a86174229e`
- final profiler binary SHA-256: `5f598d085451c6193fac61cb07c1afa52d9996e3835f7b424c2d7614c722a258`
- boundary: pre-start syncからdevice completion、またはowned contiguous F32 CPU waveformまで

旧`/tmp` artifactや旧sessionは新campaignへpoolしていない。

## copy単体の実測

`profile_k7_weight_repacks`は、production launchと同じ
`permute_nchw_to_nhwc -> into_contiguous_pitched`だけをCubeCL device timestampで測る。
3 warmup + 10 measuredの全repeatで12 copyを確認した。

| decoder width | weights | bytes/weight | representative device time |
|---:|---:|---:|---:|
| 768 | 3 | 8,257,536 | 0.030–0.037 ms |
| 384 | 3 | 2,064,384 | 0.011–0.014 ms |
| 192 | 3 | 516,096 | 0.006–0.008 ms |
| 96 | 3 | 129,024 | 0.004–0.008 ms |

最終10 repeatの12本合計は`0.170–0.188 ms/request`、中央値`0.177 ms`だった。物理layoutは
全shapeで次の契約を満たした。

```text
source OIK contiguous
  -> logical OKI strides [I*7, 1, 7], RHS vector size 1
  -> materialized OKI strides [I*7, I, 1], RHS vector size 4
```

copy削減だけから求めた理想上限は約1.1%であり、PyTorchとの差全体を埋める規模ではない。

## exact prepared OKI

`PreparedK7Weight`は、現行repackと同じ`into_contiguous_pitched`でprepare時にOKIを作り、source
shape、physical stride、bytesをreceiptとして保持する。decode時はこの型付きweightをCubeKへ直接渡し、
`correct_layout`がcopy不要と判定する。convolution、bias、Snakeを含めてoperator全体が1 dispatchになる。

同一modelにsource OIKとprepared OKIを一時的に併存させ、ABBA/BAABで比較した。preparedを使う
最小weightサイズを変えたsystematic sweepでは、どの条件も明確な勝者にならなかった。最も差が小さかった
`>=516,097 bytes`、つまりwidth 384/768だけをpreparedにするhybridを5 fresh sessionで再確認した。

| session | hybrid device ms | repack device ms | delta ms |
|---:|---:|---:|---:|
| 1 | 15.840 | **15.796** | +0.044 |
| 2 | 16.117 | **16.001** | +0.116 |
| 3 | 16.517 | **16.303** | +0.214 |
| 4 | 16.234 | **16.099** | +0.135 |
| 5 | 16.417 | **16.285** | +0.133 |

5 sessionすべてでhybridが遅い。session medianのmedianはhybrid`16.234 ms`、repack
`16.099 ms`で、prepared側が`0.135 ms`遅い。全sampleは同じwaveform SHA-256
`113ba560546d82a3112332ac67b3cea5d5b83b407109d3df3817e5b82b609e05`で、WGPU errorは0だった。

全preparedでは一つの40 sample sessionで`16.032 vs 15.625 ms`、別の30 sample sessionで
`16.148 vs 15.791 ms`だった。旧single-storage比較の遅化は別model比較だけが原因ではない。

## source OIK direct

CubeKへcaller strideをそのまま渡す`SimpleStridedPostCastEpilogueConv`を追加し、元weight以外の
persistent allocationを持たないone-dispatch routeも実装した。出力hashとaccuracyはproductionに
bitwise一致し、WGPU errorは0だった。一方、logical OKIの最終strideは7であり、materialized OKIの
最終stride 1と異なる。RHSのcoalesced/vectorized load条件を失い、結果は次だった。

| route | device-complete | readback-complete |
|---|---:|---:|
| production repack | 約15.6 ms | 約16.5 ms |
| source OIK direct | **20.784 ms** | **21.867 ms** |

copy `0.177 ms`の削減に対し、convolution側の低速化が約5 msと桁違いに大きい。これは不採用とする。

## production判断と次候補

高速化を維持する条件では、copyを減らすべきではない。productionは次を維持する。

```text
source OIK
  -> vectorized OKI layout copy
  -> CubeK convolution + bias + Snake
```

次にone-dispatchを狙う場合、metadata操作やpersistent化の延長ではなく、CubeKのk7 RHS loader自体を
変更する必要がある。候補は、source OIKのkernel-contiguous成分をvector loadし、workgroup内で
K-major fragmentへ変換するloaderである。合格条件は、materialized OKIと同じMMA tile/vectorized
throughputを保ち、copyを含む現productionよりend-to-endで速いこととする。

同様にfixed-k7 1D halo LHS loader、tile/vector Snake epilogue、shape別CubeK探索は独立候補である。
単にdispatch counterを1へする案は採用条件にしない。

## final production non-regression

最終binaryでproduction routeを5 fresh process、各5 warmup + 10 measuredで再測定した。

| session | device ms | readback ms |
|---:|---:|---:|
| 1 | 15.617 | 16.400 |
| 2 | 15.715 | 16.566 |
| 3 | 15.638 | 16.559 |
| 4 | 15.818 | 16.645 |
| 5 | 15.710 | 16.629 |
| median | **15.710** | **16.566** |

直前採用campaignの`15.585 / 16.486 ms`との差は`+0.80% / +0.49%`で、2% gate内である。
全repeatのwaveform hashとaccuracyは一致し、WGPU errorは0だった。NVML 100 ms samplingは401点、
peak used `1,210 MiB`、minimum free `10,565 MiB`である。直前の1,186 MiBとの差24 MiBは
sampling/allocator変動を含み、production graphはprepared weightを保持していない。

## 2026-08-19: fixed-k7 channel-major halo loader

上記の次候補を実装し、`c7ebac47087fa1a892d254a4efcb854cd589db6d`でprofile-onlyに接続した。
新しいCubeK readerはcheckpoint-native OIKをそのままbindingし、NHWC inputのchannel vectorを
fixed-k7 haloへ一度loadしてから既存MMA stageへ展開する。convolution、bias、post-cast F32 Snake、
F16 storeまでを1 dispatchで実行し、operator前のweight-layout copyは発生しない。

端数M tileでは、同じphysical inputを表す候補のうち最大kernel indexを選び、synthetic output rowを
最小化する。これにより末尾partitionが存在しない次batchへ跨ぐことを防ぐ。C=32の複数K stage、
length=65の端数M tileを含むGPU直接回帰と、96,000 sampleのcodec accuracy gateは通過した。

fresh screening artifactは
`/home/sanzentyo/benchmark-artifacts/irodori-v4-k7-halo-screen-20260819-attempt1`であり、
`SHA256SUMS`を検証済みである。GPUはRTX 5070 Ti Laptop 12,227 MiB、driver `595.71.05`、
Vulkan adapter 0、CUDA/NVML index 0、PCI `00000000:01:00.0`。profiler binary SHA-256は
`02cb5234988efaee98efcf2d9ae8d1c3ef6748b627ab9e7ce4fa17c4759ffcb9`である。旧`/tmp`値や
旧sessionはpoolしていない。

standalone block-boundary、3 warmup + 10 measured、fresh process各1 sessionのscreening結果は次である。
これは大差による早期reject判定であり、5 fresh sessionの採用campaignではない。

| route | device-complete median | readback-complete median | NVML sampled peak |
|---|---:|---:|---:|
| production repack | **15.338 ms** | **16.244 ms** | 1,172 MiB |
| channel-major halo | 26.253 ms | 27.196 ms | 1,180 MiB |

haloはdeviceで`+71.2%`、readbackで`+67.4%`遅い。candidate waveformはSNR `55.794 dB`、
max abs `4.883e-3`、cosine `0.999998696442`でF16 gateを通り、10 measuredのhashも一定だった。
WGPU uncaptured errorは0である。一方、同じbinaryで分離した12本のlayout copyは中央値
`0.174 ms/request`にすぎない。したがって、追加したshared halo、stage展開、K stageごとのbarrierを
償却できず、copy削減の理想上限を大幅に超えるloader costが発生したと判断する。

この経路は「本当に1 dispatch」の正しい比較候補として残すが、productionには採用しない。
次の構造改善では、二段shared stagingを避けてMMAが直接消費できるfixed-k7 stage layout、または
producer/consumer間の中間allocation・dispatchを複数本まとめて消す変更を優先する。

## verification

- `cargo test --lib --features inference,codec`: 507 passed、0 failed、17 ignored
- partial-tile + k7 route Vulkan GPU tests: 2 passed
- vendored CubeK typed-contract test: 1 passed
- `cargo clippy --all-targets --features inference,codec,cli,profile -- -D warnings`: pass
- rustfmt: pass
- `uvx ruff check scripts`: pass
