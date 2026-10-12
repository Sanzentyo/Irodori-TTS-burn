# RTX 5070 Ti Laptop: decoder block境界Snake融合（2026-08-19）

## 結論

decoder内の`res0 -> res1`と`res1 -> res2`は既にpointwise projection、bias、residual、
次のSnakeを1 dispatchへ融合していたが、decoder block間の同じdataflowだけは
`pointwise -> F16 raw store -> standalone Snake -> ConvTranspose`のまま残っていた。

C384とC192の二つのblock境界について、pointwise kernelが次block用のSnake activationだけを
直接生成する経路を実装した。F16ではpointwise residualを一度F16へ丸めてからF32へ戻してSnakeを
評価するため、従来のstorage境界を保つ。不要になったraw output allocation、raw write/read、
standalone Snake 2 dispatchを除去する。

50 latent framesの5 fresh sessionでは、device-complete中央値の中央値が`15.345 ms`から
`15.234 ms`へ`0.111 ms`（`0.72%`）短縮した。productionへ昇格した最終binaryでは
`15.196 / 16.120 ms`（device/readback）だった。直前の採用済みproduction 5-session値
`15.388 / 16.284 ms`に対して`1.25% / 1.01%`短い。

六長すべてでstandalone controlとwaveform SHA-256がbitwise一致し、WGPU uncaptured errorは0だった。
685 frames相当も3 fresh sessionすべてで融合側が速かった。最終production測定のNVML peakは
`1,196 MiB`で、追加の
persistent weightやworkspaceは持たない。

## 演算境界

従来のblock 1/2終端は次の2 dispatchだった。

```text
pointwise projection + bias + residual -> F16 raw NCL
F16 raw NCL -> F32 Snake -> F16 activated NCL
```

採用経路は次の1 dispatchである。

```text
F32 pointwise accumulator
  -> bias + residual
  -> F16 round
  -> F32 Snake
  -> F16 activated NCL
```

最終raw値は次blockのSnake以外にconsumerを持たないため保存しない。これは演算順序の変更ではなく、
owned intermediateのlifetimeをgraphから除いたものである。C768のblock 0終端は現在のdirect
pointwise kernel contract外なので、従来境界を維持する。

launcherはoutput contractを`Raw`、`Pair`、`ActivatedOnly`へ分離し、それぞれbinding数を
5、7、6として検証する。`ActivatedOnly`はraw outputのpage-size検査やallocationを行わない。
unsupported dtype、layout、shape、device、resource limitでは従来のpointwise + standalone Snakeへ
fail closedする。

## 初版を採用しなかった理由

最初のprofile-only実装は既存のtwo-output pair kernelを流用し、F32 residualから直接Snakeを評価した。
これはF16 raw store後にSnakeを読む従来境界と丸め位置が異なり、waveform hashが変わった。また不要な
rawとactivatedを両方writeしたため、50-frame device中央値は`15.604 ms`でcontrolの
`15.329 ms`より遅かった。

この結果は成功campaignへpoolしていない。採用版は専用のactivated-only kernelでpost-cast意味を
明示し、全比較でcontrolとbitwise一致した。

## 50-frame fresh session

同じ最終候補binaryを用い、各processで5 warmup + 20 measuredを実行した。candidate/controlは交互に
実行し、automatic retryは0とした。

| session | fused device ms | standalone device ms | fused readback ms | standalone readback ms |
|---:|---:|---:|---:|---:|
| 1 | 15.240 | 15.337 | 16.106 | 16.257 |
| 2 | 15.232 | 15.319 | 15.972 | 16.168 |
| 3 | 15.205 | 15.345 | 16.146 | 16.280 |
| 4 | 15.234 | 15.366 | 16.126 | 16.205 |
| 5 | 15.284 | 15.365 | 16.040 | 16.217 |
| median | **15.234** | **15.345** | **16.106** | **16.217** |

5組すべてで融合側が速い。全sampleのwaveform SHA-256は
`113ba560546d82a3112332ac67b3cea5d5b83b407109d3df3817e5b82b609e05`、SNRは
`56.074 dB`、max absは`3.41796875e-3`だった。

## 長さ回帰

旧F32 accuracy campaignのpinned latent/waveform fixtureをF16 executionへ入力し、同じbinaryの
fused/standaloneを比較した。低warmupの単発wallにはpipeline/clock外れ値があったため性能値を
session間でpoolせず、ここではbitwise accuracyを主張する。

| audio相当 | latent frames | fused/control waveform SHA-256 |
|---:|---:|---|
| 1.8 s | 45 | `9dc44ecfbffe71a26049cd94aee9530b56080e4a3c22dd969b526682a2f03f33` |
| 4.48 s | 112 | `ea84a605ce16d178b3b22e7d89dfd675209c9e851df1d66e7e4140b54b305d65` |
| 10.2 s | 255 | `4c54d0059ac05758a9a9002c46fd2e4f26902d7b6485f62b1947145b176da902` |
| 13.32 s | 333 | `45ca463082a269f0849ae7c591e4f78e358a290a6d47f6cc99cc62f6f398fafa` |
| 19.56 s | 489 | `adf611e8c111e4e617f023423d517dd6645b3aad36e93a6584a6d7df2752d1c2` |
| 27.4 s | 685 | `cae7d8471e7fecbf6bb05bdc12eef480129fed1598bcafc538c88a94525a776a` |

685-frame性能はwarmup 5 + measured 5を3 fresh sessionで再確認した。

| session | fused device ms | standalone device ms |
|---:|---:|---:|
| 1 | **222.071** | 229.693 |
| 2 | **222.875** | 236.079 |
| 3 | **224.519** | 236.767 |

F32 50-frame smokeもcandidate/controlで同じwaveform SHA-256
`dcf32ebeb57f1213e59748c96604a04b89904ed12095c8b7e061d63e7cec1516`、SNR
`113.197 dB`、max abs `5.2601e-6`だった。

## final production non-regression

production昇格後のbinaryを5 fresh process、各5 warmup + 10 measuredで測定した。

| session | device ms | readback ms |
|---:|---:|---:|
| 1 | 15.196 | 16.161 |
| 2 | 15.138 | 16.191 |
| 3 | 15.269 | 16.120 |
| 4 | 15.216 | 16.017 |
| 5 | 15.155 | 16.057 |
| median | **15.196** | **16.120** |

- final profiler binary SHA-256:
  `8a3a2ee510c905f960d879388f2afcb30b1426e0f9ae3724b59bf515e1271718`
- NVML: 449 samples、peak used `1,196 MiB`、minimum free `10,579 MiB`
- GPU: NVIDIA GeForce RTX 5070 Ti Laptop GPU、12,227 MiB
- driver: `595.71.05`
- WGPU: Vulkan discrete adapter 0
- CUDA/NVML index: 0
- PCI bus ID: `00000000:01:00.0`

## portability

実装はWGSL F32/F16の同一source設計で、WGPUのVulkan、Metal、DX12へcompile可能な演算だけを使う。
型付きoutput contract、post-cast意味、fallback、shape validationはbackend非依存の設計である。
ただし性能とpipeline compileを実測したのはVulkan/NVIDIAだけであり、Metal/DX12で同じ短縮率を
主張しない。backend別tileやGPU名による分岐は追加していない。

## artifact

- implementation commit: `d34a7c4`
- `/home/sanzentyo/benchmark-artifacts/irodori-v4-cross-block-fusion-20260819-attempt1`
- F16 oracle SHA-256:
  `08663e52df6c63eea7ebf5ae2ba35778bdb5bc7d20cd06e430e855a86174229e`
- codec SHA-256:
  `b14a25da5d68bf779d8c44a40706ddc7c4819cb60489b2a80a6408ca03c514fb`

旧`/tmp` artifactや旧session値は今回の集計へpoolしていない。
