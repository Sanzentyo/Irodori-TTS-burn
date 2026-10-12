# RTX 5070 Ti / strict-FP32 40-step RF structural screen (2026-08-24)

## 結論

2026-08-24 09:00 JSTまでの探索では、WGPUはPythonを上回らなかった。489-frame
voice-design、40 Euler evaluations、B3×20 + B1×20、12 layers / 480 block callsの
device-complete中央値は、最終hot-path確認のWGPUが4.857645 s、正式Python比較値が4.548211 sである。
残差は0.309 s、6.80%である。

一方、SDPAのhead-major出力をtoken-major gated tensorへ物理化せず、`wo` projectionと
block residualまで一dispatchで処理する汎用境界削減を実装した。同じCubeCL cacheを使った
same-binary相当A/Bでは4.844138 sから4.840792 sへ3.35 ms（0.069%）短縮し、最終audio
SHA-256も完全一致した。このrouteはexact-device tuner候補へ追加し、NVIDIA family defaultには
昇格していない。

今回明らかになった支配要因はdispatch submit回数ではなく、大規模matmulのthroughputと
CubeCL内部autotuneの選択安定性である。特にSDPAの`[512,512] @ [512,64]`相当の`P@V`
matmulは、同一deviceでもCubeCL cacheのfastest indexが変わり、40-step RFを約145--170 ms
動かし得た。Irodoriのroute profileだけをsealしても、内部matmul選択が固定されなければ
exact profile全体は固定できない。

## 固定条件

- source base: `a866baa0f33e3f02cdc5a42013a315afb9cccade`
- model revision: `e4aaac4df355ff560dcd35e0dae272c3a759317b`
- model SHA-256: `5863c986345d9f6d20b7d8748fee1af02079c5161cf0c9e52557da0a0c378593`
- decoder-only codec SHA-256: `1b1ceb3f620525cf4252af508c0fde80e3779582d47fc7fc879410d2e4abe231`
- fixture SHA-256: `9a1e00e667f960983b62ebc9188c6b430acf0c00d0721ef9ffdf8fc8b9fd4b3f`
- final measured binary SHA-256: `ae2e8e20af11ac7473c6cc88731a681015f387d9d61c4234161c74c38b2b2ea0`
- GPU: NVIDIA GeForce RTX 5070 Ti Laptop GPU, PCI `0000:01:00.0`
- driver: NVIDIA 595.71.05; WGPU adapter 0; Vulkan
- VRAM: 12,227 MiB total
- precision: strict FP32; TF32 off; autocast off
- RF: 40 Euler evaluations、forward batchesは`[3×20, 1×20]`、effective rows 80
- voice/length: official Voice Design相当caption CFG 4、489 latent frames、19.56 s audio
- boundary: pre-stage device syncからRF device completionまで。Pythonの同じdevice-complete境界と比較
- allocator / residency: `ExclusivePages`、exact-manifest RF weights、decode-only codec、all-resident

旧campaignの計測値や`/tmp` artifactは今回のsession poolへ混ぜていない。Python値だけは同一入力・
同一40-step意味論を満たす直前の正式fresh campaignを比較基準として明示的に引用した。

## 採用した実装

`PostSdpaRoute::DirectOutputResidual`を追加した。成立条件は型付きroute tableとlauncherの双方で
fail-closedに検査する。compact gate `[B,S,D]`とportable QKV+gate `[B,S,4D]`を同じkernelの
明示stride/offset contractで扱い、特定のQKV materialization routeへ依存しない。

```text
SDPA [B,H,S,Dh]
  + compact attention gate [B,S,D]
  + packed wo [D,D]
  + residual [B,S,D]
  + block gate [B,D]
        ↓ one dispatch
final block output [B,S,D]
```

これにより、従来の次の境界を削除する。

```text
head-major → token-major layout + attention gate
token-major gated tensor → wo projection + residual
```

入力はread-only storage binding、出力だけread-writeで宣言する。shape、stride、dtype、device、
binding alignment、shared-memory、workgroup/grid limit、packed-weight admissionを全てlaunch前に検査し、
一つでも不成立なら従来の`FusedLayoutGate` routeへ戻る。F16は未検証なのでこのlauncherはstrict
F32だけを受理する。

route ABIは`v4-dit-route-6`へ更新した。direct variant自身がpacked `wo` layoutを要求し、portable
fallbackがsource weightを使うcellでは両layoutを保持する。既存のsealed manifestは新kernelを暗黙に
選ばず、exact-device tunerだけがB1/B2/B3の各cellを個別に候補化できる。

## 計測結果

### Direct SDPA output fusion

同じfast CubeCL databaseを復元したA/Bで比較した。

| route | measured RF (s) | median (s) | audio SHA-256 |
|---|---|---:|---|
| direct candidate | 4.850427 / 4.840792 / 4.839569 | 4.840792 | `fe1abefe...` |
| restored control | 4.854338 / 4.828795 / 4.844138 | 4.844138 | `fe1abefe...` |

差は小さいが、1 requestあたり480 block callsで中間tensorとdispatchを一つずつ減らし、数値結果を
変えないため、構造候補として保持する。別processで得た5 measuredの再確認中央値は4.849866 s、
read-only binding版の3 measured中央値は4.869431 s、環境変数lookupをhot pathから除いた最終
5 measuredは4.857645 sだった。process間の約20 ms差はclock/process noiseの範囲で、binding宣言や
enum-only route判定による回帰とは判定しない。

### WGPU submission batching

harnessが`RuntimeOptions { tasks_max: 32 }`を明示していたため、従来の
`CUBECL_WGPU_MAX_TASKS`環境変数screenは実際には無効だった。`--wgpu-tasks-max`を
`NonZeroUsize`として追加し、schema 12 raw JSONへ値を記録して測り直した。

| actual tasks_max | measured count | RF median (s) | disposition |
|---:|---:|---:|---|
| 32 | 5 | 4.849866 | keep |
| 64 | 5 | 4.855172 | reject、+5.31 ms |
| 128 | 3 | 4.900048 | reject、+50.18 ms |

したがってRTX 5070 Tiの今回のgraphでは32を維持する。`WgpuExecutionPolicy`のproduction Rust APIは
既に型付き`tasks_max`を持つため、他GPUは同じAPIでexact-device測定できる。

### Rejected structural candidates

| candidate | RF median | reason |
|---|---:|---|
| handwritten MLP expand R128 | 5.550877 s | 既定より約14.6%遅い |
| restored CubeK compressed SwiGLU | 5.079317 s | 現Burn/CubeK matmul改善後は既定graphより遅い |
| B3 handwritten T64 expand | 約4.94 s | default CubeK expansionより約0.10 s遅い |
| vectorized compressed writer | invalid | CubeK stage linear orderをlogical adjacent pairとみなせず、partial-tile CPU oracle不一致・process内hash不一致 |

vectorized writerは性能不採用ではなくcorrectness rejectであり、全変更を撤回した。正しく実装するには
logical coordinateからstage offsetへの逆写像、またはpair-aware output stage layoutが必要である。

## AccuracyとVRAM

direct route on/offのaudioはbitwise同一であり、今回の融合によるaccuracy divergenceはゼロである。
Python referenceと現WGPU routeの差はSNR 67.91 dB、max abs 0.002753、RMSE
`6.46e-5`、cosine 0.999999919だった。これは今回ユーザーが許容した演算順序差として性能探索からは
除外しなかったが、従来の80 dB numerical reproducibility gateには通らない。音質同等性を主張する
聴覚試験は実施していないため、production accuracy receiptではこの事実を明示する必要がある。

final exact-manifest runのpersistent RFは3,417,084,544 B、codecを含むconsumer後in-useは
3,555,988,096 Bだった。direct routeは新しいpersistent weightを要求せず、削除した中間tensorの
live rangeだけを短くする。all-resident reserved 6.4 GB前後はallocator poolでありpersistent in-useと
混同しない。

## Pythonとの差が残る理由と次の優先順位

現行WGPUは直前の正式WGPU 5.413518 sから4.857645 sへ10.27%改善したが、Python 4.548211 sには
まだ0.309 s届かない。dispatch一つの削減が約3 ms、submission batchを倍にしても改善しなかったため、
残りを小さなWGSL epilogue追加だけで埋める見込みは低い。

次は次の順で進める。

1. CubeCL内部matmul候補をIrodoriのsealed exact-device receiptへ昇格する。候補indexだけでなくstable
   algorithm ID、shape/stride、driver、source hash、実測sampleを保存し、同じ`P@V`で約145--170 ms
   変わる選択揺れを排除する。
2. Burnの現default matmul candidate集合そのものへtyped compressed-output epilogueを接続する。
   現在のSimpleUnit専用SwiGLU routeではfull expansionを消せてもmatmul本体が遅い。defaultと同じ
   high-throughput routineを使い、pair-aware stage layoutで`[M,2H]`を一度も物理化しないことが必要。
3. raw-only graphをBurn 0.22 custom Fusion providerへ段階移行する。QKV、SDPA、SwiGLUの大型matmulは
   tuned CubeKを維持し、そのepilogueと周辺elementwiseだけをproviderへ取り込む。これは単純な巨大
   monolithic shaderよりportableで、Metal/DX12/他Vulkanにも同じ候補機構を使える。
4. GPU timestampで12 layers×B3/B1のmatmul algorithm、writer drain、allocator live rangeを同一request
   内で採取する。CPU wallやstageごとの強制syncを最適化判断へ使わない。

NVIDIA/Appleのfamily profileは候補順のpriorに留める。AMD、Intel、旧Apple、他NVIDIA世代では
portable fallbackからexact-device tunerが承認し、上記algorithm IDを含むreceiptが揃うまでsource
weightを解放しない。

## Artifacts

fresh campaign root:
`/home/sanzentyo/benchmark-artifacts/irodori-v4-rf-python-beat-20260824`

root `SHA256SUMS` SHA-256:
`dd6c8e6085e38a28ac082cf42a86ee3a8e35bdf49d7ce02af3a5c98ac4317c67`

- direct A/B: `direct-sdpa-output-candidate-s2/`、`direct-sdpa-output-control-restored-s1/`
- final direct runs: `direct-output-tasks64-final-s1/`（実値32）、
  `direct-output-readonly-final-s1/`、`direct-output-no-hotpath-env-final-s1/`
- generalized portable-gate smoke: `direct-output-portable-gate-final-smoke/`
- actual batching: `direct-output-tasks64-real-s1/`、`direct-output-tasks128-real-s1/`
- rejected: `mlp-expand-r128-candidate-s1/`、`b3-compressed-direct-output-tasks64-s1/`、
  `b3-handwritten-tasks64-s1/`、`b3-compressed-vector4-max-s1/`
- route profiles: `profiles/`

各採用/比較directoryにraw `result.json`、stdout/stderr、CubeCL database、audio、`SHA256SUMS`を置いた。
失敗したvectorized writerの成果値は採用値へpoolしていない。
