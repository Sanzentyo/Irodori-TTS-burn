# RTX 5070 Ti / plane SDPA・inner seal・Fusion・GPU timestamp (2026-08-24)

## 結論

要求された4項目を実装した。RTX 5070 Tiでの採否は次のとおりである。

1. CubeCL DSLのstrict-FP32 plane SDPAは、32-lane `plane_sum`とonline softmaxを使う独立route候補として
   実装し、実機mask smokeを通した。ただし489-frame Voice Design、40-step RFでは既定Burn SDPAの
   4.787814 sに対し6.877308 sで、43.642%遅かった。RTX defaultには採用しない。
2. CubeCL内部autotuneは、fastest indexだけでなく選択候補名、全候補集合digest、exact key、route ABI、
   route-profile digestを持つreceiptへsealした。候補ごとのcacheに加え、最終route集合を一つのfresh cacheで
   再実行して単一のcomposed receiptを作る。これがない状態は
   `PreparedModel<RoutesSealed>`へ遷移できない。
3. Burn 0.22 custom Fusion providerをWGPU-onlyのopt-in featureとして実装した。最初のproviderは
   `sigmoid(gate) * gate * value`を完全なdependency patternとして認識し、Burnのportable elementwise
   codegenへ委譲する。productionの`WgpuRaw`は変更せず、未承認のFusion graphを暗黙に有効化しない。
4. RF 40-stepの4,320区間を、stageごとのdevice syncなしでdeferred GPU timestamp化した。全区間が
   hardware device timestampであり、支配項はMLP expand、QKV projection、SDPA、MLP contractだった。

今回のplane候補は速度rejectである。候補を保持する理由は、CubeCL DSLなのでMetal/DX12/他Vulkanでも
同じsourceをcompileでき、exact-device tunerが各adapterで個別に採否を決められるためである。ただし
32-lane plane operationを満たさないadapterではlaunch前にfail-closedし、portable Burn SDPAへ戻す。

## Pinsと境界

- branch: `codex/v4-post-seal-priority-1-4`
- measurement implementation: `92a8e50005227da657f3c8c115cb8bf6b22fa86e`
- composed-seal follow-up: `6507cfe7eb342043080c4c0008927d39fc5c8fae`
- final source: `8da3089efc7f0e5c088fcadd154e58261fad42e4`
- measured binary SHA-256: `3bfb07c53003b44d44edc7d1b07bbaa4856aac29281ca101bbf2ae7d7302f982`
- GPU: NVIDIA GeForce RTX 5070 Ti Laptop GPU
- driver: `595.71.05`
- WGPU adapter: index 0、Vulkan
- CUDA/NVML index: 0
- PCI bus ID: `00000000:01:00.0`
- physical VRAM: 12,227 MiB、campaign前available: 11,774 MiB
- model revision: `e4aaac4df355ff560dcd35e0dae272c3a759317b`
- model SHA-256: `5863c986345d9f6d20b7d8748fee1af02079c5161cf0c9e52557da0a0c378593`
- decoder-only codec SHA-256:
  `1b1ceb3f620525cf4252af508c0fde80e3779582d47fc7fc879410d2e4abe231`
- fixture SHA-256: `9a1e00e667f960983b62ebc9188c6b430acf0c00d0721ef9ffdf8fc8b9fd4b3f`
- precision: strict FP32、TF32 off、autocast off
- work: 40 Euler evaluations、B3 x 20 + B1 x 20、12 layers、480 block calls
- request: 489 latent frames、19.56 s audio、Voice Design、caption CFG 4.0
- RF latency boundary: pre-stage device syncからRF device completionまで。codec/readbackを含めない
- timestamp boundary: 各stage closureのGPU stream timestamp。全request完了後にまとめてresolve

測定値はこのfresh campaignだけから集計した。過去の`/tmp` artifactや旧WGPU測定値をsessionへpoolして
いない。Python 4.548211 sは同一request・同一40-step意味論・同一device-complete境界の直前formal
campaignを位置づけのため引用した値であり、今回のfresh session値ではない。PyTorchとWGPUはsame
semantic workであり、same operator graphではない。

## CubeCL DSL plane SDPA

`src/kernels/plane_sdpa.rs`は、1つの`(batch, head, query)`を1 planeへ割り当てる。D=64なので各laneが
2要素を保持し、dot productを`plane_sum`でreduceする。score matrixは作らず、key loop中のmax、sum、
value accumulatorだけをregisterへ保持する。

```text
one 32-lane plane
  q[D=64] x k[D=64]
      -> plane_sum(F32)
      -> online softmax(F32)
      -> value accumulator(F32)
      -> output[D=64]
```

成立条件はhost launcherで検査する。

- q/k/vはrank 4、strict F32、同一device
- maskはrank 2、U32-backed bool、`nonzero = attend`
- head dimensionは64
- plane min/maxはともに32、`Plane::Ops`あり
- cube/grid/binding limitを満たす

全mask falseではNaNを出さずzero rowを返す。条件外は`Option::None`でportable routeへ戻る。
`SdpaRoute::CubeClPlane`はexact B/S route tableの候補で、GPU名による選択はしない。

### 40-step A/B

同一binary、同一campaign内のCubeCL cacheを使った。baseline側がfresh tuneし、plane側は同じ内部matmul
cacheをrestoreした。warmupを測定値から分離している。

| SDPA route | measured RF (s) | median (s) | baseline比 | disposition |
|---|---|---:|---:|---|
| Burn fallback | 4.786036 / 4.789593 | **4.787814** | reference | keep |
| CubeCL plane | 6.875079 / 6.879537 | **6.877308** | +43.642% | reject |

planeはscore matrixを除くが、RTXではBurn/CubeKの大規模matmul throughputに勝てなかった。これは
「temporaryを減らせば必ず速い」わけではなく、1 query rowを1 planeで逐次key走査する構造がtensor-core
型matmulよりcompute-boundになった結果と推定する。GPU timestampのSDPA自体も全profiled stageの
20.37%を占めるため、attentionは重要だが、このscalar/plane構成はRTX向け解ではない。

### Accuracy differential

既定Burn routeをreferenceにした同一入力差分である。ユーザー方針どおり演算順序差だけを自動reject理由
にはしないが、性能rejectとは独立に数値を保存した。

| boundary | max abs | RMSE | SNR | cosine |
|---|---:|---:|---:|---:|
| first RF forward | `3.397e-6` | `4.933e-7` | 126.751 dB | 0.999999999999857 |
| final RF latent | `1.037e-3` | `2.279e-5` | 93.470 dB | 0.999999999775648 |
| decoded waveform | `7.291e-4` | `1.720e-5` | 79.402 dB | 0.999999994261988 |

局所差は小さいが40-step trajectoryで累積する。今回は速度が明確に負けるため、accuracy thresholdを
緩めて採用する理由はない。

## 内部matmul選択のsealed receipt

CubeCLのraw `fastest_index`は候補順変更に弱いためauthorityにしない。新しい
`SealedInnerKernelSelection`は次を保存する。

- exact CubeCL cache key
- selected candidate indexとstable candidate name
- 全candidate result vectorのcanonical SHA-256

`SealedInnerKernelReceipt`はさらにroute ABI、route profile SHA-256、raw recorder SHA-256、選択vector全体の
SHA-256を持つ。JSON object keyをcanonical順へ並べてdigestし、duplicate key、unknown index、tamper、
空receiptをfail-closedにする。

候補測定と最終compositionは分けた。

```text
candidate route
  -> candidate-local fresh CubeCL cache
  -> record + seal
  -> warm/measured sessions restore the same cache

all selected routes
  -> one fresh composed CubeCL cache
  -> every exact B/S caseを40-step validation
  -> recorder fragmentsを一つのselection vectorへseal
  -> ExactRouteManifest
```

型状態遷移は次である。

```text
PreparedModel<LayoutsSelected>
  -- ExactRouteManifest + matching composed receipt -->
PreparedModel<RoutesSealed>
  -- source/layout lock -->
PreparedModel<ProfileLocked>
```

候補ごとのreceipt集合を合成済みreceiptの代用にはしない。異なるcandidate-local cacheで選ばれた内部
algorithmを同時に使ったと誤認するためである。composed receiptのSHAが一致しなければlockできず、
`ProfileLocked`にはtune APIを実装していない。

## Burn custom Fusion provider

feature `fusion-provider`を追加し、次をopt-inで公開した。

- `WgpuFusion = burn_fusion::Fusion<WgpuRaw>`
- `register_irodori_fusion_providers()`
- `SwiGluPostprocessProvider`

providerは最初のTensor operationより前に、exactな`WgpuRuntime`型へ登録する。patternは
`sigmoid(gate)`、その出力とgateのmultiply、続くvalue multiplyのdependency IDを検査する。途中に別operation
が入る、不完全、またはdependencyが異なる場合は`Closed`となる。complete時だけreadyになり、generic
elementwise providerとのtieを1点だけ上回る。

現在のexecutorはBurnのelementwise engineへ委譲する。これによりcustom providerのregistry、stable name、
serialized state、fallback境界を先に固定しつつ、production raw graphと数値・速度・VRAMを変えない。
大規模projection matmulやraw WGSL executorを同時に移す変更ではない。次段階でcustom epilogue executorを
差し替えても、public registration/receipt contractは維持できる。

featureを有効にしてもCPU/CUDA backendは追加されず、hardware backendはWGPUだけである。provider sourceは
WGPUのVulkan/Metal/DX12で共有できるが、このcampaignで実測したのはVulkan/NVIDIAだけである。

## GPU timestampによる局所化

従来の`IRODORI_RF_DETAIL_PROFILE=1`はstageごとにdevice syncするため、dispatch batchingとtemporary lifetimeを
変えていた。新しいcollectorは各closureを`ComputeClient::profile`で囲み、4,320個の
`ProfileDuration`をrequest後までresolveしない。receiptはtimestamp sourceを各区間に保存する。RTXでは
4,320 / 4,320が`device_timestamp`で、system-clock fallbackは0だった。

profiled stage合計は4.657997 sである。timestamp queryを含むdiagnostic request全体は4.988082 sなので、
この値を通常latencyとして使わない。通常A/Bは別processのdevice-complete 4.787814 sである。

| component / stage | calls | GPU total (s) | profiled比 | mean/call (ms) |
|---|---:|---:|---:|---:|
| MLP expand | 480 | 1.512602 | 32.47% | 3.151 |
| attention QKV+gate | 480 | 0.977353 | 20.98% | 2.036 |
| SDPA | 480 | 0.948612 | 20.37% | 1.976 |
| MLP contract | 480 | 0.827259 | 17.76% | 1.723 |
| direct output projection | 480 | 0.295749 | 6.35% | 0.616 |
| QKV materialization | 480 | 0.040984 | 0.88% | 0.085 |
| pitched SwiGLU | 480 | 0.032727 | 0.70% | 0.068 |
| AdaLN attention + MLP | 960 | 0.022711 | 0.49% | - |

B3半分のstage合計は3.423997 s、73.51%、B1半分は1.233999 s、26.49%だった。残差は小さなWGSL
elementwiseやsubmit回数ではなく、B3の4つの大規模stageへ集中している。上位4 stageだけでprofiled時間の
91.58%を占める。今後の優先順位は次である。

1. B3 MLP expandのdefault CubeK matmulへcompressed-output epilogueを接続し、full `[M,2H]`を書かない。
2. QKV projection matmulの後処理をprovider/typed epilogueへ移し、packed Q/K/V/gateへ直接storeする。
3. SDPAはplane逐次版ではなく、tuned matmul/Flash kernelの内部algorithmをsealed receiptで安定化する。
4. MLP contractとdirect output projectionは現行packed weightを維持し、writer drainをtile-awareにする。

SDPA allocator probeではB3/B1ともstage内`bytes_in_use`追加peakは0 Bだった。B3開始時
4,243,562,816 B、B1開始時4,212,839,360 Bが各stage peakと一致した。7 reservation eventはあったが既存pool
内で処理されている。従って、新しいpersistent SDPA workspace arenaを置く根拠はない。

## VRAM

formalなresidency確認は`exact-manifest`で別fresh processを使った。

| stage | bytes in use | MiB |
|---|---:|---:|
| RF resident | 3,417,207,424 | 3,258.90 |
| all-resident after consumer | 3,556,110,976 | 3,391.37 |
| all-resident reserved after consumer | 6,397,925,184 | 6,101.54 |

100 ms NVMLのdevice peakは6,287 MiBだった。12 GiB all-residentは維持される。plane A/Bとroute tuningでは
全fallback layoutを保持する`production-prepared`を使ったため、persistent in-use 4,175,424,640 Bを
formal exact residencyとして引用しない。

## 性能の位置づけ

今回のfresh Burn baseline screenは4.787814 sで、直前formal Python 4.548211 sより0.239603 s、5.27%遅い。
直前formal WGPUの6.765%差より小さいが、今回は1 fresh process・2 measuredのscreenなので正式な更新値とは
しない。差が縮んだ主因候補はcampaign-local fresh CubeCL internal selectionであり、この変動こそ
inner selection sealが必要な理由である。

GPU timestampは「WGPU内のどこを削るべきか」を局所化するが、PyTorchのoperator graphとの一対一対応を
証明しない。CUDAとの差をstage別に断定するには、同じ入力境界でPyTorch側もNVTX/CUDA eventを挿入した
別campaignが必要である。

## Verification

- `cargo check --all-targets --all-features`: pass
- route autotune focused: 22 passed
- autotune approval focused: 6 passed
- plane SDPA WGPU ignored smoke: 1 passed
- Fusion provider focused: 2 passed
- full lib: 602 passed、21 ignored、0 failed
- `cargo clippy --all-targets --all-features -- -D warnings`: pass
- `cargo fmt --all -- --check`: pass
- `uvx ruff check scripts`: pass

## Artifacts

fresh campaign root:
`/home/sanzentyo/benchmark-artifacts/irodori-v4-plane-fusion-timestamp-20260824-attempt1`

root `SHA256SUMS` SHA-256:
`0176f8258b1f7bea015a38b275cd1860c738460b70f43063c7bc04737cea6d9c`

- A/B raw JSON/log/audio/diagnostic: `baseline/`、`plane/`
- GPU timestamp receipt: `profile/rf-device-timestamps.json`
- timestamp batch aggregate: `profile/by-batch.json`
- exact-residency NVML: `final-exact/nvml.csv`
- environment/model/source/binary pins: `environment/`
- machine-readable A/B summary: `campaign-summary.json`

旧測定値や失敗候補を今回のA/B中央値へpoolしていない。
