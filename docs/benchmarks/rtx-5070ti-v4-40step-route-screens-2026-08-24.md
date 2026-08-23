# RTX 5070 Ti Laptop: 40-step RF route screens (2026-08-24)

## 結論

489-frame voice-designのstrict FP32 product pathで、演算順序差を許容したうえでrouteを再評価した。
このcycleでRTX既定へ採用したのは、B1のS333/S489におけるBurn/CubeK SwiGLU graph、
compact Q/gate storage、B1/B3 S489のpitched in-place SwiGLU contract、およびQKV projectionから
packed K/V materializationまでのone-dispatch subgroup routeである。
入力projection broadcast、CFG+Euler融合、長尺native SDPA、過去に採用していたcompressed-output
SwiGLUは、現binaryでは40-step全体を改善しなかったため既定化していない。

演算順序差はbitwise/hashでrejectしない。hard failureはNaN/Inf、WGPU error、形状・schedule・work
manifest不一致、および実用的な複合gateを外れる出力である。従来の狭いgateはtargetとして維持し、
許容範囲内だがtarget外の経路を`ApprovedWithWarning`としてautotunerが比較できるようにした。

## Pinsと測定条件

- branch: `codex/v4-post-seal-priority-1-4`
- logging prerequisite: `c8ac489`
- screen binary SHA-256: `81b3c32b0b5cf9e53f07292469c9afd4fa20861d4a850f7224d12c36cf0d161c`
- GPU: NVIDIA GeForce RTX 5070 Ti Laptop GPU
- driver: `595.71.05`
- WGPU: Vulkan adapter index 0
- CUDA/NVML index: 0
- PCI bus ID: `00000000:01:00.0`
- physical VRAM: 12,227 MiB
- campaign前available VRAM: 11,774 MiB
- model revision: `e4aaac4df355ff560dcd35e0dae272c3a759317b`
- model SHA-256: `5863c986345d9f6d20b7d8748fee1af02079c5161cf0c9e52557da0a0c378593`
- decoder-only codec SHA-256: `1b1ceb3f620525cf4252af508c0fde80e3779582d47fc7fc879410d2e4abe231`
- fixture SHA-256: `9a1e00e667f960983b62ebc9188c6b430acf0c00d0721ef9ffdf8fc8b9fd4b3f`
- strict FP32、TF32 off、autocast off、40 Euler evaluations
- 前半B3 x 20、後半B1 x 20、12 layers、480 block calls
- `ExclusivePages`、`ProductionPrepared`、`ConditionAndAdaLn`

各fresh sessionは専用CubeCL directoryを使い、旧campaignのautotune値をpoolしていない。steady値は
各processの2 warmup後の3 measured request中央値である。

## 採用したB1 SwiGLU route

B1 S333/S489では、handwritten T64 expand+SwiGLUより、現Burn/CubeK default graphの方が40-step
全体で速かった。このcrossoverはshape単調ではなく、S255ではhandwrittenが勝ち、S112/S685は
ノイズ範囲だった。従ってrange heuristicではなくexact cellとしてNVIDIA既定tableへ入れた。

この変更後の489-frame design controlはRF約5.08 sだった。その後のfresh native-SDPA比較ではGPUが
約4.84 sと約5.00 sの二つのclock帯を行き来したため、別campaignの単一中央値だけを比較しない。

## 改善しなかった構造候補

### Broadcast input projection

Independent CFGで物理B1 latentを先に`[32,1280]`へprojectionし、B2/B3へ直接broadcastする
一dispatch kernelを追加した。40-stepの20回すべてでcandidate hitを確認し、音声hashもcontrolと
一致したが、RF差は約2.7 ms（0.05%）に留まった。中間B2/B3入力のallocation削減候補として
exact-device tunerには残すが、RTX既定にはしない。

### Fused CFG + Euler

velocity combineとEuler updateを一dispatchにした。40-step全体では改善がノイズ範囲で、既定経路は
従来graphのままとした。

### Projection/weight route

B1 QKV、attention output、MLP contractをdefault graphへ戻すscreen、およびB1/B3 packed weight layoutの
代替を測った。全条件で音声は同一で、RFは約6--55 ms悪化した。現handwritten projectionと
B1-flat/B3-rank3のweight routeを維持する。

### CubeK compressed-output SwiGLU

過去binaryでは有効だったinterleaved compressed-output routeを現binaryへ復元すると、RF中央値は
約5.08 sから約5.48 sへ悪化した。一方でRF residentは約290 MiB減った。substage計測ではB3/B1
expandが合計約420 ms遅く、現Burn/CubeK matmul改善後には旧winnerがstaleになっていることを確認した。
VRAM優先profileの候補にはできるが、latency優先productionには戻さない。

## 長尺native SDPA

native WGSL kernelはshader自体が動的B/SKVを扱っていたが、host validationがB1/B2かつ`SKV=SQ+3`
へ不要に固定されていた。これをB1--B3、`SKV>=SQ`へ一般化し、S489 B1/B3をexact-profile候補へ
追加した。既定tableは変更していない。

B3 S489 candidateとBurn controlをそれぞれ5 fresh processで測った。

| route | fresh-session RF medians (ms) | median of sessions |
|---|---|---:|
| native WGSL | 4835.22, 5021.09, 4989.45, 5023.60, 4842.83 | 4989.45 |
| Burn control | 4988.99, 4839.47, 4845.03, 4845.86, 4846.15 | 4845.86 |

GPU clock帯を対応させると差はほぼ消え、session中央値ではnativeが143.60 ms（2.96%）遅い。
従ってRTX既定には採用しない。他GPUではsubgroup、shared memory、driver compiler特性が異なるため、
exact-device autotunerが選べる汎用候補として保持する。persistent RF VRAMは両者とも
4,178,447,488 B、853 allocationsで同じだった。

候補と同時期controlの一組を比較すると次の通りである。

| max abs | mean abs | RMSE | SNR | cosine |
|---:|---:|---:|---:|---:|
| 5.284e-4 | 1.678e-6 | 1.245e-5 | 82.208 dB | 0.999999996993 |

全sampleはfiniteで、WGPU errorは0、同一process内のrequestは同一hashだった。fresh process間では
CubeCL candidate選択に伴う演算順序差でhashが複数存在したため、hashをcross-process determinismの
代理にしない。

## Accuracy policy

route tunerは次の二段階で扱う。

- hard gate: latent/waveformがfiniteで、波形SNR 80 dB以上、cosine 0.999999以上、かつ緩和した
  absolute/RMSE bounds内。外れた候補だけrejectする。
- target gate: 従来のlatent 90 dB、waveform 85 dBと狭いabsolute/RMSE bounds。hardは通るがtargetを
  外れる候補は`ApprovedWithWarning`として性能比較へ残す。

これにより、referenceの加算順序再現を性能選択の必須条件にせず、明白な破損は引き続きfail closedにする。

## Compact Q/gate live storage

`c8e892b`ではdirect packed-K/V materializationの出力を次のように変更した。

```text
before: combined QKV+gate [B,S,4D] + Q [B,H,S,Dh] + K/V
after:  packed (Q,gate) [2,B,S,D]    + K/V
```

Qとgateは一つのallocationの非重複viewであり、shaderにはallocation全体を一つのstorage bindingとして
渡す。従ってdirect kernelは従来と同じ8 bindings、1 dispatchで、WebGPU最低保証を超えない。
post-SDPA kernelはcompact gate `[B,S,D]`とfallback combined `[B,S,4D]`の両方をshapeから判別する。

B3/S489でprojection combinedの不要なQ/K/V部分をSDPA前に解放でき、live bytesを正確に
22,533,120 B削減した。4-step同期stage profileでは、warm後のSDPA開始時in-useが全blockで
4,341,311,552 B、SDPA内部の追加in-use peakは0 Bだった。persistent RF residency
4,178,447,488 B、853 allocationsは変わらない。

| route | fresh-session RF medians (ms) | median of sessions |
|---|---|---:|
| compact Q/gate | 4832.88, 4844.40, 4990.03, 4843.45, 4849.38 | **4844.40** |
| prior combined control | 4988.99, 4839.47, 4845.03, 4845.86, 4846.15 | 4845.86 |

GPUの二つのclock帯を考慮すると速度差はノイズ範囲だが、回帰はない。FP32/F16の実GPU smoke、
attention module 30 tests、materialization focused tests、WGPU error monitorを通した。最終再build binary
SHA-256は`63488491fab2dd8eae300ad698f6a37b6f36b3aee67d97a8ff7e2bb086c5dc2f`である。

## Tuned projection + pitched in-place SwiGLU

Burn/CubeKのtuned projectionを維持し、その`[rows, gate | value]`出力の先頭半分へSwiGLUを
in-placeで書く候補を追加した。後段のhandwritten MLP contractは明示的なrow strideを受け取り、
pitched `[rows, hidden]` viewを直接読む。generic contractionへpitched viewを渡すinvalid stateを避けるため、
routeは`MlpContractRoute::{DefaultGraph, HandwrittenT64Contiguous, HandwrittenT64Pitched}`として型を分けた。

B3/S489 stage profileでは、通常SwiGLUの追加allocation 21,594,240 Bが0 Bになり、1 allocation/blockを
削除した。4-step出力hashはcontrolと一致し、40-stepの全requestはfinite、WGPU errorは0だった。

| route | fresh-session RF medians (ms) | median of sessions |
|---|---|---:|
| pitched in-place | 5017.74, 4844.98, 4990.92, 4845.71, 4997.22 | 4990.92 |

GPUは既存controlと同じ約4.84秒／約5.00秒の二つのclock帯を示した。対応する帯では速度差は
ノイズ範囲で、20.59 MiBのlive peak削減には回帰がない。NVIDIA既定ではexact B1/B3 S489 cellだけへ
採用し、他shape/deviceではexact-device tunerの候補に留める。route schema/ABIはこのstorage contractを
含めてv3へ更新し、古いselection cacheをreuseしない。

## Source-free all-voice residencyの修正

従来の`LongAllVoicePreparedOnly`は、source-freeであることを旧CubeK compressed-output layoutと誤って
結び付けていた。現default projectionは`SwiGluFused`だけでw1/w3 sourceを解放できるため、residency
layoutをinterleavedからfusedへ変更した。これにより、`wo`/`w2` sourceを含む304,740,864 Bを解放した
まま、遅いcompressed projectionを通らない。

| profile | RF median (ms) | persistent RF bytes | productionとの差 |
|---|---:|---:|---:|
| ProductionPrepared + pitched | clock帯により約4,845--4,997 | 4,178,447,488 | reference |
| LongAllVoicePreparedOnly + fused + pitched | 4,840.45 / 4,851.77 | 3,873,706,624 | -304,740,864 B (-290.62 MiB) |
| 旧LongAllVoice + compressed | 約5,480 | 約3,873,706,624 | 約+0.63 s |

新経路は489-frame voice-designの40-stepで同一process内hash一致、finite、WGPU error 0だった。
profileのrequest admissionは従来どおりbatch-one request、100+ frames、text/design/prepared-cloneに限定し、
未承認形状をsource-free modelへ流さない。
型付きNVIDIA defaultから環境変数なしで再実行した最終screenはRF中央値4,840.45 msだった。最終binary
SHA-256は`eec7d9efba48d48c0012973df7ff455702515d0fd4a27c1421c006d1629bc042`である。

## CubeK compressed-output候補の再探索

pairwise compressed writerをsimple-unit以外のCubeK routineへ接続できるよう一般化し、同一B1/B3 S489
40-stepで比較した。RTXではいずれも現tuned projectionに勝たなかったためproduction既定にはしない。

| candidate | RF median (ms) | disposition |
|---|---:|---|
| SimpleUnit default tile | 約5,480 | rejected |
| SimpleUnit minimum tile | 5,302.96 | improved candidate, still rejected |
| DoubleUnit compressed writer | 5,561.88 | rejected |
| GEMM Dot pairwise | 17,142.88 | rejected |
| staged GEMM Dot pairwise | 19,040.68 | rejected |

minimum-tileは旧compressed routeを約177 ms改善したため、VRAM優先のdevice tuner候補として残す。
GEMM Dot二案はparallelism/tilingが現shapeに不適であり、RTXの候補集合からは優先度を下げる。
これらの負の結果を、演算順序差によるaccuracy rejectとは混同しない。

## Exact manifestからのweight layout導出

広い`LongAllVoicePreparedOnly` profileは100--685 framesとB1/B2/B3を一つのmodelで扱うため、
row/column両方のQKV+gate cacheを保持する。一方、S489 design固定sessionではNVIDIA route tableの
B1/B3がともにhandwritten T64 projectionを選び、row layoutだけを読む。従来はこのexact coverageを
residencyへ反映できず、未使用column cache 300 MiBを保持していた。

`WeightResidencyPlan::derive_for_manifest`はstrict warmup manifestから実際のCFG batch classを展開し、
解決済み`ResolvedRouteTable`の各variantが必要とするlayoutのunionを作るようにした。GPU名や世代名から
layoutを推測せず、B/S/topologyのexact cellだけを使う。B4またはcompile-on-demandは従来のportable
fallback layoutを維持する。CLI harnessには`--rf-weight-residency exact-manifest`を追加し、schema 11で
導出receiptをraw JSONへ保存する。

S489 designの導出結果は次の5 layoutだけである。

```text
QkvGateRow
QkNormPacked
SwiGluFused
AttentionOutputPacked
MlpContractPacked
```

5 fresh process、各2 warmup + 3 measured、40 Euler / B3x20+B1x20 / 12 layers / 480 callsで測った。

| session | RF median (ms) | RF persistent (B) | all-resident after consumer (B) |
|---:|---:|---:|---:|
| 1 | 5,006.98 | 3,559,133,824 | 3,698,037,376 |
| 2 | 5,016.64 | 3,559,133,824 | 3,698,037,376 |
| 3 | 4,844.48 | 3,559,133,824 | 3,698,037,376 |
| 4 | 4,847.80 | 3,559,133,824 | 3,698,037,376 |
| 5 | 5,272.92 | 3,559,133,824 | 3,698,037,376 |

median-of-session-mediansは5,006.98 ms。既存の約4.84/5.00秒clock帯と、session 5の低clock側変動を
含み、dispatch graphを変えないresidency変更として速度回帰の証拠はない。広いsource-free all-voice
profile比でRF persistentとall-residentを正確に314,572,800 B（300 MiB）削減した。通常の
`ProductionPrepared`比ではRF persistentを619,313,664 B（590.62 MiB）削減している。

5/5で全sample finite、WGPU error 0、work manifest一致、process内audio hash一致だった。process間hashは
CubeCL candidateの演算順序差により異なり得るためreject条件にしていない。測定binary SHA-256は
`6fb0d0f4693d86cbff0739bbfcafd45462830045930880c7d58e4d15742855f2`。

## Cross-layer AdaLNのsingle-storage化

24個のLowRank AdaLNは、cross-layer batched precompute用にdown/up/biasをmodule-majorへpackする一方、
portable fallback用の6 Linear/moduleも同じ論理weightを所有していた。F32ではこの重複が
141,926,400 B（135.35 MiB）ある。高速なcross-layer pathとfallbackのどちらも捨てず、各Linearを
canonical packed allocationの非重複viewへrebindした。学習済みparameterと派生cacheのpaired stateを
増やさず、inference wrapper構築時の不可逆遷移に閉じ込めている。

最初にBurnの高水準`narrow`で実装したscreenは、backendが一部viewを物理sliceへmaterializeしたため
RF persistentが3,559,133,824 Bのままであり不採用とした。採用版はWGPU binding offsetとcontiguous
metadataだけを作る。runtimeが報告するalignment、dtype、shape、device、buffer範囲をdispatch前に検証し、
満たせないadapterではstorage dedupだけを行わず、従来のfast cache + source fallbackへ戻る。Vulkan固有APIや
GPU名分岐は使っていないため、Metal/DX12/WebGPUでも同じWGPU契約で利用できる。

候補binaryで5 fresh process、各2 warmup + 3 measuredを取得した。

| session | RF median (ms) | RF persistent (B) | all-resident after consumer (B) |
|---:|---:|---:|---:|
| 1 | 5,122.97 | 3,417,207,424 | 3,556,110,976 |
| 2 | 5,036.58 | 3,417,207,424 | 3,556,110,976 |
| 3 | 4,849.35 | 3,417,207,424 | 3,556,110,976 |
| 4 | 5,002.33 | 3,417,207,424 | 3,556,110,976 |
| 5 | 4,852.18 | 3,417,207,424 | 3,556,110,976 |

median-of-session-mediansは5,002.33 msで、直前exact-manifest controlの5,006.98 msと同等だった。
RF persistentとall-residentはどちらも正確に141,926,400 B削減し、RF allocation数は841から625へ減った。
5/5で40 Euler、B3x20+B1x20、12 layers、480 block calls、process内audio hash一致、WGPU error 0を確認した。
候補binary SHA-256は`5edf11b19fbd264c6ef138375397dacd6c146fe221d2ba4a020c2fb2c4d9f5a2`。

alignment fallbackとstorage-sharing testを加えたcommit `b862f78`の最終binaryでもfresh processを再確認した。
RF measured中央値は4,983.20 ms、RF persistentは3,417,207,424 B、all-resident after consumerは
3,556,110,976 Bだった。全requestのaudio hashはprocess内で一致し、40/40 AdaLN schedule hit、finite、
WGPU error 0だった。最終binary SHA-256は
`cf3713d53fd394476a5be962caf41b98f5fda751c833bb3bbc68212ba708f80d`。

## Exact-subgroup SDPA候補

S489の40-step profileではSDPAがRF block時間の最大部分を占める。既存native WGSLは
`Q8_KV32`で1 attention rowを32 laneへ割り当てていたが、online softmaxのmax/exp/sumだけは
lane 0が32 scoreを逐次処理していた。GPU名や世代名ではなく、runtime capabilityが次を全て
満たす場合だけ選べる`SdpaRoute::SubgroupWgsl`を追加した。

- strict F32 storage
- CubeCL plane operation support
- subgroup minimum = maximum = 32
- `TILE_KV = 32`
- 通常native WGSLのshape、layout、binding、shared-memory契約も全て成立

subgroup routeは`subgroupMax`/`subgroupAdd`でsoftmax reductionを並列化する。最終版では各laneの
private weightを`subgroupShuffle`で配布し、score/weight用workgroup allocationとshared-memoryの
write/readも除去した。可変subgroup幅のAMD/Intelや、subgroupを公開しないWebGPU adapterでは
Burn fallbackへ閉じる。Metal/DX12/Vulkanで共有可能なWGSL設計だが、このcampaignで実測したのは
NVIDIA/Vulkanだけであり、他backendの承認を意味しない。

最初の並列softmax版を5 fresh processで測った。session 1は2 warmup + 1 measuredのcompile/accuracy
screen、session 2--5は各2 warmup + 3 measuredである。

| session | RF median (ms) |
|---:|---:|
| 1 | 5,011.14 |
| 2 | 4,840.03 |
| 3 | 4,845.08 |
| 4 | 4,849.26 |
| 5 | 5,031.61 |

median-of-session-mediansは4,849.26 ms。private-weight shuffle版のfresh screenは4,852.38 msだった。
既存controlの高速clock帯4,844--4,848 msと実質同等で、RTX defaultを置き換える速度改善は確認できない。
従ってbuilt-in NVIDIA profileは変更せず、exact-device tunerがBurn、逐次native WGSL、subgroup WGSLの
3候補を比較するための候補としてのみ残す。

private-weight shuffle版のcontrol waveform比はmax abs `1.5259e-4`、mean abs `1.3943e-6`、
RMSE `7.1614e-6`、SNR `87.0109 dB`、cosine `0.9999999990`。演算順序差を許容する現在の複合gateを
通過した。全request finite、process内hash一致、40 Euler、B3x20+B1x20、12 layers、480 block callsを
維持した。RF persistentは3,417,207,424 B、allocation 625でcontrolから変化していない。

## One-dispatch QKV projection + packed K/V

従来のattention front endは、handwritten T64 projectionが巨大な`[B,S,4D]`を一度書き、次の
materialization kernelがQ/K RMSNorm、RoPE、packed K/V、compact gateを生成していた。新しい
`ProjectionDirectPackedKv`は同じrow-major prepared weightを読み、projection tileからQ/K/V/gateの
consumer layoutへ直接書くため、この中間bufferと1 dispatchを除去する。9 storage bindingsとstrict
F32を要求し、shape、stride、device、binding数、shared memoryをdispatch前に検証する。契約を満たさない
adapterは既存の二段経路へfail-closedする。

portable workgroup版はQ/K norm reductionに32 KiB shared memoryと4 barriersを用いる。一方、exact
32-lane subgroup capability（CubeCL plane ops、min=max=32）がある場合だけ使える第二候補は、各
16-lane halfを1 headとして`subgroupShuffleXor`でreduceする。これによりshared memoryは24 KiBとなり、
norm barriersを除去した。GPU名による分岐ではなくruntime capabilityでguardしているためWGSL sourceは
Vulkan/Metal/DX12で共有できるが、この既定採用の実測根拠はRTX 5070 Ti/VulkanのB1/B3 S489だけである。

同一最終binaryで候補とcontrolを各5 fresh process、各2 warmup + 3 measuredで測った。各sessionは
独立CubeCL cacheを使い、表は各processのRF中央値である。

| route | fresh-session RF medians (ms) | median of sessions |
|---|---|---:|
| projection-direct subgroup | 4844.63, 5018.15, 4855.72, 4848.51, 5001.77 | **4855.72** |
| current control | 5027.53, 4853.20, 5046.92, 5034.43, 5046.33 | 5034.43 |

改善は178.71 ms、**3.55%**。subgroupなしのone-dispatch screenは5000.08 msであり、dispatch統合だけで
なくbarrier/shared-memory削減が主な追加効果だった。RF+duration+codec residentは両経路とも
3,556,334,656 Bで、persistent VRAMの増加はない。候補は全sessionで40 Euler、B3x20+B1x20、12 layers、
480 block calls、finite、WGPU error 0を維持した。

同一binary control waveformとの差はmax abs `4.6320e-5`、mean abs `9.0996e-7`、RMSE
`2.2947e-6`、SNR `96.8964 dB`、cosine `0.999999999898`でhard/target gateをともに通った。
fresh process間ではCubeCL内の別matmul選択によりhashが変わる場合があるため、演算順序差をhashだけで
rejectしていない。RTX built-in tableはB1/B3 S489 exact cellだけsubgroup routeを選び、他shapeと
Apple/AMD/Intelはexact-device tunerの候補またはportable fallbackのままとした。

最終built-in profileをroute overrideなしで再build・fresh実行したRF中央値は4852.66 msだった。
binary SHA-256は`e900b9a750903ab73fb08234d92e226c336c10e13b4f1b2c9dc722b346e6b7cd`。
このprocessの出力は別のCubeCL内部候補を選んだため、同じcontrol比でmax abs `5.7228e-4`、RMSE
`1.4770e-5`、SNR `80.7234 dB`、cosine `0.999999995767`となり、hard gate通過・target外の
`ApprovedWithWarning`である。これは今回明示的に許容した演算順序差の範囲であり、性能値から除外して
いない。

## Artifacts

fresh root:
`/home/sanzentyo/benchmark-artifacts/irodori-v4-route-screens-20260824`

- candidate: `b3-f489-sdpa-native-40step-screen-2/`、`b3-f489-sdpa-native-fresh-s2/` -- `s5/`
- control: `b3-f489-sdpa-control-fresh-s1/` -- `s5/`
- unsealed profile: `profiles/b3-f489-sdpa-native.json`
- compact Q/gate: `compact-q-gate-f489-fresh-s1/` -- `s5/`
- compact stage profile: `compact-q-gate-f489-stage-profile/`
- F16 compile/accuracy smoke: `compact-q-gate-f16-f489-smoke/`
- pitched stage profile: `pitched-swiglu-f489-stage-profile/`
- pitched fresh sessions: `pitched-swiglu-f489-40step-screen/`、
  `pitched-swiglu-f489-fresh-s2-attempt3/` -- `s5-attempt3/`
- source-free fused all-voice: `longall-fused-pitched-f489-fresh-s1/`
- typed NVIDIA default確認: `longall-fused-pitched-auto-f489-fresh-s1/`
- compressed routine screens: `simple-unit-min-compressed-b13-f489-screen/`、
  `double-unit-compressed-b13-f489-screen/`、`gemm-dot-compressed-b13-f489-screen/`、
  `gemm-dot-staged-compressed-b13-f489-screen-attempt2/`
- exact manifest residency: `exact-route-derived-f489-fresh-s1/` -- `s5/`
- rejected high-level AdaLN slice: `adaln-single-storage-f489-fresh-s1/`
- AdaLN single-storage candidate: `adaln-single-storage-raw-f489-fresh-s1/` -- `s5/`
- final hardened AdaLN confirmation: `adaln-single-storage-final-f489-fresh-s1/`
- subgroup softmax candidate: `subgroup-sdpa-f489-screen/`、
  `subgroup-sdpa-f489-fresh-s2/` -- `s5/`
- subgroup private-weight shuffle: `subgroup-shuffle-sdpa-f489-fresh-s1/`
- projection-direct workgroup: `projection-direct-packed-kv-f489-fresh-s1-attempt2/`
- projection-direct subgroup: `projection-direct-packed-kv-subgroup-f489-fresh-s1/` -- `s5/`
- same-binary control: `control-route5-f489-fresh-s1/` -- `s5/`
- adopted default confirmation: `adopted-projection-direct-subgroup-f489-fresh-s1/`

各採用session directoryに`result.json`、raw f32 audio、専用CubeCL database、`SHA256SUMS`を保持した。
比較WAVと`control-comparison.json`は最初のcandidate directoryに置いた。fixture誤指定でwork manifestが
fail-closedした最初の試行は性能集計から除外した。

## 次の構造候補

1. Burn/CubeK matmulへ汎用compressed-output epilogueを接続し、B3 SwiGLUをfull expansionなしで処理しつつ、
   現default graphのmatmul性能を維持する。
2. SDPAのstage内allocator peakとbinding lifetimeをreceipt化し、attention単位のworkspaceを再利用する。
3. 40-step exact-device tunerをB1/B3各phaseとcontext length込みで実行し、世代heuristicではなくsealed
   profileからweight residencyを導出する。
