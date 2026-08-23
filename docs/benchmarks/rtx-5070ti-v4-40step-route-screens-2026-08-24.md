# RTX 5070 Ti Laptop: 40-step RF route screens (2026-08-24)

## 結論

489-frame voice-designのstrict FP32 product pathで、演算順序差を許容したうえでrouteを再評価した。
このcycleでRTX既定へ採用したのは、B1のS333/S489におけるBurn/CubeK SwiGLU graph、
compact Q/gate storage、およびB1/B3 S489のpitched in-place SwiGLU contractである。
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

各採用session directoryに`result.json`、raw f32 audio、専用CubeCL database、`SHA256SUMS`を保持した。
比較WAVと`control-comparison.json`は最初のcandidate directoryに置いた。fixture誤指定でwork manifestが
fail-closedした最初の試行は性能集計から除外した。

## 次の構造候補

1. B3 QKV projectionからpacked K/V materializationまでを同一producerへ統合し、巨大なcombined tensorの
   live rangeとlayout変換を短縮する。
2. Burn/CubeK matmulへ汎用compressed-output epilogueを接続し、B3 SwiGLUをfull expansionなしで処理しつつ、
   現default graphのmatmul性能を維持する。
3. SDPAのstage内allocator peakとbinding lifetimeをreceipt化し、attention単位のworkspaceを再利用する。
4. 40-step exact-device tunerをB1/B3各phaseとcontext length込みで実行し、世代heuristicではなくsealed
   profileからweight residencyを導出する。
