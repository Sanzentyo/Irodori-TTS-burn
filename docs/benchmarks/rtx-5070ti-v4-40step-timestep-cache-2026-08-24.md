# RTX 5070 Ti Laptop 12 GiB: 40-step timestep cache (2026-08-24)

## 結論

strict FP32の40-step Euler product pathについて、各stepで繰り返していたtimestep condition生成を
session構築時へ移した。演算の意味、CUDA互換schedule bits、CFG window、whole-model forward 40回、
12 layer、480 block callは変えていない。

- `ConditionOnly` はRF device-complete中央値を18.01 ms（0.351%）短縮し、persistent RF VRAMの
  追加は2.05 MiBだった。
- `ConditionAndAdaLn` はRF中央値を30.67 ms（0.598%）短縮し、追加は51.27 MiBだった。
- 直前に採用したB3 cross-layer AdaLN cacheも含めると、同じ489-frame design条件のRF中央値は
  5.16736 sから5.09386 sへ73.51 ms（1.422%）短縮した。
- 新旧WGPU波形差はSNR 90.11 dB、RMSE `5.01e-6`、cosine
  `0.999999999512`で、NaN/Inf、WGPU validation error、形状破損はなかった。

演算順序だけに由来する微小差はcorrectness failureとしない。以後はbitwise/hash一致を必須にせず、
非有限値、局所operator破損、新旧production経路の複合数値gate、40-step実音声を判定に使う。
PyTorch CUDAとの自由走行trajectory差は従来から存在するため、candidate自身の回帰と分離する。

## 実装

`TimestepConditionCachePolicy`を次のADTとして公開した。

- `Disabled`: request内で従来どおり全condition/AdaLNを計算する。
- `ConditionOnly`: exact Euler scheduleのcondition embeddingだけを準備する。一般builderの既定であり、
  startupとVRAMを抑える。
- `ConditionAndAdaLn`: conditionに加え、全step・全B1/B2/B3 topologyのcross-layer AdaLNを準備する。
  long-lived `OnlineSession` とbenchmarkの既定であり、steady latencyを優先する。

cache keyはmodel generation、device、dtype/layout、step count、CUDA互換schedule bits、guidance window、
step index、physical batch topologyを含む。Euler/Independent、1--40 step、B1--B3だけを受理し、
それ以外は通常経路へfail closedする。scale、temporal rescale、speaker-KV scale、context-KV cache policyは
timestep embeddingを変えないためcache keyには含めない。

40行を一度にB40 matmulへ入れる案はsteadyがさらに約12 ms短かったが、別のcompile/autotune shapeを作り、
profile preparationが約0.5 s長かった。採用版はproductionですでに使うB3/B1 shapeに分割して生成する。
また、AdaLNをstepごとの小tensorとして保持すると`ExclusivePages` allocatorでページ数が膨らむため、
通常のB1/B2/B3 projection後にschedule単位の1 allocationへpackした。RF resident allocation数は
Disabled 847、ConditionOnly 850、full 853である。

## Pinsと環境

- branch: `codex/v4-post-seal-priority-1-4`
- implementation commit: `cd0ed3b`
- prerequisite B3 AdaLN commit: `62a7d93`
- measured binary SHA-256: `714990608e6eb21ba1c4cf33c1b4fa4201ce866c9a17bfa555b9b352110310e9`
- GPU: NVIDIA GeForce RTX 5070 Ti Laptop GPU
- driver: `595.71.05`
- WGPU adapter: index 0、Vulkan、vendor `0x10de`、device `0x2f18`
- CUDA/NVML index: 0
- PCI bus ID: `00000000:01:00.0`
- physical VRAM: 12,227 MiB
- campaign前available VRAM: 11,774 MiB
- precision: strict FP32、TF32 off、autocast off
- allocator: `ExclusivePages`
- RF weight profile: `ProductionPrepared`
- model revision: `e4aaac4df355ff560dcd35e0dae272c3a759317b`
- model SHA-256: `5863c986345d9f6d20b7d8748fee1af02079c5161cf0c9e52557da0a0c378593`
- codec revision: `47376ee24834d7a05a48ebabfe3cde29b3c5e214`
- decoder-only codec SHA-256: `1b1ceb3f620525cf4252af508c0fde80e3779582d47fc7fc879410d2e4abe231`
- fixture SHA-256: `9a1e00e667f960983b62ebc9188c6b430acf0c00d0721ef9ffdf8fc8b9fd4b3f`

各条件は独立process、独立CubeCL environment、独立result/logとして取得した。旧`/tmp` artifactや
別campaignのlatencyをpoolしていない。性能screenは489 frames、voice design、40 Euler evaluations、
前半B3×20、後半B1×20、effective rows 80、2 warmup + 5 measuredである。

## 性能とVRAM

| policy | profile preparation | first RF | steady RF median | consumer median | RF resident in-use | allocs |
|---|---:|---:|---:|---:|---:|---:|
| Disabled | 0.1260 s | 14.1603 s | 5.12452 s | 5.53616 s | 4,124,687,488 B | 847 |
| ConditionOnly | 1.0332 s | 12.9201 s | 5.10652 s | 5.51548 s | 4,126,837,888 B | 850 |
| ConditionAndAdaLn | 1.0855 s | 13.0148 s | **5.09386 s** | **5.50568 s** | 4,178,447,488 B | 853 |

`ConditionOnly`はcache作成時間が約0.91 s増えるため、単発CLIでは必ずしも回収できない。full policyは
約51 MiBを使う代わりに、2回目以降のrequestを約31 ms短縮する。従ってcrate既定をmemory-boundedな
`ConditionOnly`、長寿命serviceを`ConditionAndAdaLn`とした。fresh first requestには依然として
process-local shader/pipeline compilationが含まれ、persistent CubeCL autotune cacheだけでは消えない。

## Accuracy disposition

同じformal fixtureで、採用cache出力を旧WGPUおよびPyTorch oracleと比較した。

| comparison | max abs | mean abs | RMSE | SNR | cosine |
|---|---:|---:|---:|---:|---:|
| new vs old WGPU | 1.6947e-4 | 1.3659e-6 | 5.0128e-6 | **90.109 dB** | 0.999999999512 |
| new vs Python | 2.5239e-3 | 7.2505e-6 | 5.8689e-5 | 68.740 dB | 0.999999933168 |

new-vs-old差はcondition/AdaLNをまとめて評価・packした際のFP32演算順序差であり、今回の許容方針では
PASSである。Pythonとの差はcandidate導入前から存在するEuler trajectory separationを含むため、
このcacheのreject理由にはしない。局所same-input比較で非連続な破損がないこと、全sampleがfiniteであること、
旧productionとの差が実用gate内であることを優先する。

## Artifacts

fresh root:
`/home/sanzentyo/benchmark-artifacts/irodori-v4-route-screens-20260824`

- `timestep-disabled-f489-design-production/`
- `timestep-chunked-cond-f489-design-production/`
- `timestep-chunked-full-f489-design-production/`
- `timestep-chunked-full-f489-design-formal-oracle/`

各directoryに`SHA256SUMS`を追加した。raw `result.json`、stdout/stderr、formal WAV 3本、accuracy textを
保持している。CubeCL environmentは各performance directory内に分離した。

## 次の候補

1. exact 489-frame B1/B3について、handwrittenとBurn/CubeK routeを同一binary・同一cache policyで
   component別に再測定する。特にB1 MLP expandは過去の4-step screenでdefault graphが勝った形跡がある。
2. B3 QKV→packed K/V→SDPAのtemporary live rangeをGPU timestampとstage peakで再確認し、
   materializationをstage内へ吸収する。
3. projection + SwiGLU compressed-output epilogueのB1/B3 candidateを、40-step全trajectoryで比較する。
4. route autotunerのexact device profileに今回の40-step steady/accuracy evidenceを渡し、device世代ごとの
   既定heuristicではなく測定済みroute tableを選択する。

