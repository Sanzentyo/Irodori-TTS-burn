# RTX 5070 Ti Laptop 12 GiB: v4 priority 1--4 follow-up (2026-08-22)

## 技術要約

優先度1--4のfollow-up実装とfresh計測を完了した。結論は次の通りである。

1. strict FP32 40-stepのaccuracy差は、最初のRF forwardでは117.7--119.6 dBと小さく、
   40回のEuler更新で累積する。長尺489/685 framesではRF finalの時点で79.5--88.3 dBまで
   下がり、codecがさらに9.5--14.4 dB程度差を増幅した。明白な単一operator破損の証拠は
   ないが、後半forwardは既に異なるlatentを入力するため、原因kernelの断定には至っていない。
2. 正式なstrict FP32 40-step比較では、WGPUは18条件すべてでPyTorchより遅く、
   device-complete差は+2.97%から+31.06%だった。4-stepの結果をproduction性能の代表値に
   してはならない。hard accuracy gateは14/18、85 dB targetは9/18である。
3. `WgslWeightProfile::ProductionPrepared`を追加し、production graphから到達不能な
   wq/wk/wv/gate/w1/w3 source storageを解放した。全長・全production layoutを維持したまま、
   persistent in-useとreservedを731.25 MiB削減し、NVML peakも724--732 MiB低下した。
   3形状すべてでA/B音声hashはbitwise一致し、steady latencyの変化は-0.08%から+0.29%だった。
4. external launchからWAV closeまでのcold E2Eでは、WGPU fresh cacheは46.58--53.73秒、
   restored cacheは7.21--8.52秒だった。Pythonは7.75--10.14秒である。persistent cacheは
   有効だが、process-local pipelineを保存するものではない。cross-platformなservice設計は
   long-lived `Runtime<Ready>`とreadiness前warmupを前提にする。
5. duration predictorは6長さすべてでPythonと同じ45/112/255/333/489/685 framesを返した。
   full predictorのdevice-complete中央値はWGPU 14.55--27.94 ms、Python 65.97--66.90 msで、
   全長WGPUが短かった。これはdurationだけの結果であり、489-frame音声accuracyをPASSにはしない。

次の最優先は、同一のlatent・condition・timestepを両runtimeへ注入するper-block differential
harnessである。現状のiterative traceだけでは、誤差を「どこで増えたか」までは示せても、
後半stepの特定kernelを原因とは断定できない。

## Pinsと実測環境

- branch: `codex/v4-post-seal-priority-1-4`
- follow-up measured source range: `4a18867`--`b2d66f2`
- GPU: NVIDIA GeForce RTX 5070 Ti Laptop GPU
- driver: `595.71.05`
- WGPU adapter: index 0、Vulkan、vendor `0x10de`、device `0x2f18`
- CUDA/NVML index: 0
- PCI bus ID: `00000000:01:00.0`
- physical VRAM: 12,227 MiB
- campaign前available VRAM: 11,774 MiB
- Rust: `1.95.0 (59807616e 2026-04-14)`
- Cargo: `1.95.0 (f2d3ce0 2026-03-21)`
- Burn: `=0.22.0-pre.2`
- CubeCL: `=0.11.0-pre.2`
- production backend dispatch: WGPUのみ
- precision: strict FP32、TF32 off、autocast off
- model revision: `e4aaac4df355ff560dcd35e0dae272c3a759317b`
- model SHA-256: `5863c986345d9f6d20b7d8748fee1af02079c5161cf0c9e52557da0a0c378593`
- codec revision: `47376ee24834d7a05a48ebabfe3cde29b3c5e214`
- converter input `weights.pth` SHA-256:
  `db120339c5ee7eca1912cdf29bc612b947a0808e69c3cebfb4936b45a762c1d5`
- `scripts/convert_dacvae_weights.py` full output SHA-256:
  `4af95181ddf010091b3aca92a17f9580062494ea425cee47063a9a917395f6f1`
- decoder-only output SHA-256:
  `1b1ceb3f620525cf4252af508c0fde80e3779582d47fc7fc879410d2e4abe231`

旧`/tmp` artifactや旧計測値を新campaignへpoolしていない。入力fixtureとprepared referenceだけは、
SHA検証したsealed accuracy campaignから明示的に継承した。各性能campaignのprocess、CubeCL root、
NVML log、result JSONは独立している。

## Accuracy差はRF反復で累積し、codecで増幅する

diagnostic専用APIは、production samplerの各whole-model forward出力をGPU上に保持し、通常の
consumer-complete後に保存する。保持tensorがallocator lifetimeを変えるため、このrequestのlatencyは
常にinvalidと記録する。Pythonも同じ`forward_with_encoded_conditions`境界を40回保存した。

| case | forward 0 | step 19 | step 20 | worst step / SNR | RF final | waveform | codecによる変化 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 112 text | 117.67 | 92.15 | 90.73 | 24 / 86.29 | 96.31 | 88.16 | -8.15 dB |
| 333 clone | 119.55 | 90.80 | 87.08 | 27 / 82.58 | 92.31 | 83.10 | -9.20 dB |
| 489 clone | 118.19 | 85.23 | 83.04 | 25 / 53.73 | 79.53 | 70.07 | -9.45 dB |
| 489 design | 117.92 | 84.20 | 83.77 | 35 / 72.90 | 82.97 | 68.53 | -14.45 dB |
| 685 clone | 118.55 | 92.22 | 87.22 | 27 / 73.28 | 88.32 | 78.58 | -9.75 dB |

単位はすべてPyTorchに対するSNR dBである。steps 0--19はCFGをbatch化し、steps 20--39は
1 rowで実行する。step 19から20への低下は0.42--5.00 dBだった。しかしstep 20の入力latent自体が
既にruntime間で異なるため、この低下だけをB1 kernelの誤りとは判定しない。

この結果が支持するのは、次の限定された診断である。

- step 0が全caseで117 dB超なので、shape全体を壊す明白なbinding/layout/operator errorは見えない。
- 489/685 framesはRF finalの時点でhard 80 dB近傍または未満になる。codecだけの問題ではない。
- codecは同じruntime由来latentをdecodeした最終波形比較で差をさらに8--14 dB増幅する。
- 112/333 framesではRF finalが92 dB以上なので、最終波形差はcodec amplificationの寄与が大きい。

原因を確定するには、同じPyTorch latentを各stepの両runtimeへ入力し、12 blockのinput/output、
QKV postprocess、SDPA、projection、SwiGLU、Euler combineを順に比較する必要がある。加えて同一の
CPU保存latentを両codecへ入力するdecode-only差分を取る。これで「既に異なる入力の増幅」と
「そのoperator固有の差」を分離できる。

判定policyはwaveform SNR 80 dBとcosine `0.99999999`をhard gate、85 dBとmax abs `2e-4`を
targetとする。80--85 dBは聴覚品質FAILではなくnumerical reproducibility warningである。

## 40-stepでは現行strict FP32 WGPUがPyTorchに届いていない

正式比較は6長さ × 3 voice × 5 fresh session/runtime、各session 2 warmup + 10 measuredである。
40 Euler evaluations、linear schedule、12 layers、480 block callsをmanifestで検証した。
text-onlyは前半B2、design/cloneは前半B3、後半はB1である。同じsemantic workだがsame operator
graphではない。

| case | PyTorch device | WGPU device | WGPU差 | waveform accuracy |
|---|---:|---:|---:|---|
| 45 text | 769.24 ms | 880.22 ms | +14.43% | target PASS / 105.34 dB |
| 112 text | 1,197.40 ms | 1,302.25 ms | +8.76% | target PASS / 92.01 dB |
| 333 clone | 3,506.53 ms | 3,629.83 ms | +3.52% | FAIL / 79.22 dB |
| 489 text | 3,852.57 ms | 4,150.55 ms | +7.73% | hard warning / 86.38 dB |
| 489 design | 4,705.98 ms | 5,674.63 ms | +20.58% | FAIL / 70.99 dB |
| 685 clone | 6,474.97 ms | 7,967.16 ms | +23.05% | FAIL / 79.83 dB |

全18条件のdevice-complete差は+2.97%から+31.06%、readback-completeでWGPUが勝った条件は
0/18だった。hard accuracy PASSは14/18、85 dB target PASSは9/18である。このため4-step比較の
勝敗を40-step product pathへ外挿しない。

489-frame designをNsight Systemsで補助profileしたrunでは、1 warmup後の同期境界がRF 5.439秒、
codec device-complete 0.438秒、consumer-complete 5.880秒だった。traceには2 request合計で
80 `dit_forward_wgsl`、40 `forward_batched_cfg`、40 `forward_uncfg`が記録され、work manifestと
一致する。startupを含むVulkan APIには176 `vkCreateComputePipelines`、合計8.790秒が記録された。

ただしNsight 2025.1.3はtargetがresult JSONと`.nsys-rep`を書いた後にstatus 1または139を返した。
attempt 1--5はすべてFAILUREとして保存し、Nsight値をformal latencyへpoolしていない。production
binaryはNsight外のfresh campaignで安定している。次のprofileは、in-process CubeCL GPU timestampを
RF stageへ追加するか、profiler/driver組合せを更新してteardownを解消してから行う。

## ProductionPreparedは731.25 MiBを速度を保って物理解放する

`ProductionPrepared`は任意のsupported frame countを受け付け、測定済みrow/column QKV layoutと
fused FFN cacheを維持する。一方、production routeから到達しない各layerの元wq/wk/wv/gateと
w1/w3をmodel構築時に解放し、直後にbackend `memory_cleanup()`を行う。wo/w2はB2/B3を含む
一部routeでsourceを使うため保持する。

| case | profile | persistent in-use | persistent reserved | NVML peak median | steady consumer median |
|---|---|---:|---:|---:|---:|
| 112 text | portable | 4,797.47 MiB | 4,799.51 MiB | 5,598 MiB | 1.30938 s |
| 112 text | prepared | **4,066.22 MiB** | **4,068.34 MiB** | **4,874 MiB** | 1.31319 s |
| 489 design | portable | 4,797.79 MiB | 4,799.82 MiB | 7,689 MiB | 5.87255 s |
| 489 design | prepared | **4,066.54 MiB** | **4,068.65 MiB** | **6,958 MiB** | 5.88129 s |
| 685 clone | portable | 4,798.12 MiB | 4,800.15 MiB | 8,820 MiB | 7.91824 s |
| 685 clone | prepared | **4,066.87 MiB** | **4,068.98 MiB** | **8,088 MiB** | 7.91157 s |

各値は3 fresh session、2 warmup + 5 measuredのsession medianのmedianである。全A/Bで音声hashは
bitwise一致した。persistent in-use/reservedは両方とも731.25 MiB減少した。steady差は112 text
+0.29%、489 design +0.15%、685 clone -0.08%であり、少なくともこのsampleでは速度退行の証拠は
ない。12GB all-residentは最長条件でもOOMなしに成立した。

残るpersistent差を減らすには、serviceの許可shape/topologyを`ProfileLocked` manifestへ固定し、
wo/w2 source fallbackが0件であることを証明してからsourceを解放する必要がある。request peakの
追加削減は別問題で、長尺workspaceのlifetime/reuseをprofileしてarena化する必要がある。

## Cold E2Eはcache restoreで実用域に入るが、long-lived sessionが本命である

cold E2Eは外部process launch、WGPU/CUDA初期化、必要model load、tokenize/reference preparation、
duration prediction、40-step RF、codec decode、CPU readback、WAV closeまでを含む。各voiceで
session 1はfresh campaign cache、session 2はpersistent cache restoreである。process-local
pipelineは両sessionで再構築される。

| voice | output | Python fresh | Python restored | WGPU fresh | WGPU restored | WGPU NVML peak |
|---|---:|---:|---:|---:|---:|---:|
| text-only | 6.60 s | 10.14 s | 7.97 s | 46.58 s | **7.21 s** | 5,596 MiB |
| voice design | 7.00 s | 7.75 s | **7.85 s** | 53.73 s | 8.08 s | 5,601 MiB |
| raw clone | 7.68 s | **8.02 s** | **8.03 s** | 51.45 s | 8.52 s | 5,584 MiB |

同一voiceではsession 1/2のruntime内WAV hashが一致した。runtime間hashは一致しないが、異なる
reduction/operator graphに由来する数値差であり、非決定性とは数えない。raw cloneだけreference
encoderが必要なのでfull converted codecをloadし、text/designはdecoder-only weightを使った。

CubeCL named environmentはautotune decisionなどをprocess間で再利用するが、WGSL
`ComputePipeline` objectを永続化しない。crateのdefault保存先は次である。

- Linux: `${XDG_CACHE_HOME:-$HOME/.cache}/Irodori-TTS-burn/cubecl`
- macOS: `$HOME/Library/Caches/Irodori-TTS-burn/cubecl`
- Windows: `%LOCALAPPDATA%\Irodori-TTS-burn\cubecl`
- override: `IRODORI_TTS_BURN_CACHE_DIR`、CLI `--cubecl-cache-dir`、または
  `RuntimeCachePolicy::Root`

adapter namespace、Burn/CubeCL version、compiler、precision、kernel profileはcache identityへ入る。
F16/F32や異なるadapterの結果をpoolしない。cross-platformなready pathは
`RuntimeBuilder<RuntimeCold>`でcache/deviceを初期化し、`WarmupSelection::{Interactive,
FullService, Custom}`を解決して`OnlineSession<SessionReady>`へ遷移する。`StrictWarmup`はmanifest外を
request前に拒否し、`CompileOnDemand`は未warm classを明示的にprocess-warmへ昇格する。

## Duration predictionは全長で一致し、full pathはWGPUが短い

6 caseそれぞれ3 fresh process/runtime、head/full各5 warmup + 10 measuredで測定した。
device-completeはpre-start syncからdevice completion、readback-completeはowned contiguous F32
1要素のCPU取得までで、両runtimeの境界を揃えた。

| output | tokens | frames | Python full device | WGPU full device | WGPU full readback |
|---:|---:|---:|---:|---:|---:|
| 1.80 s | 3 | 45 | 66.90 ms | **14.55 ms** | 14.90 ms |
| 4.48 s | 12 | 112 | 66.50 ms | **18.10 ms** | 18.40 ms |
| 10.20 s | 23 | 255 | 66.52 ms | **20.01 ms** | 20.41 ms |
| 13.32 s | 28 | 333 | 66.31 ms | **21.10 ms** | 21.58 ms |
| 19.56 s | 49 | 489 | 66.37 ms | **25.41 ms** | 25.95 ms |
| 27.40 s | 61 | 685 | 65.97 ms | **27.94 ms** | 28.28 ms |

predicted floatはruntime間で一致し、round/clamp後のframes、target samples、secondsも全sessionで
一致した。head-onlyは約0.97--1.42 ms同士の小差で、readbackを含む全点でWGPUが勝つとは限らない。
full predictorは全長でWGPUが明確に短い。489 framesのduration結果が正しいことは、489-frame
RF/codec waveform accuracyを保証しないため、accuracy failureとは独立に管理する。

## Crate ergonomicsと型の境界

- `WgslWeightProfile::{PortableFallback, ProductionPrepared, Fixed112OneLayout,
  Fixed112PackedOnly}`がweight lifetimeを閉じたenumで表し、paired `Option`を使わない。
- `InferenceBuilder<Ready>::build_wgsl_with_profile`がprofile preparationと物理解放を所有し、
  callerへ隠れた`memory_cleanup`手順を要求しない。
- `sample_with_diagnostic_trace`は通常の`sample`と別methodで、保持tensorとinvalid latencyを
  API/documentation上で明示する。
- `RuntimeBuilder<RuntimeCold> -> RuntimeBuilder<RuntimeConfigured> -> RuntimeBuilder<RuntimeLoaded> ->
  Runtime<RuntimeReady>`と`OnlineSession<Unwarmed> -> OnlineSession<SessionReady>`が、traffic前の
  cache/device/model/warmup順序をtype-stateで表す。
- `WarmupManifest`はframe、CFG topology、duration policyを検証し、duplicate/invalid stateを
  construction時に拒否する。
- `RuntimeCachePolicy`、`RequestAdmissionPolicy`、`ResidencyPolicy`により、CLIだけでなくRust crateを
  embeddingするserver/GUIも同じpolicyを使える。

今回のinstrumentationはproductionの演算、同期、scheduleを変更していない。diagnostic traceだけは
GPU tensor lifetimeを変えるため、runnerが性能値を無効化する。

## 制約、失敗attempt、robustness

- accuracy attempt 2はPythonの誤ったmethodをhookし、forwardを0件捕捉してFAILURE。attempt 3で
  official `forward_with_encoded_conditions`へ修正した。attempt 1のboundary-only値とtrace値を
  performanceとしてpoolしていない。
- VRAM cleanup smoke attempt 1は既存audio directoryを検出して計測前FAILURE。正式A/B attempt 2を
  最初から取り直した。
- cold attempt 1は存在しない旧codec pathでpreflight FAILURE、attempt 2はraw cloneへdecoder-only
  codecを渡してFAILURE。attempt 3はfull/decoder-onlyをvoiceごとに型相当の分岐で固定して全条件を
  取り直した。
- Nsight attempt 1--5は前述のprofiler exit問題によりすべてFAILURE。生成reportの数値は診断補助で、
  正式比較や採否判定へ使っていない。
- VRAM A/Bの3 fresh sessionは退行の大きなシグナルを検出するには十分だが、微小な+0.15--0.29%を
  因果効果と断定するsample数ではない。
- waveform 80/85 dB gateは聴覚MOSではなくnumerical reproducibility基準である。聴覚的同等性を
  主張するにはblind listening testが別途必要である。

## 次の優先順位

1. **同一入力per-block differential**: step 19/20/25/27/35を中心に、両runtimeへ同じlatent、
   condition、timestepを入力し、12 blockとQKV/SDPA/MLP/Euler境界を保存する。最初に80/85 dBを
   割るoperator familyを特定する。
2. **codec standalone差分**: 同じCPU保存latentをPyTorch/WGPU codecへ入力し、各decoder blockと
   final waveformを比較する。RF入力差の増幅とcodec固有差を分ける。
3. **40-step B3/B1 profile**: design/cloneの長尺差が大きいため、in-process GPU timestampをRFへ
   追加し、projection、attention、MLP、CFG combineをdevice timeで分解する。外部Nsightのstatus
   139問題へ依存しない。
4. **次のpersistent削減**: service manifestでwo/w2 source fallback 0件を証明する
   `ProfileLocked`遷移を追加する。source削除前後で全許可shapeのhash/accuracyをgateする。
5. **長尺request peak削減**: workspace allocation traceを取り、shape-keyed reusable arena、
   buffer alias可能範囲、temporary tensor lifetimeをA/Bする。RF latentはcodecまでGPU residentを維持し、
   final audio以外のreadbackを追加しない。
6. **readiness時間短縮**: sealed CubeCL bundle、DryRun compile、少数real validationを
   `WarmupSelection`ごとに測定し、readyまでのwallとfirst admitted requestを別々に報告する。

## Fresh artifactsとSHA256SUMS

| campaign | status | source | `SHA256SUMS` SHA-256 |
|---|---|---|---|
| `irodori-v4-accuracy-localization-20260822-attempt3` | COMPLETE | `4a18867` | `f4e9a446201988eaabbd2ce7a4f4fdbe6ace8a8b219c21cbff24f4c138b487e7` |
| `irodori-v4-production-prepared-vram-20260822-attempt2` | COMPLETE | `b83c369` | `18e365607f0eb75d55b3b4b7a74f1d296456a76b138c4f53489d899591117b9e` |
| `irodori-v4-cold-e2e-20260822-attempt3` | COMPLETE | `b2d66f2` | `a9f385d73b970a8b79d0dadf44815bb89cea71b8bb404d138e46ee8a2e955a9b` |
| `irodori-v4-duration-refresh-20260822-attempt1` | COMPLETE | `b2d66f2` | `2093e333099dba0afa45eaacd233e5f80ea27e94ef16060e3e6bb3a94f68905a` |
| `irodori-v4-40step-formal-20260822-attempt2` | COMPLETE | `8a19782` | `b5fe310825d3eeaf2b19a2a17460a9c8678c8a94bbc606e11589c9332485ecd3` |
| `irodori-v4-40step-rf-profile-20260822-attempt2` | FAILURE / diagnostic report generated | `418d66b` | `822ec39dc9b32c5ac3a9eb57d89fc6ef8eab8537af62f9fd50cf28e3b24069cd` |

全pathのrootは`/home/sanzentyo/benchmark-artifacts/`である。各COMPLETE directoryにはraw session
JSON/log/NVML、binary/model/source pin、失敗なしの`SHA256SUMS`検証結果がある。duration campaignは
runner/Python/binary/model/codec hashを`pins.txt`へ保存する。

## 再開手順

```bash
git switch codex/v4-post-seal-priority-1-4
git pull --ff-only
git rev-parse HEAD

# Artifact integrity
for d in \
  /home/sanzentyo/benchmark-artifacts/irodori-v4-accuracy-localization-20260822-attempt3 \
  /home/sanzentyo/benchmark-artifacts/irodori-v4-production-prepared-vram-20260822-attempt2 \
  /home/sanzentyo/benchmark-artifacts/irodori-v4-cold-e2e-20260822-attempt3 \
  /home/sanzentyo/benchmark-artifacts/irodori-v4-duration-refresh-20260822-attempt1; do
  (cd "$d" && sha256sum -c SHA256SUMS)
done

# Accuracyの次の入力としてtrace summaryを再表示
uv run scripts/summarize_v4_accuracy_localization.py \
  --root /home/sanzentyo/benchmark-artifacts/irodori-v4-accuracy-localization-20260822-attempt3

# 現行のVRAM A/B runner
bash scripts/run_v4_production_prepared_vram.sh \
  --output-dir /home/sanzentyo/benchmark-artifacts/NEW-FRESH-VRAM-CAMPAIGN \
  --input-campaign /home/sanzentyo/benchmark-artifacts/irodori-v4-accuracy-localization-20260822-attempt3
```

次cycleは上記1の同一入力differentialから開始し、accuracy判断が終わる前に新しいkernel/tile調整を
採用しない。
