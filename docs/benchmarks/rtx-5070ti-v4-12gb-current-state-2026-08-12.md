# Irodori-TTS v4 12 GiB baseline — 2026-08-12

## 結論

このcycleではproduction演算を最適化せず、12 GiB環境の現行実装をfreshに再計測した。
WGPUではduration/RF/codecのall-residentが実測上可能であり、同居直後はallocator
in-use 4,901.8 MiB、reserved 7,288 MiB、1 requestを含む外部NVML peakは
7,466 MiBだった。OOM、retry、条件選択はない。

既存の高水準`pipeline`は8 GiB向けにRFをdropしてからcodecをloadするため、
all-residentとonline sessionを表現できない。baseline取得後に追加した
`bench_v4_residency`は低水準production APIを構成する測定専用probeであり、演算、
precision、同期条件を変更していない。online residentとprepared speakerを表す
高水準Rust APIは依然として未実装である。

fresh strict-FP32 accuracy campaignでは489 framesを含む6長さすべてが明示gateを
通過した。旧8 GiB campaignの489-frame失敗値はpoolしておらず、今回のfresh source
noise、model pin、binaryだけから判定した。したがって489/685 framesの性能値は
accuracyと同時に記録できたが、長尺RFはWGPUがPyTorchより常に速いわけではない。

追加調査ではautotune選択結果が既にprocess間cacheされている一方、WGPU pipelineはprocess内
cache、wgpu application pipeline cacheは未使用であることを確認した。portableな対策は
long-lived warmed sessionであり、disk pipeline cacheはVulkan限定の補助策になる。WGPUの
2,386 MiBの予約余白に対してallocator policyを再計測し、`ExclusivePages`を採用した。decode-only
codecと合わせ、control比でin-use 104.3 MiB、reserved 2,469.0 MiB、外部NVML peak median
1,669 MiBを削減した。ここで2,469.0 MiBはload後idle reservedで、request中allocator peakの
削減は1,851.9 MiBである。steady consumer latency差は+0.04%で、packed/fused高速化cacheは維持した。

## Campaignとpin

fresh artifact root:

`/home/sanzentyo/benchmark-artifacts/irodori-v4-12gb-baseline-20260812-attempt1`

- source branch: `codex/v4-wgsl-fusion`
- measurement-start source commit: `b275147b63542d37be20e28e89b39bf2ed9421d6`
- upstream runtime commit: `9f19d9a9048099a4b978a762d0509228fe624e3f`
- Irodori-TTS-v4-Small revision: `e4aaac4df355ff560dcd35e0dae272c3a759317b`
- `model.safetensors`: `5863c986345d9f6d20b7d8748fee1af02079c5161cf0c9e52557da0a0c378593`
- Semantic-DACVAE revision: `47376ee24834d7a05a48ebabfe3cde29b3c5e214`
- input `weights.pth`: `db120339c5ee7eca1912cdf29bc612b947a0808e69c3cebfb4936b45a762c1d5`
- converter: `scripts/convert_dacvae_weights.py`, SHA `604a9bc6adae11aa1ba2ad5197a883078a495df9b0b1e0d32dac5092ffd0a15d`
- converted codec: `4af95181ddf010091b3aca92a17f9580062494ea425cee47063a9a917395f6f1`
- fresh source-noise file: `17e9016569e9e087001bebde393d7039d84e0beaee81a3fef7438a91bcdf186b`
- source-noise tensor: `948dbddb2a33925be58369a3073137d08930272d68468596bf7a808dfb6fba7a`
- production `pipeline`: `84c9c363646b2c3b9eccb2d1a0d12f7e4cdc5d1a323fd24c9e8be966aa562050`
- accuracy `validate_v4_precision`: `0a08b81e3ea220eb84059280d655e63ed227e82f7cd87afe742adef6f6c71d9e`
- residency/phase runner: `de93ad8c33eeae88de2c01020e4c6804b011f6c4452827ed6ab1571644336c8d`

過去の`/tmp` artifactは入力にも統計poolにも使用していない。最初のduration attemptは
Python 3.12のoffline wheel不足でGPU実行前にfail-closedし、`duration/FAILURE`として
保存した。成功値は新規`duration-attempt2`だけから得た。ほかの条件にautomatic retryは
ない。

baseline後のclean-build確認でconverterのCPU reductionがthread数によりbyte-level SHAを
変えることを検出した。現行scriptはPython 3.10、Torch 2.10.0、safetensors 0.7.0、
NumPy 2.2.6をpinし、OpenMP/MKLを1 threadで再execする。通常の`uv run`から上記
`4af951…`を再現する。campaignで使用したconverter source SHAは上記`604a9b…`のままで、
測定済みartifactや統計は更新していない。

## 実測環境

| 項目 | 実測値 |
|---|---|
| GPU | NVIDIA GeForce RTX 5070 Ti Laptop GPU |
| NVML/CUDA index | 0 |
| PCI bus ID | `00000000:01:00.0` |
| driver | 595.71.05 |
| NVIDIA-SMI CUDA compatibility | 13.2 |
| nvcc toolkit | 12.9.86 |
| compute capability | 12.0 |
| total / initial used / initial free | 12,227 / 2 / 11,774 MiB |
| WGPU adapter index | 0 |
| WGPU backend / type | Vulkan / DiscreteGpu |
| WGPU vendor / device | 4318 / 12056 |
| WGPU driver | NVIDIA 595.71.05 |
| CPU | Intel Core Ultra 9 275HX, 24 logical CPUs |
| Rust / Cargo / uv | 1.95.0 / 1.95.0 / 0.11.7 |

raw identityは`environment/nvidia-smi*`と`environment/wgpu-adapter.json`にある。
WGPU adapter indexとCUDA/NVML indexは同じ0だが、同じordinalであることを仮定せず、
名前とPCI bus IDを個別にgateした。

## Cold full E2E

外部process launchからfinal CPU readback、WAV write/close、process exitまでを
`/usr/bin/time`で測った単発baselineである。各first-shape条件は独立processと独立
`XDG_CACHE_HOME`を使用した。

| 条件 | duration prediction | RF | codec | cold E2E | output | NVML peak |
|---|---:|---:|---:|---:|---:|---:|
| text-only, fixed | — | 8,669 ms | 4,991 ms | 19.08 s | 4.48 s | 6,986 MiB |
| voice design, fixed | — | 8,891 ms | 1,548 ms | 16.58 s | 4.48 s | 7,048 MiB |
| raw clone ref1, fixed | — | 9,359 ms | 759 ms | 22.19 s | 4.48 s | 7,023 MiB |
| text-only, predicted, 4-step | 833.1 ms | 5,602 ms | 5,112 ms | 16.58 s | 5.40 s | 7,044 MiB |
| voice design, predicted, 4-step | 8,221.1 ms | 4,082 ms | 3,964 ms | 21.36 s | 5.68 s | 7,046 MiB |
| raw clone ref1, predicted, 4-step | 2,311.8 ms | 7,751 ms | 4,002 ms | 21.02 s | 7.24 s | 7,018 MiB |
| text-only, fixed, cache-warm | — | 808 ms | 774 ms | 6.87 s | 4.48 s | 6,972 MiB |
| text-only, predicted, default 40-step | 736.0 ms | 4,475 ms | 957 ms | 11.54 s | 5.40 s | 7,040 MiB |

40-step値は4-step比較から分離した。40-step processはfreshだったがdriver/backendの
永続cacheを完全に隔離できた証拠がなく、4-step first-shapeより短い非直感的結果のため、
cold compiler costの比較や最適化判断には使用しない。raw clone fixedではmodel load後、
reference codec load/encode/cleanupにログtimestamp上約6.27 sを要した。load後idleを置く
public cold CLI stateはなく、cold条件のload-after-idle VRAMは未定義である。代わりに次節の
明示all-resident probeでpersistent値を取得した。

## 12 GiB all-resident

| runtime | load wall | load後persistent | request peak | 結果 |
|---|---:|---:|---:|---|
| PyTorch, 5 fresh sessions | 4.666–13.336 s（median 4.970） | allocated 3,448.5 MiB / reserved 3,558 MiB | allocated 4,151.9 MiB / reserved 4,756 MiB、NVML 4,972 MiB | 成功 |
| WGPU probe | 3.967 s | in-use 4,901.8 MiB / reserved 7,288 MiB | NVML 7,466 MiB | 成功 |

WGPUはduration predictorを含むRF modelとcodecを同時保持したまま4.48 s requestを完了した。
allocator snapshotはrequest後もin-use 4,901.8 MiB、reserved 7,328 MiBだった。OOM時retryを
禁止したがOOMは起きなかった。これは「12GBなら入る」という仮定ではなく、このadapter、
driver、sub-slices allocatorでの実測である。

### Paired steady-state remeasurement

初回probeのWGPU 1 requestとPython steady値はwarmup境界が非対称だったため、比較用のfresh
campaignを別途実行した。採用campaignは
`/home/sanzentyo/benchmark-artifacts/irodori-v4-12gb-all-resident-compare-20260812-attempt2`。
機械可読集計は
[`runtime-scenarios-12gb-2026-08-12/all-resident-comparison.json`](runtime-scenarios-12gb-2026-08-12/all-resident-comparison.json)
に保存した。campaign root `SHA256SUMS`のSHA-256は
`732673cf9fa1df71591505e6dd810c6c424f4bc8be4a309d45fed2d5174968f1`。
両runtimeとも5 fresh sessions、各2 warmup + 10 measured、同一4.48 s / 112-frame
precision fixture、text `こんにちは。`、unconditioned voice、固定initial noise、strict FP32、
4 Euler evaluationsを使用した。consumer-completeはfinal owned CPU audioまでを含み、
intermediate latentは含まない。attempt 1はPython probeがconsumer interval内に診断用latent
readbackを挿入していたため不採用とし、attempt 2へpoolしていない。

| 指標（5 sessionのmedian） | Python | WGPU | WGPU/Python |
|---|---:|---:|---:|
| load wall | 5.430 s | 4.231 s | 1.28x faster |
| first request consumer-complete | 302.6 ms | 4,724.8 ms | 15.61x slower |
| second warmup | 278.1 ms | 196.7 ms | 1.41x faster |
| steady consumer-complete | 315.8 ms | 214.5 ms | 1.47x faster |
| steady requests/s | 2.957 | 4.506 | 1.52x |
| steady audio-s/wall-s | 13.247 | 20.186 | 1.52x |
| RF device-complete | 188.6 ms | 135.6 ms | 1.39x faster |
| codec device-complete | 126.3 ms | 73.9 ms | 1.71x faster |
| persistent in-use/allocated | 3,449.4 MiB | 4,902.0 MiB | WGPU +1,452.6 MiB |
| persistent reserved | 3,558 MiB | 7,288 MiB | WGPU +3,730 MiB |
| external NVML peak | 4,756 MiB | 7,464 MiB | WGPU +2,708 MiB |

steady session median rangeはPython 309.0–318.0 ms、WGPU 206.7–219.0 ms。50 measuredを
poolせず確認用に全row medianを取るとPython 314.9 ms、WGPU 214.4 msで同じ結論だった。
全60 request/runtimeはruntime内でdeterministicだった。runtime間のaudio hashは数値実装差により
異なるが、このfixtureの別途strict accuracy gateはpassしている。

WGPU first requestの4.72 sはload後のpipeline生成を含み、Python first requestの0.303 sより
大幅に遅い。後述の調査により、これは「autotune結果が永続化されていない」だけでは説明できない。
低遅延serviceではsession warmupが必須である。一方、warm後はWGPUがconsumer-completeで1.47x
速い。VRAM trade-offは明確で、WGPUはNVML peakを約2.65 GiB多く使う。同一RF意味論だがsame
operator graphではなく、WGPU harnessはpre-tokenized fixture tensor、Python public runtimeは
request内tokenizationを通る。

### 初回requestのshader compilation / autotune cache調査

このcampaignではCubeCL autotune結果は既に`target/autotune/0.10.0-pre.3/`へ永続化されていた。
matmul、attention、reduceの3 logはpaired 5 sessionsより前に更新され、その後も共有されている。
それでも各fresh processのfirst requestは4.704–4.989 s、second warmupは0.193–0.199 sだった。
したがって確認済みの結論は「autotune選択結果はprocessをまたいでcache済みだが、4.7 s級の
first-request penaltyは残る」である。各stageのcompile traceをまだ採っていないため、残り時間を
shader compiler、pipeline生成、driver machine-code生成へ秒単位で配分することはしない。

現在のCubeCL WGPU serverは`KernelId -> ComputePipeline`をprocess内`HashMap`へ保持する一方、
`ComputePipelineDescriptor.cache`へ`None`を渡す。さらに測定runnerは各fresh session専用の
`XDG_CACHE_HOME`を作る。実際、各sessionには約2.85 MiBのNVIDIA `GLCache`と2 MiBの
`mesa_shader_cache/index`が新規生成された。この隔離はcold境界を守る測定仕様であり、production
deploymentで毎process同じ隔離を行う理由はない。以上から、残るpenaltyの主因がprocess-local
pipeline cacheとfresh driver cacheである可能性は高いが、これはtrace取得前の推定である。

| 手段 | process間で再利用 | cross-platform性 | このrepositoryでの判断 |
|---|---|---|---|
| CubeCL autotune log | 可能 | filesystemを持つnative WGPUでは実装可能。browser WebGPUには別storageが必要 | 既に有効。adapter/runtime/kernel checksumが変わる場合は別cacheにする |
| long-lived process + shape warmup | processを終了しない | WGPUが動くVulkan/Metal/DX12/WebGPUで原理上共通 | 最優先。service ready前に必要shape/topologyをwarmupする |
| vendor driver disk cache | driver依存で可能 | OS/vendor固有で、portable APIではない | productionでは通常cache directoryを共有してよいが、正しさやlatency SLOの契約には使わない |
| wgpu `PipelineCache`の保存/復元 | 可能 | wgpu 29ではVulkanのみ | CubeCLが現在`cache: None`なので未使用。Vulkan限定のoptional accelerationとして検証する |
| CubeCL compilation cache | 条件付きで可能 | 現行WGPU実装では`spirv` feature時のSPIR-V cache。今回の`wgpu_wgsl` pathには非適用 | 今回のWGSL cold penaltyのportable解にはしない |
| build時に全shaderをdevice machine code化 | device/driver固定時以外は不可 | 不可 | 採用しない。shape manifestによるruntime warmupを使う |

wgpuのapplication-managed `PipelineCache`は、同一または類似adapterでpipeline生成結果を次回processへ
渡すAPIである。ただし現versionではVulkan限定であり、desktop driver自身のcacheと効果が重なる場合も
ある。実装する場合は`wgpu::util::pipeline_cache_key`に加えてdriver、wgpu/CubeCL、precision、
bounds-check、kernel manifest、binary versionをkeyへ含め、atomic writeする。cache miss、破損、非対応
backendでは通常compileへfallbackし、cacheを正しさの前提にしない。

portableなproduction契約は`RuntimeBuilder<Cold> -> Runtime<Loaded> -> Runtime<Warmed>`とし、
`OnlineSession<Ready>`を全required shapeのwarmup完了後だけ構築するのがよい。warmup manifestには少なく
とも45/112/255/333/489/685 latent frames、B1/B2 CFG topology、codec output shape、duration head、
reference encodeを含める。Vulkan `PipelineCache`とvendor cacheはこの契約を速めるoptional layerであり、
Metal/DX12/WebGPUを排除するAPIにはしない。

### WGPU VRAM差の内訳

paired campaignのmedianを同じ境界で分解すると次になる。WGPUはrequest中もallocator in-useが
増えておらず、7.288 GiB全体をlive tensorが占めているわけではない。

| 指標 | Python | WGPU | 差 |
|---|---:|---:|---:|
| request peak live（allocated / in-use） | 4,025.1 MiB | 4,902.0 MiB | WGPU +876.9 MiB |
| request peak reserved | 4,540 MiB | 7,288 MiB | WGPU +2,748 MiB |
| reserved - live | 514.9 MiB | 2,386.0 MiB | WGPU +1,871.1 MiB |
| external NVML peak | 4,756 MiB | 7,464 MiB | WGPU +2,708 MiB |

`reserved - live`の2,386 MiBはCubeCL allocatorが再利用のため確保したpage/slice余白である。完全に
freeなpageは`memory_cleanup`で解放できるが、live allocationを含むpageの未使用sliceはcompact
できない。したがってこの2,386 MiBをそのまま安全な削減可能量とは扱わない。解放後の次requestでpage
再確保が起きればfirst/steady latencyまたはpeakが悪化し得るため、VRAMだけでなく2 warmup + 10
measured x 5 fresh sessionsを再度gateする必要がある。

一方、persistent liveはPython 3,449.4 MiBに対してWGPU 4,902.0 MiBで1,452.6 MiB多い。現行WGPU
高速経路がsource weightに加えて保持するcacheをshapeから積算すると次のとおりである。

| WGPU inference cache | 12 blocks合計 |
|---|---:|
| combined QKV+gate row-major | 300.000 MiB |
| combined QKV+gate column-major | 300.000 MiB |
| packed attention output projection | 75.000 MiB |
| fused FFN w1+w3 | 431.250 MiB |
| packed FFN w2 | 215.625 MiB |
| Q/K norm cache | 0.117 MiB |
| cross-layer AdaLN cache | 135.352 MiB |
| 合計 | 1,457.344 MiB |

この1,457.344 MiBは実測persistent差1,452.561 MiBと4.783 MiB以内で一致する。これはallocatorを
含むsize accountingであり各byteのprovenance traceではないが、「Rust/WGPUが約7 GiBなのは単なる
leak」という説明は支持しない。主な差は、steady RF 1.39x高速化に使うfused/packed layoutの重複と、
SubSlicesの予約余白である。codec loadによるin-use増分も486.245 MiBで、converted checkpoint
409.546 MiBより約76.7 MiB大きく、decoder向けcacheとlive allocator overheadを含む。

### 高速化を維持したVRAM削減候補

結論は「削減できる可能性は高いが、予約余白と高速化cacheでは手段が異なる」である。優先順位と
portable性は次のとおり。

| 候補 | 期待する対象/上限 | 高速化維持の条件 | cross-platform性 |
|---|---|---|---|
| load + warmup後の明示`memory_cleanup` | 完全にfreeなpage | 後続実測では追加削減0、throughput低下傾向のため不採用 | CubeCL backend共通API。今回確認済みなのはVulkanのみ |
| `SubSlices` / `ExclusivePages` / page policy sweep | 主に予約余白 | `ExclusivePages`を採用。request reserved peakを1,851.9 MiB削減しsteady同等 | 設定APIはbackend共通、最適値はadapter/backend別 |
| decode-only codec loader | checkpoint encoder約104.1 MiBと不要なencode-side state | 実装済み。全6長さでfull codecとbitwise同値 | backend非依存で最もportable |
| exact-duration residency | duration predictor約83.1 MiB | `Duration::Exact/Frames`専用sessionに限定。`Predict`ではresidentを維持 | backend非依存。ただしall-model-resident baselineとは別profile |
| packed-only RF model | sourceと高速layoutの重複。安全性未確認の候補はw1/w3 431.25 MiB、AdaLN 135.35 MiB、QKV+gate 300 MiB | profile内の全shapeがpacked/fused pathを通り、fallback不要とaccuracyで証明してからsourceをdrop | type-stateはportableだがweight policy/kernelはWGPU profile固有 |
| QKV layoutをprofileごとに1種だけ準備 | rowまたはcolumn 300 MiB | 固定shape/topologyでは未使用layoutを証明。多様な長さとfallbackを維持する場合は両方必要 | policy表現はportable、layout選択はbackend/kernel固有 |

decode-only codecは`Codec<DecodeOnly>`のようにencoderを構築しない型にすれば、演算経路を変えずに
cross-platformで削減できる可能性が最も高い。duration headも`ModelResidency::{Predictive,
ExactOnly}`でinvalidなrequestを型で拒否できる。

packed-only化は効果が大きいが、現在の`prepared_wo_route`と`prepared_w2_route`は短いB1/B2を含む
一部shapeでsource projectionを意図的に使う。w2/woのsourceをさらに捨てれば理論上約290.6 MiB減る
が、現時点では速度維持条件を満たさないため候補量へ算入しない。固定112-frameではcombined row pathが
選ばれるとしても、全長serviceからcolumn cacheを即削除してよい根拠にもならない。

実装するなら`PreparedModel<PortableFallback> -> PreparedModel<ProfileLocked>`の不可逆遷移とし、
`KernelProfile`に許可shape、CFG batch、backend、precision、fallback policyを持たせる。source weightを
dropした後にunsupported shapeを受け付ける状態を表現不能にする。これにより高速layoutは保持したまま
元weightだけを解放できるが、489/685 accuracyと全manifest shapeのlatencyを通すまでproduction default
にはしない。

phase batchは既にRF drop後にN=12 latent-only 0.16 MiB in-use / 16 MiB reservedまで解放できており、
明示cleanupが完全free pageへ効く証拠である。ただしall-resident latencyを維持する手段ではなく、別の
throughput/VRAM policyとして残す。

### VRAM削減の実装と採用結果

採用campaignは
`/home/sanzentyo/benchmark-artifacts/irodori-v4-12gb-vram-opt-20260812-attempt1`。
`SHA256SUMS`のSHA-256は
`d2545e94db403c2c20034b863ea9ce0a2e96e6e22fa95b403afdb05cc39b4fff`で、機械可読集計は
[`runtime-scenarios-12gb-2026-08-12/vram-optimization.json`](runtime-scenarios-12gb-2026-08-12/vram-optimization.json)
にも保存した。4条件をinterleaveし、各5 fresh sessions、2 warmup + 10 measured、同一112-frame
strict-FP32 requestで測定した。automatic retryはなく、各runtime内60 requestのaudio hashは全条件で
`faf8ea…e3d`に完全一致した。

| 条件 | steady consumer | requests/s | in-use | idle reserved | request reserved peak | NVML peak | control比 |
|---|---:|---:|---:|---:|---:|---:|---:|
| full codec + SubSlices control | 210.10 ms | 4.545 | 4,902.0 MiB | 7,288.0 MiB | 7,288.0 MiB | 7,464 MiB | 1.000x |
| decode-only + SubSlices | 207.51 ms | 4.531 | 4,797.7 MiB | 7,136.0 MiB | 7,264.0 MiB | 7,424 MiB | 1.012x |
| decode-only + cleanup | 210.98 ms | 4.422 | 4,797.6 MiB | 7,136.0 MiB | 7,264.0 MiB | 7,424 MiB | 0.996x |
| decode-only + ExclusivePages | 210.18 ms | 4.515 | 4,797.7 MiB | 4,819.0 MiB | 5,436.1 MiB | 5,795 MiB | 1.000x |

`decode-only + ExclusivePages`はcontrol比でlive 104.315 MiB、idle reserved 2,469.047 MiB、
request reserved peak 1,851.859 MiB、NVML peak median 1,669 MiBを削減した。consumer latencyは
+0.080 ms / +0.038%、RF device medianは
132.77→130.90 ms、codec device medianは74.64→75.51 msで、総合steady性能は測定揺れの範囲で
維持された。reserved/live比は1.487から1.004へ下がり、未使用予約の大部分を除けた。load wallは
control 4.128 s、採用条件4.182 sであり、cold loadの優位は主張しない。

cleanupはdecode-only単独からreservedを1 byteも減らさず、throughput medianも4.531→4.422
requests/sだったためall-resident defaultには採用しない。これはlive allocationがSubSlices pageへ
広く残り、完全free pageだけを解放するcleanupではfragmentationを解消できないという事前分析と一致する。

実装では`DacVaeDecoder<B>`と`load_decoder`を追加し、encoderとencode-side `in_proj`を構築不能な
decode-only型にした。generic `decode`は元のbackend operation、WGPU `decode_wgsl`は元のproduction
WGSL operationをそのまま呼ぶ。production `pipeline`はreference encodeが必要な間だけfull
`DacVaeCodec`を使用し、final decodeでは`DacVaeDecoder`だけをloadする。raw clone機能を失わず、
decode resident stateからencoderを排除した。WGPU production initializationは
`MemoryConfiguration::ExclusivePages`を使う。

allocator APIとdecode-only model分離はVulkan/Metal/DX12を含むnative WGPU backendで共通に実装
できる。今回の性能・VRAM実測はVulkan/NVIDIAだけなので、他backendで同じ削減率や速度を仮定しない。
decode-onlyはbackend非依存に不要weightを構築しないためportableなdefaultである。allocator policyは
backend共通に指定できるが、将来adapter/backend別profileへ分離できるようpolicy値として扱う。

### 固定112-frame profileのsource-weight解放

追加campaignは
`/home/sanzentyo/benchmark-artifacts/irodori-v4-fixed112-vram-20260812-attempt1`。
top-level `SHA256SUMS`のSHA-256は
`ec8e594295fd0996ad4f9f584a33f77f10ba9880cb92de6d415dd8ff8cb8f2b7`であり、機械可読抜粋は
[`runtime-scenarios-12gb-2026-08-12/fixed112-vram-decomposition.json`](runtime-scenarios-12gb-2026-08-12/fixed112-vram-decomposition.json)
に保存した。旧sessionはpoolせず、campaign内でcacheを新規生成してから6条件をinterleaveし、各5 fresh
process、2 warmup + 10 measured、automatic retryなしで測定した。境界は4.48秒/112 frames、
strict FP32、4 Euler評価、forward batch `[2,2,1,1]`、final owned CPU audioまでである。

| 条件 | steady consumer | persistent in-use | control比削減 | request reserved peak | NVML peak |
|---|---:|---:|---:|---:|---:|
| portable/predictive control | 215.00 ms | 4,797.44 MiB | 0 | 5,436.14 MiB | 5,953 MiB |
| exact-duration only | 222.67 ms | 4,678.31 MiB | 119.13 MiB | 5,273.04 MiB | 5,737 MiB |
| RF fixed112 / QKV one-layout | 217.49 ms | 4,497.44 MiB | 300.00 MiB | 5,138.64 MiB | 5,539 MiB |
| RF fixed112 / packed-only | 213.52 ms | 3,766.19 MiB | 1,031.25 MiB | 4,761.22 MiB | 5,697 MiB |
| codec fixed112 / packed-only | 219.12 ms | 4,689.44 MiB | 108.00 MiB | 5,328.14 MiB | 5,845 MiB |
| combined packed-only | 213.28 ms | 3,539.06 MiB | 1,258.38 MiB | 4,514.19 MiB | 5,501 MiB |

combinedはpersistentを1,258.38 MiB、request reserved peakを921.95 MiB削減し、consumer medianは
control比1.008xであった。4.515 requests/s、20.229 audio-seconds/wall-secondであり、112-frame条件では
高速化を維持した。単独差は加算的で、duration 119.13 MiB、未使用column QKV layout 300 MiB、
さらにQKV sourceとw1/w3 source 731.25 MiB、codec first upsampler source 108 MiBに分解できる。
全360 requestはcampaign controlのaudio hash `0e1ac1…cacd`にbitwise一致した。

Python比較も旧値を流用せず、
`/home/sanzentyo/benchmark-artifacts/irodori-v4-python-all-resident-refresh-20260812-attempt1`
で5 fresh process、各2 warmup + 10 measuredを取り直した。`SHA256SUMS`のSHA-256は
`d2a9703be227d808cbfa08a321aea61355b3aca63441e51e5b5adf94964d4f84`である。

| 同一consumer境界 | Python all-resident | Rust combined fixed112 | Rust差 |
|---|---:|---:|---:|
| steady latency | 313.29 ms | 213.28 ms | 1.47x高速 |
| persistent allocated / in-use | 3,449.44 MiB | 3,539.06 MiB | +89.62 MiB |
| request peak reserved | 4,540.00 MiB | 4,514.19 MiB | -25.81 MiB |
| external NVML peak | 4,756 MiB | 5,501 MiB | +745 MiB |

これにより以前のRust約7 GiB対Python約4 GiBという差の大半は、fused/packed高速化cacheのsource重複、
未使用QKV layout、codec source、allocator page余白だったと確認できた。packed layoutは保持したためsteady
高速化は失っていない。allocator内部のpersistentはほぼPythonと同水準になったが、NVMLにはWGPU/Vulkan
device・pipeline・driver allocationがCubeCL tensor accounting外に約745 MiB多く残る。この差をlive tensor
削除量とみなして無理に回収しない。

APIはshape sentinelを使わない。`WgslWeightProfile::{PortableFallback, Fixed112OneLayout,
Fixed112PackedOnly}`をengine構築時に選び、固定profileは112以外をsampling前に`Result::Err`で拒否する。
codecは`DacVaeDecoder -> Result<Fixed112DacVaeDecoder>`の消費的遷移で、変換時にsource weight、
prepared polyphase cache、deviceを検証してからsource allocationを解放する。`weight.dims()==[1,1,1]`を
状態として判定するassert/panicは採用していない。tombstoneは解放済み`Param`の型を保つ内部実装に限り、
状態と入力契約はenum/newtype/`Result`が持つ。
profile lock後のengineから汎用model参照を取り出せないよう、`portable_model()`は
`PortableFallback`の場合だけ`Some`を返す。固定profileの操作はframe数を検証するengine methodを経由する。

portable性は二層に分かれる。exact-only loaderとcodec decoder分離はbackend非依存である。profile ADT、
消費的遷移、fail-closed validationも他backendへ移植できる。一方、どのphysical layout/sourceを捨てて
よいかはWGPU kernel profile固有であり、Metal/DX12でも同じ速度・削減量になるとは仮定しない。
固定profileは多長serviceのdefaultではなく、許可shapeをmanifestで閉じたsession専用policyである。

temporary/reserved差については、combinedのpersistent in-use 3,539.06 MiBに対しrequest reserved peakは
4,514.19 MiBで、約975.13 MiBがrequest中の再利用allocation/page余白として残る。これを毎request
cleanupすると再確保を招き、既存campaignでもthroughputが低下したため採用しない。次の削減対象は
allocatorの数字を強制的に下げることではなく、first RF/codecで生存期間が重なるtensorをstage traceで
特定してbuffer reuseすることである。

### persistent autotune cache再測定

`configure_cubecl_persistent_cache`を追加し、applicationがadapter/backend fingerprintを含むrootを渡すと、
autotuneとcompilation cacheをCargo `target`外に置けるようにした。`pipeline`、residency benchmark、
precision validatorはいずれもCubeCL初期化前に`--cubecl-cache-dir`を適用する。これはnative filesystemを
持つVulkan/Metal/DX12では同じAPIで実装可能だが、browser WebGPUは別storage adapterが必要である。

空cacheの最初の112-frame requestはRF 5.968 s + codec 3.631 sだった。同じcacheを使う次processでは
first consumerが0.515 s、steadyが0.199–0.200 sになった。diskで確認できたのは約108 KiBのautotune
logであり、WGSL compiled pipeline blobは生成されなかった。従ってcross-processで確認済みなのは
autotune winner再利用であり、portableなshader machine-code cacheとは呼ばない。process-local WGPU
pipeline cacheとvendor driver cacheは別層で、production ready条件にはlong-lived processのshape warmupを
引き続き使う。

cacheはaccuracy approvalではない。fresh campaignの112 framesはPyTorch oracleに対してlatent SNR
106.16 dB、waveform SNR 93.46 dBだった一方、45 framesはreduction treeにより82.94–85.60 dBへ
変動した。codec GEMMを`MatmulStrategy::Cube`へ固定する切り分けは82.94 dBで改善せず、
不採用/revertした。失敗artifactは
`/home/sanzentyo/benchmark-artifacts/irodori-v4-codec-cube-accuracy-20260812-attempt1`に保存した。

### 6長さの同値性とfresh autotune accuracy注意

`/home/sanzentyo/benchmark-artifacts/irodori-v4-12gb-vram-opt-accuracy-20260812-attempt2`
（`SHA256SUMS` SHA-256
`af2596d1c34fbe89985e1a27d60194f8d3aa98960ae1f4528a76636afa0fcc9f`）で
45/112/255/333/489/685 framesを各3 request、
同一current binaryのfull codec / decode-onlyで直接比較した。全6長さで各runtime内deterministicかつ
full/decode-onlyのaudio SHA-256が一致した。したがってencoder除去はdecode数値を変えていない。

一方、旧campaignのaudio hashをcurrent binaryのoracleにしたattempt 1は、旧測定値を新campaignへ
流用しない原則により45 framesでfail-closedした。さらにcurrent `validate_v4_precision`を同じfixtureへ
85 dB waveform gate付きで実行したところ、45 framesは82.939 dBで失敗した。同じcurrent binaryの
SubSlices controlも同一hash・同一82.939 dBであり、ExclusivePagesやdecode-onlyによる回帰ではない。
baseline時は85.605 dBだったためautotuneを再調査した。

5 fresh sessionの45-frame campaignでは85.605 dBのPASSが2件、82.939 dBのFAILが3件だった。最速sessionは
FAIL側であり、最速winnerを無条件に保存できない。PASS/FAILの完全selection vectorは主にreduce winnerで
異なり、あるvectorでは単一matmul keyの変更で改善しても、別vectorで同じ変更を行うと悪化した。
したがってcandidate単体を独立に承認せず、runtime identityと全cache entryからなるselection vector全体を
fixture evidenceと一緒にsealする。45-frame成功campaignは
`/home/sanzentyo/benchmark-artifacts/irodori-v4-autotune-accuracy-tristate-20260812-attempt6`
（`SHA256SUMS` SHA-256
`8a8847057756d1f8aa2ad01f936da94c5acad91a1bd5f3c0f6354d757f8bdd02`）である。

### accuracy gateの役割分離

85 dBは音声品質の境界ではなく、PyTorch FP32の丸め順に非常に近いことを要求する
numerical-reproducibility targetとして扱う。実際、45-frameの83.00 dB条件もmax abs
`1.30e-4`、RMSE `8.74e-6`、cosine `0.999999997496`であり、85.60 dB条件とのRMSE差は
`6.48e-6`対`8.74e-6`である。reduction treeの加算順を再現するためだけに全長共通vectorを失うのは、
production correctnessの目的と一致しない。

version 2のapproval policyは次の三層に分ける。

- latent hard gate: max abs `2e-4`、mean abs `1e-5`、RMSE `2e-5`、SNR 90 dB、
  cosine `0.99999999`。RF意味論の回帰を厳しく止める。
- waveform hard gate: max abs `1.5e-4`、mean abs `5e-6`、RMSE `1e-5`、SNR 80 dB、
  cosine `0.99999999`。すべてを満たす必要がある。
- waveform target: SNR 85 dB。未達はwarningとして保存するがhard failureにはしない。

80 dB hard gateは知覚的同一性を証明するものではない。現時点では聴取試験、ABX、PESQ/STOI等を
実施していないため、「不可聴」とは断定せず、strict-FP32 referenceに対するengineering toleranceと呼ぶ。
同一approved vector、device identity、fixture内ではlatent/waveform SHA-256の一致を要求するが、異なる
正当なreduction tree間のhash一致は要求しない。

この定義で実行した正式campaignは
`/home/sanzentyo/benchmark-artifacts/irodori-v4-six-length-approved-autotune-20260812-attempt3`
で、top-level `SHA256SUMS`のSHA-256は
`7a11414ec0a7f339b34fc450817415cf98ea0fee80f1c3341ce71889a4c1838e`である。各長さ5 fresh session、
計30/30がhard PASSした。86-entry selection vectorをseal後、別cacheへrestoreし、全6長さを各2 repeatして
全hashの決定性とvectorの完全一致を再検証した。automatic retryはなく、旧測定値をpoolしていない。

| frames | latent max abs / SNR | waveform max abs / RMSE / SNR | 判定 |
|---:|---:|---:|---:|
| 45 | `6.07e-5` / 104.22 dB | `1.27e-4` / `8.56e-6` / 83.18 dB | hard PASS、target warning |
| 112 | `3.22e-5` / 106.16 dB | `3.16e-5` / `1.92e-6` / 93.46 dB | hard/target PASS |
| 255 | `5.51e-5` / 100.66 dB | `5.79e-5` / `2.19e-6` / 92.56 dB | hard/target PASS |
| 333 | `9.16e-5` / 99.22 dB | `3.41e-5` / `2.06e-6` / 92.62 dB | hard/target PASS |
| 489 | `4.41e-5` / 103.53 dB | `9.71e-5` / `2.21e-6` / 91.63 dB | hard/target PASS |
| 685 | `1.84e-4` / 99.41 dB | `3.90e-5` / `1.91e-6` / 92.47 dB | hard/target PASS |

85 dBのみをhard gateにしたattempt 2は333 framesで5/5失敗し、retryせず
`irodori-v4-six-length-approved-autotune-20260812-attempt2/FAILURE`として保存した。これは削除せず、
gate変更理由のnegative evidenceとする。

### Burn 0.22移行で維持する実行基盤方針

この移行とfresh再計測は
[`rtx-5070ti-burn-0.22-cache-migration-2026-08-12.md`](rtx-5070ti-burn-0.22-cache-migration-2026-08-12.md)
で完了した。以下は移行前に固定した設計方針であり、実測値とOOM条件はリンク先を正とする。

Burn `0.22.0-pre.2`、burn-cubecl `0.22.0-pre.2`、CubeCL `0.11.0-pre.2`へexact pinで移行する。
最初はFusionを無効にしたraw pathで全6長さparityを取り、0.21のapproved cacheや計測値を0.22へ
流用しない。0.22用のruntime/device/kernel identityで新しいapproval manifestを作る。

production backendはdispatch costと状態空間を抑えるためWGPUだけに固定する。`burn`は
`default-features = false`で`std`、`wgpu`、`autotune`、`template`、`extension`だけを有効化し、
`cubecl`もdefault featureを切って`std`、`stdlib`、`template`、`wgpu`だけを明示する。
`burn-cubecl`はこのraw-parity段階では`std`だけで、Fusionはまだ有効にしない。feature tree上、
CPU、CUDA、ROCm、NdArray、LibTorch、Vulkan専用、WebGPU専用のdispatch variantは存在しない。
`burn-flex` crateは`burn-dispatch`のnon-optionalな内部依存としてlink closureに残るが、`flex`
featureと`Flex` dispatch variantは無効である。これはCPU fallbackをproductionへ残すことを意味しない。
PyTorchはrepository外の比較harnessだけに維持する。Metal/DX12/Vulkanの選択はWGPU内部のadapter
選択であり、Burn backendを増やさずcross-platform性を維持できる。
`burn/template`は`burn-wgpu`の`SourceKernel`、`SourceTemplate`、`into_contiguous`公開を
gateするため必要で、直接依存の`cubecl/template`と両方を維持する。

productionの最終形は通常のBurn graph/built-in Fusionを維持し、Irodori固有WGSLをcustom Fusion
providerとして登録する。大規模projection matmulはBurnに残し、QKV postprocess、長尺attention、
post-SDPA、必要なcodec residualだけを段階的にprovider化する。unsupported shapeは明示fallbackへ流し、
model本体でpanicやweight shape sentinelによりrouteを判定しない。raw backendはparity oracle、kernel単体
benchmark、fallback検証へ限定する。

0.22.0-pre.2の配布READMEには旧`Tensor<B, D>`例が残るが、実際に解決された`burn-tensor`と`burn-nn`は
`Tensor<const D, K>`とbackend genericなしのmoduleを公開する。コンパイル対象の型定義を正として
`Tensor<D>`/`Device`へ移行し、strict FP32をdevice policyとwarmup/approval manifestで検査する。
`CubeBackend`もruntime型だけを取る新signatureへ更新する。primitive/handle/launcherは
`backend_bridge`へ隔離する。sessionは
`RuntimeBuilder<Cold> -> Runtime<Loaded> -> Runtime<Warmed> -> OnlineSession<Ready>`、weight解放は
`PreparedModel<PortableFallback> -> PreparedModel<ProfileLocked>`の不可逆遷移で表し、required shapeの
fallbackが0件になるまでsource weightをdropしない。既存のdecode-only codec、`ExclusivePages`、
fixed112 packed-onlyの速度/VRAM成果は0.22移行後の回帰gateとする。

## Online resident / speaker switching（PyTorch現行public runtime）

各fresh sessionで2 warmup + 10 measured、計5 sessions。wallはfinal owned CPU audioまでを
含む。first requestは各scenarioの最初のwarmup request、steadyはsession内measured medianを
さらに5 sessionsでmedianした。

| 条件 | first request | steady latency | requests/s | audio-s/wall-s | peak allocated |
|---|---:|---:|---:|---:|---:|
| text-only | 370.3 ms | 345.1 ms | 2.898 | 12.982 | 4,022.9 MiB |
| voice design fixed | 331.3 ms | 356.4 ms | 2.806 | 12.572 | 4,123.0 MiB |
| voice design A/B | 354.2 ms | 356.6 ms | 2.804 | 12.562 | 4,124.5 MiB |
| prepared clone fixed | 350.7 ms | 361.0 ms | 2.770 | 12.411 | 4,145.9 MiB |
| prepared clone ref1/ref2 | 374.5 ms | 361.9 ms | 2.763 | 12.378 | 4,149.4 MiB |
| raw clone fixed | 462.2 ms | 469.2 ms | 2.131 | 9.548 | 4,149.4 MiB |
| raw clone ref1/ref2 | 506.7 ms | 480.6 ms | 2.081 | 9.321 | 4,151.9 MiB |

design A/Bのswitch差はfixed比median -4.2 ms、prepared cloneは+0.94 msであり、session間の
揺れより小さく、追加switch costを検出できなかった。raw clone switchは+7.2 msだが、
requestごとのreference encodeを含む。`prepare_reference` stage medianはprepared clone
0.39 ms、raw fixed 127.7 ms、raw switch 145.1 ms。one-time grouped preparationはref1/ref2で
各session 101–304 msだった。hash determinismはvoiceごとに判定し、A/Bまたはref1/ref2の
異なるhashを非決定性として扱っていない。

0.21 baseline時点のWGPUにはmodelと`PreparedSpeaker`を保持して交互requestを受ける高水準session
APIがなく、同じonline matrixは測定不能だった。0.22移行で`OnlineSession<SessionReady>`の基礎は
追加したが、`PreparedSpeaker` public ADTと同じspeaker-switching matrixは次cycleに残る。

## Strict FP32 PyTorch / WGPU comparison

両runtimeともstrict FP32、TF32 off、autocast off、4 Euler evaluations、forward batches
`[2,2,1,1]`、effective rows 6、12 layers、48 block calls。runtime work manifestで
schedule bits `[1065336439,1061146329,1056947831,1048559223,0]`を照合した。同じRF意味論と
request/source noiseを使うが、same operator graphではない。

各長さは同一process内の2 warmupを除外し10 measured。device-completeはpre-start syncから
device completion、readback-completeはowned contiguous FP32 CPU resultまでで、片runtimeだけ
readbackを含めていない。

| sec / frames | Py RF dev/rb | WG RF dev/rb | Py codec dev/rb | WG codec dev/rb | accuracy |
|---|---:|---:|---:|---:|---|
| 1.80 / 45 | 146.531 / 146.573 ms | 109.765 / 109.927 ms | 35.180 / 35.288 ms | 34.448 / 34.828 ms | pass |
| 4.48 / 112 | 187.090 / 187.131 ms | 117.360 / 117.854 ms | 122.342 / 122.543 ms | 81.772 / 82.306 ms | pass |
| 10.20 / 255 | 248.278 / 248.331 ms | 202.534 / 203.010 ms | 288.586 / 289.025 ms | 195.369 / 196.378 ms | pass |
| 13.32 / 333 | 310.693 / 310.754 ms | 254.230 / 254.678 ms | 359.934 / 360.311 ms | 274.298 / 275.305 ms | pass |
| 19.56 / 489 | 373.648 / 373.723 ms | 377.704 / 378.227 ms | 606.935 / 607.556 ms | 400.116 / 401.375 ms | pass |
| 27.40 / 685 | 514.845 / 514.938 ms | 538.231 / 538.673 ms | 787.418 / 788.240 ms | 578.079 / 579.817 ms | pass |

489-frame waveformはmax abs `8.296966553e-5`、RMSE `2.457548602e-6`、SNR
`90.724931 dB`、cosine `0.999999999577`で、85 dBを含むwaveform gateを通過した。
685 framesもpassした。WGPU RFは489/685 framesでPyTorchより遅く、全長で優位という
performance PASSは出さない。codecは全長でWGPU medianが短い。

## Duration prediction

duration predictionも6長さで実行し、各runtime/長さ3 fresh process、各scope 5 warmup +
10 measured。resolved framesは45、112、255、333、489、685で一致した。

| frames | WGPU full device | PyTorch full device |
|---:|---:|---:|
| 45 | 13.214 ms | 65.989 ms |
| 112 | 23.287 ms | 66.245 ms |
| 255 | 24.589 ms | 66.046 ms |
| 333 | 26.957 ms | 66.125 ms |
| 489 | 29.445 ms | 65.684 ms |
| 685 | 29.798 ms | 65.840 ms |

readback-completeも両runtimeで別記録されており、full scopeは全caseでWGPUの全測定点が
PyTorch minimum未満だった。head scopeの一部tailは重なるため、headの全点優位は主張しない。

## VRAM節約型 phase batch

測定runnerは既存の型状態遷移
`PhaseBatch<RfResident → LatentsResident → CodecResident → Complete>`を直接使用した。
全requestをRF中にsampleし、latentのCPU readbackなしでRFをdrop、backend cleanup後にcodecを
loadし、consumerがfinal audioだけをCPUへ取得した。実行wallにはRF phase、codec load、codec
phaseを含み、初期RF model loadは別の`load_wall_seconds`である。各条件はretryなしの単発
fresh processなので、非単調な値を選別していない。

| N | speaker | length | latency/request | requests/s | audio-s/wall-s | NVML peak |
|---:|---|---|---:|---:|---:|---:|
| 1 | same | same | 8,481.0 ms | 0.118 | 0.528 | 7,042 MiB |
| 2 | same | same | 4,326.9 ms | 0.231 | 1.035 | 7,042 MiB |
| 2 | multi | same | 4,605.2 ms | 0.217 | 0.973 | 7,042 MiB |
| 4 | same | same | 2,274.2 ms | 0.440 | 1.970 | 7,042 MiB |
| 4 | multi | same | 2,318.7 ms | 0.431 | 1.932 | 7,042 MiB |
| 8 | same | same | 1,310.3 ms | 0.763 | 3.419 | 7,042 MiB |
| 8 | multi | same | 1,301.2 ms | 0.769 | 3.443 | 7,042 MiB |
| 12 | same | same | 967.4 ms | 1.034 | 4.631 | 7,042 MiB |
| 12 | multi | same | 973.5 ms | 1.027 | 4.602 | 7,042 MiB |
| 2 | same | mixed | 6,539.8 ms | 0.153 | 0.480 | 7,042 MiB |
| 2 | multi | mixed | 6,200.7 ms | 0.161 | 0.506 | 7,042 MiB |
| 4 | same | mixed | 5,316.7 ms | 0.188 | 1.401 | 7,106 MiB |
| 4 | multi | mixed | 4,484.3 ms | 0.223 | 1.661 | 7,106 MiB |
| 8 | same | mixed | 4,994.0 ms | 0.200 | 2.078 | 7,426 MiB |
| 8 | multi | mixed | 3,278.0 ms | 0.305 | 3.167 | 7,234 MiB |
| 12 | same | mixed | 2,484.5 ms | 0.402 | 5.149 | 7,234 MiB |
| 12 | multi | mixed | 2,427.2 ms | 0.412 | 5.271 | 7,234 MiB |

N=1のmixedは複数長を含められないためdegenerateであり、mixed tableから除外した。
same-length N=12ではfirst itemのRF/codec device-completeが5.796/2.162 s、以降の代表値は
RF 181–243 ms、codec 81–126 msだった。これはfirst-shape compiler/autotune costと
cache-warm workを分離して保持している。stage persistent allocatorはRF residentでin-use
4,415.8 MiB/reserved 6,816 MiB、N=12 latents-onlyで0.16/16 MiB、codec+latentsで
486.2/1,000 MiB。same-length CFG topologyはtensor micro-batch候補である。

mixed N=8 same-speakerの39.95 sという外れ気味の単発値も除外しなかった。長さごとの初回
shader/autotuneを含むため、このcycleではphase schedulingの方向性は示せるが、安定した
capacity curveとは扱わない。

## Crate ergonomics

良い点は、`PhaseBatch`がRFとcodecの同居を誤って起こせない型状態、uniqueな`BatchItemId`、
`SpeakerKey`、`VoiceIdentity`、final GPU audioを一度だけ渡すconsumer境界を既に持つこと。
今回もlatent readbackなしの契約をproduction APIのまま実行できた。

今回追加した`DacVaeDecoder`はencode capabilityを型から除き、decode-only serviceがencoderを
誤ってresidentにする状態を表現不能にした。`PhaseBatch<LatentsResident>::with_decoder`も同じ型を
受け取り、full codecを渡す互換APIはconsumeしてdecode-only stateへ遷移する。

Burn 0.22移行ではさらにbackend genericをpublic tensor/moduleから除き、
`OnlineSession<Unwarmed> -> OnlineSession<SessionReady>`、`WarmupManifest`、`WarmupPlan`を追加した。
ready sessionはmanifest外のframe/topologyをsampling前に拒否する。named environment、bundle receipt、
strict F32/I32 device policyもruntime初期化境界へ集約した。

一方、実測runnerでは次の不足により低水準tensor shapeとpaired `Option`を直接扱う必要があった。

- `RuntimeBuilder<Cold> -> Runtime<Loaded> -> Runtime<Warmed>`を一つのpublic builderへ統合していない。
- reference preparationは高水準library APIではなくbinary/private preprocessingに閉じている。
- `PreparedSpeaker`、`Voice::{Unconditioned, Clone, Designed}`、`Duration::{Predict, Exact, Frames}`
  がrequest boundaryにない。
- `SamplingRequest`の`ref_latent/ref_mask`と`caption_ids/caption_mask`はpaired `Option`で、invalid
  stateを表現できる。
- `RequestId`、`SpeakerId`、`ModelGroupId`、`OutputSeconds`、`LatentFrames`、
  `DiffusionSteps`がnewtypeでなく、単位とidentityをcallerが維持する。
- high-level all-residentとphase batchのpolicy選択、persistent prepared-speaker cache、
  requestごとのconsumer-complete telemetryが統合されていない。

## 次cycleの優先順位

0.22 raw parity、WGPU-only feature closure、named environment/bundle、DryRun、
`OnlineSession<SessionReady>`、6長accuracy再承認までは完了した。以後の優先順位は次である。

1. v3 six-shape bundleと12/12 accuracy結果をapproval manifestへ固定し、fixed112
   `OnlineSession<SessionReady>`をpublic pipelineへ接続する。
2. 12GBでOOMしたuniversal all-residentを再試行せず、RF/codec phaseを分けたfail-closed cache builderを
   実装する。任意長runtimeはphase batchを使う。
3. primitive/handle変換を`backend_bridge`へ隔離し、その後SwiGLU postprocessでcustom Fusion providerを
   最小実証する。provider hit/fallback、compile、autotune、kernel hashを記録する。
4. `PreparedSpeaker`、`Voice`、`Duration`、ID/unit newtypeを追加し、paired `Option`をrequest boundaryから
   除く。
5. `ExclusivePages`とdecode-only codecをMetal/DX12 adapterでも測り、backend別policyが必要か決める。
   warmup後cleanupは追加削減がなく不採用。Vulkan限定`PipelineCache`はcold startup実験へ分離する。
6. `PreparedSpeaker`とreference cacheをlibrary化し、raw clone encodeとcache-hit switchingを
   WGPUでも同じmatrixで測れるようにする。
7. 489/685-frame hard gateを回帰testとして固定した上で、長尺RF providerを優先する。短尺より
   489/685でPyTorchに逆転されている。
8. 112-frame profile-locked source解放をpublic sessionへ接続する。45/489/685を同じprofileへ入れず、
   長さごとにaccuracyとroute manifestを通したprofileだけを追加する。
9. RF latentをcodecまでGPU residentのまま維持し、finite checkをGPU reductionへ移す。
   tail検出のfull latent readbackは導入しない。
10. same length/CFG topologyのtensor micro-batchを検討し、sequential phase batch N=12の
   1.03 requests/sを次の比較基準にする。
11. all-resident（latency）とphase batch（VRAM/throughput）を同じpublic request ADTで再測定する。
12. reject済みscript/kernel、WGSL文字列assertion、非WGPU backendの整理はその後に行う。

BF16はこのcampaignで実行していない。

## Artifactと再開手順

machine-readable集計はartifact rootの`evidence.json`、raw dataは`cold/`、`sessions/`、
`duration-attempt2/`、`accuracy-campaign/`、`phase-batch/measurements/`にある。各conditionは
JSON/stdout/stderr/wall/NVML/SHA256SUMSを持つ。nested manifestはすべて`sha256sum -c`で
検証済みである。

VRAM最適化のraw 20 sessions/NVMLは
`/home/sanzentyo/benchmark-artifacts/irodori-v4-12gb-vram-opt-20260812-attempt1`、6長さの
full/decode同値性は`irodori-v4-12gb-vram-opt-accuracy-20260812-attempt2`にある。旧hashを要求して
停止したaccuracy attempt 1と45-frame oracle gate failureも削除せず、それぞれ`FAILURE`付きで保存した。
accuracy-approved cacheのraw 30 fresh sessions、全6長さrestore logs、NVML、manifestは
`/home/sanzentyo/benchmark-artifacts/irodori-v4-six-length-approved-autotune-20260812-attempt3`にある。
Burn 0.22/cache migrationのfresh/restored JSON、accuracy logs、OOM条件、NVML、bundleは
`/home/sanzentyo/benchmark-artifacts/irodori-v4-burn022-cache-20260812-attempt1`にある。

再開時は次の順で行う。

1. `git switch codex/v4-wgsl-fusion`後、このreportを追加したcommit以降であることを確認する。
2. artifact rootのtop-level `SHA256SUMS`を`sha256sum -c SHA256SUMS`で検証する。
3. `environment/nvidia-smi-query.csv`と`wgpu-adapter.json`を再確認し、GPU名、PCI、driver、
   adapterが変わった場合は新campaignにする。
4. model/codec/source/binary SHAを`models/SHA256.txt`、各campaignの`pins.sha256`で確認する。
5. `approved/cache-manifest.json`とruntime identityを照合し、0.21の再現時は全6長さhard gateを
   fresh outputで実行する。45-frameの85 dB未達はwarning、hard gate失敗はretryなしでfreezeする。
6. 0.22 v3 environment/bundleを使う場合もruntime/device/kernel identityと12/12 accuracyを照合する。
   0.21 cacheはコピーしない。
7. `PreparedSpeaker`を`OnlineSession`へ接続し、2 warmup + 10 measured、可能なら5 fresh sessionsで
   WGPU online matrixを埋める。
8. `Runtime<Warmed>`のshape manifestを固定し、all-residentでcleanupなし/あり、SubSlices/
   ExclusivePagesをadapterごとに独立条件として再測定する。旧campaignとsampleをpoolしない。
9. fresh autotune hard gateを通した後に、profile-locked source weight解放、長尺RF、GPU finite
   reduction、tensor micro-batchの順で進める。
