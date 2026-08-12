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

WGPUにはmodelと`PreparedSpeaker`を保持して交互requestを受ける高水準session APIがなく、
同じonline matrixは測定不能だった。これは0 msや未試行成功ではなくAPI制約である。

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

一方、実測runnerでは次の不足により低水準tensor shapeとpaired `Option`を直接扱う必要があった。

- `RuntimeBuilder<Cold> -> Runtime<Ready>`と`OnlineSession<Ready>`がない。
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

1. production演算の前に、高水準ADT/type-state session APIを追加し、今回のall-resident probeを
   `RuntimeBuilder<Cold> -> Runtime<Ready>`と`OnlineSession<Ready>`へ昇格する。
2. `PreparedSpeaker`とreference cacheをlibrary化し、raw clone encodeとcache-hit switchingを
   WGPUでも同じmatrixで測れるようにする。
3. 489/685-frame accuracy gateを回帰testとして固定した上で、長尺RFを優先する。短尺より
   489/685でPyTorchに逆転されている。
4. RF latentをcodecまでGPU residentのまま維持し、finite checkをGPU reductionへ移す。
   tail検出のfull latent readbackは導入しない。
5. same length/CFG topologyのtensor micro-batchを検討し、sequential phase batch N=12の
   1.03 requests/sを次の比較基準にする。
6. all-resident（latency）とphase batch（VRAM/throughput）を同じpublic request ADTで再測定する。
7. reject済みscript/kernel、WGSL文字列assertion、非WGPU backendの整理はその後に行う。

BF16はこのcampaignで実行していない。

## Artifactと再開手順

machine-readable集計はartifact rootの`evidence.json`、raw dataは`cold/`、`sessions/`、
`duration-attempt2/`、`accuracy-campaign/`、`phase-batch/measurements/`にある。各conditionは
JSON/stdout/stderr/wall/NVML/SHA256SUMSを持つ。nested manifestはすべて`sha256sum -c`で
検証済みである。

再開時は次の順で行う。

1. `git switch codex/v4-wgsl-fusion`後、このreportを追加したcommit以降であることを確認する。
2. artifact rootのtop-level `SHA256SUMS`を`sha256sum -c SHA256SUMS`で検証する。
3. `environment/nvidia-smi-query.csv`と`wgpu-adapter.json`を再確認し、GPU名、PCI、driver、
   adapterが変わった場合は新campaignにする。
4. model/codec/source/binary SHAを`models/SHA256.txt`、各campaignの`pins.sha256`で確認する。
5. まず489/685 accuracyをfresh outputで実行する。失敗時は性能値をPASSにせず、その条件を
   retryなしでfreezeする。
6. `OnlineSession`/`PreparedSpeaker`の最小APIを実装し、2 warmup + 10 measured、可能なら
   5 fresh sessionsでWGPU online matrixを埋める。
7. その後に長尺RF、GPU finite reduction、tensor micro-batchの順で最適化し、旧campaignと
   sampleをpoolしない。
