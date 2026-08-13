# RTX 5070 Ti Laptop 12 GiB: v4 F16 WGPU optimization (2026-08-13)

## 結論

F16はproduction defaultにはせず、明示選択するexperimental policyとして実装した。50 latent
frames（48 kHz、96,000 samples、約2.0秒）の固定fixtureでは、WGPU AutoCompilerへ
CubeCL SPIR-V compilerを追加し、Burn生成matmulをCMMMAへ送る構成が、PyTorch F16より速くなった。
高速SubSlices profileの5 fresh process median-of-session-mediansはRF + codecのdevice-completeが
56.333 ms、readback-completeが58.014 msである。PyTorch F16の同じ境界は66.892 / 67.375 msなので、
Rustはそれぞれ15.79% / 13.89%短い。低VRAM ExclusivePages profileもscalar-store時点で59.551 / 60.971 ms、
PyTorchより10.97% / 9.50%短かった。最終vec4-store sourceで取り直したExclusivePagesは
60.198 / 61.270 msで、それでもPyTorchより10.01% / 9.06%短い。

Burnのdispatch backendはWGPU一つだけである。Vulkanでは生成演算をSPIR-Vへcompileする一方、
44組のF32/F16 kernelと2本のF16専用kernelはWGSL `SourceKernel`のまま動く。Metal/DX12等ではAutoCompilerが対応する
表現を選ぶため、この依存設定自体はcross-platformである。ただしCMMMAによる今回の速度向上は
Vulkan/NVIDIAでの実測であり、他platformの速度を外挿しない。

PyTorch F16との最終waveform一致はSNR 31.384 dB、cosine 0.999636727754だった。
これは音声一致として良好だが、1長・1条件だけなのでproduction採用gateには
足りない。F32 defaultは維持し、六つの長さ、text-only/design/clone、fresh/restored cacheを
通すまでF16を自動選択しない。

## WGPU-only hybrid compilerでPyTorch F16を上回った

比較requestは2.0秒、50 latent frames、text-only、Euler 4 evaluations、forward batches
`[2,2,1,1]`、effective rows 6、12 layers、48 block callsで固定した。各stageはpre-start device syncから
device completion、またはowned contiguous CPU F32 readbackまでを測る。RFとcodecは別々にloopした
stage medianの和であり、一回のconsumer-complete E2E latencyとは呼ばない。PyTorchとWGPUはsame
semantic workだがsame operator graphではない。

| runtime | aggregation | RF device / readback | codec device / readback | independent stage sum device / readback |
|---|---|---:|---:|---:|
| Rust WGPU AutoCompiler F16、SubSlices fast profile | RF/codecを独立に5 fresh process × 10、repeat 1除外、session medianのmedian | **31.497 / 32.063 ms** | 24.836 / 25.951 ms | **56.333 / 58.014 ms** |
| Rust WGPU AutoCompiler F16、ExclusivePages low-VRAM profile | 最終vec4 source、同protocol | 34.915 / 35.359 ms | 25.283 / 25.911 ms | **60.198 / 61.270 ms** |
| PyTorch CUDA F16 | 1 loaded process × 6、repeat 1除外、median | 53.501 / 53.532 ms | **13.391 / 13.843 ms** | 66.892 / 67.375 ms |

direct residue store前のSubSlices session別device-complete和は56.688、59.920、57.970、57.502、
58.550 msだった。採用後のcodec session medianは25.825、38.309、25.573、25.691、25.763 msで、
GPU競合直後の38.309 msを残したまま中央値25.763 msだった。後述のdilation=1 zero-copyと
vec4 store追加後は24.836 msまで短縮した。Rustはcodec単体ではPyTorchより11.445 ms遅いが、
RFを22.004 ms短縮し、独立stage中央値の和で10.559 ms上回る。したがって次の
性能優先箇所はcodecであり、RFのCMMMA routeを崩してまで
巨大なmonolithic shaderへ置換しない。

CubeCL environment DBには`wgpu<spirv>` namespaceが作られ、fresh runで観測した34個のmatmul
autotune keyはすべてCMMMA候補を選んだ。内訳はdouble cyclic specialized 14、simple cyclic 8、
double cyclic 7、ordered double 4、simple multi-row 1である。SPIR-V binary storeは193 entriesだった。
WGSL-only compilerではcooperative-matrix候補がtile未対応として落ち、通常matmulを選んでいたため、
今回の主因はdispatch backend追加ではなく、同じWGPU runtime内でのcompiler/candidate解禁である。

精度は5 processでhashまで一致した。PyTorch F16に対するlatentはSNR 43.220 dB、cosine
0.999976822061、max abs 0.041015625、最終waveformはSNR 31.384 dB、cosine
0.999636727754、max abs 0.055977だった。codec-onlyはwaveform SNR 56.327 dB、cosine
0.999998842855、max abs 0.004150だった。F16 gateはPyTorch F16とのsame-precision parityとして
評価し、F32 numerical-reproducibility用85 dB gateをそのまま適用しない。

strict F32も新compilerで回帰した。fresh autotune後のsteadyはRF 86.383--86.821 ms、codec
33.831--36.018 ms、latent SNR 104.791 dB、waveform SNR 92.081 dBで、各repeatのhashも同一だった。
したがってF16の速度向上のためにF32の精度や速度を落としてはいない。

## first request、cache、VRAMは別条件で扱う

fresh CubeCL environmentでは最初のRFが22.194--22.326秒、codecが9.843--10.951秒だった。これは
autotuneとprocess-local pipeline compileを含む。v5 cache restore済みfresh processの代表値は最初の
RF/codecが51.743 / 38.585 msだった。ただしSubSlicesの5 processではRF firstが45.985--145.639 msと
process-local compileの外れ値を残したため、steadyへ混ぜない。以後のsteadyは上表の範囲へ落ちた。
SubSlicesでのmodel loadは5 processで5.505--5.795秒だった。永続environmentは
autotune選択とVulkan SPIR-V storeを再利用するが、process-local pipeline objectと手書きWGSLのdriver
compileを完全には永続化しない。long-lived session、readiness前warmup、実requestによるaccuracy確認は
引き続き必要である。

NVML process peakはfresh autotune時3,381--3,383 MiBだった。restored/process-warmはSubSlices fast
profileが最終sourceで3,093 MiB、ExclusivePages low-VRAM profileの最終再計測が2,367 MiBである。Rust allocatorは
RF終了直前に2,328,647,744 bytes（約2,221 MiB）をreservedし、RF-to-codec cleanup後13,184 bytesまで
解放した。PyTorchはsteady peak allocated 2,101.6 MiB、reserved 3,708 MiBである。NVML process値と
PyTorch allocator値は同じmetricではないため、RustがPyTorchより何MiB多いという直接差には使わない。
fresh autotuneの一時bufferをsteady persistent値へ混ぜず、restored peakをservice設計の基準とする。

`tasks_max`はSPIR-V有効後、direct residue store導入前に再探索した。SubSlicesの16/64は明確に遅く、
48も5-session中央値58.100 msで32の57.970 msを下回らなかったため、production候補は32を維持する。
最終sourceのExclusivePagesでもtasks48をscreenしたが、RF 34.560 / 35.254 msでtasks32の
5-session中央値34.435 / 34.838 msを下回らなかったため、追加sessionへ昇格しなかった。
F16 accumulatorへ
全面変更する案はwaveform SNR 43.51 dBまで落ち、
im2col + generic CubeCL matmul codec案はaccuracyを通したがcodec 36.85 msへ悪化したため、どちらも
不採用にした。失敗logはfresh campaign内に保存し、成功値へpoolしていない。

profile feature限定の一時sync instrumentationでresidue pack/coreも分離した。processとpipelineがwarmな
profile repetitionでは、C384/C192/C96のd3/d9を合わせた6 packが2.417732 ms、対応する6 coreが
9.676377 msだった。したがってpack完全除去の上限はこの2秒fixtureで約2.4 msであり、codecの
PyTorch差13.082 msすべてを説明しない。一時instrumentationは計測後にsourceから除去し、raw logだけを
`codec-f16-residue-split-instrumentation.log`として保存した。

この計測に基づき、`PreparedActivation::{Ncl, ResiduePacked { tensor, dilation }}`を実装した。
F16のunit 0/1 pointwise-residual/next-Snake finalizerは次unitのd3/d9 compact layoutへ直接scatterし、
次unitはpack dispatchなしでcoreを実行する。F32、非対応shape、cache/resource mismatchは従来NCL経路へ
fail-closedで戻る。coreが実行直前に拒否した場合もraw identity shortcutから`act0`を再生成できるため、
paired `Option`やfallback不能状態は作らない。

5 fresh processのcodec device-complete中央値は26.473 msから25.763 msへ0.710 ms（2.68%）短縮した。
readback-completeは27.127 msから26.461 msへ0.666 ms（2.45%）短縮した。profile上のpack 2.418 msを
全回収できないのは、copy dispatchを消しても同じ総要素数の非連続scatter storeがpair finalizer側へ移る
ためである。codec-only hash `eef3a021…`、full pipeline latent hash `aaa97505…`、waveform hash
`e53ee7bf…`は変更前と一致した。F32 codecもSNR 113.197 dB、uncaptured error 0で回帰を通した。
採用後の全pipeline NVML peakは3,093 MiBで、変更前3,091 MiBとの差2 MiBは125 ms sampling粒度の
変動範囲であり、activated NCLと同要素数のcompact tensorを置換する設計どおりVRAM回帰は観測しなかった。
誤って同時起動した4 processの競合logは`codec-f16-direct-residue-pair-session{2..5}.log`として失敗条件を
保存したが、性能集計にはpoolしていない。採用集計は`experiment.log`と`sequential-session{2..5}.log`だけを使う。

さらにdilation=1ではcompact `[residue][channel][q]` layoutが元のcontiguous NCLと同一なので、bufferを
コピーせず1次元viewとしてCin16 residue coreへ渡すrouteを追加した。F16かつC96/C192/C384の測定shape
だけを許可し、F32は従来T256 routeを維持する。5 fresh processのdevice-completeは25.122、24.889、
25.169、25.123、25.066 ms（median 25.122 ms）、readbackは26.018、25.915、26.451、26.244、
25.950 ms（median 26.018 ms）だった。direct store導入前26.473 msからの累計短縮は1.351 ms
（5.10%）である。codec-only hash/SNRは不変、codec単体NVML peakも1,210 MiBで不変だった。
Cin8も5 fresh processで測ったが、device中央値25.151 msでCin16の25.122 msを下回らず、readback差も
0.014 msだけだったため不採用とした。ログ`codec-f16-residue-d1-cin8-{experiment,session2..5}.log`は
成功値へpoolせず保存した。
同期profileのshape別Cin4/Cin8 winnerを混在させるhybridも5 processで再測定したが、device/readback
中央値25.180 / 26.149 msで全Cin4の25.122 / 26.018 msを下回らず不採用とした。

dilation=1は全許可lengthが4の倍数で、各laneの`q`も4要素境界に揃う。専用F16 shaderで4回のscalar
storeを1回の`vec4<f16>` storeへ置換した。5 fresh processのdevice中央値は24.836 ms、readbackは
25.951 msで、scalar storeより0.286 / 0.067 ms短い。codec hash/SNRは全sessionで同一だった。
ExclusivePagesでも最終binaryを5 fresh processで取り直し、RF 34.915 / 35.359 ms、codec
25.283 / 25.911 ms、独立stage合60.198 / 61.270 msと確定した。NVML peakは2,367 MiBで、
SubSlicesの3,093 MiBから726 MiB低い。速度はSubSlicesより6.9%程度遅くなるが、PyTorchへの優位は維持した。

## fresh campaignとpin

- optimization output:
  `/home/sanzentyo/benchmark-artifacts/irodori-v4-f16-vs-pytorch-opt-20260813-attempt1`
- optimization campaign開始HEAD: `11e1336a4213e6242e236efefa2707c072e2edab`
- adopted zero-copy implementation commit: `c95d59b6e4d07b8e5cfa63ccc455439628d55d6c`
- adopted vec4-store implementation commit: `5789dfa2e54d4e86fd54f3ec281921ce676d5e17`
- measured hybrid validator SHA-256:
  `55020f60fe3a70fe54a3d1af54f996ae14dec5c1766fe7a6e95009f2879965e3`
- final v5 validator SHA-256（commit前tree）:
  `200a657fc4c60ec431852ed2e7979e3d4051b0a69c230e8c780a614f51abb8f2`
- final d1-zero-copy validator SHA-256:
  `d464271a4cefb683ee49600b1a906e15b3ee652a9bbbb8409d36a7fb2138870a`
- final d1-zero-copy codec profiler SHA-256:
  `8a222cda6e5d0b8588a9174bfa7223e2d981ccc25c3aeeef09a1869486fa2051`
- final d1-vec4 validator SHA-256:
  `2fde308eb401fe4ebfb48f931571525a1a31110b672d185eb7f68cf099f6845a`
- final d1-vec4 codec profiler SHA-256:
  `09ef5ea0cf8f877f384fedadebaf2d06c79615dee9487f59e35e7c6aaf148e64`
- output: `/home/sanzentyo/benchmark-artifacts/irodori-v4-f16-20260813-attempt1`
- campaign開始HEAD: `cffa878485ac0adc85ab2837c99b4a55b18d46b4`
- measured implementation commit: `41dfca86521111067016887aa649ec703f4bd996`
- branch: `codex/v4-wgsl-fusion`
- model revision: `e4aaac4df355ff560dcd35e0dae272c3a759317b`
- model SHA-256: `5863c986345d9f6d20b7d8748fee1af02079c5161cf0c9e52557da0a0c378593`
- codec revision: `47376ee24834d7a05a48ebabfe3cde29b3c5e214`
- codec input SHA-256: `db120339c5ee7eca1912cdf29bc612b947a0808e69c3cebfb4936b45a762c1d5`
- fresh converted codec SHA-256:
  `b14a25da5d68bf779d8c44a40706ddc7c4819cb60489b2a80a6408ca03c514fb`
- PyTorch F16 oracle SHA-256:
  `08663e52df6c63eea7ebf5ae2ba35778bdb5bc7d20cd06e430e855a86174229e`
- PyTorch F32 oracle SHA-256:
  `5ea1fcddac1160780dfb53377ecf8fed935fc6f0bab2e2e55464a06868637094`
- Burn `0.22.0-pre.2`, CubeCL `0.11.0-pre.2`, rustc `1.95.0`
- measured validator binary SHA-256:
  `9f25b3c162df36ea749578f09b74876ec1293b43c2dd3506f9ebfccc9c660e2b`
- final committed-source validator binary SHA-256:
  `6eccd109c3e8d74dcd63d4e0ab76346babc134a1f0120d25e0a63e415b4467aa`

旧`/tmp` artifactや旧campaignの数値はpoolしていない。失敗条件も同じfresh output内へ別logとして
保存した。converted codecは全255 tensorが旧artifactとbitwise同一で、file SHA差はsafetensors
metadataのkey順序だけだった。converterの再実行環境へ`PYTHONHASHSEED=0`を追加した。

## 実測環境

- GPU: NVIDIA GeForce RTX 5070 Ti Laptop GPU
- driver: 595.71.05
- Vulkan adapter: index 0、DiscreteGpu
- CUDA/NVML index: 0
- PCI bus ID: `00000000:01:00.0`
- VRAM: total 12,227 MiB、campaign開始時free 11,774 MiB
- OS: Ubuntu 26.04、kernel 7.0.0-27-generic

PyTorchの最初のF16実行は`CUBLAS_STATUS_INVALID_VALUE`でfail-closedした。原因はshellの
`LD_LIBRARY_PATH=/usr/local/cuda/lib64...`がPyTorch cu128同梱libraryを上書きしたことだった。
`env -u LD_LIBRARY_PATH`でF16/BF16 GEMMを再検証してからoracleを作り直した。失敗runの結果は
oracleや性能値へ流用していない。

## 参考: SPIR-V導入前campaignの同一条件

- strict F16またはstrict F32を明示選択
- PyTorchはTF32 off、autocast off
- Euler 4 evaluations、forward batches `[2,2,1,1]`
- effective rows 6、12 layers、48 block calls
- 同一source F32 noiseをtarget dtypeへ一回だけcast
- device-completeはpre-start syncからstage device completionまで
- readback-completeはowned contiguous CPU F32取得まで
- すべて同じ2.0秒fixture、text-only CFG topology

PyTorchとWGPUはsame semantic workだがsame operator graphではない。この節のPyTorch timingは旧oracle
export中の単発観測であり、Rust旧3-repeat device-completeとの厳密な性能比較値には使用しない。現在の
比較結論には、冒頭のfresh optimization campaignで別途取得した6-repeat Python baselineだけを使う。

## 参考: SPIR-V導入前のWGSL-only結果

| runtime | precision | condition | RF device-complete | codec device-complete | NVML peak |
|---|---:|---|---:|---:|---:|
| Rust WGSL | F16 | first process request | 1,221.382 ms | 710.045 ms | 3,802 MiB |
| Rust WGSL | F16 | repeat 2 | 97.247 ms | 29.703 ms | 同一session |
| Rust WGSL | F16 | repeat 3 | **90.591 ms** | **29.419 ms** | 同一session |
| Rust WGSL | F32 | first process request | 968.383 ms | 525.363 ms | 7,964 MiB |
| Rust WGSL | F32 | repeat 3 | 92.734 ms | 33.795 ms | 同一session |
| Rust Burn graph | F16 | earlier same-campaign repeat | 約95.4 ms | 約660 ms | 診断run |

first時間はCubeCL environmentをfresh directoryにしたが、vendor driver cacheのhost状態まではresetして
いない。したがってattempt間のfirst差を特定実装の短縮として因果解釈しない。process内steadyと
同一runのNVML peakだけを採用する。

精度・音質は次の通り。

| 比較 | latent SNR | waveform SNR | cosine | STOI |
|---|---:|---:|---:|---:|
| Rust WGSL F16 vs PyTorch F16 | 45.033 dB | 31.550 dB | 0.999650145 | 0.999860711 |
| codec-only Rust F16 vs PyTorch F16 | — | 56.453 dB | 0.999998875 | — |
| PyTorch F16 vs PyTorch F32 | 48.654 dB | 36.566 dB | 0.999889824 | 0.999947360 |
| Rust WGSL F32 vs PyTorch F32 | 102.617 dB | 90.828 dB | 0.999999999589 | — |

85 dBはF32 numerical reproducibility targetとして維持できる。一方、F16-vs-F32の音声品質に同じ
85 dBを課すとPyTorch自身のF16（36.566 dB）も失格になる。F16は「PyTorch F16とのsame-precision
parity」と「F32に対する知覚品質」を分離し、SNR、max abs、cosine、STOI、NaN/Inf、複数fixtureを
複合gateにする。

## 実装

- WGPU precisionを`WgpuFloatPrecision::{Fp32,Fp16}`で閉じ、device default、checkpoint cast、
  reportを同じ値から導出した。
- F32/F16のCubeCL environment名を分離し、異なるdtypeのautotune結果を共有しない。
- 実棚卸し41 execution shader + 3 preparation shader = 44本すべてへ、既存F32 sourceを変更せず
  `*_f16.wgsl`を追加した。
- 35 launcherでprecisionをKernelIdへ含め、2-byte scalar、8-byte vec4、mixed dtype rejectionを
  実装した。
- F16 shaderはstorage/outputだけF16とし、conv/GEMM/reduction/RMSNorm/softmax/workgroup accumulatorは
  F32にした。
- RoPE、timestep embedding、RMSNormはreference同様F32で計算してからactivation dtypeへ戻す。
- QKVはF16 activationとF32 RoPE tableの混在時、homogeneous-storage shaderへ入れずportable
  segment fallbackへ送る。
- fixed Euler timestep condition cacheはdtype付きにし、F16でも4/4 lookup hitをmanifestで確認した。
- codecのprepared weightとroute contractはF32/F16同一dtypeを許可し、mixed dtypeはfail-closedにした。

最初のF16 WGSL runはQKV mixed binding panic、次はvalidatorのF32専用fixed-cache期待で停止した。
どちらもraw log/NVMLを保存し、成功条件へretry値を混ぜていない。

## cross-platform性とcache

shader source、dtype selector、named CubeCL environment、long-lived sessionはVulkan/Metal/DX12の
WGPU経路で共通化できる。`cubecl/wgpu-spirv`はBurnの別`vulkan` dispatch backendではなく、WGPU
AutoCompilerがVulkan上で利用できるcompilerを追加する。Vulkan以外では対応compilerへ戻るため
crate/APIは共通だが、CMMMA、SPIR-V store、速度、first requestはplatformごとに再測定する。ただしF16
shaderはadapterのshader-f16 capabilityが必要で、未対応GPUやbrowser WebGPUでは起動時に明示拒否または
F32 policyへ明示選択し直す。暗黙fallbackで精度policyを変えない。

cacheのapplication directoryは`Irodori-TTS-burn`で、OS user cache root配下に置く。compiler policy変更を
旧WGSL-only cacheと混ぜないようprofileをv5へ上げ、F16は
`irodori-v4-burn-0.22.0-pre.2-cubecl-0.11.0-pre.2-wgpu-auto-fp16-kernel-v5`、F32は
`...wgpu-auto-fp32-kernel-v5`という別environmentにする。Linux既定は
`${XDG_CACHE_HOME:-$HOME/.cache}/Irodori-TTS-burn/cubecl`で、Windows/macOSもOS user cache conventionを
使う。CLIの`--cubecl-cache-dir`または`IRODORI_TTS_BURN_CACHE_DIR`で上書きできる。

CubeCL bundle/autotune metadataとVulkan SPIR-V storeはprocess間利用できるが、WGPU pipeline objectと
手書きWGSLのdriver compileはprocess-localなので、service startupのDryRun/real validationとlong-lived
sessionは引き続き必要である。今回のartifact内DBは計測時binaryの旧v4ファイル名を保持するが、内部
namespaceが`wgpu<spirv>`であることをreceipt JSONで記録した。ファイルをv5結果と偽ってrenameせず、
最終binaryでは新environmentをfreshに作る。

## crate ergonomics

precisionはboolやpaired `Option`ではなく`WgpuFloatPrecision`で表し、device configuration、weight
cast、CubeCL environment、validator reportを同じ値から導出する。既存の`strict_fp32_device`と
`load_model`はF32 production互換のまま残し、F16は`wgpu_device_with_precision`と
`load_model_with_float_dtype`を明示的に呼ぶため、checkpoint dtypeだけで暗黙にpolicyが変わらない。

不足点は、通常の高水準session builderがまだprecision/profileを型状態として保持しないことと、
shader-f16 capability errorを起動前のtyped receiptとして返していないことである。次cycleでは
`RuntimeBuilder<Cold>`が`PrecisionProfile`とadapter capabilityを検証し、`Runtime<Warmed>`へ渡す形に
まとめる。GUI/server側も文字列precisionではなくこのreceiptを受け取る。

## 次の優先順位

1. codec 24.836 ms対PyTorch 13.391 msの残差を、convtranspose、residual k7 core、pointwise、dispatch別にprofileする。
2. vec4化後も残るd3/d9 compact scatterとresidual coreを、長さ別・channel別に追加profileする。
3. 45/112/255/333/489/685 frames、B1/B2、text/design/cloneでF16 accuracy campaignを行う。
4. v5 environmentでfresh-autotune、restored-autotune、process-warmを分離し、配布bundleも検証する。
5. all-resident sessionとphase batchの両方でpersistent/request peakを取り直す。
6. 固定長・同じCFG topologyをtensor micro-batch候補として記録し、CMMMA routeのbatch効率を測る。
7. Metal/DX12でcompiler選択、F16 capability、accuracy、warmupを独立campaignとして測る。

F16はこの時点ではexperimental opt-inであり、production default F32、F32 shader、非WGSL oracle経路は
削除しない。

## 再開手順

1. branch `codex/v4-wgsl-fusion`をcheckoutし、HEADと本reportのimplementation commitを確認する。
2. campaignの`SHA256SUMS`、`summary.json`、`environment/`、成功・失敗log、NVML CSVを検証する。
3. model/codec revisionとSHAを再検証し、旧`/tmp`や別campaignのcacheを指定しない。
4. F16専用の新しいCubeCL v5 cache rootを用意し、次の形でreport-only replayを実行する。

```bash
target/release/validate_v4_precision \
  --execution wgsl --precision fp16 \
  --fixture /path/to/oracle-fp16.safetensors \
  --fixture-sha256 08663e52df6c63eea7ebf5ae2ba35778bdb5bc7d20cd06e430e855a86174229e \
  --checkpoint /path/to/Irodori-TTS-v4-Small/model.safetensors \
  --codec-weights /path/to/converted-codec.safetensors \
  --codec-weights-sha256 b14a25da5d68bf779d8c44a40706ddc7c4819cb60489b2a80a6408ca03c514fb \
  --cubecl-cache-dir /new/campaign/cache-fp16 \
  --tasks-max 32 --memory-config sub-slices --repeats 10
```

5. 低VRAM profileは同じcommandを`--memory-config exclusive-pages`へ替え、別sessionとして比較する。
6. repeat 1をprocess-local compileとして分離し、最低5 fresh processのsession medianを集約する。
7. まず六つの長さと三つのvoice条件のaccuracyを通し、その結果を報告してから次のkernel変更へ進む。
