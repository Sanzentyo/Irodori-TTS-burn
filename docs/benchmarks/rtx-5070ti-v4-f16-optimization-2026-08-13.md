# RTX 5070 Ti Laptop 12 GiB: v4 F16 WGPU optimization (2026-08-13)

## 結論

F16はproduction defaultにはせず、明示選択するexperimental policyとして実装した。50 latent
frames（48 kHz、96,000 samples、約2.0秒）の固定fixtureでは、手書きWGSL経路は3回目の
device-completeでRF 90.591 ms、codec 29.419 ms、process NVML peak 3,802 MiBだった。同じ
sourceのF32回帰はRF 92.734 ms、codec 33.795 ms、peak 7,964 MiBである。F16の主な成果は
RFの速度ではなく、codecの短縮と4,162 MiBのpeak削減である。

PyTorch F16との最終waveform一致はSNR 31.550 dB、cosine 0.999650145、STOI
0.999860711だった。これは音声一致として良好だが、1長・1条件だけなのでproduction採用gateには
足りない。F32 defaultは維持し、六つの長さ、text-only/design/clone、fresh/restored cacheを
通すまでF16を自動選択しない。

## fresh campaignとpin

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

## 同一条件

- strict F16またはstrict F32を明示選択
- PyTorchはTF32 off、autocast off
- Euler 4 evaluations、forward batches `[2,2,1,1]`
- effective rows 6、12 layers、48 block calls
- 同一source F32 noiseをtarget dtypeへ一回だけcast
- device-completeはpre-start syncからstage device completionまで
- readback-completeはowned contiguous CPU F32取得まで
- すべて同じ2.0秒fixture、text-only CFG topology

PyTorchとWGPUはsame semantic workだがsame operator graphではない。PyTorch timingはoracle export中の
単発観測であり、Rustの3-repeat device-completeとの厳密な性能比較値には使用しない。

## 結果

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
WGPU経路で共通化できる。ただしF16 shaderはadapterのshader-f16 capabilityが必要で、未対応GPUや
browser WebGPUでは起動時に明示拒否またはF32 policyへ明示選択し直す。暗黙fallbackで精度policyを
変えない。

cacheのapplication directoryは`Irodori-TTS-burn`で、OS user cache root配下に置く。F16は
`irodori-v4-burn-0.22.0-pre.2-cubecl-0.11.0-pre.2-wgsl-fp16-kernel-v4`、F32は`...fp32...`
という別environmentである。CubeCL bundle/autotune metadataはprocess間利用できるが、WGSL
ComputePipelineはprocess-localなので、service startupのDryRun/real validationとlong-lived sessionは
引き続き必要である。

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

1. 45/112/255/333/489/685 frames、B1/B2、text/design/cloneでF16 accuracy campaignを行う。
2. fresh-autotune、restored-autotune、process-warmを別sessionで各5回取り、first時間を分離する。
3. F16 cross-layer AdaLN cacheをdtype付きで有効化し、RF dispatchとsteadyを再測定する。
4. QKV F16 storage + F32 RoPE bindingを表せるmixed-input shaderを、portable fallbackとA/Bする。
5. F16 fixed-timestep cacheのbatch-shapeによる丸め差をfixture横断で評価する。
6. all-resident sessionとphase batchの両方でpersistent/request peakを取り直す。

F16はこの時点ではexperimental opt-inであり、production default F32、F32 shader、非WGSL oracle経路は
削除しない。

## 再開手順

1. branch `codex/v4-wgsl-fusion`をcheckoutし、HEADと本reportのimplementation commitを確認する。
2. campaignの`SHA256SUMS`、`summary.json`、`environment/`、成功・失敗log、NVML CSVを検証する。
3. model/codec revisionとSHAを再検証し、旧`/tmp`や別campaignのcacheを指定しない。
4. F16専用の新しいCubeCL cache rootを用意し、次の形でreport-only replayを実行する。

```bash
target/release/validate_v4_precision \
  --execution wgsl --precision fp16 \
  --fixture /path/to/oracle-fp16.safetensors \
  --fixture-sha256 08663e52df6c63eea7ebf5ae2ba35778bdb5bc7d20cd06e430e855a86174229e \
  --checkpoint /path/to/Irodori-TTS-v4-Small/model.safetensors \
  --codec-weights /path/to/converted-codec.safetensors \
  --codec-weights-sha256 b14a25da5d68bf779d8c44a40706ddc7c4819cb60489b2a80a6408ca03c514fb \
  --cubecl-cache-dir /new/campaign/cache-fp16 \
  --tasks-max 32 --memory-config sub-slices --repeats 3
```

5. まず六つの長さと三つのvoice条件のaccuracyを通し、その結果を報告してから次のkernel変更へ進む。
