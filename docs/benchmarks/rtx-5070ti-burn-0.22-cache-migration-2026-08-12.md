# RTX 5070 Ti Laptop / Burn 0.22 cache migration (2026-08-12)

## 結論

Burn `0.22.0-pre.2`、burn-cubecl `0.22.0-pre.2`、CubeCL
`0.11.0-pre.2`へexact pinで移行し、production backendはWGPUだけに固定した。
`irodori_tts_burn` crate名も維持している。通常の推論演算を変更せず、CubeCL named
environment、bundle import/export、startup-only `DryRun`、real validationを
`OnlineSession<Unwarmed> -> OnlineSession<SessionReady>`のready条件として実装した。

固定112-frame all-resident profileでは、5 fresh processのbundle-restored DryRun medianは
`0.480 s`で、空cacheの`1.733 s`から72.3%短縮した。steady throughput medianは
`4.433 requests/s`で0.21の`4.515 requests/s`から-1.84%、persistent liveは
`3,711,226,752 bytes`（3,539.29 MiB）で実質同値だった。NVML peakは`5,697 MiB`で
0.21の`5,501 MiB`より196 MiB増えた。allocator live/reservedではなく、WGPU 30の
process-local pipeline/driver allocation側の増分であり、速度を保つため現cycleでは残す。

45/112/255/333/489/685 framesのstrict-FP32 accuracyは、空のfinal-v4 environmentへfixture順に
蓄積し、各2 repeat、計12/12でhard PASSした。途中の失敗条件によるwinner選別やretryはない。完了後に
同environmentをsix-shape bundleとしてsealした。45 framesは
waveform 84.02 dBなので85 dB target warningである。489 framesはaccuracy PASSを確認しており、
性能だけを合格扱いしていない。

一方、portable source weightを保持した全6長all-resident DryRunは12 GiBで
NVML `11,448–11,481 MiB`に達してOOMした。RF/codec compileを二相化して一時bufferを
重ねないrunnerでも失敗したため、全長universal all-residentを「12GBなら入る」とは扱わない。
固定profileのonline sessionと、全長用phase-batch/cache-buildを分ける。

## pinとfresh campaign

- branch: `codex/v4-wgsl-fusion`
- migration開始時HEAD: `fd45b73e30bc2cd11d8177bee842498e88df30fe`
- artifact root:
  `/home/sanzentyo/benchmark-artifacts/irodori-v4-burn022-cache-20260812-attempt1`
- CubeCL environment:
  `irodori-v4-burn-0.22.0-pre.2-cubecl-0.11.0-pre.2-wgsl-fp32-kernel-v4`
- Irodori-TTS-v4-Small revision:
  `e4aaac4df355ff560dcd35e0dae272c3a759317b`
- model SHA-256:
  `5863c986345d9f6d20b7d8748fee1af02079c5161cf0c9e52557da0a0c378593`
- Semantic-DACVAE revision:
  `47376ee24834d7a05a48ebabfe3cde29b3c5e214`
- source `.pth` SHA-256:
  `db120339c5ee7eca1912cdf29bc612b947a0808e69c3cebfb4936b45a762c1d5`
- `scripts/convert_dacvae_weights.py`によるfresh converted SHA-256:
  `4af95181ddf010091b3aca92a17f9580062494ea425cee47063a9a917395f6f1`
- final six-shape bundle SHA-256:
  `071968ca561956cc2c8c87ea51e4bf962706f849e28eb70935b6505c661adf14`
- fixed112 bundle SHA-256:
  `b3da6fb1553672cb1863002bb0e40c4fbe06ac7c730bc21313aafc07a7b3ceac`

過去の`/tmp` artifactと0.21 cacheは存在しないものとして扱った。0.21のprecision
oracle safetensorsはimmutableな比較入力としてSHAを再検証してcurrent campaignへコピーしたが、
旧測定値、旧winner、旧sampleはpoolしていない。campaign開始時に`cargo clean`後、release binaryを
buildし直した。最終runner binary SHAは
`1388eab70e9970b76c5643fd075a3810800632e73d5f898103880cf77648b02d`、validatorは
`b2d87adc67733410dff7f99ea0eca8c858fb9507104175115fe6f2eef0627fa4`である。

## GPU / runtime identity

- GPU: NVIDIA GeForce RTX 5070 Ti Laptop GPU
- NVIDIA driver: `595.71.05`
- total VRAM: `12,227 MiB`
- capture前available VRAM: `11,774 MiB`
- NVML/CUDA index: `0`
- PCI bus ID: `00000000:01:00.0`
- WGPU adapter index: `0`
- WGPU backend: Vulkan
- WGPU vendor/device: `4318 / 12056`

GPU名だけでcacheを共有しない。environment/bundleの運用keyには最低でもBurn/CubeCL version、
compiler path、precision、kernel-profile version、adapter/device identity、driver、model revisionを
含める。今回のCLIはcache rootとenvironment名をreceiptへ保存し、bundle input/outputの既存pathを
暗黙に上書きしない。

## WGPU-only feature closure

`burn`は`default-features = false`で`std`、`wgpu`、`autotune`、`template`、`extension`だけ、
`cubecl`は`std`、`stdlib`、`template`、`wgpu`だけ、`burn-cubecl`は`std`だけを有効にした。
CPU、CUDA、ROCm、NdArray、LibTorch backend featureをproduction dispatchへ入れていない。
`burn-flex` crateがBurn内部の非optional依存としてlink closureに現れても、`flex` featureと
Flex dispatch variantは無効である。PyTorchは同一意味論比較用repository外harnessだけに残す。

Burn 0.22ではuser-facing tensor/moduleからbackend genericを除き、strict FP32/F32+I32を
`Device`初期化時に設定・検査する。`CubeBackend<WgpuRuntime<AutoCompiler>>`だけをproduction
raw launcherに用いる。custom Fusion providerは今回のbaseline後であり、まだproduction graphを
変更していない。

## cache実装とwarmup境界

`src/backend_config.rs`はWGPU初期化より前にnamed environmentを設定し、任意のpersistent rootへ
autotune/throughput namespaceを保存する。bundle importはruntime初期化前、exportはrun完了後に行い、
path、SHA-256、imported/skipped namespace数をJSONへ保存する。fixed112 final bundleのrestoreは
attention、matmul、reduce、throughputの4 namespace、31 entryを各fresh processへimportした。移行中の
v3 six-shape bundleでは6 namespace、172 entryのimportも検証したが、final-v4結果へpoolしていない。

`src/online_session.rs`は次の契約を持つ。

1. `WarmupManifest`とruntime tensor inputを`WarmupPlan::prepare`で一対一に検証する。
2. data-dependent mask/conditioningのcompactionは`DryRun`前に完了させる。
3. RF shapeとcodec shapeを`DryRun`でcompile/autotuneする。
4. duration predictionとtopologyごとの少数real requestを実行し、finiteなfinal audioをreadbackする。
5. 全検査後だけ`OnlineSession<SessionReady>`を返す。
6. ready sessionはmanifest外のframe/topologyをsampling前に拒否する。

`DryRun`はprocess-wide状態へ作用するのでreadiness=falseのstartup中だけ使う。buffer内容は更新されない
ため、duration scalar、tail、final waveform accuracy/hashはreal validationへ残す。全shapeを一つの
all-resident processでwarmするのではなく、12GBでOOMするprofileはRF/codecをphase分離したcache-build
processで同じenvironmentへ追加し、bundleをsealする。

### process間で何をcacheできるか

| 層 | process間 | cross-platform性 | 採用判断 |
|---|---|---|---|
| CubeCL named environment / bundle | autotune winner、throughput等 | native Vulkan/Metal/DX12はfilesystemで共通。browserは別storageが必要 | 採用 |
| process-local CubeCL pipeline map | 不可 | WGPU backend共通 | long-lived sessionで保持 |
| startup `DryRun` | processごとに再生成 | Vulkan/Metal/DX12/WebGPUで原理上共通 | 採用 |
| vendor driver cache | driver依存 | vendor/OS固有 | 補助。正しさの前提にしない |
| wgpu `PipelineCache` | 可能 | Vulkan限定 | 今回は不採用。別campaign |
| SPIR-V compiled store | 条件付き | WGSL/Metal/DX12と同一ではない | compiler変更を別campaign化 |

WGSL経路の`ComputePipeline` objectはbundleへ入らない。従ってportableな構成は
`bundle restore + process-start DryRun + long-lived SessionReady`であり、bundleだけでfirst requestを
完全なsteady latencyにはできない。

### crate APIとproduction CLIのcache位置

crate利用者は`backend_config::{configure_cubecl_persistent_cache,
import_cubecl_environment_bundle, export_cubecl_environment_bundle}`と
`OnlineSession<Unwarmed>::warm`を直接利用できる。production `pipeline` CLIは
`--cubecl-cache-dir`を省略した場合、`IRODORI_TTS_BURN_CACHE_DIR`、次にOS標準user cacheの順で
解決する。application directory名はPython版と混同しないよう全OSで`Irodori-TTS-burn`とする。

| OS | default root |
|---|---|
| Linux / BSD | `$XDG_CACHE_HOME/Irodori-TTS-burn/cubecl`、未設定時`$HOME/.cache/Irodori-TTS-burn/cubecl` |
| macOS | `$HOME/Library/Caches/Irodori-TTS-burn/cubecl` |
| Windows | `%LOCALAPPDATA%\\Irodori-TTS-burn\\cubecl` |

benchmark/accuracy CLIはfresh campaignのcacheを暗黙に共有しないため、引き続きdirectoryを明示する。
異なるGPU/driver/backendのbundleを同じapproval条件として扱わず、必要なら
`--cubecl-cache-dir`でhardware fingerprintを含むsubdirectoryへ分離する。

## fixed112 warmup / latency / VRAM

条件はstrict FP32、TF32 off、autocast off、4 Euler evaluations、forward batches
`[2,2,1,1]`、effective rows 6、12 layers、48 block calls、text-only/unconditioned、
2 warmup + 10 measured、final owned CPU audioまでである。fresh 1 processと、bundleをそれぞれ空の
cache rootへimportした5 processを測定した。runner schema 2では、warmupを分子へ含めていた旧集計を
修正し、requests/sは10 measuredのconsumer-complete合計だけから算出する。

| session | DryRun | load wall | steady consumer median | requests/s | persistent live | NVML peak |
|---|---:|---:|---:|---:|---:|---:|
| fresh | 1.733 s | 9.787 s | 225.95 ms | 4.406 | 3,711,226,752 B | 5,830 MiB |
| restored 1 | 0.480 s | 7.767 s | 213.41 ms | 4.456 | 3,711,226,752 B | 5,697 MiB |
| restored 2 | 0.477 s | 7.825 s | 213.80 ms | 4.459 | 3,711,226,752 B | 5,697 MiB |
| restored 3 | 0.476 s | 7.702 s | 220.55 ms | 4.372 | 3,711,226,752 B | 5,697 MiB |
| restored 4 | 0.480 s | 8.162 s | 214.60 ms | 4.433 | 3,711,226,752 B | 5,697 MiB |
| restored 5 | 0.484 s | 7.714 s | 222.96 ms | 4.326 | 3,711,226,752 B | 5,697 MiB |
| restored median | 0.480 s | 7.767 s | 214.60 ms | 4.433 | 3,711,226,752 B | 5,697 MiB |

GPU graphics clockはactive sample平均約1.07–1.15 GHz（fresh 1.13 GHz）で揺れ、session latencyの
分散が大きい。throughput medianは旧4.515から-1.84%で2% regression gate内、persistent liveは
旧3,539.06 MiBに対して3,539.29 MiBで同値である。request後allocator reservedは約4.18 GiBで旧
4,514.19 MiBより小さい。NVMLだけ196 MiB増え、tensor accounting外のpipeline/driver allocationを
含む。pipelineを破棄すればwarmup効果を失うため、今回の「速度維持」と両立する回収対象にはしない。

steadyは4.48秒/112-frame音声相当である。5-process medianのRF device-completeは135.49 ms、codecは
73.65 msである。same semantic workだが0.21/PyTorchとsame operator graphではない。

PyTorchの同じtext-only 4.48秒条件はsteady `345.1 ms`、`2.898 requests/s`、peak allocated
`4,022.9 MiB`だった。WGPU all-residentはconsumer-complete `214.60 ms`、`4.433 requests/s`、
allocator live `3,539.29 MiB`、NVML peak `5,697 MiB`である。Rust側のNVML値にはtensor以外の
process-local pipeline、driver、allocator reservationが入り、PyTorchの`peak allocated`とは同じ
memory境界ではない。速度はWGPUが1.61倍だが、VRAM比較はWGPU live対Python allocated、または両者の
NVML同士でなければならず、`4 GB対5.7 GB`だけからweight量の差とは結論しない。

## strict-FP32 accuracy approval

最初のgateはlatent max abs `2e-4`、mean abs `1e-5`、RMSE `2e-5`、SNR 90 dB、cosine
`0.99999999`だった。333/685 framesはmax absだけがそれぞれ`2.221e-4`、`2.199e-4`でfailし、
失敗logを保存した。333のportable Burn WGPU oracleも`2.258e-4`であり、custom WGSL破損ではなく
0.22の正当な加算順で一点の外れ値が境界を越えたものだった。mean/RMSE/SNR/cosineとwaveformは通る。

winnerをretry選別せず同じselection vectorのまま、latent max absだけを`2.5e-4`へ改訂した。
他のlatent gateとwaveform hard gate（max abs `1.5e-4`、mean abs `5e-6`、RMSE `1e-5`、
SNR 80 dB、cosine `0.99999999`）は据え置いた。85 dBはnumerical target/warningである。

| frames | latent max abs / RMSE / SNR | waveform max abs / RMSE / SNR | 判定 |
|---:|---:|---:|---:|
| 45 | `4.98e-5` / `5.08e-6` / 104.08 dB | `1.12e-4` / `7.77e-6` / 84.02 dB | hard PASS、target warning |
| 112 | `4.33e-5` / `4.31e-6` / 103.91 dB | `3.44e-5` / `2.19e-6` / 92.30 dB | hard/target PASS |
| 255 | `5.36e-5` / `6.50e-6` / 99.79 dB | `5.77e-5` / `2.45e-6` / 91.59 dB | hard/target PASS |
| 333 | `2.22e-4` / `1.12e-5` / 95.51 dB | `8.31e-5` / `3.41e-6` / 88.25 dB | hard/target PASS |
| 489 | `6.82e-5` / `4.92e-6` / 102.56 dB | `7.99e-5` / `2.22e-6` / 91.61 dB | hard/target PASS |
| 685 | `2.20e-4` / `8.64e-6` / 97.80 dB | `5.16e-5` / `2.41e-6` / 90.47 dB | hard/target PASS |

各長2 repeatはlatent/audio SHA-256が長さ内で一致した。異なる話者や正当なreduction tree間のhash一致は
要求しない。ModernBERTではBurn 0.22 tuned attentionがbroadcast-strided maskで誤計算したため、
maskを`[B,1,Q,K]`へmaterializeしてから反転する。これによりcondition SNR 122.74 dBを回復し、
portable fallbackより高速なtuned attentionを維持した。

## all-resident可否

- fixed112 packed-only: 成功。persistent 3.54 GiB、NVML peak 5.70 GiB。
- 6長portable/universal: 失敗。NVML 11.45 GiB付近でdriver OOM。
- OOM条件はretryで消さず、runnerの誤ったmixed-manifest assertion失敗、cleanup前OOM、
  RF/codec二相化後OOMを別artifactとして残した。
- 6 shape cache自体はshape別processで同じnamed environmentへ蓄積し、172 entry bundleとしてseal/restoreできた。

12GBのproduction policyは、低latency固定profileではall-resident、任意長・大batchでは
`PhaseBatch<RfResident -> LatentsResident -> CodecResident -> Complete>`を使う。RF latentはcodecまで
GPU resident、final audio以外は原則readbackしない。

## crate ergonomics

改善済み:

- public model/tensor型からbackend genericを除去し、WGPU-only `Device` policyへ集約した。
- `OnlineSession<Unwarmed> -> OnlineSession<SessionReady>`、`WarmupManifest`、`WarmupPlan`を追加した。
- manifest外shape/topologyはready sessionでもsampling前に拒否する。
- duration-required/exact-only、topology、real-validation presenceをADTで表現した。
- `PreparedSamplingRequest`がdata-dependent compactionをDryRunより前へ固定する。
- decode-only/fixed112 codec、profile-locked RF、phase-batch type-stateは維持した。

未完:

- `RuntimeBuilder<Cold> -> Runtime<Loaded> -> Runtime<Warmed>`はまだ一つのpublic builderに統合されていない。
- `PreparedSpeaker`、`Voice::{Unconditioned, Clone, Designed}`、`Duration::{Predict, Exact, Frames}`、
  `RequestId`等のnewtypeがrequest boundaryにない。
- `SamplingRequest`にはref/captionのpaired `Option`が残る。
- full-service cache builderはRF/codec phaseをCLI coordinatorとして明示化する必要がある。
- custom Fusion providerとbackend bridge隔離はbaseline後の次cycleである。

## 次回の最適化優先順位

1. 今回のv4 environment identity、six-shape bundle、12/12 accuracy結果をapproval manifestへ固定する。
2. fixed112 `OnlineSession<SessionReady>`をpublic pipelineへ接続し、manifest外requestをCLIでも拒否する。
3. 全長cache buildをRF/codec別phaseで実行するfail-closed coordinatorを追加し、12GB OOMを避ける。
4. primitive/handle変換を`backend_bridge`へ隔離し、SwiGLU postprocessでcustom Fusionを最小実証する。
5. 489/685 accuracyを必須gateにしたlong-sequence providerを評価する。
6. NVMLの追加196 MiBをpipeline数/driver allocationへ分解する。速度を落とすcache破棄は採用しない。
7. Vulkan限定wgpu `PipelineCache`、SPIR-V、Metal/DX12はそれぞれ独立campaignにする。
8. final scalar finite checkのGPU reduction、tail full-readback除去、same-length/CFG tensor micro-batchを続ける。

### model / codec load短縮候補

final checkpointはRF modelが`3,064,295,596 bytes`、converted codecが`429,440,040 bytes`である。
現runnerのrestored profileはmodel、codec、profile preparation、DryRunを含むload wallがmedian
`7.767 s`で、内訳を完全には分離していない。validatorではRF model load/buildが`7.09–8.93 s`、
codec loadがpage-cache状態で`0.15–0.29 s`だった。

優先順位は次の通り。

1. 通常RF loadで`TensorStore::load`が3.06 GB全体をread/copyしてmetadataを取得した後、
   `SafetensorsStore`が同じfileをmmapして再度loadする二重経路を廃止する。metadata headerだけを読み、
   tensorはBurn Storeのmmap経路だけにする。演算・weight値を変えないため最も低riskである。
2. decode-only codec用checkpointを生成するか、mmap/filter loaderでencoder tensorをCPUへcopyしない。
   現状の`TensorStore::load`はdecode-onlyでも429 MB全体をmaterializeする。
3. RF parse、GPU upload、prepared-weight生成、codec parse/upload、fixed-profile packing、DryRunを個別計時し、
   page-cache cold/warmを分ける。現在の合算値だけでparallel loadを採用しない。
4. fixed profileのprepared layoutをversioned checkpointへ保存する案を評価する。startup packingは減るが、
   kernel/profile/device identity、source SHA、accuracy approvalをmanifestに含め、source weight解放後の
   fallback 0件を必須にする。
5. RF/codecの並列loadはCPU parsingが支配的と確認できた場合だけ試す。WGPU queueへの同時uploadで
   peak VRAM、driver allocation、load determinismを悪化させないことをgateにする。

load最適化はsteady演算を変えずに実施できるが、cold filesystem cache、warm filesystem cache、
bundle restoreの3条件を別campaignとして再計測する。

BF16は実行していない。reject済みkernel/scriptの整理はこのbaseline報告後のcycleへ残す。

## 最終QA

- `cargo test --lib --all-features`: 511 passed、0 failed、16 ignored
- `cargo clippy --all-targets --all-features -- -D warnings`: PASS
- `cargo fmt --all -- --check`: PASS
- `uvx ruff check scripts`: PASS
- `uvx ruff format --check scripts`: 13 files formatted
- 旧crate名`irodori_tts_wgpu` / `irodori-tts-wgpu`: source tree残存0件
- Rust source文字列を検索する`include_str!("*.rs")` assertionと、weight shapeをruntime stateの
  sentinelにするfixed112 assertion: 残存0件

## 再開手順

1. `git switch codex/v4-wgsl-fusion`し、このreportを追加したcommit以降であることを確認する。
2. artifact rootの`SHA256SUMS`とnested `build/`、`models/`、`inputs/` manifestを検証する。
3. GPU、driver、WGPU adapter、CUDA/NVML index、PCI bus ID、free VRAMを再取得する。差があれば新campaignにする。
4. model/codec revisionとSHA、strict F32/I32 device policy、Burn/CubeCL exact versionを照合する。
5. 旧0.21 cacheや別GPU bundleをコピーしない。v4 bundle import receiptのenvironment名とentry数を確認する。
6. 45/112/255/333/489/685をhard gateで再実行する。45の85 dB未達はwarning、hard failureはretryしない。
7. fixed112は2 warmup + 10 measured x 5 fresh processでlatency/VRAMを再確認する。
8. universal all-residentを12GBで再試行しない。全長はphase-separated cache build/phase batchから再開する。
9. 上記が通った後だけcustom Fusionまたは長尺最適化へ進む。
