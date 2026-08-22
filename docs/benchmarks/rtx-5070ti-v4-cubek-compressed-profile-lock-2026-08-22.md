# CubeK compressed SwiGLU / SDPA internal peak / layout-set profile lock (2026-08-22)

## 結論

3項目を実装し、B3 SwiGLU経路はproductionの`LongAllVoicePreparedOnly`へ採用した。

- CubeKへ、隣接する2 accumulator列を1 scalarへ縮約してhalf-width outputだけをstoreする
  汎用compressed-output epilogueを追加した。v4のweightは
  `[w1[0], w3[0], w1[1], w3[1], ...]`へload時に一度だけcanonicalizeする。
- 489-frame Voice Design、40-step、3 fresh pair、各1 warmup + 5 measuredでは、全15 requestの
  consumer-complete中央値が5.76042 sから5.62377 sへ136.65 ms（2.37%）短縮した。
  RF device-completeは5.34567 sから5.21632 sへ129.35 ms短縮した。全30 waveform hashは一致した。
- interleaved weightは従来のhalf-separated fused weightを置換し、併存しない。12-layerの
  431.25 MiB cacheは増えていない。`LongAllVoicePreparedOnly`のpersistent in-useも従来値と同じ
  3,959,266,048 bytesである。
- CubeCL WGPU allocatorへstage-scoped high-water probeを追加した。f489 design・4-stepの48 SDPA
  block callでは、全callでstage中のlive in-use peakがstage入口を超えなかった。reserved pageは
  初回に増えるがlive tensor peakとは区別できた。
- `WeightResidencyPlan.profile`を再度matchして全cacheを作る経路を高水準runtimeから外し、検証済み
  `WeightLayoutSet`から`PreparedModel<LayoutsSelected> -> PreparedModel<ProfileLocked>`を直接構築する。
  state payloadをgeneric型に持たせ、paired `Option`は使っていない。

## Pinとcampaign境界

- branch: `codex/v4-post-seal-priority-1-4`
- compressed SwiGLU 40-step source: `6cbf71a`
- SDPA internal-peak source: `58cd3ae2e690eb40561d2d68821d4df506ce3199`
- GPU: NVIDIA GeForce RTX 5070 Ti Laptop GPU
- driver: `595.71.05`
- WGPU adapter: index 0、Vulkan、vendor `0x10de`、device `0x2f18`
- CUDA/NVML index: 0
- PCI bus ID: `00000000:01:00.0`
- physical VRAM: 12,227 MiB
- campaign前available VRAM: 11,774 MiB
- Rust: `1.95.0 (59807616e 2026-04-14)`
- Burn: `=0.22.0-pre.2`
- CubeCL: `=0.11.0-pre.2`
- precision: strict FP32、TF32 off、autocast off
- model revision: `e4aaac4df355ff560dcd35e0dae272c3a759317b`
- model SHA-256: `5863c986345d9f6d20b7d8748fee1af02079c5161cf0c9e52557da0a0c378593`
- codec revision: `47376ee24834d7a05a48ebabfe3cde29b3c5e214`
- decoder-only codec SHA-256:
  `1b1ceb3f620525cf4252af508c0fde80e3779582d47fc7fc879410d2e4abe231`
- f489 design fixture SHA-256:
  `9a1e00e667f960983b62ebc9188c6b430acf0c00d0721ef9ffdf8fc8b9fd4b3f`

過去の`/tmp` artifactや旧計測値を新しい中央値へpoolしていない。40-step campaignはfresh output、
fresh process、variantごとに独立したCubeCL/XDG cache、AB/BA交互、automatic retry 0である。
attempt 1/2の失敗も上書きせず保存した。attempt 1はB3 packed-output routeの不正なassertion、attempt 2は
同じ契約がfallback helperにも残っていたことを検出した。両方ともsource layoutを選んだ時点で
packed-only kernelへ入らないようrouteを先に判定する修正とした。

## 汎用CubeK compressed-output epilogue

### 契約

matmulの論理shapeは`M x N`のまま、RHSの隣接列を一組としてepilogueへ渡し、物理outputは
`M x (N/2)`だけを持つ。

```text
LHS [M,K]
RHS [K,N] = [a0,b0,a1,b1,...]
  -> CubeK accumulator pair (2j, 2j+1)
  -> PairwiseAccumulatorGlobalEpilogue::apply(first, second, coordinate)
  -> physical output [M,N/2]
```

writerはoutput storeより前にrow/column tailをmaskする。端数tileのepilogueが範囲外parameterを
読む状態は作らない。logical matmul strideとcompressed physical strideはlauncherで分離し、通常の
PlaneWriter/UnitWriterは従来どおり自分のvalid sliceだけを受け取る。

strict FP32の現在のWGPU adapterではf32 cooperative-matrix tileを選べないため、portableな
`SimpleUnitPairwiseCompressedAlgorithm`を使用する。accelerated plane writerも同じtraitで実装したが、
今回実GPUで通したのはunit writerである。小型の`M=17, K=128, N=34` partial-tile testはCPU参照と
`2e-5`以内で一致した。

### B3 SwiGLU

v4-SmallのB3/f489では次のgeometryになる。

- `M = 3 * 489 = 1,467`
- `K = 1,280`
- logical `N = 7,360`
- physical `N = 3,680`
- 従来full expansion: 43,188,480 bytes
- 新しいcompressed output: 21,594,240 bytes
- 消えたtemporary: 21,594,240 bytes（20.594 MiB）/ block / forward

weightはrequest時にpackしない。load時に一度だけcolumn-major pair-interleaved storageを生成する。
launcherはdtype、rank、shape、stride、device/clientを検査し、implicit contiguous copyを行わない。
成功したhot pathはprojectionとSwiGLUを合わせて1 dispatchである。

## 40-step性能・精度・VRAM

条件は489-frame Voice Design、strict FP32、40 Euler evaluations、forward batches
`[3,3,1,1]`、effective rows 8、12 layers、480 block callsである。比較は同一binaryの
`ProductionPrepared`と`LongAllVoicePreparedOnly`をfresh processで交互に実行した。

| fresh pair | Production consumer median | compressed LongAll median | paired差 | RF paired差 |
|---:|---:|---:|---:|---:|
| 1 | 5.77652 s | 5.61537 s | -161.15 ms | -156.12 ms |
| 2 | 5.74190 s | 5.62377 s | -118.13 ms | -115.88 ms |
| 3 | 5.75254 s | 5.63990 s | -112.63 ms | -115.35 ms |
| all 15 measured | 5.76042 s | **5.62377 s** | **-136.65 ms** | **-129.35 ms** |

全30 measured waveformはf32 SHA-256
`4df3cc4bd811517ff3ff49fb8406643ab3286925fc3dafc58409de64f3bd930f`で一致した。

| metric | ProductionPrepared | compressed LongAll | delta |
|---|---:|---:|---:|
| persistent in-use | 4,264,006,912 B | **3,959,266,048 B** | -304,740,864 B (-290.624 MiB) |
| persistent reserved | 4,266,205,312 B | **3,961,530,560 B** | -304,674,752 B |
| NVML peak pair 1 | 6,957 MiB | **6,667 MiB** | -290 MiB |
| NVML peak pair 2 | 6,957 MiB | **6,667 MiB** | -290 MiB |
| NVML peak pair 3 | 6,957 MiB | **6,669 MiB** | -288 MiB |

persistent差の大部分はLongAll profileが`wo/w2` sourceを解放する既存効果である。重要なのは、新しい
interleaved cacheを加えてもLongAllの旧persistent 3,959,266,048 bytesから増えていない点である。
half-separated 431.25 MiB cacheを一旦作ってdropすることもなく、profile preparation中央値は
Productionの約162.9 msに対してLongAllは約123.3 msだった。

## SDPA内部のstage peak

従来のbefore/after snapshotでは、SDPA中に生成して同じstage内で解放したallocationを見落とす。
`cubecl-wgpu`のmain-pool reserve境界へprocess-local high-water probeを追加し、attentionのpre-sync後に
windowを開始、post-sync後に終了する。重複windowは`begin()`がfalseを返し、callerがfail closedする。

独立diagnosticはf489 design、4-step、48 block callsで、B3 24回・B1 24回だった。各SDPAで10 reserve
eventを観測したが、live `bytes_in_use` peakは48回すべてstage入口と同じだった。

| batch | calls | stage-entry / internal peak in-use | max reserved growth |
|---:|---:|---:|---:|
| B3 | 24 | 4,335,747,776 B / 同値 | 67,819,968 B |
| B1 | 24 | 4,295,581,376 B / 同値 | 2,616,320 B |

B3の通常stage出口ではin-useが8,186,880 bytes減る。すなわちallocator event上、SDPA workspaceは
入力live-rangeの解放と入れ替わり、stage入口を越える追加live peakを作っていない。初回のreserved
growthはpool pageの保持であり、同時生存tensor量ではない。したがって、これだけを根拠にpersistent
SDPA arenaを追加するとVRAMを増やす可能性が高い。

この診断は各stageにsyncを入れるためlatency比較へ使わない。NVMLはNVIDIA固有だが、このhigh-water
probe自体はCubeCLのWGPU allocator境界にあり、Vulkan/Metal/DX12で共有できる。今回の実測対象は
NVIDIA/Vulkanだけである。

## `PreparedModel<ProfileLocked>`

高水準runtimeの不可逆遷移は次になった。

```text
WeightResidencyPlan
  -> WeightLayoutSet::new(...)       // sort/dedup/coverage validation
  -> PreparedModel<LayoutsSelected>  // loaded logical source modelを所有
  -> lock()
  -> PreparedModel<ProfileLocked>    // selected physical representationだけを所有
  -> WgslInferenceEngine
```

`LayoutsSelected`と`ProfileLocked`がそれぞれ実体を所有するため、raw/lockedのpaired `Option`はない。
QKV row/column、Q/K norm、SwiGLU fused/interleaved、`wo/w2` packedはlayout集合に含まれるものだけを
prepareする。不要なprepared layoutを作ってからprofile enumでdropする経路は、`RuntimeBuilder` /
`OnlineSession`の通常loadから外れた。duration predictorはRF residency setとは別の固定WGSL contract
なので、従来どおり専用cacheをprepareする。

`SwiGluInterleaved`を含む集合は、long B3に必要なQKV row+column、Q/K norm、packed `wo/w2`が全て
含まれなければconstruction前に拒否する。layoutが欠けたprojectionも同様に拒否する。

### crate ergonomics

- 良い点: public receiptと実model transitionが同じ`WeightLayoutSet`を使い、profile名と物理状態の
  二重管理を減らした。stateを消費するためlock後にsource modelへ戻れない。
- 良い点: `WeightLayout`追加で新representationを列挙でき、booleanの組合せをAPIへ増やしていない。
- 残る点: legacyな`build_wgsl_with_profile`は低水準caller互換のため残っている。新規service codeは
  `RuntimeBuilder::load_for`とmanifest-derived planを使うべきである。
- 残る点: checkpoint loaderは一度learned sourceをloadしてからselected representationを作る。
  safetensorsから最終layoutへ直接streamするloaderは別変更であり、今回は数値意味を変えない。

## 移植性

- pairwise epilogueとunit/plane writerはCubeCL sourceでありWGPU runtime共通。NVIDIA固有APIは使わない。
- strict FP32のportable unit pathはVulkan以外でもsource共有可能だが、Metal/DX12でのcompile/accuracy/
  performanceは未検証であり「対応済み」とは表現しない。
- allocator probeもWGPU main pool共通。ただしprocess-globalで単一windowなので、並行service request中
  ではなくstartup/diagnostic sessionだけで使う。
- layout-set/type-stateはpure Rustでdevice vendor非依存。実layoutのkernel coverageはWGPU production
  contractとして検証する。

## Fresh artifacts

| campaign | status | source | `SHA256SUMS` SHA-256 |
|---|---|---|---|
| `irodori-v4-cubek-compressed-swiglu-f489-design-20260822-attempt1` | FAILURE / assertion | `af3eb6e` | 保存済み |
| `irodori-v4-cubek-compressed-swiglu-f489-design-20260822-attempt2` | FAILURE / fallback assertion | `365f830` | 保存済み |
| `irodori-v4-cubek-compressed-swiglu-f489-design-20260822-attempt3` | COMPLETE | `6cbf71a` | `05ab23e1129b866be115b5f472c34feaa7ad62a7ddafc53a828515fa7365ab0f` |
| `irodori-v4-sdpa-internal-peak-20260822-attempt1` | COMPLETE / superseded diagnostic | `fd52a9f` | 保存済み |
| `irodori-v4-sdpa-internal-peak-20260822-attempt2` | COMPLETE / diagnostic | `58cd3ae` | `eb72b06f11681b26eee964635dd9d862f40e1bf0df82de155259b158ede6c247` |

COMPLETE directoryはbinary/runner/model/codec/input/cache bundle pin、raw JSON/stdout/stderr/NVML、
GPU inventory、`SHA256SUMS`を持つ。各directory内で`sha256sum -c SHA256SUMS`が成功した。

## QA

- `cargo test --all-features --lib`: 564 passed / 20 ignored / 0 failed
- `cargo test --all-features --bins`: 32 passed / 0 failed
- `cargo test --all-features --doc`: 4 ignored / 0 failed
- `cargo clippy --all-targets --all-features -- -D warnings`: PASS
- `cargo fmt --all -- --check`: PASS
- `uvx ruff check scripts`: PASS
- `bash -n scripts/run_v4_sdpa_internal_peak.sh`: PASS
- `git diff --check`: PASS

全機能testを並列実行した際、partial-tile GPU testだけが共有WGPU deviceを再初期化してCubeCL serverと
競合した。演算の失敗ではない。testを既存deviceのF32 policyを検査して共有する形へ変更し、同じ並列
584 test条件を再実行して上記結果を得た。

## 残る本質的な改善

1. strict FP32を維持するaccelerated CubeK tileがWGPU adapterで利用可能になった時に、同じ
   compressed writerをunit pathと比較する。TF32へ下げた結果は同じcampaignへ混ぜない。
2. SDPAの10 reserve eventへsemantic tagを付け、score/mask/softmax/output別のlifetimeを出す。
   現状のstage peakは正しいが、個々のbuffer attributionまでは行わない。
3. layout setをcheckpoint indexへ渡し、不要source tensorをCPU/GPUへmaterializeせず最終layoutへ
   streamする。load短縮とload中peak削減を別々に測る。
4. Metal/DX12でpartial-tile compile smokeと短尺accuracyを実行する。source共有可能性と実検証を
   混同しない。

## 再開手順

```bash
git switch codex/v4-post-seal-priority-1-4
git pull --ff-only
git rev-parse HEAD

(cd /home/sanzentyo/benchmark-artifacts/irodori-v4-cubek-compressed-swiglu-f489-design-20260822-attempt3 \
  && sha256sum -c SHA256SUMS)
(cd /home/sanzentyo/benchmark-artifacts/irodori-v4-sdpa-internal-peak-20260822-attempt2 \
  && sha256sum -c SHA256SUMS)

cargo test --lib --features inference,codec,profile \
  pairwise_writer_matches_cpu_on_partial_tiles -- --nocapture

bash scripts/run_v4_sdpa_internal_peak.sh \
  --output-dir /home/sanzentyo/benchmark-artifacts/NEW-FRESH-SDPA-PEAK \
  --input-campaign /home/sanzentyo/benchmark-artifacts/irodori-v4-accuracy-localization-20260822-attempt3
```
