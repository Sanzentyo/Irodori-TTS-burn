# B3 buffer / SwiGLU epilogue / manifest residency follow-up (2026-08-22)

## 結論

今回の4項目のうち、productionへ新たに採用したのはB3 JointAttentionのdirect packed K/Vと
post-SDPA layout+gateである。489-frame、40-step Voice Designの同一binary paired screenでは
15 measured requestをまとめたconsumer-complete中央値が5.41472 sから5.40300 sへ11.71 ms
（0.216%）短縮し、3 fresh pairすべてのsession medianも短縮した。出力f32 hashは旧経路と
bit exact、persistent allocator usageも同一だった。

既存のB1/B2 projection+SwiGLU一dispatch経路も、独立toggleで再確認した。489-frame B1では
13.729 MiBの展開temporaryと1 dispatchを除去し、NVML peakは3 fresh sessionすべてで
12--15 MiB低かった。一方、40-step Voice Design全体のconsumer中央値は+13.79 ms（+0.256%）で
明確な速度改善ではない。これは今回新たに導入した経路ではなく既存production経路の再評価であり、
メモリ削減を保ったまま維持する。B3へ同じ手書きGEMMを広げる案は前campaignで遅くaccuracyも悪化した
ため再採用していない。

長尺temporaryのlive-rangeはprofile feature内で、各attention/MLP substageの同期前後にCubeCL
allocatorのin-use/reserved/allocation countを機械可読形式で出すようにした。manifest-derived
weight residencyは`load_for`前に導出・検証し、選択profileだけでなく実際にresidentとなる論理layout
集合もreceiptへ残す。

## Pinsと環境

- branch: `codex/v4-post-seal-priority-1-4`
- B3 attention campaign source: `198039e1e58102c92dccd0acd974d35e7280e23b`
- live-range campaign source: `d6bd59233835820bf605440210a721b2406aac50`
- SwiGLU screen source: `390f1f0f07f87d383dc3775ade52c1ef7e67f9b0`
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
- production backend: WGPUのみ
- precision: strict FP32、TF32 off、autocast off
- sampler: 40-step Euler（live-range診断だけ4-step、性能値にはpoolしない）
- model revision: `e4aaac4df355ff560dcd35e0dae272c3a759317b`
- model SHA-256: `5863c986345d9f6d20b7d8748fee1af02079c5161cf0c9e52557da0a0c378593`
- codec revision: `47376ee24834d7a05a48ebabfe3cde29b3c5e214`
- converted decoder-only codec SHA-256:
  `1b1ceb3f620525cf4252af508c0fde80e3779582d47fc7fc879410d2e4abe231`
- f489 design fixture SHA-256:
  `9a1e00e667f960983b62ebc9188c6b430acf0c00d0721ef9ffdf8fc8b9fd4b3f`

過去campaignの数値は新しいA/Bへpoolしていない。各screenはfresh output、fresh process、独立CubeCL
cache directory、1 warmup + 5 measured、3 fresh session、variant順序AB/BA交互、automatic retry 0である。

## B3 QKVからSDPAまでの中間buffer削減

旧B3経路は次の順だった。

```text
QKV+gate projection
  -> Q/Kself/Vself materialization
  -> Kself+Kctx cat / Vself+Vctx cat
  -> SDPA
  -> head-major layout conversion
  -> gate multiply
  -> wo
```

採用経路はQ/K norm、half-RoPE、gate sigmoidを一度行い、Qと最終形の
`[self | context]` K/Vへ直接書く。SDPA後もlayout conversionとgate multiplyを一dispatchにした。
shaderはbatchをtemplate値として扱っていたため、B1/B2だけだったhost-side dtype/shape/stride/device/
hardware-limit contractとcontext K/V packをB3まで一般化した。B4は許可していない。

f489/B3/FP32の1 block・1 forward当たり、direct K/Vはself K/Vのwrite/readを30,044,160 bytes
（28.652 MiB）、post-SDPAは15,022,080 bytes（14.326 MiB）削減する。合計logical traffic削減は
45,066,240 bytes（42.979 MiB）である。これはallocator in-useの単純差ではなく、消えたread/writeの
厳密なshape計算である。

| session | disabled consumer median | enabled consumer median | paired差 | hash |
|---|---:|---:|---:|---|
| 1 | 5.40939 s | 5.36520 s | -44.19 ms | exact |
| 2 | 5.41472 s | 5.40508 s | -9.63 ms | exact |
| 3 | 5.43435 s | 5.41350 s | -20.86 ms | exact |
| all 15 measured | 5.41472 s | 5.40300 s | -11.71 ms | exact |

RF device-completeのall-request中央値は5.00477 sから4.98689 sへ17.88 ms短縮した。
load後persistentは両variantとも3,959,266,048 bytes in-use（3,775.850 MiB）、
3,961,530,560 bytes reserved（3,778.010 MiB）。NVML peakはdisabled 6,664 MiB、enabled
6,664--6,666 MiBで差を主張できない。一方、consumer後allocator reservedは全pairでenabledが
196,056,256 bytes（186.974 MiB）小さく、同じtemporary shapeがpoolへ残る量は減った。

## 長尺temporary live-range

`IRODORI_RF_DETAIL_PROFILE=1`では、既存のdevice-complete同期境界に次を追加した。

```text
before/after bytes_in_use
before/after bytes_reserved
before/after number_allocs
各delta
```

production featureでは環境参照もmemory queryもコンパイルされない。診断campaignはf489 design、
4-step、1 warmup + 1 measuredであり、stage同期が通常実行を変えるため性能比較へpoolしない。

| B3 stage | enabled in-use delta | disabled in-use delta | 解釈 |
|---|---:|---:|---|
| QKV+gate projection | +30,044,160 | +30,044,160 | combined projectionの生存開始 |
| materialize Q/K/V | +23,208,960 | +23,208,960 | stage出口では最終Q/K/V shapeが同じ |
| SDPA | -8,186,880 | -8,186,880 | SDPA workspace解放を含む |
| layout+gate | **-22,533,120** | 0 | combinedの寿命を`wo`前に終了 |
| output projection入口の差 | **-22,533,120** | baseline | 21.489 MiBのlive-range短縮 |

direct K/Vが削ったself K/V+catの一時peakは同一stage内で発生・解放されるため、stage出口snapshotだけ
では差にならない。ここは前節のexact logical trafficと40-step A/Bのreserved receiptで補完する。
次にallocator arenaを作るなら、同時生存が確認できたbufferだけを対象にし、stage境界で既に解放される
bufferを常駐workspaceへ移してはいけない。

MLPではB3の非融合経路がexpandで+43,188,480 bytes、SwiGLUでさらに+21,594,240 bytesを持ち、
一時的に61.782 MiB増える。B1の既存融合経路は`expand_swiglu`一stageで出力7,198,080 bytesだけを
生成し、14,396,160-byteのfull expansionを作らない。

## Projection + SwiGLU epilogue

現行B1/B2は`w1 || w3` projectionのF32 accumulatorからSwiGLUを同じWGSL dispatchで評価し、
`[rows, hidden]`だけをstoreする。独立toggleでは無効側をBurn/CubeK matmul + standalone SwiGLUへ戻した。

| session | disabled consumer median | enabled consumer median | paired差 |
|---|---:|---:|---:|
| 1 | 5.38669 s | 5.37567 s | -11.02 ms |
| 2 | 5.39256 s | 5.40804 s | +15.49 ms |
| 3 | 5.40350 s | 5.41676 s | +13.26 ms |
| all 15 measured | 5.39298 s | 5.40677 s | +13.79 ms (+0.256%) |

NVML peak中央値はdisabled 6,679 MiB、enabled 6,664 MiBで15 MiB低く、persistentは同一だった。
hashはenabled `d2bc...246d`、disabled `fc09...923c`で異なる。これはmatmul reduction orderの違う
same semantic workであり、このscreen単独をaccuracy比較には使わない。enabledは今回導入した新しい
数値経路ではなく既存production経路である。

B3で真にBurn tuned matmulを保ったまま同じ削減を得るには、通常のscalar epilogueでは足りない。
`[gate | value]`の離れた2 accumulator列を1出力へ縮約するpaired-column writer、またはweightを
interleaveするsingle-storage canonicalizationとcompressed output contractがCubeK側に必要である。
既存weightを複製すると12-layerで約431 MiB増えるため採用しない。これは残課題であり、現在の遅い
B3手書きprojectionを有効化したとは報告しない。

## Manifest-derived weight residency plan

public APIに次を追加した。

```text
WeightResidencyPolicy::{Explicit, FromWarmupManifest}
WeightResidencyPlan
WeightResidencyBasis::{Explicit, StrictManifest, CompileOnDemandFallback}
WeightLayout

RuntimeBuilder<Cold>
  -> derive_weight_profile_from_manifest()
  -> load_for(WarmupSelection)
  -> RuntimeBuilder<Loaded>::warm_planned(inputs)
```

manifestはweight解放前にschema、空、zero frame、duplicate、topologyごとのreal validation、duration
validationを再検証する。deserializationやpublic field構築でconstructorを迂回してもfail closedである。
`warm_planned`または`warm_with_plan`へ異なるmanifestを渡すことも拒否する。

`StrictWarmup`ではexact-112、long text-only、long text/design/prepared-cloneをそれぞれ最小の既存profile
へ写像する。combined design+cloneはB4なので`LongAllVoicePreparedOnly`へ入れず
`ProductionPrepared`へ戻す。`CompileOnDemand`もmanifest外requestを許すため常に
`ProductionPrepared`を選ぶ。receiptはprofile名だけでなく、QKV source/row/column、Q/K norm pack、
SwiGLU source/fused、`wo`/`w2` source/packedのresident集合を列挙する。

これらはRust ADTとtype-state transitionで表し、paired `Option`やshape sentinelは追加していない。

## 移植性

- B3 materializationとSwiGLU shaderはWGSL/WGPU経路で、Vulkan専用APIを直接使用しない。source設計は
  Metal/DX12でも共有可能だが、今回実測したのはNVIDIA/Vulkanだけである。
- CubeCL allocator live-range queryはWGPU client APIなのでbackend横断で利用できる。NVML peak採取だけは
  NVIDIA固有であり、他vendorでは別monitorが必要である。
- manifest-derived planはdevice vendor非依存のRust APIである。実際のlayoutを解放するprofileは
  WGPU production graphのcontractなので、別backendを暗黙に許可しない。
- profile-only environment togglesはproduction buildでは定数化され、request hot pathに文字列参照を
  持ち込まない。

## Fresh artifacts

| campaign | status | source | `SHA256SUMS` SHA-256 |
|---|---|---|---|
| `irodori-v4-b3-attention-materialization-20260822-attempt1` | COMPLETE | `198039e` | `47cea0c23f408421770a14bebc29ecaefcb482c991ade0e2df2717de1bd270ac` |
| `irodori-v4-long-live-range-20260822-attempt1` | COMPLETE / diagnostic | `d6bd592` | `d9f588e6b91da20bd0d177ee6cf4e11a4b1b27b22cfcc3de903e82f07a16ecad` |
| `irodori-v4-projection-swiglu-epilogue-20260822-attempt1` | COMPLETE | `390f1f0` | `be2d1b144e8fbf82bfed84fe0079b74ed62e45f2872e5bc72a19230d782979c6` |

各directoryはraw result JSON、stdout/stderr、NVML、GPU inventory、binary、input/model/cache bundle pin、
audio f32 artifact、`SHA256SUMS`を保持する。検証はdirectory内で`sha256sum -c SHA256SUMS`を実行する。

## QA

- `cargo test --all-features --no-fail-fast`: library 561 passed / 20 ignored / 0 failed、
  全binary testとdoc-testも成功
- `cargo clippy --all-targets --all-features -- -D warnings`: PASS
- `cargo fmt --all -- --check`: PASS
- `uvx ruff check scripts`: PASS
- `git diff --check`: PASS

全体testで、genericな小型SwiGLUのsource fallbackまでv4固定の`[3680, 1280]` shape assertionで
拒否していた既存問題も検出した。このassertionはprepared-only状態の証明になっておらず、通常のshape
整合検査と重複していたため削除した。修正後は該当focused test 2件と全体testの両方が成功している。

## 残る優先順位

1. ~~CubeKにpaired-column/compressed-output epilogueを一般化し、weight duplicateなしでB3
   projection+SwiGLUを実装する。~~ 完了。実装・40-step・VRAM・SDPA internal peakは
   [`rtx-5070ti-v4-cubek-compressed-profile-lock-2026-08-22.md`](rtx-5070ti-v4-cubek-compressed-profile-lock-2026-08-22.md)
   に記録した。
2. ~~live-range pointをSDPA内部workspaceとB3 materialization kernel内部へ細分化し、stage内peakをGPU
   timestamp/allocator eventで捕捉する。~~ stage-scoped allocator high-water計測まで完了。個々の
   workspaceへのsemantic tag付与は上記follow-up reportの残課題へ移した。
3. ~~`WeightResidencyPlan`のlayout集合からprofile enumを経由せず直接`PreparedModel<ProfileLocked>`を
   構築する。~~ `WeightLayoutSet`のvalidationと
   `PreparedModel<LayoutsSelected> -> PreparedModel<ProfileLocked>`で完了。
4. Metal/DX12でcompile smokeと短尺accuracyを取得し、「共有可能なsource」を「検証済みbackend」へ
   段階的に昇格する。
