# v4 cross-platform route autotune design (2026-08-23)

## 結論

Apple M5で実測された広いDiT t64 route envelopeは、カーネルの物理能力として取り込む。一方、
全deviceのproduction既定値にはしない。現行RTX 5070 Tiの40-step campaignでは、B3のQKV、
attention output、MLP contractは489 framesで勝つが685 framesで逆転し、B3 MLP expandは489
framesでも遅かった。M5では短尺とB3でhandwritten routeが大幅に勝つため、勝者はOS名ではなく
device世代、driver、shape、CFG phaseを含むexact workloadごとに決める必要がある。

今回の統合では次の境界を採用した。

- `sequence >= 13`、`batch <= 3`はhandwritten t64 kernelのphysical capabilityとする。
- production既定値はRTXで承認済みの範囲を維持する。B3 projection/contractは512 frames以下、
  B3 MLP expandは無効である。
- M5で勝った拡張範囲は`ExtendedCandidate`としてprofile buildからだけ選択可能にする。
- `cfg(target_os = "macos")`、adapter名、vendor名だけではrouteを選ばない。
- `allow_b3_packed_wo_wgsl` / `allow_b3_packed_w2_wgsl`によるsource-free layout証明を維持する。
- `SwiGluInterleaved`の存在から他stageのB3 route admissionを推測しない。正式autotunerでは各
  routeが必要layoutを個別に宣言する。

## 実測から分かるplatform差

M5とRTXのcampaignはprecision、入力、process protocolが異なるため、中央値をpoolしたり直接の
speed ratioとして扱ったりしない。以下は各platform内のA/Bから得たroute特性である。

| platform | 実測された特性 | productionへの意味 |
|---|---|---|
| Apple M5 / Metal | t64 kernelは約1.9 TFLOPS、Burn eager matmulは約0.3 TFLOPS。40-stepの255-frame design/cloneはroute拡張で約1.51--1.52x、textは1.14x。685 textは1.01x | small-M、B3、短中尺でdispatch/tiling overheadを避けるhandwritten routeが広く勝つ候補。ただしM5以外へ一般化しない |
| RTX 5070 Ti Laptop / Vulkan | B3 QKV、attention output、MLP contractは489で勝つが685で負ける。B3 MLP expandは489でも負ける。CubeK compressed SwiGLUは489 design 40-step E2Eを2.37%短縮 | CubeK matmulが長尺で強く、componentごとのcrossoverが必要。現行のB3上限512を既定値として維持 |
| Apple M1--M4、将来Apple GPU | 未計測 | UMA、subgroup、Metal compiler、matmul実装の世代差でcrossoverが動く。M5結果はcandidate優先順位のpriorにだけ使う |
| NVIDIA Ampere/Ada/他Blackwell | 未計測 | tensor/core構成、shared memory、driver shader compiler、power limitでwinnerが変わる。5070 Tiのselectionをpersistent reuseしない |
| AMD/Intel Vulkan・DX12 | 未計測 | CubeK tile、subgroup幅、register pressureとdriver compilerの差が大きい。portable fallbackから測る |
| browser WebGPU | 未計測 | adapter identityと永続cacheの信頼性が低い場合はprocess-local tuningだけを許す |

M5ではRFのnative MPS matmulが全cellで速い一方、WGPU codecが20--50 msとほぼ長さ非依存で、
PyTorch MPS codecの最大約3.5秒に対してE2Eを逆転する。したがって「WGPUが速い／遅い」ではなく、
RF、SDPA、codecを別々のroute problemとして扱う必要がある。AppleのUMAではpersistent weightの
意味もdedicated VRAMとは異なるため、resident bytes、OS memory pressure、request peakを別々に記録する。

## 今回追加したprofile candidate

`profile` feature付きbuildでは、process内の最初のroute解決前に次を設定するとM5で実測された
拡張envelopeを試せる。

```bash
IRODORI_DIT_ROUTE_ENVELOPE=extended-candidate \
  cargo run --release --features inference,codec,cli,profile --bin pipeline -- ...
```

これは診断用candidateであり、accuracy承認済みcacheでもproduction policyでもない。production
buildは環境変数を読まず、`ProductionApproved`を選ぶ。process途中の変更は`OnceLock`により反映
されない。正式autotuner導入後はこの文字列経路を削除し、startupで構築したtyped route tableへ
置き換える。

## 正式autotunerの型状態

route tuning中はfallback sourceを保持し、全routeの選択と40-step validationが終わるまでweightを
解放しない。

```text
PreparedModel<SourcesResident>
  -> prepare_candidates(RouteWorkloadManifest)
PreparedModel<CandidatesReady>
  -> benchmark_and_filter()
PreparedModel<RoutesMeasured>
  -> validate_40step()
PreparedModel<RoutesApproved>
  -> WeightResidencyPlan::from_approved(RouteCoverageReceipt<AccuracyApproved>)
  -> lock()
PreparedModel<ProfileLocked>
```

`ProfileLocked`からtune APIへ戻る遷移は実装しない。未知shapeをcompile-on-demandで許すprofileは
portable fallback layoutをresidency unionへ含め、source-free lockを禁止する。

主なrouteはclosed ADTで表す。

```rust,ignore
enum QkvRoute {
    BurnSource,
    BurnPacked,
    T64Packed,
}

enum AttentionOutRoute {
    BurnSourceRank3,
    BurnPackedFlat,
    BurnPackedRank3,
    T64PackedResidual,
}

enum SwiGluRoute {
    BurnHalfSeparated,
    T64HalfSeparated,
    CubeKCompressedInterleaved { algorithm: StableAlgorithmId },
}

enum MlpContractRoute {
    BurnSourceRank3,
    BurnPackedFlat,
    BurnPackedRank3,
    T64PackedResidual,
}

enum SdpaRoute {
    Burn,
    CubeK,
    Native(NativeFaConfig),
}
```

各variantが`required_layouts()`を所有し、`RouteRequirementSet`へunionする。B3 wo admission、B3 w2
admission、interleaved SwiGLUは独立したproofとし、あるlayoutから別stageのrouteを推論しない。
serialize可能な`RouteChoice`とtensor/client handleを持つ`ResolvedRoute`を分離する。request hot pathは
startup時に構築済みの`ResolvedRouteTable`を`RouteClassId`で引くだけにし、環境変数、文字列比較、
hash計算、adapter照会を行わない。

Eulerでは前半のguided B2/B3と後半のunguided B1で問題shapeが変わるため、keyに`CfgPhase`を含める。
SDPAはquery長だけでなくKV/context長、head数、head dim、mask、layout、stride classまで区別する。

## tuning手順

1. device capabilityとworkload manifestを確定し、exact problemを列挙する。
2. sourceを保持したまま全candidateをcompile smokeする。launch error、OOM、NaN/Inf、timeoutはその
   candidateだけをrejectする。
3. canonical portable routeと同一入力でoperator differentialを取り、local accuracyを通過した
   candidateだけを残す。
4. pre-start syncからdevice completionまでを、同一processのpaired round-robin/ABBAで測る。
5. provisional selection vector全体を実fixture・40 Euler stepsでE2E validationする。
6. approved routeの`required_layouts()`のunionとexact manifest coverage receiptを作る。
7. receiptが完全なときだけ不要sourceを解放し、`PreparedModel<ProfileLocked>`へ遷移する。
8. 選択済みgraphだけをDryRun warmupし、少数のreal validation後に`Runtime<Ready>`を公開する。

全candidate失敗時はcanonical portable routeへ戻す。policyは
`TuningFailurePolicy::{FailStartup, KeepPortable}`、結果は
`PreparationOutcome::{Locked, Portable(FailureReceipt)}`のADTで表し、paired `Option`を作らない。
locked後やfirst request中にtuningは行わない。

## accuracy承認

最速candidateをそのまま保存せず、次のhard gateをすべて通る候補の中から選ぶ。

| boundary | hard gate | target / warning |
|---|---|---|
| local latent/operator | max abs `<= 2e-4`、mean abs `<= 1e-5`、RMSE `<= 2e-5`、SNR `>= 90 dB`、cosine `>= 0.99999999` | hard pass後に性能比較 |
| final waveform | max abs `<= 1.5e-4`、mean abs `<= 5e-6`、RMSE `<= 1e-5`、SNR `>= 80 dB`、cosine `>= 0.99999999` | 80--85 dBはnumerical warning、85 dB以上はtarget |

PCM16 hash一致だけでは承認しない。単発forwardのlocal pass後もtrajectory差が累積するため、40-step
validationを必須にする。canonical baseline自体が80 dBを下回るclassでは、新candidateを承認せず
portable incumbentへ戻す。

## sealed cacheのauthority

CubeCL 0.11のpersistent tunerは候補列のdigestとfastest indexを中心に保存し、top-level routeを
安全に再利用するためのdriver、source、E2E accuracy proofを十分には表さない。そのためCubeCL cacheは
candidate内部のmatmul autotuneと計時補助に限定し、Irodori側のsealed manifestをroute authorityにする。

cache keyは少なくとも次を含む。

```text
schema / route ABI
backend / compiler / allocator / bounds-check policy
exact device fingerprint
  Vulkan UUID or PCI identity / DXGI LUID / Metal registry ID
  vendor ID / device ID / device type / driver / driver info
OS / architecture
app / Burn / burn-cubecl / CubeCL / CubeK / wgpu versions
float/int dtype and strict-precision policy
model / config / codec SHA-256
candidate-set / WGSL / vendored-source SHA-256
manifest / fixture digest
exact RouteProblemKey
  op, B, Sq, Skv, M, K, N, heads, head dim,
  topology, CFG phase, mask, layout and stride classes
```

GPUの世代名やadapter表示名は`GenerationHint`としてcandidate順序にだけ使う。cache hitは
`ExactDeviceFingerprint`で判定する。stable identityの一部でも取れないplatformではpersistent
selection reuseを無効化し、そのprocess内だけでtuneする。shape bucketは使わず、rangeを再利用する
場合は全境界shapeを検証した別receiptを必要とする。

cache valueはcandidate indexではなくstable enum ID、全sample、median、MAD、reject理由、local metrics、
40-step latent/waveform metrics、required layouts、selection-vector digestを保存する。schema不一致、
unknown candidate、破損、途中writeはcache missにし、file lockとatomic replaceを使う。

## test matrix

- route境界: B1--B4、S=12/13/45/99/100/512/513/685/686。
- serde golden、identity各field変更でcache miss、旧schema、破損、truncate、concurrent writer。
- unapproved stateからlockできないこと、layout proof不足、source解放後fallbackへ到達しないこと。
- candidate launch error、OOM、NaN、timestamp欠落時のfail-closed動作。
- M5/RTXのhardware differential fixture、全voice topologyの40-step gate。
- production hot pathの環境変数readと動的route判定が0件であること。

## 実装順序

1. 今回のcapability/policy分離とstatic production behaviorを維持する。
2. typed route table、exact device identity、report-only cacheを追加する。
3. source-retained candidate runnerとpaired GPU timingを追加する。
4. operator differentialと40-step approval receiptを追加する。
5. `WeightResidencyPlan`生成をapproval後へ移し、route/layout proofを独立させる。
6. opt-in campaignをM5、RTX、AMD/Intel、Metal世代別に実施してからproduction defaultにする。

## 関連資料

- [Apple M5 v4 F32 report](m5-v4-f32-2026-08-22.md)
- [RTX 5070 Ti priority 1--4 follow-up](rtx-5070ti-v4-priority-1-4-followup-2026-08-22.md)
- [CubeK compressed SwiGLU and profile lock](rtx-5070ti-v4-cubek-compressed-profile-lock-2026-08-22.md)
