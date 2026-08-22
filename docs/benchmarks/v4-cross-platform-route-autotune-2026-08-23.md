# v4 cross-platform route autotune design (2026-08-23)

## 結論

Apple M5で実測された広いDiT t64 route envelopeは、カーネルの物理能力として取り込む。一方、
全deviceのproduction既定値にはしない。現行RTX 5070 Tiの40-step campaignでは、B3のQKV、
attention output、MLP contractは489 framesで勝つが685 framesで逆転し、B3 MLP expandは489
framesでも遅かった。M5では短尺とB3でhandwritten routeが大幅に勝つため、勝者はOS名ではなく
device世代、driver、shape、CFG phaseを含むexact workloadごとに決める必要がある。

今回の実装では次の境界を採用した。

- `sequence >= 13`、`batch <= 3`はhandwritten t64 kernelのphysical capabilityとする。
- 旧production policyは再現用`LegacyProduction`として残すが、通常の`RuntimeBuilder`と`pipeline`
  は`Auto`を使う。exact cache hitがなければ全cellをportable routeにする。
- M5で勝った拡張範囲は候補能力として保持するが、M5以外はもちろん、別driver/binaryのM5にも
  自動継承しない。
- `cfg(target_os = "macos")`、adapter名、vendor名だけではrouteを選ばない。
- `allow_b3_packed_wo_wgsl` / `allow_b3_packed_w2_wgsl`によるsource-free layout証明を維持する。
- QKV projection、attention output projection、MLP expand projection、MLP contract、`wo` layout、
  `w2` layoutは別々の`RouteOperation`である。ある成分の承認から別成分を許可しない。
- `SwiGluInterleaved`の存在から他stageのB3 route admissionを推測しない。layoutとrouteの完全な
  proof統合はprofile lock前に行う後続段階である。

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

## 実装済みのbinary側自動選択

`src/route_autotune.rs`にclosed ADT、全候補catalog、accuracy-aware selector、複数device用の
`ApprovedRouteManifestSet`、exact identity照合、直接indexの`ResolvedRouteTable`を実装した。
`admits_full_b3()`のような共有booleanから複数成分を派生させず、次の6成分を独立に選ぶ。

| operation | candidate |
|---|---|
| QKV projection | default graph / handwritten t64 |
| attention output projection | default graph / handwritten t64 |
| MLP expand projection override | default graph / handwritten t64 |
| MLP contract | default graph / handwritten t64 |
| attention output weight | source-column flat / packed-row flat / packed-row rank-3 |
| MLP contract weight | source-column flat / packed-row flat / packed-row rank-3 |

`default graph`は「Burnである」と過剰に約束する名前ではない。例えばMLPには別のCubeK compressed
candidateが先行するprofileがあり得るため、ここで選ぶのはhandwritten t64 overrideの有無である。
SDPA、CubeK compressed SwiGLU、codec selectorを同じtop-level authorityへ加える際も、それぞれ別の
`RouteOperation`にし、既存operationのbooleanへ同居させない。

`RuntimeBuilder`では次が通常経路になる。

```rust,ignore
RuntimeBuilder::new(model, codec)
    .routes(RuntimeRoutePolicy::Auto)
    .initialize()?;
```

`Auto`はstartupでmanifest setを一度だけ解決する。exact hitなら承認済みtable、missならportable tableを
`OnceLock`へsealする。request hot pathは`(batch, sequence)`から固定長配列を直接indexするだけで、adapter
名、vendor、環境変数、JSON、HashMapを参照しない。別tableへのprocess途中の差し替えは拒否する。
portableまたはapproved tableの選択を`allow_b3_packed`のようなprofile flagで上書きすることも禁止した。
source-free residencyが選択tableの全warmup caseをcoverしなければ、model load前にfail closedする。

`pipeline`も既定が`--route-selection auto`である。明示的な動作は次の3つである。

```text
--route-selection auto                 exact hit、なければportable
--route-selection portable             常にportable
--route-selection legacy-production    旧static policyの再現専用
--route-manifest <manifest-set.json>    autoが読むimmutable setを明示
```

manifest setが壊れている場合はstartup error、正常なsetに当該deviceがない場合はcache missとしてportableへ
戻る。壊れたcacheを「未計測」と同一扱いにして隠さない。

## cacheの場所と作成

既定のroute cacheはCubeCL environmentの中には置かず、同じapplication cache下の兄弟directoryへ置く。

```text
Linux:   $XDG_CACHE_HOME/Irodori-TTS-burn/routes/v4-approved-routes.json
         または ~/.cache/Irodori-TTS-burn/routes/v4-approved-routes.json
macOS:   ~/Library/Caches/Irodori-TTS-burn/routes/v4-approved-routes.json
Windows: %LOCALAPPDATA%\Irodori-TTS-burn\routes\v4-approved-routes.json
```

CubeCLのautotune cacheは引き続き`.../Irodori-TTS-burn/cubecl/`であり、IrodoriのE2E accuracy承認済み
route setとはauthorityを分ける。route setはserviceが上書きする可変cacheではなく、campaignで新規fileへ
sealして配置するimmutable artifactである。

作成手順は次である。

```bash
# production binary、adapter、model、codecを含むexact identityを生成
cargo run --release --features inference,codec,cli \
  --bin approve_v4_autotune -- build-route-identity \
  --checkpoint "$MODEL" --codec-weights "$CODEC" \
  --production-binary target/release/pipeline \
  --adapter-index 0 --output-identity identity.json

# 40-step evidenceから各exact problemのwinnerを選ぶ
cargo run --release --features inference,codec,cli \
  --bin approve_v4_autotune -- select-routes \
  --identity identity.json \
  --measurement qkv-default.json --measurement qkv-t64.json \
  --output-manifest device-routes.json

# 複数GPU/driverの独立manifestを1つのimmutable setへまとめる
cargo run --release --features inference,codec,cli \
  --bin approve_v4_autotune -- assemble-route-set \
  --manifest m5-routes.json --manifest rtx5070ti-routes.json \
  --manifest amd-routes.json --output-set v4-approved-routes.json
```

利用可能なcandidateは全て、測定JSONまたは`--rejection`の明示的なfail-closed JSONを必要とする。
OOM、compile/launch失敗、non-finite、timeout、timestamp欠落は他candidateをretryで選び直す理由ではなく、
そのcandidate固有の保存済みrejectionとなる。portable candidateの正常な40-step測定は常に必須である。
各evidenceはexact identityのSHA-256、41個のschedule bits、40 forwardのbatch topology、12 layers、
480 block callsを持ち、identityやRF意味論が異なるsessionの混入をrejectする。

この変更は既存M5/RTX値を新campaignへpoolしていない。現時点のsetに未計測AMD、Intel、旧Apple、別
NVIDIA世代のentryはなく、それらはportableから個別campaignで昇格する。

crate利用者向けには`RouteCandidateRunner` traitと`autotune_routes()`も公開した。workload内の全exact
operation/candidateを列挙し、runnerが返す`Measured`または`Rejected` ADTからmanifestを生成する。
これによりGPU実行harnessはfresh child process、組込みstartup calibration、CI workerのいずれでも同じ
選択ロジックを使える。現在のrepositoryに残る作業は、既存40-step benchmark/validatorをこのtraitへ接続し、
測定JSONの手渡しを不要にするconcrete runnerである。

## profile lockまでの型状態

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

top-level selectorへ今後追加するrouteもclosed ADTで表す。

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
serialize可能な`RouteChoice`とstartup後の`ResolvedRouteTable`は既に分離した。tensor/client handleを
含む`ResolvedRoute`と`required_layouts()` receiptは、sourceを解放するprofile lockへ選択結果を接続する
次段階である。

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

実装済みcache identityは次を完全一致で照合する。

```text
schema / route ABI
backend / compiler / allocator / bounds-check policy
adapter name / backend / vendor ID / device ID / device type / driver / driver info
OS / kernel-platform version / architecture
app / Burn / burn-cubecl / CubeCL / CubeK / wgpu versions
precision / compiler / allocator policy
model / codec / production binary SHA-256
exact RouteProblemKey
  operation, B, sequence
```

現在のv4 Small projectionはM/K/N/head geometryがmodel SHAとoperationで固定されるため、table keyはBと
sequenceまでに縮約している。汎用SDPA等を追加する際はSq/Skv/head/mask/layout/strideをkeyへ追加する。
stable hardware/driver identityを取得できないbrowser/compatibility adapterはpersistent reuseを禁止する。
MetalはdriverがOS統合で空の場合があるため、exact adapter名とkernel/platform versionを併用する。

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

## 残る実装順序

1. source-retained candidate runnerから、現在のtyped evidence形式を直接出力する。
2. operator differentialと40-step approvalを同じcampaign directoryで生成する。
3. SDPA、CubeK compressed SwiGLU、codec selectorを独立`RouteOperation`として追加する。
4. `required_layouts()`とcoverage receiptから`WeightResidencyPlan`を生成し、approval後だけsourceを解放する。
5. M5、RTX、AMD/Intel、Metal/NVIDIA世代別にfresh campaignを実行し、immutable setを増やす。

## 関連資料

- [Apple M5 v4 F32 report](m5-v4-f32-2026-08-22.md)
- [RTX 5070 Ti priority 1--4 follow-up](rtx-5070ti-v4-priority-1-4-followup-2026-08-22.md)
- [CubeK compressed SwiGLU and profile lock](rtx-5070ti-v4-cubek-compressed-profile-lock-2026-08-22.md)
