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
  は`Auto`を使う。優先順位は`exact実測profile > NVIDIA RTX既定 / Apple M5既定 > portable`である。
- NVIDIA/Apple既定はfamily priorでありexact approvalではない。startup receiptは根拠campaignとexact
  miss理由を保持し、実測profileに見せかけない。
- M5で勝った拡張範囲はApple Metalのfamily priorとして使う。ただしこれはexact承認ではない。
  別driver/binaryへ継承されるのは明示的に表示されるpriorだけで、exact manifestは完全一致時しか使わない。
- `cfg(target_os = "macos")`やadapter名のsubstringではrouteを選ばない。family priorはvendor IDとbackend、
  exact profileはdevice ID、driver、binary/model SHAを含む完全identityで選ぶ。
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
| Apple M1--M4、将来Apple GPU | 未計測 | 既定ではM5 family priorを使うがexact承認とは表示しない。UMA、subgroup、Metal compiler、matmul実装の世代差でcrossoverが動くため個別profileを推奨 |
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
`admits_full_b3()`のような共有booleanから複数成分を派生させず、次の9成分を独立に選ぶ。

| operation | candidate |
|---|---|
| QKV projection | default graph / handwritten t64 |
| QKV/K/V materialization | reference graph / direct packed K/V |
| SDPA | Burn fallback / native WGSL（現在のv4固定shapeではS=13/25/50） |
| post-SDPA | reference layout+gate / fused layout+gate |
| attention output projection | default graph / handwritten t64 |
| SwiGLU expand+activation | default graph / handwritten t64 / CubeK compressed interleaved（S>=100） |
| MLP contract | default graph / handwritten t64 |
| attention output weight | source-column flat / packed-row flat / packed-row rank-3 |
| MLP contract weight | source-column flat / packed-row flat / packed-row rank-3 |

`default graph`は「Burnである」と過剰に約束する名前ではない。SwiGLUはprojection出力を作らない
CubeK compressed writerを別variantにしたため、default/t64/CubeKを同時に表せないinvalid stateはない。
codec selectorはRF tableとは問題keyが異なるため、既存のcodec selector authorityを維持する。

`RuntimeBuilder`では次が通常経路になる。

```rust,ignore
RuntimeBuilder::new(model, codec)
    .routes(RuntimeRoutePolicy::Auto)
    .initialize()?;
```

`Auto`はstartupでmanifest setを一度だけ解決する。exact hitなら承認済みtable、missならvendor IDと
backendからNVIDIA RTXまたはApple M5のfamily priorを選び、その他はportable tableを`OnceLock`へsealする。
adapter名のsubstring判定は行わない。request hot pathは`(batch, sequence)`から固定長配列を直接indexするだけで、adapter
名、vendor、環境変数、JSON、HashMapを参照しない。別tableへのprocess途中の差し替えは拒否する。
portableまたはapproved tableの選択を`allow_b3_packed`のようなprofile flagで上書きすることも禁止した。
なお、exact tableからsource-free residencyを自動導出する処理はまだ接続していない。通常の
`ProductionPrepared`は候補に必要なlayoutを保持し、明示的な`ProfileLocked`だけが既存layout証明で解放する。

`pipeline`も既定が`--route-selection auto`である。明示的な動作は次である。

```text
--route-selection auto                 exact hit、次にGPU family既定、最後にportable
--route-selection portable             常にportable
--route-selection nvidia-rtx            RTX 5070 Ti campaign由来のNVIDIA既定
--route-selection apple-m5              M5 campaign由来のApple Metal既定
--route-selection legacy-production    旧static policyの再現専用
--route-manifest <manifest-set.json>    autoが読むimmutable setを明示
```

manifest setが壊れている場合はstartup error、正常なsetに当該deviceがない場合はfamily既定へ進む。
`RouteInstallDecision::BuiltInDefault`はprofile、根拠campaign、exact miss理由を記録する。壊れたcacheを
「未計測」と同一扱いにして隠さない。

## cacheの場所と作成

既定のroute cacheはCubeCL environmentの中には置かず、同じapplication cache下の兄弟directoryへ置く。

```text
Linux:   $XDG_CACHE_HOME/Irodori-TTS-burn/routes/v4-approved-routes-v2.json
         または ~/.cache/Irodori-TTS-burn/routes/v4-approved-routes-v2.json
macOS:   ~/Library/Caches/Irodori-TTS-burn/routes/v4-approved-routes-v2.json
Windows: %LOCALAPPDATA%\Irodori-TTS-burn\routes\v4-approved-routes-v2.json
```

CubeCLのautotune cacheは引き続き`.../Irodori-TTS-burn/cubecl/`であり、IrodoriのE2E accuracy承認済み
route setとはauthorityを分ける。route setはserviceが上書きする可変cacheではなく、campaignで新規fileへ
sealして配置するimmutable artifactである。

手動evidenceをsealする従来手順に加え、concrete fresh-process runnerを実装した。通常は次の一コマンドを
使う。既存output directoryは拒否し、`--install`なしではapplication cacheを書き換えない。

```bash
cargo build --release --all-features \
  --bin bench_v4_residency --bin tune_v4_routes --bin pipeline

target/release/tune_v4_routes \
  --workload route-workload.json \
  --benchmark-binary target/release/bench_v4_residency \
  --checkpoint "$MODEL" --codec-weights "$CODEC" \
  --production-binary target/release/pipeline \
  --output-directory "$FRESH_OUT" \
  --adapter-index 0 --base-profile auto \
  --fresh-sessions 5 --warmups 2 --measured 10 \
  --install
```

`route-workload.json`のcaseはexact `(B,S)`、独立operation、voice topology、input fixture、2つのprepared
reference、canonical 40-step patched latentとwaveformのf32le oracleを持つ。B1は全voiceの後半、B2は
text-only前半、B3はdesigned/prepared-clone前半で実行されるため、runnerは指定voiceが対象batchを実際に
通ることを検査する。oracleは同じmodel SHA、noise、40-step schedule、CFG意味論で作ったものだけを使う。

```json
{
  "schema_version": 1,
  "cases": [{
    "problem": {"batch_class": "guided_triple", "sequence": 489},
    "operations": [
      "attention_qkv_projection", "attention_materialization", "sdpa", "post_sdpa",
      "attention_output_projection", "mlp_expand", "mlp_contract",
      "attention_output_weight", "mlp_contract_weight"
    ],
    "fixture": "/absolute/fixtures/design-489.safetensors",
    "references": ["/absolute/ref1.safetensors", "/absolute/ref2.safetensors"],
    "voice": "designed",
    "oracle_patched_f32le": "/absolute/oracle/rf_final_patched.f32le",
    "oracle_waveform_f32le": "/absolute/oracle/request-01.f32le"
  }]
}
```

runnerはcandidateごとにunsealed typed profileを作り、WGPU初期化前にfresh child processへ固定する。
各candidateは別accuracy runと5 fresh performance sessionを持つ。performance runは2 warmup + 10 measured、
pre-syncからRF device completionまでのsession medianを使う。全candidate選択後、selection vector全体を
portableではなく測定時のbase profileへ重ね、別の40-step E2E runでlatent/waveform hard gateを再検証する。
このcomposed receiptが失敗したmanifestは保存・installしない。

利用可能なcandidateは全て、測定JSONまたは`--rejection`の明示的なfail-closed JSONを必要とする。
OOM、compile/launch失敗、non-finite、timeout、timestamp欠落は他candidateをretryで選び直す理由ではなく、
そのcandidate固有の保存済みrejectionとなる。portable candidateの正常な40-step測定は常に必須である。
各evidenceはexact identityのSHA-256、41個のschedule bits、40 forwardのbatch topology、12 layers、
480 block callsを持ち、identityやRF意味論が異なるsessionの混入をrejectする。

この変更は既存M5/RTX値を新campaignへpoolしていない。現時点のsetに未計測AMD、Intel、旧Apple、別
NVIDIA世代のexact entryはない。NVIDIA/Appleは明示されたfamily priorから、その他はportableから
個別campaignでexact profileへ昇格する。

crate利用者向けには`RouteCandidateRunner` trait、`autotune_routes_on_base()`、concrete
`FreshProcessRouteTuner`も公開した。workload内の全exact
operation/candidateを列挙し、runnerが返す`Measured`または`Rejected` ADTからmanifestを生成する。
これによりGPU実行harnessはfresh child process、組込みstartup calibration、CI workerのいずれでも同じ
選択ロジックを使える。runnerは`bench_v4_residency`の40-step work report、strict FP32、TF32/autocast off、
12 layers/480 calls、warmup除外timingを再検査し、raw JSON/stdout/stderr/accuracy tensor/SHA256SUMSを残す。
child failureはOOM/compile/launch rejectionとしてcandidate固有に保存され、別条件へのretryにはならない。

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
4. pre-start syncからdevice completionまでを5 fresh process、各2 warmup + 10 measuredで測り、
   session medianを比較する。同一process ABBAはclock driftをさらに減らす後続候補である。
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

現実装はcandidate indexではなくstable enum IDを保存し、raw campaignには全session median、reject理由、
local latent/final waveform metricsを残す。sealed manifestには選択medianとaccuracy dispositionを残す。
MAD、required-layout union、selection-vector digestはprofile-lock接続時に追加する。schema不一致、unknown
candidate、破損、途中writeはcache missにし、installはfile lockとatomic replaceを使う。

## test matrix（実装済みと後続を含む）

- route境界: B1--B4、S=12/13/45/99/100/512/513/685/686。
- serde golden、identity各field変更でcache miss、旧schema、破損、truncate、concurrent writer。
- unapproved stateからlockできないこと、layout proof不足、source解放後fallbackへ到達しないこと。
- candidate launch error、OOM、NaN、timestamp欠落時のfail-closed動作。
- M5/RTXのhardware differential fixture、全voice topologyの40-step gate。
- production hot pathの環境変数readと動的route判定が0件であること。

## 残る実装順序

1. `RouteChoice::required_layouts()`とexact coverage receiptから`WeightResidencyPlan`を生成し、approval後だけ
   sourceを解放する。現在もCubeK interleavedが選ばれたtableだけは自動prepareするが、未covered cellの
   fallback layout unionと最小VRAM化は未完である。
2. SDPA keyを現在のv4固定contextから一般化する場合、Sq/Skv/head/mask/layout/strideをADTへ追加する。
3. codec selectorはRFとは別のproblem keyで同じfresh-process/composed-validation規約へ接続する。
4. exact profile用canonical oracleを公式Python実行から直接exportするRust/Python campaign wrapperを追加する。
   現在のtunerはoracle pathを明示必須とし、異なる意味論の自動生成をしない。
5. M5、RTX、AMD/Intel、旧Apple/NVIDIA世代別にfresh campaignを実行し、v2 setを増やす。

exact manifestはproduction binary SHA-256にも固定される。crateを埋め込んだ別applicationやbinary rebuildは
cache missになり、family priorへ戻る。将来binary全体のhashをroute-relevant source/kernel digestへ狭める場合も、
ABI、dependency、compiler policyを含む同等以上に厳しいidentityを維持する必要がある。

## 関連資料

- [Apple M5 v4 F32 report](m5-v4-f32-2026-08-22.md)
- [RTX 5070 Ti priority 1--4 follow-up](rtx-5070ti-v4-priority-1-4-followup-2026-08-22.md)
- [CubeK compressed SwiGLU and profile lock](rtx-5070ti-v4-cubek-compressed-profile-lock-2026-08-22.md)
