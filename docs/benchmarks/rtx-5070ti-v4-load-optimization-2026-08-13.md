# Irodori v4 model / codec load最適化（12GB、2026-08-13）

## 結論

現行演算、同期境界、strict FP32、4 Euler evaluationsを変えず、all-residentのload wallを
control median `7.703 s`から統合条件median `5.195 s`へ`2.508 s`（`32.56%`）短縮した。
採用したのは次の4点である。

1. RF checkpointのmetadataはsafetensors headerだけから読み、tensor本体はBurn Storeの
   file-backed経路で一度だけloadする。
2. converterに`decoder-only` profileを追加し、codec encoderと`quantizer.in_proj`をartifactから除く。
3. codecの`TensorStore`を全file read + tensor別copyから、検証済みoffsetによるsingle-copy streamingへ変える。
4. crate APIからRFとcodecを並列loadできる`OnlineSession<Unwarmed>::load_parallel`を追加する。

全測定でfinal audio f32 SHA256は
`45854abebba0af7f74833a261412a3d303031c747817ac5da2b44e65f8c96821`、
all-resident live bytesは`3,710,995,136` bytes（`3,539.08 MiB`）で一致した。
steady演算を変更していないため、load短縮と引き換えの精度、steady latency、GPU memory増加はない。

## pinと測定環境

- source branch: `codex/v4-wgsl-fusion`
- control source HEAD: `52c27734c42faba8cfbd5c532b60bc666d648625`
- checkpoint optimization commit: `571bcf6`
- parallel session commit: `098845f`
- GPU: NVIDIA GeForce RTX 5070 Ti Laptop GPU、`12,227 MiB`
- driver: `595.71.05`
- CUDA/NVML index: `0`
- PCI bus ID: `00000000:01:00.0`
- Burn / burn-cubecl: `0.22.0-pre.2`
- CubeCL: `0.11.0-pre.2`
- model revision: `e4aaac4df355ff560dcd35e0dae272c3a759317b`
- model SHA256: `5863c986345d9f6d20b7d8748fee1af02079c5161cf0c9e52557da0a0c378593`
- codec revision: `47376ee24834d7a05a48ebabfe3cde29b3c5e214`
- codec input SHA256: `db120339c5ee7eca1912cdf29bc612b947a0808e69c3cebfb4936b45a762c1d5`
- full converted codec SHA256: `4af95181ddf010091b3aca92a17f9580062494ea425cee47063a9a917395f6f1`
- decoder-only codec SHA256: `1b1ceb3f620525cf4252af508c0fde80e3779582d47fc7fc879410d2e4abe231`
- final benchmark binary SHA256: `46d5bbc4aa6d1d834fb1b1e0f02447ddb632e0807f4ed253a4068c921b702ad9`

fresh campaignは
`/home/sanzentyo/benchmark-artifacts/irodori-v4-load-opt-20260813-attempt1`である。
過去の`/tmp` artifact、旧計測値、旧CubeCL environmentはpoolしていない。immutable fixtureとreferenceは
campaignへcopyしてSHAを再取得した。CubeCL bundleもこのcampaign内で新規生成した。

filesystem coldは対象model/codec fileだけに`fadvise --advice dontneed`を実行し、`fincore`を保存した。
warmとは分離して記録した。process初回にDryRunが約15秒になったrunは失敗/first-shape evidenceとして保存し、
load medianにはretry選択で混ぜず除外した。これはRF/codec load stageの外側で発生している。

## stage別結果

| 条件 | n | load wall median | RF checkpoint median | codec checkpoint median | RF profile prep median |
|---|---:|---:|---:|---:|---:|
| control | 8 | 7.703 s | 6.750 s | 0.289 s | 0.242 s |
| RF metadata-only | 7 | 5.421 s | 4.426 s | 0.395 s | 0.142 s |
| decoder-only artifact | 8 | 5.572 s | 4.576 s | 0.303 s | 0.190 s |
| pre-stream sequential cohort | 4 | 5.567 s | 4.636 s | 0.345 s | 0.131 s |
| parallel load | 5 | 5.308 s | 4.697 s | 0.390 s | 0.191 s |
| streaming codec sequential | 4 | 5.189 s | 4.366 s | 0.200 s | 0.163 s |
| integrated parallel + streaming | 5 | 5.195 s | 4.535 s | 0.278 s | 0.191 s |

final committed-source確認runはload wall `4.969 s`、DryRun `0.344 s`、request wall `0.221 s`で、
同じaudio hashとlive bytesを再確認した。

control、metadata-only、decoder-onlyはcold/warmを個別にも保存している。上表の各行は同じ条件内の要約であり、
coldとwarmを相互poolして有意差検定には使っていない。統合条件はpage-cache warm、restored autotune bundle、
fresh processである。

## 採用判断

### RF metadata-only + Burn Store file-backed load

controlではRF loadがload wallの`86–91%`を占めた。従来は3.06 GB checkpoint全体を
`TensorStore`へread/copyしてmetadataを得た後、Burn Storeでも同じcheckpointを開いていた。
header-only metadataへ変えることでRF checkpoint medianは`6.750 s`から`4.426 s`へ
`34.43%`短縮した。model値とrecord mappingは変えていない。

### decoder-only codec artifact

full artifactは`429,440,040` bytes、decoder-onlyは`320,048,128` bytesで`25.47%`小さい。
残すkeyは`decoder.*`と`quantizer.out_proj.*`だけで、Rust decoderが要求する全164 tensorを含む。
artifact metadataの`irodori_codec_profile=decoder-only`を検査し、full codec APIからの誤用は
明示errorにする。従来のfull artifactはmetadata keyなしを`full`として互換維持し、converterの
default full出力SHAも不変である。

生成例:

```bash
uv run scripts/convert_dacvae_weights.py \
  --pth /path/to/weights.pth \
  --profile decoder-only \
  --output /path/to/dacvae-decoder-only.safetensors
```

### codec single-copy streaming

従来のcodec loaderはfile全体の`Vec<u8>`に加え、各tensorのowned bytesを同時に保持した。
新loaderはsafetensors headerを完全にparse/offset検証し、file lengthとの一致を確認してから、
tensor payloadを最終owned bufferへ直接読む。pre-stream同条件に対してcodec medianは
`0.345 s`から`0.200 s`へ`41.99%`短縮した。追加unsafe、OS固有mmap API、backend依存はない。

### parallel all-resident load

pre-stream cohortのsequential median `5.567 s`に対しparallel medianは`5.308 s`で、
`0.259 s`（`4.65%`）短縮した。RFとcodecのhost I/O/GPU uploadを同一Burn Deviceのcloneで重ね、
join後にRF profileを構築する。Rust標準threadとBurn/WGPU DeviceだけなのでVulkan、Metal、DX12で
同じAPIを使用できる。phase batchはresident modelが同時不要なのでsequentialのままにする。

crateからは次の型付きAPIを使う。

```rust,ignore
let (session, load_report) = OnlineSession::<SessionUnwarmed>::load_parallel(
    device,
    model_path,
    decoder_path,
    sampling,
    WgslWeightProfile::PortableFallback,
    DurationModelResidency::Predictive,
)?;
```

戻り値の`SessionLoadReport`はwall、RF checkpoint、codec checkpoint、RF profile preparationを
別々に持つ。`OnlineSession<Unwarmed>`のまま返すため、manifest-driven warmupとreal validationを
完了するまで`SessionReady`のsynthesis APIへ到達できない。

## 不採用案

### prepared-weight checkpoint永続化

今回は実装しない。RF profile preparation全体のmedianは統合条件で`0.191 s`、観測最大でも
`0.312 s`であり、保存による短縮上限が小さい。一方、prepared cacheは約1.4 GiB、複数の
`#[module(skip)]` tensorからなり、shape/profile/device/kernel identityに依存する。永続化には
record設計、source SHA、kernel hash、device identity、accuracy approval、fallback 0件の契約が必要になる。
load wallの主要因を解消した後でもprofile prepが1秒を超える環境が確認された場合にだけ再評価する。

### 独自unsafe mmap

採用しない。RFはBurn Storeのfile-backed pathを利用でき、codecはsafe streamingで十分な改善を得た。
独自mmapはplatform別lifetime/locking、unsafe slice、Windows file replacementの複雑性を増やす。

## 精度・速度・memory gate

- strict FP32、TF32 off、autocast off
- 4 Euler evaluations、forward batches `[2,2,1,1]`
- effective rows 6、12 layers、48 block calls
- fixed112 final audio hash: 全run一致
- all-resident live bytes: 全run `3,710,995,136`
- request path、RF/codec shader、sync/readback/consumer境界: 変更なし
- unit tests: `515 passed / 0 failed / 16 ignored`

load optimizationはstartupだけに作用する。process初回のpipeline compile/autotuneやsteady latencyを
load timeへ混ぜず、CubeCL cache/DryRunの結果は既存cache migration reportを参照する。

## 再開手順

1. branchとsource commitを確認し、artifact `SHA256SUMS`を検証する。
2. GPU/driver/adapter/index/PCI/free VRAMを再取得し、差があれば新campaignを作る。
3. pinned source `.pth`からconverterでfullとdecoder-onlyを再生成し、上記SHAを照合する。
4. CubeCL environmentは`Irodori-TTS-burn`のplatform cache rootを使い、別GPU/旧versionを流用しない。
5. all-residentは`OnlineSession::load_parallel`、phase batchはsequential loadを選ぶ。
6. manifest DryRun後にreal validationを通し、その後だけservice readyにする。
7. 45/112/255/333/489/685のaccuracy、steady latency、VRAMを再確認してから次の演算最適化へ進む。

## 残存するload短縮候補

統合後もload wall `5.195 s`のうちRF checkpoint stageがmedian `4.535 s`、約87%を占める。
次cycleでは次の順序で評価する。

1. `rf_checkpoint_seconds`をmodel構造初期化、仮weight生成、safetensors index、tensor materialize、
   upload submission、device syncへ分解する。現値はstage末尾device syncを含まないため、
   submission-completeとdevice-completeを混同しない。
2. checkpointで直ちに置換される約3 GBの仮weight初期化を省くcheckpoint-first constructorを検討する。
   Burn 0.22にstoreから未初期化moduleを直接構築する高水準APIは確認できないため、まず初期化時間を実測し、
   module不変条件とparameter IDを保てる場合だけ専用constructorへ進む。
3. Burn Storeのsafetensors lazy closureがtensor materializeごとに約86 KBのheaderを再parseして
   tensor名を検索する経路を、初回parseで検証済みoffset、length、dtype、shapeをcaptureする形と比較する。
   logical tensor値を変えず、OSに依存しないため、残候補の中では比較的低riskである。
4. PyTorch互換safetensorsとBurn-native checkpointを比較する。native形式ではregex key remap、
   `PyTorchToBurnAdapter`、load時transposeを省ける可能性がある。logical FP32 tensorの全件一致、
   source/converter/output SHA、format schemaを必須gateにし、prepared WGSL cacheは混ぜない。
5. parameterごとのhost bufferとGPU uploadが支配的なら、module group単位のstaging uploadと、
   module traversal順に並べたcheckpointを別campaignで評価する。allocator/handle所有に踏み込むため高riskである。
6. restored-cache時のDryRunは約0.34秒なので、pipeline keyによる重複除去は上記RF load候補の後に行う。

codecの追加短縮、約0.19秒のprepared-weight永続化、独自unsafe mmap、Vulkan限定pipeline cacheは、
現時点では上記候補より費用対効果またはportabilityが低い。
