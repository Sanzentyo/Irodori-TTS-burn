# RTX 5070 Ti / strict-FP32 RF route validation (2026-08-24)

## 結論

2026-08-24 09:00 JSTまでの最適化では、489-frame Voice Design相当の40-step RFでPythonを
上回らなかった。最終WGPU 5 fresh sessionのdevice-complete中央値は4.855899 s、同一意味論・
同一境界の正式Python値は4.548211 sで、WGPUは0.307688 s、6.765%遅い。直前WGPU
4.857645 sとの差は-1.746 ms、-0.036%に留まり、速度改善とは判定しない。

今回の主成果は性能値ではなくroute選択のcorrectnessである。以前3.55%改善として採用した
`ProjectionDirectPackedKvSubgroup`は、実際の22-token contextをkernelが受理せずfallbackを測って
いた。実経路を通すように直して再測定すると、通常の二段direct materializationがreferenceより
20.28 ms速くbitwise同一、projection-directは200.76 ms遅かった。raw WGSL subgroup版は現Naga
frontendではcompile不能である。このためRTX既定は`DirectPackedKv`へ戻し、raw subgroup候補を
fail-closedにした。

CubeKのstrict-F32 `FlashUnit`も候補化したが、同一process・同一fixtureのwarmupとmeasuredで
audio hashが変わったためrejectした。route tunerには、この種の候補を
`NonDeterministicOutput`として永続receiptへ残し、選択しない契約を追加した。

## Pinsと測定境界

- branch: `codex/v4-post-seal-priority-1-4`
- implementation commit: `c4e059192666b3bd2b232a4b89fa677dd9154013`
- measurement base: `57a0be19028f82aa08af7ecce4e2a437d6ea5fe6` + measured diff SHA-256
  `1eae8c6d3cdf1b6748c8990b50505c58b2a656d11d1cd6385d7889a202252dcc`
- five-session measured binary SHA-256:
  `a2ec5db0c13969622268c2eed5992dc0a4dc8e0ff89afe0bdd591ef6b4a3ed65`
- post-commit rebuild binary SHA-256:
  `c868152a641172f98201c0957ce55725bc94be7a797f4749c3805ce078138f20`
- `Cargo.lock` SHA-256:
  `b6e4da0e76c391f863821c6ea911fc8d61b649303a16a63aa6b4d199ef6187d8`
- Burn `0.22.0-pre.2`、burn-cubecl `0.22.0-pre.2`、CubeCL `0.11.0-pre.2`
- rustc `1.95.0 (59807616e 2026-04-14)`
- GPU: NVIDIA GeForce RTX 5070 Ti Laptop GPU
- driver: `595.71.05`
- WGPU adapter: index 0、Vulkan、vendor `0x10de`、device `0x2f18`
- CUDA/NVML index: 0
- PCI bus ID: `00000000:01:00.0`
- physical VRAM: 12,227 MiB、campaign前available: 11,774 MiB
- model revision: `e4aaac4df355ff560dcd35e0dae272c3a759317b`
- model SHA-256: `5863c986345d9f6d20b7d8748fee1af02079c5161cf0c9e52557da0a0c378593`
- decoder-only codec SHA-256:
  `1b1ceb3f620525cf4252af508c0fde80e3779582d47fc7fc879410d2e4abe231`
- fixture SHA-256: `9a1e00e667f960983b62ebc9188c6b430acf0c00d0721ef9ffdf8fc8b9fd4b3f`
- strict FP32、TF32 off、autocast off
- 40 Euler evaluations、B3 x 20 + B1 x 20、effective rows 80、12 layers、480 block calls
- 489 latent frames、19.56 s audio、caption CFG 4.0
- `ExclusivePages`、all-resident、exact-manifest RF weights、decode-only codec
- 時間境界: pre-stage device syncからRF device completionまで。readbackやcodecをRF時間へ含めない

five-session測定後にrustfmtとtest-only修正を行ったため、測定binaryとpost-commit rebuildのhashは
異なる。production経路の意味は同一だが、bit-identicalなbinary再現を偽ってはいけないため両方を
記録した。さらにpost-commit rebuildを別fresh processで再確認し、5-session集計にはpoolしていない。

## 発見したroute検証上の問題

### 1. 固定3-token contextによるsilent fallback

公式Voice Design fixtureのcompact contextは22 tokenである。一方、direct materializationと
projection-direct kernelは`CTX=3`をsourceとhost validationへ固定していた。そのため以前の
projection-direct profileはselector上では候補でも、実行時に`None`を返してportable graphへ
fallbackしていた。

kernel template、kernel ID、出力shape、packed context cache、resource accountingを動的contextへ
一般化した。新contractは`1 <= context <= latent sequence`で、B1/B2/B3、F32/F16のstorage layoutを
同じhost contractで検査する。実fixtureでQ/K/Vが次のcontiguous layoutになることも確認した。

```text
Q:   shape [3,20,489,64], strides [625920,31296,64,1]
K/V: shape [3,20,511,64], strides [654080,32704,64,1]
```

### 2. CubeCL handleの`offset_end`解釈

packed Q/gate allocationのview分割で、`offset_end`を絶対end addressとして扱っていた。しかし
CubeCL 0.11の`Handle::offset_end`はallocation末尾から除外するbyte数である。sub-allocationでも
正しい`size_in_used()`になる`split_leading_views`へ一本化し、Qとgateの非重複viewを作る。

### 3. hardware subgroupとsource compiler capabilityの混同

adapterが32-lane plane operationを報告しても、raw `SourceKernel` WGSLはNaga 30のWGSL frontendを
通る。現frontendは`enable subgroups;`を実装していないため、hardware capabilityだけを根拠にraw
subgroup shaderを選ぶと初めて実経路に入った時点でcompileに失敗する。

`SubgroupWgsl`と`ProjectionDirectPackedKvSubgroup`は自動candidate集合から外し、support predicateも
falseを返すようにした。これはVulkan/NVIDIA分岐ではなく、使用中source compilerの能力を表せない
限り全platformでfail-closedにする判断である。CubeCL DSLから生成するkernelはVulkanでSPIR-V等の
別compiler pathを取れるため、今後のsubgroup attentionはraw WGSLではなくCubeCL plane APIで実装する。

### 4. route-derived residencyの不足

`WeightResidencyPlan`は`DirectPackedKv`だけを`QkNormPacked`利用者として扱い、projection-direct二種を
漏らしていた。source-free lock後に実経路が初めて通ると、必要weightが存在しないinvalid stateに
なり得る。全direct variantが`QkNormPacked`を要求するよう修正し、現在選択可能な二経路をfocused
testで固定した。

## Route A/B

同じbinary family、同じ復元済みCubeCL database、各2 warmup + 5 measuredで測った。sessionを交互に
実行した完全なpaired ABBAではないため、単一sessionの差を最終性能値には使わない。

| materialization route | RF median (s) | referenceとの差 | audio SHA-256 | disposition |
|---|---:|---:|---|---|
| reference graph | 4.870326 | - | `4a9f4fdf...` | control |
| direct packed K/V | 4.850049 | -20.277 ms (-0.416%) | `4a9f4fdf...` | RTX default |
| projection-direct | 5.051082 | +180.756 ms (+3.711%) | `0b0afca5...` | performance reject |

projection-directはreferenceに対して約3.71%、direct packed K/Vに対して約4.15%遅い。hash差は今回
許容した演算順序差だけではrejectしないが、速度で明確に負けるため採用しない。

CubeK `FlashUnit`は4-step smokeのみで、40-step性能値として比較してはいけない。warmup hash
`c4d766f1...`とmeasured hash `9ee77396...`が同一process内で異なった。route tunerは今後、各fresh
sessionのwarmupを含む全request hashを比較し、一つでも異なれば
`RouteCandidateRejectionReason::NonDeterministicOutput`としてfail-closedにする。cross-processで
許容する演算順序差と、同一process・同一選択routeの不安定性を区別した。

## 最終40-step性能

復元済みの同一CubeCL databaseを使い、5 fresh processで各2 warmup + 5 measuredを取得した。
restored cache hitは新しいtune decisionではないため、各sessionの`autotune.json.log`は空である。

| fresh session | RF device-complete median (s) | NVML peak (MiB) |
|---:|---:|---:|
| 1 | 4.829014 | 6,288 |
| 2 | 4.837701 | 6,289 |
| 3 | 4.855899 | 6,282 |
| 4 | 4.861617 | 6,289 |
| 5 | 4.870354 | 6,287 |
| median of session medians | **4.855899** | **6,288** |

全35 request（warmup 10 + measured 25）のaudio f32 SHA-256は
`4a9f4fdf90b725ab9d16ff48273ca816395030274e9ab072fba5829f34c57cb4`で一致した。

post-commit rebuildの独立verificationは4.829770 / 4.830245 / 4.820321 s、中央値
4.829770 s、同一audio hash、NVML peak 6,289 MiBだった。これは再現確認であり、上表へpoolして
いない。

正式Python値4.548211 sとの比較は次のとおりである。

| runtime | RF device-complete median (s) | Python比 |
|---|---:|---:|
| PyTorch/CUDA strict FP32 | **4.548211** | reference |
| WGPU/Vulkan strict FP32 | 4.855899 | +0.307688 s、+6.765% |

Python値は同一request、同一fixture、同一40-step schedule、同一RF意味論、同一device-complete境界の
直前正式campaignから引用した。今回の5 WGPU sessionへ旧WGPU値や`/tmp` artifactをpoolしていない。
same semantic workであり、operator graphまで同一ではない。

## VRAM

最終5 sessionでallocator値は全て一致した。

| stage | bytes in use | MiB |
|---|---:|---:|
| RF resident | 3,417,207,424 | 3,258.90 |
| all-resident after consumer | 3,556,110,976 | 3,391.37 |
| all-resident reserved after consumer | 6,401,099,456 | 6,104.56 |

NVML device peakは6,282--6,289 MiBである。reservedは`ExclusivePages` allocator poolであり、
persistent in-useではない。今回のdirect route修正はweightを追加せず、VRAMは直前のexact-manifest
構成を維持した。12 GiB all-residentは引き続き成立する。

## Cacheとcrate ergonomics

`bench_v4_residency`へ`--cubecl-autotune-record PATH`を追加し、persistent cache directoryと同時にだけ
指定可能にした。raw JSON schemaは13へ更新し、cache rootとrecorder pathを別fieldで保存する。
recorderはfresh decisionのJSONL証拠であり、restored hitでは空になる。これを第二のcacheやroute選択の
authorityとして扱わない。

Rust APIではrouteをenum、problemをvalidated newtype、候補失敗を
`RouteCandidateRejectionReason`で表す。今回`NonDeterministicOutput`を追加したため、文字列解析や
paired `Option`なしで失敗理由を保存できる。route ABIを`v4-dit-route-7`へ上げ、以前の誤った
subgroup selection manifestを自動reuseしない。

現状の不足は、CubeCL内部matmul algorithm receiptがIrodoriのtype-state lockへまだ統合されていない
点である。CubeCL recorderを観測用に露出できたが、stable algorithm ID、driver/source hash、全sample、
accuracy dispositionを含むIrodori側sealed receiptが完成するまでは、内部candidate indexをexact profileの
authorityにしない。

## 次に残る本質的な改善

1. **CubeCL DSLのplane-based SDPA**: raw WGSL subgroupではなく、CubeCL IRの`plane_sum`/
   `plane_max`を使うstrict-F32 attentionを、B/S/SKV/maskを型付きproblem keyに持つ汎用candidateとして
   実装する。現`FlashUnit`の非決定性を先に局所化し、operator differentialと40-step approvalを必須にする。
2. **inner matmul selectionのseal**: CubeCL recorderからstable algorithm identityを抽出し、adapter、driver、
   compiler、dtype、shape/stride、source hashを含むIrodori receiptへ昇格する。cache missやunknown candidateは
   retune、`PreparedModel<ProfileLocked>`後はtune不能にする。
3. **Burn custom Fusion provider**: tuned large matmulは維持し、QKV postprocess、post-SDPA、SwiGLUの
   elementwise/epilogueだけをFusion providerへ接続する。特定GPUのtile定数ではなく中間allocationと
   dispatch境界を減らす。
4. **GPU timestampとlive-rangeの同時計測**: 12 layers x B3/B1についてmatmul本体、writer drain、SDPA、
   allocator live bytesを同期なしtimestampで採取する。残り6.8%をoperator単位へ帰属してから候補を増やす。

AMD、Intel、旧Apple、他NVIDIA世代はportable fallbackから個別承認する。family defaultは候補順のprior
だけに使い、raw WGSL compiler capability、subgroup幅、shared memory、driver compile品質をGPU名から
推測しない。

## Verification

- `cargo test --lib --features inference,codec,cli,profile`: 582 passed、20 ignored、0 failed
- focused route autotune tests: 21 passed
- focused route tuner tests: 3 passed
- focused materialization tests: 5 passed
- `cargo clippy --all-targets --all-features -- -D warnings`: pass
- `cargo fmt --all -- --check`: pass
- `uvx ruff check scripts`: pass
- `git diff --check`: pass

## Artifacts

fresh campaign root:
`/home/sanzentyo/benchmark-artifacts/irodori-v4-rf-matmul-seal-20260824`

root `SHA256SUMS` SHA-256:
`a06e5c6b67cb119811b13ef4250aaa080d2d02711546f5c1a3f0a85150af5c95`

- final 5 sessions: `final-built-in-v7-s1/` ... `final-built-in-v7-s5/`
- exact post-commit verification: `postcommit-verification-s1/`
- A/B: `abba-reference-s1/`、`abba-direct-s1/`、`abba-projection-direct-s1/`
- rejected CubeK attention: `flash-unit-smoke-v7/`
- fresh CubeCL decision evidence: `fresh-tune/autotune.json.log`
- environment/source/binary pins: `environment/`
- machine-readable aggregate: `campaign-summary.json`

各sessionにはraw `result.json`、stdout/stderr、NVML、wall timing、autotune recorder、保存対象audioを置いた。
失敗候補と4-step smokeは最終40-step中央値へpoolしていない。
