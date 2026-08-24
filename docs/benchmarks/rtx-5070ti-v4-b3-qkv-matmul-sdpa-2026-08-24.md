# RTX 5070 Ti / B3 epilogue・direct QKV・matmul SDPA (2026-08-24)

## 結論

3つの構造候補を実装し、489 latent frames、Voice Design、40 Euler evaluationsで同じ新binaryから
個別に測定した。

- **matmul型SDPAは採用**。既存のdevice-tuned `QK^T` / `P@V` matmulを維持し、間のscale、
  key-padding mask、NaN-safe softmaxだけをin-place 1 dispatchへまとめた。5 fresh process、各
  2 warmup + 10 measuredのRF session中央値は `5.024849 s -> 4.521771 s`、**10.01%短縮**した。
- **B3 MLP compressed-output max tileは候補保持・RTX既定ではreject**。full `[M,2H]` outputを
  作らない1-dispatch epilogueは成立したが、RFはscreenで `5.197774 s`、control比4.65%遅かった。
- **CubeK QKV accumulator scatterは候補保持・RTX既定ではreject**。`[B,S,4D]` projectionを
  作らずQ/gateとpacked K/Vへ直接storeできたが、RFは `5.251409 s`、control比5.73%遅かった。

matmul SDPAのpersistent in-use bytesはcontrolと同一だった。一方、明示score matrixのlive rangeに
よりNVML request peakはformal中央値相当で約 `6,288 -> 6,445 MiB`、157 MiB増えた。12 GiB
all-residentは維持するが、速度とrequest peakのtrade-offとして扱う。

同じcampaignでPyTorchも取り直した。PyTorch RFは `4.126792 s`で、改善後WGPUはなお**9.57%遅い**。
過去のPython値は今回の集計へpoolしていない。

## 実装

### B3 MLP compressed-output epilogue

既存のtyped CubeK pairwise writerへ、永続化可能なrouteとしてminimum / maximum unit tileを分離した。

```text
[M,K] @ interleaved [K,2H]
  -> accumulator pair (gate, value)
  -> SiLU(gate) * value
  -> compact [M,H]
```

`CubeKSwiGluTile::{Min, Max}`はdevice名から推測せず、`SwiGluRoute`のstable candidate IDで選ぶ。
partial M/N tileはwriter側でmaskしてからepilogueを呼ぶため、捨てるlaneからparameter readしない。

### QKVからpacked Q/K/Vへの直接store

vendored CubeKへgeneric `AccumulatorGlobalScatter<RuntimeArgs>`とtyped launch pathを追加した。Irodori側の
`QkvProjectionScatter`はmatmul accumulatorのlogical `(row,column)`を次へ直接写像する。

```text
Q      -> [B,H,S,Dh]
K / V  -> [B,H,S+context,Dh]
gate   -> [B,S,D]
context K/V -> packed tail
```

通常のprimary matmul outputは物理化しない。Q/KのDh-wide reductionが必要なRMSNorm+RoPEだけは、
続く1 dispatchでin-place実行する。従ってprojection/scatter本体は1 dispatchだが、attention front-end
全体は2 dispatchであり、「QKV operator全体が1 dispatch」とは表現しない。

CubeK scatterはM/N tailをgeneric writerで先にmaskする。Irodori launcherはF32、shape、stride、
column-major prepared weight、binding device/client、u32 address範囲、workgroup/grid/shared-memoryを
dispatch前に検査し、一つでも不成立なら従来routeへ戻る。

### matmul型SDPA / Flash候補集合

採用経路は次である。

```text
QK^T: existing tuned matmul
  -> one in-place WGSL dispatch
       scale + mask + row max + exp/sum + normalize
  -> P@V: existing tuned matmul
```

softmaxは1つの`[B,H,Sq]` rowを256 invocationのworkgroupへ割り当て、F32 shared reductionを行う。
全key masked行は厳密にzeroを返す。maskは`true = masked-out`へruntime境界で正規化する。

既存のscore-free `CubeClPlane`と`CubeKFlashUnit`も同じ`SdpaRoute`候補集合に残す。前者は489-frame
RTX測定で遅かったため既定化せず、他adapterのexact tuner候補である。今回採用したmatmul routeは
score matrixを保持するためFlashではなく、matmul throughputを優先した中間案である。

route ABIは`v4-dit-route-9`へ更新した。NVIDIA family priorでは、実測したexact cell
`B1/S489`と`B3/S489`だけmatmul SDPAを既定にした。Apple M5、他shape、AMD、Intel、他backendは
従来routeのままで、個別profileが承認できる。

## 固定条件

- feature commit: `8191864666af882f56ef29bad64d6a12c96acdbd`
- measurement binary SHA-256: `3bfb07c53003b44d44edc7d1b07bbaa4856aac29281ca101bbf2ae7d7302f982`
- final default-enabled binary SHA-256: `001e78c27d3cbf889244fca79756d3713805c23f0ab14b2b2d0b47f0e4ad574c`
- model revision: `e4aaac4df355ff560dcd35e0dae272c3a759317b`
- model SHA-256: `5863c986345d9f6d20b7d8748fee1af02079c5161cf0c9e52557da0a0c378593`
- codec revision: `47376ee24834d7a05a48ebabfe3cde29b3c5e214`
- converted decoder-only codec SHA-256: `1b1ceb3f620525cf4252af508c0fde80e3779582d47fc7fc879410d2e4abe231`
- fixture SHA-256: `9a1e00e667f960983b62ebc9188c6b430acf0c00d0721ef9ffdf8fc8b9fd4b3f`
- source-noise SHA-256: `17e9016569e9e087001bebde393d7039d84e0beaee81a3fef7438a91bcdf186b`
- GPU: NVIDIA GeForce RTX 5070 Ti Laptop GPU
- driver: `595.71.05`; Vulkan adapter 0; WGPU vendor/device `0x10de/0x2f18`
- CUDA/NVML index: 0; PCI bus ID `00000000:01:00.0`
- physical VRAM: 12,227 MiB; campaign開始時free 11,774 MiB
- strict FP32、TF32 off、autocast off
- 40 Euler evaluations、forward batches `[3 x 20, 1 x 20]`、effective rows 80
- 12 layers、480 block calls、scheduleの全41 FP32 bits一致
- exact 489 latent frames、Voice Design、caption CFG 4.0、19.56秒audio
- WGPU: `ExclusivePages`、exact-manifest RF weights、decode-only codec、all-resident
- boundary: pre-stage device syncからRF device completion。readback/consumerは別記録

候補ごとに空のCubeCL cacheから開始した。formal sessionはこの新campaign内cacheのrestore条件だが、
WGPU ComputePipelineは各fresh processで再生成される。過去の`/tmp`や旧campaignの計測値は使っていない。

## 性能

### 構造候補screen

各routeは1 fresh process、2 warmup + 2 measured。値は2 measuredの中央値である。

| route | RF device-complete | consumer-complete | persistent in-use | NVML peak | disposition |
|---|---:|---:|---:|---:|---|
| control | 4.966595 s | 5.379099 s | 3,556,388,224 B | 6,400 MiB | control |
| B3 compressed max tile | 5.197774 s | 5.609139 s | 4,008,586,624 B | 7,347 MiB | reject |
| CubeK direct QKV scatter | 5.251409 s | 5.732192 s | 3,870,961,024 B | 6,795 MiB | reject |
| matmul fused-softmax SDPA | 4.467952 s | 4.890639 s | 3,556,388,224 B | 6,493 MiB | formalへ進行 |

MLPとQKV候補のpersistent増分は、それぞれB3で必要なinterleaved w1/w3（約431 MiB）と
column-major QKV（約300 MiB）である。中間buffer削減だけを見てVRAM改善と主張しない。

### matmul SDPA formal

| runtime/route | fresh session RF medians (s) | median of medians | consumer median |
|---|---|---:|---:|
| WGPU control | 4.987777 / 5.017800 / 5.024849 / 5.030986 / 5.029509 | **5.024849** | 5.441808 |
| WGPU matmul SDPA | 4.495941 / 4.505876 / 4.521771 / 4.524471 / 4.526851 | **4.521771** | 4.944179 |
| PyTorch/CUDA | 4.067959 / 4.111279 / 4.126792 / 4.144437 / 4.148982 | **4.126792** | CUDA full 4.696992 |

WGPU route差はRF 10.01%、consumer-complete 9.14%改善。PyTorch比ではWGPU RFが9.57%遅い。
PyTorchは2.10.0+cu128 / CUDA 12.8 / cuDNN 91002、`highest` matmul precision、TF32と
autocastを無効にしている。同じsemantic workだがoperator graphは同一ではない。

GPU timestampでは480 SDPA calls合計が`1.164656 s -> 0.665590 s`、**42.85%短縮**した。
RF全体の短縮約503 msとSDPA短縮約499 msが一致し、他stageの偶然の変動を改善理由にしていない。

## Accuracy

WGPU controlをoracleとした489-frame最終waveform比較:

| route | max abs | RMSE | SNR | cosine |
|---|---:|---:|---:|---:|
| B3 compressed max tile | 2.331e-4 | 6.876e-6 | 87.364 dB | 0.999999999082 |
| CubeK direct QKV scatter | 5.709e-4 | 1.453e-5 | 80.867 dB | 0.999999995905 |
| matmul SDPA | 8.014e-5 | 3.287e-6 | 93.774 dB | 0.999999999790 |

matmul SDPAは80 dB hard gateと85 dB / max abs 2e-4 targetの両方を通る。QKV scatterは演算順序差を
許容するhard gateを通るが、速度でrejectした。各routeは全formal measured requestでprocess内・
process間hashが一つに固定された。

default昇格後、route profileを指定せず空のCubeCL cacheから起動した追加smokeはRF
`4.469385 / 4.465338 s`となり、matmul routeが既定で実行された。このprocessは別の内部matmul候補を
選んだためcontrol比SNR 84.039 dB、max abs `4.021e-4`、RMSE `1.008e-5`、cosine
`0.999999998027`だった。hard gateは通るが85 dB / max abs 2e-4 target外であり、formal restored-cache
結果と区別する。演算順序差を許容する現在の方針に基づき、これだけでrouteを撤回しない。

PyTorchと採用WGPUの最終waveform差はSNR 67.928 dB、max abs `2.770e-3`、RMSE `6.444e-5`、
cosine `0.999999919434`である。ユーザー方針どおり演算順序差だけでは性能routeをrejectしないが、
従来の80 dB numerical reproducibility gateを通ったとは記載しない。

## 移植性とcrate ergonomics

- matmul SDPAの周辺shaderはWGSLで、256-thread workgroup、2 KiB shared memory、2 bindingsに収まり、
  Vulkan固有extensionを使わない。sourceはMetal/DX12/WebGPUへ共有できるが、実測承認はRTX/Vulkanだけ。
- QKV scatterはCubeK DSLのgeneric writerとtyped runtime argsに分離した。Irodori固有shape mappingは
  `QkvProjectionScatter`へ閉じ、CubeK matmul coreへmodel定数を入れていない。
- `SwiGluRoute`、`AttentionMaterializationRoute`、`SdpaRoute`が候補IDを保持する。request hot pathは
  adapter名や環境変数を解釈せず、startupで解決済みのroute tableを読む。
- `RouteRequirementSet`がrouteから必要weight layoutを導出する。QKV scatterはcolumn QKVとpacked
  Q/K norm、compressed MLPはinterleaved w1/w3を明示要求し、paired `Option`で暗黙表現しない。
- dtype/shape/layout/deviceが合わないlaunchは`Option` fallbackで既存graphへ戻る。F16は今回の3 routeを
  未承認のため選ばない。

## 検証

- `cargo clippy --all-targets --features inference,codec,cli,profile -- -D warnings`
- lib: 574 passed / 21 ignored
- CubeK compressed partial-tail GPU/CPU differential: pass
- matmul softmax mask / fully-masked-row GPU test: pass
- CubeK direct QKV scatter GPU integration: pass
- route/runtime focused tests: 22 + 13 passed
- `cargo fmt --all -- --check`
- `uvx ruff check scripts/bench_python_runtime_scenarios.py scripts/compare_f32_audio.py`

## 残る優先事項

1. PyTorchとの差9.57%は主にQKV projection、MLP expand、MLP contractへ移った。SimpleUnit専用writerでは
   なく、現defaultのhigh-throughput CubeK routineへtyped compressed/scatter writerを接続する。
2. matmul SDPAのscore workspaceをattention層間で安全に再利用するprepared plan、または高throughput
   Flash kernelで157 MiBのrequest peak増分を取り戻す。層ごとの12重persistent workspaceは作らない。
3. QKV scatterはscalar storeのcoalescingとRMSNorm reductionをtile writerへ持ち上げる。column layout
   300 MiBを追加したままでは採用しない。
4. MLP compressed routeはpair-aware writerをdouble-buffer / ordered routineまで一般化し、max/min tileの
   手動選択ではなくexact-device tunerへ候補集合として渡す。
5. M5、旧Apple、他NVIDIA、AMD、Intel、DX12で個別profileを作る。family defaultは候補順のpriorに留める。

## Artifactと再開

fresh campaign root:
`/home/sanzentyo/benchmark-artifacts/irodori-v4-b3-qkv-sdpa-20260824-attempt1`

主要raw data:

- screen: `results/{baseline,swiglu_max,qkv_scatter,matmul_sdpa}.json`
- formal WGPU: `results/{baseline,matmul_sdpa}-formal-{1..5}.json`
- formal Python: `python-formal/session-{1..5}-run/result.json`
- GPU timestamps: `results/{baseline,matmul_sdpa}-rf-device-profile.json`
- waveform metrics: `results/*-accuracy.json`、`results/matmul-sdpa-vs-python-waveform.json`
- raw NVML/log/CubeCL/audio、environment pin、source/binary hashes

再開時はcommitをcheckoutし、model/codec/fixture SHAを検証する。route profileでcontrolとcandidateを明示し、
新しいCubeCL directoryを使って同じ`2 warmup + 10 measured`を実行する。未計測shapeへS489の結果を
rangeとして流用せず、exact cellごとにaccuracyと5 fresh sessionを承認してからdefaultを広げる。
