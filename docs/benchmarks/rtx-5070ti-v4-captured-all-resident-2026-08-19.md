# RTX 5070 Ti Laptop: all-resident codec software graph（2026-08-19）

## 結論

112 latent frames（4.48秒相当）、strict FP32、RF/duration/decoder同居のonline条件では、
codec software graphを既定経路にしない。5 fresh sessionのmedian-of-mediansは次の通りだった。

| 境界 | eager | captured | captured - eager |
|---|---:|---:|---:|
| codec device-complete | 76.179 ms | 77.665 ms | +1.487 ms（+1.95%） |
| codec readback-complete | 76.648 ms | 78.183 ms | +1.535 ms（+2.00%） |
| request consumer-complete | 222.260 ms | 226.358 ms | +4.098 ms（+1.84%） |

capturedの最初のcodec device-complete中央値は70.869 msで、eagerの75.332 msより
4.463 ms短い。しかしgraph構築時に実decodeを行い、captureだけで中央値295.494 msを前払いする。
steady 10 requestでは逆に遅いため、これは総仕事量の削減ではなくwarmup位置の移動である。

`CapturedDacVaeDecoder`と`CapturedOnlineSession`は明示的なfixed-shape service policyとして残すが、
`bench_v4_residency`と通常sessionのdefaultはeagerを維持する。50-frame F16 codec単体で得た
software graphの小さな利益を、112-frame FP32 all-residentへ一般化しない。

## 固定条件

- source: `1620f539a7a072e410eeb2a5a167151976214c36`
- branch: `codex/v4-wgsl-fusion-autotune`
- GPU: NVIDIA GeForce RTX 5070 Ti Laptop GPU、12,227 MiB
- driver: 595.71.05
- Vulkan/WGPU adapter: discrete adapter index 0
- NVML/CUDA index: 0
- PCI bus: `00000000:01:00.0`
- allocator: `ExclusivePages`
- model SHA-256: `5863c986345d9f6d20b7d8748fee1af02079c5161cf0c9e52557da0a0c378593`
- decoder-only codec SHA-256:
  `1b1ceb3f620525cf4252af508c0fde80e3779582d47fc7fc879410d2e4abe231`
- fixture SHA-256: `f90e785823da3a0ec05caddadfc3d337bf833ad003daa9ff968f42086043d032`
- precision: strict FP32、TF32 off、autocast off
- RF: Euler 4 evaluations、forward batch `[2,2,1,1]`、effective rows 6、
  12 layers、48 block calls、runtime manifestでschedule bits照合
- RF weight profile: `Fixed112PackedOnly`
- codec weight profile: `PortableFallback`
- voice: unconditioned
- startup: fresh process、fresh XDG driver cache、fresh CubeCL environment、今回primeした
  environment bundleだけを各processへimport
- sample: conditionごとに5 fresh session、各2 warmup + 10 measured
- condition order: session単位でeager/capturedを交互に反転
- automatic retry: 0

旧`/tmp`、過去campaignのcache、旧計測値は使用・poolしていない。prime processはcache生成専用で、
集計へ含めていない。

## session別結果

| session | eager device ms | captured device ms | eager readback ms | captured readback ms |
|---:|---:|---:|---:|---:|
| 1 | 79.375 | 72.029 | 79.802 | 72.537 |
| 2 | 77.686 | 77.665 | 78.103 | 78.183 |
| 3 | 74.760 | 79.867 | 75.077 | 80.398 |
| 4 | 76.179 | 76.349 | 76.648 | 76.823 |
| 5 | 74.905 | 79.610 | 75.345 | 80.081 |

各値はそのsessionの10 measured requestの中央値である。session間の変動が大きいため、
1 session目だけならcapturedを誤採用する。5 sessionの中央値ではdevice/readbackとも非回帰gateを
通らない。

`codec_readback_complete_seconds`はcodec開始から、pre-start device sync、dispatch、device sync、
owned contiguous CPU F32 result取得までを含む。hash consumerはその後に行う。
片方だけreadbackを含めた比較ではない。

## 精度と仕事量

eager/capturedの全120 requestは、最終CPU F32 audio hash
`5c22e03be6864d320a7881939b318d0d066b06af3005942457a7dc7e1e43c8b9`
でbitwise一致した。各session内のhashも一意で、stderr/WGPU uncaptured errorは0だった。

graph replayは通常decodeと同じoperator列を保持し、その前に3,584要素のRF latentをstable GPU
bufferへbit-preserving copyする。latentやintermediateのCPU readbackは行わず、最終audioだけを
CPUへ渡す。RF runtime manifestも全requestで同じ4-step意味論を確認した。

## 起動時間とVRAM

| 項目 | eager | captured |
|---|---:|---:|
| load wall median | 10.483 s | 10.926 s |
| graph capture wall median | - | 0.295 s |
| NVML peak range | 4,444--4,446 MiB | 4,441--4,443 MiB |

graph capture直前から直後のCubeCL allocator差は全sessionで同じだった。

- in-use: +874,496 bytes（約0.834 MiB）
- reserved: +629,698,560 bytes（約600.527 MiB）

reserved増分はgraph arenaの論理page予約である。eagerもtraffic中にmain poolを拡張するため、100 ms
NVML peakではcapturedの増加として現れなかった。ただし複数shapeをcaptureすればarenaはshapeごとに
増えるため、NVML peakが同じことを「無償の600 MiB」と解釈してはいけない。

## 修正したgraph allocator不具合

最初のall-resident smokeでは、82--165 MiBのgraph allocationに対して約8--9 EiBを予約しようとして
失敗した。原因はoversize用`ExclusivePages`を`u64::MAX`一個で構成したことだった。CubeCLの
exclusive poolは初回平均allocationを最大値の半分で初期化するため、実要求より桁違いに大きいbufferを
作ろうとしていた。

修正後は64 MiB以下をsliced poolへ、超過分を128 MiBからdeviceのmax page sizeまでの倍々の
exclusive bucketへ送る。各bucketが受理するrequestはbucket上半分にあるため、初回allocationは実要求の
aligned sizeになる。vendor単体testとall-resident実GPU smokeを通した。修正commitは
`41421b0d77082f3fb8f61437f1c40423b33cca57`である。

APIでは`CapturedCodecOutput<'session>`が`&mut CapturedDacVaeDecoder`を借用し、CPU resultを消費するまで
次のreplayを型で禁止する。reusable GPU output tensorは外へ逃がさない。このguardによりdevice-completeと
readback-completeを分離できる。

## artifact

authoritative artifact:

`/home/sanzentyo/benchmark-artifacts/irodori-v4-captured-all-resident-20260819-attempt5`

binary/source/runner pin、environment、fresh bundle、raw JSON/stdout/stderr、100 ms NVML、wall/RSS、
各sessionの`SHA256SUMS`、campaign全体の`SHA256SUMS`と`COMPLETE`を含む。最終manifestは
`sha256sum --quiet --strict --check SHA256SUMS`で再検証した。

非authoritative attemptはpoolしていない。

- attempt1: allocator修正前、graph OOMで失敗
- attempt2: bucket修正後のsmokeのみ
- attempt3: `vulkaninfo`未導入をGPU処理前に検出してfail-closed
- attempt4: 5-session完走だがcodec単独readback境界をまだ明示しないschema 5
- attempt5: schema 6 authoritative campaign

## 次の優先順位

1. k7 haloをaffine stageへ全面scatterせず、non-affine
   `(m, channel, kernel) -> (input_time, channel)` readerをCubeK MMAへ接続する。
2. ConvTranspose/NHWCはlane入替やdual outputを繰り返さず、raw shortcutのglobal write/read自体を
   消せるproducer-consumer境界だけを検討する。
3. 構造変更後にaccuracy-approved selector manifestを作り直す。
4. software graphは112-frame FP32の速度施策として再調整しない。新しいdispatch topology、別precision、
   または複数request replayにより固定launch overheadが支配的になった場合だけ再測定する。
