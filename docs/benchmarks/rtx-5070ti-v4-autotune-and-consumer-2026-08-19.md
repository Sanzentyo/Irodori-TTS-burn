# RTX 5070 Ti Laptop: pointwise autotune / consumer境界（2026-08-19）

## 結論

50 latent frames / 96,000 waveform samplesのF16 codecについて、pointwiseの行分割を
decoder graph内で再測定した。`rows >= channels`で一律にmulti-rowを選ぶ旧条件を、
`rows >= channels * 64`へ変更した。これによりC384の2本だけをsingle-rowへ戻し、
C192/C96のmulti-rowを維持する。

同一processのABBA/BAAB比較では、候補-controlのdevice-complete block中央値が5 fresh
sessionすべてで負になった。

| session | paired delta ms | improved blocks |
|---:|---:|---:|
| 1 | -0.243688 | 10/15 |
| 2 | -0.211338 | 11/15 |
| 3 | -0.211567 | 12/15 |
| 4 | -0.166384 | 11/15 |
| 5 | -0.233964 | 12/15 |

最終waveformは全sessionでcontrolとbitwise一致し、WGPU uncaptured errorは0だった。
このポリシーをproductionへ採用した。

採用後のprocess-local software graphは5 fresh sessionのdevice-complete中央値が
`13.556 / 13.716 / 13.585 / 13.952 / 13.610 ms`、median-of-mediansが
**13.610 ms**だった。旧graph campaignの13.759 msから約0.149 ms短縮した。
同じcampaignのreadback-complete median-of-mediansは14.439 msである。
比較対象のPyTorch codec device 13.391 msとの差は約0.219 ms（1.64%）まで縮小した。

## 固定条件

- source: `49939566d011f0f56a11154e97d103b8ef3e2714`（採用commit）
- GPU: NVIDIA GeForce RTX 5070 Ti Laptop GPU、12,227 MiB
- driver: 595.71.05
- adapter: Vulkan discrete adapter index 0
- PCI bus: `00000000:01:00.0`
- precision: F16 storage、F32 accumulator/Snake、TF32/autocastなし
- codec SHA-256: `b14a25da5d68bf779d8c44a40706ddc7c4819cb60489b2a80a6408ca03c514fb`
- fixture SHA-256: `08663e52df6c63eea7ebf5ae2ba35778bdb5bc7d20cd06e430e855a86174229e`
- automatic retry: 0

旧`/tmp`、別campaignのsample、旧CubeCL cacheはpoolしていない。各fresh processは専用の
cache directoryを使用した。

## pointwise stageの切り分け

single-rowを8本すべてへ強制する候補は全体では不採用だったが、GPU timestampにより原因を
形状別に切り分けた。

- C768: 実質中立
- C384: 1本あたり約0.08--0.09 ms短縮
- C192: 1本あたり約0.08 ms悪化
- C96: 1本あたり約0.06 ms悪化

このため個別shape tableではなく、行列の縦横比を表す汎用条件を採用した。artifact:

- `/home/sanzentyo/benchmark-artifacts/irodori-v4-pointwise-single-row-20260819-attempt1`
- `/home/sanzentyo/benchmark-artifacts/irodori-v4-pointwise-row-stages-20260819-attempt1`
- `/home/sanzentyo/benchmark-artifacts/irodori-v4-pointwise-tall-rows-20260819-attempt1`
- `/home/sanzentyo/benchmark-artifacts/irodori-v4-pointwise-tall-rows-graph-20260819-attempt1`

各directoryにsource/binary pin、raw log、NVML（該当campaign）、`SHA256SUMS`を保存した。
software graph campaignのNVML peakは全5 sessionで1,319 MiB、CubeCL memoryはcapture前
246,761,344 bytes in-use、capture後247,145,344 bytes in-useだった。graph固有の増分は
384,000 bytes in-use、134,217,728 bytes reservedで、ポリシー変更はbuffer topologyを
変えない。

## CubeK selector screen

production accumulator-store routineと同じepilogueへ、cache-keyで識別できるselector型を追加し、
single-rowの`default / no-swizzle / auto-partition / double-partition /
no-swizzle+auto-partition`をscreenした。

最初のattemptはcontrolだけproduction cross-block graphを通し、candidateはstandalone graphを
通していたため境界不一致だった。この値は採用判断に使用せず、attempt2へpoolしていない。

attempt2は両経路をstandalone graphとtall-row policyへ固定した。各selectorのpointwise stage
paired中央値はcontrol比`-0.004〜+0.010 ms`で、2% gateに届かなかった。全候補はbitwise一致、
WGPU error 0だが不採用とした。

- invalid boundary evidence:
  `/home/sanzentyo/benchmark-artifacts/irodori-v4-pointwise-selector-screen-20260819-attempt1`
- corrected evidence:
  `/home/sanzentyo/benchmark-artifacts/irodori-v4-pointwise-selector-screen-20260819-attempt2`

## 最終consumer境界

### WmHeadからF32へ直接store

F16 WmHeadがF32 consumer tensorへ直接書く候補を実装した。最初の版は最終F16丸めを省いたため
hashが変わり、同一意味論比較として失格とした。修正版は
`f32(f16(tanh(accumulator)))`をstoreしてcontrolとのbitwise一致を回復した。

修正版のcandidate-control device paired中央値は
`+0.021 / +0.012 / -0.026 / -0.030 / +0.033 ms`で中立だった。独立cast dispatchを
消してもF32 store帯域が相殺し、productionへは採用しない。

- invalid precision boundary:
  `/home/sanzentyo/benchmark-artifacts/irodori-v4-f32-consumer-head-20260819-attempt1`
- corrected bitwise comparison:
  `/home/sanzentyo/benchmark-artifacts/irodori-v4-f32-consumer-head-20260819-attempt2`

### F16を直接CPUへreadback

最終F16をreadbackしてCPUでF32へ変換する候補も測定した。device-completeだけでなく、96,000
sampleのCPU変換を含むreadback-completeを主判定にした。candidate-control readback paired中央値は
`-0.144 / +0.147 / +0.028 / +0.070 / +0.136 ms`で4/5 sessionが悪化した。
bitwise一致、WGPU error 0だが不採用である。

artifact:
`/home/sanzentyo/benchmark-artifacts/irodori-v4-cpu-f16-consumer-20260819-attempt1`

## long-lived sessionへのsoftware graph接続

codec software graphはbench専用のborrowed objectだけでなく、decoder weightと複数の固定shape graphを
同じownerへ保持する`CapturedDacVaeDecoder`として扱えるようにした。graph mapをdecoderより先にdropする
field order、`&mut self`によるstable input更新・replay・readbackの直列化、warmup manifest外shapeの
capture前拒否を型とAPIの契約にしている。`OnlineSession<SessionReady>`は明示したgeometryだけをcaptureした
`CapturedOnlineSession`へconsumeでき、RF latentをCPUへ戻さず最終contiguous F32 audioだけを返す。

複数shape captureではpriming runが通常allocatorへ残した未使用pageを各capture後にcleanupする。graph固有arena、
reusable output、stable input、decoder weightは保持するため、steady graphを壊さず通常poolとの二重reservedを避ける。
実装commitは`14793242a128b07f20f8a5e7395bc4684c72964a`である。

その後、112-frame strict-FP32 all-resident条件で5 fresh sessionのeager/captured比較を行った。
capturedはcodec device-completeで+1.95%、同一CPU readback境界で+2.00%遅く、既定経路には
採用しなかった。graph allocatorのoversize bucket不具合は修正したが、capture中央値295 msと
約600.5 MiBのallocator reserved増分も残る。詳細とauthoritative artifactは
[`rtx-5070ti-v4-captured-all-resident-2026-08-19.md`](rtx-5070ti-v4-captured-all-resident-2026-08-19.md)
を参照する。

RF最終出力をstable inputへ直接書く案も再点検した。50-frame latentはF16で3,200 bytesに過ぎず、直接書込には
productionの5演算CFG/Euler経路をcustom output bindingへ置換する必要がある。この融合自体は既存campaignで
bitwise同値ながらdevice/readbackともneutralだったため、copyだけを消すためのsolver分岐拡大は行わない。

## NHWC residual-state channel mapping screen

旧NHWC residual-state候補が遅かった原因であるpointwise lane mappingを変更し、同じT64/O96/K32演算で
`workgroup_size(32 time, 8 channel-group)`を`(8 channel-group, 32 time)`へ入れ替えた。後者は同一timeの
96 channelを8 lane cohortが連続storeするが、input/weight tileとFMA順序は変えない設計だった。

1 fresh screenでは候補のdevice中央値が`17.653 ms`、同一binary production controlが`13.939 ms`で、候補は
約26.6%遅かった。readbackも`18.560 ms`対`14.811 ms`だった。accuracy gate、finite、WGPU error 0は通ったが、
candidate hash `fecb5da6…`はcontrol `04daa965…`と一致しなかった。store coalescingよりもtime cohortごとの
input/weight利用効率低下が大きい。差が明確なため追加fresh sessionは行わず、candidate sourceは戻した。

- invalid configuration（decoder-only checkpointをfull loaderへ渡しdispatch前fail-closed、値は不使用）:
  `/home/sanzentyo/benchmark-artifacts/irodori-v4-nhwc-channel-mapping-20260819-attempt1`
- valid screen、raw log、100 ms NVML、source/binary/model/fixture pin、`SHA256SUMS`:
  `/home/sanzentyo/benchmark-artifacts/irodori-v4-nhwc-channel-mapping-20260819-attempt2`

valid screenのNVML peakはcandidate 4,260 MiB、control 4,252 MiBだった。旧NHWC campaignや`/tmp`の値は
今回へpoolしていない。

## 次の優先順位

小さいselector、最終cast/readback境界は今回で打ち止めとする。次はパラメータ調整より先に、
以下の構造を優先する。

1. k7 haloを通常のaffine `StridedStageMemory`へ全面scatterしないため、non-affine
   `(m, channel, kernel) -> (input_time, channel)` viewをMMA readerが直接消費できるstage contractを
   CubeKへ追加する。過去のchannel-major scalar loaderは再利用しない。
2. ConvTranspose/NHWCは単なるdual-outputやlane入替を繰り返さず、raw shortcutのwrite/read自体を消せる
   producer-consumer fusionだけを再検討する。
3. 構造変更後にaccuracy-approved selector manifestを作り直す。

同じ長さ・同じCFG topologyは将来のtensor micro-batch候補として維持する。
