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

## 次の優先順位

小さいselector、最終cast/readback境界は今回で打ち止めとする。次はパラメータ調整より先に、
以下の構造を優先する。

1. graph入力copyをRFのstable output slotと統合し、requestごとのlatent copyを除去する。
2. k7の現行kernel-major vector loadを維持したまま、kernel plane間でhaloを再利用できる
   shared-cache contractを設計する。過去のchannel-major scalar loaderは再利用しない。
3. ConvTranspose finalizerからresidual consumerまでの物理layoutを一本化し、dual-outputの
   追加帯域を発生させずにSnake境界を除去する。
4. 構造変更後にのみ、accuracy-approved selector manifestを作り直す。

同じ長さ・同じCFG topologyは将来のtensor micro-batch候補として維持する。
