# RTX 5070 Ti Laptop: ConvTranspose finalizer / Snake融合screen（2026-08-19）

## 結論

cached-col2im ConvTranspose finalizerから最初のresidual unitへ渡る境界を対象に、raw NCLと
post-storage-cast Snake済みNHWCを一つのdispatchで生成する`GlobalStoreTransform`候補を実装した。
F16/F32、shape、device、binding、buffer sizeをlaunch前に検査する型付きpairであり、未対応のcase 0と
契約不一致は従来経路へfail closedする。

演算量とdispatchは減るが、全case融合は50 latent framesでproductionより遅かった。NCL順の単純実装は
NHWC storeがstridedになり、16×16 shared-memory transposeで両出力をcoalescedにした版も、barrierと
dual-output drainのcostを回収できなかった。case別screenではcase 1/2が明確に不利で、最も近いcase 3も
5 fresh sessionでcontrolに負けた。

したがってproductionへ昇格しない。既存の
`coalesced col2im finalizer -> tiled Snake + NCHW-to-NHWC transpose`を維持する。候補は`profile` featureの
明示selectorからのみ利用し、今後CubeCL/WGPUのbackend変更時に同一binaryで再測定できるようにする。

## dataflowと数値境界

controlは次の2 dispatchである。

```text
columns -> col2im + bias -> F16 raw NCL
F16 raw NCL -> F32 Snake -> F16 activated NHWC
```

候補は次を1 dispatchで生成する。

```text
columns -> col2im + bias -> F16 raw NCL
                         -> F16 round -> F32 Snake -> F16 activated NHWC
```

F16 rawへ丸めた値をF32へ戻してSnakeを評価するため、storage境界を除去しても演算順序は変えない。
candidate/controlの最終waveform SHA-256は全測定で
`113ba560546d82a3112332ac67b3cea5d5b83b407109d3df3817e5b82b609e05`に一致した。

## 単純版を不採用にした理由

最初のkernelはNCL順で連続laneを割り当てた。raw NCL storeとcolumns readはcoalescedになるが、activated
NHWC storeはchannel strideとなる。5 warmup + 10 measuredの単独screenはcandidate
`15.668 / 16.460 ms`、同じbinaryのcontrol `15.090 / 16.099 ms`（device/readback）で、candidateが
明確に遅かった。この値は採用campaignへpoolしない。

次に16×16のpadded workgroup tileを導入した。各laneはcolumnsをtime方向に読み、raw NCLをcoalesced
storeする。Snake結果をshared memoryへ置き、barrier後に転置してactivated NHWCもcoalesced storeする。
単純版より改善したが、全case融合は`15.375 / 16.222 ms`でcontrolを上回らなかった。

## case別screen

同じtiled binaryでcase 1、2、3を個別に有効化した。各条件は5 warmup + 10 measured、accuracy gate、
WGPU uncaptured error 0を通した。

| fused finalizer | device median ms | readback median ms | 判断 |
|---|---:|---:|---|
| case 1: 768→384 | 15.387 | 16.233 | reject |
| case 2: 384→192 | 15.512 | 16.458 | reject |
| case 3: 192→96 | 15.185 | 15.959 | fresh-session再確認へ |
| all cases | 15.375 | 16.222 | reject |

case 3は単発screenが近かったため、同じ最終binaryを使い、candidate/controlを交互に5 fresh process、
各5 warmup + 20 measuredで再確認した。automatic retryは0である。

| session | case 3 device ms | control device ms | case 3 readback ms | control readback ms |
|---:|---:|---:|---:|---:|
| 1 | 15.273 | 15.269 | 16.074 | 16.094 |
| 2 | 15.127 | 15.175 | 16.139 | 15.910 |
| 3 | 15.434 | 15.133 | 16.235 | 16.103 |
| 4 | 15.333 | 15.297 | 16.226 | 16.188 |
| 5 | 15.317 | 15.212 | 16.127 | 16.005 |
| median | **15.317** | **15.212** | **16.139** | **16.094** |

deviceは5組中4組、readbackは5組中4組でcandidateが遅い。追加dispatchを消すこと自体は目的にせず、
end-to-end latencyを維持するproduction条件に従って不採用とする。

## 解釈

既存のSnake kernelはactivationとlayout transposeを一つのtiled dispatchで処理している。今回の候補は
rawとactivatedを同じproducerから生成するため、global raw readと1 dispatchを除去できる一方、finalizer
へworkgroup barrier、shared-memory traffic、二つ目のstore streamを追加する。今回のgeometryでは、既存の
独立したcoalesced producer/consumerの方が速い。

これはmulti-output Fusion一般を否定する結果ではない。raw側の後続consumerまでNHWCで維持できれば、
dual-layout出力自体を不要にできる。ただしそれにはresidual shortcut、pointwise residual、次block境界を
まとめてlayout-polymorphicにする設計変更が必要で、今回の小さな境界融合とは別campaignにする。

## portabilityとartifact

kernelはWGSLのF16/F32、workgroup memory、barrierだけを使い、Vulkan/Metal/DX12で共有可能なsource設計で
ある。実測したのはNVIDIA/Vulkanだけであり、他backendのcompile/性能は未検証である。GPU固有の名前や
tile selectorでproduction分岐していない。

- output: `/home/sanzentyo/benchmark-artifacts/irodori-v4-convtranspose-snake-fusion-20260819-attempt1`
- source base: `1f9636a3df95c2ccaa96df8e6c5ed56af92abeac`
- profile implementation commit: `5e42498`
- profiler binary SHA-256: `153a8d62efc69157c7bca1b6029e21dc69a8779ccb7f9a5e6909bb3f8f69fbe2`
- GPU: NVIDIA GeForce RTX 5070 Ti Laptop GPU、12,227 MiB
- driver: `595.71.05`
- Vulkan adapter: 0
- CUDA/NVML index: 0
- PCI bus ID: `00000000:01:00.0`
- codec SHA-256: `b14a25da5d68bf779d8c44a40706ddc7c4819cb60489b2a80a6408ca03c514fb`
- oracle SHA-256: `08663e52df6c63eea7ebf5ae2ba35778bdb5bc7d20cd06e430e855a86174229e`

旧`/tmp` artifactと旧session値は今回の集計へpoolしていない。
