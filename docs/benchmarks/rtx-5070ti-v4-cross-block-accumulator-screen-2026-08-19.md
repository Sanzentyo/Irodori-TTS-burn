# RTX 5070 Ti Laptop: CubeK block-boundary accumulator store screen (2026-08-19)

## 結論

採用済みのC384/C192 decoder block境界は、pointwise projection、bias、residual、
post-storage-cast Snakeを手書きWGSL 1 dispatchで実行している。このprojection coreを
CubeK CMMAへ置き換え、中間Tensorとdispatch数を変えずに高速化する候補をscreenしたが、
**productionへ採用しない**。

50 latent framesの同一process ABBA/BAAB 10 blockでは、候補のdevice-complete中央値が
`14.090521 ms`、controlが`13.962290 ms`だった。block内のpaired差は候補-controlで
中央値`+0.427855 ms`、候補が速かったのは4/10 blockだけである。

## 実装した構造

profile-onlyの`F16ResidualPostCastSnakeStore`はCubeKのF32 accumulatorへF16 shortcutを
加算し、従来どおり一度F16へ丸め、その値をF32へpromoteしてSnakeを評価する。出力は
次blockが必要とするactivated stateだけであり、raw intermediateをallocateしない。
dtype、device、contiguous storage、長さをtyped launch argumentsで検査し、contract missは
測定前にerrorとして閉じる。

演算は1 dispatchだが、CubeKの論理出力はNHWC、consumerが必要とする物理出力はNCLである。
zero-copy stride viewへ直接storeするためchannel軸のvector storeが弱まり、CMMA projectionの
利点を相殺した。先にrejectしたblock-final raw storeと同じ構造上の制約であり、tile値の調整で
先に解く問題ではない。

## 精度

候補はaccuracy gateを通過し、WGPU uncaptured errorは0だった。

- candidate: SNR `56.626626 dB`、max abs `3.662109375e-3`、cosine `0.999998917164`
- control: SNR `56.622776 dB`、max abs `3.417968750e-3`、cosine `0.999998916470`
- candidate hash: `b128653b557153dfadea20327ea4f21e8bcd1a63f913993f2c5bd33d34f0d348`
- control hash: `04daa96513fe33c680bc0ca475b2182936074a4578312a76f3dfab821f49cc38`

hash差はCubeKと手書きprojectionのaccumulation order差であり、post-storage-cast境界の欠落ではない。

## 条件とartifact

- GPU: NVIDIA GeForce RTX 5070 Ti Laptop GPU、12,227 MiB
- driver: 595.71.05
- WGPU: Vulkan discrete adapter index 0
- precision: F16 storage、F32 accumulation/Snake
- warmup 5、measured ABBA/BAAB 10 block、automatic retry 0
- binary SHA-256: `ca46903f2b6b4aa85fb654d61e2b7f2502d099f2c24d737c31c6f54fba7fb43d`
- fixture SHA-256: `08663e52df6c63eea7ebf5ae2ba35778bdb5bc7d20cd06e430e855a86174229e`
- codec SHA-256: `b14a25da5d68bf779d8c44a40706ddc7c4819cb60489b2a80a6408ca03c514fb`
- fresh artifact: `/home/sanzentyo/benchmark-artifacts/irodori-v4-cross-block-accumulator-screen-20260819-attempt1`

旧`/tmp`や別campaignのsampleはpoolしていない。このcandidateはprofile featureのdifferential
routeとしてのみ保持し、production defaultは採用済みdirect WGSL経路のままとする。

## 次の設計判断

同じNCL stride viewへscalar storeする別のCubeK epilogueは繰り返さない。再検討条件は、
CubeK writerがtransform固有の物理store layoutを所有し、NCLの連続time軸へvector storeできる
ようになった場合に限る。次は最大支配項であるk7+Snakeについて、MMA coreの前後ではなく
入力haloの再利用とglobal read量を構造的に減らせるかを調べる。
