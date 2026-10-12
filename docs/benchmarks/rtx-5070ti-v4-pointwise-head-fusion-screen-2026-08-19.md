# RTX 5070 Ti Laptop: final pointwise / WmHead fusion screen (2026-08-19)

## 結論

decoder block 3終端のpointwise residualと、既に一dispatch化されている
`Snake -> Conv1d(96 -> 1, k=7) -> tanh`を一つのconsumer kernelへ統合したが、
**productionへ採用しない**。

候補は96-channelの巨大なfinal residual stateをglobal memoryへwrite/readせず、64 time sampleと
左右3 sampleのhaloを一workgroupで完結させる。これはdispatch削減だけでなくintermediate lifetimeを
除去する本質的なproducer/consumer fusionだが、96x96 pointwise projectionを既存の高速経路から
scalar F32 FMAへ移すcostを回収できなかった。

## 同一process screen

- F16 storage、F32 accumulation/Snake
- 50 latent frames、96,000 waveform samples
- 5 warmup + ABBA/BAAB 10 block
- candidate/controlは同一binary・同一model
- automatic retry 0、WGPU uncaptured error 0

| boundary | candidate median | production median |
|---|---:|---:|
| device-complete | 19.006820 ms | 14.093218 ms |
| readback-complete | 19.703846 ms | 14.883019 ms |

block内のcandidate-production device差は中央値`+5.224788 ms`で、candidateが速かったblockは
0/10だった。単発差として十分大きいため5 fresh process採用campaignへ進めない。

候補はSNR `56.609249 dB`、max abs `3.417968750e-3`、cosine
`0.999998913322`でaccuracy gateを通過した。waveform SHA-256はcandidate
`1191a0d2cde21ca22afe061c97314032159a00c5396d91e1928b8ce52d88bad5`、production
`04daa96513fe33c680bc0ca475b2182936074a4578312a76f3dfab821f49cc38`である。

## 設計上の知見

一workgroupはpointwise input/weight、post-cast Snake state、WmHead weightを47,040 bytesの
workgroup memoryへ保持する。これにより中間global trafficは消えるが、workgroup間barrierが使えない
ため各time tileがpointwise projection全体を独立に計算する必要がある。既存pointwise/CubeK経路の
matrix accelerationを維持しつつ、その全channel結果を次のtime convolutionへ直接渡すportableな
WGPU primitiveは現在ない。

従って「巨大kernelなら速い」という方向は止める。再検討条件は、subgroup matrix resultを
workgroup-wide producer/consumer tileへ直接公開できる汎用CubeK store/continuation APIが得られた
場合に限る。profile-only候補は同一binaryで将来runtimeを再評価できるよう保持する。

## pinとartifact

- source base: `70c860e`
- profiler binary SHA-256: `a6d3335aa539f85ad3e5ae35700dfcadf65047e84f4df005070b1271186e03a7`
- fixture SHA-256: `08663e52df6c63eea7ebf5ae2ba35778bdb5bc7d20cd06e430e855a86174229e`
- codec SHA-256: `b14a25da5d68bf779d8c44a40706ddc7c4819cb60489b2a80a6408ca03c514fb`
- GPU: NVIDIA GeForce RTX 5070 Ti Laptop GPU、driver 595.71.05、Vulkan adapter 0、12,227 MiB
- fresh artifact: `/home/sanzentyo/benchmark-artifacts/irodori-v4-pointwise-head-fusion-20260819-attempt1`

旧`/tmp`、旧campaign、別shapeのsampleはpoolしていない。

