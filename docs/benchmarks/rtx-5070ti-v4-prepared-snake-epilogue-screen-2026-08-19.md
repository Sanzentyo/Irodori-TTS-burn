# RTX 5070 Ti Laptop: prepared Snake epilogue screen (2026-08-19)

## 結論

各channelの `1 / (alpha + eps)` をmodel preparation時にF32で一度だけ計算し、
CubeK post-cast Snake epilogueのper-output除算を乗算へ置換する候補は、
**productionへ昇格しない**。

5 fresh processの同一binary ABBA/BAAB比較ではdevice-complete差の中央値が
`-0.034683 ms`だったが、改善は3/5 sessionに留まり、差は現在のlaptop GPUの
clock/power noise floorに近い。readback-complete差の中央値も`-0.011186 ms`である。
数値結果は全候補・controlでbitwise一致したためcorrectness rejectionではなく、
効果量不足によるperformance rejectionである。

## 条件

- source: `bc4a307` (`perf(codec): screen prepared Snake reciprocals`)
- GPU: NVIDIA GeForce RTX 5070 Ti Laptop GPU
- driver: 595.71.05
- WGPU adapter: Vulkan adapter index 0
- precision: F16 storage、F32 Snake arithmetic
- shape: 50 latent frames、96,000 waveform samples
- 5 fresh paired sessions
- 各session: 5 warmup + 10 ABBA/BAAB blocks
- scalar-only VRAM control: 5 fresh sessions
- automatic retry: 0
- 過去campaignのsample pooling: なし

fresh evidence directory:

```text
/home/sanzentyo/benchmark-artifacts/
  irodori-v4-prepared-snake-epilogue-20260819-attempt1/
```

`SHA256SUMS`自身のSHA-256は次である。

```text
fa719fe56d52767123bdbfd086cc9953800fd12a2d878fe206b4576a037454b6
```

## fresh-session結果

差は `prepared - scalar` で、負がprepared候補に有利である。

| session | device Δ ms | readback Δ ms | prepared device ms | scalar device ms |
|---:|---:|---:|---:|---:|
| 1 | -0.034683 | +0.948298 | 15.547335 | 15.582018 |
| 2 | -0.043107 | -0.111305 | 15.353883 | 15.396990 |
| 3 | +0.020954 | -0.011186 | 15.468551 | 15.447597 |
| 4 | -0.229895 | -0.209867 | 15.427281 | 15.657176 |
| 5 | +0.072266 | +0.078760 | 15.589509 | 15.517243 |
| median | **-0.034683** | **-0.011186** | 15.468551 | 15.517243 |

paired processのNVML peak中央値は1,187 MiB、scalar-only processは1,192 MiBだった。
prepared parameterは小さいため増加はNVMLの1 MiB sampling粒度では検出されず、
この5 MiB差は節約効果ではなくprocess間noiseとして扱う。

## accuracy

全5 sessionでprepared/scalarとも次のwaveform hashとなり、route間でもbitwise一致した。

```text
113ba560546d82a3112332ac67b3cea5d5b83b407109d3df3817e5b82b609e05
```

fixture比較も全runで同一だった。

- max abs: `3.417968750e-3`
- RMSE: `2.139710145e-4`
- SNR: `56.074203 dB`
- cosine: `0.999998775055`
- uncaptured WGPU errors: 0

F16 outputへstoreされる前にreciprocalをF32へ丸めても、このfixtureでは最終F16値が
変化しなかった。

## 実装判断

`PreparedSnakeEpilogue`、interleaved `[alpha, reciprocal]` cache、汎用paired-plan
harnessは `profile` feature内のdifferential routeとして保持する。productionの
`AccuracyApproved`はscalar division routeのままとし、通常buildのmodel memoryと
実行graphは変更しない。

先行したnative vector Snake epilogueの一時screenもbitwise一致したが、device中央値は
scalarより約`0.021 ms`遅かった。この `/tmp` screenは正式campaignへpoolせず、vector
assemblyとWGSL backend上のtranscendental scalarizationが相殺する可能性を示す探索結果
としてのみ記録する。汎用writerへ追加した分岐はscreen後に撤回済みである。

次の本質的候補はRFのtext-only Independent CFGで現在別々になっている、

```text
v_cond - v_uncond
scale multiply
v_cond add
Euler dt multiply
x_t add
```

を、既存のdispatchごとのF16丸め位置を明示的に維持したportable WGSL
`cfg + Euler update`へ統合することである。1 Euler step当たり5 elementwise dispatchを
1 dispatchへ減らせるため、Snake除算単体より効果幅が大きい。
