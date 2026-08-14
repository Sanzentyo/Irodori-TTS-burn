# RTX 5070 Ti Laptop: k7 geometry-selected multi-row評価（2026-08-14）

## 結論

F16 DACVAE decoderのCubeK k=7 convolutionで、出力行列の形状に応じて
`SimpleArgs.multi_rows`を有効化する。productionの`AccuracyApproved`へ次のselectorを採用した。

```text
multi_rows = output_length >= output_channels && output_channels >= 384
```

この条件はdecoder block 0/1のwide convolutionだけを対象にし、実測で遅かった
block 2/3（192/96 channels）は従来single-rowのままにする。45 latent framesではblock 1だけ、
112 frames以上のcampaignではblock 0/1がmulti-rowになる。

6つのpinned F32 oracle長で同一process ABBA/BAAB比較を行った。対象k=7段のdevice timestampは
全長で短縮し、112〜685 framesでは各10/10 block、45 framesでも9/10 blockで勝った。
全出力はsingle-row controlとbitwise一致し、全oracle accuracy gateを通過、WGPU errorは0だった。

## campaign

- output: `/home/sanzentyo/benchmark-artifacts/irodori-v4-k7-geometry-20260814-attempt1`
- source start: `041acfdf38ebbdf650a3585cd3a384f40db69185`
- GPU: NVIDIA GeForce RTX 5070 Ti Laptop GPU、12,227 MiB
- driver: `595.71.05`
- WGPU: Vulkan discrete adapter 0
- CUDA/NVML index: 0
- PCI bus ID: `00000000:01:00.0`
- execution: F16 storage、F32 accumulator/Snake
- codec SHA-256: `b14a25da5d68bf779d8c44a40706ddc7c4819cb60489b2a80a6408ca03c514fb`
- final F16 oracle SHA-256: `08663e52df6c63eea7ebf5ae2ba35778bdb5bc7d20cd06e430e855a86174229e`
- final profiler binary SHA-256: `f3228e24a886ff39bb1570580b1ba3ad053326c9e39f367c6f51f56d7fc3f9d6`

stage比較は各routeの前にdevice syncし、CubeCL stream timestampで各decoder stageを記録した。
ABBA/BAABの各block内で同route 2 sampleを平均し、candidate minus controlを求めた。
絶対host時間にはtimestamp解決とreadbackが含まれるため、採否は主にblock内device timestamp差で判断した。

## shape selector探索

最初に`multi_rows=true`を全12本へ強制し、decoder blockと入力長を系統的に確認した。

- block 0、768 channels: 出力行数が768未満の45-frameケースでは遅く、それ以上では速い
- block 1、384 channels: 全campaign長で速い
- block 2、192 channels: 全campaign長で遅い
- block 3、96 channels: 全campaign長で遅い

この結果から、frame数やblock番号のhard-codeではなく、GEMM geometryで
`M >= N && N >= 384`を選んだ。profile-onlyにはforced multi-rowとgeometry-selectedの明示route、
single-row controlを残している。

## 6-length device timestamp campaign

各長は2 warmup + 10 ABBA/BAAB blockで、candidate/control各20 sample。下表は
multi-row選択対象であるblock 0/1 k=7段のsample medianである。

| latent frames | geometry-selected | single-row control | delta | improvement |
|---:|---:|---:|---:|---:|
| 45 | 1.880 ms | 2.129 ms | -0.249 ms | 11.68% |
| 112 | 3.958 ms | 5.164 ms | -1.206 ms | 23.36% |
| 255 | 8.833 ms | 12.001 ms | -3.168 ms | 26.40% |
| 333 | 12.129 ms | 14.911 ms | -2.782 ms | 18.66% |
| 489 | 15.059 ms | 21.367 ms | -6.308 ms | 29.52% |
| 685 | 21.720 ms | 29.892 ms | -8.173 ms | 27.34% |

block内平均差のmedianは順に`-0.248 / -1.272 / -3.043 / -3.533 / -5.961 /
-8.827 ms`だった。112 frames以上は全10 blockでcandidateが短く、45 framesは9/10だった。

長いcaseではGPU clock/power driftが大きく、全decoder host wall timeだけをpoolすると試行位置に
強く依存した。一方、変更対象6段の同block timestamp差は一貫した。そのためABBA/BAABの
block内差とoperator timestampを採用根拠にし、pooled host medianだけでは判断していない。

## final promoted-route comparison

productionへ昇格後、50-frame F16 oracleで3 warmup + 20 ABBA/BAAB block、各route 40 sampleを
最終計測した。対照は明示的なCubeK single-row routeである。

| boundary | geometry-selected | single-row control | pooled median delta |
|---|---:|---:|---:|
| selected block 0/1 k=7 device timestamp | 2.214 ms | 2.591 ms | -0.377 ms |
| all 12 k=7 device timestamps | 7.459 ms | 7.806 ms | -0.346 ms |
| all decoder-stage device timestamps | 14.285 ms | 14.637 ms | -0.351 ms |
| device-complete host wall | 17.136 ms | 17.500 ms | -0.364 ms |
| readback-complete host wall | 17.489 ms | 17.849 ms | -0.360 ms |

より頑健なblock内平均差では、selected k=7がmedian `-0.390 ms`、20/20勝、全stageが
`-0.386 ms`、19/20勝、device-completeが`-0.204 ms`、14/20勝だった。

両routeのwaveform SHA-256は
`113ba560546d82a3112332ac67b3cea5d5b83b407109d3df3817e5b82b609e05`でbitwise一致した。
F16 oracle gateはmax abs `3.417968750e-3`、mean abs `8.517037084e-5`、
RMSE `2.139710145e-4`、SNR `56.074203 dB`、cosine `0.999998775055`で通過した。

## accuracy

F32 oracle campaignの45/112/255/333/489/685 framesすべてで、candidateとcontrolのhashは一致した。
最大長685では1,315,200 waveform samplesを比較し、max abs `1.855909824e-3`、
RMSE `8.149367973e-5`、SNR `59.883002 dB`、cosine `0.999999487097`だった。
各campaignは`wgpu_uncaptured_errors=0`で完了した。

## verification

- `cargo test --lib --features inference,codec`: 508 passed、0 failed、17 ignored
- geometry-selected k=7 Vulkan bitwise route test: 1 passed
- `cargo clippy --all-targets --features inference,codec,cli,profile -- -D warnings`: pass
- `cargo fmt --all -- --check`: pass
- `uvx ruff check scripts`: pass

production behaviorはF16の`AccuracyApproved`だけを変更する。F32 packed-residue routeは不変で、
明示`CubeClImplicitGemm`は測定用single-row controlとして従来挙動を維持する。
