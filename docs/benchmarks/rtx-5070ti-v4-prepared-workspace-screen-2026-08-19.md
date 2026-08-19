# RTX 5070 Ti Laptop: pointwise prepared workspace screen（2026-08-19）

## 結論

decoderのC384/C192/C96 residual unitについて、pointwise出力をrequestごとにallocatorから取得せず、
固定shapeの3-slot workspaceへ直接書く構造を実装してscreenした。出力はproduction controlと全sampleで
bitwise一致し、WGPU uncaptured errorも0だった。しかし同一processのABBA/BAAB block差では
device-completeを短縮せず、F16・50 latent framesで52.734 MiBのpersistent bufferを追加する。
したがってproductionには採用せず、実験用実装も残さない。

この結果は、現在の約15 ms decodeではpointwise出力bufferの取得がGPU device時間の支配項ではないことを
示す。caller-owned arenaを再検討する場合は、pointwiseだけでなくk7、ConvTranspose、最終headを含む
liveness解析済みのdecoder全体arenaとして設計し、persistent VRAMとの交換条件を先に定義する。

## 候補の契約

候補は最大出力である`[1, 96, latent_frames * 1920]`を収容するF16 bufferを3本確保した。
各viewはsame device、same dtype、contiguous NCL/NHWC、必要byte数をdispatch前に検証した。

```text
res0 -> raw slot0 + activation slot1
res1 -> raw slot1 + activation slot2
res2 -> final slot0
```

mutable workspace borrowで同一session内の同時request構築を禁止し、同一WGPU queueのsubmit順序により
consumer完了前のslot上書きを防ぐ。GPU名、tile値、Vulkan固有APIには依存しないためsource設計は
Metal/DX12へ移植可能だが、実測したのはNVIDIA/Vulkanだけである。

## 同一process paired結果

各fresh processでcandidate/controlを5回ずつwarmupし、4 blockをABBA/BAAB交互順序で測った。
各routeは1 processあたり8 samples。時間境界は両routeで同一である。

| session | workspace device中央値 ms | control device中央値 ms | workspace enqueue中央値 ms | control enqueue中央値 ms | block device差中央値 ms |
|---:|---:|---:|---:|---:|---:|
| 1 | 15.103 | 15.123 | 0.486 | 0.502 | +0.028 |
| 2 | 15.168 | 15.160 | 0.489 | 0.521 | +0.013 |
| 3 | 15.300 | 15.212 | 0.456 | 0.482 | +0.028 |
| 4 | 15.107 | 15.216 | 0.497 | 0.578 | -0.005 |
| 5 | 15.126 | 15.277 | 0.478 | 0.549 | +0.249 |
| session中央値 | **15.126** | **15.212** | **0.486** | **0.521** | **+0.028** |

route別sample中央値だけならworkspaceはdeviceで0.086 ms短いが、途中に15→19 ms級のclock/power変動があり、
この集計は順序biasを残す。linear driftを相殺するblock内平均の候補−control差は中央値`+0.028 ms`で、
workspaceのdevice改善を支持しない。enqueueは`-0.036 ms`だが絶対量が小さく、52.734 MiBの常駐追加を
正当化しない。readback-completeも安定した改善を示さなかった。

全candidate/control waveform SHA-256は
`113ba560546d82a3112332ac67b3cea5d5b83b407109d3df3817e5b82b609e05`でbitwise一致した。
F16 oracleに対してmax abs `3.41796875e-3`、RMSE `2.139710145e-4`、SNR `56.074203 dB`、
cosine `0.999998775055`である。

## 付随して得た実装知見

- main threadのdecode graph enqueueはproduction controlで約0.52 msであり、15 ms級device時間の主因ではない。
- `RuntimeOptions.tasks_max`の32/64/128/256単発screenにも安定改善はなく、default 32を維持する。
- process-local bind-group cacheの小規模screenはdeviceで約0.4%の信号に留まり、readbackは悪化したため採用しない。
- output arenaはallocation回数を減らせても、allocatorがbuffer reuse済みならGPU workとmemory trafficは減らない。

profile CLIには今後の調査用として`--tasks-max`、`--profile-repeats 0`、
`decode_enqueue_complete_ms`だけを残す。workspace実装は誤ってproductionへ昇格しないよう削除した。

## pinとartifact

- source base: `6995018b3a2fc559bb2e82d1a4d63f96c58a118f`
- profiler binary SHA-256: `fd089f7bc9a7d8eb2cc6246cd05bdae304b2e52f5a125c62b4e9fb24f7307583`
- F16 oracle SHA-256: `08663e52df6c63eea7ebf5ae2ba35778bdb5bc7d20cd06e430e855a86174229e`
- codec SHA-256: `b14a25da5d68bf779d8c44a40706ddc7c4819cb60489b2a80a6408ca03c514fb`
- GPU: NVIDIA GeForce RTX 5070 Ti Laptop GPU、driver `595.71.05`、Vulkan adapter 0、
  CUDA/NVML index 0、PCI `00000000:01:00.0`、VRAM `12,227 MiB`
- artifact: `/home/sanzentyo/benchmark-artifacts/irodori-v4-prepared-workspace-20260819-attempt1`

artifactには実行済みcandidate binary、5 fresh session raw log、environment記録、`SHA256SUMS`を保存した。
候補sourceは不採用のためtreeへ残しておらず、source baseに対する一時差分は再利用対象ではない。
旧`/tmp` artifactや別campaignの測定値は集計へpoolしていない。
