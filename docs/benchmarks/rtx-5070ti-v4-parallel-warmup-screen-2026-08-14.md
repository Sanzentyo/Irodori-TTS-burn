# RTX 5070 Ti Laptop: RF / codec parallel DryRun screen（2026-08-14）

## 結論

同一WGPU device上でRFとcodecのcompile-only DryRunを2 threadへ分けても、fixed 112-frame
strict FP32のstartup wall短縮は中央値`0.433 s`（`1.42%`）に留まった。並列中はRFとcodecの
両branchが大きく遅くなり、DryRun直後の最初の実requestも中央値`0.371 s`で、逐次の
`0.197 s`より`88.5%`遅かった。追加validationでは並列後のfirst requestだけ
`0.370 s`、second requestは`0.201 s`となり、readiness完了後へ遅延costを残した。

波形hashもfresh process間でsingletonにならず、既知hash
`5c22e03be6864d320a7881939b318d0d066b06af3005942457a7dc7e1e43c8b9`に加えて
`ae6510616fb18a0acb2bd27ff62df52fa4235f9802d815553b64add4cb8f40b2`を観測した。
後者のoracle accuracyはこのscreenでは算出していない。wall差が小さく、first-request latencyと
cross-process determinismのgateを満たさないため、単純なRF/codec 2-thread DryRunは不採用とし、
実験用CLI分岐はproduction sourceから戻した。

## 条件

- source HEAD: `96037a2`
- GPU: NVIDIA GeForce RTX 5070 Ti Laptop GPU、12,227 MiB
- driver: `595.71.05`
- runtime: Vulkan WGPU、strict FP32、TF32 off、autocast off
- fixture: 4.48秒 / 112 latent frames、unconditioned voice
- RF work: 4 Euler evaluations、forward batches `[2,2,1,1]`
- model/codec: all-resident、RF/codec checkpointは並列load
- warmup work: 同じprepared RF request 1件とcodec 112-frame shape 1件
- CubeCL: 全processで同じ既存autotune bundleをimport
- isolation: processごとに新しいCubeCL cache directoryと`XDG_CACHE_HOME`
- aggregation: 各route 5 fresh process、順序`S,P,P,S,S,P,P,S,S,P`
- automatic retry: 0

実験binaryのSHA-256は
`b084232303164928048a99fd5f30c2334200d46b71565c747835ec1e2cab4320`。

## 5-process結果

| route | DryRun wall median (range) | RF branch median | codec branch median | first real request median (range) |
|---|---:|---:|---:|---:|
| sequential | **30.477 s** (30.082–31.169) | **23.477 s** | **6.951 s** | **0.197 s** (0.193–0.197) |
| parallel RF/codec | 30.044 s (29.998–30.685) | 30.035 s | 19.862 s | 0.371 s (0.316–0.382) |

並列routeのwallは逐次より`0.433 s`短いが、RF branchは`27.9%`、codec branchは
`185.7%`遅い。二つのcompile streamは独立にCPU compileを進めるより、共有WGPU server、
driver pipeline生成、autotune実行、allocator cleanupで強く競合していると解釈するのが妥当である。
この内訳は競合の存在を示すが、server/driver別のtraceを採っていないため時間を各原因へ配分しない。

main campaignのhashはsequentialが既知4 process / alternate 1 process、parallelが既知5 processだった。
別の2-request validationではsequentialが2回とも既知、parallelが2回ともalternateとなった。
したがって「parallelなら常にalternate」ではなく、fresh environmentでcache/algorithm selectionが
単一hashへ収束しないことだけを確認済み事実とする。

## 追加validation

| route | DryRun wall | request 1 | request 2 | process内hash |
|---|---:|---:|---:|---|
| sequential | 30.041 s | 0.192 s | 0.202 s | 既知hashで一致 |
| parallel RF/codec | 30.240 s | 0.370 s | 0.201 s | alternate hashで一致 |

parallel routeはsecond requestで通常範囲へ戻るため、compile-only pass終了後のdevice syncだけでは
逐次routeと同じfirst-request readinessを作れていない。service ready前にreal validationを置けば
user requestへの漏出は防げるが、そのreal validationが約`0.17 s`の残差を支払うため、DryRun wallの
`0.43 s`短縮だけをstartup利得として数えることはできない。

## artifactと判断範囲

- 5-process A/B:
  `/home/sanzentyo/benchmark-artifacts/irodori-v4-parallel-warmup-ab-QY5HKYrj`
  (`SHA256SUMS`: `e454efdd3976952bd2ce8b79e1d95f4396a7ac694c84e2347d67ac79d10604be`)
- 2-request validation:
  `/home/sanzentyo/benchmark-artifacts/irodori-v4-parallel-warmup-validation-8jVCoONZ`
  (`SHA256SUMS`: `8dba4a04c0e35a72cc2622e15a5012ce0bc47dc18e5759ef28066e5d76cad901`)
- initial smoke:
  `/home/sanzentyo/benchmark-artifacts/irodori-v4-parallel-warmup-smoke-uplZJG13`
  (`SHA256SUMS`: `3ab400aaf2db4c29e6ada20214f106f1afad10793ff28c30e2c239060d1223e2`)

これはfixed112 strict FP32で単純な同一device 2-thread化を落とすscreenであり、F16六長manifest、
Metal/DX12、別device、WGPU pipeline cache restoreの性能主張には拡張しない。別deviceで作った
process-local pipeline objectはproduction deviceへ移せないため、現行contractの代替にはならない。
次にstartupを短縮する場合はcompile並列化ではなく、manifest内の重複pipeline identityをtraceして
同じkeyのDryRun発行を除く方を先に評価する。
