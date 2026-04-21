# User Input — 2026-07-21 (Mac Metal Continuation)

## Message 1 (session handoff)
> 現在、別デバイスでやられていたセッションをあなたは引き継ぐ必要があります。想定通りにモデルなどを配置したうえで、セッションを継続しなさい。a6000のubuntuからm5 mac miniになっています。作業の方針の決定や作業の進捗、benchの結果などが出るたびに細かく、逐次的なdocsのreadやupdate(sync)、必要とされるrust,python,justfileのskillsのload,rubber duckの呼び出しをしなさい。uvなどは入っていますが、足りないものは適宜インストールして作業を進めなさい。現在のtaskはwgpu kernelによるwgpu backendの最適化のはずです。とりあえず、サポートされるbackend(CPUを除く)に対するbenchやtestなどを回しつつ、問題がないことを確認し、あったら修正を加えつつ、作業を自律的に進めていきなさい

**Context**: Session handoff from RTX A6000 Ubuntu → Apple M4 Pro Mac Mini. Task is WGPU kernel optimization continuation.

## Message 2 (pull from other hardware)
> 別のハードウェアで更新が行われました。masterで作業をし、最初にpullしてから作業を継続してください

**Context**: Another device (RTX 5070 Ti Windows) pushed 15 new commits. Pulled and continued from updated master.

## Work Done This Session (Metal Micro-Benchmarks)
- git pull: integrated 15 commits from Windows session (5 kernels, 6 benchmark binaries, WGSL docs)
- Build clean on macOS arm64 / Metal
- Tests: 223 passed, 0 failed
- bench_rmsnorm: Metal 1.58× (dim=1024), 0.90× (dim=64)
- bench_fused_adaln: Metal 1.77-2.90×, ~130ms savings/inference
- bench_tiled_fa: All custom FA variants 1.82-2.84× SLOWER than burn on Metal
- bench_fused_sdpa: Row-streaming 0.17× burn (5.9× slower)
- Subgroup diagnostic: enable subgroups; causes all-zero output on Metal (same naga bug)
- Full model: Wgpu f32=35,745ms, Wgpu f16=18,155ms (RTF 0.61), WgpuRaw=36,451ms
- Fix: WgpuDevice::DefaultDevice for gpu_id=0 (was DiscreteGpu which panics on Apple Silicon)
- Created docs/benchmarks/m4-pro.md with all results
