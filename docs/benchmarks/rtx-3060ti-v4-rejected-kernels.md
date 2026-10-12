# RTX 3060 Ti v4 rejected kernel candidates

This file records candidates that passed numerical checks but lost the aligned
performance gate.  They must not be reintroduced without a materially different
design.

## Block2 C192/L48000 packed-weight residue core

- Date: 2026-08-11
- GPU: NVIDIA GeForce RTX 3060 Ti, Vulkan, PCI `00000000:07:00.0`
- Evidence: `/tmp/irodori-v4-block2-packed-weight-fair-ab-attempt1-20260811`
- Workload attempts: 1; automatic retries: 0
- Shapes: two independent k7 residue calls, dilation 3 and 9,
  `[1,192,48000]`, FP32
- Protocol: 10 warmups, 5 trials x 50 measured single-call samples per
  variant/case, rotating order
- Primary boundary: pre-sync to device-complete sync
- Secondary boundary: the same start through complete CPU readback of all
  9,216,000 FP32 output values
- Accuracy: 18,432,000 compared output values, bit mismatches 0, max abs 0

| Boundary | Current d3+d9 median | Candidate d3+d9 median | Result |
|---|---:|---:|---:|
| Device complete | 13.744 ms | 14.900 ms | 8.4% slower |
| Full CPU readback complete | 70.635 ms | 71.991 ms | 1.9% slower |

The candidate packed OIK weights into `[Cin,K7,Cout]`, changed the core from
WG16x16 to WG32x8, and reduced shared memory per workgroup from 31,104 to
16,768 bytes.  Despite preserving bit-exact output, its subgroup-uniform global
weight reads and output mapping were slower for both dilations.  The production
route was never changed.
