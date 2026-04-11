調べた限り、**Burn の「公式」FlashAttention 実装は、現時点では “低レベル primitive は入っているが、高レベルの `nn` モジュール全体が完全にそれへ乗り切っている段階ではない”** というのがいちばん正確です。最新の安定版は **`v0.20.1`（2026-01-23）** で、attention 周りの大きな拡張は **`v0.21.0-pre.2`** の pre-release 側に多く入っています。([GitHub][1])

まず安定版 `0.20.1` には、公式の低レベル API として **`burn::tensor::module::attention()`** があります。公開 docs ではこれは **scaled dot-product attention** を計算する関数で、引数は `query, key, value, mask`、mask は **4D の bool mask のみ**、さらに **dropout 非対応** で **inference 向け** と明記されています。つまり、安定版で公式に使える attention primitive はあるものの、PyTorch の SDPA に近いフル機能版ではまだありません。([Docs.rs][2])

一方で Burn 側はこれを単なる naive 実装として見ているわけではなく、2026-02-08 の公式 issue では `burn::tensor::module::attention()` を **“backend-dispatched SDPA (enabling flash attention on CubeCL)”** のための正しい抽象化だと説明しています。つまり、**公式の設計思想としては `attention()` を backend ごとの最適化、特に CubeCL 側の flash attention に接続するつもり**で進んでいます。([GitHub][3])

その流れ自体は `v0.20.0` でかなり明確になっていて、公式 release notes には **“Add CubeCL Flash Attention module”** が入り、公式ブログでも新しい Flash Attention 実装が 0.20 の主要モチベーションのひとつで、**CubeCL が backend ごとに命令選択や tile サイズを自動で選んで Flash Attention kernel を生成する** と説明されています。少なくとも **公式実装が Burn 本体 / CubeCL 系に入っている** こと自体は、ここではっきり確認できます。([GitHub][4])

ただし、ここが重要ですが、**高レベルの `burn::nn` モジュールはまだ全面的にその primitive に統一されていません**。`burn-nn 0.20.1` の `MultiHeadAttention` のソースを見ると、forward は `query/key/value` を線形変換したあと、**`query.matmul(key.transpose())` → `softmax` → `weights.matmul(value)`** という手順を明示的に実装しており、`weights` も出力として返しています。つまり、**少なくとも安定版 0.20.1 の `nn::MultiHeadAttention` は、`tensor::module::attention()` を直接ラップする形にはなっていません**。([Docs.rs][5])

同じことは `CrossAttention` にもほぼ当てはまります。`CrossAttention` のソースには **GQA / MQA、quiet_softmax、KV cache** など高レベル機能がありつつ、forward の中身はやはり **score 計算 → softmax → `matmul(v)`** の流れを明示的に書いています。なので、**Burn 公式の attention 系 high-level module は機能豊富だが、stable 版ではまだ flash-attention primitive へ一枚岩では接続されていない**、という理解が妥当です。([Docs.rs][6])

では「どこまで進んでいるのか」を最新版の流れで見ると、かなり前進しています。`v0.21.0-pre.2` の公式 changelog には、attention 関連として **`attention()` に `scale`, `attn_bias`, `softcap`, `is_causal` を追加する変更 (#4476)**、**causal で flash attention を許可する変更 (#4509)**、**attention の default impl をやめて全 backend で実装する変更 (#4544)**、さらに **attention autotune (#4552, #4554)** が並んでいます。これは、**安定版 0.20 系の “最小限 primitive” から、より PyTorch の SDPA に近い汎用・backend-aware 実装へ寄せている途中**だと読めます。([GitHub][1])

ただし、完成度については慎重に見た方がよくて、2026-01-14 に開かれた公式 issue **“[burn-cubecl] FlashAttention doesn't work on most backends”** はまだ open のままで、本文には **Vulkan では incorrect values、CUDA では assertion failure** とあります。つまり、**公式コードベースに FlashAttention 実装はあるが、backend 横断で「もう完全に安定」とまでは言いにくい** 状況です。([GitHub][7])

なので、結論をかなり実務寄りに言うと、今の Burn 公式実装は次の段階です。
**1. 低レベル SDPA/FlashAttention の公式 primitive はある。**
**2. CubeCL 側に公式 FlashAttention kernel は入っている。**
**3. ただし stable 0.20.1 の high-level `nn` attention はまだ手書き実装が中心。**
**4. 本格的な機能拡張と backend 全面実装は 0.21 pre 系で進行中。**
**5. backend ごとの安定性はまだ発展途中。** ([Docs.rs][2])

いま Burn で使う前提なら、判断はこうです。
**安定版重視なら**: `0.20.1` の `tensor::module::attention()` は「あるが機能は絞られている」。
**最新機能を追うなら**: `0.21.0-pre.2` 以降の attention 改善はかなり大きい。
**`nn::MultiHeadAttention` をそのまま使う場合**: 少なくとも stable 版の source 上は、まだ自動的に flash-attention primitive に全面移行していません。([Docs.rs][2])

必要なら次に、
**「stable 0.20.1 で何ができて何ができないか」一覧** か、
**「0.20.1 と 0.21.0-pre.2 の attention API 差分」** をコード付きで整理します。

[1]: https://github.com/Tracel-AI/burn/releases?utm_source=chatgpt.com "Releases · tracel-ai/burn"
[2]: https://docs.rs/burn/latest/burn/tensor/module/fn.attention.html "attention in burn::tensor::module - Rust"
[3]: https://github.com/tracel-ai/burn/issues/4475 "Enhance burn::tensor::module::attention() with scale, float mask, and softcap · Issue #4475 · tracel-ai/burn · GitHub"
[4]: https://github.com/Tracel-AI/burn/releases "Releases · tracel-ai/burn · GitHub"
[5]: https://docs.rs/burn-nn/latest/src/burn_nn/modules/attention/mha.rs.html "mha.rs - source"
[6]: https://docs.rs/burn-nn/latest/src/burn_nn/modules/attention/cross_attention.rs.html "cross_attention.rs - source"
[7]: https://github.com/tracel-ai/burn/issues/4325 "[burn-cubecl] FlashAttention doesn't work on most backends · Issue #4325 · tracel-ai/burn · GitHub"

