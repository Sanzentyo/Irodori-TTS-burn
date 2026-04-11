# Initial Request — 2026-04-11

## Raw Input

https://github.com/Aratako/Irodori-TTS をsubmoduleに追加しない形でcloneして、それを参考にし、同等の機能をrustのburnと言う機械学習ライブラリで再現実装(構造的なリファクタリング自体は行い、rustのskillを読んで、それに従って、適度なRustらしい構造にしていく)し、結果の正当性とパフォーマンスで同等の性能が出るまでとりあえず再現実装して行ってください。Irodori-TTS自体はpythonなのでuvで環境構築して動かしてください。ラッパーとかではなく、フルスクラッチの再現実装(pytorchからburnへの移植)を念頭に置いて下さい。ライブラリ自体は常に最新のものをつかうため、Cargo.tomlではなく、cargo addで追加してください。定期的なレビュー(複数観点(保守性やパフォーマンス、rustらしさ(pythonに引っ張られていないか、enumやtraitなどを使った本当にRustらしいコードになっているか)、これ以外の観点も必要なら定義し、これを元にしたレビューと自分とsubagentとrubber duckなどを使って定期的にレビューをしていってください)。適宜、docsに計画や現状などをちゃんと構造化してかいていってください。type stateパターンなども使っていくようにしなさい。こちらからの入力は、コピペで明らかに大きいものをのぞいて、全部、markdownに個別のファイルにフルのこちらの入力を残していってください。ghでprivateでこのCargoのプロジェクト(Irodori-TTS-burn)をあっぷしてください。task runnerはjustをつかいなさい。あと、contextを圧縮後にskillを必ず読み込むように必要そうなskillsをlistにし、書いて、必ず読み込まれた状態で動くようにしなさい

## Summary

- Clone https://github.com/Aratako/Irodori-TTS (not as submodule)
- Implement equivalent functionality in Rust using the `burn` ML library (full scratch rewrite, NOT a wrapper)
- PyTorch → burn port
- Set up Python (Irodori-TTS) with `uv`
- Use `cargo add` always (never edit Cargo.toml directly for deps)
- Use `just` as task runner
- Regular reviews: maintainability, performance, Rust idiomaticity (enums, traits, type state pattern)
- Save all user inputs to `docs/user-inputs/` markdown files
- Write structured docs in `docs/`
- Upload to GitHub as private repo (`Irodori-TTS-burn`)
- After context compression: reload skill list:
  - `rust-best-practices`
  - `justfile`
  - `python-uv-enforcer`
- Use type state pattern where applicable
