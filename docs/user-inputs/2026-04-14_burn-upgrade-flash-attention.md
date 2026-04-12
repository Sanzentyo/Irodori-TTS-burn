# User Input — 2026-04-14

## Message

burn_upgradeとflash_attentionの方向で、一旦これをcommitしてから、worktree使って検証をすすめてください。
docs/review/other_advice_1.md にburnのflash_attensionの状況を、2の方にflash_attention自体についての解説をいれました。参考にしなさい。
あと、docs/user-inputs/ の内容をちゃんと覚えてますか?最初にこれらを読み込み、状況を整理し、自動的に読み込まれるこのリポジトリ固有のinstructionのmarkdownを作成し、それを逐次更新して行って、作業を進めなさい。必要な事項については、ちゃんと抽象化して的確に抜き出してまとめておきなさい

## Summary

- Approach: burn 0.21 upgrade + Flash Attention (not cuBLAS/unsafe)
- Use git worktree for exploration/validation
- Read docs/review/other_advice_1.md (burn Flash Attention status) and other_advice_2.md (Flash Attention mechanics)
- Create a repo-specific `.github/copilot-instructions.md` that auto-loads with key project context
- Keep docs/user-inputs/ updated, abstracts key decisions
