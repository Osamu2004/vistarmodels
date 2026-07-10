# Vendored DreamCD Source

- Upstream repository: <https://github.com/tangkai-RS/DreamCD>
- Upstream commit: `d4750ff6f7d35fe9640059d7b9cdfe6902fcf9c5`
- Vendored scope: official inference/config/model source required by the
  `baselines/dreamcd` wrapper.
- Excluded: nested Git metadata, checkpoints, examples, figures, previews,
  Python caches, and standalone degradation demo assets.

The wrapper keeps checkpoints outside Git under
`/root/data/weight/dreamcd/second/`.
