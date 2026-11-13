## What and why

## Checked

- [ ] `uv run ruff check .` and `uv run pytest` pass (local mode)
- [ ] `uv run pyright` passes if `src/` changed
- [ ] A `CHANGELOG.md` line for user-visible changes
- [ ] Docs and example READMEs match the change ("Requires" lines and the table in `examples/README.md` if the number of MicroVMs changed)
- [ ] No tokens, `.env` contents, endpoint URLs or personal data in the diff
- [ ] Live runs (if any): the team was empty before and after, and clean-up was run
