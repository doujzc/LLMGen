# Vendored ToolWeaver model code

`layers.py`, `vq.py`, `rq.py`, and `rqvae.py` are copied from
`Fwibo/ToolWeaver`, commit `3a102bad2d85f9674a7febdbaed0235d137e7222`.
Only end-of-file whitespace was normalized; executable source is unchanged.

The files were copied from the local source checkout at
`a7684edaf2bb3af7ff6928c34e27a324599deda0`; their hashes match the pinned
official ToolWeaver revision above.

Source: <https://github.com/Fwibo/ToolWeaver>

Only the RQ-VAE model dependency is vendored. LLMGen owns the SkillRet data
adapter, sparse graph training, checkpointing, indexing, router training, and
constrained inference around these files.

The pinned upstream snapshot did not include a LICENSE, COPYING, or NOTICE
file. These four vendored modules are therefore not represented as being
covered by LLMGen's MIT license. Confirm redistribution rights with the
ToolWeaver authors before publishing a redistributed source or wheel copy. A
source checkout also contains the repository-level `THIRD_PARTY_NOTICES.md`.
