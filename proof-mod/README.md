The test mod behind the in-game proofs on #19758, #19767, #19802 and #19913.

- `pipeline_proof.rs`: a `horizon-mod-host` example. Blocks marked `//@N{` .. `//@}` need part N;
  `python cut.py pipeline_proof.rs N out.rs` cuts the later parts for part N's branch.
- `shaders/`: the resource pack's `assets/proof/shaders/pipeline/` files.

Built on each pull request's head with
`cargo rustc --profile fast -p horizon-mod-host --example pipeline_proof -- -C link-arg=-Wl,--stack,142606336`,
with the package `target/fast/examples/horizon-native-mods/proof/` as `tools/hot_paths.py` writes one.
