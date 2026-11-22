use super::*;
use mod_api::__private::game::RegistrationContextNew;
use mod_api::{Identifier, PipelineBlend, PipelineTexture, RenderPipeline, WorldDepth};

fn id(text: &str) -> Identifier {
    Identifier::parse(text).unwrap()
}

/// The pipelines a mod registers: a mod's sprite, a vanilla item sprite with a buffer, and one
/// untextured that reads the scene depth and two buffers.
fn pipelines() -> [RenderPipeline; 3] {
    let pipeline = |name: &str| {
        RenderPipeline::new(
            id(name),
            id(name),
            PipelineBlend::Additive,
            WorldDepth::Tested,
        )
    };
    [
        pipeline("wire_test:orb").textured(PipelineTexture::Sprite(id("wire_test:block/charge"))),
        pipeline("wire_test:glow")
            .textured(PipelineTexture::Sprite(id("minecraft:item/diamond")))
            .reads_buffer("glow", 16)
            .writes_depth()
            .optional(),
        pipeline("wire_test:plain")
            .reads_scene_depth()
            .reads_buffer("blasts", 256)
            .reads_buffer("grid", 16 << 20),
    ]
}

/// Every refusal of a pipeline draw or buffer write crosses the wire as itself.
#[test]
fn pipeline_draw_refusals_round_trip() {
    use mod_api::{PipelineDrawError as E, RenderPipelineError};
    for cause in [
        E::Pipeline(RenderPipelineError::Stale),
        E::NonFinite,
        E::CapacityExceeded,
        E::BufferRange,
    ] {
        let mut out = Writer::new();
        out.put(&cause);
        assert_eq!(Reader::decode::<E>(out.bytes()).unwrap(), cause);
    }
}

/// Every named texture refusal crosses the wire as itself.
#[test]
fn pipeline_texture_refusals_round_trip() {
    use mod_api::PipelineTextureError as E;
    for cause in [
        E::NoSuchSprite,
        E::NoSuchTextureFile,
        E::NotASpriteKey,
        E::OverCap,
        E::Undecodable,
        E::NoSlot,
    ] {
        let mut out = Writer::new();
        out.put(&cause);
        assert_eq!(Reader::decode::<E>(out.bytes()).unwrap(), cause);
    }
}

#[test]
fn pipeline_textures_round_trip() {
    for pipeline in pipelines() {
        let mut out = Writer::new();
        out.put(&pipeline);
        assert_eq!(
            Reader::decode::<RenderPipeline>(out.bytes()).unwrap(),
            pipeline
        );
        if let Some(texture) = &pipeline.texture {
            let mut out = Writer::new();
            out.put(texture);
            assert_eq!(
                &Reader::decode::<PipelineTexture>(out.bytes()).unwrap(),
                texture
            );
        }
    }
}

pub(super) struct PipelineTexturesHooks;
impl PipelineTexturesHooks {
    pub(super) fn register_render_pipelines(
        &self,
        _: &mod_api::RegistrationContext,
        event: &mut mod_api::RegisterRenderPipelinesEvent<'_>,
    ) {
        event.pipelines.extend(pipelines());
    }
}

struct Native;
#[mod_api::__subscriptions]
impl Hooks for Native {
    fn register_render_pipelines(
        &self,
        cx: &mod_api::RegistrationContext,
        event: &mut mod_api::RegisterRenderPipelinesEvent<'_>,
    ) {
        PipelineTexturesHooks.register_render_pipelines(cx, event);
    }
}

#[test]
fn a_wasm_mods_textured_pipelines_reach_the_host_as_a_native_mods_do() {
    for guest in [false, true] {
        let hooks: Box<dyn Hooks> = if guest {
            Box::new(WasmMod {
                group: 0,
                subscribed: &[Some(mod_api::HookId::RegisterRenderPipelines)],
                listeners: &[],
                corrupt: false,
            })
        } else {
            Box::new(Native)
        };
        mod_api::testing::with_mods(
            vec![mod_api::ModEntry::new("wire_test", hooks)].into(),
            || {
                let mut registered = Vec::new();
                mod_api::dispatch_register_render_pipelines(
                    &mod_api::RegistrationContext::new(),
                    &mut registered,
                );
                let textures: Vec<_> = registered.iter().map(|p| p.texture.clone()).collect();
                assert_eq!(
                    textures,
                    [
                        Some(PipelineTexture::Sprite(id("wire_test:block/charge"))),
                        Some(PipelineTexture::Sprite(id("minecraft:item/diamond"))),
                        None,
                    ],
                    "guest {guest}"
                );
                assert_eq!(
                    registered.iter().map(|p| p.optional).collect::<Vec<_>>(),
                    [false, true, false],
                    "guest {guest}"
                );
                assert_eq!(
                    registered.iter().map(|p| p.scene_depth).collect::<Vec<_>>(),
                    [false, false, true],
                    "guest {guest}"
                );
                assert_eq!(
                    registered
                        .iter()
                        .map(|p| p.buffers.iter().map(|b| (b.name.as_str(), b.size)).collect())
                        .collect::<Vec<Vec<_>>>(),
                    [vec![], vec![("glow", 16)], vec![("blasts", 256), ("grid", 16 << 20)]],
                    "guest {guest}"
                );
                assert_eq!(
                    registered.iter().map(|p| p.depth_write).collect::<Vec<_>>(),
                    [false, true, false],
                    "guest {guest}"
                );
                assert!(registered.iter().all(|p| p.owner() == "wire_test"));
            },
        );
    }
}
