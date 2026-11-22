//! Render pipelines a client mod registers as plain data, and the quads it draws with them.
use crate::{Identifier, WorldDepth};

/// A pipeline registered by `Hooks::register_render_pipelines`: a fragment shader from the
/// mod's assets and the fixed state it draws with. This is a restricted form: the mod supplies
/// the fragment stage and the host fixes the rest. Vanilla requires a vertex shader and a
/// primitive type; this API fixes both. The engine builds every GPU object.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
#[derive(horizon_mod_api_macros::Wire)]
pub struct RenderPipeline {
    owner: String,
    /// The pipeline's own name (`mymod:glow`), in a namespace the mod owns.
    pub id: Identifier,
    /// The fragment shader (`mymod:glow` is `assets/mymod/shaders/pipeline/glow.wgsl`), as the
    /// mod's assets or a resource pack supply it.
    pub fragment: Identifier,
    /// What the pipeline's draws are made of.
    pub vertex_format: PipelineVertexFormat,
    /// How the shader's result joins the frame.
    pub blend: PipelineBlend,
    /// Whether nearer world surfaces hide the draw. No pipeline writes depth.
    pub depth: WorldDepth,
    /// Whether a resource load may go on without this pipeline (NeoForge's
    /// `registerOptionalPipeline`). A required pipeline (NeoForge's `registerPipeline`, the
    /// default) that a load cannot build fails the load: the game does not start, and a reload
    /// fails and keeps the previous resources. An optional one is logged once by id and resolves
    /// to [`RenderPipelineError::Invalid`] while the other pipelines draw.
    pub optional: bool,
    /// What the shader samples with `pipeline_texture(uv)`, if anything ([`Self::textured`]).
    /// `None`, the default, binds no texture: the shader reads only its input and `frame`.
    pub texture: Option<PipelineTexture>,
    /// Whether the shader reads the opaque scene's depth ([`Self::reads_scene_depth`]).
    pub scene_depth: bool,
    /// The buffers of the mod's own data the shader reads, in the order registered
    /// ([`Self::reads_buffer`]).
    pub buffers: Vec<PipelineBuffer>,
}
impl RenderPipeline {
    /// Describe a required pipeline that draws [`PipelineVertexFormat::PositionTexColor`]
    /// geometry.
    pub fn new(
        id: Identifier,
        fragment: Identifier,
        blend: PipelineBlend,
        depth: WorldDepth,
    ) -> Self {
        Self {
            owner: String::new(),
            id,
            fragment,
            vertex_format: PipelineVertexFormat::PositionTexColor,
            blend,
            depth,
            optional: false,
            texture: None,
            scene_depth: false,
            buffers: Vec::new(),
        }
    }
    /// The owning mod, assigned by the host after registration.
    pub fn owner(&self) -> &str {
        &self.owner
    }

    /// Let a resource load that cannot build this pipeline go on without it (NeoForge's
    /// `registerOptionalPipeline`).
    pub fn optional(mut self) -> Self {
        self.optional = true;
        self
    }

    /// Let the shader sample `texture` with `pipeline_texture(uv)` (vanilla's `Sampler0`, as
    /// `RenderPipeline.withSampler` declares it). A texture the load refuses
    /// ([`PipelineTextureError`]) refuses the pipeline as a missing shader does: a required one
    /// fails the load, an optional one is logged with the named cause and resolves to
    /// [`RenderPipelineError::Invalid`].
    pub fn textured(mut self, texture: PipelineTexture) -> Self {
        self.texture = Some(texture);
        self
    }

    /// Let the shader read the depth of the opaque world, terrain without its translucent
    /// layers, at its pixel: `pipeline_scene_depth`, `pipeline_scene_position` and
    /// `pipeline_scene_distance`, for geometry a mod traces or places itself (ray marching,
    /// volumes, decals). The game copies the depth once a frame, after the opaque terrain, and
    /// only when such a pipeline draws: at `AfterSky` everything reads as sky, and from
    /// `AfterOpaqueBlocks` on the copy holds the opaque terrain, never later draws.
    pub fn reads_scene_depth(mut self) -> Self {
        self.scene_depth = true;
        self
    }

    /// Give the shader a buffer of `size` bytes of the mod's own data, read in the shader as
    /// `name`, an `array<vec4<f32>>` (bitcast for other types). The mod writes it with
    /// [`crate::WorldDrawContext::write_pipeline_buffer`], by its place among this pipeline's
    /// buffers (the first is 0); it starts as zeros and keeps what was written across frames and
    /// resource reloads. A pipeline reads at most 4 buffers, each 16 bytes to 16 MiB in steps of
    /// 16, and a mod's buffers hold at most 64 MiB together; `name` starts with a letter, has
    /// only letters, digits and `_`, and isn't the game's (`frame`, or starting `horizon_`,
    /// `pipeline_` or `Pipeline`). A registration past these is logged and dropped.
    pub fn reads_buffer(mut self, name: &str, size: u32) -> Self {
        self.buffers.push(PipelineBuffer::new(name, size));
        self
    }
}

/// A buffer of a mod's own data a pipeline's shader reads ([`RenderPipeline::reads_buffer`]).
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
#[derive(horizon_mod_api_macros::Wire)]
pub struct PipelineBuffer {
    /// The name the shader reads it by.
    pub name: String,
    /// Its size in bytes.
    pub size: u32,
}
impl PipelineBuffer {
    /// Describe a buffer; the registration checks it.
    pub fn new(name: &str, size: u32) -> Self {
        Self {
            name: name.to_owned(),
            size,
        }
    }
}

/// A texture a pipeline's shader samples ([`RenderPipeline::textured`]): always a rectangle of
/// the atlas the host binds, never a texture of its own. `pipeline_texture(uv)` maps `fract(uv)`
/// into that rectangle and loads one texel by integer coordinates at mip level 0 (nearest, no
/// filtering, so no neighbouring sprite's texel enters): `uv` 0 to 1 spans it once and beyond
/// that it repeats. It returns linear light (the mod render colour rule) with the texel's alpha.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
#[derive(horizon_mod_api_macros::Wire)]
pub enum PipelineTexture {
    /// A sprite of the game's block or item atlas, by its texture id: `powah:block/energy_charge`
    /// is `assets/powah/textures/block/energy_charge.png` as the `minecraft:blocks` atlas holds
    /// it, `minecraft:item/diamond` an item sprite. An animated sprite animates. A sprite in
    /// neither atlas refuses the pipeline ([`PipelineTextureError::NoSuchSprite`]). A texture
    /// file of the mod's own, packed into the runtime atlas's slot pool (arismoko/horizon#4438) and drawn the
    /// same way, follows once that pool exists.
    Sprite(Identifier),
}

/// Why a pipeline's texture was refused: the named cause its refusal is logged with.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
#[derive(horizon_mod_api_macros::Wire)]
pub enum PipelineTextureError {
    /// No sprite of that id in the block or item atlas.
    NoSuchSprite,
    /// No texture file of that id in the mod's assets or a resource pack.
    NoSuchTextureFile,
    /// The id names a file (a `textures/` path or a `.png` name), not a sprite key.
    NotASpriteKey,
    /// A texture file over the pool's cap of 1024 x 1024 pixels.
    OverCap,
    /// A texture file that isn't an image the game can decode.
    Undecodable,
    /// No free atlas slot, or over the mod's image budget.
    NoSlot,
}

impl std::fmt::Display for PipelineTextureError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::NoSuchSprite => "no such sprite in the block or item atlas",
            Self::NoSuchTextureFile => "no such texture file",
            Self::NotASpriteKey => "the id names a file, not a sprite key",
            Self::OverCap => "the texture file is over 1024 x 1024 pixels",
            Self::Undecodable => "the texture file cannot be decoded",
            Self::NoSlot => "no free atlas slot, or over the mod's image budget",
        })
    }
}

impl std::error::Error for PipelineTextureError {}

/// The vertex data a pipeline's draws carry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
#[derive(horizon_mod_api_macros::Wire)]
pub enum PipelineVertexFormat {
    /// A position, a texture coordinate and a colour: what [`PipelineQuad`] gives.
    PositionTexColor,
}

/// How a pipeline's fragments join the frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
#[derive(horizon_mod_api_macros::Wire)]
pub enum PipelineBlend {
    /// Laid over the frame by the fragment's alpha, as [`crate::WorldQuad`] is.
    Translucent,
    /// Added to the frame, scaled by the fragment's alpha.
    Additive,
}

/// A registered pipeline as the current resources built it, resolved through
/// [`crate::RenderAssets::render_pipeline`]. Retain it between frames; resolve it again after a
/// successful resource reload.
// no wire: decoding from_parts rejects generation zero (an unpublished pipeline handle).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
// exhaustive: private fields; only the engine resolves pipeline handles
pub struct RenderPipelineRef(pub(crate) u64, pub(crate) u32);

impl RenderPipelineRef {
    pub(crate) fn from_parts(generation: u64, slot: u32) -> Result<Self, &'static str> {
        if generation == 0 {
            return Err("unpublished pipeline generation");
        }
        Ok(Self(generation, slot))
    }
}

/// Why a pipeline could not be resolved or drawn with.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
#[derive(horizon_mod_api_macros::Wire)]
pub enum RenderPipelineError {
    /// No mod registered a pipeline of this id.
    Unknown,
    /// The pipeline is registered, but the current resources give it no usable shader. The log
    /// names the pipeline and the reason.
    Invalid,
    /// A successful resource reload superseded the handle's generation.
    Stale,
}

/// A two-sided quad drawn by a mod pipeline. Corners run top left, bottom left, bottom right,
/// top right as seen from the front.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
#[derive(horizon_mod_api_macros::Wire)]
pub struct PipelineQuad {
    /// Four world-space corners in blocks, before camera subtraction.
    pub corners: [[f64; 3]; 4],
    /// Each corner's texture coordinate, handed to the shader as given.
    pub uv: [[f32; 2]; 4],
    /// Packed 0xAARRGGBB colour in vanilla's space; the shader reads it as linear light.
    pub color: u32,
}
impl PipelineQuad {
    /// Describe a quad; the draw context validates it when submitting the batch.
    pub fn new(corners: [[f64; 3]; 4], uv: [[f32; 2]; 4], color: u32) -> Self {
        Self { corners, uv, color }
    }
    /// A quad whose texture coordinates span 0 to 1: (0, 0) at the top left corner and (1, 1)
    /// at the bottom right.
    pub fn spanning(corners: [[f64; 3]; 4], color: u32) -> Self {
        Self::new(
            corners,
            [[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]],
            color,
        )
    }
}

/// Why a whole pipeline batch was refused; nothing from that batch was appended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
#[derive(horizon_mod_api_macros::Wire)]
pub enum PipelineDrawError {
    /// The handle is stale, or names no usable pipeline.
    Pipeline(RenderPipelineError),
    /// A coordinate is not finite, or a camera-relative coordinate overflows f32.
    NonFinite,
    /// The batch would exceed this view's quad capacity, which pipeline quads share with
    /// [`crate::WorldDrawContext::quads`]; or a buffer write would bring what this view wrote
    /// into that buffer past its size.
    CapacityExceeded,
    /// The pipeline has no buffer at that place, or the write reaches past its end, or its
    /// offset or length is not a multiple of 4.
    BufferRange,
}

/// The most buffers a pipeline reads.
const MAX_BUFFERS: usize = 4;
/// The most bytes one buffer holds: 16 MiB.
const MAX_BUFFER_BYTES: u32 = 16 << 20;
/// The most bytes a mod's buffers hold together: 64 MiB.
const MAX_MOD_BUFFER_BYTES: u64 = 64 << 20;

/// Why `pipeline`'s buffers are refused, when its mod's earlier pipelines hold `held` bytes.
fn buffers_refusal(pipeline: &RenderPipeline, held: u64) -> Option<&'static str> {
    let buffers = &pipeline.buffers;
    if buffers.len() > MAX_BUFFERS {
        return Some("it reads more than 4 buffers");
    }
    if buffers
        .iter()
        .any(|b| b.size == 0 || !b.size.is_multiple_of(16) || b.size > MAX_BUFFER_BYTES)
    {
        return Some("a buffer's size is not 16 bytes to 16 MiB in steps of 16");
    }
    if buffers.iter().any(|b| !shader_name(&b.name)) {
        return Some("a buffer's name is not one of its own the shader can use");
    }
    if buffers
        .iter()
        .enumerate()
        .any(|(i, b)| buffers[..i].iter().any(|other| other.name == b.name))
    {
        return Some("two of its buffers have the same name");
    }
    let size: u64 = buffers.iter().map(|b| u64::from(b.size)).sum();
    (held + size > MAX_MOD_BUFFER_BYTES).then_some("the mod's buffers would hold more than 64 MiB")
}

/// A name a mod may give a buffer: a WGSL identifier of letters, digits and `_` that starts with
/// a letter and is none of the names the game puts before the mod's file.
fn shader_name(name: &str) -> bool {
    let mut chars = name.chars();
    chars.next().is_some_and(|c| c.is_ascii_alphabetic())
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
        && name.len() <= 64
        && !["horizon_", "pipeline_", "Pipeline"]
            .iter()
            .any(|prefix| name.starts_with(prefix))
        && name != "frame"
}

/// What `owner` added to `pipelines` in `register_render_pipelines`, checked as it returns: a
/// pipeline outside the mod's namespaces or whose id an earlier registration took is logged
/// with the mod and dropped.
pub(crate) fn validate_render_pipelines(
    owner: &str,
    previous: &[RenderPipeline],
    pipelines: &mut Vec<RenderPipeline>,
) {
    validate_in(owner, previous, pipelines, &|namespace| {
        crate::parts::owns_namespace(owner, namespace)
    });
}

fn validate_in(
    owner: &str,
    previous: &[RenderPipeline],
    pipelines: &mut Vec<RenderPipeline>,
    owns: &dyn Fn(&str) -> bool,
) {
    let added = pipelines.split_off(previous.len().min(pipelines.len()));
    for mut pipeline in added {
        let refusal = if !owns(pipeline.id.namespace()) {
            Some("its id is outside the mod's namespaces")
        } else if pipelines.iter().any(|kept| kept.id == pipeline.id) {
            Some("an earlier registration has that id")
        } else {
            let held = pipelines
                .iter()
                .filter(|kept| kept.owner == owner)
                .flat_map(|kept| &kept.buffers)
                .map(|b| u64::from(b.size))
                .sum();
            buffers_refusal(&pipeline, held)
        };
        match refusal {
            Some(reason) => log::warn!(
                "register_render_pipelines: mod {owner}: pipeline {} refused: {reason}",
                pipeline.id
            ),
            None => {
                pipeline.owner = owner.to_owned();
                pipelines.push(pipeline);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pipeline(id: &str) -> RenderPipeline {
        let id = Identifier::parse(id).unwrap();
        RenderPipeline::new(
            id.clone(),
            id,
            PipelineBlend::Translucent,
            WorldDepth::Tested,
        )
    }
    fn ids(pipelines: &[RenderPipeline]) -> Vec<String> {
        pipelines.iter().map(|p| p.id.to_string()).collect()
    }

    #[test]
    fn a_mod_keeps_only_new_pipelines_in_its_own_namespace() {
        let previous = vec![pipeline("other:glow")];
        let mut pipelines = previous.clone();
        pipelines.extend([
            pipeline("mine:glow"),
            pipeline("other:beam"),
            pipeline("mine:glow"),
            pipeline("mine:beam"),
        ]);
        validate_in("owning_mod", &previous, &mut pipelines, &|namespace| {
            namespace == "mine"
        });
        assert_eq!(ids(&pipelines), ["other:glow", "mine:glow", "mine:beam"]);
        assert_eq!(pipelines[1].owner(), "owning_mod");
    }

    #[test]
    fn a_mod_cannot_take_an_id_registered_before_it() {
        let previous = vec![pipeline("shared:glow")];
        let mut pipelines = previous.clone();
        pipelines.push(pipeline("shared:glow"));
        validate_in("late", &previous, &mut pipelines, &|_| true);
        assert_eq!(pipelines, previous);
    }

    /// A pipeline's buffers are refused with the pipeline unless each is 16 bytes to 16 MiB in
    /// steps of 16, there are at most 4, each has a name of its own the shader can use, and the
    /// mod's buffers come to no more than 64 MiB.
    #[test]
    fn a_pipelines_buffers_must_keep_to_the_limits() {
        let with = |id: &str, buffers: &[(&str, u32)]| {
            buffers
                .iter()
                .fold(pipeline(id), |p, (name, size)| p.reads_buffer(name, *size))
        };
        let mut pipelines = vec![
            with("mine:ok", &[("blasts", 256), ("grid_2", 16 << 20)]),
            with("mine:odd", &[("blasts", 100)]),
            with("mine:empty", &[("blasts", 0)]),
            with("mine:huge", &[("blasts", (16 << 20) + 16)]),
            with(
                "mine:many",
                &[("a", 16), ("b", 16), ("c", 16), ("d", 16), ("e", 16)],
            ),
            with("mine:twice", &[("a", 16), ("a", 16)]),
            with("mine:games", &[("horizon_view", 16)]),
            with("mine:prelude", &[("pipeline_data", 16)]),
            with("mine:frame", &[("frame", 16)]),
            with("mine:digit", &[("1st", 16)]),
            with("mine:space", &[("my data", 16)]),
        ];
        validate_in("mine", &[], &mut pipelines, &|_| true);
        assert_eq!(ids(&pipelines), ["mine:ok"]);

        // 64 MiB for the mod: the fifth 16 MiB buffer is refused, another mod's still fits.
        let previous = vec![with("other:big", &[("data", 16 << 20)])];
        let mut big = previous.clone();
        big.extend((0..5).map(|i| with(&format!("mine:b{i}"), &[("data", 16 << 20)])));
        validate_in("mine", &previous, &mut big, &|_| true);
        assert_eq!(
            ids(&big),
            ["other:big", "mine:b0", "mine:b1", "mine:b2", "mine:b3"]
        );
    }

    #[test]
    fn a_mod_can_register_more_than_vanillas_pipeline_count() {
        let mut pipelines: Vec<_> = (0..131).map(|i| pipeline(&format!("mine:p{i}"))).collect();
        validate_in("mine", &[], &mut pipelines, &|_| true);
        assert_eq!(pipelines.len(), 131);
        assert_eq!(pipelines.last().unwrap().id.to_string(), "mine:p130");

        let previous = pipelines.clone();
        pipelines.push(pipeline("next:glow"));
        validate_in("next", &previous, &mut pipelines, &|_| true);
        assert_eq!(pipelines.len(), 132);
    }
}
