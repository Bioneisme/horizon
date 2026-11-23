//! Mods' render pipelines on the GPU (`Hooks::register_render_pipelines`): the pipelines built
//! from a resource load's checked shaders (`crate::mod_pipelines`), and the quads a frame draws
//! with them. They draw with the world primitives, at their stages and into the same targets,
//! so they take the world's reversed depth, its draw resolution and TAA's jitter. Pipelines are
//! built with the renderer a load is applied with ([`super::Renderer::build_mod_pipelines`]),
//! never while quads are being submitted, so a required pipeline the GPU refuses fails that
//! load's renderer and the reload with it.

use super::frame_targets::Kind;
use super::world_primitives::Upload;
use crate::mod_pipelines::{
    Built, BuiltTexture, FIRST_BUFFER_BINDING, FRAGMENT_ENTRY, Pipelines, VERTEX_ENTRY,
};
use glam::Mat4;
use mod_api::{PipelineBlend, WorldDepth};

/// One corner of a pipeline quad, camera-relative.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PipelineVertex {
    pub pos: [f32; 3],
    pub uv: [f32; 2],
    /// Red in the low byte, then green, blue and alpha, as vanilla encodes them.
    pub color: u32,
}

/// Quads that follow each other in a frame's vertices and draw with one pipeline at one stage.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PipelineRun {
    /// The pipeline's slot in the frame's [`Pipelines`].
    pub slot: u32,
    /// The stage it draws at (`RenderLevelStage::ALL`'s index).
    pub stage: u32,
    pub quads: u32,
}

/// A write of a frame into a pipeline's buffer ([`mod_api::WorldDrawContext::write_pipeline_buffer`]),
/// checked when the game queued it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PipelineBufferWrite {
    /// The pipeline's slot.
    pub slot: u32,
    /// The buffer's place among the pipeline's.
    pub buffer: u32,
    /// Where in the buffer it goes, in bytes.
    pub offset: u32,
    /// What it writes, in [`Batch::bytes`].
    pub bytes: std::ops::Range<u32>,
}

/// A frame's pipeline quads, as the game queued them: 4 vertices a quad, in the runs' order.
#[derive(Clone, Copy, Debug, Default)]
pub struct Batch<'a> {
    /// The pipelines of the resources the frame draws with; `None` draws nothing.
    pub set: Option<&'a Pipelines>,
    pub vertices: &'a [PipelineVertex],
    pub runs: &'a [PipelineRun],
    /// `PipelineFrame.time`.
    pub time: f32,
    /// The frame's writes into the pipelines' buffers, in the order written.
    pub writes: &'a [PipelineBufferWrite],
    pub bytes: &'a [u8],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct FrameUniform {
    viewport: [f32; 2],
    time: f32,
    _pad: f32,
    /// `PipelineFrame.eye`, at a `vec3`'s alignment.
    eye: [f32; 3],
    _pad_eye: f32,
}

/// Where the rays of the view `view_proj` start, in blocks relative to the camera: the camera,
/// but for a translation the projection carries (the view bob's). A projection maps that point
/// to `w` = 0, so it is the inverse's image of the direction along depth.
fn view_eye(view_proj: Mat4) -> [f32; 3] {
    let eye = view_proj.inverse() * glam::Vec4::Z;
    if eye.w.abs() > f32::EPSILON && eye.is_finite() {
        (eye.truncate() / eye.w).to_array()
    } else {
        [0.0; 3]
    }
}

/// A run as uploaded: where its quads start in the vertex buffer.
struct Run {
    slot: usize,
    stage: usize,
    quads: std::ops::Range<u32>,
}

/// Where textured pipelines' textures come from when the renderer builds them
/// ([`mod_api::RenderPipeline::textured`]).
#[derive(Clone, Copy)]
pub struct TextureSources<'a> {
    /// The 2D atlas the blocks and items atlases' sprites are drawn from.
    pub atlas: &'a wgpu::TextureView,
    /// A sprite's `[u0, v0, u1, v1]` in `atlas`, by its name there; `None` for no such sprite.
    pub sprite: &'a dyn Fn(&str) -> Option<[f32; 4]>,
}

/// What mods wrote into their pipelines' buffers, kept by the game as each write is accepted
/// ([`keep_write`]), outside any renderer.
#[derive(Default)]
struct Kept {
    /// By pipeline id and the buffer's place.
    bytes: std::collections::HashMap<(String, u32), Vec<u8>>,
    /// How many writes were kept: a renderer whose frames uploaded fewer missed some.
    writes: u64,
}

/// The kept writes. A renderer that missed some (a resource reload builds a new one, a frame
/// returned before preparing its writes, or one was given up before its uploads reached the GPU,
/// #13624) uploads every buffer from here before its next draw.
static CONTENTS: std::sync::LazyLock<std::sync::Mutex<Kept>> =
    std::sync::LazyLock::new(Default::default);

fn contents() -> std::sync::MutexGuard<'static, Kept> {
    CONTENTS.lock().unwrap_or_else(|e| e.into_inner())
}

/// Keeps a write the game accepted into buffer `buffer` (`size` bytes) of pipeline `pipeline`,
/// as it queues the write for the frame: the renderer uploads the queued write, or, if it missed
/// any, every buffer as kept here.
pub fn keep_write(
    pipeline: &mod_api::Identifier,
    buffer: u32,
    size: u32,
    offset: u32,
    bytes: &[u8],
) {
    // The game queues no empty write, so keeping one would read as a write the renderer missed.
    if bytes.is_empty() {
        return;
    }
    let mut kept = contents();
    let at = offset as usize;
    let Some(end) = at
        .checked_add(bytes.len())
        .filter(|&end| end <= size as usize)
    else {
        return;
    };
    let buffer = kept
        .bytes
        .entry((pipeline.to_string(), buffer))
        .or_insert_with(|| vec![0; size as usize]);
    buffer.resize(size as usize, 0);
    buffer[at..end].copy_from_slice(bytes);
    kept.writes += 1;
}

#[cfg(test)]
thread_local! {
    /// How many of the next buffer allocations find no room, as a device out of memory would.
    static NO_ROOM: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
}

/// A built pipeline, with the bind group of its texture and buffers (group 2) when it has
/// either.
struct Compiled {
    pipeline: wgpu::RenderPipeline,
    texture: Option<wgpu::BindGroup>,
    /// It binds the scene depth in group 1.
    scene_depth: bool,
}

/// The opaque world's depth as this frame copied it, for the pipelines that read it.
struct SceneDepth {
    texture: wgpu::Texture,
    group: wgpu::BindGroup,
    /// Made under `frame_targets::made`'s own scopes. A scaled screenshot's frame makes it
    /// without them (its own scopes catch a failure), so the next window frame makes it again.
    checked: bool,
}

/// The pipelines and the buffer that draw a [`Batch`].
pub struct ModPipelines {
    color_format: wgpu::TextureFormat,
    group_layout: wgpu::BindGroupLayout,
    /// The scene depth and the inverse view.
    depth_layout: wgpu::BindGroupLayout,
    /// The inverse of the jittered view, which the scene depth functions read.
    inverse: wgpu::Buffer,
    /// Group 1 over a depth of 0 everywhere: before the copy, and on a frame without one.
    sky: wgpu::BindGroup,
    /// What the last copy went into, sized as its source.
    scene: Option<SceneDepth>,
    /// Whether this frame copied the scene depth, since [`Self::prepare`].
    copied: bool,
    /// The mods' buffers by pipeline slot, made when a load first builds a pipeline that reads
    /// them and kept from then on: registrations don't change, so neither do their sizes.
    buffers: Vec<Vec<wgpu::Buffer>>,
    /// Each slot's pipeline id, the key of its buffers' [`CONTENTS`].
    buffer_ids: Vec<Option<String>>,
    /// How many kept writes this renderer's buffers hold; `None` when it can't tell (its
    /// buffers are new, or a frame's uploads were lost: [`Self::forget`]). A frame whose queued
    /// writes don't bring it to [`Kept::writes`] uploads every buffer as kept.
    synced: Option<u64>,
    /// The atlas's one sampler binding; `pipeline_texture` loads integer texels itself.
    sampler: wgpu::Sampler,
    group: wgpu::BindGroup,
    view: wgpu::Buffer,
    frame: wgpu::Buffer,
    vertices: wgpu::Buffer,
    /// The load `pipelines` was built from; 0 before any.
    generation: u64,
    /// By slot; `None` for a pipeline the load refused.
    pipelines: Vec<Option<Compiled>>,
    runs: Vec<Run>,
}

impl ModPipelines {
    /// Storage for pipelines drawing into `color_format` over a reversed Depth32Float depth.
    pub fn new(device: &wgpu::Device, color_format: wgpu::TextureFormat) -> Self {
        let uniform = |binding, visibility| wgpu::BindGroupLayoutEntry {
            binding,
            visibility,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Mod pipelines"),
            entries: &[
                // A depth-writing pipeline's fragment stage projects its surface's point too.
                uniform(0, wgpu::ShaderStages::VERTEX_FRAGMENT),
                uniform(1, wgpu::ShaderStages::FRAGMENT),
            ],
        });
        let depth_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Mod pipeline scene depth"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                uniform(1, wgpu::ShaderStages::FRAGMENT),
            ],
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Mod pipeline texture"),
            ..Default::default()
        });
        let buffer = |label, size, usage| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: usage | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let view = buffer("Mod pipeline view", 64, wgpu::BufferUsages::UNIFORM);
        let frame = buffer(
            "Mod pipeline frame",
            std::mem::size_of::<FrameUniform>() as u64,
            wgpu::BufferUsages::UNIFORM,
        );
        let inverse = buffer("Mod pipeline inverse view", 64, wgpu::BufferUsages::UNIFORM);
        // A depth texture reads 0 until something is drawn into it: the sky everywhere.
        let sky = depth_texture(device, [1, 1], wgpu::TextureUsages::RENDER_ATTACHMENT);
        let sky = depth_group(device, &depth_layout, &sky, &inverse);
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Mod pipelines"),
            layout: &group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: view.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: frame.as_entire_binding(),
                },
            ],
        });
        Self {
            color_format,
            group_layout,
            depth_layout,
            inverse,
            sky,
            scene: None,
            copied: false,
            buffers: Vec::new(),
            buffer_ids: Vec::new(),
            synced: None,
            sampler,
            group,
            view,
            frame,
            vertices: buffer("Mod pipeline quads", 0, wgpu::BufferUsages::VERTEX),
            generation: 0,
            pipelines: Vec::new(),
            runs: Vec::new(),
        }
    }

    /// One load's pipeline: its checked shader with the fixed state it registered, and its
    /// texture's bind group when it has one, or the GPU's error.
    fn pipeline(
        &self,
        device: &wgpu::Device,
        built: &Built,
        buffers: &[wgpu::Buffer],
        textures: Option<TextureSources<'_>>,
    ) -> Result<Compiled, String> {
        if built.buffers.len() as u32 > device.limits().max_storage_buffers_per_shader_stage {
            return Err("this GPU can't read the pipeline's buffers in a fragment shader".into());
        }
        let resources = resources_layout(device, built);
        let texture = match &resources {
            None => None,
            Some(layout) => Some(self.resources_group(device, layout, built, buffers, textures)?),
        };
        // What every depth reader shares comes before what differs between pipelines, so a switch
        // between two depth readers never disturbs the depth's binding.
        let mut groups = vec![
            Some(&self.group_layout),
            built.scene_depth.then_some(&self.depth_layout),
            resources.as_ref(),
        ];
        while groups.last().is_some_and(Option::is_none) {
            groups.pop();
        }
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Mod pipeline"),
            bind_group_layouts: &groups,
            ..Default::default()
        });
        // The shader text is a mod's or a pack's: a backend failing to compile it is an
        // internal error, and neither kind may reach the device's uncaptured handler.
        let validation = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let internal = device.push_error_scope(wgpu::ErrorFilter::Internal);
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Mod pipeline"),
            source: wgpu::ShaderSource::Wgsl(built.shader.as_str().into()),
        });
        let attributes = wgpu::vertex_attr_array![0=>Float32x3,1=>Float32x2,2=>Unorm8x4];
        // Blending happens in the frame's storage, as every vanilla draw's does. Neither mode
        // writes the frame's alpha past what covering it means.
        let blend = match built.blend {
            PipelineBlend::Additive => wgpu::BlendState {
                color: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::SrcAlpha,
                    dst_factor: wgpu::BlendFactor::One,
                    operation: wgpu::BlendOperation::Add,
                },
                alpha: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::Zero,
                    dst_factor: wgpu::BlendFactor::One,
                    operation: wgpu::BlendOperation::Add,
                },
            },
            _ => wgpu::BlendState::ALPHA_BLENDING,
        };
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Mod pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some(VERTEX_ENTRY),
                compilation_options: Default::default(),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<PipelineVertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &attributes,
                })],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                // Two-sided, as world quads are.
                cull_mode: None,
                ..Default::default()
            },
            // Depth is written only by a pipeline that writes its surface's. Reversed depth:
            // nearer is greater.
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(built.depth_write),
                depth_compare: Some(match built.depth {
                    WorldDepth::ThroughWalls => wgpu::CompareFunction::Always,
                    _ => wgpu::CompareFunction::GreaterEqual,
                }),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some(FRAGMENT_ENTRY),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: self.color_format,
                    blend: Some(blend),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });
        let internal = pollster::block_on(internal.pop());
        let validation = pollster::block_on(validation.pop());
        match internal.or(validation) {
            None => Ok(Compiled {
                pipeline,
                texture,
                scene_depth: built.scene_depth,
            }),
            Some(error) => Err(error.to_string()),
        }
    }

    /// The bind group of a pipeline's texture (the atlas and its sprite's rect) and buffers.
    /// Made when the pipeline is built, never in a frame; a sprite the atlas lacks is refused by
    /// name.
    fn resources_group(
        &self,
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        built: &Built,
        buffers: &[wgpu::Buffer],
        sources: Option<TextureSources<'_>>,
    ) -> Result<wgpu::BindGroup, String> {
        use wgpu::util::DeviceExt;
        let texture = match &built.texture {
            None => None,
            Some(BuiltTexture::Sprite(name)) => {
                let sources = sources.ok_or("no texture can be bound here")?;
                let Some(rect) = (sources.sprite)(name) else {
                    return Err(format!(
                        "its texture sprite {name}: {}",
                        crate::mod_pipelines::texture_refusal(
                            mod_api::PipelineTextureError::NoSuchSprite
                        )
                    ));
                };
                let rect = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Mod pipeline texture rect"),
                    contents: bytemuck::cast_slice(&rect),
                    usage: wgpu::BufferUsages::UNIFORM,
                });
                Some((sources.atlas.clone(), rect))
            }
        };
        if buffers.len() != built.buffers.len() {
            return Err("its buffers are not made".into());
        }
        let validation = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let mut entries = Vec::new();
        if let Some((view, rect)) = &texture {
            entries.extend([
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: rect.as_entire_binding(),
                },
            ]);
        }
        entries.extend(
            buffers
                .iter()
                .zip(FIRST_BUFFER_BINDING..)
                .map(|(buffer, binding)| wgpu::BindGroupEntry {
                    binding,
                    resource: buffer.as_entire_binding(),
                }),
        );
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Mod pipeline resources"),
            layout,
            entries: &entries,
        });
        match pollster::block_on(validation.pop()) {
            None => Ok(group),
            Some(error) => Err(format!("its texture or buffers: {error}")),
        }
    }

    /// Makes the buffers of `set`'s pipelines that have none yet, by slot; the next frame fills
    /// them from [`CONTENTS`]. None are made for a pipeline the GPU can't read buffers for, and
    /// none when the device has no room for them: [`Self::pipeline`] then refuses it, under the
    /// pipeline's optional or required policy.
    fn make_buffers(&mut self, device: &wgpu::Device, set: &Pipelines) {
        let limit = device.limits().max_storage_buffers_per_shader_stage as usize;
        self.buffers.resize_with(set.built().len(), Vec::new);
        self.buffer_ids.resize(set.built().len(), None);
        for (slot, built) in set.built().iter().enumerate() {
            let Some(built) = built else { continue };
            self.buffer_ids[slot] = Some(built.id.to_string());
            let have: Vec<u64> = self.buffers[slot].iter().map(wgpu::Buffer::size).collect();
            let want: Vec<u64> = built.buffers.iter().map(|&size| u64::from(size)).collect();
            if have == want {
                continue;
            }
            self.buffers[slot] = Vec::new();
            if want.is_empty() || want.len() > limit {
                continue;
            }
            let validation = device.push_error_scope(wgpu::ErrorFilter::Validation);
            let memory = device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
            let made: Vec<_> = want
                .iter()
                .map(|&size| {
                    device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("Mod pipeline buffer"),
                        size,
                        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    })
                })
                .collect();
            let memory = pollster::block_on(memory.pop());
            let validation = pollster::block_on(validation.pop());
            #[cfg(test)]
            let memory = memory.map(|e| e.to_string()).or_else(|| {
                let left = NO_ROOM.get().checked_sub(1)?;
                NO_ROOM.set(left);
                Some("no room (test)".to_owned())
            });
            if memory.is_none() && validation.is_none() {
                self.buffers[slot] = made;
                self.synced = None;
            }
        }
    }

    /// After a frame whose uploads never reached the GPU: the next frame uploads every buffer's
    /// contents again.
    pub fn forget(&mut self) {
        self.synced = None;
    }

    /// Builds the pipelines of `set` unless they are built already. A required pipeline the GPU
    /// refuses is the error, by id, and leaves this renderer and `set` as they were; an optional
    /// one is logged once, draws nothing and resolves invalid.
    /// A textured pipeline whose texture can't be bound (`textures`) is refused the same way.
    fn build(
        &mut self,
        device: &wgpu::Device,
        set: &Pipelines,
        textures: Option<TextureSources<'_>>,
    ) -> Result<(), String> {
        if set.generation() != self.generation {
            self.make_buffers(device, set);
        }
        self.build_with(set, |this, slot, built| {
            let buffers = this.buffers.get(slot).map_or(&[][..], Vec::as_slice);
            this.pipeline(device, built, buffers, textures)
        })
    }

    fn build_with(
        &mut self,
        set: &Pipelines,
        compile: impl Fn(&Self, usize, &Built) -> Result<Compiled, String>,
    ) -> Result<(), String> {
        if set.generation() == self.generation {
            return Ok(());
        }
        let mut refused = Vec::new();
        let pipelines: Vec<_> = set
            .built()
            .iter()
            .enumerate()
            .map(|(slot, built)| {
                let built = built.as_ref()?;
                compile(self, slot, built)
                    .map_err(|error| refused.push((slot, built, error)))
                    .ok()
            })
            .collect();
        let required: Vec<_> = refused
            .iter()
            .filter(|(_, built, _)| !built.optional)
            .map(|(_, built, error)| format!("required {}", gpu_refusal(built, error)))
            .collect();
        if !required.is_empty() {
            return Err(required.join("; "));
        }
        for (slot, built, error) in refused {
            if let Some(buffers) = self.buffers.get_mut(slot) {
                buffers.clear();
            }
            if set.invalidate(slot) {
                log::warn!("Horizon: {}", gpu_refusal(built, &error));
            }
        }
        for (slot, pipeline) in pipelines.iter().enumerate() {
            if pipeline.is_some() {
                set.revalidate(slot);
            }
        }
        self.generation = set.generation();
        self.pipelines = pipelines;
        Ok(())
    }

    /// Uploads the frame's quads, drawn with `view_proj` into a frame `viewport` pixels big. The
    /// number of quads to draw: the caller's quad indices must reach that far. A set this
    /// renderer was not built with draws nothing.
    pub fn prepare(
        &mut self,
        device: &wgpu::Device,
        upload: Upload<'_>,
        batch: &Batch<'_>,
        view_proj: Mat4,
        viewport: [u32; 2],
    ) -> u32 {
        self.runs.clear();
        self.copied = false;
        // Before the frame's own checks: a buffer belongs to its slot, whatever load drew.
        let kept = contents();
        let caught_up = self
            .synced
            .is_some_and(|synced| synced + batch.writes.len() as u64 == kept.writes);
        if caught_up {
            for write in batch.writes {
                let buffer = self
                    .buffers
                    .get(write.slot as usize)
                    .and_then(|b| b.get(write.buffer as usize));
                let bytes = batch
                    .bytes
                    .get(write.bytes.start as usize..write.bytes.end as usize);
                if let (Some(buffer), Some(bytes)) = (buffer, bytes)
                    && u64::from(write.offset) + bytes.len() as u64 <= buffer.size()
                {
                    upload(buffer, u64::from(write.offset), bytes);
                }
            }
        } else {
            // Some writes never reached these buffers: every one is uploaded as kept, the
            // frame's own writes with them.
            for (slot, buffers) in self.buffers.iter().enumerate() {
                let Some(Some(id)) = self.buffer_ids.get(slot) else {
                    continue;
                };
                for (index, buffer) in (0..).zip(buffers) {
                    if let Some(bytes) = kept.bytes.get(&(id.clone(), index))
                        && bytes.len() as u64 == buffer.size()
                    {
                        upload(buffer, 0, bytes);
                    }
                }
            }
        }
        self.synced = Some(kept.writes);
        drop(kept);
        if batch
            .set
            .is_none_or(|set| set.generation() != self.generation)
        {
            return 0;
        }
        let mut next = 0;
        for run in batch.runs {
            let quads = next..next + run.quads;
            next = quads.end;
            // A run past the vertices given, or of a pipeline this load has none for, has
            // nothing to draw with.
            if quads.end as usize * 4 > batch.vertices.len()
                || !matches!(self.pipelines.get(run.slot as usize), Some(Some(_)))
            {
                continue;
            }
            self.runs.push(Run {
                slot: run.slot as usize,
                stage: run.stage as usize,
                quads,
            });
        }
        let Some(last) = self.runs.last() else {
            return 0;
        };
        let quads = last.quads.end;
        let bytes: &[u8] = bytemuck::cast_slice(&batch.vertices[..quads as usize * 4]);
        if self.vertices.size() < bytes.len() as u64 {
            self.vertices = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Mod pipeline quads"),
                size: (bytes.len() as u64).next_power_of_two(),
                usage: self.vertices.usage(),
                mapped_at_creation: false,
            });
        }
        upload(&self.vertices, 0, bytes);
        upload(
            &self.view,
            0,
            bytemuck::cast_slice(&view_proj.to_cols_array()),
        );
        if self.runs.iter().any(|run| self.reads_depth(run.slot)) {
            let inverse = view_proj.inverse().to_cols_array();
            upload(&self.inverse, 0, bytemuck::cast_slice(&inverse));
        }
        upload(
            &self.frame,
            0,
            bytemuck::bytes_of(&FrameUniform {
                viewport: [viewport[0] as f32, viewport[1] as f32],
                time: batch.time,
                _pad: 0.0,
                eye: view_eye(view_proj),
                _pad_eye: 0.0,
            }),
        );
        quads
    }

    fn reads_depth(&self, slot: usize) -> bool {
        matches!(self.pipelines.get(slot), Some(Some(c)) if c.scene_depth)
    }

    /// Whether a prepared run reads the scene depth at a stage past `AfterSky`: then the frame
    /// copies it with [`Self::copy_scene_depth`] where `AfterOpaqueBlocks` draws.
    pub fn wants_scene_depth(&self) -> bool {
        self.runs
            .iter()
            .any(|run| run.stage > 0 && self.reads_depth(run.slot))
    }

    /// Copies `depth`, the world's depth after its opaque terrain, for the pipelines of the
    /// stages from `AfterOpaqueBlocks` on, when [`Self::wants_scene_depth`]; it needs
    /// `COPY_SRC`. Outside a pass: the caller ends the world pass first and starts another.
    pub fn copy_scene_depth(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        depth: &wgpu::Texture,
    ) {
        if !self.wants_scene_depth() {
            return;
        }
        let size = depth.size();
        let checked = !super::frame_targets::screenshot_frame();
        if self
            .scene
            .as_ref()
            .is_none_or(|s| s.texture.size() != size || (checked && !s.checked))
        {
            // The old copy goes first, so the two are never held together. Without the room for
            // the new one nothing invalid is kept: this frame reads sky and the next tries again.
            self.scene = None;
            self.scene = super::frame_targets::made(device, Kind::SceneDepth, || {
                let texture = depth_texture(
                    device,
                    [size.width, size.height],
                    wgpu::TextureUsages::COPY_DST,
                );
                let group = depth_group(device, &self.depth_layout, &texture, &self.inverse);
                SceneDepth {
                    texture,
                    group,
                    checked,
                }
            });
        }
        let Some(scene) = &self.scene else {
            return;
        };
        encoder.copy_texture_to_texture(depth.as_image_copy(), scene.texture.as_image_copy(), size);
        self.copied = true;
    }

    /// Draws the prepared runs of stages `first..=last` in the order queued, with `indices`
    /// (6 a quad, `u32`). Leaves the pass with its own pipeline, bind group 0 and buffers, bind
    /// group 1 after a pipeline that reads the depth and bind group 2 after one with a texture
    /// or buffers.
    pub fn draw(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        indices: &wgpu::Buffer,
        first: usize,
        last: usize,
    ) {
        let mut bound = false;
        for run in &self.runs {
            if run.stage < first || run.stage > last {
                continue;
            }
            let Some(Some(Compiled {
                pipeline,
                texture,
                scene_depth,
            })) = self.pipelines.get(run.slot)
            else {
                continue;
            };
            if !bound {
                bound = true;
                pass.set_bind_group(0, &self.group, &[]);
                pass.set_vertex_buffer(0, self.vertices.slice(..));
                pass.set_index_buffer(indices.slice(..), wgpu::IndexFormat::Uint32);
            }
            pass.set_pipeline(pipeline);
            // Only a textured pipeline has group 1; the others draw as they did before it.
            if let Some(texture) = texture {
                pass.set_bind_group(2, texture, &[]);
            }
            // At `AfterSky`, and on a frame that took no copy, the depth is the sky's.
            if *scene_depth {
                let scene = self.scene.as_ref().filter(|_| self.copied && run.stage > 0);
                pass.set_bind_group(1, scene.map_or(&self.sky, |s| &s.group), &[]);
            }
            pass.draw_indexed(run.quads.start * 6..run.quads.end * 6, 0, 0..1);
        }
    }
}

/// Group 2's layout for `built`: its texture, its sampler and where it lies in what is bound,
/// then its buffers; none for a pipeline with neither.
fn resources_layout(device: &wgpu::Device, built: &Built) -> Option<wgpu::BindGroupLayout> {
    let fragment = wgpu::ShaderStages::FRAGMENT;
    let mut entries = Vec::new();
    if built.texture.is_some() {
        entries.extend([
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: fragment,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: fragment,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: fragment,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ]);
    }
    entries.extend(
        (FIRST_BUFFER_BINDING..)
            .take(built.buffers.len())
            .map(|binding| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: fragment,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }),
    );
    (!entries.is_empty()).then(|| {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Mod pipeline resources"),
            entries: &entries,
        })
    })
}

/// A depth texture the scene depth functions can read, `size` pixels big.
fn depth_texture(
    device: &wgpu::Device,
    [width, height]: [u32; 2],
    usage: wgpu::TextureUsages,
) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Mod pipeline scene depth"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: usage | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    })
}

/// Group 1 over `depth`, with the inverse view.
fn depth_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    depth: &wgpu::Texture,
    inverse: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Mod pipeline scene depth"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(
                    &depth.create_view(&Default::default()),
                ),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: inverse.as_entire_binding(),
            },
        ],
    })
}

fn gpu_refusal(built: &Built, error: &str) -> String {
    format!(
        "render pipeline {} from mod {} refused by the GPU: {error}",
        built.id, built.owner
    )
}

impl super::Renderer {
    /// Builds the mods' pipelines of the load this renderer is made for, before the load is
    /// published: the required ones the GPU refuses are the error, by id. A textured pipeline's
    /// sprite is looked up with `sprite` (its `[u0, v0, u1, v1]` in the entity atlas, the 2D
    /// atlas world primitives draw from), and a sprite it doesn't find refuses that pipeline.
    pub fn build_mod_pipelines(
        &mut self,
        set: &Pipelines,
        sprite: &dyn Fn(&str) -> Option<[f32; 4]>,
    ) -> Result<(), String> {
        let textures = TextureSources {
            atlas: &self.entity_atlas,
            sprite,
        };
        self.world_primitives
            .pipelines
            .build(&self.device, set, Some(textures))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gfx::world_primitives::{self, STAGES, WorldPrimitives};
    use glam::Vec3;
    use mod_api::{Identifier, RenderPipeline};

    const SIZE: u32 = 64;
    /// The frame before any quad: a quarter grey, stored as 64 in 8 bits.
    const CLEAR: f64 = 64.0 / 255.0;

    const PASS: &str =
        "fn pipeline_fragment(input: PipelineInput) -> vec4<f32> { return input.color; }";
    const HALF: &str = "fn pipeline_fragment(input: PipelineInput) -> vec4<f32> { return vec4<f32>(vec3<f32>(0.5), 1.0); }";
    const INPUTS: &str = "fn pipeline_fragment(input: PipelineInput) -> vec4<f32> {
    return vec4<f32>(step(0.5, input.uv.x), frame.time, step(0.5, input.uv.y), 1.0);
}";

    fn device() -> Option<(wgpu::Device, wgpu::Queue)> {
        let instance = crate::gfx::test_gpu::instance();
        let adapter = pollster::block_on(instance.request_adapter(&Default::default())).ok()?;
        pollster::block_on(adapter.request_device(&Default::default())).ok()
    }

    /// The load of `generation` with one optional pipeline for each shader, in slot order.
    fn load(generation: u64, shaders: &[(&str, PipelineBlend, WorldDepth)]) -> Pipelines {
        let registered: Vec<_> = (0..shaders.len())
            .map(|slot| {
                let id = Identifier::parse(&format!("test:p{slot}")).unwrap();
                RenderPipeline::new(id.clone(), id, shaders[slot].1, shaders[slot].2).optional()
            })
            .collect();
        Pipelines::build(&registered, generation, |file| {
            let slot: usize = file
                .strip_prefix("assets/test/shaders/pipeline/p")
                .and_then(|rest| rest.strip_suffix(".wgsl"))
                .unwrap()
                .parse()
                .unwrap();
            Ok(shaders[slot].0.to_owned())
        })
        .unwrap()
    }

    thread_local! {
        /// A translation the projection carries, as the view bob's: its rays start there,
        /// off the camera.
        static BOB: std::cell::Cell<Vec3> = const { std::cell::Cell::new(Vec3::ZERO) };
    }

    fn view_proj() -> Mat4 {
        crate::gfx::world_projection(70.0, 1.0, 512.0) * Mat4::from_translation(BOB.get())
    }

    /// A quad facing the camera `z` blocks ahead of it, over the pixels `x0..x1`, `y0..y1` of
    /// the frame (y down), its texture coordinates spanning 0 to 1 and `color` as stored bytes.
    fn quad([x0, y0, x1, y1]: [f32; 4], z: f32, color: [u8; 4]) -> [PipelineVertex; 4] {
        let depth = view_proj().project_point3(Vec3::new(0.0, 0.0, -z)).z;
        let inverse = view_proj().inverse();
        let at = |x: f32, y: f32, uv: [f32; 2]| PipelineVertex {
            pos: inverse
                .project_point3(Vec3::new(
                    x / SIZE as f32 * 2.0 - 1.0,
                    1.0 - y / SIZE as f32 * 2.0,
                    depth,
                ))
                .to_array(),
            uv,
            color: u32::from_le_bytes(color),
        };
        [
            at(x0, y0, [0.0, 0.0]),
            at(x0, y1, [0.0, 1.0]),
            at(x1, y1, [1.0, 1.0]),
            at(x1, y0, [1.0, 0.0]),
        ]
    }

    /// The depth of a wall 5 blocks ahead, as the frame stores it.
    fn wall() -> f32 {
        view_proj().project_point3(Vec3::new(0.0, 0.0, -5.0)).z
    }

    /// Draws `batch`'s stages up to `last` over the cleared frame of `format`, behind a wall
    /// 5 blocks ahead, and reads the frame back as RGBA floats.
    fn render(
        gpu: &(wgpu::Device, wgpu::Queue),
        primitives: &mut WorldPrimitives,
        format: wgpu::TextureFormat,
        batch: Batch<'_>,
        last: usize,
    ) -> impl Fn(u32, u32) -> [f32; 4] + use<> {
        render_over(gpu, primitives, format, batch, last, wall())
    }

    /// [`render`] over a world whose depth is `ground` everywhere, as a frame draws it: the
    /// first stage, then the scene depth's copy where `AfterOpaqueBlocks` draws, then the rest
    /// in a pass that loads both targets.
    fn render_over(
        (device, queue): &(wgpu::Device, wgpu::Queue),
        primitives: &mut WorldPrimitives,
        format: wgpu::TextureFormat,
        batch: Batch<'_>,
        last: usize,
        ground: f32,
    ) -> impl Fn(u32, u32) -> [f32; 4] + use<> {
        let texture = |size: u32, format, usage| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: None,
                size: wgpu::Extent3d {
                    width: size,
                    height: size,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage,
                view_formats: &[],
            })
        };
        let color = texture(
            SIZE,
            format,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        );
        let depth = texture(
            SIZE,
            wgpu::TextureFormat::Depth32Float,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        );
        let atlas = texture(
            1,
            wgpu::TextureFormat::Rgba8Unorm,
            wgpu::TextureUsages::TEXTURE_BINDING,
        );
        if let Some(set) = batch.set {
            primitives.pipelines.build(device, set, None).unwrap();
        }
        let drawn = primitives.prepare(
            device,
            &mut |buffer, offset, bytes| queue.write_buffer(buffer, offset, bytes),
            &world_primitives::Batch {
                pipelines: batch,
                ..Default::default()
            },
            view_proj(),
            [SIZE, SIZE],
            &atlas.create_view(&Default::default()),
        );
        let mut encoder = device.create_command_encoder(&Default::default());
        let color_view = color.create_view(&Default::default());
        let depth_view = depth.create_view(&Default::default());
        let pass = |encoder: &mut wgpu::CommandEncoder, clear: bool| {
            encoder
                .begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: None,
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &color_view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: if clear {
                                wgpu::LoadOp::Clear(wgpu::Color {
                                    r: CLEAR,
                                    g: CLEAR,
                                    b: CLEAR,
                                    a: 1.0,
                                })
                            } else {
                                wgpu::LoadOp::Load
                            },
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &depth_view,
                        depth_ops: Some(wgpu::Operations {
                            load: if clear {
                                wgpu::LoadOp::Clear(ground)
                            } else {
                                wgpu::LoadOp::Load
                            },
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    ..Default::default()
                })
                .forget_lifetime()
        };
        let mut next = 0;
        let mut first = pass(&mut encoder, true);
        if drawn {
            primitives.draw_through(&mut first, &mut next, 0);
        }
        drop(first);
        primitives
            .pipelines
            .copy_scene_depth(device, &mut encoder, &depth);
        let mut rest = pass(&mut encoder, false);
        if drawn {
            primitives.draw_through(&mut rest, &mut next, last);
        }
        drop(rest);
        let float = format == wgpu::TextureFormat::Rgba16Float;
        let texel = if float { 8 } else { 4 };
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: u64::from(SIZE * SIZE * texel),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_texture_to_buffer(
            color.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(SIZE * texel),
                    rows_per_image: None,
                },
            },
            color.size(),
        );
        queue.submit([encoder.finish()]);
        readback.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        let bytes = readback.slice(..).get_mapped_range().unwrap().to_vec();
        move |x: u32, y: u32| {
            let i = ((y * SIZE + x) * texel) as usize;
            std::array::from_fn(|channel| {
                if float {
                    let at = i + channel * 2;
                    half::f16::from_le_bytes([bytes[at], bytes[at + 1]]).to_f32()
                } else {
                    f32::from(bytes[i + channel])
                }
            })
        }
    }

    fn assert_near(got: [f32; 4], expected: [f32; 4], tolerance: f32, what: &str) {
        assert!(
            got.iter()
                .zip(expected)
                .all(|(got, expected)| (got - expected).abs() <= tolerance),
            "{what}: got {got:?}, expected {expected:?}"
        );
    }

    /// One frame with every part of a pipeline's contract in its own corner. Expected bytes
    /// are vanilla's 2.2 encoding worked out by hand: linear 0.5 is 0.5^(1/2.2) = 0.7297,
    /// stored as 186.
    #[test]
    fn gpu_validation_failure_is_contained_and_invalidates_handle() {
        let _gpu = crate::gfx::test_gpu::lock();
        let Some((device, _)) = device() else {
            return;
        };
        let set = load(
            41,
            &[(PASS, PipelineBlend::Translucent, WorldDepth::Tested)],
        );
        let id = Identifier::parse("test:p0").unwrap();
        let handle = set.resolve(&id).unwrap();
        let renderer = ModPipelines::new(&device, wgpu::TextureFormat::Rgba8Unorm);
        let mut broken = set.built()[0].clone().unwrap();
        broken.owner = "owning_mod".into();
        assert_eq!(
            gpu_refusal(&broken, "compiler refusal"),
            "render pipeline test:p0 from mod owning_mod refused by the GPU: compiler refusal"
        );
        broken.shader = "not wgsl".into();
        assert!(renderer.pipeline(&device, &broken, &[], None).is_err());
        assert!(set.invalidate(0));
        assert_eq!(set.resolve(&id), Err(mod_api::RenderPipelineError::Invalid));
        assert_eq!(set.slot(handle), Err(mod_api::RenderPipelineError::Invalid));
    }

    #[test]
    fn optional_gpu_refusal_is_published_only_once() {
        let _gpu = crate::gfx::test_gpu::lock();
        let Some((device, _)) = device() else {
            return;
        };
        let set = load(
            41,
            &[(PASS, PipelineBlend::Translucent, WorldDepth::Tested)],
        );
        let id = Identifier::parse("test:p0").unwrap();
        let handle = set.resolve(&id).unwrap();
        let mut renderer = ModPipelines::new(&device, wgpu::TextureFormat::Rgba8Unorm);
        renderer
            .build_with(&set, |_, _, _| Err("injected refusal".into()))
            .unwrap();
        assert_eq!(renderer.generation, 41);
        assert!(renderer.pipelines[0].is_none());
        assert_eq!(set.slot(handle), Err(mod_api::RenderPipelineError::Invalid));
        assert!(!set.invalidate(0));
        renderer
            .build_with(&set, |_, _, _| panic!("already compiled"))
            .unwrap();
    }

    #[test]
    fn a_later_successful_build_makes_a_refused_slot_usable_again() {
        let _gpu = crate::gfx::test_gpu::lock();
        let Some((device, _)) = device() else {
            return;
        };
        let set = load(
            41,
            &[(PASS, PipelineBlend::Translucent, WorldDepth::Tested)],
        );
        let id = Identifier::parse("test:p0").unwrap();
        let handle = set.resolve(&id).unwrap();
        let mut refusing = ModPipelines::new(&device, wgpu::TextureFormat::Rgba8Unorm);
        refusing
            .build_with(&set, |_, _, _| Err("injected refusal".into()))
            .unwrap();
        assert_eq!(set.slot(handle), Err(mod_api::RenderPipelineError::Invalid));
        let mut recovered = ModPipelines::new(&device, wgpu::TextureFormat::Rgba8Unorm);
        recovered.build(&device, &set, None).unwrap();
        assert!(recovered.pipelines[0].is_some());
        assert_eq!(set.slot(handle), Ok(0));
        assert!(set.resolve(&id).is_ok());
    }

    #[test]
    fn required_gpu_failure_does_not_publish_and_mixed_success_does() {
        let _gpu = crate::gfx::test_gpu::lock();
        let Some((device, _)) = device() else {
            return;
        };
        let set = load(
            41,
            &[(PASS, PipelineBlend::Translucent, WorldDepth::Tested)],
        );
        let mut renderer = ModPipelines::new(&device, wgpu::TextureFormat::Rgba8Unorm);
        renderer.build(&device, &set, None).unwrap();
        let shaders = [
            (PASS, PipelineBlend::Translucent, WorldDepth::Tested),
            (PASS, PipelineBlend::Translucent, WorldDepth::Tested),
        ];
        let mut registered: Vec<_> = (0..shaders.len())
            .map(|slot| {
                let id = Identifier::parse(&format!("test:p{slot}")).unwrap();
                RenderPipeline::new(id.clone(), id, shaders[slot].1, shaders[slot].2).optional()
            })
            .collect();
        registered[0].optional = false;
        let next = Pipelines::build(&registered, 42, |_| Ok(PASS.to_owned())).unwrap();
        let id = Identifier::parse("test:p0").unwrap();
        let handle = next.resolve(&id).unwrap();
        let error = renderer
            .build_with(&next, |_, _, _| Err("injected refusal".into()))
            .unwrap_err();
        assert_eq!(
            error,
            "required render pipeline test:p0 from mod  refused by the GPU: injected refusal"
        );
        assert_eq!(renderer.generation, 41);
        assert_eq!(renderer.pipelines.len(), 1);
        assert_eq!(next.slot(handle), Ok(0));
        assert!(next.resolve(&Identifier::parse("test:p1").unwrap()).is_ok());
        renderer.build(&device, &next, None).unwrap();
        assert_eq!(renderer.generation, 42);
        assert_eq!(renderer.pipelines.len(), 2);
        assert!(renderer.pipelines.iter().all(Option::is_some));
    }

    #[test]
    fn pipelines_convert_colour_both_ways_and_keep_their_blend_depth_and_stage() {
        let _gpu = crate::gfx::test_gpu::lock();
        let Some(gpu) = device() else {
            eprintln!("no GPU adapter: skipped");
            return;
        };
        use PipelineBlend::{Additive, Translucent};
        use WorldDepth::{Tested, ThroughWalls};
        let set = load(
            3,
            &[
                (PASS, Translucent, Tested),
                (HALF, Translucent, ThroughWalls),
                // Refused by the load: the pipelines after it still draw.
                ("fn nothing() {}", Translucent, Tested),
                (HALF, Additive, Tested),
                (INPUTS, Translucent, Tested),
            ],
        );
        assert!(set.built()[2].is_none());
        let grey = [128, 128, 128, 255];
        let white = [255; 4];
        let vertices = [
            // Top left: a grey quad returned as it came, in front of the wall and behind it.
            quad([0.0, 0.0, 16.0, 32.0], 2.0, grey),
            quad([16.0, 0.0, 32.0, 32.0], 10.0, grey),
            // Top right: linear 0.5 through walls, at a later stage.
            quad([32.0, 0.0, 64.0, 32.0], 10.0, white),
            // Bottom left: linear 0.5 added to the frame.
            quad([0.0, 32.0, 32.0, 64.0], 2.0, white),
            // Bottom right: the texture coordinate and the frame's time.
            quad([32.0, 32.0, 64.0, 64.0], 2.0, white),
        ]
        .concat();
        let run = |slot, stage, quads| PipelineRun { slot, stage, quads };
        let runs = [run(0, 0, 2), run(1, 2, 1), run(3, 0, 1), run(4, 0, 1)];
        let batch = Batch {
            set: Some(&set),
            vertices: &vertices,
            runs: &runs,
            time: 1.0,
            ..Default::default()
        };
        let format = wgpu::TextureFormat::Rgba8Unorm;
        let mut primitives = WorldPrimitives::new(&gpu.0, (format, false));

        let at = render(&gpu, &mut primitives, format, batch, STAGES - 1);
        let clear = [64.0, 64.0, 64.0, 255.0];
        assert_near(
            at(8, 16),
            [128.0, 128.0, 128.0, 255.0],
            1.0,
            "decoded on the way in and encoded on the way out, a colour comes back as it was",
        );
        assert_near(
            at(24, 16),
            clear,
            0.0,
            "depth tested: hidden behind the wall",
        );
        assert_near(
            at(48, 16),
            [186.0, 186.0, 186.0, 255.0],
            1.0,
            "linear 0.5 through walls",
        );
        assert_near(
            at(16, 48),
            [250.0, 250.0, 250.0, 255.0],
            1.0,
            "additive: 186 over the frame's 64",
        );
        assert_near(
            at(40, 40),
            [0.0, 255.0, 0.0, 255.0],
            0.0,
            "the quad's first quarter: uv under 0.5, time 1",
        );
        assert_near(
            at(56, 56),
            [255.0, 255.0, 255.0, 255.0],
            0.0,
            "the quad's last quarter: uv over 0.5",
        );

        // Only the first stage: the later stage's quad is not drawn yet.
        let at = render(&gpu, &mut primitives, format, batch, 0);
        assert_near(
            at(48, 16),
            clear,
            0.0,
            "a later stage's quad waits for its stage",
        );
        assert_near(
            at(8, 16),
            [128.0, 128.0, 128.0, 255.0],
            1.0,
            "the first stage draws",
        );
    }

    /// In a float frame light above white survives, stored as vanilla encodes it (linear 2.0
    /// is 2^(1/2.2) = 1.37035), and a new load's shader replaces the old one's pipeline.
    #[test]
    fn a_float_frame_keeps_light_above_white_and_a_reload_builds_the_new_shader() {
        let _gpu = crate::gfx::test_gpu::lock();
        let Some(gpu) = device() else {
            eprintln!("no GPU adapter: skipped");
            return;
        };
        let bright = "fn pipeline_fragment(input: PipelineInput) -> vec4<f32> { return vec4<f32>(2.0, 1.0, 0.0, 1.0); }";
        let green = "fn pipeline_fragment(input: PipelineInput) -> vec4<f32> { return vec4<f32>(0.0, 1.0, 0.0, 1.0); }";
        let state = (PipelineBlend::Translucent, WorldDepth::Tested);
        let vertices = quad([0.0, 0.0, 64.0, 64.0], 2.0, [255; 4]);
        let runs = [PipelineRun {
            slot: 0,
            stage: 4,
            quads: 1,
        }];
        let format = wgpu::TextureFormat::Rgba16Float;
        let mut primitives = WorldPrimitives::new(&gpu.0, (format, false));
        let mut draw = |set: &Pipelines| {
            let batch = Batch {
                set: Some(set),
                vertices: &vertices,
                runs: &runs,
                time: 0.0,
                ..Default::default()
            };
            render(&gpu, &mut primitives, format, batch, STAGES - 1)(32, 32)
        };

        let before = load(1, &[(bright, state.0, state.1)]);
        assert_near(
            draw(&before),
            [1.37035, 1.0, 0.0, 1.0],
            0.002,
            "linear 2.0 in a float frame",
        );
        let after = load(2, &[(green, state.0, state.1)]);
        assert_near(
            draw(&after),
            [0.0, 1.0, 0.0, 1.0],
            0.002,
            "the reloaded shader",
        );
    }

    /// A textured pipeline samples a 2x1 sprite's two texels across its quad, and one naming
    /// half the atlas samples only that sprite's part, repeating it past 1. A sprite the
    /// atlas lacks refuses the optional pipeline, and an untextured one in the same load still
    /// draws as before. Expected bytes: stored red and green stay 255 through decode and encode.
    #[test]
    fn textured_pipelines_sample_their_file_or_their_sprite() {
        use mod_api::PipelineTexture;
        use wgpu::util::DeviceExt;
        let _gpu = crate::gfx::test_gpu::lock();
        let Some(gpu) = device() else {
            eprintln!("no GPU adapter: skipped");
            return;
        };
        let (device, queue) = (&gpu.0, &gpu.1);
        let sample = "fn pipeline_fragment(input: PipelineInput) -> vec4<f32> { return pipeline_texture(input.uv); }";
        let id = |slot: usize| Identifier::parse(&format!("test:p{slot}")).unwrap();
        let state = (PipelineBlend::Translucent, WorldDepth::Tested);
        let pipeline = |slot| RenderPipeline::new(id(slot), id(slot), state.0, state.1).optional();
        let registered = [
            // A 2x1 sprite, the whole atlas: red on the left, green on the right.
            pipeline(0).textured(PipelineTexture::Sprite(id(0))),
            // The atlas's right half, green.
            pipeline(1).textured(PipelineTexture::Sprite(id(1))),
            pipeline(2).textured(PipelineTexture::Sprite(
                Identifier::parse("test:none").unwrap(),
            )),
            pipeline(3),
        ];
        let red_green = vec![255, 0, 0, 255, 0, 255, 0, 255];
        let set = Pipelines::build(&registered, 7, |file| {
            Ok(if file.ends_with("p3.wgsl") {
                PASS
            } else {
                sample
            }
            .to_owned())
        })
        .unwrap();
        let atlas = device.create_texture_with_data(
            queue,
            &wgpu::TextureDescriptor {
                label: None,
                size: wgpu::Extent3d {
                    width: 2,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            },
            wgpu::util::TextureDataOrder::LayerMajor,
            &red_green,
        );
        let atlas = atlas.create_view(&Default::default());
        let sprite = |name: &str| match name {
            "test:p0" => Some([0.0, 0.0, 1.0, 1.0]),
            "test:p1" => Some([0.5, 0.0, 1.0, 1.0]),
            _ => None,
        };
        let format = wgpu::TextureFormat::Rgba8Unorm;
        let mut primitives = WorldPrimitives::new(device, (format, false));
        let sources = TextureSources {
            atlas: &atlas,
            sprite: &sprite,
        };
        primitives
            .pipelines
            .build(device, &set, Some(sources))
            .unwrap();
        assert_eq!(
            set.resolve(&id(2)),
            Err(mod_api::RenderPipelineError::Invalid),
            "a sprite the atlas lacks"
        );
        assert!(
            primitives.pipelines.pipelines[0]
                .as_ref()
                .unwrap()
                .texture
                .is_some()
        );
        assert!(
            primitives.pipelines.pipelines[3]
                .as_ref()
                .unwrap()
                .texture
                .is_none()
        );
        let mut vertices = Vec::new();
        for (row, slot) in [(0.0, 0), (16.0, 1), (32.0, 3)] {
            let mut quad = quad([0.0, row, 64.0, row + 16.0], 2.0, [255, 255, 255, 255]);
            if slot == 1 {
                // Twice across: the sprite repeats.
                quad.iter_mut().for_each(|v| v.uv[0] *= 2.0);
            }
            vertices.extend(quad);
        }
        let runs = [0, 1, 3].map(|slot| PipelineRun {
            slot,
            stage: 4,
            quads: 1,
        });
        let pixel = render(
            &gpu,
            &mut primitives,
            format,
            Batch {
                set: Some(&set),
                vertices: &vertices,
                runs: &runs,
                time: 0.0,
                ..Default::default()
            },
            STAGES - 1,
        );
        let (red, green) = ([255.0, 0.0, 0.0, 255.0], [0.0, 255.0, 0.0, 255.0]);
        assert_near(pixel(8, 8), red, 1.0, "the 2x1 sprite's left texel");
        assert_near(pixel(56, 8), green, 1.0, "the 2x1 sprite's right texel");
        assert_near(pixel(8, 24), green, 1.0, "the sprite, not the atlas's red");
        assert_near(pixel(40, 24), green, 1.0, "the sprite repeated");
        assert_near(pixel(8, 40), [255.0; 4], 1.0, "the untextured pipeline");
    }

    /// The frame's light as a float frame stores it: vanilla's 2.2 encoding.
    fn stored(linear: f32) -> f32 {
        linear.powf(1.0 / 2.2)
    }

    /// A pipeline that reads the scene depth gets the opaque world behind each fragment: its
    /// distance along the fragment's ray and its position, from the copy taken after the first
    /// stage. At that first stage, and over the sky, it reads sky; a frame whose depth-reading
    /// quads are all at the first stage takes no copy.
    #[test]
    fn a_pipeline_reads_the_opaque_depth_behind_its_fragment() {
        let _gpu = crate::gfx::test_gpu::lock();
        let Some(gpu) = device() else {
            eprintln!("no GPU adapter: skipped");
            return;
        };
        // Red: the distance in tens of blocks; green: sky; blue: how far ahead the surface is.
        let reads = "fn pipeline_fragment(input: PipelineInput) -> vec4<f32> {
    let distance = pipeline_scene_distance(input);
    let sky = distance > 1.0e37;
    let ahead = -pipeline_scene_position(input).z;
    return vec4<f32>(select(distance / 10.0, 0.0, sky), select(0.0, 1.0, sky), select(ahead / 10.0, 0.0, sky), 1.0);
}";
        let id = Identifier::parse("test:p0").unwrap();
        let registered = [RenderPipeline::new(
            id.clone(),
            id,
            PipelineBlend::Translucent,
            WorldDepth::ThroughWalls,
        )
        .reads_scene_depth()];
        let set = Pipelines::build(&registered, 5, |_| Ok(reads.to_owned())).unwrap();
        let vertices = [
            quad([0.0, 0.0, 64.0, 32.0], 2.0, [255; 4]),
            quad([0.0, 32.0, 64.0, 64.0], 2.0, [255; 4]),
        ]
        .concat();
        let run = |stage| PipelineRun {
            slot: 0,
            stage,
            quads: 1,
        };
        // The top half at AfterSky, the bottom half at AfterOpaqueBlocks.
        let runs = [run(0), run(1)];
        let batch = Batch {
            set: Some(&set),
            vertices: &vertices,
            runs: &runs,
            time: 0.0,
            ..Default::default()
        };
        let format = wgpu::TextureFormat::Rgba16Float;
        let mut primitives = WorldPrimitives::new(&gpu.0, (format, false));
        let at = render(&gpu, &mut primitives, format, batch, STAGES - 1);
        assert!(primitives.pipelines.wants_scene_depth());
        let sky = [0.0, 1.0, 0.0, 1.0];
        assert_near(at(32, 8), sky, 0.002, "AfterSky reads the sky");
        // The wall 5 blocks ahead, behind the pixel (x, y): along its ray from the camera.
        let behind = |x: u32, y: u32| {
            let screen = Vec3::new(
                (x as f32 + 0.5) / SIZE as f32 * 2.0 - 1.0,
                1.0 - (y as f32 + 0.5) / SIZE as f32 * 2.0,
                wall(),
            );
            view_proj().inverse().project_point3(screen)
        };
        for (x, y) in [(32, 48), (0, 63), (63, 32)] {
            let wall = behind(x, y);
            assert!(
                (wall.z + 5.0).abs() < 1e-3,
                "the wall is 5 blocks ahead: {wall}"
            );
            assert_near(
                at(x, y),
                [stored(wall.length() / 10.0), 0.0, stored(0.5), 1.0],
                0.003,
                &format!(
                    "the wall behind ({x}, {y}), {} blocks along its ray",
                    wall.length()
                ),
            );
        }

        let at = render_over(&gpu, &mut primitives, format, batch, STAGES - 1, 0.0);
        assert_near(at(32, 48), sky, 0.002, "nothing opaque: the sky");

        // A bobbed view's rays start off the camera: the distance runs from where they start.
        BOB.set(Vec3::new(1.5, 1.0, 0.0));
        let bobbed = [
            quad([0.0, 0.0, 64.0, 32.0], 2.0, [255; 4]),
            quad([0.0, 32.0, 64.0, 64.0], 2.0, [255; 4]),
        ]
        .concat();
        let at = render(
            &gpu,
            &mut primitives,
            format,
            Batch {
                vertices: &bobbed,
                ..batch
            },
            STAGES - 1,
        );
        let eye = Vec3::new(-1.5, -1.0, 0.0);
        for (x, y) in [(32, 48), (0, 63), (63, 32)] {
            let along = behind(x, y).distance(eye);
            assert_near(
                at(x, y),
                [stored(along / 10.0), 0.0, stored(0.5), 1.0],
                0.003,
                &format!(
                    "the wall behind ({x}, {y}) in a bobbed view, {along} blocks along its ray"
                ),
            );
        }
        BOB.set(Vec3::ZERO);

        let early = [run(0), run(0)];
        let batch = Batch {
            runs: &early,
            ..batch
        };
        drop(render(&gpu, &mut primitives, format, batch, STAGES - 1));
        assert!(
            !primitives.pipelines.wants_scene_depth(),
            "AfterSky alone needs no copy"
        );
    }

    /// A depth-reading pipeline drawn right after a textured one in the same pass still reads
    /// the scene depth: switching pipelines whose texture groups differ keeps the depth bound.
    #[test]
    fn a_depth_reader_after_a_textured_one_still_reads_the_depth() {
        use mod_api::PipelineTexture;
        use wgpu::util::DeviceExt;
        let _gpu = crate::gfx::test_gpu::lock();
        let Some(gpu) = device() else {
            eprintln!("no GPU adapter: skipped");
            return;
        };
        let (device, queue) = (&gpu.0, &gpu.1);
        // Green over the wall, red over the sky; the textured one adds its white texel's nothing.
        let wall = "vec4<f32>(0.0, 1.0, 0.0, 1.0), pipeline_scene_depth(input) > 0.0)";
        let textured = format!(
            "fn pipeline_fragment(input: PipelineInput) -> vec4<f32> {{
    return pipeline_texture(input.uv) * 0.0 + select(vec4<f32>(1.0, 0.0, 0.0, 1.0), {wall};
}}"
        );
        let plain = format!(
            "fn pipeline_fragment(input: PipelineInput) -> vec4<f32> {{
    return select(vec4<f32>(1.0, 0.0, 0.0, 1.0), {wall};
}}"
        );
        let id = |slot: usize| Identifier::parse(&format!("test:p{slot}")).unwrap();
        let pipeline = |slot| {
            RenderPipeline::new(
                id(slot),
                id(slot),
                PipelineBlend::Translucent,
                WorldDepth::ThroughWalls,
            )
            .reads_scene_depth()
        };
        let registered = [
            pipeline(0).textured(PipelineTexture::Sprite(id(0))),
            pipeline(1),
        ];
        let set = Pipelines::build(&registered, 9, |file| {
            Ok(if file.ends_with("p0.wgsl") {
                &textured
            } else {
                &plain
            }
            .clone())
        })
        .unwrap();
        let atlas = device.create_texture_with_data(
            queue,
            &wgpu::TextureDescriptor {
                label: None,
                size: wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            },
            wgpu::util::TextureDataOrder::LayerMajor,
            &[255; 4],
        );
        let atlas = atlas.create_view(&Default::default());
        let sprite = |_: &str| Some([0.0, 0.0, 1.0, 1.0]);
        let format = wgpu::TextureFormat::Rgba8Unorm;
        let mut primitives = WorldPrimitives::new(device, (format, false));
        let sources = TextureSources {
            atlas: &atlas,
            sprite: &sprite,
        };
        primitives
            .pipelines
            .build(device, &set, Some(sources))
            .unwrap();
        let vertices = [
            quad([0.0, 0.0, 32.0, 64.0], 2.0, [255; 4]),
            quad([32.0, 0.0, 64.0, 64.0], 2.0, [255; 4]),
        ]
        .concat();
        let runs = [0, 1].map(|slot| PipelineRun {
            slot,
            stage: 1,
            quads: 1,
        });
        let batch = Batch {
            set: Some(&set),
            vertices: &vertices,
            runs: &runs,
            time: 0.0,
            ..Default::default()
        };
        let at = render(&gpu, &mut primitives, format, batch, STAGES - 1);
        let green = [0.0, 255.0, 0.0, 255.0];
        assert_near(at(16, 32), green, 0.0, "the textured reader");
        assert_near(at(48, 32), green, 0.0, "the plain reader after it");
    }

    /// Without the video memory for the copy, the frame's depth-reading pipelines read sky and
    /// nothing invalid is kept: the next frame makes the copy and reads the wall.
    #[test]
    fn a_scene_depth_copy_without_room_reads_sky_and_is_made_again() {
        let _gpu = crate::gfx::test_gpu::lock();
        let Some(gpu) = device() else {
            eprintln!("no GPU adapter: skipped");
            return;
        };
        let reads = "fn pipeline_fragment(input: PipelineInput) -> vec4<f32> {
    return select(vec4<f32>(1.0, 0.0, 0.0, 1.0), vec4<f32>(0.0, 1.0, 0.0, 1.0), pipeline_scene_depth(input) > 0.0);
}";
        let id = Identifier::parse("test:p0").unwrap();
        let registered = [RenderPipeline::new(
            id.clone(),
            id,
            PipelineBlend::Translucent,
            WorldDepth::ThroughWalls,
        )
        .reads_scene_depth()];
        let set = Pipelines::build(&registered, 5, |_| Ok(reads.to_owned())).unwrap();
        let vertices = quad([0.0, 0.0, 64.0, 64.0], 2.0, [255; 4]);
        let runs = [PipelineRun {
            slot: 0,
            stage: 1,
            quads: 1,
        }];
        let batch = Batch {
            set: Some(&set),
            vertices: &vertices,
            runs: &runs,
            time: 0.0,
            ..Default::default()
        };
        let format = wgpu::TextureFormat::Rgba8Unorm;
        let mut primitives = WorldPrimitives::new(&gpu.0, (format, false));
        let (sky, wall) = ([255.0, 0.0, 0.0, 255.0], [0.0, 255.0, 0.0, 255.0]);
        crate::gfx::frame_targets::NO_ROOM.set(1);
        let at = render(&gpu, &mut primitives, format, batch, STAGES - 1);
        crate::gfx::frame_targets::NO_ROOM.set(0);
        assert_near(at(32, 32), sky, 0.0, "no room for the copy: the sky");
        assert!(primitives.pipelines.scene.is_none(), "nothing kept");
        let at = render(&gpu, &mut primitives, format, batch, STAGES - 1);
        assert_near(at(32, 32), wall, 0.0, "the next frame makes it");

        // A scaled screenshot's frame makes it unchecked (its own scopes catch a failure), so
        // the next window frame makes it again rather than keep what may be invalid.
        primitives.pipelines.scene = None;
        let at = crate::gfx::frame_targets::in_screenshot(|| {
            render(&gpu, &mut primitives, format, batch, STAGES - 1)
        });
        assert_near(at(32, 32), wall, 0.0, "the screenshot reads the wall");
        assert!(
            primitives
                .pipelines
                .scene
                .as_ref()
                .is_some_and(|s| !s.checked)
        );
        drop(render(&gpu, &mut primitives, format, batch, STAGES - 1));
        assert!(
            primitives
                .pipelines
                .scene
                .as_ref()
                .is_some_and(|s| s.checked),
            "made again, checked, by the next window frame"
        );
    }

    /// `input.front_facing` is true on the side a quad's corners run top left, bottom left,
    /// bottom right, top right, as world quads are given, and false on the other.
    #[test]
    fn front_facing_tells_the_two_sides_of_a_quad_apart() {
        let _gpu = crate::gfx::test_gpu::lock();
        let Some(gpu) = device() else {
            eprintln!("no GPU adapter: skipped");
            return;
        };
        let side = "fn pipeline_fragment(input: PipelineInput) -> vec4<f32> {
    return select(vec4<f32>(1.0, 0.0, 0.0, 1.0), vec4<f32>(0.0, 1.0, 0.0, 1.0), input.front_facing);
}";
        let set = load(
            6,
            &[(side, PipelineBlend::Translucent, WorldDepth::ThroughWalls)],
        );
        let front = quad([0.0, 0.0, 32.0, 64.0], 2.0, [255; 4]);
        let mut back = quad([32.0, 0.0, 64.0, 64.0], 2.0, [255; 4]);
        back.reverse();
        let vertices = [front, back].concat();
        let runs = [PipelineRun {
            slot: 0,
            stage: 4,
            quads: 2,
        }];
        let format = wgpu::TextureFormat::Rgba8Unorm;
        let mut primitives = WorldPrimitives::new(&gpu.0, (format, false));
        let batch = Batch {
            set: Some(&set),
            vertices: &vertices,
            runs: &runs,
            time: 0.0,
            ..Default::default()
        };
        let at = render(&gpu, &mut primitives, format, batch, STAGES - 1);
        assert_near(at(16, 32), [0.0, 255.0, 0.0, 255.0], 0.0, "the front");
        assert_near(at(48, 32), [255.0, 0.0, 0.0, 255.0], 0.0, "the back");
    }

    /// A pipeline reads what the mod wrote into its buffer; a later frame's write of part of it
    /// keeps the rest. What was written outlives a frame whose uploads were lost and a resource
    /// reload, which builds a new renderer.
    #[test]
    fn a_pipeline_reads_what_the_mod_wrote_and_a_partial_write_keeps_the_rest() {
        let _gpu = crate::gfx::test_gpu::lock();
        let Some(gpu) = device() else {
            eprintln!("no GPU adapter: skipped");
            return;
        };
        let reads = "fn pipeline_fragment(input: PipelineInput) -> vec4<f32> {
    return select(data[0], data[1], input.uv.y > 0.5);
}";
        // An id of its own: what was written is kept by pipeline id, for the whole process.
        let id = Identifier::parse("test:written").unwrap();
        let kept_id = id.clone();
        let registered = [RenderPipeline::new(
            id.clone(),
            id,
            PipelineBlend::Translucent,
            WorldDepth::ThroughWalls,
        )
        .reads_buffer("data", 32)];
        let load = |generation| {
            Pipelines::build(&registered, generation, |_| Ok(reads.to_owned())).unwrap()
        };
        let vertices = quad([0.0, 0.0, 64.0, 64.0], 2.0, [255; 4]);
        let runs = [PipelineRun {
            slot: 0,
            stage: 4,
            quads: 1,
        }];
        let format = wgpu::TextureFormat::Rgba8Unorm;
        let mut primitives = WorldPrimitives::new(&gpu.0, (format, false));
        let floats = |values: &[[f32; 4]]| -> Vec<u8> { bytemuck::cast_slice(values).to_vec() };
        let (red, green, blue) = (
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
        );
        let draw = |primitives: &mut WorldPrimitives,
                    set: &Pipelines,
                    writes: &[PipelineBufferWrite],
                    bytes: &[u8]| {
            // As the game does: each write is kept as it is queued.
            for write in writes {
                let at = write.bytes.start as usize..write.bytes.end as usize;
                keep_write(&kept_id, write.buffer, 32, write.offset, &bytes[at]);
            }
            let batch = Batch {
                set: Some(set),
                vertices: &vertices,
                runs: &runs,
                writes,
                bytes,
                ..Default::default()
            };
            let at = render(&gpu, primitives, format, batch, STAGES - 1);
            (at(32, 8), at(32, 56))
        };
        let write = |offset, bytes| PipelineBufferWrite {
            slot: 0,
            buffer: 0,
            offset,
            bytes,
        };
        let first = load(1);
        let both = floats(&[red, green]);
        let (top, bottom) = draw(&mut primitives, &first, &[write(0, 0..32)], &both);
        assert_near(top, [255.0, 0.0, 0.0, 255.0], 0.0, "the first vec4");
        assert_near(bottom, [0.0, 255.0, 0.0, 255.0], 0.0, "the second vec4");

        let second = floats(&[blue]);
        let (top, bottom) = draw(&mut primitives, &first, &[write(16, 0..16)], &second);
        assert_near(
            top,
            [255.0, 0.0, 0.0, 255.0],
            0.0,
            "kept from the last frame",
        );
        assert_near(bottom, [0.0, 0.0, 255.0, 255.0], 0.0, "written over");

        // A frame that returned before preparing: its write was kept, never queued to a frame.
        keep_write(&kept_id, 0, 32, 0, bytemuck::cast_slice(&[green]));
        let (top, bottom) = draw(&mut primitives, &first, &[], &[]);
        assert_near(
            top,
            [0.0, 255.0, 0.0, 255.0],
            0.0,
            "a write of a skipped frame",
        );
        assert_near(bottom, [0.0, 0.0, 255.0, 255.0], 0.0, "the rest as it was");
        keep_write(&kept_id, 0, 32, 0, bytemuck::cast_slice(&[red]));
        let (top, _) = draw(&mut primitives, &first, &[], &[]);
        assert_near(top, [255.0, 0.0, 0.0, 255.0], 0.0, "written back");

        // An empty write is accepted and never queued: no frame missed it, so none uploads every
        // buffer again (the GPU's copy, cleared behind the renderer's back, stays cleared).
        keep_write(&kept_id, 0, 32, 0, &[]);
        gpu.1
            .write_buffer(&primitives.pipelines.buffers[0][0], 0, &[0; 32]);
        let (top, _) = draw(&mut primitives, &first, &[], &[]);
        assert!(top[0] < 128.0, "uploaded again for an empty write: {top:?}");

        // A frame given up: its uploads never reached the buffer.
        gpu.1
            .write_buffer(&primitives.pipelines.buffers[0][0], 0, &[0; 32]);
        primitives.pipelines.forget();
        let (top, bottom) = draw(&mut primitives, &first, &[], &[]);
        assert_near(
            top,
            [255.0, 0.0, 0.0, 255.0],
            0.0,
            "uploaded again after a lost frame",
        );
        assert_near(
            bottom,
            [0.0, 0.0, 255.0, 255.0],
            0.0,
            "uploaded again after a lost frame",
        );

        let mut reloaded = WorldPrimitives::new(&gpu.0, (format, false));
        let (top, bottom) = draw(&mut reloaded, &load(2), &[], &[]);
        assert_near(top, [255.0, 0.0, 0.0, 255.0], 0.0, "kept in a new renderer");
        assert_near(
            bottom,
            [0.0, 0.0, 255.0, 255.0],
            0.0,
            "kept in a new renderer",
        );
    }

    /// A depth-writing pipeline's surface hides what draws behind it afterwards, at the depth of
    /// the point it returned rather than its quad's: a quad between that point and the wall is
    /// hidden, one in front of it is drawn.
    #[test]
    fn a_depth_writing_pipeline_hides_what_draws_behind_its_surface() {
        let _gpu = crate::gfx::test_gpu::lock();
        let Some(gpu) = device() else {
            eprintln!("no GPU adapter: skipped");
            return;
        };
        // Blue, on a surface 3 blocks ahead behind a quad 2 ahead.
        let surface = "fn pipeline_fragment(input: PipelineInput) -> PipelineSurface {
    return PipelineSurface(vec4<f32>(0.0, 0.0, 1.0, 1.0), input.position * 1.5);
}";
        let id = |name: &str| Identifier::parse(name).unwrap();
        let registered = [
            RenderPipeline::new(
                id("test:surface"),
                id("test:surface"),
                PipelineBlend::Translucent,
                WorldDepth::Tested,
            )
            .writes_depth(),
            RenderPipeline::new(
                id("test:plain"),
                id("test:plain"),
                PipelineBlend::Translucent,
                WorldDepth::Tested,
            ),
        ];
        let set = Pipelines::build(&registered, 4, |file| {
            Ok(if file.ends_with("surface.wgsl") {
                surface
            } else {
                PASS
            }
            .to_owned())
        })
        .unwrap();
        let red = [255, 0, 0, 255];
        let vertices = [
            quad([0.0, 0.0, 64.0, 64.0], 2.0, [255; 4]),
            // Behind the surface (and in front of the wall 5 ahead), just behind it, just in front.
            quad([0.0, 0.0, 21.0, 64.0], 4.0, red),
            quad([21.0, 0.0, 42.0, 64.0], 3.1, red),
            quad([42.0, 0.0, 64.0, 64.0], 2.9, red),
        ]
        .concat();
        let runs = [
            PipelineRun {
                slot: 0,
                stage: 4,
                quads: 1,
            },
            PipelineRun {
                slot: 1,
                stage: 4,
                quads: 3,
            },
        ];
        let batch = Batch {
            set: Some(&set),
            vertices: &vertices,
            runs: &runs,
            ..Default::default()
        };
        let format = wgpu::TextureFormat::Rgba8Unorm;
        let mut primitives = WorldPrimitives::new(&gpu.0, (format, false));
        let at = render(&gpu, &mut primitives, format, batch, STAGES - 1);
        let (blue, red) = ([0.0, 0.0, 255.0, 255.0], [255.0, 0.0, 0.0, 255.0]);
        assert_near(at(10, 32), blue, 0.0, "behind the surface: hidden");
        assert_near(
            at(31, 32),
            blue,
            0.0,
            "a tenth of a block behind it: hidden",
        );
        assert_near(
            at(53, 32),
            red,
            0.0,
            "a tenth of a block in front of it: drawn",
        );
    }

    /// A depth-writing pipeline that returns a position not a number or infinite draws nothing
    /// there, and writes no depth: the quad behind it is drawn.
    #[test]
    fn a_depth_writing_pipeline_draws_nothing_at_a_position_not_finite() {
        let _gpu = crate::gfx::test_gpu::lock();
        let Some(gpu) = device() else {
            eprintln!("no GPU adapter: skipped");
            return;
        };
        // NaN on the left third, +infinity in the middle, -infinity on the right; blue.
        let surface = "fn pipeline_fragment(input: PipelineInput) -> PipelineSurface {
    var bits = 0x7fc00000u;
    if input.uv.x > 0.66 { bits = 0xff800000u; } else if input.uv.x > 0.33 { bits = 0x7f800000u; }
    let odd = vec3(bitcast<f32>(bits));
    return PipelineSurface(vec4<f32>(0.0, 0.0, 1.0, 1.0), input.position + odd);
}";
        let id = |name: &str| Identifier::parse(name).unwrap();
        let registered = [
            RenderPipeline::new(
                id("test:odd"),
                id("test:odd"),
                PipelineBlend::Translucent,
                WorldDepth::Tested,
            )
            .writes_depth(),
            RenderPipeline::new(
                id("test:plain"),
                id("test:plain"),
                PipelineBlend::Translucent,
                WorldDepth::Tested,
            ),
        ];
        let set = Pipelines::build(&registered, 5, |file| {
            Ok(if file.ends_with("odd.wgsl") {
                surface
            } else {
                PASS
            }
            .to_owned())
        })
        .unwrap();
        let vertices = [
            quad([0.0, 0.0, 64.0, 64.0], 2.0, [255; 4]),
            quad([0.0, 0.0, 64.0, 64.0], 4.0, [255, 0, 0, 255]),
        ]
        .concat();
        let runs = [
            PipelineRun {
                slot: 0,
                stage: 4,
                quads: 1,
            },
            PipelineRun {
                slot: 1,
                stage: 4,
                quads: 1,
            },
        ];
        let batch = Batch {
            set: Some(&set),
            vertices: &vertices,
            runs: &runs,
            ..Default::default()
        };
        let format = wgpu::TextureFormat::Rgba8Unorm;
        let mut primitives = WorldPrimitives::new(&gpu.0, (format, false));
        let at = render(&gpu, &mut primitives, format, batch, STAGES - 1);
        for (x, what) in [(10, "NaN"), (32, "+infinity"), (53, "-infinity")] {
            assert_near(at(x, 32), [255.0, 0.0, 0.0, 255.0], 0.0, what);
        }
    }

    /// A pipeline whose buffers the device has no room for is refused as the GPU refuses a
    /// shader: an optional one resolves invalid and the others build.
    #[test]
    fn a_pipeline_without_room_for_its_buffers_is_refused() {
        let _gpu = crate::gfx::test_gpu::lock();
        let Some(gpu) = device() else {
            eprintln!("no GPU adapter: skipped");
            return;
        };
        let reads = "fn pipeline_fragment(input: PipelineInput) -> vec4<f32> { return data[0]; }";
        let id = |name: &str| Identifier::parse(name).unwrap();
        let pipeline = |name: &str| {
            RenderPipeline::new(
                id(name),
                id(name),
                PipelineBlend::Translucent,
                WorldDepth::Tested,
            )
            .optional()
        };
        let registered = [
            pipeline("test:roomless").reads_buffer("data", 16),
            pipeline("test:plain"),
        ];
        let set = Pipelines::build(&registered, 3, |file| {
            Ok(if file.ends_with("plain.wgsl") {
                PASS
            } else {
                reads
            }
            .to_owned())
        })
        .unwrap();
        let mut renderer = ModPipelines::new(&gpu.0, wgpu::TextureFormat::Rgba8Unorm);
        NO_ROOM.set(1);
        renderer.build(&gpu.0, &set, None).unwrap();
        NO_ROOM.set(0);
        assert_eq!(
            set.resolve(&id("test:roomless")),
            Err(mod_api::RenderPipelineError::Invalid)
        );
        assert!(set.resolve(&id("test:plain")).is_ok());
    }

    /// A real device made with one storage buffer per shader stage, as a GPU below the
    /// pipeline's count reports it: the optional pipeline reading two is refused before anything
    /// is made for it, with no error on the device, while one reading a single buffer and a
    /// plain one build.
    #[test]
    fn an_optional_pipeline_reading_more_buffers_than_the_gpu_allows_is_refused() {
        let _gpu = crate::gfx::test_gpu::lock();
        let instance = crate::gfx::test_gpu::instance();
        let Ok(adapter) = pollster::block_on(instance.request_adapter(&Default::default())) else {
            eprintln!("no GPU adapter: skipped");
            return;
        };
        let descriptor = wgpu::DeviceDescriptor {
            required_limits: wgpu::Limits {
                max_storage_buffers_per_shader_stage: 1,
                ..Default::default()
            },
            ..Default::default()
        };
        let (device, _queue) = pollster::block_on(adapter.request_device(&descriptor)).unwrap();
        assert_eq!(device.limits().max_storage_buffers_per_shader_stage, 1);
        let id = |name: &str| Identifier::parse(name).unwrap();
        let pipeline = |name: &str| {
            RenderPipeline::new(
                id(name),
                id(name),
                PipelineBlend::Translucent,
                WorldDepth::Tested,
            )
            .optional()
        };
        let registered = [
            pipeline("test:two")
                .reads_buffer("a", 16)
                .reads_buffer("b", 16),
            pipeline("test:one").reads_buffer("a", 16),
            pipeline("test:plain"),
        ];
        let set = Pipelines::build(&registered, 3, |file| {
            Ok(if file.ends_with("two.wgsl") {
                "fn pipeline_fragment(input: PipelineInput) -> vec4<f32> { return a[0] + b[0]; }"
            } else if file.ends_with("one.wgsl") {
                "fn pipeline_fragment(input: PipelineInput) -> vec4<f32> { return a[0]; }"
            } else {
                PASS
            }
            .to_owned())
        })
        .unwrap();
        let errors = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let mut renderer = ModPipelines::new(&device, wgpu::TextureFormat::Rgba8Unorm);
        renderer.build(&device, &set, None).unwrap();
        assert!(pollster::block_on(errors.pop()).is_none());
        assert_eq!(
            set.resolve(&id("test:two")),
            Err(mod_api::RenderPipelineError::Invalid)
        );
        assert!(set.resolve(&id("test:one")).is_ok());
        assert!(set.resolve(&id("test:plain")).is_ok());
    }
}
