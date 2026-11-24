//! In-game proof for #19057's mod pipelines (#19758, #19767, #19802, #19913), built per branch:
//! blocks marked `//@N{` .. `//@}` need part N and are cut from the earlier parts' builds.
//! Near the player it draws, at `AfterOpaqueBlocks`:
//! - `proof:shore`, a flat translucent plane just under the eyes that reads the scene depth;
//! - a plain cyan `WorldQuad` 6 blocks east (no pipeline);
//! - `proof:stripes`, four stripes from a buffer the mod writes once (`/proof recolor` writes one);
//! - `proof:ball`, a sphere traced in a proxy quad that writes its depth, half in the ground,
//!   crossed by a `proof:plain` quad drawn after it (`/proof ball sky` draws it at `AfterSky`);
//! - `proof:deck`, drawn over the whole view at `AfterTranslucentBlocks` (`/proof deck <y>`).
use mod_api::__private::native::BuiltMod;
use mod_api::{
    ClientHost, CommandEvent, Flow, Hooks, Identifier, ModEntry, PipelineBlend, PipelineQuad,
    RegisterClientCommandsEvent, RenderLevelStage, RenderLevelStageEvent, RenderPipeline,
    RenderPipelineRef, WorldDepth,
};
use std::sync::Mutex;

const BUILT: &[BuiltMod] = &[BuiltMod { id: "proof", version: "1.0.0", side: mod_api::Side::Client }];

#[derive(Default)]
struct State {
    shore: Option<RenderPipelineRef>,
    //@2{
    stripes: Option<RenderPipelineRef>,
    written: bool,
    recolor: bool,
    //@}
    //@3{
    ball: Option<RenderPipelineRef>,
    plain: Option<RenderPipelineRef>,
    /// Draw the ball at `AfterSky` instead of `AfterOpaqueBlocks`.
    ball_sky: bool,
    no_ball: bool,
    //@}
    //@4{
    deck: Option<RenderPipelineRef>,
    deck_y: Option<f64>,
    //@}
    anchor: Option<[f64; 3]>,
    off: bool,
}

#[derive(Default)]
struct Proof(Mutex<State>);

fn id(text: &str) -> Identifier {
    Identifier::parse(text).unwrap()
}

#[allow(dead_code)]
fn floats(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|c| c.to_le_bytes()).collect()
}

impl Proof {
    fn resolve(&self, host: &mut dyn ClientHost) {
        let resolve = |name: &str| {
            let handle = host.render_assets().render_pipeline(&id(name));
            log::info!("proof: {name} resolved: {handle:?}");
            handle.ok()
        };
        let shore = resolve("proof:shore");
        //@2{
        let stripes = resolve("proof:stripes");
        //@}
        //@3{
        let (ball, plain) = (resolve("proof:ball"), resolve("proof:plain"));
        //@}
        //@4{
        let deck = resolve("proof:deck");
        //@}
        let mut state = self.0.lock().unwrap();
        state.shore = shore;
        //@2{
        state.stripes = stripes;
        //@}
        //@3{
        (state.ball, state.plain) = (ball, plain);
        //@}
        //@4{
        state.deck = deck;
        //@}
    }
}

/// A square of half size `half` facing the camera at `camera`, centred on `center`, corners top
/// left, bottom left, bottom right, top right as seen.
#[allow(dead_code)]
fn facing(camera: [f64; 3], center: [f64; 3], half: f64) -> [[f64; 3]; 4] {
    let f = [center[0] - camera[0], center[1] - camera[1], center[2] - camera[2]];
    let len = (f[0] * f[0] + f[1] * f[1] + f[2] * f[2]).sqrt();
    let f = [f[0] / len, f[1] / len, f[2] / len];
    // right = f x up, up' = right x f
    let r = [-f[2], 0.0, f[0]];
    let rl = (r[0] * r[0] + r[2] * r[2]).sqrt().max(1e-6);
    let r = [r[0] / rl, 0.0, r[2] / rl];
    let u = [r[1] * f[2] - r[2] * f[1], r[2] * f[0] - r[0] * f[2], r[0] * f[1] - r[1] * f[0]];
    let at = |a: f64, b: f64| {
        [
            center[0] + (r[0] * a + u[0] * b) * half,
            center[1] + (r[1] * a + u[1] * b) * half,
            center[2] + (r[2] * a + u[2] * b) * half,
        ]
    };
    [at(-1.0, 1.0), at(-1.0, -1.0), at(1.0, -1.0), at(1.0, 1.0)]
}

#[mod_api::__subscriptions]
impl Hooks for Proof {
    fn register_render_pipelines(
        &self,
        _: &mod_api::RegistrationContext,
        event: &mut mod_api::RegisterRenderPipelinesEvent<'_>,
    ) {
        let new = |name: &str| {
            RenderPipeline::new(id(name), id(name), PipelineBlend::Translucent, WorldDepth::Tested).optional()
        };
        event.pipelines.push(new("proof:shore").reads_scene_depth());
        //@2{
        event.pipelines.push(new("proof:stripes").reads_buffer("colors", 64));
        //@}
        //@3{
        event.pipelines.push(new("proof:ball").reads_buffer("ball", 16).writes_depth());
        event.pipelines.push(new("proof:plain"));
        //@}
        //@4{
        event.pipelines.push(new("proof:deck").reads_scene_depth().reads_buffer("deck", 16));
        //@}
    }

    fn client_setup(&self, host: &mut dyn ClientHost, _: &mut mod_api::ClientSetupEvent) {
        self.resolve(host);
    }

    fn client_resources_reloaded(
        &self,
        host: &mut dyn ClientHost,
        _: &mut mod_api::ClientResourcesReloadedEvent,
    ) {
        self.resolve(host);
    }

    fn register_client_commands(
        &self,
        _: &mut dyn ClientHost,
        event: &mut RegisterClientCommandsEvent<'_>,
    ) {
        event.commands.push(mod_api::Command::new("proof"));
    }

    fn command(&self, _: &mut dyn ClientHost, event: &mut CommandEvent<'_>) -> Flow {
        let mut words = event.command.split_whitespace();
        if words.next() != Some("proof") {
            return Flow::Continue;
        }
        let mut state = self.0.lock().unwrap();
        match words.next() {
            Some("off") => state.off = true,
            Some("on") => state.off = false,
            Some("here") => state.anchor = None,
            //@2{
            Some("recolor") => state.recolor = true,
            //@}
            //@3{
            Some("ball") => match words.next() {
                Some("sky") => (state.ball_sky, state.no_ball) = (true, false),
                Some("off") => state.no_ball = true,
                _ => (state.ball_sky, state.no_ball) = (false, false),
            },
            //@}
            //@4{
            Some("deck") => state.deck_y = words.next().and_then(|y| y.parse().ok()),
            //@}
            other => log::warn!("proof: unknown argument {other:?}"),
        }
        log::info!("proof: command {:?} done", event.command);
        Flow::Cancel
    }

    fn render_level_stage(&self, _: &dyn ClientHost, event: &mut RenderLevelStageEvent<'_>) {
        let mut state = self.0.lock().unwrap();
        let camera = event.view.camera;
        //@4{
        if event.stage == RenderLevelStage::AfterTranslucentBlocks
            && let (Some(deck), Some(y)) = (state.deck, state.deck_y)
        {
            let above = (y - camera[1]) as f32;
            if let Err(error) = event.draw.write_pipeline_buffer(deck, 0, 0, &floats(&[above, 0.0, 0.0, 0.0])) {
                log::warn!("proof: deck height not written: {error:?}");
            }
            if let Err(error) = event.draw.pipeline_screen(deck) {
                log::warn!("proof: deck not drawn: {error:?}");
            }
        }
        //@}
        if state.off {
            return;
        }
        let Some([x, y, z]) = state.anchor else {
            if event.stage == RenderLevelStage::AfterSky {
                state.anchor = Some([camera[0], camera[1] - 1.0, camera[2]]);
            }
            return;
        };
        //@3{
        // The ball: half in the ground 10 blocks north and 4 west of the anchor, radius 2.5.
        let center = [x - 4.0, y - 0.6, z - 10.0];
        let ball_stage = if state.ball_sky {
            RenderLevelStage::AfterSky
        } else {
            RenderLevelStage::AfterOpaqueBlocks
        };
        if event.stage == ball_stage
            && !state.no_ball
            && let Some(ball) = state.ball
        {
            let rel = [center[0] - camera[0], center[1] - camera[1], center[2] - camera[2]];
            let bytes = floats(&[rel[0] as f32, rel[1] as f32, rel[2] as f32, 2.5]);
            if let Err(error) = event.draw.write_pipeline_buffer(ball, 0, 0, &bytes) {
                log::warn!("proof: ball not written: {error:?}");
            }
            let proxy = facing(camera, center, 2.5 * 1.6);
            if let Err(error) = event.draw.pipeline_quads(ball, &[PipelineQuad::spanning(proxy, !0)]) {
                log::warn!("proof: ball not drawn: {error:?}");
            }
        }
        //@}
        if event.stage != RenderLevelStage::AfterOpaqueBlocks {
            return;
        }
        // A plain quad (no pipeline) 6 blocks east.
        let (e, top, bottom) = (x + 6.0, y + 4.5, y + 1.5);
        let plain = [[e, top, z - 3.0], [e, bottom, z - 3.0], [e, bottom, z + 3.0], [e, top, z + 3.0]];
        if let Err(error) = event.draw.quads(&[mod_api::WorldQuad::new(plain, 0xFF20E0E0, WorldDepth::Tested)]) {
            log::warn!("proof: plain quad not drawn: {error:?}");
        }
        if let Some(shore) = state.shore {
            let r = 40.0;
            let corners = [[x - r, y, z - r], [x - r, y, z + r], [x + r, y, z + r], [x + r, y, z - r]];
            if let Err(error) = event.draw.pipeline_quads(shore, &[PipelineQuad::spanning(corners, 0xFFE040C0)]) {
                log::warn!("proof: shore not drawn: {error:?}");
            }
        }
        //@3{
        // A white pipeline quad through the ball's centre, east to west, drawn after it: the part
        // inside the ball is hidden by the depth the ball wrote.
        if !state.no_ball
            && let Some(plain) = state.plain
        {
            let (w, e, top, bottom) = (center[0] - 4.5, center[0] + 4.5, center[1] + 1.2, center[1] - 1.2);
            let n = center[2];
            let corners = [[w, top, n], [w, bottom, n], [e, bottom, n], [e, top, n]];
            if let Err(error) = event.draw.pipeline_quads(plain, &[PipelineQuad::spanning(corners, 0xFFFFFFFF)]) {
                log::warn!("proof: crossing quad not drawn: {error:?}");
            }
        }
        //@}
        //@2{
        if let Some(stripes) = state.stripes {
            if !state.written {
                let colors = [
                    1.0, 0.1, 0.1, 1.0, 0.1, 1.0, 0.1, 1.0, 0.1, 0.3, 1.0, 1.0, 1.0, 0.9, 0.1, 1.0,
                ];
                match event.draw.write_pipeline_buffer(stripes, 0, 0, &floats(&colors)) {
                    Ok(()) => state.written = true,
                    Err(error) => log::warn!("proof: colours not written: {error:?}"),
                }
            }
            if std::mem::take(&mut state.recolor) {
                if let Err(error) = event.draw.write_pipeline_buffer(stripes, 0, 16, &floats(&[1.0; 4])) {
                    log::warn!("proof: recolour not written: {error:?}");
                }
            }
            let (n, top, bottom) = (z - 6.0, y + 4.5, y + 1.5);
            let corners = [[x - 3.0, top, n], [x - 3.0, bottom, n], [x + 3.0, bottom, n], [x + 3.0, top, n]];
            if let Err(error) = event.draw.pipeline_quads(stripes, &[PipelineQuad::spanning(corners, !0)]) {
                log::warn!("proof: stripes not drawn: {error:?}");
            }
        }
        //@}
    }
}

fn main() {
    horizon_mod_host::exit_if_compile_child();
    let startup = horizon_client::entry::prepare();
    horizon_mod_host::main_with(
        vec![ModEntry::new("proof", Box::<Proof>::default())],
        &[],
        BUILT,
        startup,
    );
}
