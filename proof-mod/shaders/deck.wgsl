// A flat cloud deck `deck[0].x` blocks above the camera (below it when negative), checkered every
// 16 blocks, out to the horizon; the terrain in front of it hides it.
fn pipeline_fragment(input: PipelineInput) -> vec4<f32> {
    let ray = normalize(input.position - frame.eye);
    let above = deck[0].x - frame.eye.y;
    if ray.y * above <= 0.0 {
        discard;
    }
    let along = above / ray.y;
    if along > pipeline_scene_distance(input) {
        discard;
    }
    let at = frame.eye + ray * along;
    let cell = floor(at.xz / 16.0);
    let light = select(0.75, 1.0, (i32(cell.x) + i32(cell.y)) % 2 == 0);
    return vec4<f32>(vec3<f32>(light, light * 0.9, light * 0.8), 0.8);
}
