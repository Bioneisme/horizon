// How deep the opaque terrain lies under this point of the plane, along the view ray from where
// the view's rays start (frame.eye): the plane fades out where the ground comes up to it and is
// solid over low ground.
fn pipeline_fragment(input: PipelineInput) -> vec4<f32> {
    let below = pipeline_scene_distance(input) - distance(input.position, frame.eye);
    return vec4<f32>(input.color.rgb, input.color.a * clamp(below / 3.0, 0.0, 1.0) * 0.85);
}
