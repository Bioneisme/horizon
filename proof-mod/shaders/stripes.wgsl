// Four stripes, each the colour the mod wrote into its buffer.
fn pipeline_fragment(input: PipelineInput) -> vec4<f32> {
    let stripe = u32(clamp(input.uv.x * 4.0, 0.0, 3.0));
    return colors[stripe];
}
