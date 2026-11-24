fn pipeline_fragment(input: PipelineInput) -> vec4<f32> {
    return vec4<f32>(input.color.rgb, input.color.a * 0.6);
}
