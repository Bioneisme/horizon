// A sphere traced inside its proxy quad: `ball[0].xyz` its centre relative to the camera, `.w`
// its radius. It returns the point it hit, so the game writes that point's depth: the terrain
// cuts it where it goes into the ground, and quads drawn later meet it where they cross it.
fn pipeline_fragment(input: PipelineInput) -> PipelineSurface {
    let center = ball[0].xyz;
    let radius = ball[0].w;
    let ray = normalize(input.position - frame.eye);
    let to_eye = frame.eye - center;
    let b = dot(to_eye, ray);
    let h = b * b - (dot(to_eye, to_eye) - radius * radius);
    if h < 0.0 {
        discard;
    }
    let along = -b - sqrt(h);
    if along <= 0.0 {
        discard;
    }
    let hit = frame.eye + ray * along;
    let normal = normalize(hit - center);
    let light = 0.3 + 0.7 * max(dot(normal, normalize(vec3<f32>(0.4, 0.9, 0.3))), 0.0);
    return PipelineSurface(vec4<f32>(vec3<f32>(1.0, 0.55, 0.1) * light, 1.0), hit);
}
