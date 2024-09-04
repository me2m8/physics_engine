struct VertexInput {
    @location(0) position: vec4<f32>, 
    @location(1) color: vec4<f32>,
    @location(2) frag_coord: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) frag_coord: vec2<f32>,
}

@vertex
fn vs_main(
    in: VertexInput
) -> VertexOutput {
    var out: VertexOutput;

    out.clip_position = in.position;
    out.color = in.color;
    out.frag_coord = in.frag_coord;

    return out;
}

@fragment
fn fs_main(
    in: VertexOutput
) -> @location(0) vec4<f32> {
    let dist = length(in.frag_coord);
    let base_alpha = 0.2;

    return vec4<f32>(in.color.xyz, cubed(1.0 - dist) * base_alpha);
}

/// Cubes the given number
fn cubed(n: f32) -> f32 {
    return n * n * n;
}
