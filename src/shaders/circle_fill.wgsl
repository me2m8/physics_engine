struct QuadVertex {
    @location(0) position: vec4<f32>, 
    @location(1) color: vec4<f32>,
    @location(2) frag_coord: vec2<f32>,

}

struct Instance2D {
    @location(0) transform: mat4x4<f32>,
    @location(1) color: vec4<f32>,
}

struct Camera2D {
    camera_matrix: mat4x4<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) frag_coord: vec2<f32>,
}

@group(0) @binding(0) var<uniform> camera: Camera2D;

@vertex
fn vs_main(
    in: QuadVertex
) -> VertexOutput {

    var out: VertexOutput;

    out.clip_position = camera.camera_matrix * in.position;
    out.color = in.color;
    out.frag_coord = in.frag_coord;

    return out;
}

@fragment
fn fs_main(
    in: VertexOutput
) -> @location(0) vec4<f32> {
    let dist = length(in.frag_coord);
    let crisp_factor = 10.0;
    let alpha = crisp_factor * (1.0 - dist);

    return vec4<f32>(in.color.xyz, min(alpha, in.color.w));
}
