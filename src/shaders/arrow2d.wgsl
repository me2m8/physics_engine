struct QuadVertex {
    @location(0) position: vec4<f32>, 
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
}

@group(0) @binding(0) var<uniform> camera: Camera2D;

@vertex
fn vs_main(
    vertex: QuadVertex,
    instance: Instance2D,
) -> VertexOutput {
    var out: VertexOutput;

    out.clip_position = camera.camera_matrix * instance.transform * vertex.position;
    out.color = instance.color;

    return out;
}

@fragment
fn fs_main(
    in: VertexOutput
) -> @location(0) vec4<f32> {
    return in.color;
}
