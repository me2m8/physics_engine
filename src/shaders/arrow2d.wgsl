struct QuadVertex {
    @location(0) position: vec4<f32>, 
    @location(1) color: vec4<f32>,
}

// Size: 80B
struct Instance2D {
    @location(2) transform_matrix_0: vec4<f32>,
    @location(3) transform_matrix_1: vec4<f32>,
    @location(4) transform_matrix_2: vec4<f32>,
    @location(5) transform_matrix_3: vec4<f32>,
    @location(6) color: vec4<f32>,
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
    instance: Instance2D,
    vertex: QuadVertex,
) -> VertexOutput {
    let transform = mat4x4<f32>(
        instance.transform_matrix_0,
        instance.transform_matrix_1,
        instance.transform_matrix_2,
        instance.transform_matrix_3,
    );

    var out: VertexOutput;

    out.clip_position = camera.camera_matrix * transform * vertex.position;
    out.color = (vertex.color + 2 * instance.color) / 2;

    return out;
}

@fragment
fn fs_main(
    in: VertexOutput
) -> @location(0) vec4<f32> {
    return in.color;
}
