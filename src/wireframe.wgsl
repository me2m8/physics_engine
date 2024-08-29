
struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) color: vec4<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
}

struct Camera2D {
    position: vec2<f32>,
    resolution: vec2<f32>,
}

// Camera Uniform Buffer
@group(0) @binding(0) var<uniform> camera2d: Camera2D;

@vertex fn vs_main(
    model: VertexInput,
) -> VertexOutput {
    var out: VertexOutput;

    let transformed_position = (model.position - camera2d.position) / camera2d.resolution;

    out.clip_position = vec4<f32>(transformed_position, 0.0, 1.0);
    out.color = model.color;

    return out;
}

@fragment fn fs_main(
    in: VertexOutput,
) -> @location(0) vec4<f32> {
    return in.color;
}
