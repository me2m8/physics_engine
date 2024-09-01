struct VertexInput {
    @location(0) frag_coord: vec2<f32>,
}

struct InstanceInput {
    @location(1) position: vec2<f32>,
    @location(2) radius: f32,
    @location(3) color: vec4<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) frag_coord: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) radius: f32,
}

struct Camera2D {
    position: vec2<f32>,
    resolution: vec2<f32>,
}

// Camera Uniform Buffer
@group(0) @binding(0) var<uniform> camera2d: Camera2D;


@vertex fn vs_main(
    model: VertexInput,
    instance: InstanceInput,
) -> VertexOutput {
    var out: VertexOutput;

    let position = instance.position + model.frag_coord * instance.radius;
    let transformed_position = (position - camera2d.position) / (camera2d.resolution / 2.);

    out.clip_position = vec4<f32>(transformed_position, 0.0, 1.0);
    out.frag_coord = model.frag_coord;
    out.color = instance.color;
    out.radius = instance.radius;

    return out;
}


@fragment fn fs_main(
    in: VertexOutput,
) -> @location(0) vec4<f32> {
    let crispness_factor = in.radius / 2.;
    let dist = crispness_factor - length(in.frag_coord) * crispness_factor;

    let rgb_color = exp(log((in.color.xyz + 0.055) / 1.055) * 2.);

    return vec4<f32>(rgb_color, dist);
}
