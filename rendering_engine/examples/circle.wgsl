
struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) color: vec3<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) fragCoord: vec3<f32>,
    @location(1) color: vec3<f32>,
}

@vertex fn vs_main(
    vertex: VertexInput,
) -> VertexOutput {
    var out: VertexOutput;

    out.clip_position = vec4<f32>(vertex.position, 1.0);
    out.fragCoord = vertex.position;
    out.color = vertex.color;

    return out;
}

@fragment fn fs_main(
    in: VertexOutput,
) -> @location(0) vec4<f32> {

    let uv = in.fragCoord * 2.0;

    let len = length(uv);

    if (len < 0.98) {
        return vec4<f32>(1.0, 0.0, 0.0, 0.8);
    } else if (len < 1.0) {
        return vec4<f32>(1.0, 0.0, 0.0, 1.0);
    } else {
        return vec4<f32>(0.0);
    }
    return vec4<f32>(uv, 1.0);
}
