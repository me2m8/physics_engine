#define_import_path globals

struct LineVertex {
    @location(0) position: vec4<f32>, 
    @location(1) color: vec4<f32>,
}

struct QuadVertex {
    @location(0) position: vec4<f32>, 
    @location(1) color: vec4<f32>,
    @location(2) frag_coord: vec2<f32>,
}

struct Camera2D {
    camera_matrix: mat4x4<f32>,
}
