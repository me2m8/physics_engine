pub mod application;
pub mod camera;
pub mod grid_simulation;
pub mod render_context;
pub mod shaders;
pub mod sph_simulation;

pub const PARTICLE_COLOR: [f32; 4] = [0.0, 0.5, 1.0, 1.0];
pub const CLEAR_COLOR: wgpu::Color = wgpu::Color {
    r: 0.0,
    g: 0.01,
    b: 0.01,
    a: 1.0,
};

pub const SAMPLE_COUNT: u32 = 4;
pub const VIEWPORT_SCALE: f32 = 1000.0;
pub const PARTICLE_COUNT: usize = 10000;

pub const MAX_CIRCLES: usize = 11000;
pub const MAX_LINES: usize = 1000;
pub const MAX_ARROWS: usize = 1096;
