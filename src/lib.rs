use std::ops::{Add, Mul};

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
pub const VIEWPORT_SCALE: f32 = 1500.0;
pub const PARTICLE_COUNT: usize = 4000;

pub const PARTICLE_RADIUS: f32 = 2.0;

pub const MAX_CIRCLES: usize = 15100;
pub const MAX_LINES: usize = 2000;
pub const MAX_ARROWS: usize = 6096;

/// Linearly interpolates a value between a and b using the value t. For example, if t is 0.5, a
/// value halfway between a and b will be given. If t is 0.25, a value 25% between a and b will be
/// given. Will clamp t to (0.0, 1.0) range.
fn lerp<V>(a: V, b: V, t: f32) -> V
where
    V: Add<Output = V> + Mul<f32, Output = V> + Copy,
{
    let t = t.clamp(0.0, 1.0);
    a * (1.0 - t) + b * t
}

/// Given a value c between a and b, returns the fraction of the way between a and b.
fn inverse_lerp(a: f32, b: f32, c: f32) -> f32 {
    if c > b || c < a {
        return 0.0;
    }
    (c - a) / (b - a)
}

/// Linearly interpolates a value between a and b using the value t. For example, if t is 0.5, a
/// value halfway between a and b will be given. If t is 0.25, a value 25% between a and b will be
/// given. Will accept values outside (0.0, 1.0) range and give a balue smaller than a or larger
/// than b.
fn lerp_unchecked(a: f32, b: f32, t: f32) -> f32 {
    a * (1.0 - t) + b * t
}

/// Bilinear interpolation
fn bilerp(a: f32, b: f32, c: f32, d: f32, t: f32) -> f32 {
    match t {
        t if t < 0.0 => a,
        t if t > 1.0 => d,
        t => {
            let l1 = lerp(a, b, t);
            let l2 = lerp(c, d, t);
            lerp(l1, l2, t)
        }
    }
}
