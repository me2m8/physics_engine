
use cgmath::{Matrix4, Vector2, Vector3};

pub struct Camera {
    eye: Vector3<f32>,
    rotations: Vector3<f32>,
    viewport: Vector2<f32>,
}
