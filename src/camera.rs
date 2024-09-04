
use cgmath::{Matrix4, Vector2, Vector3};

pub struct Camera {
    eye: Vector3<f32>,
    rotations: Vector3<f32>,
    viewport: Vector2<f32>,
    fov: f32,
}

pub struct RawCamera {
    projection_matrix: Matrix4<f32>,
}

//
// a = h/w
//
// f = 1 / tan(fov / 2)
//
// zfar is the farthest we can render
// znear is the nearest we can render
//
// lambda = zfar(1 - znear) / (zfar - znear)
// lambda is the z scaling factor

//
// [afx, fy, lambda*z - lambda * znear]


