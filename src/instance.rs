use cgmath::{Vector2, Vector4};
use wgpu::{vertex_attr_array, BufferAddress, VertexAttribute, VertexBufferLayout};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Zeroable, bytemuck::Pod)]
pub struct RawInstance2D {
    transform: [[f32; 4]; 4],
    color: [f32; 4],
}

pub struct Instance2D {
    position: Vector2<f32>,
    rotation: f32,
    scale: Vector2<f32>,
    color: Vector4<f32>,
}

impl Instance2D {
    pub fn new(p: Vector2<f32>, rotation: f32, scale: Vector2<f32>, color: Vector4<f32>) -> Self {
        Self {
            position: p,
            rotation,
            scale,
            color,
        }
    }
}

pub trait Instance {
    /// The associated raw type
    type Raw;

    /// Vertex attributes
    const ATTRIBS: &'static [VertexAttribute];

    const DESC: VertexBufferLayout<'static> = VertexBufferLayout {
        array_stride: std::mem::size_of::<Self::Raw>() as BufferAddress,
        step_mode: wgpu::VertexStepMode::Instance,
        attributes: Self::ATTRIBS,
    };

    /// Converts the instance to its assicated raw instance
    fn to_raw(&self) -> Self::Raw;
}

impl Instance for Instance2D {
    type Raw = RawInstance2D;

    const ATTRIBS: &'static [VertexAttribute] = &vertex_attr_array![
        0 => Float32x4, // Matrix4x4
        1 => Float32x4, // Matrix4x4
        2 => Float32x4, // Matrix4x4
        3 => Float32x4, // Matrix4x4
        4 => Float32x4, // Color
    ];

    fn to_raw(&self) -> Self::Raw {
        // Get the transform values from the struct
        let Vector2 { x, y } = self.position;
        let Vector2 { x: s_x, y: s_y } = self.scale;
        let (sin, cos) = self.rotation.sin_cos();

        // Build matrix
        // This matrix is the result of scaling, then rotating, then translating
        let transform = [
            [s_x * cos, s_y * sin, 0.0, x],
            [-s_x * sin, s_y * cos, 0.0, y],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];

        let color = self.color;

        // Return the raw instance
        Self::Raw {
            transform,
            color: color.into(),
        }
    }
}
