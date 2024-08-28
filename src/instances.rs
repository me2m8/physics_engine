pub mod circle {
    use super::*;

    #[derive(Copy, Clone, Debug)]
    pub struct Circle {
        pub position: [f32; 2],
        pub radius: f32,
        pub color: [f32; 4],
    }

    #[repr(C)]
    #[derive(Copy, Clone, Debug, bytemuck::Zeroable, bytemuck::Pod)]
    pub struct RawCircle {
        position: [f32; 2],
        radius: f32,
        color: [f32; 4],
    }

    #[repr(C)]
    #[derive(Copy, Clone, Debug, bytemuck::Zeroable, bytemuck::Pod)]
    pub struct CircleVertex {
        frag_coord: [f32; 2],
    }

    #[rustfmt::skip]
    pub const CIRCLE_VERTS: &[CircleVertex] = &[
        CircleVertex { frag_coord: [ 1.0,  1.0] },
        CircleVertex { frag_coord: [-1.0,  1.0] },
        CircleVertex { frag_coord: [-1.0, -1.0] },
        CircleVertex { frag_coord: [ 1.0, -1.0] },
    ];

    #[rustfmt::skip]
    pub const CIRCLE_INDICIES: &[u16] = &[
        0, 1, 2, 
        0, 2, 3
    ];

    impl CircleVertex {
        const ATTRIBS: &'static [wgpu::VertexAttribute] = &wgpu::vertex_attr_array![0 => Float32x2];

        pub fn desc() -> wgpu::VertexBufferLayout<'static> {
            wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<CircleVertex>() as wgpu::BufferAddress,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: Self::ATTRIBS,
            }
        }
    }

    impl Circle {
        pub fn new(
            position: [f32; 2],
            radius: f32,
            color: [f32; 4],
        ) -> Self {
            Self {
                position,
                radius,
                color,
            }
        }

        pub fn to_raw(&self) -> RawCircle {
            RawCircle {
                position: self.position,
                radius: self.radius,
                color: self.color,
            }
        }
    }

    impl RawCircle {
        const ATTRIBS: &'static [wgpu::VertexAttribute] =
            &wgpu::vertex_attr_array![1 => Float32x2, 2 => Float32, 3 => Float32x4];

        pub fn desc() -> wgpu::VertexBufferLayout<'static> {
            wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<RawCircle>() as wgpu::BufferAddress,
                step_mode: wgpu::VertexStepMode::Instance,
                attributes: Self::ATTRIBS,
            }
        }
    }
}
