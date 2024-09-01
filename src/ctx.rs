use cgmath::Vector2;

use crate::camera::Camera2D;
use crate::instances::circle::*;
use crate::instances::polygon_frame::*;
use crate::simulation_context::SimulationCtx;
use crate::state::State;

pub struct Ctx {
    pub size: Vector2<f32>,
    pub camera: Camera2D,
    pub circle_renderer: CircleRender,
    pub polygon_frame_render: PolygonFrameRender,

    pub simulation_ctx: SimulationCtx,
}

impl Ctx {
    pub fn new(state: &State, size: Vector2<f32>) -> Self {
        let camera = Camera2D::new(state, size);

        let mut circle_renderer = CircleRender::new(state, &camera);
        let polygon_frame_render = PolygonFrameRender::new(state, &camera);

        let simulation_ctx = SimulationCtx::new();
        simulation_ctx.draw_particles(state, &mut circle_renderer);

        Self {
            size,
            circle_renderer,
            polygon_frame_render,
            camera,
            simulation_ctx,
        }
    }
}
