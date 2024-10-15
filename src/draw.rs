use std::f32::consts::PI;

use cgmath::{vec2, vec4, Vector2, Vector4};
use wgpu::Queue;

use crate::{camera::Camera, instance::Instance2D, render_context::RenderContext};

pub fn render<T: Camera + Sized>(render_context: &mut RenderContext<T>, queue: &Queue) {
    render_context.begin_scene(queue);

    let (draw_circle, draw_arrow) = {
        let draw_circle = |p: Vector2<f32>, scale: Vector2<f32>, color: Vector4<f32>| {
            let instance = Instance2D::new(p, 0.0, scale, color);
            render_context.circles().add_instance(instance);
        };

        let draw_arrow =
            |p: Vector2<f32>, rotation: f32, scale: Vector2<f32>, color: Vector4<f32>| {
                let instance = Instance2D::new(p, rotation, scale, color);
                render_context.arrows().add_instance(instance);
            };

        (draw_circle, draw_arrow)
    };

    draw_arrow(
        vec2(0.0, 0.0),
        PI / 2.0,
        vec2(1.0, 1.0),
        vec4(0.0, 1.0, 1.0, 1.0),
    );
    // draw_circle(vec2(20.0, 0.0), vec2(1.0, 1.0), vec4(1.0, 1.0, 1.0, 1.0));
}
