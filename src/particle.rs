use cgmath::{Vector2, Vector4, Zero};
use crate::{ctx::Ctx, instances::circle::Circle, state::State};

pub struct Particle {
    circle: Circle,
    position: Vector2<f32>,
    radius: f32,
    velocity: Vector2<f32>,
    mass: f32,
}

impl Particle {
    // Measured in cm/s^2
    const G: Vector2<f32> = Vector2::new(0.0, -981.0);

    pub fn new(
        size: f32,
        mass: f32,
        position: Vector2<f32>,
        velocity: Option<Vector2<f32>>,
    ) -> Self {
        let circle = Circle::new(
            position,
            size,
            Vector4::<f32>::new(
                0.5,
                0.5,
                1.0,
                1.0
            ),
        );

        Self {
            circle,
            radius: size,
            position,
            mass,
            velocity: velocity.unwrap_or(Vector2::zero()),
        }
    }

    pub fn physics_update(&mut self, state: &State, dt: std::time::Duration) {
        let dt = dt.as_secs_f32();

        if self.position.y + self.velocity.y * dt - self.radius - crate::BORDER_WIDTH < -(state.config().height as f32) {
            self.velocity = -self.velocity;
        } else {
            self.position += self.velocity * dt;
        }

        self.velocity += Self::G * dt;

        self.circle.position = self.position;
    }

    pub fn circle(&self) -> Circle {
        self.circle
    }
}
