use crate::{instances::circle::Circle, *};
use cgmath::{InnerSpace, Matrix2, Vector2, Vector4, VectorSpace};

#[allow(dead_code)]
#[derive(Debug)]
pub struct Particle {
    position: Vector2<f32>,
    last_position: Vector2<f32>,
    vel: Vector2<f32>,
    radius: f32,
    mass: f32,
    color: Vector4<f32>,
}

impl Particle {
    // Measured in cm/s^2
    const G: Vector2<f32> = Vector2::new(0.0, -981.0);

    pub fn new(
        radius: f32,
        mass: f32,
        position: Vector2<f32>,
        vel: Vector2<f32>,
        color: Vector4<f32>,
    ) -> Self {
        Self {
            position,
            last_position: position,
            vel,
            radius,
            mass,
            color,
        }
    }

    pub fn circle(&self) -> Circle {
        Circle {
            position: self.position,
            radius: self.radius,
            color: self.color,
        }
    }

    pub fn update(&mut self, dt: f32) {
        let _acceleration = Self::G;

        self.last_position = self.position;

        // self.vel += _acceleration * dt;

        self.handle_border_collisions(dt);
    }

    fn handle_border_collisions(&mut self, dt: f32) {
        let dr = self.vel * dt;
        let next_pos = self.position + dr;

        let top = next_pos.y + self.radius;
        let bottom = next_pos.y - self.radius;
        let left = next_pos.x - self.radius;
        let right = next_pos.x + self.radius;

        if bottom < BORDER_BOTTOM {
            let tc = (BORDER_BOTTOM + self.radius - self.position.y) / dr.y;
            dbg!(tc);
            let p1 = self.position.lerp(next_pos, tc);
            let p2 = Matrix2::new(1.0, 0.0, 0.0, -1.0) * (next_pos - p1);

            dbg!(p1, p2, p1 + p2);

            self.position = p1 + p2;
            self.vel.y = -self.vel.y;
            
            return;
        }

        if top > BORDER_TOP {
            let tc = (BORDER_TOP - self.radius - self.position.y) / dr.y;
            let p1 = self.position.lerp(next_pos, tc);
            let p2 = Matrix2::new(1.0, 0.0, 0.0, -1.0) * (next_pos - p1);

            self.position = p1 + p2;
            self.vel.y = -self.vel.y;
            
            return;
        }

        if left < BORDER_LEFT {
            let tc = (BORDER_LEFT + self.radius - self.position.x) / dr.x;
            let p1 = self.position.lerp(next_pos, tc);
            let p2 = Matrix2::new(1.0, 0.0, 0.0, -1.0) * (next_pos - p1);

            self.position = p1 + p2;
            self.vel.x = -self.vel.x;
            
            return;
        }

        if right > BORDER_RIGHT {
            let tc = (BORDER_RIGHT - self.radius - self.position.x) / dr.x;
            let p1 = self.position.lerp(next_pos, tc);
            let p2 = Matrix2::new(1.0, 0.0, 0.0, -1.0) * (next_pos - p1);

            self.position = p1 + p2;
            self.vel.x = -self.vel.x;
            
            return;
        }

        self.position = next_pos;
    }

    pub fn handle_particle_collision(&mut self, other: &mut Self) {
        let dist = (other.position - self.position).magnitude();
        if dist > self.radius + other.radius {
            return;
        }

        let normal = (other.position - self.position).normalize();
        let relative_velocity = other.vel - self.vel;

        let impulse = normal * 2. * relative_velocity.dot(normal) / 2.;

        self.vel += impulse;
        other.vel -= impulse;
    }
}
