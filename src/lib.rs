
pub mod state;
pub mod ctx;
pub mod instances;
pub mod camera;
pub mod particle;
pub mod simulation_context;
pub mod render_context;

pub const BORDER_WIDTH: f32 = 0.;

pub const WINDOW_WIDTH: f32 = 1920.;
pub const WINDOW_HEIGHT: f32 = 1080.;

pub const VIEWPORT_WIDTH: f32 = WINDOW_WIDTH * 2.0;
pub const VIEWPORT_HEIGHT: f32 = WINDOW_HEIGHT * 2.0;

pub const BORDER_TOP: f32 = VIEWPORT_HEIGHT / 2. - BORDER_WIDTH;
pub const BORDER_BOTTOM: f32 = -VIEWPORT_HEIGHT / 2. + BORDER_WIDTH;
pub const BORDER_LEFT: f32 = -VIEWPORT_WIDTH / 2. + BORDER_WIDTH;
pub const BORDER_RIGHT: f32 = VIEWPORT_WIDTH / 2. - BORDER_WIDTH;
