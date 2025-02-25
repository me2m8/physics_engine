use cgmath::{vec2, vec4, Deg, InnerSpace, Rad, Vector2, Vector3, Vector4, Zero};
use itertools::Itertools;
use rand::random;
use std::collections::HashMap;
use std::error::Error;
use std::f32::consts::PI;
use std::process::exit;
use std::sync::mpsc::{Receiver, Sender};
use wgpu::TextureUsages;
use wgpu::{Device, Queue, Surface};
use winit::application::ApplicationHandler;
use winit::dpi::{PhysicalPosition, PhysicalSize};
use winit::event::{ElementState, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{KeyCode, ModifiersState, PhysicalKey};
#[cfg(not(target_os = "macos"))]
use winit::platform::startup_notify::{
    self, EventLoopExtStartupNotify, WindowAttributesExtStartupNotify,
};
use winit::window::{Window, WindowAttributes, WindowId};

use crate::camera::Camera2D;
use crate::render_context::shapes::{draw_arrow_2d, draw_circle_2d, draw_line_2d};
use crate::render_context::{shapes, RenderContext};
use crate::sph_simulation::{
    w_spiky, Simulation, HALF_HEIGHT, HALF_WIDTH, RHO_0, SMOOTHING_RADIUS,
};
use crate::{inverse_lerp, lerp, PARTICLE_COUNT, PARTICLE_RADIUS, SAMPLE_COUNT, VIEWPORT_SCALE};

pub struct Application {
    receiver: Receiver<Action>,
    sender: Sender<Action>,

    wgpu_instance: wgpu::Instance,
    windows: HashMap<WindowId, WindowState>,
    monitor_size: PhysicalSize<u32>,
}

impl Application {
    pub fn new(
        _event_loop: &EventLoop,
        receiver: Receiver<Action>,
        sender: Sender<Action>,
    ) -> Self {
        let wgpu_instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            flags: wgpu::InstanceFlags::empty(),
            ..Default::default()
        });

        Self {
            receiver,
            sender,
            wgpu_instance,
            windows: Default::default(),
            monitor_size: Default::default(),
        }
    }

    /// Creates a window based on the window type
    /// Should only be used to create the main window but can create other template types
    pub fn create_window(
        &mut self,
        event_loop: &dyn ActiveEventLoop,
        window_size: PhysicalSize<u32>,
        _tab_id: Option<String>,
    ) -> Result<WindowId, Box<dyn Error>> {
        #[allow(unused_mut)]
        let mut window_attributes = WindowAttributes::default()
            .with_inner_size(window_size)
            .with_transparent(true)
            .with_active(true)
            .with_title("Winit Window");

        #[cfg(not(target_os = "macos"))]
        if let Some(token) = event_loop.read_token_from_env() {
            startup_notify::reset_activation_token_env();

            window_attributes = window_attributes.with_activation_token(token);
        }

        let window = event_loop.create_window(window_attributes)?;

        // get window id
        let window_id = window.id();
        // Create window state
        let window_state = pollster::block_on(WindowState::new(window, &self.wgpu_instance))?;

        println!("Created Window: {window_id:?}");

        // Insert new window state into windows hashmap
        self.windows.insert(window_id, window_state);

        Ok(window_id)
    }

    /// Creates a winit window with the given attributes and window type
    pub fn create_window_with_attributes(
        &mut self,
        event_loop: &dyn ActiveEventLoop,
        mut window_attributes: WindowAttributes,
        _tab_id: Option<String>,
    ) -> Result<WindowId, Box<dyn Error>> {
        #[cfg(not(target_os = "macos"))]
        if let Some(token) = event_loop.read_token_from_env() {
            startup_notify::reset_activation_token_env();
            window_attributes = window_attributes.with_activation_token(token);
        }

        let window = event_loop.create_window(window_attributes)?;

        // get window id
        let window_id = window.id();
        // Create window state
        let window_state = pollster::block_on(WindowState::new(window, &self.wgpu_instance))?;

        println!("Created Window: {window_id:?}");

        // Insert new window state into windows hashmap
        self.windows.insert(window_id, window_state);

        Ok(window_id)
    }

    pub fn handle_action(&mut self, event_loop: &dyn ActiveEventLoop, action: Action) {
        match action {
            Action::Ping => println!("Pong!"),
            _ => {}
        }
    }

    pub fn handle_action_with_window(
        &mut self,
        event_loop: &dyn ActiveEventLoop,
        window_id: WindowId,
        _action: Action,
    ) {
        let Some(window) = self.windows.get_mut(&window_id) else {
            return;
        };

        #[allow(clippy::single_match)]
        match _action {
            Action::CreateNewMainWindow => {
                if let Err(err) = self.create_window(event_loop, self.monitor_size, None) {
                    println!("Failed to Create Window: {err:?}");
                }
            }
            Action::CloseWindow => {
                self.windows.remove(&window_id);

                println!("Removes window: {window_id:?}");
                let windows = self.windows.keys().collect_vec();

                println!("Remaining windows: {windows:?}");

                if self.windows.is_empty() {
                    event_loop.exit();
                }
            }
            Action::ToggleDecorations => window.toggle_decorations(),
            Action::ToggleFullscreen => window.toggle_fullscreen(),
            Action::StepSimulation => window.simulation.step_simulation_once(),
            Action::TogglePauseSimulation => window.simulation.toggle_pause(),
            Action::DeselectParticle => window.selected_particle = None,
            Action::ToggleDebugMode => window.debug_mode = !window.debug_mode,
            Action::RestartSimulation => window.simulation = Simulation::new(PARTICLE_COUNT),
            Action::SpawnParticle => window.simulation.add_new_particle(window.mouse_position),
            Action::SpawnManyParticles => window
                .simulation
                .add_many_new_particles(window.mouse_position, 9.0),
            _ => {}
        }
    }

    fn process_key_bindings(key: KeyCode, mods: ModifiersState) -> Option<Action> {
        KEYBINDS.iter().find_map(|binding| {
            binding
                .is_triggered(key, mods)
                .then_some(binding.action.clone())
        })
    }
}

impl ApplicationHandler for Application {
    /// Called when a proxy is told to wake this
    fn proxy_wake_up(&mut self, _event_loop: &dyn ActiveEventLoop) {
        for i in self.receiver.try_iter() {
            println!("Received: {i:?}");
        }
    }

    fn can_create_surfaces(&mut self, event_loop: &dyn ActiveEventLoop) {
        let mut monitor_result = event_loop.primary_monitor();

        if monitor_result.is_none() {
            monitor_result = event_loop.available_monitors().next();
        }

        let monitor = monitor_result.expect("Exppected to be able to get any monitor");

        let video_mode = monitor
            .current_video_mode()
            .expect("Expected to be able to get video mode of all monitors");
        let size = video_mode.size();
        let PhysicalSize { width, height } = size;

        println!("Monitor Size: {width} x {height}");

        self.create_window(event_loop, size, None)
            .expect("Failed to create window");
    }

    fn window_event(
        &mut self,
        event_loop: &dyn ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        let Some(window) = self.windows.get_mut(&window_id) else {
            return;
        };

        match event {
            WindowEvent::ActivationTokenDone {
                serial: _,
                token: _,
            } => todo!(),
            WindowEvent::Moved(position) => {}
            WindowEvent::CursorMoved {
                position: PhysicalPosition { x, y },
                ..
            } => {
                // Translate mouse screen coordinates to mouse world coordinates
                let PhysicalSize { width, height } = window.window.inner_size();
                let x = VIEWPORT_SCALE * (x as f32 - width as f32 / 2.0) / width as f32;
                let y = -VIEWPORT_SCALE * (y as f32 - height as f32 / 2.0) / width as f32;
                window.mouse_position = vec2(x, y);
            }
            WindowEvent::CursorLeft { .. } => {}
            WindowEvent::CursorEntered { .. } => {}
            WindowEvent::MouseInput { state, button, .. } => {
                if button == MouseButton::Left && state == ElementState::Pressed {
                    let pos = window.mouse_position;
                    let target = window.simulation.get_closest_particle(pos);

                    dbg!(window.simulation.color_field[target]);
                    dbg!(window.simulation.density[target]);
                    dbg!(window.simulation.pressure[target]);
                    dbg!(window.simulation.acceleration[target]);
                    dbg!(window.simulation.half_step_velocity[target]);

                    window.selected_particle = Some(target);
                }
            }
            WindowEvent::Resized(new_size) => window.resize(new_size),
            WindowEvent::CloseRequested => {
                println!("Removed window: {window_id:?}");
                self.windows.remove(&window_id);
            }
            // WindowEvent::Focused(_) => todo!(),
            WindowEvent::KeyboardInput { event, .. } => {
                let mods = window.modifiers;

                if event.state == ElementState::Pressed {
                    let action = if let PhysicalKey::Code(key_code) = event.physical_key {
                        Self::process_key_bindings(key_code, mods)
                    } else {
                        None
                    };

                    if let Some(action) = action {
                        self.handle_action_with_window(event_loop, window_id, action);
                    }
                }
            }
            WindowEvent::ModifiersChanged(mods) => {
                window.modifiers = mods.state();
            }
            WindowEvent::Focused(f) => {
                if f {
                    window.clear_color = wgpu::Color {
                        r: 0.0,
                        g: 0.3,
                        b: 1.0,
                        a: 1.0,
                    }
                } else {
                    window.clear_color = wgpu::Color {
                        r: 1.0,
                        g: 0.0,
                        b: 0.0,
                        a: 1.0,
                    }
                }
                window.window.request_redraw();
            }
            // WindowEvent::Ime(_) => todo!(),
            // WindowEvent::CursorMoved { device_id, position } => todo!(),
            // WindowEvent::CursorEntered { device_id } => todo!(),
            // WindowEvent::CursorLeft { device_id } => todo!(),
            // WindowEvent::MouseWheel { device_id, delta, phase } => todo!(),
            // WindowEvent::MouseInput { device_id, state, button } => todo!(),
            WindowEvent::RedrawRequested => {
                window.simulation.step_simulation();
                window.draw();
            }
            _ => {}
        }
    }

    fn exiting(&mut self, event_loop: &dyn ActiveEventLoop) {
        let _ = event_loop;
    }

    fn new_events(&mut self, event_loop: &dyn ActiveEventLoop, cause: winit::event::StartCause) {
        let _ = (event_loop, cause);
    }

    fn resumed(&mut self, event_loop: &dyn ActiveEventLoop) {
        let _ = event_loop;
    }

    fn device_event(
        &mut self,
        event_loop: &dyn ActiveEventLoop,
        device_id: winit::event::DeviceId,
        event: winit::event::DeviceEvent,
    ) {
        let _ = (event_loop, device_id, event);
    }

    /// This function is called when there are no more events
    fn about_to_wait(&mut self, event_loop: &dyn ActiveEventLoop) {
        if self.windows.is_empty() {
            event_loop.exit();
        }

        self.windows
            .values()
            .for_each(|window| window.window.request_redraw());
    }

    fn suspended(&mut self, event_loop: &dyn ActiveEventLoop) {
        let _ = event_loop;
    }

    fn destroy_surfaces(&mut self, _event_loop: &dyn ActiveEventLoop) {
        self.windows.clear();
    }

    fn memory_warning(&mut self, event_loop: &dyn ActiveEventLoop) {
        let _ = event_loop;
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Action {
    Ping,
    CreateNewMainWindow,
    CloseWindow,
    ToggleDecorations,
    ToggleFullscreen,
    StepSimulation,
    TogglePauseSimulation,
    PauseSimulation,
    DeselectParticle,
    ToggleDebugMode,
    RestartSimulation,
    SpawnParticle,
    SpawnManyParticles,
}

pub struct WindowState {
    // NOTE: The surface is dropped before the window
    //
    /// The window surface
    surface: wgpu::Surface<'static>,
    msaa: wgpu::Texture,
    config: wgpu::SurfaceConfiguration,
    _adapter: wgpu::Adapter,
    device: Device,
    queue: Queue,

    /// Context for rendering to the window
    renderer: RenderContext<Camera2D>,

    /// The simulation
    simulation: Simulation,
    selected_particle: Option<usize>,
    debug_mode: bool,

    // Miscellaneous window information
    mouse_position: Vector2<f32>,
    modifiers: ModifiersState,
    clear_color: wgpu::Color,

    /// The actual window
    window: Box<dyn Window>,
}

type CameraType = Camera2D;

impl WindowState {
    pub async fn new(
        window: Box<dyn Window>,
        instance: &wgpu::Instance,
    ) -> Result<Self, Box<dyn Error>> {
        let PhysicalSize { width, height } = window.inner_size();

        // NOTE: This is safe because the surface is dropped before the window
        let surface = unsafe {
            std::mem::transmute::<Surface<'_>, Surface<'static>>(instance.create_surface(&window)?)
        };

        let _adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::LowPower,
                force_fallback_adapter: false,
                compatible_surface: Some(&surface),
            })
            .await
            .unwrap();

        let (device, queue) = _adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("Device"),
                    required_features: wgpu::Features::POLYGON_MODE_LINE,
                    required_limits: wgpu::Limits::default(),
                    ..Default::default()
                },
                None,
            )
            .await
            .unwrap();

        let surface_caps = surface.get_capabilities(&_adapter);
        let format = *surface_caps
            .formats
            .iter()
            .find(|s| s.is_srgb())
            .unwrap_or(&wgpu::TextureFormat::Bgra8UnormSrgb);

        dbg!(&surface_caps.alpha_modes);
        dbg!(&surface_caps.present_modes);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width,
            height,
            present_mode: surface_caps.present_modes[0],
            desired_maximum_frame_latency: 2,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
        };

        surface.configure(&device, &config);

        let clear_color = wgpu::Color {
            r: 0.0,
            g: 0.3,
            b: 1.0,
            a: 1.0,
        };

        let renderer = RenderContext::<CameraType>::new(&device, &config);
        let simulation = Simulation::new(PARTICLE_COUNT);

        let msaa = Self::create_msaa_texture(&device, &config);

        Ok(Self {
            surface,
            msaa,
            config,
            _adapter,
            device,
            queue,

            renderer,

            simulation,
            selected_particle: None,
            debug_mode: true,

            mouse_position: Vector2::zero(),
            modifiers: Default::default(),
            clear_color,

            window,
        })
    }

    fn resize(&mut self, new_size: PhysicalSize<u32>) {
        let width = new_size.width;
        let height = new_size.height;

        if width == 0 || height == 0 {
            return;
        }

        self.config.width = width;
        self.config.height = height;

        self.surface.configure(self.device(), self.config());
        self.msaa = Self::create_msaa_texture(self.device(), self.config());
        self.window.request_redraw();
    }

    fn toggle_decorations(&mut self) {
        let decor = self.window.is_decorated();
        self.window.set_decorations(!decor);
    }

    fn toggle_fullscreen(&mut self) {
        let maximized = self.window.is_maximized();
        self.window.set_maximized(!maximized);
    }

    pub fn draw(&mut self) {
        if self.debug_mode {
            self.draw_spatial_grid();

            (0..self.simulation.particles).for_each(|i| {
                let pos = self.simulation.position[i];
                draw_circle_2d(
                    &self.renderer,
                    pos,
                    PARTICLE_RADIUS,
                    vec4(1.0, 1.0, 1.0, 1.0),
                );
            });

            self.draw_boundary_particles();
            self.draw_debug_selected_particle();
        } else {
            self.draw_particles();
            self.draw_boundary_square();
        }

        self.renderer
            .present_scene(&self.queue, &self.device, &self.surface, &self.msaa);
    }

    fn draw_spatial_grid(&self) {
        let vp_size = self.renderer.viewport_size();

        let grid_spacing = self.simulation.spacial_hashing.spacing;
        let n_lines = (vp_size.x / grid_spacing).ceil() as usize;

        let half_w = vp_size.x / 2.;
        let half_h = vp_size.y / 2.;

        // Draw Vertical Grid Lines
        (0..n_lines).for_each(|i| {
            let x = (i as f32) * grid_spacing;
            draw_line_2d(
                &self.renderer,
                vec2(x, -half_h),
                vec2(x, half_h),
                vec4(1.0, 1.0, 1.0, 0.1),
            );
            draw_line_2d(
                &self.renderer,
                vec2(-x, -half_h),
                vec2(-x, half_h),
                vec4(1.0, 1.0, 1.0, 0.1),
            );
        });
        // Draw Horizontal Grid Lines
        (0..n_lines).for_each(|i| {
            let y = (i as f32) * grid_spacing;
            draw_line_2d(
                &self.renderer,
                vec2(-half_w, y),
                vec2(half_w, y),
                vec4(1.0, 1.0, 1.0, 0.1),
            );
            draw_line_2d(
                &self.renderer,
                vec2(-half_w, -y),
                vec2(half_w, -y),
                vec4(1.0, 1.0, 1.0, 0.1),
            );
        });
    }

    fn draw_debug_selected_particle(&self) {
        if let Some(pi) = self.selected_particle {
            let pos = self.simulation.position[pi];

            draw_circle_2d(
                &self.renderer,
                pos,
                crate::sph_simulation::SMOOTHING_RADIUS,
                vec4(1.0, 1.0, 1.0, 0.2),
            );

            draw_circle_2d(
                &self.renderer,
                pos,
                PARTICLE_RADIUS,
                vec4(0.0, 1.0, 0.0, 1.0),
            );

            let neighbours = self
                .simulation
                .spacial_hashing
                .query_position(pos, SMOOTHING_RADIUS);

            for n in neighbours {
                if n == pi {
                    continue;
                }

                let npos = self.simulation.position[n];
                let r = npos - pos;

                let influence =
                    w_spiky(r.magnitude(), SMOOTHING_RADIUS) * PI * SMOOTHING_RADIUS.powi(4) / 6.0;
                draw_circle_2d(
                    &self.renderer,
                    npos,
                    PARTICLE_RADIUS,
                    vec4(1.0, influence, 0.0, 1.0),
                );
            }

            // Draw arrow for pressure
            let pressure_f = self.simulation.pressure_force[pi];
            let pf_dir = pressure_f.angle(Vector2::unit_x());
            draw_arrow_2d(
                &self.renderer,
                pos,
                pf_dir,
                pressure_f.magnitude() / 500.0,
                1.0,
                0.5,
                0.5,
                vec4(0.0, 1.0, 1.0, 1.0),
            );
            // Draw arrow for viscosity
            let viscous_f = self.simulation.viscous_force[pi];
            let vf_dir = viscous_f.angle(Vector2::unit_x());
            draw_arrow_2d(
                &self.renderer,
                pos,
                vf_dir,
                viscous_f.magnitude() / 500.0,
                1.0,
                0.5,
                0.5,
                vec4(0.0, 0.2, 1.0, 1.0),
            );
        }
    }

    fn draw_particles(&self) {
        let speed_gradient = &[
            ColorStop::new_rgb(2.0, 120.0, 235.0, 0.0),
            ColorStop::new_rgb(53.0, 229.0, 89.0, 0.25),
            ColorStop::new_rgb(213.0, 135.0, 45.0, 0.80),
            ColorStop::new_rgb(255.0, 75.0, 67.0, 1.0),
        ];

        (0..self.simulation.particles).for_each(|i| {
            let pos = self.simulation.position[i];
            let v = self.simulation.half_step_velocity[i];
            let scaled_v = (v.magnitude() / 200.0).clamp(0.0, 1.0);

            let color = self.linear_color_gradient(speed_gradient, scaled_v, 1.0);

            draw_circle_2d(&self.renderer, pos, PARTICLE_RADIUS, color)
        })
    }

    fn draw_boundary_particles(&self) {
        let color = vec4(0.8, 0.8, 0.8, 0.5);

        (0..self.simulation.boundary_particles).for_each(|i| {
            let p = self.simulation.boundary_particle_position[i];

            draw_circle_2d(&self.renderer, p, PARTICLE_RADIUS, color)
        })
    }

    fn draw_boundary_square(&self) {
        let color = vec4(0.8, 0.8, 0.8, 1.0);

        let padding = 1.0;

        let x = HALF_WIDTH + PARTICLE_RADIUS + padding;
        let y = HALF_HEIGHT + PARTICLE_RADIUS + padding;

        let v1 = vec2(-x, y);
        let v2 = vec2(-x, -y);
        let v3 = vec2(x, -y);
        let v4 = vec2(x, y);

        draw_line_2d(&self.renderer, v1, v2, color);
        draw_line_2d(&self.renderer, v2, v3, color);
        draw_line_2d(&self.renderer, v3, v4, color);
        draw_line_2d(&self.renderer, v4, v1, color);
    }

    fn linear_color_gradient(&self, colors: &[ColorStop], t: f32, a: f32) -> Vector4<f32> {
        match colors.len() {
            0 => vec4(0.0, 0.0, 0.0, a),
            1 => colors[0].0.xyz().extend(a),
            l => {
                for i in 0..(l - 1) {
                    if t >= colors[i].t() && t <= colors[i + 1].t() {
                        let t = inverse_lerp(colors[i].t(), colors[i + 1].t(), t);
                        let c1 = colors[i].color();
                        let c2 = colors[i + 1].color();
                        let c = lerp(c1, c2, t);

                        return c.extend(a);
                    }
                }

                unreachable!();
            }
        }
    }

    fn create_msaa_texture(device: &Device, config: &wgpu::SurfaceConfiguration) -> wgpu::Texture {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Multisampling Texture"),
            size: wgpu::Extent3d {
                width: config.width,
                height: config.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: SAMPLE_COUNT,
            dimension: wgpu::TextureDimension::D2,
            format: config.format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        })
    }

    pub const fn config(&self) -> &wgpu::SurfaceConfiguration {
        &self.config
    }

    pub const fn surface(&self) -> &wgpu::Surface<'static> {
        &self.surface
    }

    pub const fn device(&self) -> &wgpu::Device {
        &self.device
    }

    pub const fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }
}

struct ColorStop(Vector4<f32>);

impl ColorStop {
    fn new_rgb(r: f32, g: f32, b: f32, t: f32) -> Self {
        if !(0.0..=1.0).contains(&t) {
            exit(1);
        }

        Self(vec4(r / 256.0, g / 256.0, b / 256.0, t))
    }

    #[inline]
    fn t(&self) -> f32 {
        self.0.w
    }

    fn color(&self) -> Vector3<f32> {
        self.0.xyz()
    }
}

pub struct Binding<T: Eq> {
    trigger: T,
    modifiers: ModifiersState,
    action: Action,
}

impl<T: Eq> Binding<T> {
    const fn new(trigger: T, modifiers: ModifiersState, action: Action) -> Self {
        Self {
            trigger,
            modifiers,
            action,
        }
    }

    pub fn is_triggered(&self, trigger: T, mods: ModifiersState) -> bool {
        self.trigger == trigger && self.modifiers == mods
    }
}

#[rustfmt::skip]
const KEYBINDS: &[Binding<KeyCode>] = &[
    Binding::new(KeyCode::KeyD, ModifiersState::CONTROL, Action::ToggleDecorations),
    Binding::new(KeyCode::KeyF, ModifiersState::CONTROL, Action::ToggleFullscreen),
    Binding::new(KeyCode::KeyQ, ModifiersState::CONTROL, Action::CloseWindow),
    Binding::new(KeyCode::Space, ModifiersState::empty(), Action::TogglePauseSimulation),
    Binding::new(KeyCode::KeyN, ModifiersState::empty(), Action::StepSimulation),
    Binding::new(KeyCode::Escape, ModifiersState::empty(), Action::DeselectParticle),
    Binding::new(KeyCode::KeyD, ModifiersState::empty(), Action::ToggleDebugMode),
    Binding::new(KeyCode::KeyR, ModifiersState::CONTROL, Action::RestartSimulation),
    Binding::new(KeyCode::KeyS, ModifiersState::empty(), Action::SpawnParticle),
    Binding::new(KeyCode::KeyS, ModifiersState::CONTROL, Action::SpawnManyParticles),
];

pub enum ButtonState {
    Down,
    Up,
}
