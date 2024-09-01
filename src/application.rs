use std::collections::HashMap;
use std::error::Error;
use std::sync::mpsc::{Receiver, Sender};
use wgpu::Surface;
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::{ElementState, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{KeyCode, ModifiersState, PhysicalKey};
use winit::platform::startup_notify::{
    self, EventLoopExtStartupNotify, WindowAttributesExtStartupNotify,
};
use winit::window::{Window, WindowAttributes, WindowId};

pub struct Application {
    reciever: Receiver<Action>,
    sender: Sender<Action>,

    wgpu_instance: wgpu::Instance,

    windows: HashMap<WindowId, WindowState>,
}

impl Application {
    pub fn new(
        _event_loop: &EventLoop,
        reciever: Receiver<Action>,
        sender: Sender<Action>,
    ) -> Self {
        let wgpu_instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            flags: wgpu::InstanceFlags::empty(),
            ..Default::default()
        });

        Self {
            reciever,
            sender,
            wgpu_instance,
            windows: Default::default(),
        }
    }
    pub fn create_window(
        &mut self,
        event_loop: &dyn ActiveEventLoop,
        _tab_id: Option<String>,
    ) -> Result<(), Box<dyn Error>> {
        let mut window_attributes = WindowAttributes::default()
            .with_title("Winit Window")
            .with_transparent(true)
            .with_active(true);

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

        Ok(())
    }

    pub fn handle_action(&mut self, _event_loop: &dyn ActiveEventLoop, action: Action) {
        if let Action::Ping = action {
            println!("Pong!")
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
            Action::CreateNewWindow => {
                if let Err(err) = self.create_window(event_loop, None) {
                    println!("Failed to Create Window: {err:?}");
                }
            }
            Action::CloseWindow => {
                self.windows.remove(&window_id);

                if self.windows.is_empty() {
                    event_loop.exit();
                }
            }
            Action::ToggleDecorations => window.toggle_decorations(),
            Action::ToggleFullscreen => window.toggle_fullscreen(),
            _ => {}
        }
    }

    fn process_key_bindings(key: KeyCode, mods: ModifiersState) -> Option<Action> {
        KEYBINDS
            .iter()
            .find_map(|binding| binding.is_triggered(key, mods).then_some(binding.action))
    }
}

impl ApplicationHandler for Application {
    /// Called when a proxy is told to wake this
    fn proxy_wake_up(&mut self, _event_loop: &dyn ActiveEventLoop) {
        for i in self.reciever.try_iter() {
            println!("Received: {i:?}");
        }
    }

    fn can_create_surfaces(&mut self, event_loop: &dyn ActiveEventLoop) {
        self.create_window(event_loop, None)
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
            WindowEvent::Resized(new_size) => window.resize(new_size),
            WindowEvent::CloseRequested => {
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
            // WindowEvent::Focused(f) => {
            //     if f {
            //         window.clear_color = wgpu::Color {
            //             r: 0.0,
            //             g: 1.0,
            //             b: 0.0,
            //             a: 0.1,
            //         }
            //     } else {
            //         window.clear_color = wgpu::Color {
            //             r: 1.0,
            //             g: 0.0,
            //             b: 0.0,
            //             a: 0.1,
            //         }
            //     }
            //     window.window.request_redraw();
            // }
            // WindowEvent::Ime(_) => todo!(),
            // WindowEvent::CursorMoved { device_id, position } => todo!(),
            // WindowEvent::CursorEntered { device_id } => todo!(),
            // WindowEvent::CursorLeft { device_id } => todo!(),
            // WindowEvent::MouseWheel { device_id, delta, phase } => todo!(),
            // WindowEvent::MouseInput { device_id, state, button } => todo!(),
            WindowEvent::RedrawRequested => {
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

    fn about_to_wait(&mut self, event_loop: &dyn ActiveEventLoop) {
        if self.windows.is_empty() {
            event_loop.exit();
        }
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

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum Action {
    Ping,
    CreateNewWindow,
    CloseWindow,
    ToggleDecorations,
    ToggleFullscreen,
}

pub struct WindowState {
    // NOTE: The surface is dropped before the window
    //
    /// The window surface
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    _adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,

    // Miscelaneous window information
    modifiers: ModifiersState,
    clear_color: wgpu::Color,

    /// The actual window
    window: Box<dyn Window>,
}

impl WindowState {
    pub async fn new(
        window: Box<dyn Window>,
        instance: &wgpu::Instance,
    ) -> Result<Self, Box<dyn Error>> {
        let PhysicalSize { width, height } = window.inner_size();

        // NOTE: This is safe beacuse the surface is dropped before the window
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

        let clear_color = wgpu::Color {
            r: 0.0,
            g: 0.3,
            b: 1.0,
            a: 1.0,
        };

        Ok(Self {
            surface,
            config,
            _adapter,
            device,
            queue,

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

        self.surface.configure(&self.device, &self.config);
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

    fn draw(&mut self) {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Command Encoder"),
            });

        let output = self.surface.get_current_texture().unwrap();
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        {
            let _render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        store: wgpu::StoreOp::Store,
                        load: wgpu::LoadOp::Clear(self.clear_color),
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();
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
    Binding::new(KeyCode::KeyN, ModifiersState::CONTROL, Action::CreateNewWindow),
    Binding::new(KeyCode::KeyD, ModifiersState::CONTROL, Action::ToggleDecorations),
    Binding::new(KeyCode::KeyF, ModifiersState::CONTROL, Action::ToggleFullscreen),
    Binding::new(KeyCode::KeyQ, ModifiersState::CONTROL, Action::CloseWindow),
    Binding::new(KeyCode::Escape, ModifiersState::empty(), Action::CloseWindow),
];
