
pub struct State<'a> {
    instance: wgpu::Instance,
    _adapter: wgpu::Adapter,
    surface: wgpu::Surface<'a>,
    multisample_texture: wgpu::Texture,
    sample_count: u32,
    config: wgpu::SurfaceConfiguration,
    device: wgpu::Device,
    queue: wgpu::Queue,

    window: &'a winit::window::Window,
}

// For more functional methods
impl<'a> State<'a> {
    pub async fn new(window: &'a winit::window::Window) -> Self {
        let size = window.inner_size();

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            flags: wgpu::InstanceFlags::empty(),
            ..Default::default()
        });

        let surface = instance.create_surface(window).unwrap();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::LowPower,
                force_fallback_adapter: false,
                compatible_surface: Some(&surface),
            })
            .await
            .unwrap();

        let surface_caps = surface.get_capabilities(&adapter);

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("Device Descriptor"),
                    required_features: wgpu::Features::POLYGON_MODE_LINE,
                    required_limits: wgpu::Limits::default(),
                    ..Default::default()
                },
                None,
            )
            .await
            .unwrap();

        let format = surface_caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(surface_caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width,
            height: size.height,
            view_formats: vec![],
            present_mode: dbg!(surface_caps.present_modes)[0], 
            alpha_mode: dbg!(surface_caps.alpha_modes)[0],
            desired_maximum_frame_latency: 2,
        };

        surface.configure(&device, &config);

        let sample_count = 4;

        let multisample_texture = device.create_texture(
            &wgpu::TextureDescriptor {
                label: Some("Multisample Texture"),
                size: wgpu::Extent3d {
                    width: size.width,
                    height: size.height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            }
        );

        Self {
            window,
            instance,
            _adapter: adapter,
            surface,
            multisample_texture,
            sample_count,
            config,
            device,
            queue,
        }
    }

    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        self.config.width = new_size.width;
        self.config.height = new_size.height;

        self.surface.configure(&self.device, &self.config);
        self.multisample_texture = self.create_multisample_texture();
    }

    pub fn handle_window_events(&mut self, event: winit::event::WindowEvent) {
    }

    fn create_multisample_texture(&self) -> wgpu::Texture {
        self.device.create_texture(
            &wgpu::TextureDescriptor {
                label: Some("Multisample Texture"),
                size: wgpu::Extent3d {
                    width: self.config.width,
                    height: self.config.height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: self.sample_count,
                dimension: wgpu::TextureDimension::D2,
                format: self.config.format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            }
        )
    }
}

// For general getter/setter functions
impl<'a> State<'a> {
    pub fn window(&self) -> &winit::window::Window {
        self.window
    }

    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    pub fn instance(&self) -> &wgpu::Instance {
        &self.instance
    }

    pub fn config(&self) -> &wgpu::SurfaceConfiguration {
        &self.config
    }

    pub fn set_config(&mut self, config: wgpu::SurfaceConfiguration) {
        self.config = config;
    }

    pub fn surface(&self) -> &wgpu::Surface<'a> {
        &self.surface
    }

    pub fn sample_count(&self) -> u32 {
        self.sample_count
    }

    pub fn multisample_texture(&self) -> &wgpu::Texture {
        &self.multisample_texture
    }

}
