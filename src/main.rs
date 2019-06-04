use wgpu;
use winit;

fn main() {
    // Create device connection and such
    let instance = wgpu::Instance::new();
    let adapter = instance.get_adapter(&wgpu::AdapterDescriptor {
        power_preference: wgpu::PowerPreference::Default,
    });
    let mut device = adapter.create_device(&wgpu::DeviceDescriptor {
        extensions: wgpu::Extensions {
            anisotropic_filtering: false,
        },
    });

    // Create rendering pipeline
    let vert_spirv = include_bytes!("../hello_triangle.vert.spv");
    let frag_spirv = include_bytes!("../hello_triangle.frag.spv");

    let vertex_shader = device.create_shader_module(vert_spirv);
    let frag_shader = device.create_shader_module(frag_spirv);
    let bind_group_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { bindings: &[] });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &bind_group_layout,
        bindings: &[],
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        bind_group_layouts: &[&bind_group_layout],
    });
    let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        layout: &pipeline_layout,
        vertex_stage: wgpu::PipelineStageDescriptor {
            module: &vertex_shader,
            entry_point: "main",
        },
        fragment_stage: wgpu::PipelineStageDescriptor {
            module: &frag_shader,
            entry_point: "main",
        },
        rasterization_state: wgpu::RasterizationStateDescriptor {
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: wgpu::CullMode::None,
            depth_bias: 0,
            depth_bias_slope_scale: 0.0,
            depth_bias_clamp: 0.0,
        },
        primitive_topology: wgpu::PrimitiveTopology::TriangleList,
        color_states: &[wgpu::ColorStateDescriptor {
            format: wgpu::TextureFormat::Bgra8Unorm,
            alpha: wgpu::BlendDescriptor::REPLACE,
            color: wgpu::BlendDescriptor::REPLACE,
            write_mask: wgpu::ColorWriteFlags::ALL,
        }],
        depth_stencil_state: None,
        index_format: wgpu::IndexFormat::Uint16,
        vertex_buffers: &[],
        sample_count: 1,
    });

    // Now create window and glue wgpu to it
    let mut events_loop = winit::EventsLoop::new();
    let window = winit::Window::new(&events_loop).unwrap();

    let surface = instance.create_surface(&window);
    let winit::dpi::LogicalSize { width, height } = window.get_inner_size().unwrap();
    let swap_chain = &mut device.create_swap_chain(
        &surface,
        &wgpu::SwapChainDescriptor {
            usage: wgpu::TextureUsageFlags::OUTPUT_ATTACHMENT,
            format: wgpu::TextureFormat::Bgra8Unorm,
            width: width as u32,
            height: height as u32,
        },
    );

    let mut continuing = true;
    while continuing {
        events_loop.poll_events(|event| match event {
            winit::Event::WindowEvent {
                event: winit::WindowEvent::CloseRequested,
                ..
            } => {
                continuing = false;
            }
            _ => (),
        });
        // present() is called automatically when next_texture is dropped.
        let next_texture = swap_chain.get_next_texture();
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            // lulz
            todo: 0,
        });
        let color_attachments = wgpu::RenderPassColorAttachmentDescriptor {
            attachment: &next_texture.view,
            load_op: wgpu::LoadOp::Clear,
            store_op: wgpu::StoreOp::Store,
            clear_color: wgpu::Color::BLUE,
        };
        {
            // The pass is finished automatically when rpass is drawn
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                color_attachments: &[color_attachments],
                depth_stencil_attachment: None,
            });

            rpass.set_pipeline(&render_pipeline);
            rpass.set_bind_group(0, &bind_group);
            rpass.draw(0..3, 0..1);
        }
        let cmd_buffer = encoder.finish();
        let mut queue = device.get_queue();
        queue.submit(&[cmd_buffer]);
    }
}
