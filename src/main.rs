use std::sync::Arc;
use std::time;

use rendy::command::QueueId;
use rendy::factory::{Config, Factory};
use rendy::graph::{present::PresentNode, render::*, GraphBuilder};
use rendy::hal;
use rendy::hal::PhysicalDevice as _;

use winit::{Event, EventsLoop, WindowBuilder, WindowEvent};

use ggraphics::*;

fn main() {
    let config: Config = Default::default();

    let (mut factory, mut families): (Factory<Backend>, _) = rendy::factory::init(config).unwrap();

    let mut event_loop = EventsLoop::new();

    let window = WindowBuilder::new()
        .with_title("Rendy example")
        .build(&event_loop)
        .unwrap();
    let window_size = window
        .get_inner_size()
        .unwrap()
        .to_physical(window.get_hidpi_factor());

    event_loop.poll_events(|_| ());

    let surface = factory.create_surface(&window);

    let mut graph_builder = GraphBuilder::<Backend, Aux<Backend>>::new();

    let size = window
        .get_inner_size()
        .unwrap()
        .to_physical(window.get_hidpi_factor());

    let window_kind = hal::image::Kind::D2(size.width as u32, size.height as u32, 1, 1);
    let color = graph_builder.create_image(
        window_kind,
        1,
        factory.get_surface_format(&surface),
        Some(hal::command::ClearValue::Color([0.1, 0.2, 0.3, 1.0].into())),
    );

    let depth = graph_builder.create_image(
        window_kind,
        1,
        hal::format::Format::D16Unorm,
        Some(hal::command::ClearValue::DepthStencil(
            hal::command::ClearDepthStencil(1.0, 0),
        )),
    );

    let render_group_desc = MeshRenderGroupDesc::new();
    let pass = graph_builder.add_node(
        render_group_desc
            .builder()
            .into_subpass()
            .with_color(color)
            .with_depth_stencil(depth)
            .into_pass(),
    );

    let present_builder = PresentNode::builder(&factory, surface, color).with_dependency(pass);

    let frames = present_builder.image_count();

    graph_builder.add_node(present_builder);

    // HACK suggested by Frizi, just use queue 0 for everything
    // instead of getting it from `graph.node_queue(pass)`.
    // Since we control in our `Config` what families we have
    // and what they have, as long as we only ever use one family
    // (which is probably fine) then we're prooooobably okay with
    // this.
    let queue_id = QueueId {
        family: families.family_by_index(0).id(),
        index: 0,
    };

    let rendy_bytes = include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/src/data/rendy_logo.png"
    ));
    let gfx_bytes = include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/src/data/gfx_logo.png"
    ));
    let heart_bytes = include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/data/heart.png"));
    let rust_bytes = include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/src/data/rust_logo.png"
    ));

    let width = window_size.width as f32;
    let height = window_size.height as f32;
    println!("dims: {}x{}", width, height);

    let texture1 = make_texture(queue_id, &mut factory, gfx_bytes);
    let texture2 = make_texture(queue_id, &mut factory, rendy_bytes);
    let texture3 = make_texture(queue_id, &mut factory, heart_bytes);
    let texture4 = make_texture(queue_id, &mut factory, rust_bytes);
    let object_mesh = Arc::new(make_quad_mesh(queue_id, &mut factory));
    let tri_mesh = Arc::new(make_tri_mesh(queue_id, &mut factory));

    let align = factory
        .physical()
        .limits()
        .min_uniform_buffer_offset_alignment;

    let draws = vec![
        DrawCall::new(texture1, object_mesh.clone()),
        DrawCall::new(texture2, object_mesh.clone()),
        DrawCall::new(texture3, object_mesh.clone()),
        DrawCall::new(texture4, tri_mesh),
    ];

    let vertex_src = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/src/data/shader.glslv"
    ));
    let fragment_src = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/src/data/shader.glslf"
    ));

    let mut aux = Aux {
        frames: frames as _,
        align,

        draws,
        camera: UniformData {
            proj: Transform3::ortho(0.0, width, height, 0.0, 1.0, 200.0),

            view: Transform3::create_translation(0.0, 0.0, 10.0),
        },

        shader: load_shaders(vertex_src, fragment_src),
    };

    let mut graph = graph_builder
        .with_frames_in_flight(frames)
        .build(&mut factory, &mut families, &aux)
        .unwrap();

    let mut frames = 0u64..;
    let mut rng = rand::thread_rng();

    let mut should_close = false;

    let started = time::Instant::now();
    // TODO: Someday actually check against MAX_OBJECTS
    while !should_close {
        for _i in &mut frames {
            factory.maintain(&mut families);
            event_loop.poll_events(|event| match event {
                Event::WindowEvent {
                    event: WindowEvent::CloseRequested,
                    ..
                } => should_close = true,
                _ => (),
            });
            graph.run(&mut factory, &mut families, &aux);
            // Add another object
            for draw_call in &mut aux.draws {
                draw_call.add_object(&mut rng, width, height);
            }
            if should_close {
                break;
            }
        }
    }
    let finished = time::Instant::now();
    let dt = finished - started;
    let millis = dt.as_millis() as f64;
    let fps = frames.start as f64 / (millis / 1000.0);
    println!(
        "{} frames over {} seconds; {} fps",
        frames.start,
        millis / 1000.0,
        fps
    );

    graph.dispose(&mut factory, &aux);
}
