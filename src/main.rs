// Suggested logging level for debugging:
// env RUST_LOG=info cargo run
//
// Next up: Render passes
// Better shader setup, multiple pipelines
// Clear color -- start refactoring it into an actual lib
// Make actual projection and stuff.
// Try out triangle strips?  idk, vertices don't seem much a bottleneck.
// Resize viewport properly

use log::*;
use std::time::Instant;

use ggraphics::*;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen(start))]
pub fn wasm_main() {
    main();
}

#[cfg(target_arch = "wasm32")]
fn run_wasm() {
    use console_error_panic_hook;
    use web_sys;

    console_error_panic_hook::set_once();
    // CONTEXT CREATION
    let (gl, render_loop, shader_version) = {
        use wasm_bindgen::JsCast;
        let canvas = web_sys::window()
            .unwrap()
            .document()
            .unwrap()
            .get_element_by_id("canvas")
            .unwrap()
            .dyn_into::<web_sys::HtmlCanvasElement>()
            .unwrap();
        let webgl2_context = canvas
            .get_context("webgl2")
            .unwrap()
            .unwrap()
            .dyn_into::<web_sys::WebGl2RenderingContext>()
            .unwrap();
        (
            glow::Context::from_webgl2_context(webgl2_context),
            glow::RenderLoop::from_request_animation_frame(),
            "#version 300 es",
        )
    };

    // GL SETUP
    let mut ctx = Some(GlContext::new(gl, shader_version));

    // RENDER LOOP
    render_loop.run(move |running: &mut bool| {
        if let Some(ictx) = &mut ctx {
            // web-sys has no Instant so we just have
            // to give it a dummy frame duration
            ictx.update(Duration::from_millis(10));
            ictx.draw();
        }

        if !*running {
            // Drop context, deleting its contents.
            ctx = None;
        }
    });
}

#[cfg(not(target_arch = "wasm32"))]
fn run_glutin() {
    pretty_env_logger::init();
    // CONTEXT CREATION
    unsafe {
        // Create a context from a glutin window on non-wasm32 targets
        let (gl, event_loop, windowed_context, shader_version) = {
            let el = glutin::event_loop::EventLoop::new();
            let wb = glutin::window::WindowBuilder::new()
                .with_title("Hello triangle!")
                .with_inner_size(glutin::dpi::LogicalSize::new(1024.0, 768.0));
            let windowed_context = glutin::ContextBuilder::new()
                //.with_gl(glutin::GlRequest::Latest)
                .with_gl(glutin::GlRequest::GlThenGles {
                    opengl_version: (4, 3),
                    opengles_version: (3, 0),
                })
                .with_gl_profile(glutin::GlProfile::Core)
                .with_vsync(true)
                .build_windowed(wb, &el)
                .unwrap();
            let windowed_context = windowed_context.make_current().unwrap();
            let context = glow::Context::from_loader_function(|s| {
                windowed_context.get_proc_address(s) as *const _
            });
            (context, el, windowed_context, "#version 410")
        };
        trace!("Window created");

        // GL SETUP
        let mut ctx = GlContext::new(gl, shader_version);
        let (vend, rend, vers, shader_vers) = ctx.get_info();
        info!(
            "GL context created.
  Vendor: {}
  Renderer: {}
  Version: {}
  Shader version: {}",
            vend, rend, vers, shader_vers
        );

        // EVENT LOOP
        {
            use glutin::event::{Event, WindowEvent};
            use glutin::event_loop::ControlFlow;

            let mut frames = 0;
            let mut loop_time = Instant::now();

            event_loop.run(move |event, _, control_flow| {
                *control_flow = ControlFlow::Poll;
                match event {
                    Event::LoopDestroyed => {
                        info!("Event::LoopDestroyed!");
                        return;
                    }
                    Event::EventsCleared => {
                        let now = Instant::now();
                        let dt = now - loop_time;
                        let num_objects = ctx.update(dt);
                        loop_time = now;

                        frames += 1;
                        const FRAMES: u32 = 100;
                        if frames % FRAMES == 0 {
                            let fps = 1.0 / dt.as_secs_f64();
                            info!("{} objects, {:.03} fps", num_objects, fps);
                        }
                        windowed_context.window().request_redraw();
                    }
                    Event::WindowEvent { ref event, .. } => match event {
                        WindowEvent::Resized(logical_size) => {
                            info!("WindowEvent::Resized: {:?}", logical_size);
                            let dpi_factor = windowed_context.window().hidpi_factor();
                            windowed_context.resize(logical_size.to_physical(dpi_factor));
                        }
                        WindowEvent::RedrawRequested => {
                            ctx.draw();
                            windowed_context.swap_buffers().unwrap();
                        }
                        WindowEvent::CloseRequested => {
                            info!("WindowEvent::CloseRequested");
                            // Don't need to drop Context explicitly,
                            // it'll happen when we exit.
                            *control_flow = ControlFlow::Exit
                        }
                        _ => (),
                    },
                    _ => (),
                }
            });
        }
    }
}

pub fn main() {
    #[cfg(target_arch = "wasm32")]
    run_wasm();
    #[cfg(not(target_arch = "wasm32"))]
    run_glutin();
}
