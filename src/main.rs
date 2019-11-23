// Suggested logging level for debugging:
// env RUST_LOG=info cargo run

use ggraphics::*;
use glow;
use oorandom;
use winit;

use std::time::Duration;

struct GameState {
    ctx: GlContext,
    lulz: oorandom::Rand32,
}

impl GameState {
    pub fn new(gl: glow::Context) -> Self {
        let mut ctx = GlContext::new(gl);
        unsafe {
            // Render our bunnies to a texture
            let mut pass1 = RenderPass::new(&mut ctx, 800, 600, (0.1, 0.2, 0.3, 1.0));
            let shader = GlContext::default_shader(&ctx);
            let mut pipeline = QuadPipeline::new(&ctx, shader);
            let texture = {
                let image_bytes = include_bytes!("data/wabbit_alpha.png");
                let image_rgba = image::load_from_memory(image_bytes).unwrap().to_rgba();
                let (w, h) = image_rgba.dimensions();
                let image_rgba_bytes = image_rgba.into_raw();
                TextureHandle::new(&ctx, &image_rgba_bytes, w as usize, h as usize).into_shared()
            };
            let drawcall = QuadDrawCall::new(&mut ctx, texture, SamplerSpec::default(), &pipeline);
            pipeline.drawcalls.push(drawcall);
            pass1.pipelines.push(pipeline);
            let texture2 = pass1.get_texture().unwrap();
            ctx.passes.push(pass1);

            // Render that texture to the screen
            let mut pass2 = RenderPass::new_screen(&mut ctx, 800, 600, (0.6, 0.6, 0.6, 1.0));
            let shader = GlContext::default_shader(&ctx);
            let mut pipeline = QuadPipeline::new(&ctx, shader);
            let drawcall = QuadDrawCall::new(&mut ctx, texture2, SamplerSpec::default(), &pipeline);
            pipeline.drawcalls.push(drawcall);
            pass2.pipelines.push(pipeline);
            ctx.passes.push(pass2);
        }

        {
            let pass2 = &mut ctx.passes[1];
            // yes I know the numbers make you cry
            // i drink your tears :D
            for pipeline in pass2.pipelines.iter_mut() {
                for drawcall in pipeline.drawcalls.iter_mut() {
                    for i in 0..10 {
                        let offset = (i as f32) * 0.2 - 1.0;
                        let quad = QuadData {
                            offset: [0.0, 0.0],
                            color: [1.0, 1.0, 1.0, 1.0],
                            src_rect: [0.0, 0.0, 1.0, 1.0],
                            dst_rect: [-offset, -offset, 0.5, 0.5],
                            rotation: 0.0,
                        };
                        drawcall.add(quad)
                    }
                }
            }
        }

        let lulz = oorandom::Rand32::new(12345);
        Self { ctx, lulz }
    }

    pub fn update(&mut self, frametime: Duration) -> usize {
        // This adds more quads as long as our frame doesn't take too long
        // We max out at 17 ms per frame; this method of measurement
        // is pretty imprecise and there will be jitter, but it should
        // be okay for order-of-magnitude.
        let mut total_instances = 0;
        if frametime.as_secs_f64() < 0.017 {
            {
                let pass1 = &mut self.ctx.passes[0];
                for pipeline in pass1.pipelines.iter_mut() {
                    for drawcall in pipeline.drawcalls.iter_mut() {
                        for _ in 0..1 {
                            drawcall.add_random(&mut self.lulz);
                        }
                        total_instances += drawcall.instances.len();
                    }
                }

                let drawcall2 = &mut self.ctx.passes[1].pipelines[0].drawcalls[0];
                for instance in drawcall2.instances.iter_mut() {
                    instance.rotation += 0.02;
                }
                // TODO: Make this API not suck.
                drawcall2.dirty = true;
            }
        }
        total_instances
    }
}

trait Window {
    fn request_redraw(&self);
    fn swap_buffers(&self);
}

/// Used for desktop
#[cfg(not(target_arch = "wasm32"))]
impl Window for glutin::WindowedContext<glutin::PossiblyCurrent> {
    fn request_redraw(&self) {
        self.window().request_redraw();
    }
    fn swap_buffers(&self) {
        self.swap_buffers().unwrap();
    }
}

/// Used for wasm
#[cfg(target_arch = "wasm32")]
impl Window for winit::window::Window {
    fn request_redraw(&self) {
        self.request_redraw();
    }
    fn swap_buffers(&self) {
        let msg = format!("swapped buffers");
        web_sys::console::log_1(&wasm_bindgen::JsValue::from_str(&msg));
    }
}

fn mainloop(
    gl: glow::Context,
    event_loop: winit::event_loop::EventLoop<()>,
    window: impl Window + 'static,
) {
    use instant::Instant;
    use log::*;
    use winit::event::{Event, WindowEvent};
    use winit::event_loop::ControlFlow;
    let mut state = GameState::new(gl);
    let (vend, rend, vers, shader_vers) = state.ctx.get_info();
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
        let mut frames = 0;
        let target_dt = Duration::from_micros(16_660);
        let mut last_frame = Instant::now();
        let mut next_frame = last_frame + target_dt;

        event_loop.run(move |event, _, control_flow| {
            *control_flow = ControlFlow::WaitUntil(next_frame);
            match event {
                Event::LoopDestroyed => {
                    info!("Event::LoopDestroyed!");
                    return;
                }
                Event::EventsCleared => {
                    println!("Events cleared");
                    let now = Instant::now();
                    let dt = now - last_frame;
                    if dt >= target_dt {
                        #[cfg(target_arch = "wasm32")]
                        {
                            let msg = format!("Events cleared: {:?}, target: {:?}", dt, target_dt);
                            web_sys::console::log_1(&wasm_bindgen::JsValue::from_str(&msg));
                        }
                        let num_objects = state.update(dt);
                        last_frame = now;
                        next_frame = now + target_dt;

                        frames += 1;
                        const FRAMES: u32 = 100;
                        if frames % FRAMES == 0 {
                            let fps = 1.0 / dt.as_secs_f64();
                            info!("{} objects, {:.03} fps", num_objects, fps);
                        }
                        window.request_redraw();
                    }
                }
                Event::WindowEvent { ref event, .. } => match event {
                    WindowEvent::Resized(logical_size) => {
                        info!("WindowEvent::Resized: {:?}", logical_size);
                        //let dpi_factor = windowed_context.window().hidpi_factor();
                        //windowed_context.resize(logical_size.to_physical(dpi_factor));
                    }
                    WindowEvent::RedrawRequested => {
                        state.ctx.draw();
                        window.swap_buffers();
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

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
fn run_wasm() {
    use glow::HasRenderLoop;
    use instant::Instant;

    console_error_panic_hook::set_once();
    use winit::event::{Event, WindowEvent};
    use winit::event_loop::ControlFlow;
    use winit::platform::web::WindowExtWebSys;
    let event_loop = winit::event_loop::EventLoop::new();
    let win = winit::window::WindowBuilder::new()
        .with_inner_size(winit::dpi::LogicalSize::new(1024.0, 768.0))
        .with_title("Heckin' winit")
        .build(&event_loop)
        .unwrap();

    let document = web_sys::window()
        .expect("Failed to obtain window")
        .document()
        .expect("Failed to obtain document");

    // Shove winit's canvas into the document
    document
        .body()
        .expect("Failed to obtain body")
        .append_child(&win.canvas())
        .unwrap();

    // Wire winit's context into glow
    let gl = {
        use wasm_bindgen::JsCast;
        let webgl2_context = win
            .canvas()
            .get_context("webgl2")
            .unwrap()
            .unwrap()
            .dyn_into::<web_sys::WebGl2RenderingContext>()
            .unwrap();
        glow::Context::from_webgl2_context(webgl2_context)
    };

    mainloop(gl, event_loop, win);
}

#[cfg(not(target_arch = "wasm32"))]
fn run_glutin() {
    use log::*;
    pretty_env_logger::init();
    // CONTEXT CREATION
    unsafe {
        // Create a context from a glutin window on non-wasm32 targets
        let (gl, event_loop, windowed_context) = {
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
            (context, el, windowed_context)
        };
        trace!("Window created");

        // GL SETUP
        mainloop(gl, event_loop, windowed_context);
    }
}

pub fn main() {
    #[cfg(target_arch = "wasm32")]
    run_wasm();
    #[cfg(not(target_arch = "wasm32"))]
    run_glutin();
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen(start))]
pub fn wasm_main() {
    main();
}
