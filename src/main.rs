// Suggested logging level for debugging:
// env RUST_LOG=info cargo run

use glow::*;
use log::*;

// Shortcuts for various OpenGL types.

type Texture = <Context as glow::HasContext>::Texture;
type Sampler = <Context as glow::HasContext>::Sampler;
type Program = <Context as glow::HasContext>::Program;
type VertexArray = <Context as glow::HasContext>::VertexArray;
type Framebuffer = <Context as glow::HasContext>::Framebuffer;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen(start))]
pub fn wasm_main() {
    main();
}

/// A type that contains all the STUFF we need for displaying graphics
/// and handling events on both desktop and web.
/// Anything it contains is specialized to the correct type via cfg flags
/// at compile time, rather than trying to use generics or such.
struct GlContext {
    gl: glow::Context,
    program: Program,
    vertex_array: VertexArray,
    pipeline: QuadPipeline,
}

impl Drop for GlContext {
    fn drop(&mut self) {
        unsafe {
            self.pipeline.dispose(&self.gl);
            self.gl.delete_program(self.program);
            self.gl.delete_vertex_array(self.vertex_array);
        }
    }
}

impl GlContext {
    fn create_program(
        gl: &glow::Context,
        vertex_src: &str,
        fragment_src: &str,
        shader_version: &str,
    ) -> Program {
        let shader_sources = [
            (glow::VERTEX_SHADER, vertex_src),
            (glow::FRAGMENT_SHADER, fragment_src),
        ];

        unsafe {
            let program = gl.create_program().expect("Cannot create program");
            let mut shaders = Vec::with_capacity(shader_sources.len());

            for (shader_type, shader_source) in shader_sources.iter() {
                let shader = gl
                    .create_shader(*shader_type)
                    .expect("Cannot create shader");
                gl.shader_source(shader, &format!("{}\n{}", shader_version, shader_source));
                gl.compile_shader(shader);
                if !gl.get_shader_compile_status(shader) {
                    panic!(gl.get_shader_info_log(shader));
                }
                gl.attach_shader(program, shader);
                shaders.push(shader);
            }

            gl.link_program(program);
            if !gl.get_program_link_status(program) {
                panic!(gl.get_program_info_log(program));
            }

            for shader in shaders {
                gl.detach_shader(program, shader);
                gl.delete_shader(shader);
            }
            program
        }
    }

    unsafe fn create_texture(gl: &glow::Context) -> Texture {
        let texture = gl.create_texture().unwrap();
        texture
    }

    fn new(gl: glow::Context, shader_version: &str) -> Self {
        // GL SETUP
        unsafe {
            let vertex_array = gl
                .create_vertex_array()
                .expect("Cannot create vertex array");
            gl.bind_vertex_array(Some(vertex_array));

            let vertex_shader_source = r#"const vec2 verts[3] = vec2[3](
                vec2(0.5f, 1.0f),
                vec2(0.0f, 0.0f),
                vec2(1.0f, 0.0f)
            );
            out vec2 vert;
            void main() {
                vert = verts[gl_VertexID];
                gl_Position = vec4(vert - 0.5, 0.0, 1.0);
            }"#;
            let fragment_shader_source = r#"precision mediump float;
            in vec2 vert;
            out vec4 color;
            void main() {
                color = vec4(vert, 0.5, 1.0);
            }"#;
            let program = Self::create_program(
                &gl,
                vertex_shader_source,
                fragment_shader_source,
                shader_version,
            );

            gl.clear_color(0.1, 0.2, 0.3, 1.0);
            //gl.use_program(Some(program));
            let mut pipeline = QuadPipeline::new(program);
            let texture = Self::create_texture(&gl);
            let drawcall = QuadDrawCall::new(texture, SamplerSpec {});
            pipeline.drawcalls.push(drawcall);
            GlContext {
                gl,
                program,
                vertex_array,
                pipeline,
            }
        }
    }

    pub fn get_sampler(&mut self, spec: &SamplerSpec) -> Sampler {
        unimplemented!()
    }
}

/// Input to an instance
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct DrawParam {
    /// TODO: euclid, vek, or what?  UGH.
    pub dest: [f32; 2],
    //pub dest: Point2,
    /*
    /// A portion of the drawable to clip, as a fraction of the whole image.
    /// Defaults to the whole image `(0,0 to 1,1)` if omitted.
    pub src: Rect,
    /// The position to draw the graphic expressed as a `Point2`.
    pub dest: mint::Point2<f32>,
    /// The orientation of the graphic in radians.
    pub rotation: f32,
    /// The x/y scale factors expressed as a `Vector2`.
    pub scale: mint::Vector2<f32>,
    /// An offset from the center for transform operations like scale/rotation,
    /// with `0,0` meaning the origin and `1,1` meaning the opposite corner from the origin.
    /// By default these operations are done from the top-left corner, so to rotate something
    /// from the center specify `Point2::new(0.5, 0.5)` here.
    pub offset: mint::Point2<f32>,
    /// A color to draw the target with.
    /// Default: white.
    pub color: Color,
    */
}

/// Data we need for each quad instance.
/// DrawParam gets turned into this, eventually.
/// We have to be *quite particular* about layout since this gets
/// fed straight to the shader.
///
/// TODO: Currently the shader doesn't use src or color though.
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct QuadData {
    transform: [f32; 16],
    rect: [f32; 4],
    color: [f32; 4],
}

/// A description of a sampler.  We cache the actual
/// samplers as needed in the GlContext.
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct SamplerSpec {}

pub struct QuadDrawCall {
    texture: Texture,
    sampler: SamplerSpec,
    instances: Vec<QuadData>,
}

impl QuadDrawCall {
    fn new(texture: Texture, sampler: SamplerSpec) -> Self {
        Self {
            texture,
            sampler,
            instances: vec![],
        }
    }

    unsafe fn draw(&self, gl: &Context) {
        // bind texture
        // bind sampler
        //let num_vertices = instances.len() * 3;
        gl.draw_arrays(glow::TRIANGLES, 0, 3);
    }

    /// Destroy this thing's resources using the given gl context.
    /// Must be the same gl context that created this, natch.
    /// Horrible things will happen if it isn't.
    ///
    /// TODO: Arc textures?
    /// TODO: Debug ID?
    unsafe fn dispose(&mut self, gl: &Context) {
        gl.delete_texture(self.texture);
    }
}

pub struct QuadPipeline {
    drawcalls: Vec<QuadDrawCall>,
    program: Program,
}

impl QuadPipeline {
    fn new(program: Program) -> Self {
        Self {
            drawcalls: vec![],
            program,
        }
    }

    unsafe fn draw(&self, gl: &Context) {
        gl.use_program(Some(self.program));
        for dc in self.drawcalls.iter() {
            dc.draw(gl);
        }
    }

    unsafe fn dispose(&mut self, gl: &Context) {
        for mut dc in self.drawcalls.drain(..) {
            dc.dispose(gl);
        }
        gl.delete_program(self.program);
    }
}

/// Currently, no input framebuffers or such.
/// We're not actually intending to reproduce Rendy's Graph type here.
/// This may eventually feed into a bounce buffer or such though.
pub struct RenderPass {
    _output_framebuffer: Framebuffer,
    _pipelines: Vec<QuadPipeline>,
}

#[cfg(target_arch = "wasm32")]
fn run_wasm() {
    unsafe {
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
            if let Some(ictx) = &ctx {
                ictx.gl.clear(glow::COLOR_BUFFER_BIT);
                //ictx.gl.draw_arrays(glow::TRIANGLES, 0, 3);
                ictx.pipeline.draw(&ictx.gl);
            }

            if !*running {
                // Drop context, deleting its contents.
                ctx = None;
            }
        });
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn run_glutin() {
    // CONTEXT CREATION
    unsafe {
        // Create a context from a glutin window on non-wasm32 targets
        let (gl, event_loop, windowed_context, shader_version) = {
            let el = glutin::event_loop::EventLoop::new();
            let wb = glutin::window::WindowBuilder::new()
                .with_title("Hello triangle!")
                .with_inner_size(glutin::dpi::LogicalSize::new(1024.0, 768.0));
            let windowed_context = glutin::ContextBuilder::new()
                .with_vsync(true)
                .build_windowed(wb, &el)
                .unwrap();
            let windowed_context = windowed_context.make_current().unwrap();
            let context = glow::Context::from_loader_function(|s| {
                windowed_context.get_proc_address(s) as *const _
            });
            (context, el, windowed_context, "#version 410")
        };

        // GL SETUP
        let ctx = GlContext::new(gl, shader_version);

        // EVENT LOOP
        {
            use glutin::event::{Event, WindowEvent};
            use glutin::event_loop::ControlFlow;

            event_loop.run(move |event, _, control_flow| {
                *control_flow = ControlFlow::Wait;
                match event {
                    Event::LoopDestroyed => {
                        info!("Event::LoopDestroyed!");
                        return;
                    }
                    Event::EventsCleared => {
                        info!("EventsCleared");
                        windowed_context.window().request_redraw();
                    }
                    Event::WindowEvent { ref event, .. } => match event {
                        WindowEvent::Resized(logical_size) => {
                            info!("WindowEvent::Resized: {:?}", logical_size);
                            let dpi_factor = windowed_context.window().hidpi_factor();
                            windowed_context.resize(logical_size.to_physical(dpi_factor));
                        }
                        WindowEvent::RedrawRequested => {
                            info!("WindowEvent::RedrawRequested");
                            ctx.gl.clear(glow::COLOR_BUFFER_BIT);
                            ctx.gl.draw_arrays(glow::TRIANGLES, 0, 3);
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
