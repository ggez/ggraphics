// Suggested logging level for debugging:
// env RUST_LOG=info cargo run
//
// Next up: Vertex array objects and buffer objects.
// VAO's basically describe what the array looks like.
// VBO's contain the data.
// See the Rust gamedev discord for more, around Nov 6 2019
// 16:00 EST

use std::mem;

use glow::*;
use log::*;

// Shortcuts for various OpenGL types.

type Texture = <Context as glow::HasContext>::Texture;
type Sampler = <Context as glow::HasContext>::Sampler;
type Program = <Context as glow::HasContext>::Program;
type VertexArray = <Context as glow::HasContext>::VertexArray;
type Framebuffer = <Context as glow::HasContext>::Framebuffer;
type Buffer = <Context as glow::HasContext>::Buffer;

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
            // TODO:
            // By default only one output so this isn't necessary
            // glBindFragDataLocation(shaderProgram, 0, "outColor");
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
            layout(std140) uniform QuadData {
                vec2 offset2,
            } quad_data;
            in vec2 offset;
            out vec2 vert;
            void main() {
                vert = verts[gl_VertexID % 3] + offset;
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
            let mut pipeline = QuadPipeline::new(program);
            let texture = Self::create_texture(&gl);
            let mut drawcall = QuadDrawCall::new(&gl, texture, SamplerSpec {}, &pipeline.program);
            drawcall.add(QuadData {
                offset: [-0.5, 0.5],
            });
            drawcall.add(QuadData {
                offset: [-0.5, 0.5],
            });
            drawcall.add(QuadData {
                offset: [-0.5, 0.5],
            });
            drawcall.add(QuadData { offset: [0.5, 0.5] });
            drawcall.add(QuadData { offset: [0.5, 0.5] });
            drawcall.add(QuadData { offset: [0.5, 0.5] });
            drawcall.add(QuadData { offset: [0.0, 0.0] });
            drawcall.add(QuadData { offset: [0.0, 0.0] });
            drawcall.add(QuadData { offset: [0.0, 0.0] });
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
/// TODO: Figure out correct alignment...
//#[repr(C, align(16))]
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct QuadData {
    //transform: [f32; 16],
    //rect: [f32; 4],
    //color: [f32; 4],
    offset: [f32; 2],
}

/// A description of a sampler.  We cache the actual
/// samplers as needed in the GlContext.
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct SamplerSpec {}

pub struct QuadDrawCall {
    texture: Texture,
    sampler: SamplerSpec,
    instances: Vec<QuadData>,
    vbo: Buffer,
    vao: VertexArray,
    ubo: Buffer,
}

impl QuadDrawCall {
    fn new(gl: &Context, texture: Texture, sampler: SamplerSpec, shader: &Program) -> Self {
        // TODO: Audit unsafe
        unsafe {
            let vao = gl.create_vertex_array().unwrap();
            gl.bind_vertex_array(Some(vao));

            // TODO: Double-check that this bind sticks to the VAO,
            // though currently it doesn't matter 'cause we re-bind it so that
            // we can fill it with instance data on draw
            // Okay, it looks like which VBO is bound IS part of the VAO data,
            // and it is stored *when glVertexAttribPointer is called*.
            // According to https://open.gl/drawing at least.
            // And, that stuff I THINK is stored IN THE ATTRIBUTE INFO,
            // in this case `offset_attrib`, and then THAT gets attached
            // to the VAO by enable_vertex_attrib_array()
            //
            // Wait, no, maybe it has to be done after the VAO is bound,
            // and the attribs are part of that?  The same link
            // says that, but says it AFTER it talks about the VBO, so
            let vbo = gl.create_buffer().unwrap();
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(vbo));

            // TODO: https://github.com/grovesNL/glow/issues/54
            let offset_attrib = gl.get_attrib_location(*shader, "offset") as u32;
            gl.vertex_attrib_pointer_f32(
                offset_attrib,
                2,
                glow::FLOAT,
                false,
                (2 * mem::size_of::<f32>()) as i32,
                0,
            );
            // TODO: Double-check if 3 is correct
            //gl.vertex_attrib_divisor(offset_attrib, 3);
            gl.enable_vertex_attrib_array(offset_attrib);

            let ubo = gl.create_buffer().unwrap();
            let quad_idx = gl.get_uniform_block_index(*shader, "QuadData").unwrap();

            gl.bind_vertex_array(None);
            Self {
                vbo,
                ubo,
                vao,
                texture,
                sampler,
                instances: vec![],
            }
        }
    }

    pub fn add(&mut self, quad: QuadData) {
        self.instances.push(quad);
    }

    /// Upload the array of instances to our VBO
    unsafe fn upload_instances(&self, gl: &Context) {
        gl.bind_buffer(glow::ARRAY_BUFFER, Some(self.vbo));
        // TODO: Invalidate buffer on change instead of refilling it all the time
        // TODO: Make instance data cast not suck
        let num_bytes = self.instances.len() * mem::size_of::<QuadData>();
        let bytes_ptr = self.instances.as_ptr() as *const u8;
        let bytes_slice = std::slice::from_raw_parts(bytes_ptr, num_bytes);

        // TODO: Make usage sensible
        gl.buffer_data_u8_slice(glow::ARRAY_BUFFER, bytes_slice, glow::STREAM_DRAW);
        gl.bind_buffer(glow::ARRAY_BUFFER, None);

        gl.bind_buffer(glow::UNIFORM_BUFFER, Some(self.ubo));
        gl.buffer_data_u8_slice(glow::UNIFORM_BUFFER, bytes_slice, glow::STREAM_DRAW);
        gl.bind_buffer(glow::UNIFORM_BUFFER, None);
    }

    unsafe fn draw(&self, gl: &Context) {
        self.upload_instances(gl);
        // bind texture
        // bind sampler
        // Use this when we figure out heckin' instancing
        //let num_vertices = self.instances.len() * 3;
        let num_vertices = self.instances.len();
        //gl.draw_arrays(glow::TRIANGLES, 0, 3);
        gl.bind_vertex_array(Some(self.vao));
        gl.draw_arrays(glow::TRIANGLES, 0, num_vertices as i32);
    }

    /// Destroy this thing's resources using the given gl context.
    /// Must be the same gl context that created this, natch.
    /// Horrible things will happen if it isn't.
    ///
    /// TODO: Arc textures?
    /// TODO: Debug ID?
    unsafe fn dispose(&mut self, gl: &Context) {
        gl.delete_texture(self.texture);
        gl.delete_buffer(self.vbo);
        gl.delete_vertex_array(self.vao);
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
                // TODO: Specific GL versions.
                .with_gl(glutin::GlRequest::Latest)
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

        // GL SETUP
        let ctx = GlContext::new(gl, shader_version);

        // EVENT LOOP
        {
            use glutin::event::{Event, WindowEvent};
            use glutin::event_loop::ControlFlow;

            event_loop.run(move |event, _, control_flow| {
                *control_flow = ControlFlow::Poll;
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
                            //ctx.gl.draw_arrays(glow::TRIANGLES, 0, 3);
                            ctx.pipeline.draw(&ctx.gl);
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
