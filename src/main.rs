// Suggested logging level for debugging:
// env RUST_LOG=info cargo run
//
// Next up: Textures

use std::convert::TryFrom;
use std::mem;

use glow::*;
use image;
use log::*;

// Shortcuts for various OpenGL types.

type GlTexture = <Context as glow::HasContext>::Texture;
type Sampler = <Context as glow::HasContext>::Sampler;
type GlProgram = <Context as glow::HasContext>::Program;
type GlVertexArray = <Context as glow::HasContext>::VertexArray;
type GlFramebuffer = <Context as glow::HasContext>::Framebuffer;
type GlBuffer = <Context as glow::HasContext>::Buffer;
type GlUniformLocation = <Context as glow::HasContext>::UniformLocation;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen(start))]
pub fn wasm_main() {
    main();
}

/// TODO: Figure out what to do with this, and whether it can work on wasm.
/// For now though, Rc is fine.
type Rc<T> = std::rc::Rc<T>;
//type Ref<T> = Arc<T>;

/// A type that contains all the STUFF we need for displaying graphics
/// and handling events on both desktop and web.
/// Anything it contains is specialized to the correct type via cfg flags
/// at compile time, rather than trying to use generics or such.
pub struct GlContext {
    gl: Rc<glow::Context>,
    pipelines: Vec<QuadPipeline>,
}

impl GlContext {
    fn new(gl: glow::Context, shader_version: &str) -> Self {
        // GL SETUP
        unsafe {
            let vertex_shader_source = r#"const vec2 verts[6] = vec2[6](
                vec2(0.0f, 0.0f),
                vec2(1.0f, 1.0f),
                vec2(0.0f, 1.0f),

                vec2(0.0f, 0.0f),
                vec2(1.0f, 0.0f),
                vec2(1.0f, 1.0f)
            );
            in vec2 offset;
            in vec2 offset2;
            out vec2 vert;
            out vec2 tex_coord;
            void main() {
                vert = verts[gl_VertexID % 6] / 10.0 + offset + offset2;
                tex_coord = verts[gl_VertexID];
                gl_Position = vec4(vert, 0.0, 1.0);
            }"#;
            let fragment_shader_source = r#"precision mediump float;
            in vec2 vert;
            in vec2 tex_coord;
            uniform sampler2D tex;

            out vec4 color;

            void main() {
                //color = vec4(vert, 0.5, 1.0);
                color = vec4(texture(tex, tex_coord).rgb, 1.0);
                //color = vec4(tex_coord, 0.5, 0.5) + vec4(texture(tex, tex_coord).rgb, 0.0);
            }"#;

            gl.clear_color(0.1, 0.2, 0.3, 1.0);
            gl.enable(glow::BLEND);
            gl.blend_func(glow::SRC_ALPHA, glow::ONE_MINUS_SRC_ALPHA);
            let mut s = GlContext {
                gl: Rc::new(gl),
                pipelines: vec![],
            };
            s.register_debug_callback();
            let shader = Shader::new(
                &s,
                vertex_shader_source,
                fragment_shader_source,
                shader_version,
            );
            let mut pipeline = QuadPipeline::new(&s, shader);
            let texture = {
                let image_bytes = include_bytes!("data/blue.png");
                let image_rgba = image::load_from_memory(image_bytes).unwrap().to_rgba();
                let (w, h) = image_rgba.dimensions();
                let image_rgba_bytes = image_rgba.into_raw();
                //make_texture(&gl, &image_rgba_bytes, w as usize, h as usize)
                Texture::new(&s, &image_rgba_bytes, w as usize, h as usize)
            };
            let drawcall = QuadDrawCall::new(&s, texture, SamplerSpec {}, &pipeline.shader);
            pipeline.drawcalls.push(drawcall);
            s.pipelines.push(pipeline);
            s
        }
    }

    /// Log OpenGL errors as possible.
    /// TODO: make it only happen in debug mode.
    /// ALSO TODO: Figure out why this panics on wasm
    fn register_debug_callback(&self) {
        #[cfg(not(target_arch = "wasm32"))]
        unsafe {
            self.gl
                .debug_message_callback(|source, typ, id, severity, message| {
                    debug!(
                        "GL error type {} id {} from {} severity {}: {}",
                        typ, id, source, severity, message
                    );
                });
        }
    }

    pub fn get_sampler(&mut self, _spec: &SamplerSpec) -> Sampler {
        unimplemented!()
    }
}

pub struct Texture {
    ctx: Rc<glow::Context>,
    tex: GlTexture,
}

impl Drop for Texture {
    fn drop(&mut self) {
        unsafe {
            self.ctx.delete_texture(self.tex);
        }
    }
}

impl Texture {
    pub fn new(ctx: &GlContext, rgba: &[u8], width: usize, height: usize) -> Self {
        assert_eq!(width * height * 4, rgba.len());
        let gl = &*ctx.gl;
        unsafe {
            let t = gl.create_texture().unwrap();
            gl.active_texture(glow::TEXTURE0);
            gl.bind_texture(glow::TEXTURE_2D, Some(t));
            gl.tex_image_2d(
                glow::TEXTURE_2D,                   // Texture target
                0,                                  // mipmap level
                i32::try_from(glow::RGBA).unwrap(), // format to store the texture in (can't fail)
                i32::try_from(width).unwrap(),      // width
                i32::try_from(height).unwrap(),     // height
                0,                                  // border, must always be 0, lulz
                glow::RGBA,                         // format to load the texture from
                glow::UNSIGNED_BYTE,                // Type of each color element
                Some(rgba),                         // Actual data
            );
            gl.bind_texture(glow::TEXTURE_2D, None);
            Self {
                ctx: ctx.gl.clone(),
                tex: t,
            }
        }
    }
}

pub struct Shader {
    ctx: Rc<glow::Context>,
    program: GlProgram,
}

impl Drop for Shader {
    fn drop(&mut self) {
        unsafe {
            self.ctx.delete_program(self.program);
        }
    }
}

impl Shader {
    pub fn new(
        ctx: &GlContext,
        vertex_src: &str,
        fragment_src: &str,
        shader_version: &str,
    ) -> Shader {
        let gl = &*ctx.gl;
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
            Shader {
                ctx: ctx.gl.clone(),
                program,
            }
        }
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
    ctx: Rc<glow::Context>,
    texture: Texture,
    _sampler: SamplerSpec,
    instances: Vec<QuadData>,
    vbo: GlBuffer,
    vao: GlVertexArray,
    instance_vbo: GlBuffer,
    texture_location: GlUniformLocation,

    lulz: oorandom::Rand32,
}

impl Drop for QuadDrawCall {
    fn drop(&mut self) {
        unsafe {
            self.ctx.delete_vertex_array(self.vao);
            self.ctx.delete_buffer(self.vbo);
            self.ctx.delete_buffer(self.instance_vbo);
        }
    }
}

impl QuadDrawCall {
    fn new(ctx: &GlContext, texture: Texture, sampler: SamplerSpec, shader: &Shader) -> Self {
        let gl = &*ctx.gl;
        // TODO: Audit unsafe
        unsafe {
            let vao = gl.create_vertex_array().unwrap();
            gl.bind_vertex_array(Some(vao));

            // Okay, it looks like which VBO is bound IS part of the VAO data,
            // and it is stored *when glVertexAttribPointer is called*.
            // According to https://open.gl/drawing at least.
            // And, that stuff I THINK is stored IN THE ATTRIBUTE INFO,
            // in this case `offset_attrib`, and then THAT gets attached
            // to the VAO by enable_vertex_attrib_array()
            let vbo = gl.create_buffer().unwrap();
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(vbo));

            // TODO: https://github.com/grovesNL/glow/issues/54
            let offset_attrib =
                u32::try_from(gl.get_attrib_location(shader.program, "offset")).unwrap();
            gl.vertex_attrib_pointer_f32(
                offset_attrib,
                2,
                glow::FLOAT,
                false,
                // Can we jsut say the stride of this guy is 0, since
                // we never actually use it (yet)?
                //(2 * mem::size_of::<f32>()) as i32,
                0,
                0,
            );
            gl.enable_vertex_attrib_array(offset_attrib);

            // Now create another VBO containing per-instance data
            let instance_vbo = gl.create_buffer().unwrap();
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(instance_vbo));
            let offset2_attrib =
                u32::try_from(gl.get_attrib_location(shader.program, "offset2")).unwrap();
            gl.vertex_attrib_pointer_f32(
                offset2_attrib,
                2,
                glow::FLOAT,
                false,
                (2 * mem::size_of::<f32>()) as i32,
                0,
            );
            gl.vertex_attrib_divisor(offset2_attrib, 1);
            gl.enable_vertex_attrib_array(offset2_attrib);

            let texture_location = gl.get_uniform_location(shader.program, "tex").unwrap();
            //gl.uniform_1_i32(Some(texture_location), 0);

            gl.bind_vertex_array(None);

            let lulz = oorandom::Rand32::new(314159);
            Self {
                ctx: ctx.gl.clone(),
                vbo,
                vao,
                texture,
                _sampler: sampler,
                instance_vbo,
                texture_location,
                instances: vec![],
                lulz,
            }
        }
    }

    pub fn add(&mut self, quad: QuadData) {
        self.instances.push(quad);
    }

    fn add_random(&mut self) {
        let x = self.lulz.rand_float() * 2.0 - 1.0;
        let y = self.lulz.rand_float() * 2.0 - 1.0;
        let quad = QuadData { offset: [x, y] };
        self.instances.push(quad);
    }

    /// Upload the array of instances to our VBO
    unsafe fn upload_instances(&mut self, gl: &Context) {
        // TODO: This is just for testing
        self.add_random();

        gl.bind_buffer(glow::ARRAY_BUFFER, Some(self.vbo));
        // TODO: Invalidate buffer on change instead of refilling it all the time
        // TODO: Make instance data cast not suck
        let num_bytes = self.instances.len() * mem::size_of::<QuadData>();
        let bytes_ptr = self.instances.as_ptr() as *const u8;
        let empty_slice = &vec![0; self.instances.len() * 8 * 6];
        let bytes_slice2 = std::slice::from_raw_parts(bytes_ptr, num_bytes);

        // TODO: Make usage sensible
        // Dummy data for per-vertex attributes
        // TODO: This is a little awful since we send a big pile of
        // data to the GPU which we never use, is there a better way?
        // Can we just give it an empty or empty-ish array?
        gl.buffer_data_u8_slice(glow::ARRAY_BUFFER, empty_slice, glow::STREAM_DRAW);

        // Fill instance buffer
        gl.bind_buffer(glow::ARRAY_BUFFER, Some(self.instance_vbo));
        gl.buffer_data_u8_slice(glow::ARRAY_BUFFER, bytes_slice2, glow::STREAM_DRAW);
        gl.bind_buffer(glow::ARRAY_BUFFER, None);
    }

    unsafe fn draw(&mut self, gl: &Context) {
        self.upload_instances(gl);
        // bind sampler
        // Bind VAO
        let num_instances = self.instances.len();
        let num_vertices = 6;
        gl.bind_vertex_array(Some(self.vao));
        // Bind texture
        // TODO: is this active_texture() call necessary?
        gl.active_texture(glow::TEXTURE0);
        gl.bind_texture(glow::TEXTURE_2D, Some(self.texture.tex));
        gl.uniform_1_i32(Some(self.texture_location), 0);
        gl.draw_arrays_instanced(
            glow::TRIANGLES,
            0,
            num_vertices as i32,
            num_instances as i32,
        );
    }
}

pub struct QuadPipeline {
    //ctx: Rc<glow::Context>,
    drawcalls: Vec<QuadDrawCall>,
    shader: Shader,
}

impl QuadPipeline {
    fn new(_ctx: &GlContext, shader: Shader) -> Self {
        Self {
            //ctx: ctx.gl.clone(),
            drawcalls: vec![],
            shader,
        }
    }

    unsafe fn draw(&mut self, gl: &Context) {
        gl.use_program(Some(self.shader.program));
        for dc in self.drawcalls.iter_mut() {
            dc.draw(gl);
        }
    }
}

/// Currently, no input framebuffers or such.
/// We're not actually intending to reproduce Rendy's Graph type here.
/// This may eventually feed into a bounce buffer or such though.
pub struct RenderPass {
    _output_framebuffer: GlFramebuffer,
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
            if let Some(ictx) = &mut ctx {
                ictx.gl.clear(glow::COLOR_BUFFER_BIT);
                for pipeline in ictx.pipelines.iter_mut() {
                    pipeline.draw(&ictx.gl);
                }
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
        trace!("Window created");

        // GL SETUP
        let mut ctx = GlContext::new(gl, shader_version);
        trace!("GL context created");

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
                        windowed_context.window().request_redraw();
                    }
                    Event::WindowEvent { ref event, .. } => match event {
                        WindowEvent::Resized(logical_size) => {
                            info!("WindowEvent::Resized: {:?}", logical_size);
                            let dpi_factor = windowed_context.window().hidpi_factor();
                            windowed_context.resize(logical_size.to_physical(dpi_factor));
                        }
                        WindowEvent::RedrawRequested => {
                            //info!("WindowEvent::RedrawRequested");
                            ctx.gl.clear(glow::COLOR_BUFFER_BIT);
                            //ctx.gl.draw_arrays(glow::TRIANGLES, 0, 3);
                            for pipeline in ctx.pipelines.iter_mut() {
                                pipeline.draw(&ctx.gl);
                            }
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
