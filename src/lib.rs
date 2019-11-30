//! A basic rendering library designed to run on desktop and in
//! web browsers.  Uses the `glow` crate for OpenGL, WebGL and
//! OpenGL ES.
//!
//! For now, this is mostly an implementation detail of the `ggez`
//! crate.
//!
//! Note this is deliberately NOT thread-safe, because threaded
//! OpenGL is not worth the trouble.  Create your OpenGL context
//! on a particular thread and do all your rendering from that
//! thread.
//! See
//! <https://github.com/FNA-XNA/FNA/blob/76554b7ca3d7aa33229c12c6ab5bf3dbdb114d59/src/FNAPlatform/OpenGLDevice.cs#L10-L39> for more info

// Next up:
// Impl mesh pipelines
// Blend modes!
// Try out termhn's wireframe shader with barycentric coords
// audit unsafes, figure out what can be safe,

#![deny(missing_docs)]
//#![deny(missing_debug_implementations)]
#![deny(unused_results)]
#![warn(bare_trait_objects)]
#![warn(missing_copy_implementations)]

use std::collections::HashMap;
use std::convert::TryFrom;
use std::mem;
use std::rc::Rc;

/// Re-export
pub use glow;

use bytemuck;
use glam::Mat4;
use glow::*;
use log::*;

// Shortcuts for various OpenGL types.

type GlTexture = <Context as glow::HasContext>::Texture;
type GlSampler = <Context as glow::HasContext>::Sampler;
type GlProgram = <Context as glow::HasContext>::Program;
type GlVertexArray = <Context as glow::HasContext>::VertexArray;
type GlFramebuffer = <Context as glow::HasContext>::Framebuffer;
type GlRenderbuffer = <Context as glow::HasContext>::Renderbuffer;
type GlBuffer = <Context as glow::HasContext>::Buffer;
type GlUniformLocation = <Context as glow::HasContext>::UniformLocation;

/// A type that contains all the STUFF we need for displaying graphics
/// and handling events on both desktop and web.
/// Anything it contains is specialized to the correct type via cfg flags
/// at compile time, rather than trying to use generics or such.
pub struct GlContext {
    /// The OpenGL context.
    pub gl: Rc<glow::Context>,
    /// The list of render passes.
    pub passes: Vec<RenderPass>,
    /// Samplers are cached and managed entirely by the GlContext.
    /// You usually only need a few of them so there's no point freeing
    /// them separately, you just ask for the one you want and it gives
    /// it to you.
    samplers: HashMap<SamplerSpec, GlSampler>,
    quad_shader: Shader,
}

fn ortho(left: f32, right: f32, top: f32, bottom: f32, far: f32, near: f32) -> [[f32; 4]; 4] {
    let c0r0 = 2.0 / (right - left);
    let c0r1 = 0.0;
    let c0r2 = 0.0;
    let c0r3 = 0.0;

    let c1r0 = 0.0;
    let c1r1 = 2.0 / (top - bottom);
    let c1r2 = 0.0;
    let c1r3 = 0.0;

    let c2r0 = 0.0;
    let c2r1 = 0.0;
    let c2r2 = -2.0 / (far - near);
    let c2r3 = 0.0;

    let c3r0 = -(right + left) / (right - left);
    let c3r1 = -(top + bottom) / (top - bottom);
    let c3r2 = -(far + near) / (far - near);
    let c3r3 = 1.0;

    // our matrices are column-major, so here we are.
    [
        [c0r0, c0r1, c0r2, c0r3],
        [c1r0, c1r1, c1r2, c1r3],
        [c2r0, c2r1, c2r2, c2r3],
        [c3r0, c3r1, c3r2, c3r3],
    ]
}

fn ortho_mat(left: f32, right: f32, top: f32, bottom: f32, far: f32, near: f32) -> Mat4 {
    Mat4::from_cols_array_2d(&ortho(left, right, top, bottom, far, near))
}
const VERTEX_SHADER_SOURCE: &str = include_str!("data/quad.vert.glsl");
const FRAGMENT_SHADER_SOURCE: &str = include_str!("data/quad.frag.glsl");

impl GlContext {
    /// Create a new `GlContext` from the given `glow::Context`.  Does
    /// basic setup and state setting.
    pub fn new(gl: glow::Context) -> Self {
        // GL SETUP
        unsafe {
            gl.enable(glow::BLEND);
            gl.blend_func(glow::SRC_ALPHA, glow::ONE_MINUS_SRC_ALPHA);
            let gl = Rc::new(gl);
            let quad_shader =
                ShaderHandle::new_raw(gl.clone(), VERTEX_SHADER_SOURCE, FRAGMENT_SHADER_SOURCE);
            let s = GlContext {
                gl,
                passes: vec![],
                samplers: HashMap::new(),
                quad_shader: quad_shader.into_shared(),
            };
            s.register_debug_callback();

            s
        }
    }

    /// Get a copy of the default quad shader.
    pub fn default_shader(&self) -> Shader {
        self.quad_shader.clone()
    }

    /// Log OpenGL errors as possible.
    /// TODO: Figure out why this panics on wasm
    fn register_debug_callback(&self) {
        #[cfg(all(debug_assertions, not(target_arch = "wasm32")))]
        unsafe {
            self.gl
                .debug_message_callback(|source, typ, id, severity, message| {
                    // The ordering of severities is basically awful, best to
                    // not even try and just match them manually.
                    match severity {
                        glow::DEBUG_SEVERITY_HIGH => {
                            error!(
                                "GL error type {} id {} from {}: {}",
                                typ, id, source, message
                            );
                        }
                        glow::DEBUG_SEVERITY_MEDIUM => {
                            warn!(
                                "GL error type {} id {} from {}: {}",
                                typ, id, source, message
                            );
                        }
                        glow::DEBUG_SEVERITY_LOW => {
                            info!(
                                "GL error type {} id {} from {}: {}",
                                typ, id, source, message
                            );
                        }
                        glow::DEBUG_SEVERITY_NOTIFICATION => (),
                        _ => (),
                    }
                });
        }
    }

    /// Get a sampler given the given spec.  Samplers are cached, and
    /// usually few in number, so you shouldn't free them and this handles
    /// caching them for you.
    pub fn get_sampler(&mut self, spec: &SamplerSpec) -> GlSampler {
        let gl = &*self.gl;
        // unsafety: This takes no inputs besides spec, which has
        // constrained types.
        *self.samplers.entry(*spec).or_insert_with(|| unsafe {
            let sampler = gl.create_sampler().unwrap();
            gl.sampler_parameter_i32(
                sampler,
                glow::TEXTURE_MIN_FILTER,
                spec.min_filter.to_gl() as i32,
            );
            gl.sampler_parameter_i32(
                sampler,
                glow::TEXTURE_MAG_FILTER,
                spec.mag_filter.to_gl() as i32,
            );
            gl.sampler_parameter_i32(sampler, glow::TEXTURE_WRAP_S, spec.wrap.to_gl() as i32);
            gl.sampler_parameter_i32(sampler, glow::TEXTURE_WRAP_T, spec.wrap.to_gl() as i32);
            sampler
        })
    }

    /// Draw all contained render passes, in order.
    pub fn draw(&mut self) {
        // unsafety: This will be safe if RenderPass::draw() is
        unsafe {
            for pass in self.passes.iter_mut() {
                pass.draw(&self.gl);
            }
        }
    }

    /// Returns OpenGL version info.
    /// Vendor, renderer, GL version, GLSL version
    pub fn get_info(&self) -> (String, String, String, String) {
        unsafe {
            let vendor = self.gl.get_parameter_string(glow::VENDOR);
            let rend = self.gl.get_parameter_string(glow::RENDERER);
            let vers = self.gl.get_parameter_string(glow::VERSION);
            let glsl_vers = self.gl.get_parameter_string(glow::SHADING_LANGUAGE_VERSION);

            (vendor, rend, vers, glsl_vers)
        }
    }

    /// Sets the viewport for the final render-to-screen pass.
    /// Negative numbers are valid, see `glViewport` for the
    /// math behind it.
    ///
    /// Panics if there is no such render pass.
    pub fn set_screen_viewport(&mut self, x: i32, y: i32, w: i32, h: i32) {
        let pass = self
            .passes
            .last_mut()
            .expect("set_screen_viewport() requires a render pass to function on");
        if let RenderTarget::Screen = pass.target {
            pass.set_viewport(x, y, w, h);
        } else {
            panic!("Last render pass is not rendering to screen, aiee!");
        }
    }
}

/// This is actually not safe to Clone, we'd have to Rc the GlTexture.
/// Having the Rc on the *outside* of this type is what we actually want.
#[derive(Debug)]
pub struct TextureHandle {
    ctx: Rc<glow::Context>,
    tex: GlTexture,
}

impl Drop for TextureHandle {
    fn drop(&mut self) {
        unsafe {
            self.ctx.delete_texture(self.tex);
        }
    }
}

/// A shared, clone-able texture type.
pub type Texture = Rc<TextureHandle>;

impl TextureHandle {
    /// Create a new texture from the given slice of RGBA bytes.
    pub fn new(ctx: &GlContext, rgba: &[u8], width: usize, height: usize) -> Self {
        assert_eq!(width * height * 4, rgba.len());
        let gl = &*ctx.gl;
        // Unsafety: This verifies size of user input, checks resource
        // creation, and does no raw pointer-manipulation-y things outside
        // of that.
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

    /// Make a new empty texture with the given format.  Note that reading from the texture
    /// will give undefined results, hence why this is unsafe.
    pub unsafe fn new_empty(
        ctx: &GlContext,
        format: u32,
        component_format: u32,
        width: usize,
        height: usize,
    ) -> Self {
        let gl = &*ctx.gl;
        let t = gl.create_texture().unwrap();
        gl.active_texture(glow::TEXTURE0);
        gl.bind_texture(glow::TEXTURE_2D, Some(t));
        gl.tex_image_2d(
            glow::TEXTURE_2D,               // Texture target
            0,                              // mipmap level
            i32::try_from(format).unwrap(), // format to store the texture in (can't fail)
            i32::try_from(width).unwrap(),  // width
            i32::try_from(height).unwrap(), // height
            0,                              // border, must always be 0, lulz
            format,                         // format to load the texture from
            component_format,               // Type of each color element
            None,                           // Actual data
        );

        gl.bind_texture(glow::TEXTURE_2D, None);
        Self {
            ctx: ctx.gl.clone(),
            tex: t,
        }
    }

    /// Turn this texture into a share-able, refcounted one.
    pub fn into_shared(self) -> Texture {
        Rc::new(self)
    }
}

/// Similar to `TextureHandle`, this
/// is a shader resource that can be Rc'ed and shared.
#[derive(Debug)]
pub struct ShaderHandle {
    ctx: Rc<glow::Context>,
    program: GlProgram,
}

impl Drop for ShaderHandle {
    fn drop(&mut self) {
        unsafe {
            self.ctx.delete_program(self.program);
        }
    }
}

/// A share-able refcounted shader.
pub type Shader = Rc<ShaderHandle>;

impl ShaderHandle {
    fn new_raw(gl: Rc<glow::Context>, vertex_src: &str, fragment_src: &str) -> ShaderHandle {
        let shader_sources = [
            (glow::VERTEX_SHADER, vertex_src),
            (glow::FRAGMENT_SHADER, fragment_src),
        ];

        // TODO: Audit unsafe
        unsafe {
            let program = gl.create_program().expect("Cannot create program");
            let mut shaders = Vec::with_capacity(shader_sources.len());

            for (shader_type, shader_source) in shader_sources.iter() {
                let shader = gl
                    .create_shader(*shader_type)
                    .expect("Cannot create shader");
                gl.shader_source(shader, shader_source);
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
            ShaderHandle { ctx: gl, program }
        }
    }

    /// Create new shader
    pub fn new(ctx: &GlContext, vertex_src: &str, fragment_src: &str) -> ShaderHandle {
        Self::new_raw(ctx.gl.clone(), vertex_src, fragment_src)
    }

    /// Consume this shader and return a clone-able one
    pub fn into_shared(self) -> Shader {
        Rc::new(self)
    }
}

/// Data we need for each quad instance.
/// DrawParam gets turned into this, eventually.
/// We have to be *quite particular* about layout since this gets
/// fed straight to the shader.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct QuadData {
    /// Color to blend the result texture with.
    pub color: [f32; 4],
    /// Source region on the texture to draw, coordinates range from 0 to 1
    pub src_rect: [f32; 4],
    /// Destination rectangle in your render target to draw the texture on,
    /// coordinates are whatever you set in your transform and viewport.
    pub dst_rect: [f32; 4],
    /// Rotation offset -- A point within your `dst_rect` to rotate around,
    /// coordinates range from 0 to 1
    pub offset: [f32; 2],
    /// Rotation, in radians, CCW.
    pub rotation: f32,
}

unsafe impl bytemuck::Zeroable for QuadData {}

unsafe impl bytemuck::Pod for QuadData {}

impl QuadData {
    /// Returns an empty `QuadData` with default values.
    pub const fn empty() -> Self {
        QuadData {
            offset: [0.0, 0.0],
            color: [1.0, 1.0, 1.0, 1.0],
            src_rect: [0.0, 0.0, 1.0, 1.0],
            dst_rect: [0.0, 0.0, 1.0, 1.0],
            rotation: 0.0,
        }
    }

    /// Returns a Vec of (element offset, element size)
    /// pairs.  This is proooobably technically a little UB,
    /// see https://github.com/rust-lang/rust/issues/48956#issuecomment-544506419
    /// but with repr(C) it's probably safe enough.
    ///
    /// Also returns the name of the shader variable associated with each field...
    unsafe fn layout() -> Vec<(&'static str, usize, usize)> {
        // It'd be nice if we could make this `const` but
        // doing const pointer arithmatic is unstable.
        let thing = QuadData::empty();
        let thing_base = &thing as *const QuadData;
        let offset_offset = (&thing.offset as *const [f32; 2] as usize) - thing_base as usize;
        let offset_size = mem::size_of_val(&thing.offset);

        let color_offset = (&thing.color as *const [f32; 4] as usize) - thing_base as usize;
        let color_size = mem::size_of_val(&thing.color);

        let src_rect_offset = (&thing.src_rect as *const [f32; 4] as usize) - thing_base as usize;
        let src_rect_size = mem::size_of_val(&thing.src_rect);

        let dst_rect_offset = (&thing.dst_rect as *const [f32; 4] as usize) - thing_base as usize;
        let dst_rect_size = mem::size_of_val(&thing.dst_rect);

        let rotation_offset = (&thing.rotation as *const f32 as usize) - thing_base as usize;
        let rotation_size = mem::size_of_val(&thing.rotation);

        vec![
            ("model_offset", offset_offset, offset_size),
            ("model_color", color_offset, color_size),
            ("model_src_rect", src_rect_offset, src_rect_size),
            ("model_dst_rect", dst_rect_offset, dst_rect_size),
            ("model_rotation", rotation_offset, rotation_size),
        ]
    }
}

/// Filter modes a sampler may have.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FilterMode {
    /// Nearest-neighbor filtering.  Use this for pixel-y effects.
    Nearest,
    /// Linear filtering.  Use this for smooth effects.
    Linear,
}

impl FilterMode {
    /// Turns the filter mode into the appropriate OpenGL enum
    fn to_gl(self) -> u32 {
        match self {
            FilterMode::Nearest => glow::NEAREST,
            FilterMode::Linear => glow::LINEAR,
        }
    }
}

/// Wrap modes a sampler may have.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum WrapMode {
    /// Clamp colors to the edge of the texture.
    Clamp,
    /// Tile/repeat the texture.
    Tile,
    /// Mirror the texture.
    Mirror,
}

impl WrapMode {
    /// Turns the wrap mode into the appropriate OpenGL enum
    fn to_gl(self) -> u32 {
        match self {
            WrapMode::Clamp => glow::CLAMP_TO_EDGE,
            WrapMode::Tile => glow::REPEAT,
            WrapMode::Mirror => glow::MIRRORED_REPEAT,
        }
    }
}

/// TODO
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BlendMode {}

/// A description of a sampler.  We cache the actual
/// samplers as needed in the GlContext.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SamplerSpec {
    min_filter: FilterMode,
    mag_filter: FilterMode,
    wrap: WrapMode,
}

impl SamplerSpec {
    /// Shortcut for creating a new `SamplerSpec`.
    pub fn new(min: FilterMode, mag: FilterMode, wrap: WrapMode) -> Self {
        Self {
            min_filter: min,
            mag_filter: mag,
            wrap,
        }
    }
}

impl Default for SamplerSpec {
    fn default() -> Self {
        Self::new(FilterMode::Nearest, FilterMode::Nearest, WrapMode::Tile)
    }
}

/// A list of quads that will be drawn in one draw call.
/// Each uses the same texture, same mesh (built in to the quad shader),
/// and may have different `QuadData` inputs.
#[derive(Debug)]
pub struct QuadDrawCall {
    ctx: Rc<glow::Context>,
    texture: Texture,
    sampler: GlSampler,
    /// The instances that will be drawn.
    pub instances: Vec<QuadData>,
    vbo: GlBuffer,
    vao: GlVertexArray,
    instance_vbo: GlBuffer,
    texture_location: GlUniformLocation,
    /// Whether or not the instances have changed
    /// compared to what the VBO contains, so we can
    /// only upload to the VBO on changes
    pub dirty: bool,
}

impl Drop for QuadDrawCall {
    fn drop(&mut self) {
        unsafe {
            self.ctx.delete_vertex_array(self.vao);
            self.ctx.delete_buffer(self.vbo);
            self.ctx.delete_buffer(self.instance_vbo);
            // Don't need to drop the sampler, it's owned by
            // the `GlContext`.
            // And the texture takes care of itself.
        }
    }
}

impl QuadDrawCall {
    unsafe fn set_vertex_pointers(ctx: &GlContext, shader: &ShaderHandle) {
        let gl = &*ctx.gl;
        let layout = QuadData::layout();
        for (name, offset, size) in layout {
            info!("Layout: {} offset, {} size", offset, size);
            let element_size = mem::size_of::<f32>();
            let attrib_location = gl.get_attrib_location(shader.program, name).unwrap();
            gl.vertex_attrib_pointer_f32(
                attrib_location,
                (size / element_size) as i32,
                glow::FLOAT,
                false,
                mem::size_of::<QuadData>() as i32,
                offset as i32,
            );
            gl.vertex_attrib_divisor(attrib_location, 1);
            gl.enable_vertex_attrib_array(attrib_location);
        }
    }

    /// New empty `QuadDrawCall` using the given pipeline.
    pub fn new(
        ctx: &mut GlContext,
        texture: Texture,
        sampler: SamplerSpec,
        pipeline: &QuadPipeline,
    ) -> Self {
        let sampler = ctx.get_sampler(&sampler);
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

            let dummy_attrib = gl
                .get_attrib_location(pipeline.shader.program, "vertex_dummy")
                .unwrap();
            gl.vertex_attrib_pointer_f32(
                dummy_attrib,
                2,
                glow::FLOAT,
                false,
                // We can just say the stride of this guy is 0, since
                // we never actually use it (yet).  That lets us use a
                // widdle bitty awway for this instead of having an
                // unused empty value for every vertex of every instance.
                0,
                0,
            );
            gl.enable_vertex_attrib_array(dummy_attrib);

            // We DO need a buffer of per-vertex attributes, WebGL gets snippy
            // if we just give it per-instance attributes and say "yeah each
            // vertex just has nothing attached to it".  Which is exactly what
            // we want for quad drawing, alas.
            //
            // But we can make a buffer that just contains one vec2(0,0) for each vertex
            // and give it that, and that seems just fine.
            // And we only need enough vertices to draw one quad and never have to alter it.
            // We could reuse the same buffer for all QuadDrawCall's, tbh, but that seems
            // a bit overkill.
            let empty_slice: &[u8] = &[0; mem::size_of::<f32>() * 2 * 6];
            gl.buffer_data_u8_slice(glow::ARRAY_BUFFER, empty_slice, glow::STREAM_DRAW);

            // Now create another VBO containing per-instance data
            let instance_vbo = gl.create_buffer().unwrap();
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(instance_vbo));
            Self::set_vertex_pointers(ctx, &pipeline.shader);

            // We can't define locations for uniforms, yet.
            let texture_location = gl
                .get_uniform_location(pipeline.shader.program, "tex")
                .unwrap();

            gl.bind_vertex_array(None);

            Self {
                ctx: ctx.gl.clone(),
                vbo,
                vao,
                texture,
                sampler,
                instance_vbo,
                texture_location,
                instances: vec![],
                dirty: true,
            }
        }
    }

    /// Add a new instance to the quad data.
    /// Instances are cached between `draw()` invocations.
    pub fn add(&mut self, quad: QuadData) {
        self.dirty = true;
        self.instances.push(quad);
    }

    /// Empty all instances out of the instance buffer.
    pub fn clear(&mut self) {
        self.dirty = true;
        self.instances.clear();
    }

    /// Upload the array of instances to our VBO
    unsafe fn upload_instances(&mut self, gl: &Context) {
        // TODO: audit unsafe
        let bytes_slice: &[u8] = bytemuck::try_cast_slice(self.instances.as_slice()).unwrap();

        // Fill instance buffer
        gl.bind_buffer(glow::ARRAY_BUFFER, Some(self.instance_vbo));
        gl.buffer_data_u8_slice(glow::ARRAY_BUFFER, bytes_slice, glow::STREAM_DRAW);
        gl.bind_buffer(glow::ARRAY_BUFFER, None);
        self.dirty = false;
    }

    unsafe fn draw(&mut self, gl: &Context) {
        if self.dirty {
            self.upload_instances(gl);
        }
        // Bind VAO
        let num_instances = self.instances.len();
        let num_vertices = 6;
        gl.bind_vertex_array(Some(self.vao));

        // Bind texture
        // TODO: is this active_texture() call necessary?
        // Will be if we ever do multi-texturing, I suppose.
        gl.active_texture(glow::TEXTURE0);
        gl.bind_texture(glow::TEXTURE_2D, Some(self.texture.tex));
        gl.uniform_1_i32(Some(self.texture_location), 0);

        // bind sampler
        // This is FUCKING WHACKO.  I set the active texture
        // unit to glow::TEXTURE0 , which sets it to texture
        // unit 0, then I bind the sampler to 0, which sets it
        // to texture unit 0.  I think.  You have to dig into
        // the ARB extension RFC to figure this out 'cause it isn't
        // documented anywhere else I can find it.
        // Thanks, Khronos.
        gl.bind_sampler(0, Some(self.sampler));
        gl.draw_arrays_instanced(
            glow::TRIANGLES,
            0,
            num_vertices as i32,
            num_instances as i32,
        );
    }
}

/// Trait for a draw call...
pub trait DrawCall {
    /// Add a new instance to the quad data.
    /// Instances are cached between `draw()` invocations.
    fn add(&mut self, quad: QuadData);

    /// Empty all instances out of the instance buffer.
    fn clear(&mut self);
    /// fdjsal
    unsafe fn draw(&mut self, gl: &Context);
}

impl DrawCall for QuadDrawCall {
    /// TODO: Refactor
    fn add(&mut self, quad: QuadData) {
        self.add(quad);
    }

    /// TODO: Refactor
    fn clear(&mut self) {
        self.clear();
    }

    /// TODO: Refactor
    unsafe fn draw(&mut self, gl: &Context) {
        self.draw(gl);
    }
}

/// TODO: Make this actually a mesh
#[derive(Debug)]
pub struct MeshDrawCall {
    ctx: Rc<glow::Context>,
    texture: Texture,
    sampler: GlSampler,
    /// The instances that will be drawn.
    pub instances: Vec<QuadData>,
    vbo: GlBuffer,
    vao: GlVertexArray,
    instance_vbo: GlBuffer,
    texture_location: GlUniformLocation,
    /// Whether or not the instances have changed
    /// compared to what the VBO contains, so we can
    /// only upload to the VBO on changes
    pub dirty: bool,
}
impl Drop for MeshDrawCall {
    fn drop(&mut self) {
        unsafe {
            self.ctx.delete_vertex_array(self.vao);
            self.ctx.delete_buffer(self.vbo);
            self.ctx.delete_buffer(self.instance_vbo);
            // Don't need to drop the sampler, it's owned by
            // the `GlContext`.
            // And the texture takes care of itself.
        }
    }
}

impl MeshDrawCall {
    unsafe fn set_vertex_pointers(ctx: &GlContext, shader: &ShaderHandle) {
        let gl = &*ctx.gl;
        let layout = QuadData::layout();
        for (name, offset, size) in layout {
            info!("Layout: {} offset, {} size", offset, size);
            let element_size = mem::size_of::<f32>();
            let attrib_location = gl.get_attrib_location(shader.program, name).unwrap();
            gl.vertex_attrib_pointer_f32(
                attrib_location,
                (size / element_size) as i32,
                glow::FLOAT,
                false,
                mem::size_of::<QuadData>() as i32,
                offset as i32,
            );
            gl.vertex_attrib_divisor(attrib_location, 1);
            gl.enable_vertex_attrib_array(attrib_location);
        }
    }

    /// New empty `MeshDrawCall` using the given pipeline.
    pub fn new(
        ctx: &mut GlContext,
        texture: Texture,
        sampler: SamplerSpec,
        pipeline: &MeshPipeline,
    ) -> Self {
        let sampler = ctx.get_sampler(&sampler);
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

            let dummy_attrib = gl
                .get_attrib_location(pipeline.shader.program, "vertex_dummy")
                .unwrap();
            gl.vertex_attrib_pointer_f32(
                dummy_attrib,
                2,
                glow::FLOAT,
                false,
                // We can just say the stride of this guy is 0, since
                // we never actually use it (yet).  That lets us use a
                // widdle bitty awway for this instead of having an
                // unused empty value for every vertex of every instance.
                0,
                0,
            );
            gl.enable_vertex_attrib_array(dummy_attrib);

            // We DO need a buffer of per-vertex attributes, WebGL gets snippy
            // if we just give it per-instance attributes and say "yeah each
            // vertex just has nothing attached to it".  Which is exactly what
            // we want for quad drawing, alas.
            //
            // But we can make a buffer that just contains one vec2(0,0) for each vertex
            // and give it that, and that seems just fine.
            // And we only need enough vertices to draw one quad and never have to alter it.
            // We could reuse the same buffer for all MeshDrawCall's, tbh, but that seems
            // a bit overkill.
            let empty_slice: &[u8] = &[0; mem::size_of::<f32>() * 2 * 6];
            gl.buffer_data_u8_slice(glow::ARRAY_BUFFER, empty_slice, glow::STREAM_DRAW);

            // Now create another VBO containing per-instance data
            let instance_vbo = gl.create_buffer().unwrap();
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(instance_vbo));
            Self::set_vertex_pointers(ctx, &pipeline.shader);

            // We can't define locations for uniforms, yet.
            let texture_location = gl
                .get_uniform_location(pipeline.shader.program, "tex")
                .unwrap();

            gl.bind_vertex_array(None);

            Self {
                ctx: ctx.gl.clone(),
                vbo,
                vao,
                texture,
                sampler,
                instance_vbo,
                texture_location,
                instances: vec![],
                dirty: true,
            }
        }
    }

    /// Add a new instance to the quad data.
    /// Instances are cached between `draw()` invocations.
    pub fn add(&mut self, quad: QuadData) {
        self.dirty = true;
        self.instances.push(quad);
    }

    /// Empty all instances out of the instance buffer.
    pub fn clear(&mut self) {
        self.dirty = true;
        self.instances.clear();
    }

    /// Upload the array of instances to our VBO
    unsafe fn upload_instances(&mut self, gl: &Context) {
        // TODO: audit unsafe
        let bytes_slice: &[u8] = bytemuck::try_cast_slice(self.instances.as_slice()).unwrap();

        // Fill instance buffer
        gl.bind_buffer(glow::ARRAY_BUFFER, Some(self.instance_vbo));
        gl.buffer_data_u8_slice(glow::ARRAY_BUFFER, bytes_slice, glow::STREAM_DRAW);
        gl.bind_buffer(glow::ARRAY_BUFFER, None);
        self.dirty = false;
    }

    unsafe fn draw(&mut self, gl: &Context) {
        if self.dirty {
            self.upload_instances(gl);
        }
        // Bind VAO
        let num_instances = self.instances.len();
        let num_vertices = 6;
        gl.bind_vertex_array(Some(self.vao));

        // Bind texture
        // TODO: is this active_texture() call necessary?
        // Will be if we ever do multi-texturing, I suppose.
        gl.active_texture(glow::TEXTURE0);
        gl.bind_texture(glow::TEXTURE_2D, Some(self.texture.tex));
        gl.uniform_1_i32(Some(self.texture_location), 0);

        // bind sampler
        // This is FUCKING WHACKO.  I set the active texture
        // unit to glow::TEXTURE0 , which sets it to texture
        // unit 0, then I bind the sampler to 0, which sets it
        // to texture unit 0.  I think.  You have to dig into
        // the ARB extension RFC to figure this out 'cause it isn't
        // documented anywhere else I can find it.
        // Thanks, Khronos.
        gl.bind_sampler(0, Some(self.sampler));
        gl.draw_arrays_instanced(
            glow::TRIANGLES,
            0,
            num_vertices as i32,
            num_instances as i32,
        );
    }
}

impl DrawCall for MeshDrawCall {
    /// TODO: Refactor
    fn add(&mut self, quad: QuadData) {
        self.add(quad);
    }

    /// TODO: Refactor
    fn clear(&mut self) {
        self.clear();
    }

    /// TODO: Refactor
    unsafe fn draw(&mut self, gl: &Context) {
        self.draw(gl);
    }
}

/// A pipeline for drawing quads.
pub struct QuadPipeline {
    /// The draw calls in the pipeline.
    pub drawcalls: Vec<QuadDrawCall>,
    /// The projection the pipeline will draw with.
    pub projection: Mat4,
    shader: Shader,
    projection_location: GlUniformLocation,
}

impl QuadPipeline {
    /// Create new pipeline with the given shader.
    pub unsafe fn new(ctx: &GlContext, shader: Shader) -> Self {
        let gl = &*ctx.gl;
        //let projection = Mat4::identity();
        let projection = ortho_mat(-1.0, 1.0, 1.0, -1.0, 1.0, -1.0);
        let projection_location = gl
            .get_uniform_location(shader.program, "projection")
            .unwrap();
        Self {
            drawcalls: vec![],
            shader,
            projection,
            projection_location,
        }
    }

    /// Draw all the draw calls in the pipeline.
    pub unsafe fn draw(&mut self, gl: &Context) {
        gl.use_program(Some(self.shader.program));
        gl.uniform_matrix_4_f32_slice(
            Some(self.projection_location),
            false,
            &self.projection.to_cols_array(),
        );
        for dc in self.drawcalls.iter_mut() {
            dc.draw(gl);
        }
    }
}

/// TODO: Docs
/// hnyrn
pub trait Pipeline {
    /// foo
    unsafe fn draw(&mut self, gl: &Context);
    /// foo
    fn new_drawcall(
        &mut self,
        ctx: &mut GlContext,
        texture: Texture,
        sampler: SamplerSpec,
    ) -> &mut dyn DrawCall;
    /// this seems the way to do it...
    fn get(&self, idx: usize) -> &dyn DrawCall;
    /// Get mut
    fn get_mut(&mut self, idx: usize) -> &mut dyn DrawCall;
    /// clear all draw calls
    fn clear(&mut self);
    ///  Returns iterator of drawcalls.  The lifetimes are a PITA.
    fn drawcalls<'a>(&'a self) -> Box<dyn Iterator<Item = &'a dyn DrawCall> + 'a>;
    ///  Returns iterator of mutable drawcalls.  The lifetimes are a PITA.
    fn drawcalls_mut<'a>(&'a mut self) -> Box<dyn Iterator<Item = &'a mut dyn DrawCall> + 'a>;
}

/// aaaaa
/// TODO: Docs
pub struct PipelineIter<'a> {
    i: std::slice::Iter<'a, QuadDrawCall>,
}

impl<'a> PipelineIter<'a> {
    /// TODO: Docs
    pub fn new(p: &'a QuadPipeline) -> Self {
        Self {
            i: p.drawcalls.iter(),
        }
    }
}

impl<'a> Iterator for PipelineIter<'a> {
    type Item = &'a dyn DrawCall;
    fn next(&mut self) -> Option<Self::Item> {
        self.i.next().map(|x| x as _)
    }
}

/// Sigh
/// TODO: Docs
pub struct PipelineIterMut<'a> {
    i: std::slice::IterMut<'a, QuadDrawCall>,
}

impl<'a> PipelineIterMut<'a> {
    /// TODO: Docs
    pub fn new(p: &'a mut QuadPipeline) -> Self {
        Self {
            i: p.drawcalls.iter_mut(),
        }
    }
}

impl<'a> Iterator for PipelineIterMut<'a> {
    type Item = &'a mut dyn DrawCall;
    fn next(&mut self) -> Option<Self::Item> {
        self.i.next().map(|x| x as _)
    }
}

impl Pipeline for QuadPipeline {
    /// foo
    /// TODO: Docs
    unsafe fn draw(&mut self, gl: &Context) {
        self.draw(gl);
    }
    /// foo
    fn new_drawcall(
        &mut self,
        ctx: &mut GlContext,
        texture: Texture,
        sampler: SamplerSpec,
    ) -> &mut dyn DrawCall {
        let x = QuadDrawCall::new(ctx, texture, sampler, self);
        self.drawcalls.push(x);
        &mut *self.drawcalls.last_mut().unwrap()
    }

    fn clear(&mut self) {
        self.drawcalls.clear()
    }
    fn get(&self, idx: usize) -> &dyn DrawCall {
        &self.drawcalls[idx]
    }
    fn get_mut(&mut self, idx: usize) -> &mut dyn DrawCall {
        &mut self.drawcalls[idx]
    }

    fn drawcalls<'a>(&'a self) -> Box<dyn Iterator<Item = &'a dyn DrawCall> + 'a> {
        let i = PipelineIter::new(self);
        Box::new(i)
    }

    fn drawcalls_mut<'a>(&'a mut self) -> Box<dyn Iterator<Item = &'a mut dyn DrawCall> + 'a> {
        let i = PipelineIterMut::new(self);
        Box::new(i)
    }
}

/// A pipeline for drawing quads.
pub struct MeshPipeline {
    /// The draw calls in the pipeline.
    pub drawcalls: Vec<MeshDrawCall>,
    /// The projection the pipeline will draw with.
    pub projection: Mat4,
    shader: Shader,
    projection_location: GlUniformLocation,
}

impl MeshPipeline {
    /// Create new pipeline with the given shader.
    pub unsafe fn new(ctx: &GlContext, shader: Shader) -> Self {
        let gl = &*ctx.gl;
        //let projection = Mat4::identity();
        let projection = ortho_mat(-1.0, 1.0, 1.0, -1.0, 1.0, -1.0);
        let projection_location = gl
            .get_uniform_location(shader.program, "projection")
            .unwrap();
        Self {
            drawcalls: vec![],
            shader,
            projection,
            projection_location,
        }
    }

    /// Draw all the draw calls in the pipeline.
    pub unsafe fn draw(&mut self, gl: &Context) {
        gl.use_program(Some(self.shader.program));
        gl.uniform_matrix_4_f32_slice(
            Some(self.projection_location),
            false,
            &self.projection.to_cols_array(),
        );
        for dc in self.drawcalls.iter_mut() {
            dc.draw(gl);
        }
    }
}

/// aaaaa
/// TODO: Docs
pub struct MeshPipelineIter<'a> {
    i: std::slice::Iter<'a, MeshDrawCall>,
}

impl<'a> MeshPipelineIter<'a> {
    /// TODO: Docs
    pub fn new(p: &'a MeshPipeline) -> Self {
        Self {
            i: p.drawcalls.iter(),
        }
    }
}

impl<'a> Iterator for MeshPipelineIter<'a> {
    type Item = &'a dyn DrawCall;
    fn next(&mut self) -> Option<Self::Item> {
        self.i.next().map(|x| x as _)
    }
}

/// Sigh
/// TODO: Docs
pub struct MeshPipelineIterMut<'a> {
    i: std::slice::IterMut<'a, MeshDrawCall>,
}

impl<'a> MeshPipelineIterMut<'a> {
    /// TODO: Docs
    pub fn new(p: &'a mut MeshPipeline) -> Self {
        Self {
            i: p.drawcalls.iter_mut(),
        }
    }
}

impl<'a> Iterator for MeshPipelineIterMut<'a> {
    type Item = &'a mut dyn DrawCall;
    fn next(&mut self) -> Option<Self::Item> {
        self.i.next().map(|x| x as _)
    }
}

impl Pipeline for MeshPipeline {
    /// foo
    /// TODO: Docs
    unsafe fn draw(&mut self, gl: &Context) {
        self.draw(gl);
    }
    /// foo
    fn new_drawcall(
        &mut self,
        ctx: &mut GlContext,
        texture: Texture,
        sampler: SamplerSpec,
    ) -> &mut dyn DrawCall {
        let x = MeshDrawCall::new(ctx, texture, sampler, self);
        self.drawcalls.push(x);
        &mut *self.drawcalls.last_mut().unwrap()
    }

    fn clear(&mut self) {
        self.drawcalls.clear()
    }
    fn get(&self, idx: usize) -> &dyn DrawCall {
        &self.drawcalls[idx]
    }
    fn get_mut(&mut self, idx: usize) -> &mut dyn DrawCall {
        &mut self.drawcalls[idx]
    }

    fn drawcalls<'a>(&'a self) -> Box<dyn Iterator<Item = &'a dyn DrawCall> + 'a> {
        let i = MeshPipelineIter::new(self);
        Box::new(i)
    }

    fn drawcalls_mut<'a>(&'a mut self) -> Box<dyn Iterator<Item = &'a mut dyn DrawCall> + 'a> {
        let i = MeshPipelineIterMut::new(self);
        Box::new(i)
    }
}

/// A render target for drawing to a texture.
#[derive(Debug)]
pub struct TextureRenderTarget {
    ctx: Rc<glow::Context>,
    output_framebuffer: GlFramebuffer,
    output_texture: Texture,
    _output_depthbuffer: GlRenderbuffer,
}

impl Drop for TextureRenderTarget {
    fn drop(&mut self) {
        unsafe {
            self.ctx.delete_framebuffer(self.output_framebuffer);
            self.ctx.delete_renderbuffer(self._output_depthbuffer);
        }
    }
}

impl TextureRenderTarget {
    /// Create a new render target rendering to a texture.
    pub unsafe fn new(ctx: &GlContext, width: usize, height: usize) -> Self {
        let gl = &*ctx.gl;

        let t = TextureHandle::new_empty(ctx, glow::RGBA, glow::UNSIGNED_BYTE, width, height)
            .into_shared();
        let depth = gl.create_renderbuffer().unwrap();
        let fb = gl.create_framebuffer().unwrap();

        // Now we have our color texture, depth buffer and framebuffer, and we
        // glue them all together.
        gl.bind_texture(glow::TEXTURE_2D, Some(t.tex));
        // We need to add filtering params to the texture, for Reasons.
        // We might be able to use samplers instead, but not yet.
        gl.tex_parameter_i32(
            glow::TEXTURE_2D,
            glow::TEXTURE_MAG_FILTER,
            glow::NEAREST as i32,
        );
        gl.tex_parameter_i32(
            glow::TEXTURE_2D,
            glow::TEXTURE_MIN_FILTER,
            glow::NEAREST as i32,
        );

        /*
         * TODO: This panics for some reason
        gl.bind_renderbuffer(glow::RENDERBUFFER, Some(depth));
        gl.renderbuffer_storage(
            glow::RENDERBUFFER,
            glow::DEPTH_COMPONENT,
            width as i32,
            height as i32,
        );
        */

        gl.bind_framebuffer(glow::FRAMEBUFFER, Some(fb));
        gl.framebuffer_texture_2d(
            glow::FRAMEBUFFER,
            glow::COLOR_ATTACHMENT0,
            glow::TEXTURE_2D,
            Some(t.tex),
            0,
        );
        /*
         * TODO: This panics for some reason
        gl.framebuffer_renderbuffer(
            glow::FRAMEBUFFER,
            glow::DEPTH_ATTACHMENT,
            glow::RENDERBUFFER,
            Some(depth),
        );
        */

        // Set list of draw buffers
        let draw_buffers = &[glow::COLOR_ATTACHMENT0];
        gl.draw_buffers(draw_buffers);

        // Verify results
        if gl.check_framebuffer_status(glow::FRAMEBUFFER) != glow::FRAMEBUFFER_COMPLETE {
            panic!("Framebuffer hecked up");
        }

        // Reset heckin bindings
        gl.bind_framebuffer(glow::FRAMEBUFFER, None);
        gl.bind_texture(glow::TEXTURE_2D, None);
        Self {
            ctx: ctx.gl.clone(),
            output_framebuffer: fb,
            output_texture: t,
            _output_depthbuffer: depth,
        }
    }
}

/// The options for what a render pass can write to.
#[derive(Debug)]
pub enum RenderTarget {
    /// A render target rendering to a texture.
    Texture(TextureRenderTarget),
    /// A render target rendering to the screen's buffer.
    Screen,
}

impl RenderTarget {
    /// Return the render target corresponding to the output display.
    pub fn screen_target() -> Self {
        Self::Screen
    }

    /// Create a new render target rendering to a texture
    pub fn new_target(ctx: &GlContext, width: usize, height: usize) -> Self {
        unsafe {
            let target = TextureRenderTarget::new(ctx, width, height);
            Self::Texture(target)
        }
    }

    /// Bind this render target so it will be drawn to.
    unsafe fn bind(&self, gl: &glow::Context) {
        let fb = match self {
            Self::Screen => None,
            Self::Texture(trt) => Some(trt.output_framebuffer),
        };
        gl.bind_framebuffer(glow::FRAMEBUFFER, fb);
    }
}

/// Currently, no input framebuffers or such.
/// We're not actually intending to reproduce Rendy's Graph type here.
/// This may eventually feed into a bounce buffer or such though.
pub struct RenderPass {
    target: RenderTarget,
    clear_color: (f32, f32, f32, f32),
    viewport: (i32, i32, i32, i32),
    /// The pipelines to draw in the render pass.
    pub pipelines: Vec<Box<dyn Pipeline>>,
}

impl RenderPass {
    /// Make a new render pass rendering to a texture.
    pub unsafe fn new(
        ctx: &mut GlContext,
        width: usize,
        height: usize,
        clear_color: (f32, f32, f32, f32),
    ) -> Self {
        let target = RenderTarget::new_target(ctx, width, height);

        Self {
            target,
            pipelines: vec![],
            viewport: (0, 0, width as i32, height as i32),
            clear_color,
        }
    }

    /// Create a new rnder pass rendering to the screen.
    pub unsafe fn new_screen(
        _ctx: &mut GlContext,
        width: usize,
        height: usize,
        clear_color: (f32, f32, f32, f32),
    ) -> Self {
        Self {
            target: RenderTarget::Screen,
            pipelines: vec![],
            viewport: (0, 0, width as i32, height as i32),
            clear_color,
        }
    }

    /// Add a new pipeline to the renderpass
    pub fn add_pipeline(&mut self, pipeline: impl Pipeline + 'static) {
        self.pipelines.push(Box::new(pipeline))
    }

    unsafe fn draw(&mut self, gl: &Context) {
        self.target.bind(gl);
        let (r, g, b, a) = self.clear_color;
        let (x, y, w, h) = self.viewport;
        // TODO: Does this need to be set every time, or does it stick to the target binding?
        gl.viewport(x, y, w, h);
        gl.clear_color(r, g, b, a);
        gl.clear(glow::COLOR_BUFFER_BIT | glow::DEPTH_BUFFER_BIT);
        for pipeline in self.pipelines.iter_mut() {
            pipeline.draw(gl);
        }
    }

    /// Get the texture this render pass outputs to, if any.
    pub fn get_texture(&self) -> Option<Texture> {
        match &self.target {
            RenderTarget::Screen => None,
            RenderTarget::Texture(trt) => Some(trt.output_texture.clone()),
        }
    }

    /// Sets the viewport for the render pass.  Negative numbers are valid,
    /// see `glViewport` for the math involved.
    pub fn set_viewport(&mut self, x: i32, y: i32, w: i32, h: i32) {
        self.viewport = (x, y, w, h);
    }
}
