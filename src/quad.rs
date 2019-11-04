/*
Okay... one way or another I have a list of per-rect properties, like position. All rects in that list are drawn with the same texture and geometry, however that geometry is specified. Then I have another list of (texture, rect list) pairs. And that gets me batched drawing in a form that's easy to make automatic.

THEN I have one or more frames in flight, and each frame in flight has its own copies of the things necessary to actually draw that frame: uniforms/push constants, descriptor set, and the GPU buffers containing the list of (texture, rect list) pairs

To actually draw, you grab the appropriate free frame-in-flight, copy your (texture, rect list) pairs into its buffers, and for each pair issue one draw call to draw that chunk of the buffer.
*/

use std::io;
use std::mem;
use std::sync::Arc;

use rendy::command::{QueueId, RenderPassEncoder};
use rendy::factory::{Factory, ImageState};
use rendy::graph::{render::*, GraphContext, NodeBuffer, NodeImage};
use rendy::hal;
use rendy::hal::device::Device;
use rendy::memory::Dynamic;
use rendy::mesh::AsVertex;
use rendy::resource::{
    Buffer, BufferInfo, DescriptorSet, DescriptorSetLayout, Escape, Filter, Handle, SamplerDesc,
    WrapMode,
};
use rendy::texture::Texture;

use rendy::init::winit::event::{Event, WindowEvent};
use rendy::init::winit::event_loop::{ControlFlow, EventLoop};
use rendy::init::winit::window::WindowBuilder;

use euclid;
use log::*;
use oorandom;

pub type Point2 = euclid::Point2D<f32, euclid::UnknownUnit>;
pub type Transform3 = euclid::Transform3D<f32, euclid::UnknownUnit, euclid::UnknownUnit>;
pub type Rect = euclid::Rect<f32, euclid::UnknownUnit>;

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

use rendy::mesh::{Attribute, VertexFormat};

/// Okay, this tripped me up.  Instance data is technically
/// part of the per-vertex data.  So we describe it as
/// part of the vertex format.  This is where that
/// definition happens.
///
/// This trait impl is basically extended from the impl for
/// `rendy::mesh::Transform`
impl AsVertex for QuadData {
    fn vertex() -> VertexFormat {
        VertexFormat {
            attributes: vec![
                Attribute::new(
                    "Transform1",
                    0,
                    hal::pso::Element {
                        format: hal::format::Format::Rgba32Sfloat,
                        offset: 0,
                    },
                ),
                Attribute::new(
                    "Transform2",
                    1,
                    hal::pso::Element {
                        format: hal::format::Format::Rgba32Sfloat,
                        offset: 16,
                    },
                ),
                Attribute::new(
                    "Transform3",
                    0,
                    hal::pso::Element {
                        format: hal::format::Format::Rgba32Sfloat,
                        offset: 32,
                    },
                ),
                Attribute::new(
                    "Transform4",
                    0,
                    hal::pso::Element {
                        format: hal::format::Format::Rgba32Sfloat,
                        offset: 48,
                    },
                ),
                // rect
                Attribute::new(
                    "rect",
                    0,
                    hal::pso::Element {
                        format: hal::format::Format::Rgba32Sfloat,
                        offset: 64,
                    },
                ),
                // color
                Attribute::new(
                    "color",
                    0,
                    hal::pso::Element {
                        format: hal::format::Format::Rgba32Sfloat,
                        offset: 80,
                    },
                ),
            ],
            stride: 96,
        }
    }
}

/// Uniform data.  Each frame contains one of these.
#[derive(Clone, Copy, Debug, Default)]
#[repr(C, align(16))]
pub struct UniformData {
    pub proj: Transform3,
    pub view: Transform3,
}

/// An instance buffer that bounds checks how many instances you can
/// put into it.  Is not generic on the instance data type, just holds
/// `QuadData`.
/// TODO: Make it resizeable someday.
///
/// The buffer in a `FrameInFlight`.
///
/// We pack data from multiple `QuadDrawCall`'s together into one `Buffer`
/// but each draw call can have a varying amount of instance data,
/// so we end up with something like:
///
/// ```
/// | Instances1 ... | Instances2 ... | ... | empty space |
/// ```
///
/// Right now we have a fixed size buffer and just limit the number
/// of objects in it.  TODO: Eventually someday we will grow the buffer
/// as needed.  Maybe shrink it too?  Not sure about that.
#[derive(Debug)]
pub struct InstanceBuffer<B>
where
    B: hal::Backend,
{
    /// Capacity, in *number of instances*.
    pub capacity: u64,
    /// Number of instances currently in the buffer
    pub length: u64,
    /// Actual buffer object.
    pub buffer: Escape<Buffer<B>>,
}

impl<B> InstanceBuffer<B>
where
    B: hal::Backend,
{
    /// Create a new empty instance buffer with the given
    /// capacity in *number of instances*
    pub fn new(capacity: u64, factory: &Factory<B>) -> Self {
        let bytes_per_instance = Self::instance_size();
        let buffer_size = capacity * bytes_per_instance;
        let buffer = factory
            .create_buffer(
                BufferInfo {
                    size: buffer_size,
                    // TODO: We probably don't need usage::Uniform here anymore.  Confirm!
                    usage: hal::buffer::Usage::UNIFORM | hal::buffer::Usage::VERTEX,
                },
                Dynamic,
            )
            .unwrap();
        Self {
            capacity,
            length: 0,
            buffer,
        }
    }

    /// Resizes the underlying buffer.  Does NOT copy the contents
    /// of the old buffer (yet), new buffer is empty.
    pub fn resize(&mut self, factory: &Factory<B>, new_capacity: u64) {
        let buffer = factory
            .create_buffer(
                BufferInfo {
                    size: new_capacity,
                    // TODO: We probably don't need usage::Uniform here anymore.  Confirm!
                    usage: hal::buffer::Usage::UNIFORM | hal::buffer::Usage::VERTEX,
                },
                Dynamic,
            )
            .unwrap();
        let old_buffer = mem::replace(&mut self.buffer, buffer);
        unsafe {
            factory.destroy_relevant_buffer(Escape::unescape(old_buffer));
        }
        self.length = 0;
        self.capacity = new_capacity;
    }

    /// Returns the size in bytes of a single instance for this type.
    /// For now, this doesn't change, but it's convenient to have.
    ///
    /// This can't be a const fn yet, 'cause trait bounds.
    /// See https://github.com/rust-lang/rust/issues/57563
    pub fn instance_size() -> u64 {
        mem::size_of::<QuadData>() as u64
    }

    /// Returns the buffer size in bytes, rounded up
    /// to the given alignment.
    /// TODO: Are the alignment requirements for this necessary?
    pub fn buffer_size(&self, align: u64) -> u64 {
        (((Self::instance_size() * self.capacity) - 1) / align + 1) * align
    }

    /// Returns an offset in bytes, pointing to free space right after
    /// the given number of instances, or None if `idx >= self.capacity`
    pub fn instance_offset(&self, idx: u64) -> Option<u64> {
        if idx >= self.capacity {
            None
        } else {
            Some(idx * Self::instance_size())
        }
    }

    /// Empties the buffer by setting the length to 0.
    /// Capacity remains unchanged.
    pub fn clear(&mut self) {
        self.length = 0;
    }

    /// Copies the instance data in the given slice into the buffer,
    /// starting from the current end of it.
    /// Returns the offset at which it started if ok, or if the buffer
    /// is not large enough returns Err.
    ///
    /// TODO: Better error types.  Do bounds checks with assert_dbg!()?
    pub fn add_slice(&mut self, factory: &Factory<B>, instances: &[QuadData]) -> Result<u64, ()> {
        if self.length + (instances.len() as u64) >= self.capacity {
            return Err(());
        }
        let offset = self.instance_offset(self.length).ok_or(())?;
        // Vulkan doesn't seem to like zero-size copies very much.
        if instances.len() > 0 {
            unsafe {
                factory
                    .upload_visible_buffer(&mut self.buffer, offset, instances)
                    .unwrap();
            }
            self.length += instances.len() as u64;
        }
        Ok(self.length)
    }

    pub fn inner(&self) -> &Buffer<B> {
        &self.buffer
    }

    pub fn inner_mut(&mut self) -> &mut Buffer<B> {
        &mut self.buffer
    }

    pub fn dispose(self, factory: &Factory<B>) {
        info!("Destroying instance buffer");
        unsafe {
            factory.destroy_relevant_buffer(Escape::unescape(self.buffer));
        }
    }
}

/// The type used for push constants sent to the shaders.
/// This is its own type 'cause there's a couple places we need
/// to use it and it's nice to keep in sync.
type PushConstantsBuffer = [u32; 32];

/// What data we need for each frame in flight.
/// Rendy doesn't do any synchronization for us
/// beyond guarenteeing that when we get a new frame
/// index everything touched by that frame index is
/// now free to reuse.
///
/// We could make different frames share buffers
/// and descriptor sets and such and only change bits
/// that aren't in use from other frames, but that's
/// more complex than I want to get into right now.
/// So basically each `FrameInFlight` contains a whole
/// copy of what writeable data we need to render a frame,
/// and that data gets re-filled each frame.
///
/// When we do want to do that though, I think the simple
/// way would be... maybe create a structure through which
/// a FrameInFlight can be altered and which records if
/// things have actually changed.  Actually, the QuadDrawCall
/// might the place to handle that?  Hm, having a separate
/// Buffer per draw call might be the way to go too?  If
/// the buffer does not change from one draw call to the
/// next, we don't need to re-record its data, just issue
/// the draw call directly with the right PrepareReuse...
/// idk, I'm rambling.
#[derive(Debug)]
struct FrameInFlight<B>
where
    B: hal::Backend,
{
    /// The buffer where we store instance data.
    buffer: InstanceBuffer<B>,
    /// The frame's local copy of uniform data.
    push_constants: PushConstantsBuffer,
}

impl<B> FrameInFlight<B>
where
    B: hal::Backend,
{
    /// All our descriptor sets use the same layout.
    /// This one!  We have an instance buffer, an
    /// image, and a sampler.
    const LAYOUT: &'static [hal::pso::DescriptorSetLayoutBinding] = &[
        // TODO: Can we get rid of this uniform buffer since we use push constants?
        // Doesn't look like it, 'cause we use the buffer for our instance data too.
        hal::pso::DescriptorSetLayoutBinding {
            binding: 0,
            ty: hal::pso::DescriptorType::UniformBuffer,
            count: 1,
            stage_flags: hal::pso::ShaderStageFlags::GRAPHICS,
            immutable_samplers: false,
        },
        hal::pso::DescriptorSetLayoutBinding {
            binding: 1,
            ty: hal::pso::DescriptorType::SampledImage,
            count: 1,
            stage_flags: hal::pso::ShaderStageFlags::FRAGMENT,
            immutable_samplers: false,
        },
        hal::pso::DescriptorSetLayoutBinding {
            binding: 2,
            ty: hal::pso::DescriptorType::Sampler,
            count: 1,
            stage_flags: hal::pso::ShaderStageFlags::FRAGMENT,
            immutable_samplers: false,
        },
    ];
    fn get_descriptor_set_layout() -> SetLayout {
        SetLayout {
            bindings: Self::LAYOUT.to_vec(),
        }
    }

    fn new(factory: &mut Factory<B>) -> Self {
        use std::convert::TryInto;
        // TODO: Figure out max length.
        let buffer_count = MAX_OBJECTS; // * draw_calls.len();
        let buffer = InstanceBuffer::new(buffer_count.try_into().unwrap(), factory);
        let ret = Self {
            buffer,
            push_constants: [0; 32],
        };
        ret
    }

    /// This happens before a frame; it should take a LIST of draw calls and take
    /// care of uploading EACH of them into the buffer so they don't clash!
    fn prepare(
        &mut self,
        factory: &Factory<B>,
        uniforms: &UniformData,
        draw_calls: &[QuadDrawCall<B>],
    ) {
        //assert!(draw_calls.len() > 0);
        // Store the uniforms to be shoved into push constants this frame
        // TODO: Be less crude about indexing and such.
        for (i, vl) in uniforms
            .proj
            .to_column_major_array()
            .into_iter()
            .enumerate()
        {
            self.push_constants[i] = vl.to_bits();
        }
        for (i, vl) in uniforms
            .view
            .to_column_major_array()
            .into_iter()
            .enumerate()
        {
            self.push_constants[16 + i] = vl.to_bits();
        }

        //println!("Preparing frame-in-flight, {} draw calls, first has {} instances.", draw_calls.len(),
        //draw_calls[0].objects.len());
        self.buffer.clear();
        for draw_call in draw_calls {
            let _offset = self
                .buffer
                .add_slice(factory, &draw_call.objects[..])
                .unwrap();
        }
    }

    /// Draws a list of QuadDrawCall's.
    fn draw(
        &mut self,
        draw_calls: &[QuadDrawCall<B>],
        layout: &B::PipelineLayout,
        encoder: &mut RenderPassEncoder<'_, B>,
    ) {
        //println!("Drawing {} draw calls", draw_calls.len());
        let mut instance_count: u64 = 0;
        for draw_call in draw_calls {
            // Skip empty draw calls, since buffers of length 0 are invalid
            if draw_call.objects.is_empty() {
                continue;
            }
            //println!("Drawing {:#?}, {:#?}, {}", draw_call, descriptor_set, draw_offset);

            // This is a bit weird, but basically tells the thing where to find the
            // instance data.  The stride and such of the instance structure is
            // defined in the `AsVertex` definition.
            //
            // TODO: Per Ralith, "don't rebind the descriptor set for every batch, it sticks
            // around".  So we might be able to do it in prepare() instead?  Not sure.  The
            // Rendy examples do it here.
            //
            // Also descriptor set binding numbers and vertex buffer binding numbers
            // are separate.
            unsafe {
                encoder.bind_graphics_descriptor_sets(
                    layout,
                    0,
                    std::iter::once(draw_call.descriptor_set.raw()),
                    std::iter::empty(),
                );
            }

            // The 0 here is to say which vertex buffer slot we're binding to, more or less...
            // if we were drawing a Mesh as well we would, say, bind that to 0 and
            // bind the instance buffer to 1.
            //
            // But, we are doing mesh-less drawing 'cause it's just quads, and
            // just creating the mesh-equivalent in the shader,
            unsafe {
                encoder.bind_vertex_buffers(
                    0,
                    std::iter::once((
                        self.buffer.inner().raw(),
                        self.buffer.instance_offset(instance_count).unwrap(),
                    )),
                );
            }
            unsafe {
                encoder.push_constants(
                    layout,
                    hal::pso::ShaderStageFlags::ALL,
                    0,
                    &self.push_constants,
                );
            }
            // This count is the *number of instances*.  What instance
            // to start at in the buffer is defined by the offset in
            // `bind_vertex_buffers()` above, and the stride/size of an instance
            // is defined in `AsVertex`.
            let instances = 0..(draw_call.objects.len() as u32);
            unsafe {
                encoder.draw(0..6, instances);
            }
            instance_count += draw_call.objects.len() as u64;
        }
    }

    pub fn dispose(self, factory: &Factory<B>) {
        info!("FrameInFlight disposed");
        self.buffer.dispose(factory)
    }
}

/// The data we need for a single draw call, which
/// gets bound to descriptor sets and
///
/// For now we re-bind EVERYTHING even if only certain
/// resources have changed (for example, texture changes
/// and mesh stays the same).  Should be easy to check
/// that in the future and make the various things
/// in here `Option`s.
#[derive(Debug)]
pub struct QuadDrawCall<B>
where
    B: hal::Backend,
{
    objects: Vec<QuadData>,
    texture: Arc<Texture<B>>,
    descriptor_set: Escape<DescriptorSet<B>>,
    /// We just need the actual config for the sampler 'cause
    /// Rendy's `Factory` can manage a sampler cache itself.
    sampler_info: SamplerDesc,
}

impl<B> QuadDrawCall<B>
where
    B: hal::Backend,
{
    pub fn new(
        texture: Arc<Texture<B>>,
        factory: &Factory<B>,
        layout: &Handle<DescriptorSetLayout<B>>,
    ) -> Self {
        let sampler_info = SamplerDesc::new(Filter::Nearest, WrapMode::Clamp);
        let descriptor_set =
            QuadDrawCall::create_descriptor_set(&*texture, &sampler_info, factory, layout);
        Self {
            objects: vec![],
            texture,
            sampler_info,
            descriptor_set,
        }
    }
    pub fn add_quad(&mut self, instance: QuadData) {
        if self.objects.len() < MAX_OBJECTS {
            self.objects.push(instance);
        }
    }

    pub fn add_random_object(
        &mut self,
        rng: &mut oorandom::Rand32,
        max_width: f32,
        max_height: f32,
    ) {
        let x = rng.rand_float() * max_width;
        let y = rng.rand_float() * max_height;
        let offset = euclid::Vector3D::new(x, y, 0.0);
        let transform = Transform3::create_scale(100.0, 100.0, 1.0).post_translate(offset);
        let src = Rect::from(euclid::Size2D::new(100.0, 100.0));
        let color = [1.0, 0.0, 1.0, 1.0];
        let instance = QuadData {
            transform: transform.to_column_major_array(),
            rect: [src.origin.x, src.origin.y, src.size.width, src.size.height],
            color,
        };
        self.add_quad(instance);
    }

    fn create_descriptor_set(
        texture: &Texture<B>,
        sampler_info: &SamplerDesc,
        factory: &Factory<B>,
        layout: &Handle<DescriptorSetLayout<B>>,
    ) -> Escape<DescriptorSet<B>> {
        // Does this sampler need to stay alive?  We pass a
        // reference to it elsewhere in an `unsafe` block...
        // It's cached in the Factory anyway, so I don't think so.
        let sampler = factory
            .get_sampler(sampler_info.clone())
            .expect("Could not get sampler");

        unsafe {
            let set = factory.create_descriptor_set(layout.clone()).unwrap();
            factory
                .device()
                .write_descriptor_sets(Some(hal::pso::DescriptorSetWrite {
                    set: set.raw(),
                    binding: 1,
                    array_offset: 0,
                    descriptors: vec![hal::pso::Descriptor::Image(
                        texture.view().raw(),
                        hal::image::Layout::ShaderReadOnlyOptimal,
                    )],
                }));
            factory.write_descriptor_sets(Some(hal::pso::DescriptorSetWrite {
                set: set.raw(),
                binding: 2,
                array_offset: 0,
                descriptors: vec![hal::pso::Descriptor::Sampler(sampler.raw())],
            }));
            set
        }
    }
}

#[derive(Debug)]
pub struct Aux<B: hal::Backend> {
    /// Number of FrameInFlight's.
    /// This is a little bass-ackwards but there's currently no better place to put it;
    /// we get this from the PresentNode but have to build that after building
    /// the QuadRenderGroup, and the QuadRenderGroupDesc is where this is mainly used.
    pub frames: usize,

    pub draws: Vec<QuadDrawCall<B>>,
    pub camera: UniformData,

    pub shader: rendy::shader::ShaderSetBuilder,
    layout: Option<Handle<DescriptorSetLayout<B>>>,
}

impl<B> Aux<B>
where
    B: hal::Backend,
{
    /// for SOME reason, doing this in a Drop impl doesn't work right, it either
    /// doesn't get called at all or gets called at the wrong time.  Even when we
    /// call it by hand.  So I guess I'm just doing this instead.
    fn dispose(&mut self) {
        /*
        for draw in self.draws.drain(..) {
            drop(draw);
        }
        */
        self.draws.clear();
        info!("Dropped draw calls");
        self.layout = None;
        info!("Dropped layout");
    }
}

const MAX_OBJECTS: usize = 10_000;

/// Render group describing a graph node that renders quads.
#[derive(Debug)]
pub struct QuadRenderGroup<B>
where
    B: hal::Backend,
{
    pipeline_layout: B::PipelineLayout,
    graphics_pipeline: B::GraphicsPipeline,
    frames_in_flight: Vec<FrameInFlight<B>>,
}

/// Descriptor for simple render group.
#[derive(Debug)]
pub struct QuadRenderGroupDesc {
    colors: Vec<hal::pso::ColorBlendDesc>,
}

impl QuadRenderGroupDesc {
    pub fn new() -> Self {
        Self {
            colors: vec![hal::pso::ColorBlendDesc {
                mask: hal::pso::ColorMask::ALL,
                blend: Some(hal::pso::BlendState::ALPHA),
            }],
        }
    }
}

impl<B> RenderGroupDesc<B, Aux<B>> for QuadRenderGroupDesc
where
    B: hal::Backend,
{
    fn buffers(&self) -> Vec<rendy::graph::BufferAccess> {
        vec![]
    }

    fn images(&self) -> Vec<rendy::graph::ImageAccess> {
        vec![]
    }

    fn colors(&self) -> usize {
        self.colors.len()
    }

    fn depth(&self) -> bool {
        true
    }

    fn build<'a>(
        self,
        _ctx: &GraphContext<B>,
        factory: &mut Factory<B>,
        _queue: QueueId,
        aux: &Aux<B>,
        framebuffer_width: u32,
        framebuffer_height: u32,
        subpass: hal::pass::Subpass<'_, B>,
        _buffers: Vec<NodeBuffer>,
        _images: Vec<NodeImage>,
    ) -> Result<Box<dyn RenderGroup<B, Aux<B>>>, hal::pso::CreationError> {
        let depth_stencil = hal::pso::DepthStencilDesc {
            depth: Some(hal::pso::DepthTest {
                fun: hal::pso::Comparison::LessEqual,
                write: true,
            }),
            depth_bounds: false,
            stencil: None,
        };
        let input_assembler_desc = hal::pso::InputAssemblerDesc {
            primitive: hal::pso::Primitive::TriangleList,
            with_adjacency: false,
            restart_index: None,
        };

        let layout_push_constants = vec![(
            hal::pso::ShaderStageFlags::ALL,
            // This size is in number of u32's
            0..((mem::size_of::<PushConstantsBuffer>() / mem::size_of::<u32>()) as u32),
        )];

        let vertices =
            vec![QuadData::vertex().gfx_vertex_input_desc(hal::pso::VertexInputRate::Instance(1))];

        // TODO: Verify this doesn't cause a double-free when combined with aux.layout
        // getting destroyed.
        let desc_set_layout_list = vec![aux.layout.as_ref().unwrap().raw()];

        let pipeline_layout = unsafe {
            factory
                .device()
                .create_pipeline_layout(desc_set_layout_list, layout_push_constants)
        }?;

        let mut vertex_buffers = Vec::new();
        let mut attributes = Vec::new();

        for &(ref elements, stride, rate) in &vertices {
            push_vertex_desc(elements, stride, rate, &mut vertex_buffers, &mut attributes);
        }

        let rect = hal::pso::Rect {
            x: 0,
            y: 0,
            w: framebuffer_width as i16,
            h: framebuffer_height as i16,
        };

        let mut shader_set = aux.shader.build(factory, Default::default()).unwrap();
        // TODO: Make disposing of the shader set nicer.  Either store it, or have a wrapper
        // that disposes it on drop, or something.  Would that cause a double-borrow?
        //
        // Actually, think about this more in general, 'cause there's other structures that
        // need similar handling: set_layouts, pipeline layouts, etc.
        let shaders = match shader_set.raw() {
            Err(e) => {
                shader_set.dispose(factory);
                // TODO: Better error type
                return Err(hal::pso::CreationError::Other);
            }
            Ok(s) => s,
        };

        let graphics_pipeline = unsafe {
            factory.device().create_graphics_pipelines(
                Some(hal::pso::GraphicsPipelineDesc {
                    shaders,
                    rasterizer: hal::pso::Rasterizer::FILL,
                    vertex_buffers,
                    attributes,
                    input_assembler: input_assembler_desc,
                    blender: hal::pso::BlendDesc {
                        logic_op: None,
                        targets: self.colors.clone(),
                    },
                    depth_stencil: depth_stencil,
                    multisampling: None,
                    baked_states: hal::pso::BakedStates {
                        viewport: Some(hal::pso::Viewport {
                            rect,
                            depth: 0.0..1.0,
                        }),
                        scissor: Some(rect),
                        blend_color: None,
                        depth_bounds: None,
                    },
                    layout: &pipeline_layout,
                    subpass,
                    flags: hal::pso::PipelineCreationFlags::empty(),
                    parent: hal::pso::BasePipeline::None,
                }),
                None,
            )
        }
        .remove(0)
        .map_err(|e| {
            shader_set.dispose(factory);
            e
        })?;

        // Create frames-in-flight
        let mut frames_in_flight = vec![];
        frames_in_flight.extend((0..aux.frames).map(|_| FrameInFlight::new(factory)));

        shader_set.dispose(factory);

        Ok(Box::new(QuadRenderGroup::<B> {
            pipeline_layout,
            graphics_pipeline,
            frames_in_flight,
        }))
    }
}

impl<B> RenderGroup<B, Aux<B>> for QuadRenderGroup<B>
where
    B: hal::Backend,
{
    fn prepare(
        &mut self,
        factory: &Factory<B>,
        _queue: QueueId,
        index: usize,
        _subpass: hal::pass::Subpass<'_, B>,
        aux: &Aux<B>,
    ) -> PrepareResult {
        self.frames_in_flight[index].prepare(factory, &aux.camera, &aux.draws);
        // TODO: Investigate this more...
        // Ooooooh in the example it always used the same draw command buffer 'cause it
        // always did indirect drawing, and just modified the draw command in the data buffer.
        // we're doing direct drawing now so we have to always re-record our drawing
        // command buffers when they change -- and the number of instances always changes
        // in this program, so!
        //PrepareResult::DrawReuse
        PrepareResult::DrawRecord
    }

    fn draw_inline(
        &mut self,
        mut encoder: RenderPassEncoder<'_, B>,
        index: usize,
        _subpass: hal::pass::Subpass<'_, B>,
        aux: &Aux<B>,
    ) {
        encoder.bind_graphics_pipeline(&self.graphics_pipeline);
        self.frames_in_flight[index].draw(&aux.draws, &self.pipeline_layout, &mut encoder);
    }

    fn dispose(self: Box<Self>, factory: &mut Factory<B>, _aux: &Aux<B>) {
        info!("Disposing of QuadRenderGroup");
        unsafe {
            for frame in self.frames_in_flight.into_iter() {
                frame.dispose(factory);
            }
            info!("Disposed frames in flight");
            factory
                .device()
                .destroy_graphics_pipeline(self.graphics_pipeline);
            info!("Destroyed pipeline");
            factory
                .device()
                .destroy_pipeline_layout(self.pipeline_layout);
            info!("Destroyed pipeline layout");
        }
    }
}

/// Add a description of our vertex buffer elements to the given
/// vertex buffer and attribute descriptors.  Maybe this can be merged
/// back into QuadRenderGroup::build() ?
fn push_vertex_desc(
    elements: &[hal::pso::Element<hal::format::Format>],
    stride: hal::pso::ElemStride,
    rate: hal::pso::VertexInputRate,
    vertex_buffers: &mut Vec<hal::pso::VertexBufferDesc>,
    attributes: &mut Vec<hal::pso::AttributeDesc>,
) {
    let index = vertex_buffers.len() as hal::pso::BufferIndex;

    vertex_buffers.push(hal::pso::VertexBufferDesc {
        binding: index,
        stride,
        rate,
    });

    let mut location = attributes.last().map_or(0, |a| a.location + 1);
    for &element in elements {
        attributes.push(hal::pso::AttributeDesc {
            location,
            binding: index,
            element,
        });
        location += 1;
    }
}

/// This is how we can load an image and create a new texture.
pub fn make_texture<B>(device: &mut GraphicsDevice<B>, image_bytes: &[u8]) -> Arc<Texture<B>>
where
    B: hal::Backend,
{
    let cursor = std::io::Cursor::new(image_bytes);
    let texture_builder = rendy::texture::image::load_from_image(cursor, Default::default())
        .expect("Could not load texture?");

    let texture = texture_builder
        .build(
            ImageState {
                queue: device.queue_id,
                stage: hal::pso::PipelineStage::FRAGMENT_SHADER,
                access: hal::image::Access::SHADER_READ,
                layout: hal::image::Layout::ShaderReadOnlyOptimal,
            },
            &mut device.factory,
        )
        .unwrap();
    Arc::new(texture)
}

/// Creates a shader builder from the given raw SPIR-V byte buffers.
/// Alignment and byte-order is handled for you.
pub fn load_shaders(vertex_src: &[u8], fragment_src: &[u8]) -> rendy::shader::ShaderSetBuilder {
    use rendy::shader::SpirvShader;
    let vert_cursor = io::Cursor::new(vertex_src);
    let vert_words = hal::pso::read_spirv(vert_cursor)
        .expect("Invalid SPIR-V buffer passed to load_shaders one way or another!");
    let vertex = SpirvShader::new(vert_words, hal::pso::ShaderStageFlags::VERTEX, "main");

    let frag_cursor = io::Cursor::new(fragment_src);
    let frag_words = hal::pso::read_spirv(frag_cursor)
        .expect("Invalid SPIR-V buffer passed to load_shaders one way or another!");
    let fragment = SpirvShader::new(frag_words, hal::pso::ShaderStageFlags::FRAGMENT, "main");

    let shader_builder: rendy::shader::ShaderSetBuilder =
        rendy::shader::ShaderSetBuilder::default()
            .with_vertex(&vertex)
            .unwrap()
            .with_fragment(&fragment)
            .unwrap();
    shader_builder
}

/// Load shaders from the given file names.
pub fn load_shader_files(
    vertex_file: &str,
    fragment_file: &str,
) -> rendy::shader::ShaderSetBuilder {
    let vertex_src = std::fs::read(vertex_file).unwrap();
    let fragment_src = std::fs::read(fragment_file).unwrap();
    load_shaders(vertex_src.as_ref(), fragment_src.as_ref())
}

/*
Exploring API

General idea: render pass -> pipeline -> draw call -> instance

From Viral:

loop {
  update_frame_data(); // This one writes into uniform buffers bound to frame level descriptor set
  for pipeline in &pipelines {
    pipeline.update_pipeline_specific_data(); // This one writes into uniform buffers bound to pipeline level descriptor set
    for material in &pipeline.materials {
      material.bind_material_descriptors_set();
      for mesh in &material.meshes {
        for object in &mesh.objects {
          object.fill_instancing_data();
        }
        mesh.draw();
      }
    }
  }
}
*/

/// An initialized graphics device context,
/// with a window.  The window itself should
/// be made optional later.  We need it for
/// building a PresentNode but should be able
/// to manage without it and add one later.
pub struct GraphicsDevice<B>
where
    B: hal::Backend,
{
    // gfx-hal types
    pub factory: Factory<B>,
    pub queue_id: QueueId,
    pub families: rendy::command::Families<B>,
}

impl<B> GraphicsDevice<B>
where
    B: hal::Backend,
{
    pub fn new() -> Self {
        use rendy::factory::Config;
        let config: Config = Default::default();

        let event_loop = EventLoop::new();
        let window = WindowBuilder::new()
            .with_title("Rendy example")
            .with_inner_size((800, 600).into());

        let rendy = rendy::init::AnyWindowedRendy::init_auto(&config, window, &event_loop).unwrap();
        rendy::with_any_windowed_rendy!((rendy)
        (mut factory, mut families, surface, window) => {

        //let (factory, families): (Factory<B>, _) = rendy::factory::init(config).unwrap();

        // TODO: HACK suggested by Frizi, just use queue 0 for everything
        // instead of getting it from `graph.node_queue(pass)`.
        // Since we control in our `Config` what families we have
        // and what they have, as long as we only ever use one family
        // (which is probably fine) then we're prooooobably okay with
        // this.
        // TODO: Check and see if this has immproved now
        let queue_id = QueueId {
            family: families.family_by_index(0).id(),
            index: 0,
        };

        Self {
            factory,
            families,
            queue_id,
        }
        })
    }
}

pub struct GraphicsWindowThing<B>
where
    B: hal::Backend,
{
    // winit window stuff
    pub window: winit::Window,
    pub event_loop: winit::EventsLoop,
    // Graph, gfx device and render targets
    pub graph: rendy::graph::Graph<B, Aux<B>>,
    pub device: GraphicsDevice<B>,
    // Our stuff
    pub aux: Aux<B>,
}

impl<B> GraphicsWindowThing<B>
where
    B: hal::Backend,
{
    pub fn make_aux(
        device: &mut GraphicsDevice<B>,
        frames: u32,
        width: f32,
        height: f32,
    ) -> Aux<B> {
        let heart_bytes =
            include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/data/heart.png"));

        let layout_set: SetLayout = FrameInFlight::<B>::get_descriptor_set_layout();
        let desc_set_layout = device
            .factory
            .create_descriptor_set_layout(layout_set.bindings)
            .expect("Bogus layout?")
            .into(); // Turn Escape into Handle

        let texture1 = make_texture(device, heart_bytes);
        let draws = vec![QuadDrawCall::new(
            texture1,
            &device.factory,
            &desc_set_layout,
        )];

        let vertex_file = concat!(env!("CARGO_MANIFEST_DIR"), "/src/data/quad.vert.spv");
        let fragment_file = concat!(env!("CARGO_MANIFEST_DIR"), "/src/data/quad.frag.spv");

        /* TODO: Hmm
        let align = device
            .factory
            .physical()
            .limits()
            .min_uniform_buffer_offset_alignment;
            */

        let aux = Aux {
            frames: frames as _,

            draws,
            camera: UniformData {
                proj: Transform3::ortho(0.0, width, height, 0.0, -100.0, 100.0),

                view: Transform3::create_translation(0.0, 0.0, 10.0),
            },

            shader: load_shader_files(vertex_file, fragment_file),
            layout: Some(desc_set_layout),
        };
        aux
    }

    pub fn new() -> Self {
        use rendy::graph::{present::PresentNode, render::*, GraphBuilder};

        let mut event_loop = EventsLoop::new();

        let window = WindowBuilder::new()
            .with_title("Arglebargle")
            .build(&event_loop)
            .unwrap();

        event_loop.poll_events(|_| ());

        let size = window
            .get_inner_size()
            .unwrap()
            .to_physical(window.get_hidpi_factor());
        let mut device = GraphicsDevice::<B>::new();

        let mut graph_builder = GraphBuilder::<B, Aux<B>>::new();
        let window_kind = hal::image::Kind::D2(size.width as u32, size.height as u32, 1, 1);

        let surface: rendy::wsi::Surface<B> = device.factory.create_surface(&window).unwrap();
        let format = device.factory.get_surface_format(&surface);
        let color = graph_builder.create_image(
            window_kind,
            1,
            device.factory.get_surface_format(&surface),
            Some(hal::command::ClearValue {
                color: hal::command::ClearColor {
                    float32: [0.1, 0.2, 0.3, 1.0],
                },
            }),
        );
        let depth = graph_builder.create_image(
            window_kind,
            1,
            hal::format::Format::D16Unorm,
            Some(hal::command::ClearValue {
                depth_stencil: hal::command::ClearDepthStencil {
                    depth: 1.0,
                    stencil: 0,
                },
            }),
        );
        let render_group_desc = QuadRenderGroupDesc::new();
        let pass = graph_builder.add_node(
            render_group_desc
                .builder()
                .into_subpass()
                .with_color(color)
                .with_depth_stencil(depth)
                .into_pass(),
        );

        println!("Surface format is {:?}", format);
        let present_builder =
            PresentNode::builder(&device.factory, surface, color).with_dependency(pass);
        let frames = present_builder.image_count();
        graph_builder.add_node(present_builder);

        let aux = Self::make_aux(&mut device, frames, size.width as f32, size.height as f32);
        let graph = graph_builder
            .with_frames_in_flight(frames)
            .build(&mut device.factory, &mut device.families, &aux)
            .unwrap();

        Self {
            window,
            event_loop,
            graph,
            device,
            aux,
        }
    }
    pub fn run(&mut self) {
        use std::time;

        let mut frames = 0u64..;
        let mut rng = oorandom::Rand32::new(12345);

        let mut should_close = false;

        let started = time::Instant::now();
        // TODO: Someday actually check against MAX_OBJECTS
        while !should_close {
            for _i in &mut frames {
                self.device.factory.maintain(&mut self.device.families);
                self.event_loop.poll_events(|event| match event {
                    Event::WindowEvent {
                        event: WindowEvent::CloseRequested,
                        ..
                    } => should_close = true,
                    _ => (),
                });
                self.graph.run(
                    &mut self.device.factory,
                    &mut self.device.families,
                    &self.aux,
                );
                // Add another object
                for draw_call in &mut self.aux.draws {
                    draw_call.add_random_object(&mut rng, 1024.0, 768.0);
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
    }

    pub fn draw(&mut self) {
        self.device.factory.maintain(&mut self.device.families);
        self.graph.run(
            &mut self.device.factory,
            &mut self.device.families,
            &self.aux,
        );
    }

    pub fn dispose(mut self) {
        // Things maybe not disposed: Texture?  DescriptorSet?
        info!("Disposing aux");
        self.aux.dispose();
        info!("Disposing graph");
        self.graph.dispose(&mut self.device.factory, &self.aux);
    }
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct DrawParam {
    pub dest: Point2,
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

/// Draws a quad with the given texture.
pub fn draw<B>(
    ctx: &mut GraphicsWindowThing<B>,
    _target: (),
    drawable: Arc<Texture<B>>,
    param: DrawParam,
) -> Result<(), ()>
where
    B: hal::Backend,
{
    /// Texture's aren't Eq and so we have a slightly ugly pointer comparison to
    /// see if two Arc<Texture> 's point to the same thing.
    fn texture_compare<B>(t1: &Arc<Texture<B>>, t2: &Arc<Texture<B>>) -> bool
    where
        B: hal::Backend,
    {
        let t1p: *const Texture<B> = t1.as_ref() as *const _;
        let t2p: *const Texture<B> = t2.as_ref() as *const _;
        t1p == t2p
    }
    // Here is where we check whether we need to create a new draw call, or just batch
    // another item onto the existing one
    let latest_draw_call = match ctx.aux.draws.last_mut() {
        Some(c) => c,
        None => {
            let c = QuadDrawCall::new(
                drawable.clone(),
                &ctx.device.factory,
                ctx.aux.layout.as_ref().unwrap(),
            );
            ctx.aux.draws.push(c);
            ctx.aux.draws.last_mut().expect("Should never happen")
        }
    };
    if texture_compare(&latest_draw_call.texture, &drawable) {
        let transform = Transform3::create_translation(param.dest.x, param.dest.y, -100.0);
        let src = Rect::from(euclid::Size2D::new(100.0, 100.0));
        let color = [1.0, 0.0, 1.0, 1.0];
        let instance = QuadData {
            transform: transform.to_column_major_array(),
            rect: [src.origin.x, src.origin.y, src.size.width, src.size.height],
            color,
        };
        latest_draw_call.add_quad(instance);
    }
    Ok(())
}
