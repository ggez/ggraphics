/*
Okay, so the first step is going to be rendering multiple things with the same
geometry and texture using instanced drawing.  DONE.

Next step is to render things with different textures.  DONE.

Last step is to render things with different textures and
different geometry.  Trivial to do currently but it would be nice
to not re-bind the mesh descriptor if we don't have to...
How much does that actually matter though?  Dunno.
DONE.

Time to clean up and refactor!

Last+1 step is going to be having multiple pipelines with
different shaders.

Last+2 step might be to make rendering quads a more efficient special case,
for example by reifying the geometry in the vertex shader a la the Rendy
quads example.  This might also be where we try to reduce
descriptor swaps if possible.

Then actual last step will be to have multiple render passes with different
render targets.
*/

use std::mem;
use std::sync::Arc;

use rendy::command::{QueueId, RenderPassEncoder};
use rendy::factory::{Factory, ImageState};
use rendy::graph::{render::*, GraphContext, NodeBuffer, NodeImage};
use rendy::hal;
use rendy::hal::Device as _;
use rendy::memory::Dynamic;
use rendy::mesh::{AsVertex, Mesh, PosColorNorm};
use rendy::resource::{
    Buffer, BufferInfo, DescriptorSet, DescriptorSetLayout, Escape, Filter, Handle, SamplerInfo,
    WrapMode,
};
use rendy::shader::{ShaderKind, SourceLanguage};
use rendy::texture::Texture;

use rand::distributions::{Distribution, Uniform};

use euclid;

pub type Transform3 = euclid::Transform3D<f32>;
pub type Rect = euclid::Rect<f32>;

// TODO: Think a bit better about how to do this.  Can we set it or specialize it at runtime perhaps?
// Perhaps.
// For now though, this is okay if not great.
// It WOULD be quite nice to be able to play with OpenGL and DX12 backends.
//
// TODO: We ALSO need to specify features to rendy to build these, so this doesn't even work currently.
// For now we only ever specify Vulkan.
//
// Rendy doesn't currently work on gfx-rs's DX12 backend though, and the OpenGL backend
// is still WIP, so...  I guess this is what we get.
#[cfg(target_os = "macos")]
pub type Backend = rendy::metal::Backend;

#[cfg(not(target_os = "macos"))]
pub type Backend = rendy::vulkan::Backend;

/// Data we need per instance.  DrawParam gets turned into this.
/// We have to be *quite particular* about layout since this gets
/// fed straight to the shader.
///
/// TODO: Currently the shader doesn't use src or color though.
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct InstanceData {
    // The euclid types don't impl PartialOrd and PartialEq so
    // we have to use the lower-level forms for
    // actually sending to the GPU.  :-/
    transform: [f32; 16],
    src: [f32; 4],
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
impl AsVertex for InstanceData {
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
/// `InstanceData`.
/// TODO: Make it resizeable someday.
///
/// The buffer in a `FrameInFlight`.
///
/// We pack data from multiple `DrawCall`'s together into one `Buffer`
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
        mem::size_of::<InstanceData>() as u64
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
    pub fn add_slice(
        &mut self,
        factory: &Factory<B>,
        instances: &[InstanceData],
    ) -> Result<u64, ()> {
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
}

/// What data we need for each frame in flight.
/// Rendy doesn't do any synchronization for us
/// beyond guarenteeing that when we get a new frame
/// index everything touched by that frame index is
/// now free to reuse.  So we just make one entire
/// set of data per frame.
///
/// We could make different frames share buffers
/// and descriptor sets and such and only change bits
/// that aren't in use from other frames, but that's
/// more complex than I want to get into right now.
///
/// When we do want to do that though, I think the simple
/// way would be... maybe create a structure through which
/// a FrameInFlight can be altered and which records if
/// things have actually changed.  Actually, the DrawCall
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
    buffer: InstanceBuffer<B>, //Escape<Buffer<B>>,
    /// One descriptor set per draw call we do.
    /// Also has the number of instances in that draw call,
    /// so we can find offsets.
    descriptor_sets: Vec<Escape<DescriptorSet<B>>>,
    /// The frame's local copy of uniform data.
    push_constants: [u32; 32],
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

    fn new(
        factory: &mut Factory<B>,
        align: u64,
        draw_calls: &[DrawCall<B>],
        descriptor_set: &Handle<DescriptorSetLayout<B>>,
    ) -> Self {
        use std::convert::TryInto;
        let buffer_count = MAX_OBJECTS * draw_calls.len();
        let buffer = InstanceBuffer::new(buffer_count.try_into().unwrap(), factory);
        let descriptor_sets = vec![];
        let mut ret = Self {
            buffer,
            descriptor_sets,
            push_constants: [0; 32],
        };
        for draw_call in draw_calls {
            // all descriptor sets use the same layout
            // we need one per draw call, per frame in flight.
            ret.create_descriptor_sets(factory, draw_call, descriptor_set, align);
        }
        ret
    }

    /// Takes a draw call and creates a descriptor set that points
    /// at its resources.
    ///
    /// For now we do not care about preserving descriptor sets between draw calls.
    fn create_descriptor_sets(
        &mut self,
        factory: &Factory<B>,
        draw_call: &DrawCall<B>,
        layout: &Handle<DescriptorSetLayout<B>>,
        _align: u64,
    ) {
        // Does this sampler need to stay alive?  We pass a
        // reference to it elsewhere in an `unsafe` block...
        // It's cached in the Factory anyway, so I don't think so.
        let sampler = factory
            .get_sampler(draw_call.sampler_info.clone())
            .expect("Could not get sampler");

        unsafe {
            let set = factory.create_descriptor_set(layout.clone()).unwrap();
            factory.write_descriptor_sets(Some(hal::pso::DescriptorSetWrite {
                set: set.raw(),
                binding: 1,
                array_offset: 0,
                descriptors: vec![hal::pso::Descriptor::Image(
                    draw_call.texture.view().raw(),
                    hal::image::Layout::ShaderReadOnlyOptimal,
                )],
            }));
            factory.write_descriptor_sets(Some(hal::pso::DescriptorSetWrite {
                set: set.raw(),
                binding: 2,
                array_offset: 0,
                descriptors: vec![hal::pso::Descriptor::Sampler(sampler.raw())],
            }));
            self.descriptor_sets.push(set);
        }
    }

    /// This happens before a frame; it should take a LIST of draw calls and take
    /// care of uploading EACH of them into the buffer so they don't clash!
    fn prepare(
        &mut self,
        factory: &Factory<B>,
        uniforms: &UniformData,
        draw_calls: &[DrawCall<B>],
        _layout: &Handle<DescriptorSetLayout<B>>,
        _align: u64,
    ) {
        assert!(draw_calls.len() > 0);
        // Store the uniforms to be shoved into push constants this frame
        // TODO: Be less crude about indexing and such.
        for (i, vl) in uniforms.proj.to_row_major_array().into_iter().enumerate() {
            self.push_constants[i] = vl.to_bits();
        }
        for (i, vl) in uniforms.view.to_row_major_array().into_iter().enumerate() {
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

    /// Draws a list of DrawCall's.
    fn draw(
        &mut self,
        draw_calls: &[DrawCall<B>],
        layout: &B::PipelineLayout,
        encoder: &mut RenderPassEncoder<'_, B>,
        _align: u64,
    ) {
        //println!("Drawing {} draw calls", draw_calls.len());
        let mut instance_count: u64 = 0;
        for (draw_call, descriptor_set) in draw_calls.iter().zip(&self.descriptor_sets) {
            //println!("Drawing {:#?}, {:#?}, {}", draw_call, descriptor_set, draw_offset);

            // This is a bit weird, but basically tells the thing where to find the
            // instance data.  The stride and such of the instance structure is
            // defined in the `AsVertex` definition.
            unsafe {
                encoder.bind_graphics_descriptor_sets(
                    layout,
                    0,
                    std::iter::once(descriptor_set.raw()),
                    std::iter::empty(),
                );
            }
            draw_call
                .mesh
                .as_ref()
                .bind(0, &[PosColorNorm::vertex()], encoder)
                .expect("Could not bind mesh?");
            // The 1 here is a LITTLE weird; TODO: Investigate!  I THINK it is there
            // to differentiate which *place* we're binding to; see the 0 in the
            // bind_graphics_descriptor_sets().
            unsafe {
                encoder.bind_vertex_buffers(
                    1,
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
            // The length of the mesh is the number of indices if it has any, the number
            // of verts otherwise.  See https://github.com/amethyst/rendy/issues/119
            let indices = 0..(draw_call.mesh.len() as u32);
            // This count is the *number of instances*.  What instance
            // to start at in the buffer is defined by the offset in
            // `bind_vertex_buffers()` above, and the stride/size of an instance
            // is defined in `AsVertex`.
            let instances = 0..(draw_call.objects.len() as u32);
            unsafe {
                encoder.draw_indexed(indices, 0, instances);
            }
            instance_count += draw_call.objects.len() as u64;
        }
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
pub struct DrawCall<B>
where
    B: hal::Backend,
{
    objects: Vec<InstanceData>,
    mesh: Arc<Mesh<B>>,
    texture: Arc<Texture<B>>,
    /// We just need the actual config for the sampler 'cause
    /// Rendy's `Factory` can manage a sampler cache itself.
    sampler_info: SamplerInfo,
}

impl<B> DrawCall<B>
where
    B: hal::Backend,
{
    pub fn new(texture: Arc<Texture<B>>, mesh: Arc<Mesh<B>>) -> Self {
        let sampler_info = SamplerInfo::new(Filter::Nearest, WrapMode::Clamp);
        Self {
            objects: vec![],
            mesh,
            texture,
            sampler_info,
        }
    }

    pub fn add_object(&mut self, rng: &mut rand::rngs::ThreadRng, max_width: f32, max_height: f32) {
        if self.objects.len() < MAX_OBJECTS {
            let rx = Uniform::new(0.0, max_width);
            let ry = Uniform::new(0.0, max_height);
            let x = rx.sample(rng);
            let y = ry.sample(rng);
            //println!("Adding object at {}x{}", x, y);
            let transform = Transform3::create_translation(x, y, -100.0);
            let src = Rect::from(euclid::Size2D::new(1.0, 1.0));
            let color = [1.0, 0.0, 1.0, 1.0];
            let instance = InstanceData {
                transform: transform.to_row_major_array(),
                src: [src.origin.x, src.origin.y, src.size.width, src.size.height],
                color,
            };
            self.objects.push(instance);
        }
    }
}

#[derive(Debug)]
pub struct Aux<B: hal::Backend> {
    pub frames: usize,
    pub align: u64,

    pub draws: Vec<DrawCall<B>>,
    pub camera: UniformData,

    pub shader: rendy::shader::ShaderSetBuilder,
}

const MAX_OBJECTS: usize = 10_000;

/// Render group that consist of simple graphics pipeline.
#[derive(Debug)]
pub struct MeshRenderGroup<B>
where
    B: hal::Backend,
{
    set_layouts: Vec<Handle<DescriptorSetLayout<B>>>,
    pipeline_layout: B::PipelineLayout,
    graphics_pipeline: B::GraphicsPipeline,
    frames_in_flight: Vec<FrameInFlight<B>>,
}

/// Descriptor for simple render group.
#[derive(Debug)]
pub struct MeshRenderGroupDesc {
    // inner: MeshRenderPipelineDesc,
    colors: Vec<hal::pso::ColorBlendDesc>,
}

impl MeshRenderGroupDesc {
    pub fn new() -> Self {
        Self {
            // inner: MeshRenderPipelineDesc,
            colors: vec![hal::pso::ColorBlendDesc(
                hal::pso::ColorMask::ALL,
                hal::pso::BlendState::ALPHA,
            )],
        }
    }
}

impl<B> RenderGroupDesc<B, Aux<B>> for MeshRenderGroupDesc
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
    ) -> Result<Box<dyn RenderGroup<B, Aux<B>>>, failure::Error> {
        let depth_stencil = hal::pso::DepthStencilDesc {
            depth: hal::pso::DepthTest::On {
                fun: hal::pso::Comparison::LessEqual,
                write: true,
            },
            depth_bounds: false,
            stencil: hal::pso::StencilTest::Off,
        };
        let input_assembler_desc = hal::pso::InputAssemblerDesc {
            primitive: hal::Primitive::TriangleList,
            primitive_restart: hal::pso::PrimitiveRestart::Disabled,
        };

        let layout_sets = vec![FrameInFlight::<B>::get_descriptor_set_layout()];
        let layout_push_constants = vec![(
            hal::pso::ShaderStageFlags::ALL,
            // Pretty sure the size of push constants is given in bytes,
            // but even putting nonsense sizes in here seems to make
            // the program run fine unless you put super extreme values in.
            // Thanks, NVidia.
            0..(mem::size_of::<UniformData>() as u32),
        )];

        // vertices: std::vec::Vec<(std::vec::Vec<gfx_hal::pso::input_assembler::Element<gfx_hal::format::Format>>, u32, gfx_hal::pso::input_assembler::VertexInputRate)>
        let vertices = vec![
            PosColorNorm::vertex().gfx_vertex_input_desc(hal::pso::VertexInputRate::Vertex),
            InstanceData::vertex().gfx_vertex_input_desc(hal::pso::VertexInputRate::Instance(1)),
        ];

        let set_layouts = layout_sets
            .into_iter()
            .map(|set| {
                factory
                    .create_descriptor_set_layout(set.bindings)
                    .map(Handle::from)
            })
            .collect::<Result<Vec<_>, _>>()?;

        let pipeline_layout = unsafe {
            factory
                .device()
                .create_pipeline_layout(set_layouts.iter().map(|l| l.raw()), layout_push_constants)
        }?;

        let mut vertex_buffers = Vec::new();
        let mut attributes = Vec::new();

        for &(ref elemets, stride, rate) in &vertices {
            push_vertex_desc(elemets, stride, rate, &mut vertex_buffers, &mut attributes);
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
                return Err(e);
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

        let (frames, align) = (aux.frames, aux.align);

        // Each `FrameInFlight` needs one descriptor set per draw call.
        let mut frames_in_flight = vec![];
        frames_in_flight.extend(
            (0..frames).map(|_| FrameInFlight::new(factory, align, &aux.draws, &set_layouts[0])),
        );

        shader_set.dispose(factory);

        Ok(Box::new(MeshRenderGroup::<B> {
            set_layouts,
            pipeline_layout,
            graphics_pipeline,
            frames_in_flight,
        }))
    }
}

impl<B> RenderGroup<B, Aux<B>> for MeshRenderGroup<B>
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
        let align = aux.align;

        let layout = &self.set_layouts[0];
        self.frames_in_flight[index].prepare(factory, &aux.camera, &aux.draws, layout, align);
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
        self.frames_in_flight[index].draw(
            &aux.draws,
            &self.pipeline_layout,
            &mut encoder,
            aux.align,
        );
    }

    fn dispose(self: Box<Self>, factory: &mut Factory<B>, _aux: &Aux<B>) {
        unsafe {
            factory
                .device()
                .destroy_graphics_pipeline(self.graphics_pipeline);
            factory
                .device()
                .destroy_pipeline_layout(self.pipeline_layout);
            drop(self.set_layouts);
        }
    }
}

pub fn push_vertex_desc(
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
pub fn make_texture<B>(
    queue_id: QueueId,
    factory: &mut Factory<B>,
    image_bytes: &[u8],
) -> Arc<Texture<B>>
where
    B: hal::Backend,
{
    let cursor = std::io::Cursor::new(image_bytes);
    let texture_builder = rendy::texture::image::load_from_image(cursor, Default::default())
        .expect("Could not load texture?");

    let texture = texture_builder
        .build(
            ImageState {
                queue: queue_id,
                stage: hal::pso::PipelineStage::FRAGMENT_SHADER,
                access: hal::image::Access::SHADER_READ,
                layout: hal::image::Layout::ShaderReadOnlyOptimal,
            },
            factory,
        )
        .unwrap();
    Arc::new(texture)
}

pub fn make_quad_mesh<B>(queue_id: QueueId, factory: &mut Factory<B>) -> Mesh<B>
where
    B: hal::Backend,
{
    let verts: Vec<[f32; 3]> = vec![
        [0.0, 0.0, 0.0],
        [0.0, 100.0, 0.0],
        [100.0, 100.0, 0.0],
        [100.0, 0.0, 0.0],
    ];
    let indices = rendy::mesh::Indices::from(vec![0u32, 1, 2, 0, 2, 3]);
    // TODO: Mesh color... how do we want to handle this?
    // It's a bit of an open question in ggez as well, so.
    // For now the shader just uses the vertex color.
    // It feels weird but ggez more or less requires both vertex
    // colors and per-model colors.
    // Unless you want to handle changing sprite colors by
    // creating entirely new geometry for the sprite.
    let vertices: Vec<_> = verts
        .into_iter()
        .map(|v| PosColorNorm {
            position: rendy::mesh::Position::from(v),
            color: [1.0, 1.0, 1.0, 1.0].into(),
            normal: rendy::mesh::Normal::from([0.0, 0.0, 1.0]),
        })
        .collect();

    let m = Mesh::<Backend>::builder()
        .with_indices(indices)
        .with_vertices(&vertices[..])
        .build(queue_id, &factory)
        .unwrap();
    m
}

pub fn make_tri_mesh<B>(queue_id: QueueId, factory: &mut Factory<B>) -> Mesh<B>
where
    B: hal::Backend,
{
    let verts: Vec<[f32; 3]> = vec![[0.0, 0.0, 0.0], [100.0, 0.0, 0.0], [50.0, 100.0, 0.0]];
    let indices = rendy::mesh::Indices::from(vec![0u32, 1, 2]);
    // TODO: Mesh color... how do we want to handle this?
    // It's a bit of an open question in ggez as well, so.
    // For now the shader just uses the vertex color.
    // It feels weird but ggez more or less requires both vertex
    // colors and per-model colors.
    // Unless you want to handle changing sprite colors by
    // creating entirely new geometry for the sprite.
    let vertices: Vec<_> = verts
        .into_iter()
        .map(|v| PosColorNorm {
            position: rendy::mesh::Position::from(v),
            color: [1.0, 1.0, 1.0, 1.0].into(),
            normal: rendy::mesh::Normal::from([0.0, 0.0, 1.0]),
        })
        .collect();

    let m = Mesh::<Backend>::builder()
        .with_indices(indices)
        .with_vertices(&vertices[..])
        .build(queue_id, &factory)
        .unwrap();
    m
}

/// Creates a shader builder from the given GLSL shader source texts.
///
/// Also takes names for the vertex and fragment shaders to be used
/// in debugging output.
pub fn load_shaders(
    vertex_src: &str,
    vertex_name: &str,
    fragment_src: &str,
    fragment_name: &str,
) -> rendy::shader::ShaderSetBuilder {
    use rendy::shader::SourceCodeShaderInfo;
    let vertex = SourceCodeShaderInfo::new(
        vertex_src,
        vertex_name,
        ShaderKind::Vertex,
        SourceLanguage::GLSL,
        "main",
    );
    let fragment = SourceCodeShaderInfo::new(
        fragment_src,
        fragment_name,
        ShaderKind::Fragment,
        SourceLanguage::GLSL,
        "main",
    );

    let shader_builder: rendy::shader::ShaderSetBuilder =
        rendy::shader::ShaderSetBuilder::default()
            .with_vertex(&vertex)
            .unwrap()
            .with_fragment(&fragment)
            .unwrap();
    shader_builder
}

pub fn load_shader_files(
    vertex_file: &str,
    fragment_file: &str,
) -> rendy::shader::ShaderSetBuilder {
    let vertex_src = std::fs::read_to_string(vertex_file).unwrap();
    let fragment_src = std::fs::read_to_string(fragment_file).unwrap();
    load_shaders(
        vertex_src.as_ref(),
        vertex_file,
        fragment_src.as_ref(),
        fragment_file,
    )
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


pub struct DrawCall {
    // Mesh, texture, sampler info instance data.
}

/// Roughly corresponds to a RenderGroup
pub struct Pipeline {
    // Draws
// Uniforms
// Shaders
}

pub struct GraphicsDevice {
    // frames in flight...
// descriptor sets...
}

impl GraphicsDevice {
    pub fn add_draw(&mut self, mesh: (), texture: (), sampler_info: (), instances: &[InstanceData]) {}
}

pub fn present() {}
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

    // Rendy types
    pub depth: rendy::graph::ImageId,
    pub color: rendy::graph::ImageId,
    pub graph: rendy::graph::Graph<B, Aux<B>>,
    pub aux: Aux<B>,
    // winit types
    pub window: winit::Window,
    pub event_loop: winit::EventsLoop,
    // Config data
    pub frame_count: u32,
    pub align: u64,
}

impl<B> GraphicsDevice<B>
where
    B: hal::Backend,
{
    pub fn new() -> Self {
        use rendy::factory::Config;
        use rendy::graph::{present::PresentNode, render::*, GraphBuilder};
        use rendy::hal::PhysicalDevice as _;
        use winit::{EventsLoop, WindowBuilder};

        let config: Config = Default::default();

        let (mut factory, mut families): (Factory<B>, _) = rendy::factory::init(config).unwrap();

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

        let mut graph_builder = GraphBuilder::<B, Aux<B>>::new();

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
        // TODO: Check and see if this has immproved now
        let queue_id = QueueId {
            family: families.family_by_index(0).id(),
            index: 0,
        };

        let heart_bytes =
            include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/data/heart.png"));
        let texture1 = make_texture(queue_id, &mut factory, heart_bytes);
        let object_mesh = Arc::new(make_quad_mesh(queue_id, &mut factory));
        let draws = vec![DrawCall::new(texture1, object_mesh.clone())];

        let vertex_file = concat!(env!("CARGO_MANIFEST_DIR"), "/src/data/shader.glslv");
        let fragment_file = concat!(env!("CARGO_MANIFEST_DIR"), "/src/data/shader.glslf");

        let align = factory
            .physical()
            .limits()
            .min_uniform_buffer_offset_alignment;

        let width = window_size.width as f32;
        let height = window_size.height as f32;
        let aux = Aux {
            frames: frames as _,
            align,

            draws,
            camera: UniformData {
                proj: Transform3::ortho(0.0, width, height, 0.0, 1.0, 200.0),

                view: Transform3::create_translation(0.0, 0.0, 10.0),
            },

            shader: load_shader_files(vertex_file, fragment_file),
        };

        let graph = graph_builder
            .with_frames_in_flight(frames)
            .build(&mut factory, &mut families, &aux)
            .unwrap();

        Self {
            factory,
            families,
            queue_id,
            depth,
            color,
            window,
            event_loop,
            graph,
            aux,
            frame_count: frames,
            align: align,
        }
    }

    pub fn run(&mut self) {
        use std::time;
        use winit::{Event, EventsLoop, WindowBuilder, WindowEvent};

        let mut frames = 0u64..;
        let mut rng = rand::thread_rng();

        let mut should_close = false;

        let started = time::Instant::now();
        // TODO: Someday actually check against MAX_OBJECTS
        while !should_close {
            for _i in &mut frames {
                self.factory.maintain(&mut self.families);
                self.event_loop.poll_events(|event| match event {
                    Event::WindowEvent {
                        event: WindowEvent::CloseRequested,
                        ..
                    } => should_close = true,
                    _ => (),
                });
                self.graph
                    .run(&mut self.factory, &mut self.families, &self.aux);
                // Add another object
                for draw_call in &mut self.aux.draws {
                    draw_call.add_object(&mut rng, 1024.0, 768.0);
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

    #[cfg(not(target_os = "macos"))]
    pub fn new_vulkan() -> GraphicsDevice<rendy::vulkan::Backend> {
        GraphicsDevice::new()
    }

    #[cfg(target_os = "macos")]
    pub fn new_metal() -> GraphicsDevice<rendy::metal::Backend> {
        GraphicsDevice::new()
    }

    pub fn dispose(mut self) {
        // TODO: This doesn't actually dispose of everything right
        self.graph.dispose(&mut self.factory, &self.aux)
    }
}

#[cfg(not(target_os = "macos"))]
pub fn new_vulkan_device() -> GraphicsDevice<rendy::vulkan::Backend> {
    GraphicsDevice::new()
}

// #[cfg(target_os = "windows")]
// pub fn new_dx12() -> GraphicsDevice<rendy::dx12::Backend> {
//     GraphicsDevice::new()
// }

// pub fn new_empty() -> GraphicsDevice<rendy::empty::Backend> {
//     GraphicsDevice::new()
// }
