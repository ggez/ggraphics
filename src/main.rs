/*
Okay, so the first step is going to be rendering multiple things with the same
geometry and texture using instanced drawing.  DONE.

Next step is to render things with different textures.  DONE.

Last step is to render things with different textures and
different geometry.  Trivial to do currently but it would be nice
to not re-bind the mesh descriptor if we don't have to...
How much does that actually matter though?  Dunno.

Last+1 step might be to make rendering quads a more efficient special case,
for example by reifying the geometry in the vertex shader a la the Rendy
quads example.

Last+2 step is going to be having multiple pipelines with
different shaders.

Then actual last step will be to have multiple render passes with different
render targets.

ALSO, at some point we should stop using SimpleGraphicsPipeline, per
viral, 'cause it's mostly intended for only the simplest of cases.
RenderGroup is what gets us the full power.  Looks mostly similar in
concept but with more fiddly bits to fill out.  Should be fine, right?
*/

use std::sync::Arc;
use std::{mem::size_of, time};

use rendy::command::{QueueId, RenderPassEncoder};
use rendy::factory::{Config, Factory, ImageState};
use rendy::graph::{
    present::PresentNode, render::*, GraphBuilder, GraphContext, NodeBuffer, NodeImage,
};
use rendy::hal;
use rendy::hal::Device as _;
use rendy::hal::PhysicalDevice as _;
use rendy::memory::Dynamic;
use rendy::mesh::{AsVertex, Mesh, PosColorNorm};
use rendy::resource::{
    Buffer, BufferInfo, DescriptorSet, DescriptorSetLayout, Escape, Filter, Handle, SamplerInfo,
    WrapMode,
};
use rendy::shader::{self, ShaderKind, SourceLanguage, StaticShaderInfo};
use rendy::texture::Texture;

use rand::distributions::{Distribution, Uniform};
use winit::{Event, EventsLoop, WindowBuilder, WindowEvent};

use euclid;

type Transform3 = euclid::Transform3D<f32>;
type Rect = euclid::Rect<f32>;

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
type Backend = rendy::metal::Backend;

#[cfg(not(target_os = "macos"))]
type Backend = rendy::vulkan::Backend;

/// Data we need per instance.  DrawParam gets turned into this.
/// We have to be *quite particular* about layout since this gets
/// fed straight to the shader.
///
/// TODO: Currently the shader doesn't use src or color though.
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
struct InstanceData {
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
struct UniformData {
    proj: Transform3,
    view: Transform3,
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
    buffer: Escape<Buffer<B>>,
    /// One descriptor set per draw call we do.
    /// Also has the number of instances in that draw call,
    /// so we can find offsets.
    descriptor_sets: Vec<Escape<DescriptorSet<B>>>,
    /// Offsets in the buffer to start each draw call from.
    draw_offsets: Vec<u64>,
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
        let buffer_size = gbuffer_size(align) * (draw_calls.len() as u64);
        let buffer = factory
            .create_buffer(
                BufferInfo {
                    size: buffer_size,
                    usage: hal::buffer::Usage::UNIFORM | hal::buffer::Usage::VERTEX,
                },
                Dynamic,
            )
            .unwrap();

        let descriptor_sets = vec![];
        let mut ret = Self {
            buffer,
            descriptor_sets,
            draw_offsets: vec![],
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

        let mut instance_count = 0;
        //println!("Preparing frame-in-flight, {} draw calls, first has {} instances.", draw_calls.len(),
        //draw_calls[0].objects.len());
        unsafe {
            for draw_call in draw_calls {
                // Upload the instances to the right offset in the
                // buffer.
                // Vulkan doesn't seem to like zero-size copies.
                //println!("Uploading {} instance data to {}", draw_call.objects.len(), ginstance_offset(instance_count));
                if draw_call.objects.len() > 0 {
                    factory
                        .upload_visible_buffer(
                            &mut self.buffer,
                            ginstance_offset(instance_count),
                            &draw_call.objects[..],
                        )
                        .unwrap();
                    self.draw_offsets.push(ginstance_offset(instance_count));
                    instance_count += draw_call.objects.len();
                }
            }
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
        let mut instance_count = 0;
        for ((draw_call, descriptor_set), _draw_offset) in draw_calls
            .iter()
            .zip(&self.descriptor_sets)
            .zip(&self.draw_offsets)
        {
            //println!("Drawing {:#?}, {:#?}, {}", draw_call, descriptor_set, draw_offset);

            // This is a bit weird, but basically tells the thing where to find the
            // instance data.  The stride and such of the instance structure is
            // defined in the `AsVertex` definition.
            encoder.bind_graphics_descriptor_sets(
                layout,
                0,
                std::iter::once(descriptor_set.raw()),
                std::iter::empty(),
            );
            draw_call
                .mesh
                .as_ref()
                .bind(0, &[PosColorNorm::vertex()], encoder)
                .expect("Could not bind mesh?");
            // The 1 here is a LITTLE weird; TODO: Investigate!  I THINK it is there
            // to differentiate which *place* we're binding to; see the 0 in the
            // bind_graphics_descriptor_sets().
            encoder.bind_vertex_buffers(
                1,
                std::iter::once((self.buffer.raw(), ginstance_offset(instance_count))),
            );

            encoder.push_constants(
                layout,
                hal::pso::ShaderStageFlags::ALL,
                0,
                &self.push_constants,
            );
            // The length of the mesh is the number of indices if it has any, the number
            // of verts otherwise.  See https://github.com/amethyst/rendy/issues/119
            let indices = 0..(draw_call.mesh.len() as u32);
            // This count is the *number of instances*.  What instance
            // to start at in the buffer is defined by the offset in
            // `bind_vertex_buffers()` above, and the stride/size of an instance
            // is defined in `AsVertex`.
            let instances = 0..(draw_call.objects.len() as u32);
            encoder.draw_indexed(indices, 0, instances);
            instance_count += draw_call.objects.len();
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
struct DrawCall<B>
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
    fn new(texture: Arc<Texture<B>>, mesh: Arc<Mesh<B>>) -> Self {
        let sampler_info = SamplerInfo::new(Filter::Nearest, WrapMode::Clamp);
        Self {
            objects: vec![],
            mesh,
            texture,
            sampler_info,
        }
    }

    fn add_object(&mut self, rng: &mut rand::rngs::ThreadRng, max_width: f32, max_height: f32) {
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
struct Aux<B: hal::Backend> {
    frames: usize,
    align: u64,

    draws: Vec<DrawCall<B>>,
    camera: UniformData,

    shader: rendy::shader::ShaderSetBuilder,
}

const MAX_OBJECTS: usize = 10_000;
const INSTANCES_SIZE: u64 = size_of::<InstanceData>() as u64 * MAX_OBJECTS as u64;

/// The size of the buffer in a `FrameInFlight`.
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
const fn gbuffer_size(align: u64) -> u64 {
    ((INSTANCES_SIZE - 1) / align + 1) * align
}

/// The offset pointing to free space right after the given number
/// of instances.
///
/// TODO: Are there alignment requirements for this?
const fn ginstance_offset(instance_count: usize) -> u64 {
    (instance_count * size_of::<InstanceData>()) as u64
}

#[derive(Debug)]
struct MeshRenderGroup<B>
where
    B: hal::Backend,
{
    frames_in_flight: Vec<FrameInFlight<B>>,
    set_layout: Handle<DescriptorSetLayout<B>>,
    pipeline_layout: B::PipelineLayout,
    graphics_pipeline: B::GraphicsPipeline,
}

impl<B> RenderGroup<B, Aux<B>> for MeshRenderGroup<B>
where
    B: hal::Backend,
{
    fn prepare(
        &mut self,
        factory: &Factory<B>,
        queue: QueueId,
        index: usize,
        _subpass: hal::pass::Subpass<B>,
        aux: &Aux<B>,
    ) -> PrepareResult {
        let align = aux.align;

        self.frames_in_flight[index].prepare(
            factory,
            &aux.camera,
            &aux.draws,
            &self.set_layout,
            align,
        );

        PrepareResult::DrawReuse
    }

    fn draw_inline(
        &mut self,
        encoder: RenderPassEncoder<B>,
        index: usize,
        subpass: hal::pass::Subpass<B>,
        aux: &Aux<B>,
    ) {
        // We own it, we can mutate it if we want, muahahahaha!
        let mut encoder = encoder;
        self.frames_in_flight[index].draw(
            &aux.draws,
            &self.pipeline_layout,
            &mut encoder,
            aux.align,
        );
    }

    fn dispose(self: Box<Self>, factory: &mut Factory<B>, aux: &Aux<B>) {
        // Our Aux doesn't store any resources, I think!
        // TODO: each FrameInFlight needs to be disposed of though.
        //self.pipeline.dispose(factory, aux);

        unsafe {
            factory
                .device()
                .destroy_graphics_pipeline(self.graphics_pipeline);
            factory
                .device()
                .destroy_pipeline_layout(self.pipeline_layout);
            drop(self.set_layout);
        }
    }
}

#[derive(Debug)]
struct MeshRenderGroupDesc {
    colors: Vec<hal::pso::ColorBlendDesc>,
}

impl MeshRenderGroupDesc {
    fn new() -> Self {
        Self {
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
    /// Creates the RenderGroup associated with this type.
    fn build(
        self,
        ctx: &GraphContext<B>,
        factory: &mut Factory<B>,
        queue: QueueId,
        aux: &Aux<B>,
        framebuffer_width: u32,
        framebuffer_height: u32,
        subpass: hal::pass::Subpass<B>,
        buffers: Vec<NodeBuffer>,
        images: Vec<NodeImage>,
    ) -> Result<Box<dyn RenderGroup<B, Aux<B>> + 'static>, failure::Error> {
        let desc_set_layout = Handle::from(
            factory.create_descriptor_set_layout(FrameInFlight::<B>::LAYOUT.to_vec())?,
        );
        /*
        let res = MeshRenderGroup {
            frames_in_flight: vec![],
            set_layout: desc_set_layout,
        };

        Ok(Box::new(res))
         */
        unimplemented!()
    }

    /*
    Can't implement this, see https://github.com/amethyst/rendy/issues/158
    TODO: Once we have a resolution to the above issue, add it to rendy's docs.
    fn builder(self) -> rendy::graph::DescBuilder<B, Aux<B>, Self> {
        rendy::graph::DescBuilder {
            desc: self,
            buffers: Vec::new(),
            images: Vec::new(),
            dependencies: Vec::new(),
            marker: std::marker::PhantomData,
        }
    }
     */

    /// Which buffers the render group uses.  Default
    /// should be okay?
    fn buffers(&self) -> Vec<rendy::graph::BufferAccess> {
        vec![]
    }
    /// Which images the render group uses. Default
    /// should be okay?
    fn images(&self) -> Vec<rendy::graph::ImageAccess> {
        vec![]
    }

    /// How many color blend desc's we use
    fn colors(&self) -> usize {
        self.colors.len()
    }

    /// Whether or not we use a depth buffer
    fn depth(&self) -> bool {
        true
    }
}

/// Okay, we NEED a default() method on this, 'cause it is
/// constructed implicitly by `MeshRenderPipeline::builder()`.
///
/// If we don't use `SimpleGraphicsPipeline` then we may be able
/// to avoid that.
#[derive(Debug, Default)]
struct MeshRenderPipelineDesc;

#[derive(Debug)]
struct MeshRenderPipeline<B: hal::Backend> {
    frames_in_flight: Vec<FrameInFlight<B>>,
}

impl<B> SimpleGraphicsPipelineDesc<B, Aux<B>> for MeshRenderPipelineDesc
where
    B: hal::Backend,
{
    type Pipeline = MeshRenderPipeline<B>;

    fn depth_stencil(&self) -> Option<hal::pso::DepthStencilDesc> {
        // The rendy default, except with LessEqual instead of just
        // Less.  This makes things with the exact same Z coord render
        // in draw-order with newer ones on top, with transparency.
        Some(hal::pso::DepthStencilDesc {
            depth: hal::pso::DepthTest::On {
                fun: hal::pso::Comparison::LessEqual,
                write: true,
            },
            depth_bounds: false,
            stencil: hal::pso::StencilTest::Off,
        })
    }

    fn layout(&self) -> Layout {
        // TODO: Figure this stuff out.
        // Currently we just have all draw call's use the same Layout,
        // having a more sophisticated approach would be nice someday.
        let push_constants = vec![(
            hal::pso::ShaderStageFlags::ALL,
            // Pretty sure the size of push constants is given in bytes,
            // but even putting nonsense sizes in here seems to make
            // the program run fine unless you put super extreme values in.
            // Thanks, NVidia.
            0..(size_of::<UniformData>() as u32),
        )];
        Layout {
            sets: vec![FrameInFlight::<B>::get_descriptor_set_layout()],
            push_constants,
        }
    }

    fn vertices(
        &self,
    ) -> Vec<(
        Vec<hal::pso::Element<hal::format::Format>>,
        u32,
        hal::pso::VertexInputRate,
    )> {
        vec![
            PosColorNorm::vertex().gfx_vertex_input_desc(hal::pso::VertexInputRate::Vertex),
            InstanceData::vertex().gfx_vertex_input_desc(hal::pso::VertexInputRate::Instance(1)),
        ]
    }

    fn load_shader_set(&self, factory: &mut Factory<B>, aux: &Aux<B>) -> shader::ShaderSet<B> {
        aux.shader.build(factory, Default::default()).unwrap()
    }

    fn build<'a>(
        self,
        _ctx: &GraphContext<B>,
        factory: &mut Factory<B>,
        _queue: QueueId,
        aux: &Aux<B>,
        buffers: Vec<NodeBuffer>,
        images: Vec<NodeImage>,
        set_layouts: &[Handle<DescriptorSetLayout<B>>],
    ) -> Result<MeshRenderPipeline<B>, failure::Error> {
        assert!(buffers.is_empty());
        assert!(images.is_empty());
        assert_eq!(set_layouts.len(), 1);

        let (frames, align) = (aux.frames, aux.align);

        // Each `FrameInFlight` needs one descriptor set per draw call.
        let mut frames_in_flight = vec![];
        frames_in_flight.extend(
            (0..frames).map(|_| FrameInFlight::new(factory, align, &aux.draws, &set_layouts[0])),
        );

        Ok(MeshRenderPipeline { frames_in_flight })
    }
}

impl<B> SimpleGraphicsPipeline<B, Aux<B>> for MeshRenderPipeline<B>
where
    B: hal::Backend,
{
    type Desc = MeshRenderPipelineDesc;

    fn prepare(
        &mut self,
        factory: &Factory<B>,
        _queue: QueueId,
        set_layouts: &[Handle<DescriptorSetLayout<B>>],
        index: usize,
        aux: &Aux<B>,
    ) -> PrepareResult {
        let align = aux.align;

        let layout = &set_layouts[0];
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

    fn draw(
        &mut self,
        layout: &B::PipelineLayout,
        mut encoder: RenderPassEncoder<'_, B>,
        index: usize,
        aux: &Aux<B>,
    ) {
        self.frames_in_flight[index].draw(&aux.draws, layout, &mut encoder, aux.align);
    }

    fn dispose(self, _factory: &mut Factory<B>, _aux: &Aux<B>) {}
}

/// This is how we can load an image and create a new texture.
fn make_texture<B>(
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

fn make_quad_mesh<B>(queue_id: QueueId, factory: &mut Factory<B>) -> Mesh<B>
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

fn load_shaders() -> rendy::shader::ShaderSetBuilder {
    let vertex: StaticShaderInfo = StaticShaderInfo::new(
        concat!(env!("CARGO_MANIFEST_DIR"), "/src/data/shader.glslv"),
        ShaderKind::Vertex,
        SourceLanguage::GLSL,
        "main",
    );

    let fragment: StaticShaderInfo = StaticShaderInfo::new(
        concat!(env!("CARGO_MANIFEST_DIR"), "/src/data/shader.glslf"),
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

    /*
    let pass = graph_builder.add_node(
        MeshRenderPipeline::builder()
            .into_subpass()
            .with_color(color)
            .with_depth_stencil(depth)
            .into_pass(),
    );
     */

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

    let align = factory
        .physical()
        .limits()
        .min_uniform_buffer_offset_alignment;

    let draws = vec![
        DrawCall::new(texture1, object_mesh.clone()),
        DrawCall::new(texture2, object_mesh.clone()),
        DrawCall::new(texture3, object_mesh.clone()),
        DrawCall::new(texture4, object_mesh.clone()),
    ];
    let mut aux = Aux {
        frames: frames as _,
        align,

        draws,
        camera: UniformData {
            proj: Transform3::ortho(0.0, width, height, 0.0, 1.0, 200.0),

            view: Transform3::create_translation(0.0, 0.0, 10.0),
        },

        shader: load_shaders(),
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
