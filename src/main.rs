/*
TODO: Make backend select-able either based on platform,
or at runtime?
TODO: Rendy has no gl backend yet.
TODO: Make shaderc less inconvenient?

Okay, so the first step is going to be rendering multiple things with the same
geometry and texture using instanced drawing.

Next step is to render things with different textures.

Last step is to render things with different textures and
different geometry.

Last+1 step might be to make rendering quads a more efficient special case,
for example by reifying the geometry in the vertex shader a la the Rendy
quads example.
 */

use {
    gfx_hal::PhysicalDevice as _,
    rendy::{
        command::{DrawIndexedCommand, QueueId, RenderPassEncoder},
        factory::{Config, Factory, ImageState},
        graph::{
            present::PresentNode, render::*, GraphBuilder, GraphContext, NodeBuffer, NodeImage,
        },
        hal as gfx_hal,
        hal::Device as _,
        memory::Dynamic,
        mesh::{AsVertex, Mesh, PosColorNorm, Transform},
        resource::{
            Buffer, BufferInfo, DescriptorSet, DescriptorSetLayout, Escape, Filter, Handle,
            SamplerInfo, WrapMode,
        },
        shader::{Shader, ShaderKind, SourceLanguage, SpirvShader, StaticShaderInfo},
        texture::Texture,
    },
};

use std::{mem::size_of, time};

use rand::distributions::{Distribution, Uniform};

use winit::{Event, EventsLoop, WindowBuilder, WindowEvent};

use euclid;

type Point2 = euclid::Vector2D<f32>;
type Vector2 = euclid::Vector2D<f32>;
type Vector3 = euclid::Vector3D<f32>;
type Transform3 = euclid::Transform3D<f32>;
type Color = [f32; 4];
type Rect = euclid::Rect<f32>;

// TODO: Think a bit better about how to do this.  Can we set it or specialize it at runtime perhaps?
// Perhaps.
// For now though, this is okay if not great.
// It WOULD be quite nice to be able to play with OpenGL and DX12 backends.
//
// TODO: We ALSO need to specify features to rendy to build these, so.  For now we only ever
// specify Vulkan.
#[cfg(target_os = "macos")]
type Backend = rendy::metal::Backend;

#[cfg(not(target_os = "macos"))]
type Backend = rendy::vulkan::Backend;

lazy_static::lazy_static! {
    static ref VERTEX: SpirvShader = StaticShaderInfo::new(
       // concat!(env!("CARGO_MANIFEST_DIR"), "/examples/meshes/shader.vert"),
        concat!(env!("CARGO_MANIFEST_DIR"), "/src/data/shader.glslv"),
        ShaderKind::Vertex,
        SourceLanguage::GLSL,
        "main",
    ).precompile().unwrap();

    static ref FRAGMENT: SpirvShader = StaticShaderInfo::new(
        //concat!(env!("CARGO_MANIFEST_DIR"), "/examples/meshes/shader.frag"),
        concat!(env!("CARGO_MANIFEST_DIR"), "/src/data/shader.glslf"),
        ShaderKind::Fragment,
        SourceLanguage::GLSL,
        "main",
    ).precompile().unwrap();
}

/// Data we need per instance.  DrawParam gets turned into this.
/// We have to be *quite particular* about layout since this gets
/// fed straight to the shader.
///
/// TODO: Currently the shader doesn't use src or color though.
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
struct InstanceData {
    // These don't impl PartialOrd and PartialEq so
    // we have to use the lower-level forms for
    // actually sending to the GPU...?
    //transform: Transform3,
    //src: Rect,
    //color: Color,

    transform: [f32;16],
    src: [f32;4],
    color: [f32;4],
}

use rendy::mesh::{Attribute, VertexFormat};
use std::borrow::Cow;

/// Okay, this tripped me up.  Instance data is technically
/// part of the per-vertex data.  So we describe it as
/// part of the vertex format.  This is where that
/// definition happens.
///
/// This trait impl is basically extended from the impl for
/// `rendy::mesh::Transform`
impl AsVertex for InstanceData {
    const VERTEX: VertexFormat<'static> =  VertexFormat {
        attributes: Cow::Borrowed(&[
            // transform as a `[vec4;4]`
            Attribute {
                format: gfx_hal::format::Format::Rgba32Float,
                offset: 0,
            },
            Attribute {
                format: gfx_hal::format::Format::Rgba32Float,
                offset: 16,
            },
            Attribute {
                format: gfx_hal::format::Format::Rgba32Float,
                offset: 32,
            },
            Attribute {
                format: gfx_hal::format::Format::Rgba32Float,
                offset: 48,
            },
            // rect
            Attribute {
                format: gfx_hal::format::Format::Rgba32Float,
                offset: 64,
            },
            // color
            Attribute {
                format: gfx_hal::format::Format::Rgba32Float,
                offset: 80,
            },
        ]),
        stride: 96,
};
}

/// Uniform data.  Each Scene contains one of these.
#[derive(Clone, Copy, Debug)]
#[repr(C, align(16))]
struct UniformData {
    proj: Transform3,
    view: Transform3,
}

/// What data we need for each frame in flight.
/// We could make different frames share buffers
/// and descriptor sets and such and only change bits
/// that aren't in use from other frames, but that's
/// more complex than I want to get into right now.
/// Rendy doesn't do any synchronization for us
/// beyond guarenteeing that when we get a new frame
/// index everything touched by that frame index is
/// now free to reuse.
#[derive(Debug)]
struct FrameInFlight<B>
where
    B: gfx_hal::Backend,
{
    /// The buffer where we store uniform and instance data.
    buffer: Escape<Buffer<B>>,
    /// One descriptor set per draw call we do.
    /// Also has the number of instances in that draw call,
    /// so we can find offsets.
    descriptor_sets: Vec<Escape<DescriptorSet<B>>>,
    /// Offsets in the buffer to start each draw call from.
    draw_offsets: Vec<u64>,
}

impl<B> FrameInFlight<B>
where
    B: gfx_hal::Backend,
{
    /// All our descriptor sets use the same layout.
    /// This one!  We have a uniform buffer, an
    /// image, and a sampler.
    const LAYOUT: &'static [gfx_hal::pso::DescriptorSetLayoutBinding] = &[
        gfx_hal::pso::DescriptorSetLayoutBinding {
            binding: 0,
            ty: gfx_hal::pso::DescriptorType::UniformBuffer,
            count: 1,
            stage_flags: gfx_hal::pso::ShaderStageFlags::GRAPHICS,
            immutable_samplers: false,
        },
        gfx_hal::pso::DescriptorSetLayoutBinding {
            binding: 1,
            ty: gfx_hal::pso::DescriptorType::SampledImage,
            count: 1,
            stage_flags: gfx_hal::pso::ShaderStageFlags::FRAGMENT,
            immutable_samplers: false,
        },
        gfx_hal::pso::DescriptorSetLayoutBinding {
            binding: 2,
            ty: gfx_hal::pso::DescriptorType::Sampler,
            count: 1,
            stage_flags: gfx_hal::pso::ShaderStageFlags::FRAGMENT,
            immutable_samplers: false,
        },
    ];

    fn get_descriptor_set_layout() -> SetLayout {
        SetLayout {
            bindings: Self::LAYOUT.to_vec(),
        }
    }

    fn new(factory: &mut Factory<B>, align: u64, draw_calls: &[DrawCall<B>], descriptor_set: &Handle<DescriptorSetLayout<B>>) -> Self {
        let buffer = factory
            .create_buffer(
                BufferInfo {
                    size: gbuffer_size(align),
                    usage: gfx_hal::buffer::Usage::UNIFORM
                        | gfx_hal::buffer::Usage::INDIRECT
                        | gfx_hal::buffer::Usage::VERTEX,
                },
                Dynamic,
            )
            .unwrap();

        let descriptor_sets = vec![];
        let mut ret = Self {
            buffer,
            descriptor_sets,
            draw_offsets: vec![],
        };
        let layout = Self::LAYOUT;
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
    /// For now we do not care about reusing descriptor sets.
    /// They all have the same layout though, so should be easy
    /// eventually, but for now meh.
    fn create_descriptor_sets(
        &mut self,
        factory: &Factory<B>,
        draw_call: &DrawCall<B>,
        layout: &Handle<DescriptorSetLayout<B>>,
        align: u64,
    ) {
        // TODO: Does this sampler need to stay alive?
        let sampler = factory
            .get_sampler(draw_call.sampler_info.clone())
            .expect("Could not get sampler");

        unsafe {
            let set = factory.create_descriptor_set(layout.clone()).unwrap();

            factory.write_descriptor_sets(Some(gfx_hal::pso::DescriptorSetWrite {
                set: set.raw(),
                binding: 0,
                array_offset: 0,
                descriptors: Some(gfx_hal::pso::Descriptor::Buffer(
                    self.buffer.raw(),
                    Some(guniform_offset())..Some(guniform_offset() + UNIFORM_SIZE),
                )),
            }));
            factory.write_descriptor_sets(Some(gfx_hal::pso::DescriptorSetWrite {
                set: set.raw(),
                binding: 1,
                array_offset: 0,
                descriptors: vec![gfx_hal::pso::Descriptor::Image(
                    draw_call.texture.view().raw(),
                    gfx_hal::image::Layout::ShaderReadOnlyOptimal,
                )],
            }));
            factory.write_descriptor_sets(Some(gfx_hal::pso::DescriptorSetWrite {
                set: set.raw(),
                binding: 2,
                array_offset: 0,
                descriptors: vec![gfx_hal::pso::Descriptor::Sampler(sampler.raw())],
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
        layout: &Handle<DescriptorSetLayout<B>>,
        align: u64,
    ) {
        let mut instance_count = 0;
        unsafe {
            factory
                .upload_visible_buffer(&mut self.buffer, guniform_offset(), &[uniforms.clone()])
                .unwrap();

            for draw_call in draw_calls {

                // Upload the instances to the right offset in the
                // buffer.
                // Vulkan doesn't seem to like zero-size copies.
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
        align: u64,
    ) {
        let mut instances_count = 0;
        for ((draw_call, descriptor_set), draw_offset) in draw_calls
            .iter()
            .zip(&self.descriptor_sets)
            .zip(&self.draw_offsets)
        {
            draw_call
                .mesh
                .bind(&[PosColorNorm::VERTEX], encoder)
                .expect("Could not bind mesh?");

            // This is a bit weird, but basically tells the thing where to find the
            // instance data.  The stride and such of the instance structure is
            // defined in the `AsVertex` definition.
            //
            // The 1 here is a LITTLE weird; TODO: Investigate!  I THINK it is there
            // to differentiate which *place* we're binding to; see the 0 in the
            // bind_graphics_descriptor_sets().
            encoder.bind_graphics_descriptor_sets(
                layout,
                0,
                std::iter::once(descriptor_set.raw()),
                std::iter::empty(),
            );
            encoder.bind_vertex_buffers(
                1,
                std::iter::once((self.buffer.raw(), ginstance_offset(instances_count))),
            );
            // The index count is wrong...?
            // Maybe not.  See https://github.com/amethyst/rendy/issues/119
            let indices = 0..(draw_call.mesh.len() as u32);
            // This count is the *number of instances*.  What instance
            // to start at in the buffer is defined by the offset in
            // `bind_vertex_buffers()` above, and the stride/size of an instance
            // is defined in `AsVertex`.
            let instances = 0..(draw_call.objects.len() as u32);
            encoder.draw_indexed(indices, 0, instances);
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
    B: gfx_hal::Backend,
{
    objects: Vec<InstanceData>,
    mesh: Mesh<B>,
    texture: Texture<B>,
    /// We just need the actual config for the sampler 'cause
    /// Rendy's `Factory` can manage a sampler cache itself.
    sampler_info: SamplerInfo,
}

impl<B> DrawCall<B>
where
    B: gfx_hal::Backend,
{
    fn new(texture: Texture<B>, mesh: Mesh<B>) -> Self {
        let sampler_info = SamplerInfo::new(Filter::Nearest, WrapMode::Clamp);
        Self {
            objects: vec![],
            mesh,
            texture,
            sampler_info,
        }
    }

    fn add_object(&mut self, rng: &mut rand::rngs::ThreadRng, max_width: f32, max_height: f32) {
        let rx = Uniform::new(0.0, max_width);
        let ry = Uniform::new(0.0, max_height);
        let x = rx.sample(rng);
        let y = ry.sample(rng);
        println!("Adding object at {}x{}", x, y);
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

#[derive(Debug)]
struct Scene<B: gfx_hal::Backend> {
    camera: UniformData,
    object_mesh: Mesh<B>,
    objects: Vec<InstanceData>,
    texture: Texture<B>,
    texture2: Texture<B>,

    /// We just need the actual config for the sampler 'cause
    /// Rendy's `Factory` can manage a sampler cache itself.
    sampler_info: SamplerInfo,
}

impl<B> Scene<B>
where
    B: gfx_hal::Backend,
{
    fn add_object(&mut self, rng: &mut rand::rngs::ThreadRng, max_width: f32, max_height: f32) {
        let rx = Uniform::new(0.0, max_width);
        let ry = Uniform::new(0.0, max_height);

        if self.objects.len() < MAX_OBJECTS {
            let transform = Transform3::create_translation(rx.sample(rng), ry.sample(rng), -100.0);
            // println!("Transform: {:?}", transform);
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
struct Aux<B: gfx_hal::Backend> {
    frames: usize,
    align: u64,
    scene: Scene<B>,

    draws: Vec<DrawCall<B>>,
    camera: UniformData,
}

const MAX_OBJECTS: usize = 10_000;
const UNIFORM_SIZE: u64 = size_of::<UniformData>() as u64;
const INSTANCES_SIZE: u64 = size_of::<InstanceData>() as u64 * MAX_OBJECTS as u64;
const INDIRECT_SIZE: u64 = size_of::<DrawIndexedCommand>() as u64;

const fn buffer_frame_size(align: u64) -> u64 {
    ((UNIFORM_SIZE + INSTANCES_SIZE + INDIRECT_SIZE - 1) / align + 1) * align
}

const fn uniform_offset(index: usize, align: u64) -> u64 {
    buffer_frame_size(align) * index as u64
}

const fn instances_offset(index: usize, align: u64) -> u64 {
    uniform_offset(index, align) + UNIFORM_SIZE
}

const fn indirect_offset(index: usize, align: u64) -> u64 {
    instances_offset(index, align) + INSTANCES_SIZE
}

/// The size of the buffer in a `FrameInFlight`.
///
/// We pack data from multiple `DrawCall`'s together into one `Buffer`
/// but each draw call can have a varying amount of instance data,
/// so we end up with something like:
///
/// ```
/// |Uniforms | Instances1 ... | Instances2 ... | ... | empty space |
/// ```
///
/// Right now we have a fixed size buffer and just limit the number
/// of objects in it.  TODO: Eventually someday we will grow the buffer
/// as needed.  Maybe shrink it too?  Not sure about that.
/// ALSO TODO: Would probably be nicer to stick the uniforms into
/// push constants someday, but one thing at a time.
const fn gbuffer_size(align: u64) -> u64 {
    ((UNIFORM_SIZE + INSTANCES_SIZE - 1) / align + 1) * align
}

/// Offset of the uniforms section in the buffer.
const fn guniform_offset() -> u64 {
    0
}

/// The offset pointing to free space right after the given number
/// of instances.
///
/// TODO: Are there alignment requirements for this?
const fn ginstance_offset(instance_count: usize) -> u64 {
    guniform_offset() + (instance_count * size_of::<InstanceData>()) as u64
}

#[derive(Debug, Default)]
struct MeshRenderPipelineDesc;

#[derive(Debug)]
struct MeshRenderPipeline<B: gfx_hal::Backend> {
    // buffer: Escape<Buffer<B>>,
    // sets: Vec<Escape<DescriptorSet<B>>>,

    frames_in_flight: Vec<FrameInFlight<B>>,
}

impl<B> SimpleGraphicsPipelineDesc<B, Aux<B>> for MeshRenderPipelineDesc
where
    B: gfx_hal::Backend,
{
    type Pipeline = MeshRenderPipeline<B>;

    fn depth_stencil(&self) -> Option<gfx_hal::pso::DepthStencilDesc> {
        None
    }

    fn layout(&self) -> Layout {
        // TODO: Figure this stuff out.
        Layout {
            sets: vec![
                SetLayout {
                    bindings: vec![
                        gfx_hal::pso::DescriptorSetLayoutBinding {
                            binding: 0,
                            ty: gfx_hal::pso::DescriptorType::UniformBuffer,
                            count: 1,
                            stage_flags: gfx_hal::pso::ShaderStageFlags::GRAPHICS,
                            immutable_samplers: false,
                        },
                        gfx_hal::pso::DescriptorSetLayoutBinding {
                            binding: 1,
                            ty: gfx_hal::pso::DescriptorType::SampledImage,
                            count: 1,
                            stage_flags: gfx_hal::pso::ShaderStageFlags::FRAGMENT,
                            immutable_samplers: false,
                        },
                        gfx_hal::pso::DescriptorSetLayoutBinding {
                            binding: 2,
                            ty: gfx_hal::pso::DescriptorType::Sampler,
                            count: 1,
                            stage_flags: gfx_hal::pso::ShaderStageFlags::FRAGMENT,
                            immutable_samplers: false,
                        },
                    ],
                },
                SetLayout {
                    bindings: vec![
                        gfx_hal::pso::DescriptorSetLayoutBinding {
                            binding: 0,
                            ty: gfx_hal::pso::DescriptorType::UniformBuffer,
                            count: 1,
                            stage_flags: gfx_hal::pso::ShaderStageFlags::GRAPHICS,
                            immutable_samplers: false,
                        },
                        gfx_hal::pso::DescriptorSetLayoutBinding {
                            binding: 1,
                            ty: gfx_hal::pso::DescriptorType::SampledImage,
                            count: 1,
                            stage_flags: gfx_hal::pso::ShaderStageFlags::FRAGMENT,
                            immutable_samplers: false,
                        },
                        gfx_hal::pso::DescriptorSetLayoutBinding {
                            binding: 2,
                            ty: gfx_hal::pso::DescriptorType::Sampler,
                            count: 1,
                            stage_flags: gfx_hal::pso::ShaderStageFlags::FRAGMENT,
                            immutable_samplers: false,
                        },
                    ],
                },
            ],
            push_constants: Vec::new(),
        }
    }

    fn vertices(
        &self,
    ) -> Vec<(
        Vec<gfx_hal::pso::Element<gfx_hal::format::Format>>,
        gfx_hal::pso::ElemStride,
        gfx_hal::pso::InstanceRate,
    )> {
        vec![
            PosColorNorm::VERTEX.gfx_vertex_input_desc(0),
            InstanceData::VERTEX.gfx_vertex_input_desc(1),
        ]
    }

    fn load_shader_set<'a>(
        &self,
        storage: &'a mut Vec<B::ShaderModule>,
        factory: &mut Factory<B>,
        _aux: &Aux<B>,
    ) -> gfx_hal::pso::GraphicsShaderSet<'a, B> {
        storage.clear();

        log::trace!("Load shader module VERTEX");
        storage.push(unsafe { VERTEX.module(factory).unwrap() });

        log::trace!("Load shader module FRAGMENT");
        storage.push(unsafe { FRAGMENT.module(factory).unwrap() });

        gfx_hal::pso::GraphicsShaderSet {
            vertex: gfx_hal::pso::EntryPoint {
                entry: "main",
                module: &storage[0],
                specialization: gfx_hal::pso::Specialization::default(),
            },
            fragment: Some(gfx_hal::pso::EntryPoint {
                entry: "main",
                module: &storage[1],
                specialization: gfx_hal::pso::Specialization::default(),
            }),
            hull: None,
            domain: None,
            geometry: None,
        }
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
        assert_eq!(set_layouts.len(), 2);

        let (frames, align) = (aux.frames, aux.align);
/*
        let buffer = factory
            .create_buffer(
                BufferInfo {
                    size: buffer_frame_size(align) * frames as u64,
                    usage: gfx_hal::buffer::Usage::UNIFORM
                        | gfx_hal::buffer::Usage::INDIRECT
                        | gfx_hal::buffer::Usage::VERTEX,
                },
                Dynamic,
            )
            .unwrap();


        let sampler = factory.get_sampler(aux.scene.sampler_info.clone())?;
        let mut sets = Vec::new();
        for index in 0..frames {
            unsafe {
                {
                    let set = factory
                        .create_descriptor_set(set_layouts[0].clone())
                        .unwrap();

                    factory.write_descriptor_sets(Some(gfx_hal::pso::DescriptorSetWrite {
                        set: set.raw(),
                        binding: 0,
                        array_offset: 0,
                        descriptors: Some(gfx_hal::pso::Descriptor::Buffer(
                            buffer.raw(),
                            Some(uniform_offset(index, align))
                                ..Some(uniform_offset(index, align) + UNIFORM_SIZE),
                        )),
                    }));
                    factory.write_descriptor_sets(Some(gfx_hal::pso::DescriptorSetWrite {
                        set: set.raw(),
                        binding: 1,
                        array_offset: 0,
                        descriptors: vec![gfx_hal::pso::Descriptor::Image(
                            aux.scene.texture.view().raw(),
                            gfx_hal::image::Layout::ShaderReadOnlyOptimal,
                        )],
                    }));
                    factory.write_descriptor_sets(Some(gfx_hal::pso::DescriptorSetWrite {
                        set: set.raw(),
                        binding: 2,
                        array_offset: 0,
                        descriptors: vec![gfx_hal::pso::Descriptor::Sampler(sampler.raw())],
                    }));
                    sets.push(set);
                }

                // Second descriptor set...?
                {
                    let set = factory
                        .create_descriptor_set(set_layouts[1].clone())
                        .unwrap();

                    factory.write_descriptor_sets(Some(gfx_hal::pso::DescriptorSetWrite {
                        set: set.raw(),
                        binding: 0,
                        array_offset: 0,
                        descriptors: Some(gfx_hal::pso::Descriptor::Buffer(
                            buffer.raw(),
                            Some(uniform_offset(index, align))
                                ..Some(uniform_offset(index, align) + UNIFORM_SIZE),
                        )),
                    }));
                    factory.write_descriptor_sets(Some(gfx_hal::pso::DescriptorSetWrite {
                        set: set.raw(),
                        binding: 1,
                        array_offset: 0,
                        descriptors: vec![gfx_hal::pso::Descriptor::Image(
                            aux.scene.texture2.view().raw(),
                            gfx_hal::image::Layout::ShaderReadOnlyOptimal,
                        )],
                    }));
                    factory.write_descriptor_sets(Some(gfx_hal::pso::DescriptorSetWrite {
                        set: set.raw(),
                        binding: 2,
                        array_offset: 0,
                        descriptors: vec![gfx_hal::pso::Descriptor::Sampler(sampler.raw())],
                    }));

                    sets.push(set);
                }
            }
        }

*/
    // Okay, so, each `FrameInFlight` needs one descriptor set per draw call.


    let mut frames_in_flight = vec![];
    frames_in_flight.extend((0..frames).map(|_| FrameInFlight::new(factory, align, &aux.draws, &set_layouts[0])));

        Ok(MeshRenderPipeline { 
            //buffer, 
            // sets, 
            frames_in_flight, 
        })
    }
}

impl<B> SimpleGraphicsPipeline<B, Aux<B>> for MeshRenderPipeline<B>
where
    B: gfx_hal::Backend,
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
        let (scene, align) = (&aux.scene, aux.align);

        let layout = &set_layouts[0];
        self.frames_in_flight[index].prepare(factory, &aux.camera, &aux.draws, layout, align);
/*
        unsafe {
            factory
                .upload_visible_buffer(
                    &mut self.buffer,
                    uniform_offset(index, align),
                    &[UniformData {
                        proj: scene.camera.proj,
                        view: scene.camera.view.inverse().unwrap(),
                    }],
                )
                .unwrap()
        };

        let half_len = scene.objects.len() as u32;
        unsafe {
            factory
                .upload_visible_buffer(
                    &mut self.buffer,
                    indirect_offset(index, align),
                    &[
                        DrawIndexedCommand {
                            index_count: scene.object_mesh.len(),
                            instance_count: half_len,
                            first_index: 0,
                            vertex_offset: 0,
                            first_instance: 0,
                        },
                        DrawIndexedCommand {
                            index_count: scene.object_mesh.len(),
                            instance_count: half_len,
                            first_index: 0,
                            vertex_offset: 0,
                            first_instance: half_len,
                        },
                    ],
                )
                .unwrap()
        };

        if !scene.objects.is_empty() {
            unsafe {
                factory
                    .upload_visible_buffer(
                        &mut self.buffer,
                        instances_offset(index, align),
                        &scene.objects[..],
                    )
                    .unwrap()
            };
        }
*/
        PrepareResult::DrawReuse
    }

    fn draw(
        &mut self,
        layout: &B::PipelineLayout,
        mut encoder: RenderPassEncoder<'_, B>,
        index: usize,
        aux: &Aux<B>,
    ) {
        self.frames_in_flight[index].draw(&aux.draws, layout, &mut encoder, aux.align);
        /*
        let descriptor_set1 = vec![self.sets[0].raw()];
        let descriptor_set2 = vec![self.sets[3].raw()];
        let vertex_buffers = vec![
            (self.buffer.raw(), instances_offset(index, aux.align)),
            // (self.buffer.raw(), instances_offset(index, aux.align)),
        ];

        aux.scene
            .object_mesh
            .bind(&[PosColorNorm::VERTEX], &mut encoder)
            .expect("Could not bind mesh?");

        encoder.bind_vertex_buffers(
            1,
            // std::iter::once((self.buffer.raw(), instances_offset(index, aux.align))),
            vertex_buffers.into_iter(),
        );
        encoder.bind_graphics_descriptor_sets(
            layout,
            0,
            descriptor_set1.into_iter(),
            std::iter::empty(),
        );
        encoder.draw_indexed_indirect(
            self.buffer.raw(),
            indirect_offset(index, aux.align),
            1,
            INDIRECT_SIZE as u32,
        );
        // encoder.draw_indexed(
        //     0..6,
        //     0,
        //     0..(aux.scene.objects.len() as u32),
        // );

        encoder.bind_graphics_descriptor_sets(
            layout,
            0,
            descriptor_set2.into_iter(),
            std::iter::empty(),
        );
        encoder.draw_indexed_indirect(
            self.buffer.raw(),
            indirect_offset(index, aux.align) + INDIRECT_SIZE,
            1,
            INDIRECT_SIZE as u32,
        );
        */
    }

    fn dispose(self, _factory: &mut Factory<B>, _aux: &Aux<B>) {}
}

/// This is how we can load an image and create a new texture.
fn make_texture<B>(queue_id: QueueId, factory: &mut Factory<B>, image_bytes: &[u8]) -> Texture<B>
where
    B: gfx_hal::Backend,
{
    let texture_builder = rendy::texture::image::load_from_image(image_bytes, Default::default())
        .expect("Could not load texture?");

    let texture = texture_builder
        .build(
            ImageState {
                queue: queue_id,
                stage: gfx_hal::pso::PipelineStage::FRAGMENT_SHADER,
                access: gfx_hal::image::Access::SHADER_READ,
                layout: gfx_hal::image::Layout::ShaderReadOnlyOptimal,
            },
            factory,
        )
        .unwrap();
    texture
}

fn make_quad_mesh<B>(queue_id: QueueId, factory: &mut Factory<B>) -> Mesh<B>
where
    B: gfx_hal::Backend,
{
    let verts: Vec<[f32; 3]> = vec![
        [0.0, 0.0, 0.0],
        [0.0, 100.0, 0.0],
        [100.0, 100.0, 0.0],
        [0.0, 0.0, 0.0],
        [100.0, 100.0, 0.0],
        [100.0, 0.0, 0.0],
    ];
    let indices = rendy::mesh::Indices::from(vec![0u32, 1, 2, 3, 4, 5]);
    let vertices: Vec<_> = verts
        .into_iter()
        // TODO: Mesh color... how do we want to handle this?
        .map(|v| PosColorNorm {
            position: rendy::mesh::Position::from(v),
            color: [1.0, 1.0, 1.0, 1.0]
                // (v[0] + 1.0) / 2.0,
                // (v[1] + 1.0) / 2.0,
                // (v[2] + 1.0) / 2.0,
                //     1.0,
                // ]
                .into(),
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

    let surface = factory.create_surface(window.into());

    let mut graph_builder = GraphBuilder::<Backend, Aux<Backend>>::new();

    let color = graph_builder.create_image(
        surface.kind(),
        1,
        factory.get_surface_format(&surface),
        Some(gfx_hal::command::ClearValue::Color(
            [0.1, 0.2, 0.3, 1.0].into(),
        )),
    );

    // let depth = graph_builder.create_image(
    //     surface.kind(),
    //     1,
    //     gfx_hal::format::Format::D16Unorm,
    //     Some(gfx_hal::command::ClearValue::DepthStencil(
    //         gfx_hal::command::ClearDepthStencil(1.0, 0),
    //     )),
    // );

    let pass = graph_builder.add_node(
        MeshRenderPipeline::builder()
            .into_subpass()
            .with_color(color)
            // .with_depth_stencil(depth)
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

    let texture_1 = make_texture(queue_id, &mut factory, gfx_bytes);
    let texture_2 = make_texture(queue_id, &mut factory, rendy_bytes);
    let object_mesh1 = make_quad_mesh(queue_id, &mut factory);

    let sampler_info = SamplerInfo::new(Filter::Nearest, WrapMode::Clamp);
    let width = window_size.width as f32;
    let height = window_size.height as f32;
    println!("dims: {}x{}", width, height);
    let scene = Scene {
        // TODO: Make view and proj separate?  Maybe.  We should only need one.
        camera: UniformData {
            proj: Transform3::ortho(0.0, width, height, 0.0, 1.0, 200.0),

            view: Transform3::create_translation(0.0, 0.0, 10.0),
        },
        objects: vec![],
        object_mesh: object_mesh1,
        texture: texture_1,
        texture2: texture_2,

        sampler_info,
    };


    let texture1 = make_texture(queue_id, &mut factory, gfx_bytes);
    let texture2 = make_texture(queue_id, &mut factory, rendy_bytes);
    let object_mesh1 = make_quad_mesh(queue_id, &mut factory);
    // TODO: We should be able to share these, investigate further.
    let object_mesh2 = make_quad_mesh(queue_id, &mut factory);

    let align = factory
            .physical()
            .limits()
            .min_uniform_buffer_offset_alignment;

    let draws = vec![
        DrawCall::new(texture1, object_mesh1),
        DrawCall::new(texture2, object_mesh2),
    ];
    let mut aux = Aux {
        frames: frames as _,
        align,
        scene,

        draws,
        camera: UniformData {
            proj: Transform3::ortho(0.0, width, height, 0.0, 1.0, 200.0),

            view: Transform3::create_translation(0.0, 0.0, 10.0),
        },
    };

    let mut graph = graph_builder
        .with_frames_in_flight(frames)
        .build(&mut factory, &mut families, &aux)
        .unwrap();

    log::info!("{:#?}", aux.scene);

    let started = time::Instant::now();

    let mut frames = 0u64..;
    let mut rng = rand::thread_rng();

    let mut fpss = Vec::new();
    let mut checkpoint = started;
    let mut should_close = false;

    while !should_close && aux.scene.objects.len() < MAX_OBJECTS {
        let start = frames.start;
        let from = aux.scene.objects.len();
        for _ in &mut frames {
            factory.maintain(&mut families);
            event_loop.poll_events(|event| match event {
                Event::WindowEvent {
                    event: WindowEvent::CloseRequested,
                    ..
                } => should_close = true,
                _ => (),
            });
            graph.run(&mut factory, &mut families, &aux);

            let elapsed = checkpoint.elapsed();
            // aux.scene.add_object(&mut rng, width, height);
            aux.draws[0].add_object(&mut rng, width, height);
            aux.draws[1].add_object(&mut rng, width, height);

            if should_close
                || elapsed > std::time::Duration::new(5, 0)
                || aux.scene.objects.len() == MAX_OBJECTS
            {
                let frames = frames.start - start;
                let nanos = elapsed.as_secs() * 1_000_000_000 + elapsed.subsec_nanos() as u64;
                fpss.push((
                    frames * 1_000_000_000 / nanos,
                    from..aux.scene.objects.len(),
                ));
                checkpoint += elapsed;
                break;
            }
        }
    }

    log::info!("FPS: {:#?}", fpss);

    graph.dispose(&mut factory, &aux);
}
