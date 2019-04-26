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

//!
//! The mighty triangle example.
//! This examples shows colord triangle on white background.
//! Nothing fancy. Just prove that `rendy` works.
//!

#![cfg_attr(
    not(any(feature = "dx12", feature = "metal", feature = "vulkan")),
    allow(unused)
)]
use {
    gfx_hal::PhysicalDevice as _,
    rendy::{
        command::{DrawIndexedCommand, QueueId, RenderPassEncoder},
        factory::{Config, Factory, ImageState},
        graph::{
            present::PresentNode, render::*, GraphBuilder, GraphContext, NodeBuffer, NodeImage,
        },
        hal::Device as _,
        hal as gfx_hal,
        memory::Dynamic,
        mesh::{AsVertex, Mesh, PosColorNorm, Transform},
        resource::{Buffer, BufferInfo, DescriptorSet, DescriptorSetLayout, Escape, Handle},
        shader::{Shader, ShaderKind, SourceLanguage, SpirvShader, StaticShaderInfo},
        texture::Texture,
    },
};

use std::{cmp::min, mem::size_of, time};


use rand::distributions::{Distribution, Uniform};

use winit::{Event, EventsLoop, WindowBuilder, WindowEvent};

#[cfg(feature = "dx12")]
type Backend = rendy::dx12::Backend;

#[cfg(feature = "metal")]
type Backend = rendy::metal::Backend;

#[cfg(feature = "vulkan")]
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

#[derive(Clone, Copy, Debug)]
#[repr(C, align(16))]
struct Light {
    pos: nalgebra::Vector3<f32>,
    pad: f32,
    intencity: f32,
}

#[derive(Clone, Copy)]
#[repr(C, align(16))]
struct UniformArgs {
    proj: nalgebra::Matrix4<f32>,
    view: nalgebra::Matrix4<f32>,
    lights_count: i32,
    pad: [i32; 3],
    lights: [Light; MAX_LIGHTS],
}

#[derive(Debug)]
struct Camera {
    view: nalgebra::Projective3<f32>,
    proj: nalgebra::Matrix4<f32>,
}

#[derive(Debug)]
struct Scene<B: gfx_hal::Backend> {
    camera: Camera,
    object_mesh: Option<Mesh<B>>,
    objects: Vec<nalgebra::Transform3<f32>>,
    lights: Vec<Light>,
}

#[derive(Debug)]
struct Aux<B: gfx_hal::Backend> {
    frames: usize,
    align: u64,
    scene: Scene<B>,
}

const MAX_LIGHTS: usize = 32;
const MAX_OBJECTS: usize = 10_000;
const UNIFORM_SIZE: u64 = size_of::<UniformArgs>() as u64;
const TRANSFORMS_SIZE: u64 = size_of::<Transform>() as u64 * MAX_OBJECTS as u64;
const INDIRECT_SIZE: u64 = size_of::<DrawIndexedCommand>() as u64;

const fn buffer_frame_size(align: u64) -> u64 {
    ((UNIFORM_SIZE + TRANSFORMS_SIZE + INDIRECT_SIZE - 1) / align + 1) * align
}

const fn uniform_offset(index: usize, align: u64) -> u64 {
    buffer_frame_size(align) * index as u64
}

const fn transforms_offset(index: usize, align: u64) -> u64 {
    uniform_offset(index, align) + UNIFORM_SIZE
}

const fn indirect_offset(index: usize, align: u64) -> u64 {
    transforms_offset(index, align) + TRANSFORMS_SIZE
}

#[derive(Debug, Default)]
struct MeshRenderPipelineDesc;

#[derive(Debug)]
struct MeshRenderPipeline<B: gfx_hal::Backend> {
    texture: Texture<B>,
    buffer: Escape<Buffer<B>>,
    sets: Vec<Escape<DescriptorSet<B>>>,
}

impl<B> SimpleGraphicsPipelineDesc<B, Aux<B>> for MeshRenderPipelineDesc
where
    B: gfx_hal::Backend,
{
    type Pipeline = MeshRenderPipeline<B>;

    
    /*
    fn depth_stencil(&self) -> Option<gfx_hal::pso::DepthStencilDesc> {
        None
    }
    */

    fn layout(&self) -> Layout {
        Layout {
            sets: vec![SetLayout {
                bindings: vec![
                    gfx_hal::pso::DescriptorSetLayoutBinding {
                        binding: 0,
                        ty: gfx_hal::pso::DescriptorType::UniformBuffer,
                        count: 1,
                        stage_flags: gfx_hal::pso::ShaderStageFlags::GRAPHICS,
                        immutable_samplers: false,
                    },
                    // ADDED
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
            }],
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
            Transform::VERTEX.gfx_vertex_input_desc(1),
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
        queue: QueueId,
        aux: &Aux<B>,
        buffers: Vec<NodeBuffer>,
        images: Vec<NodeImage>,
        set_layouts: &[Handle<DescriptorSetLayout<B>>],
    ) -> Result<MeshRenderPipeline<B>, failure::Error> {
        assert!(buffers.is_empty());
        assert!(images.is_empty());
        assert_eq!(set_layouts.len(), 1);

        let (frames, align) = (aux.frames, aux.align);

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

        // This is how we can load an image and create a new texture.
        let image_bytes = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/src/data/logo.png"
        ));

        let texture_builder =
            rendy::texture::image::load_from_image(image_bytes, Default::default())?;

        let texture = texture_builder
            .build(
                ImageState {
                    queue,
                    stage: gfx_hal::pso::PipelineStage::FRAGMENT_SHADER,
                    access: gfx_hal::image::Access::SHADER_READ,
                    layout: gfx_hal::image::Layout::ShaderReadOnlyOptimal,
                },
                factory,
            )
            .unwrap();


        let mut sets = Vec::new();
        for index in 0..frames {
            unsafe {
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
                // ADDED
                factory.write_descriptor_sets(Some(gfx_hal::pso::DescriptorSetWrite {
                    set: set.raw(),
                    binding: 1,
                    array_offset: 0,
                    descriptors: vec![gfx_hal::pso::Descriptor::Image(
                        texture.view().raw(),
                        gfx_hal::image::Layout::ShaderReadOnlyOptimal,
                    )],
                }));
                factory.write_descriptor_sets(Some(gfx_hal::pso::DescriptorSetWrite {
                    set: set.raw(),
                    binding: 2,
                    array_offset: 0,
                    descriptors: vec![gfx_hal::pso::Descriptor::Sampler(texture.sampler().raw())],
                }));

                sets.push(set);
            }
        }

        Ok(MeshRenderPipeline { texture, buffer, sets })
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
        _set_layouts: &[Handle<DescriptorSetLayout<B>>],
        index: usize,
        aux: &Aux<B>,
    ) -> PrepareResult {
        let (scene, align) = (&aux.scene, aux.align);

        unsafe {
            factory
                .upload_visible_buffer(
                    &mut self.buffer,
                    uniform_offset(index, align),
                    &[UniformArgs {
                        pad: [0, 0, 0],
                        proj: scene.camera.proj,
                        view: scene.camera.view.inverse().to_homogeneous(),
                        lights_count: scene.lights.len() as i32,
                        lights: {
                            let mut array = [Light {
                                pad: 0.0,
                                pos: nalgebra::Vector3::new(0.0, 0.0, 0.0),
                                intencity: 0.0,
                            }; MAX_LIGHTS];
                            let count = min(scene.lights.len(), 32);
                            array[..count].copy_from_slice(&scene.lights[..count]);
                            array
                        },
                    }],
                )
                .unwrap()
        };

        unsafe {
            factory
                .upload_visible_buffer(
                    &mut self.buffer,
                    indirect_offset(index, align),
                    &[DrawIndexedCommand {
                        index_count: scene.object_mesh.as_ref().unwrap().len(),
                        instance_count: scene.objects.len() as u32,
                        first_index: 0,
                        vertex_offset: 0,
                        first_instance: 0,
                    }],
                )
                .unwrap()
        };

        if !scene.objects.is_empty() {
            unsafe {
                factory
                    .upload_visible_buffer(
                        &mut self.buffer,
                        transforms_offset(index, align),
                        &scene.objects[..],
                    )
                    .unwrap()
            };
        }

        PrepareResult::DrawReuse
    }

    fn draw(
        &mut self,
        layout: &B::PipelineLayout,
        mut encoder: RenderPassEncoder<'_, B>,
        index: usize,
        aux: &Aux<B>,
    ) {
        encoder.bind_graphics_descriptor_sets(
            layout,
            0,
            Some(self.sets[index].raw()),
            std::iter::empty(),
        );
        assert!(aux
            .scene
            .object_mesh
            .as_ref()
            .unwrap()
            .bind(&[PosColorNorm::VERTEX], &mut encoder)
            .is_ok());
        encoder.bind_vertex_buffers(
            1,
            std::iter::once((self.buffer.raw(), transforms_offset(index, aux.align))),
        );
        encoder.draw_indexed_indirect(
            self.buffer.raw(),
            indirect_offset(index, aux.align),
            1,
            INDIRECT_SIZE as u32,
        );
    }

    fn dispose(self, _factory: &mut Factory<B>, _aux: &Aux<B>) {}
}

#[cfg(any(feature = "dx12", feature = "metal", feature = "vulkan"))]
fn main() {

    let config: Config = Default::default();

    let (mut factory, mut families): (Factory<Backend>, _) = rendy::factory::init(config).unwrap();

    let mut event_loop = EventsLoop::new();

    let window = WindowBuilder::new()
        .with_title("Rendy example")
        .build(&event_loop)
        .unwrap();

    event_loop.poll_events(|_| ());

    let surface = factory.create_surface(window.into());
    let aspect = surface.aspect();

    let mut graph_builder = GraphBuilder::<Backend, Aux<Backend>>::new();

    let color = graph_builder.create_image(
        surface.kind(),
        1,
        factory.get_surface_format(&surface),
        Some(gfx_hal::command::ClearValue::Color(
            [1.0, 1.0, 1.0, 1.0].into(),
        )),
    );

    let depth = graph_builder.create_image(
        surface.kind(),
        1,
        gfx_hal::format::Format::D16Unorm,
        Some(gfx_hal::command::ClearValue::DepthStencil(
            gfx_hal::command::ClearDepthStencil(1.0, 0),
        )),
    );

    let pass = graph_builder.add_node(
        MeshRenderPipeline::builder()
            .into_subpass()
            .with_color(color)
            .with_depth_stencil(depth)
            .into_pass(),
    );

    let present_builder = PresentNode::builder(&factory, surface, color).with_dependency(pass);

    let frames = present_builder.image_count();

    graph_builder.add_node(present_builder);

    let scene = Scene {
        camera: Camera {
            // proj: nalgebra::Matrix4::new_perspective(aspect, 3.1415 / 4.0, 1.0, 200.0),
            proj: nalgebra::Matrix4::new_orthographic(-100.0, 100.0, -100.0, 100.0, 1.0, 200.0),
            view: nalgebra::Projective3::identity() * nalgebra::Translation3::new(0.0, 0.0, 10.0),
        },
        object_mesh: None,
        objects: vec![],
        lights: vec![
            Light {
                pad: 0.0,
                pos: nalgebra::Vector3::new(0.0, 0.0, 0.0),
                intencity: 10.0,
            },
            Light {
                pad: 0.0,
                pos: nalgebra::Vector3::new(0.0, 20.0, -20.0),
                intencity: 140.0,
            },
            Light {
                pad: 0.0,
                pos: nalgebra::Vector3::new(-20.0, 0.0, -60.0),
                intencity: 100.0,
            },
            Light {
                pad: 0.0,
                pos: nalgebra::Vector3::new(20.0, -30.0, -100.0),
                intencity: 160.0,
            },
        ],
    };

    let mut aux = Aux {
        frames: frames as _,
        align: factory
            .physical()
            .limits()
            .min_uniform_buffer_offset_alignment,
        scene,
    };

    log::info!("{:#?}", aux.scene);

    let mut graph = graph_builder
        .with_frames_in_flight(frames)
        .build(&mut factory, &mut families, &aux)
        .unwrap();

/*
    let icosphere = genmesh::generators::IcoSphere::subdivide(4);
    let indices: Vec<_> = genmesh::Vertices::vertices(icosphere.indexed_polygon_iter())
        .map(|i| i as u32)
        .collect();
    let vertices: Vec<_> = icosphere
        .shared_vertex_iter()
        .map(|v| PosColorNorm {
            position: v.pos.into(),
            color: [
                (v.pos.x + 1.0) / 2.0,
                (v.pos.y + 1.0) / 2.0,
                (v.pos.z + 1.0) / 2.0,
                1.0,
            ]
            .into(),
            normal: v.normal.into(),
        })
        .collect();

    aux.scene.object_mesh = Some(
        Mesh::<Backend>::builder()
            .with_indices(&indices[..])
            .with_vertices(&vertices[..])
            .build(graph.node_queue(pass), &factory)
            .unwrap(),
    );
    */
        let verts: Vec<[f32;3]> = vec![
        [0.0, 0.0, 0.5],
        [0.0, 10.0, 0.5],
        [10.0, 10.0, 0.5],
        [0.0, 0.0, 0.5],
        [10.0, 10.0, 0.5],
        [10.0, 0.0, 0.5],
    ];
    let indices = rendy::mesh::Indices::from( vec![
        0u32, 1, 2, 3, 4, 5
        ]);
    let vertices: Vec<_> = verts
        .into_iter()
        .map(|v| PosColorNorm {
            position: rendy::mesh::Position::from(v),
            color: [
                (v[0] + 1.0) / 2.0,
                (v[1] + 1.0) / 2.0,
                (v[2] + 1.0) / 2.0,
                1.0,
            ]
                .into(),
            // TODO: Double-check this; z goes into screen, right?
            normal: rendy::mesh::Normal::from([0.0, 0.0, 1.0])
        })
        .collect();

    aux.scene.object_mesh = Some(
        Mesh::<Backend>::builder()
            .with_indices(indices)
            .with_vertices(&vertices[..])
            .build(graph.node_queue(pass), &factory)
            .unwrap(),
    );


    let started = time::Instant::now();

    let mut frames = 0u64..;
    let mut rng = rand::thread_rng();
    let rxy = Uniform::new(-1.0, 1.0);
    let rz = Uniform::new(0.0, 185.0);

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

            if aux.scene.objects.len() < MAX_OBJECTS {
                    let z = rz.sample(&mut rng);
                    let trans = nalgebra::Transform3::identity()
                        * nalgebra::Translation3::new(
                            rxy.sample(&mut rng) * (z / 2.0 + 4.0),
                            rxy.sample(&mut rng) * (z / 2.0 + 4.0),
                            -z,
                        );
                        println!("Creating scene object at {:?}", trans);
                aux.scene.objects.push(trans);
            }

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

#[cfg(not(any(feature = "dx12", feature = "metal", feature = "vulkan")))]
fn main() {
    panic!("Specify feature: { dx12, metal, vulkan }");
}
