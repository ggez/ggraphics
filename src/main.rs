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


/// ggez's input draw params.  This just gets turned into
/// an InstanceData but it's good to keep in mind what we're
/// doing here.
#[derive(Clone, Copy, Debug)]
pub struct DrawParam {
    pub src: Rect,
    pub dest: Point2,
    pub rotation: f32,
    pub scale: Vector2,
    pub offset: Point2,
    pub color: Color,
}

/// Data we need per instance.  DrawParam gets turned into this.
/// We have to be *quite particular* about layout since this gets
/// fed straight to the shader.
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug)]
struct InstanceData {
    transform: Transform3,
    src: Rect,
    color: Color,
}

/// Uniform data.  Each Scene contains one of these.
#[derive(Clone, Copy)]
#[repr(C, align(16))]
struct UniformArgs {
    proj: Transform3,
    view: Transform3,
}

/// Basically UniformArgs
#[derive(Debug)]
struct Camera {
    view: Transform3,
    proj: Transform3,
}

#[derive(Debug)]
struct Scene<B: gfx_hal::Backend> {
    camera: Camera,
    object_mesh: Mesh<B>,
    objects: Vec<InstanceData>,
    texture: Texture<B>,

    /// We just need the actual config for the sampler 'cause
    /// Rendy's `Factory` can manage a sampler cache itself.
    sampler_info: SamplerInfo,
}

impl<B> Scene<B> where B: gfx_hal::Backend {
    fn add_object(&mut self, rng: &mut rand::rngs::ThreadRng) {
        let rxy = Uniform::new(-1.0, 1.0);
        let rz = Uniform::new(0.0, 185.0);

        if self.objects.len() < MAX_OBJECTS {
                let z = rz.sample(rng);
                 let transform = Transform3::create_translation(
                    rxy.sample(rng) * (z / 2.0 + 4.0),
                    rxy.sample(rng) * (z / 2.0 + 4.0),
                    -z,
                    );

                /*
                let transform = Transform3::create_translation(
                    rxy.sample(rng),
                    rxy.sample(rng),
                    -50.0,
                );
            let transform = Transform3::create_translation(
                rxy.sample(rng) * 1000.0,
                rxy.sample(rng) * 1000.0,
                100.0,
            );
            */
                println!("Transform: {:?}", transform);
                    let src = Rect::from(
                        euclid::Size2D::new(1.0, 1.0)
                    );
                    let color = [1.0, 0.0, 1.0, 1.0];
                    let instance = InstanceData {
                        transform,
                        src,
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
    }

    const MAX_OBJECTS: usize = 10_000;
    const UNIFORM_SIZE: u64 = size_of::<UniformArgs>() as u64;
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

    #[derive(Debug, Default)]
    struct MeshRenderPipelineDesc;

    #[derive(Debug)]
    struct MeshRenderPipeline<B: gfx_hal::Backend> {
        buffer: Escape<Buffer<B>>,
        sets: Vec<Escape<DescriptorSet<B>>>,
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
            }

            Ok(MeshRenderPipeline { buffer, sets })
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
                            proj: scene.camera.proj,
                            view: scene.camera.view.inverse().unwrap(),
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
                            index_count: scene.object_mesh.len(),
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
                            instances_offset(index, align),
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
            aux.scene
                .object_mesh
                .bind(&[PosColorNorm::VERTEX], &mut encoder)
                .expect("Could not bind mesh?");

            encoder.bind_vertex_buffers(
                1,
                std::iter::once((self.buffer.raw(), instances_offset(index, aux.align))),
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
            [0.0, 0.0, 0.5],
            [0.0, 10.0, 0.5],
            [10.0, 10.0, 0.5],
            [0.0, 0.0, 0.5],
            [10.0, 10.0, 0.5],
            [10.0, 0.0, 0.5],
        ];
        let indices = rendy::mesh::Indices::from(vec![0u32, 1, 2, 3, 4, 5]);
        let vertices: Vec<_> = verts
            .into_iter()
            // TODO: Mesh color... how do we want to handle this?
            .map(|v| PosColorNorm {
                position: rendy::mesh::Position::from(v),
                color: [
                    (v[0] + 1.0) / 2.0,
                    (v[1] + 1.0) / 2.0,
                    (v[2] + 1.0) / 2.0,
                    1.0,
                ]
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
        let window_size = window.get_inner_size().unwrap().to_physical(window.get_hidpi_factor());

        event_loop.poll_events(|_| ());

        let surface = factory.create_surface(window.into());
        let aspect = surface.aspect();

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

        let rendy_bytes = include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/data/rendy_logo.png"));
        let gfx_bytes = include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/data/gfx_logo.png"));
        
        let texture = make_texture(queue_id, &mut factory, rendy_bytes);
        let _texture2 = make_texture(queue_id, &mut factory, gfx_bytes);
        let object_mesh = make_quad_mesh(queue_id, &mut factory);

        let sampler_info = SamplerInfo::new(Filter::Linear, WrapMode::Clamp);
        let width = window_size.width as f32;
        let height = window_size.height as f32;
        println!("dims: {}x{}", width, height);
        let scene = Scene {
            // TODO: Make view and proj separate?  Maybe.  We should only need one.
            camera: Camera {
            proj: Transform3::ortho(-100.0, 100.0, -100.0, 100.0, 1.0, 200.0),

                view: Transform3::create_translation(0.0, 0.0, 10.0),
            },
            objects: vec![],
            object_mesh,
            texture,

            sampler_info,
        };

        let mut aux = Aux {
            frames: frames as _,
            align: factory
                .physical()
                .limits()
                .min_uniform_buffer_offset_alignment,
            scene,
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
                aux.scene.add_object(&mut rng);

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
