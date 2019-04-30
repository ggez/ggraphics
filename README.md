# ggraphics

Eventually, hopefully, a 2D graphics library using gfx-rs, to serve as a
basis for ggez as well as anything else that wants it.

Run with

> cargo run --features=vulkan

# Expected backends

Essentially dictated by gfx-hal, which is dictated by WebRender

 * OpenGL 3.2 or better
 * OpenGL ES 3.0
 * WebGL (version?)
 * Vulkan (version?)
 * Metal (version?)
 * DirectX 11 or better

# Notes and things to examine

 * https://github.com/omni-viral/rendy -- Amethyst's equivalent, might be better to just use that.
 * https://github.com/draw2d/rfcs/issues/1 and other content by that project
 * https://raphlinus.github.io/rust/graphics/2018/10/11/2d-graphics.html
 * https://nical.github.io/posts/rust-2d-graphics-01.html
 * https://nical.github.io/posts/rust-2d-graphics-02.html
 * https://github.com/Connicpu/direct2d-rs
 * Can we use webrender?  I doubt it, but it'd be interesting to examine.  Talk to kvark
   and nical about it.
 * Can we make a good particle system as part of this?  It sounds fun.  :D

# Design thoughts

See https://github.com/ggez/ggez/issues/418

## Notes on Rendy

I need to go through and document it good and hard.  :F

If factory::Factory is a higher level device interface, what's the
lower level one?

I'm not sure that `Config` being a collection of trait-y things
actually helps anything.  You're literally just selecting things from
a list.  Just select things from a list.

rendy::memory: What's the difference between LinearConfig and
DynamicConfig?

For our API, we want to be able to present the user with a list of
GPU's (with text strings) and have them choose one, or have the
ability to automatically choose one based on broad criteria
("performance", "low-power", "whatever looks best").

Init: We create a Factory, create a Winit window, then create a
Surface from the factory that gets our final render target.  This
should also be configurable, cause we might want our final render
target to be off screen!

Either way, once that's all set up we make our `Graph` stuff, which is
a chain of nodes.  The possible node types seem to be `DescBuilder`,
whatever that is, `RenderPassNodeBuilder`, and `PresentBuilder`.
RenderPass's contain Subpasses, which contain images, colors(?), depth
stencils (irrelevant), and RenderGroup's.  Not sure yet what a
RenderGroup represents.

Why does disposing of a graph involve `Graph::dispose(self, &mut
Factory, ...)` instead of `Factory::dispose(&mut self, Graph)`?  I
guess it comes down to the same thing, it's just weird.  Especially
since we have `Factory::destroy_semaphore(Semaphore)`.

In the sprite example we have `SpriteGraphicsPipeline` and
`SpriteGraphicsPipelineDesc`, which impl `SimpleGraphicsPipeline`
and `SimpleGraphicsPipelineDesc` respectively.  The `Desc` object
appears to create shaders, hand out shader layouts, and create shader
resources like textures(?).  The `SpriteGraphicsPipeline` then does
the actual drawing, but by that point it's just binding vertex buffers
and stuff and pushing the "go" button.

Why are there these `aux: &mut T` arguments everywhere?  They seem
reminiscent of Vulkan's various hooks.  Can we get rid of them?  Are
they actually used for anything?

`image::Filter` only has Nearest and Linear so far XD.  Well, that's
all I need, so...  However, there's also no mention of `blend` in the
docs.  Can we not set blending modes yet?  Texture address modes?
Oooooh, blending is in the Pipeline, as `colors` which has type
`Vec<ColorBlendDesc>` which is a `gfx_hal` construct and so doesn't
appear in the `rendy` doc search.

Okay, so this has definitely been a reality check for my nice simple
model vs. reality.  `rendy::graph::render::Pipeline` seems to be the
heart of how it all connects together, but the docs don't quite match
what's in git, and the simplified versions of it might not actually be
that simple, and it might not be directly what we want.  Either way:
it contains a `Layout` which connects our shader variables to the
whatever they're coming from -- buffers, textures+samplesr, etc.  It
also contains vertex info, in the form of vertex formats (offsets,
strides, etc).

So the hard part is going to be constructing a `Layout`, but once we
do it should probably serve just fine for ALL drawing with a
particular set of shaders.

The other hard thing is resource management, so.  As Rendy does it,
resources seem to be connected to a particular `Pipeline` instance?
That's not really how ggez does it, any resource may be visible from
any pipeline.  That may be more awkward since certain resources seem
to be specific to particular pipeline's or queues.  Hmmmm.



Okay, so we're going to be drawing with
rendy::command::DrawIndexedCommand.  This deals with two buffers, one
of instance data and one of geometry data, though only indirectly
through descriptors of some kind.  This takes three offset+length's
into those buffers: uniform, vertex, and instance starts.  Great, that
lets us have our conceptual Geometry and DrawParam types, and globals
like projection and such go into a global uniform buffer.  So far so
good.

We then have different descriptor sets or something like that for each
Geometry, keep track of which one we're currently using, and only
switch when we change Geometry?  100% sure how that all fits together
yet; Rendy does some of the stuff for you but it's kinda hidden.  But
that model basically gives us automatic sprite batching, as well as
automatic batching of whatever other geometry we happen to end up
using, so we can give sprites an arbitrary Mesh and it will be just as
fast as drawing quads.  Sauce.

We ARE going to have to do memory management of instance data, it
looks like.  This should be fairly simple since we just have to fill a
buffer up to a point, and expand the buffer when necessary.
Contracting the buffer if it gets small MIGHT be useful.  Or you can
provide a hint to the renderer or whatever someday; for now just
expand it if it runs out and don't sweat it otherwise.  Rendy probably
can help with this but idk how yet.

Okay, how do we fit textures in with instanced drawing?  Looks easy
actually; a texture and sampler get put into the `SetLayout`.  Then
changing those is basically the same as changing the geometry.  Looks
like rendy mostly just handles samplers for us?  Make sure.  Either
way, since the sampler settings we are actually interested in are
discrete, there's realistically a fairly limited number of samplers
that may get used.  We can can maybe just create all the ones possible
ahead of time, if necessary.  Samplers and textures are passed to the
shader in uniforms, basically like anything else.

Need to wrap my mind around how instanced vs. vertex/pixel data works
in shaders more.  I'm just shaky on which is specified how.

Blending is controlled by SimpleGraphicsPipelineDesc::colors(). I'm
not sure exactly where render targets happen yet but it appears to be
in the Graph, which is several levels above the Pipeline; it looks as
though it's something like Graph's contain Pass's which contain
SubPass's which contain Pipeline's.  Each `RenderPassNodeBuilder` just
takes subpass's, and a single subpass can be trivially turned into a
pass containing just that.  It looks like the `SubpassBuilder` takes
color and depth buffers (as inputs or outputs???) as well as some
dependency info that the `Graph` uses to order its various `Node`s
appropriately(?).  

ooooooh, color and depth buffers are created by
`GraphBuilder::create_image()` and then handed to a subpass with
`with_color()` and such.  That subpass is created from your
`FooPipeline` that impl's `SimpleGraphicsPipeline`  Then the
`PresentNode` connects the color buffer up to the `surface` that
actually gets presented.

example from `meshes`:

```rust
    let mut graph_builder = GraphBuilder::<Backend, Aux<Backend>>::new();

    let color = graph_builder.create_image(
        surface.kind(),
        1,
        factory.get_surface_format(&surface),
        Some(gfx_hal::command::ClearValue::Color(
            [1.0, 1.0, 1.0, 1.0].into(),
        )),
    );

    let pass = graph_builder.add_node(
        MeshRenderPipeline::builder()
            .into_subpass()
            .with_color(color)
            .into_pass(),
    );

    let present_builder = PresentNode::builder(&factory, surface, color).with_dependency(pass);

    graph_builder.add_node(present_builder);
```

This code is kinda hacked up, don't trust it to be 100% correct.  Also
just trivially omitting a depth buffer causes a crash somewhere, so.

Is this a reasonable place to ask for feedback with how to design stuff with Rendy, or should I just bug their issue tracker directly?
So I don't forget what I'm talking about while making lasagna, I am trying to make a setup that just renders many quads with one texture, based off the meshes example, and I'm getting some dependency inversion stuffs.  If I load a Texture inside a PipelineDesc it feels weird 'cause then textures can't be shared between them, but if I try to create it outside it I need to be able to get my hands on the sampler and imageview for the texture to pass it into the pipeline's build() method.
FriziToday at 6:15 PM
you should use TextureBuilder to build a texture outside and slap it into your aux type
in amethyst_rendy, we basically store those Texture<B> as asset in asset storage, then reference them on ecs via handles. That way I can query the world for objects which can share textures, then dereference and bind those textures in the render pass.
I don't want to spoil the fun, we have that one already implemented. If all you need is a working implementation of batching sprite renderer, go ahead and use it.
unless you need that outside of amethyst that is :wink:
icefoxToday at 6:27 PM
that's what I'm more or less trying to do but I need the Graph to build the texture and properties from the texture to build the Graph node
I think.  Let me look at it more.
Yeah, TextureBuilder::build() takes an ImageState which needs a NodeId which comes from graph.node_queue(pass)
so the Graph has to be built before the texture
and GraphBuilder::build() calls my SimpleGraphicsPipelineDesc::build() which sets up the descriptor sets, and Descriptor::Image()needs the ViewKind, which looks like it's just plain ol data
but Descriptor::Sampler() needs a Sampler which is a handle you need to get from somewhere.
And this is for a Rendy graphics backend for ggez, so depending on Amethyst would be hilarious but not what I want.  :stuck_out_tongue:
Anyway, I'm going a little by voodoo at the moment still, so there might be a way to reorder that.  Maybe I should dig into Amethyst... but I"m already dealing with the inner workings of two poorly-documented and in-flux codebases so I don't really want to start trying to figure out a third one.
FriziToday at 7:06 PM
ImageState doesn't need NodeId for anything
you only need queue id and image layout
ImageState::new
icefoxToday at 7:07 PM
queue id might be what I'm thinknig of then...
let me look at that.
looks like the main way you get a QueueId is from Graph::node_queue though?
FriziToday at 7:08 PM
yeah, that one is a bit tough to be foolproof. Currently i just
hardcoded queueid 0 :/
```
        let queue_id = QueueId {
            family: self.families.as_mut().unwrap().family_by_index(0).id(),
            index: 0,
        };
```
icefoxToday at 7:09 PM
hmmmm
FriziToday at 7:09 PM
where system has families: Option<Families<B>>
it's not ideal ofc
but we really only use this queue for everything involving textures anyway
and there is currently no way of knowing how else it would be used. It's just an assumption that textures loaded as assets are always for queue 0 of family 0
you can still do a queue transition if needed later, so it's not that big of a deal
icefoxToday at 7:10 PM
Aha.
I will try that then, thank you.

Okay, so of course, handling descriptor sets is a little more
complicated than I had thought.  From termhn:

from omniviral:

 * If resources are changing from one frame to the next, you basically
   need one copy of them per frame-in-flight.
 * "Data from uniform/storage buffer can be accessed by device any
   time between moment set is bound and last command recorded to that
   buffer, and also between command buffers submission and fence
   submitted with it or later signalled.  ie at time of command
   recording and execution.  Which means you should use separate
   buffer ranges for data that gets updated each frame because
   multiple frames can be in flight.  That's what frame indices serve
   for.  They help you choose unused buffer range to write data
   into".  THIS EXPLAINS A LOT ABOUT THE MESH EXAMPLE.  "Each time
   particular index received, rendy guarantee that commands recorded
   last time this index was received are complete.  Which means you
   can safely access resources used previously with this index."
 * Again per omniviral, we can have one set of buffers/descriptors per
   frame-in-flight but it 
 * In ggez's case, meshes and textures are not going to change, but
   uniforms and instance data will.  Those aren't really in the main
   buffer anyway, so.

Okay, so for each frame in flight we need:

 * A buffer (uniforms, instances, maybe indirect draw info)
 * A descriptor set
 * Some way of knowing what mesh, texture and sampler we're using.

For each different thing we're drawing, we need:

 * A mesh, a texture, a sampler, a DescriptorSetLayout, and a
   descriptor set matchin the layout.
 * 

Read-only data (meshes, textures, samplers, descriptor set layouts) can be shared between
frames in flight.  R/W data (buffers, descriptor sets) cannot.  

Switching shaders involves switching pipelines, which we're not doing
yet.

We have three places to do stuff:
`SimpleGraphicsPipelineDesc::build()` which is called when the graph
node is built.  (Other methods on it are as well but they mostly seem
to return read-only data.)  `SimpleGraphicsPipeline::prepare()` is
called at the beginning of a frame before any drawing is done.  It's
intended to do things like upload instance/uniform data per frame.  And
`SimpleGraphicsPipeline::draw()` does the actual drawing.

Do we want to actually used `draw_indexed_indirect()`?  It may or may
not make life simpler.  It appears to basically walk down a list of
`draw_indexed` commands and execute each one.  However those are
stored in the buffer along with instance and uniform data, so it's one
more thing to manage.  Hmm, well, if we had multiple pieces of
geometry that needed to be drawn per `InstanceData` that would be
great, but we *don't*, so `draw_indirect()` seems like it would be
just fine.

For reference:

> Rendy provides `Graph` abstraction.
> `Graph` is made of `Node`s
> Some nodes can be `RenderPass`'s
> `RenderPass`'s are made of `Subpass`'s
> `Subpass`'s are made of `RenderGroups`s
> And finally `RenderGroup`s can be made of one or multiple pipelines.
> And for simple case `SimpleRenderingPipeline` can serve as `RenderGroup`

And, again from omniviral:

> Note that SimpleGraphicsPipeline is called Simple because it is simplifiction, good for learning, PoCs and stuff like that.
>RenderGroup should be used for anything serious in mind.

> For instance you would like to create pipelines on the fly which is not feasible with SimpleGraphicsPipeline as you'd have rebuild whole graph to insert new one.
> Also RenderGroup would allow to conveniently share state between
> pipelines.

Sooooo.  Looks like `SimpleGraphicsPipeline` and
`SimpleGraphicsPipelineDesc` are implemented through
`SimpleRenderGroup` and `SimpleRenderGroupDesc`.  Those don't look too
complex.  The main differences seem to be:

 * Slightly simplified inputs
 * A few shortcuts
 * A little extra mongling of pipelines and subpasses that I don't yet
   fully understand.

# Outstanding questions

Depth buffer???

Selectable backends

