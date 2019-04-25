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
