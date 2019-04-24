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

We should use [wgpu](https://crates.io/crates/wgpu) on the assumption that browsers will implement it
someday, and that will then let us finally Actually Run Stuff In Browser.

Or, you know, if we want to actually run stuff in browser RIGHT NOW, we should use OpenGL... hmmm....

Well, most of this isn't really specific to a particular backend, more thinking about how things
SHOULD be organized.

From ggez's perspective, our options for what we can draw basically are:

 * A single quad with an image
 * A list of quads, each with an image attached and 
 * A list of arbitrary geometry, each with an image attached.

We also have a pile of other state that affects how these are drawn:

 * Drawparam (transform, source UV info, etc.)
 * A canvas (render target).
 * A shader.
 * A bunch of random other state like projection, screen size, transform stack, etc.
 * Glyph-brush image cache

The way ggez does it is super stateful: you set canvas and shader, then you draw a piece of geometry
with an image.  Or many pieces of geometry with the same image.  Either way.

So if we were going to refactor this a little, our types might be

 * Geometry
 * Texture
 * Drawparam
 * Image: geometry+texture
 * Mesh: geometry+texture
 * Spritebatch: geometry+texture+drawparams

So really, our fundamentals are geometry, texture and drawparam.  The differences are mainly in terms
of count and staticness: All Image's have the same geometry, all Mesh's have different geometry, all
members in a Spritebatch have the same geometry and texture, etc.

Most of ggez's geometry is quads.  Is this common enough to be special cased in the actual drawing?
Not sure.

Write usage case first: https://caseymuratori.com/blog_0025 , really ggez's actual drawing API is
quite good for its purpose.  It's basically a list of draw commands.  Or, if shaders or canvases
change, it's a list of `(shader, canvas, draw_commands)`, where each draw command is a list of
`(geometry, texture, drawparam)`.  Soooo, each of those former things is kinda like a Vulkan pipeline.
Ok, not sure where Vulkan render targets happen (I think the relevant type is vkImage though).
However it works, our "pipeline" now looks more or less like this.

```rust
// This is illustrative of how things relate to each other,
// not intended to be how things will actually be implemented.
struct GlobalData {
    geometry: HashMap<VertexIdx, (VertexBuffer, IndexBuffer)>,
    textures: HashMap<TextureIdx, Texture>,
    uniforms: HashMap<UniformIdx, DrawParam>,
}

struct DrawCall {
    geometry: VertexIdx,
    texture: TextureIdx,
    drawparam: UniformIdx,

    // plus...
    sampler: ...,
    shader_instance_parameters: ...,
}

struct Pipeline {
    shaders: Shaders,
    render_target: Canvas,
    commands: Vec<DrawCall>,

    // plus...
    shader_uniform_parameters: ...,
}

struct OtherStuff {
    // plus...
    projection: Matrix4,
    glyph_brush_stuff: ...,
}
```

90% of the time we're only ever using one pipeline.

Some tricky parts: Texture properties (samplers?), shader parameters, operations like texture copies
(used in screenshots, gfx-glyph, etc), not-quite-constrained-to-one-draw-call things like blend
modes, shaders need parameters both per-draw-call and per-pipeline-call (instance and uniform
parameters)...

We also need to make heckin' sure that draw call's and geometry render in the drawn order rather than
overlapping/z-fighting!

Actual features/enhancements we want:

 * Being able to select which adapter to work with
 * Headless mode, for testing
 * Simpler/better handling of the various icky edge cases.
 * Faster (or even automatic) sprite batches.
 * We MIGHT be able to fit the info in DrawCall into a push constant?
 * We want to be able to expose the guts of the underlying graphics system to the user.

Compromises that we make:

 * All textures have the same color format, layout etc.
 * Don't bother with depth buffer, layers, etc
 * We never really have more than one window or Device at once
 * For complicated things, we expose the guts of the underlying graphics system to the user.

Questions to ask:

 * Can we have custom vertex types that get fed into shaders?
