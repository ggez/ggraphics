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
