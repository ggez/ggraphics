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
