# ggraphics

A small, swift 2D graphics rendering library written in Rust.

Currently, this is essentially an implmentation detail of
[`ggez`](https://ggez.rs/).  It's not really *designed* to be
used as a general-purpose thing, but other people may be interested
in it (ie, other people making 2D games or game engines in Rust)
but want to otherwise make different design decisions than `ggez` does.

# Design

Currently it is a simple 2D-only(ish) quad renderer using OpenGL.
It uses [`glow`](https://crates.io/crates/glow) as a thin OpenGL
portability layer, and takes a somewhat Vulkan/WebGPU-y approach of
render passes containing pipelines containing draw commands.

## Goals

 * Work on desktop, and web via WASM with minimal extra effort
 * Work on OpenGL 4 and comparable API's -- OpenGL ES 3, WebGL 2.
 * Draws textured quads and maybe arbitrary meshes.
 * Support user-defined shaders and render passes.

## Anti-goals

 * Absolutely cross-platform with no effort involved.  Weird platforms
   are always going to require extra work to build for and integrate.
   This should make that process easy, but doesn't need to try to make
   it invisible.
 * General-purpose.  Avoid success at all costs.  Most people should not
   need to use this directly.
 * Support every hardware ever.  Sorry.
 * Include windowing, input, etc.  Those are the jobs of other tools.
 * Sophisticated 3D rendering.  It'd be nice if this were extensible
   enough that you could add your own such things, but currently,
   "textured geometry with shaders" is minimum viable product.
 * Absolute top-shelf performance.  It should not be gratuitously slow,
   and should draw fast enough to be considered Pretty Good, but it
   doesn't need to be gratuitously fast.

## Maybe someday goals

 * Mobile devices as a first-class target
 * Use `gfx-hal`, `rendy`, `wgpu` or some other next-gen graphics
   portability layer.  Currently, the portability is not there.  :-(
 * Work on OpenGL 2 and comparable API's -- OpenGL ES 2, WebGL 1.


# License

MIT
