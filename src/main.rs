// use rendy::command::QueueId;
// use rendy::factory::{Config, Factory};
// use rendy::graph::{present::PresentNode, render::*, GraphBuilder};
// use rendy::hal;
// use rendy::hal::PhysicalDevice as _;

// use winit::{Event, EventsLoop, WindowBuilder, WindowEvent};

use ggraphics::quad::*;

fn main() {
    let mut x: GraphicsWindowThing<rendy::vulkan::Backend> = GraphicsWindowThing::new();
    x.run();
    x.dispose();
}
