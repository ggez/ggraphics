// use rendy::command::QueueId;
// use rendy::factory::{Config, Factory};
// use rendy::graph::{present::PresentNode, render::*, GraphBuilder};
// use rendy::hal;
// use rendy::hal::PhysicalDevice as _;

// use winit::{Event, EventsLoop, WindowBuilder, WindowEvent};

use ggraphics::*;

fn main() {
    let mut x = new_vulkan_device();
    x.run();
    x.dispose();
}
