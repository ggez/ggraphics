// use rendy::command::QueueId;
// use rendy::factory::{Config, Factory};
// use rendy::graph::{present::PresentNode, render::*, GraphBuilder};
// use rendy::hal;
// use rendy::hal::PhysicalDevice as _;

// use winit::{Event, EventsLoop, WindowBuilder, WindowEvent};

// Suggested logging level for resource debugging:
// env RUST_LOG=warn,ggraphics=info cargo run

use ggraphics::quad::*;
use pretty_env_logger;
use log::*;

fn main() {
    pretty_env_logger::init();
    let mut x: GraphicsWindowThing<rendy::vulkan::Backend> = GraphicsWindowThing::new();
    info!("Window set up");
    x.run();
    info!("Window run finished");
    x.dispose();
    info!("Window disposed");
}
