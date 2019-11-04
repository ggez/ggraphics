// use rendy::command::QueueId;
// use rendy::factory::{Config, Factory};
// use rendy::graph::{present::PresentNode, render::*, GraphBuilder};
// use rendy::hal;
// use rendy::hal::PhysicalDevice as _;

// use winit::{Event, EventsLoop, WindowBuilder, WindowEvent};

// Suggested logging level for resource debugging:
// env RUST_LOG=warn,ggraphics=info cargo run

use ggraphics::quad::*;
use log::*;
use pretty_env_logger;

fn main() {
    pretty_env_logger::init();
    info!("Logging started");
    GraphicsWindowThing::run();
    info!("Window run finished");
    //x.dispose();
    info!("Window disposed");
}
