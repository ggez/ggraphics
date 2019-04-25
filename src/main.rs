/*
TODO: Make these select-able either based on platform,
or at runtime?
TODO: Rendy has no gl backend yet.
TODO: dx12, metal, whatever
TODO: Make shaderc less inconvenient?

#[cfg(feature = "dx12")]
type Backend = rendy::dx12::Backend;
#[cfg(feature = "gl")]
type Backend = rendy::gl::Backend;
 */


use rendy::factory::{Config, Factory};

#[cfg(feature = "vulkan")]
type Backend = rendy::vulkan::Backend;

#[cfg(any(feature = "vulkan"))]
fn main() {
    env_logger::Builder::from_default_env()
        .filter_module("init", log::LevelFilter::Trace)
        .init();

    let config: Config = Default::default();

    let (factory, families): (Factory<Backend>, _) = rendy::factory::init(config).unwrap();
    drop(families);
    drop(factory);
}

#[cfg(not(any(feature = "vulkan")))]
fn main() {
    panic!("Specify feature: { vulkan }");
}
