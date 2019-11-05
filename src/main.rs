// Suggested logging level for debugging:
// env RUST_LOG=info cargo run

use glow::*;

#[cfg(all(target_arch = "wasm32", feature = "stdweb"))]
use std_web::{
    traits::*,
    unstable::TryInto,
    web::{document, html_element::*},
};
#[cfg(all(target_arch = "wasm32", feature = "stdweb"))]
use webgl_stdweb::WebGL2RenderingContext;

fn main() {
    unsafe {
        #[cfg(all(target_arch = "wasm32", feature = "stdweb"))]
        let (_window, gl, _events_loop, render_loop, shader_version) = {
            let canvas: CanvasElement = document()
                .create_element("canvas")
                .unwrap()
                .try_into()
                .unwrap();
            document().body().unwrap().append_child(&canvas);
            canvas.set_width(640);
            canvas.set_height(480);
            let webgl2_context: WebGL2RenderingContext = canvas.get_context().unwrap();
            (
                (),
                glow::Context::from_webgl2_context(webgl2_context),
                (),
                glow::RenderLoop::from_request_animation_frame(),
                "#version 300 es",
            )
        };

        // Create a context from a glutin window on non-wasm32 targets
        #[cfg(not(target_arch = "wasm32"))]
        let (gl, event_loop, windowed_context, shader_version) = {
            let el = glutin::event_loop::EventLoop::new();
            let wb = glutin::window::WindowBuilder::new()
                .with_title("Hello triangle!")
                .with_inner_size(glutin::dpi::LogicalSize::new(1024.0, 768.0));
            let windowed_context = glutin::ContextBuilder::new()
                .with_vsync(true)
                .build_windowed(wb, &el)
                .unwrap();
            let windowed_context = windowed_context.make_current().unwrap();
            let context = glow::Context::from_loader_function(|s| {
                windowed_context.get_proc_address(s) as *const _
            });
            (context, el, windowed_context, "#version 410")
        };

        let vertex_array = gl
            .create_vertex_array()
            .expect("Cannot create vertex array");
        gl.bind_vertex_array(Some(vertex_array));

        let program = gl.create_program().expect("Cannot create program");

        let (vertex_shader_source, fragment_shader_source) = (
            r#"const vec2 verts[3] = vec2[3](
                vec2(0.5f, 1.0f),
                vec2(0.0f, 0.0f),
                vec2(1.0f, 0.0f)
            );
            out vec2 vert;
            void main() {
                vert = verts[gl_VertexID];
                gl_Position = vec4(vert - 0.5, 0.0, 1.0);
            }"#,
            r#"precision mediump float;
            in vec2 vert;
            out vec4 color;
            void main() {
                color = vec4(vert, 0.5, 1.0);
            }"#,
        );

        let shader_sources = [
            (glow::VERTEX_SHADER, vertex_shader_source),
            (glow::FRAGMENT_SHADER, fragment_shader_source),
        ];

        let mut shaders = Vec::with_capacity(shader_sources.len());

        for (shader_type, shader_source) in shader_sources.iter() {
            let shader = gl
                .create_shader(*shader_type)
                .expect("Cannot create shader");
            gl.shader_source(shader, &format!("{}\n{}", shader_version, shader_source));
            gl.compile_shader(shader);
            if !gl.get_shader_compile_status(shader) {
                panic!(gl.get_shader_info_log(shader));
            }
            gl.attach_shader(program, shader);
            shaders.push(shader);
        }

        gl.link_program(program);
        if !gl.get_program_link_status(program) {
            panic!(gl.get_program_info_log(program));
        }

        for shader in shaders {
            gl.detach_shader(program, shader);
            gl.delete_shader(shader);
        }

        gl.use_program(Some(program));
        gl.clear_color(0.1, 0.2, 0.3, 1.0);

        // We handle events very differently between targets

        #[cfg(not(target_arch = "wasm32"))]
        {
            use glutin::event::{Event, WindowEvent};
            use glutin::event_loop::ControlFlow;

            event_loop.run(move |event, _, control_flow| {
                *control_flow = ControlFlow::Wait;
                match event {
                    Event::LoopDestroyed => {
                        println!("Event::LoopDestroyed!");
                        return;
                    }
                    Event::EventsCleared => {
                        println!("EventsCleared");
                        windowed_context.window().request_redraw();
                    }
                    Event::WindowEvent { ref event, .. } => match event {
                        WindowEvent::Resized(logical_size) => {
                            println!("WindowEvent::Resized: {:?}", logical_size);
                            let dpi_factor = windowed_context.window().hidpi_factor();
                            windowed_context.resize(logical_size.to_physical(dpi_factor));
                        }
                        WindowEvent::RedrawRequested => {
                            println!("WindowEvent::RedrawRequested");
                            gl.clear(glow::COLOR_BUFFER_BIT);
                            gl.draw_arrays(glow::TRIANGLES, 0, 3);
                            windowed_context.swap_buffers().unwrap();
                        }
                        WindowEvent::CloseRequested => {
                            println!("WindowEvent::CloseRequested");
                            gl.delete_program(program);
                            gl.delete_vertex_array(vertex_array);
                            *control_flow = ControlFlow::Exit
                        }
                        _ => (),
                    },
                    _ => (),
                }
            });
        }

        #[cfg(target_arch = "wasm32")]
        render_loop.run(move |running: &mut bool| {
            gl.clear(glow::COLOR_BUFFER_BIT);
            gl.draw_arrays(glow::TRIANGLES, 0, 3);

            if !*running {
                gl.delete_program(program);
                gl.delete_vertex_array(vertex_array);
            }
        });
    }
}
