#version 450
#extension GL_ARB_separate_shader_objects : enable

// per-instance data below here?.
// vec4[4] is used instead of mat4 due to spirv-cross bug for dx12 backend
layout(location = 0) in vec4 model[4];
// Skip some locations for the members of the vec above
layout(location = 4) in vec4 rect;
layout(location = 5) in vec4 model_color;

layout(push_constant) uniform PushConstantTest {
    mat4 proj;
    mat4 view;
};

vec2 vertices[6] = {
    vec2(0.0, 0.0),
    vec2(0.0, 1.0),
    vec2(1.0, 1.0),
    vec2(0.0, 0.0),
    vec2(1.0, 1.0),
    vec2(1.0, 0.0),
};


layout(location = 0) out vec4 frag_pos;
layout(location = 1) out vec4 frag_color;
layout(location = 2) out vec4 uv;

void main() {
    mat4 model_mat = mat4(model[0], model[1], model[2], model[3]);

    frag_color = model_color;
    vec2 vertex = vertices[gl_VertexIndex % 6];
    frag_pos = model_mat * vec4(vertex, 0.0, 1.0);
    // TODO: Have unit quad's and scale verts properly by some multiplier,
    // instead of having to divide by the size of the quad
    // We also invert Y here to make things right-side up.
    //uv = vec4(pos.x / 100.0, 1 - (pos.y / 100.0), 0, 0);
    uv = vec4(vertex.x, 1 -vertex.y, 0.0, 0.0);

    // TODO: Fix depth crap!
    gl_Position = frag_pos * proj * view;
    gl_Position.z = 0.5;
}
