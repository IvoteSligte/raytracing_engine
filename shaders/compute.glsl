#version 460

#include "utilities.glsl"

layout(local_size_x = 8, local_size_y = 8) in;

layout(push_constant) uniform PushConstantData {
    vec4 rot;
    vec3 pos;
    uint iter;
    vec2 imageSize; // size of the current image in world space
} pc;

// 9 images allows for a maximum resolution of 2048x2048 pixels
layout(binding = 0, r32f) uniform image2D imgs[9];

layout(binding = 1) uniform readonly MutableData {
    uint matCount;
    uint objCount;
    uint lightCount;
    Material mats[MAX_MATERIALS];
    Object objs[MAX_OBJECTS];
    Light lights[MAX_LIGHTS];
} buf;

layout(binding = 2) uniform readonly ConstantBuffer {
    vec2 view; // window size
    vec2 ratio; // view mapper
} cs;

// camera
layout(constant_id = 0) const float RENDER_DIST = 1000.0;

float traceCone(vec3 origin, vec3 step, float threshold) {
    float distances[MAX_OBJECTS];

    for (uint i = 0; i < buf.objCount; i++) {
        distances[i] = sphereSDF(origin, buf.objs[i]);
    }

    float len = 0.0;
    float last = 0.0;

    while (len < RENDER_DIST) {
        vec3 position = origin + step * len;

        // tracing algorithm 3: only checks real distance when necessary,
        // but unlike algorithm 2 the edges look clean
        float dist = RENDER_DIST;
        float radius = (len + 1.0) * threshold;
        for (uint i = 0; i < buf.objCount; i++) {
            distances[i] -= last;
            if (distances[i] <= radius) {
                distances[i] = sphereSDF(position, buf.objs[i]);
            }
            dist = min(dist, distances[i]);
        }

        last = max(dist, 0.0);
        len += last;

        if (dist <= radius) {
            len -= radius;
            break;
        }
    }
    return len;
}

void main() {
    vec2 normCoord = (gl_GlobalInvocationID.xy * 2 + 1) * pc.imageSize - 1.0;
    normCoord *= cs.ratio;

    // sqrt(2) * (smallest image size in pixels) * (image size in coords)
    float threshold = 1.4142135 * gl_WorkGroupSize.x * pc.imageSize.x;

    vec3 step = normalize(rotate(pc.rot, vec3(normCoord.x, 1.0, normCoord.y)));

    float len = 1.0;
    if (pc.iter > 0) {
        len = imageLoad(imgs[pc.iter - 1], ivec2(gl_GlobalInvocationID.xy * 0.5)).r;
    }

    len += traceCone(pc.pos + step * len, step, threshold);

    imageStore(imgs[pc.iter], ivec2(gl_GlobalInvocationID.xy), vec4(max(len, 0.0)));
}