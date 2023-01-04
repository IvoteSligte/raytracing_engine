#version 460

#include "utilities.glsl"

layout(location = 0) out vec4 fragColor;

layout(push_constant) uniform PushConstantData {
    vec4 rot;
    vec3 pos;
    uint iter;
    vec2 imageSize;
} pc;

// direct depth image (compute)
layout(binding = 0, r32f) uniform readonly image2D img;

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
    vec2 ratio; // window height / width
} cs;

// specialization constants
layout(constant_id = 0) const float RENDER_DIST = 1000.0;

// constants
const float CAM_FALL_OFF = 0.01;
const float LIGHT_FALL_OFF = 0.01;
const float RAY_RADIUS = 0.01;

vec3 sphereNorm(vec3 p, Object s) {
    return normalize(p - s.pos);
}

float diffuse(vec3 normal, vec3 lightDir) {
    return max(dot(normal, lightDir), 0.0);
}

float specular(vec3 normal, vec3 lightDir, vec3 camDir, float diffuse, float shine) {
    vec3 reflection = reflect(-lightDir, normal);
    return max(diffuse * pow(dot(reflection, camDir), shine), 0.0);
}

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

// TODO: fix weird shadow on green ball (probably not related to the function)
float shadowRay(vec3 origin, vec3 step, float end) {
    float distances[MAX_OBJECTS];

    for (uint i = 0; i < buf.objCount; i++) {
        distances[i] = sphereSDF(origin, buf.objs[i]);
    }

    float last = 0.0;
    float nearest = 1.0;

    for (float len = 0.0; len < end; len += last + RAY_RADIUS) {
        vec3 position = origin + step * len;

        // tracing algorithm 3: only checks real distance when necessary,
        // but unlike algorithm 2 the edges look clean
        float dist = end;
        for (uint i = 0; i < buf.objCount; i++) {
            distances[i] -= last;
            if (distances[i] <= nearest) {
                distances[i] = sphereSDF(position, buf.objs[i]);
            }
            dist = min(dist, distances[i]);
        }

        if (dist <= RAY_RADIUS) {
            return 0.0;
        }

        last = max(dist, 0.0);
        nearest = min(nearest, dist);
    }
    return nearest;
}

// TODO: shadows
// TODO: transparency
// TODO: reflection
// TODO: refraction
void main() {
    // maps FragCoord to xy range [-1.0, 1.0]
    vec2 normCoord = gl_FragCoord.xy * 2 / cs.view - 1.0;
    // maps normCoord to a different range (e.g. for FOV and non-square windows)
    normCoord *= cs.ratio;

    vec3 step = normalize(rotate(pc.rot, vec3(normCoord.x, 1.0, normCoord.y)));

    float totalDist = imageLoad(img, ivec2(gl_FragCoord.xy)).x;

    if (totalDist >= RENDER_DIST) {
        fragColor.xyz = vec3(0.0);
        return;
    }

    vec3 position = pc.pos + step * totalDist;

    Material mat = buf.mats[0];
    Object object = buf.objs[0];
    float dist = sphereSDF(position, object);

    for (uint i = 1; i < buf.objCount; i++) {
        Object newObject = buf.objs[i];
        float newDist = sphereSDF(position, newObject);
        if (newDist < dist) {
            object = newObject;
            dist = newDist;
            mat = buf.mats[i];
        }
    }

    // initialize fragColor
    fragColor.rgb = vec3(0.0);

    // camera
    float camDist = distance(position, pc.pos);
    float camDistFallOff = max(CAM_FALL_OFF * (camDist * camDist + 1.0), 1.0);

    // object
    vec3 normal = sphereNorm(position, object);
    float normalFallOff = max(dot(normal, -step), 0.0);

    // sum the colors received from all lights
    for (uint i = 0; i < buf.lightCount; i++) {
        Light light = buf.lights[i];

        vec3 lightDir = normalize(light.pos - position);
        float lightDist = distance(position, light.pos);

        float softShadow = min(shadowRay(position + lightDir, lightDir, lightDist), 1.0);

        float lightDistFallOff = max(LIGHT_FALL_OFF * lightDist * lightDist, 1.0);
        
        float diffuse = diffuse(normal, lightDir);
        float specular = specular(normal, lightDir, -step, diffuse, mat.shine);

        vec3 direct = max(diffuse + specular, 0.0) * light.color / lightDistFallOff * softShadow;

        fragColor.rgb += (mat.ambient + direct) / camDistFallOff * normalFallOff * mat.color;
    }
}