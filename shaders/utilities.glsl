
const uint MAX_MATERIALS = 8;
const uint MAX_OBJECTS = 8;
const uint MAX_LIGHTS = 8;

const float RAY_WIDTH = 0.001;

struct Material {
    vec3 color;
    float diffuse;
    float specular;
    float shine;
    float ambient;
};

struct Object {
    vec3 pos;
    float size;
};

struct Light {
    vec3 pos;
    vec3 color; // length(color) = strength
};

vec3 rotate(vec4 q, vec3 v) {
    vec3 t = cross(q.xyz, v) + q.w * v;
    return v + 2.0 * cross(q.xyz, t);
}

// p = point, r = repetition radius
vec3 repeat(vec3 p, vec3 r) {
    return mod(p + 0.5*r, r) - 0.5*r;
}

float sphereSDF(vec3 p, Object s) {
    return distance(p, s.pos) - s.size;
}
