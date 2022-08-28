use std::{f32::consts::PI, sync::Arc, time::Instant};

use fps_counter::FPSCounter;
use glam::{DVec2, Quat, UVec2, Vec2, Vec3};
use vulkano::{
    buffer::{BufferAccess, BufferUsage, DeviceLocalBuffer},
    command_buffer::{
        AutoCommandBufferBuilder, ClearColorImageInfo, CommandBufferUsage,
        PrimaryAutoCommandBuffer, RenderPassBeginInfo, SubpassContents,
    },
    descriptor_set::{DescriptorSetsCollection, PersistentDescriptorSet, WriteDescriptorSet},
    device::{
        physical::{PhysicalDevice, PhysicalDeviceType, QueueFamily},
        Device, DeviceCreateInfo, DeviceExtensions, Queue, QueueCreateInfo,
    },
    format::Format,
    image::{
        view::ImageView, ImageAccess, ImageCreateFlags, ImageDimensions, ImageUsage,
        ImageViewAbstract, StorageImage, SwapchainImage,
    },
    instance::{Instance, InstanceCreateInfo},
    pipeline::{
        graphics::{
            input_assembly::InputAssemblyState,
            vertex_input::VertexInputState,
            viewport::{Viewport, ViewportState},
        },
        ComputePipeline, GraphicsPipeline, Pipeline, PipelineBindPoint,
    },
    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass},
    shader::ShaderModule,
    swapchain::{
        self, AcquireError, Surface, Swapchain, SwapchainCreateInfo, SwapchainCreationError,
    },
    sync::{self, FenceSignalFuture, FlushError, GpuFuture},
    Version, VulkanLibrary,
};
use vulkano_win::VkSurfaceBuild;
use winit::{
    event::{ElementState, VirtualKeyCode},
    event_loop::{ControlFlow, EventLoop},
    window::{CursorGrabMode, Fullscreen, Window, WindowBuilder},
};
use winit_event_helper::EventHelper;

mod vertex_shader {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: "
#version 460

vec2 positions[3] = vec2[](
    vec2(-1.0, -1.0),
    vec2(3.0, -1.0),
    vec2(-1.0, 3.0)
);

void main() {
    gl_Position = vec4(positions[gl_VertexIndex], 0.0, 1.0);
}
        "
    }
}

mod fragment_shader {
    vulkano_shaders::shader! {
        ty: "fragment",
        types_meta: { #[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)] },
        src: "
#version 460

layout(location = 0) out vec4 fragColor;

layout(push_constant) uniform PushConstantData {
    vec4 rot;
    vec3 pos;
    vec2 view;
} pc;

// direct depth image (compute)
layout(binding = 0, r32f) uniform readonly image2D img;

// materials
const uint MAX_MATERIALS = 8;

struct Material {
    vec3 color;
    float diffuse;
    float specular;
    float shine;
};

// material
layout(binding = 1) uniform readonly ImmutableData {
    // TODO: implement object-material indexing
    Material mats[MAX_MATERIALS];
} buf;

// objects
const uint MAX_OBJECTS = 8;

struct Object {
    vec3 pos;
    float size;
};

layout(binding = 2) uniform readonly MutableData {
    uint matCount;
    uint objCount;
    Object objs[MAX_OBJECTS];
} mutBuf;

// scene
const float ambientScene = 0.2;

// camera
const float renderDist = 1000.0;
const float camFallOffFactor = 0.01;

// light
// TODO: implement multiple light source support
vec4 light = vec4(4.0, 2.0, 2.0, 1.0); // w = strength
const float lightFallOffFactor = 0.01;

vec3 rotate(vec4 q, vec3 v) {
    vec3 t = cross(q.xyz, v) + q.w * v;
    return v + 2.0 * cross(q.xyz, t);
}

// iq infinite repetition
vec3 repeat(vec3 p, vec3 r) {
    return mod(p + 0.5*r, r) - 0.5*r;
}

float diffuse(vec3 normal, vec3 lightDir) {
    return max(dot(normal, lightDir), 0.0);
}

float specular(vec3 normal, vec3 lightDir, vec3 camDir, float diffuse, float shine) {
    vec3 reflection = reflect(-lightDir, normal);
    return diffuse * pow(max(dot(reflection, camDir), 0.0), shine);
}

float sphereSDF(vec3 p, Object s) {
    return distance(p, s.pos) - s.size;
}

// TODO: implement custom FOVs
void main() {
    // maps FragCoord to xy range [-1.0, 1.0]
    vec2 normCoord = gl_FragCoord.xy * 2 / pc.view - 1.0;
    // assuming height <= width, width = 1.0 and height <= 1.0
    normCoord.y *= pc.view.y / pc.view.x; // TODO: move to CPU

    vec3 step = normalize(rotate(pc.rot, vec3(normCoord.x, 1.0, normCoord.y)));

    float totalDist = imageLoad(img, ivec2(gl_FragCoord.xy)).x;

    if (totalDist >= renderDist) {
        fragColor.xyz = vec3(0.0);
        return;
    }

    vec3 position = pc.pos + step * totalDist;

    Object object = mutBuf.objs[0];
    Material mat = buf.mats[0];
    float dist = sphereSDF(position, object);

    for (uint i = 1; i < mutBuf.objCount; i++) {
        Object newObject = mutBuf.objs[i];
        float newDist = sphereSDF(position, newObject);
        if (newDist < dist) {
            object = newObject;
            dist = newDist;
            mat = buf.mats[i];
        }
    }

    // light
    vec3 lightDir = normalize(light.xyz - position);

    float lightDist = distance(position, light.xyz);
    float lightDistFallOff = max(lightFallOffFactor * lightDist * lightDist, 0.1);

    // camera
    float camDist = distance(position, pc.pos);
    float camDistFallOff = max(camFallOffFactor * (camDist * camDist + 1.0), 1.0);

    // object
    // WARNING: only works for spheres
    vec3 normal = normalize(position - object.pos);
    
    float diffuse = diffuse(normal, lightDir);
    float specular = specular(normal, lightDir, -step, diffuse, mat.shine);

    float direct = max(diffuse + specular, 0.0) * light.w / lightDistFallOff;

    fragColor.xyz = (ambientScene + direct) / camDistFallOff * dot(normal, -step) * mat.color;
}"
    }
}

// TODO: increase initial resolution to MAX_RES / 16 or MAX_RES / 32
mod compute_shader {
    vulkano_shaders::shader! {
        ty: "compute",
        types_meta: { #[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)] },
        src: "
#version 460

layout(local_size_x = 8, local_size_y = 8) in;

layout(push_constant) uniform PushConstantData {
    vec4 rot;
    vec3 pos;
    uint iter;
    vec2 view;
    uint max_img; // highest image index
} pc;

// 9 images allows for a maximum resolution of 2048x2048 pixels
layout(binding = 0, r32f) uniform image2D imgs[9];

// objects
const uint MAX_OBJECTS = 8;

struct Object {
    vec3 pos;
    float size;
};

layout(binding = 1) uniform readonly MutableData {
    uint matCount;
    uint objCount;
    Object objs[MAX_OBJECTS];
} mutBuf;

// camera
const float renderDist = 1000.0;

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

// TODO: make cpu provide general values
void main() {
    uint unitSize = 1 << (pc.max_img - pc.iter); // TODO: move to CPU

    vec2 normCoord = ((gl_GlobalInvocationID.xy * 2 + 1) * unitSize) / pc.view - 1.0;
    normCoord.y *= pc.view.y / pc.view.x; // TODO: move to CPU

    // sqrt(2) * (nearest ceil base 2 of image width) / (screen width in pixels)
    float thresholdDist = 1.4142135 * gl_WorkGroupSize.x * unitSize / pc.view.x; // TODO: move to CPU

    vec3 step = normalize(rotate(pc.rot, vec3(normCoord.x, 1.0, normCoord.y)));

    float totalDist = 1.0;
    if (pc.iter > 0) {
        totalDist = imageLoad(imgs[pc.iter - 1], ivec2(gl_GlobalInvocationID.xy * 0.5)).r;
    }

    float distances[MAX_OBJECTS];

    for (uint i = 0; i < mutBuf.objCount; i++) {
        distances[i] = 0.0;
    }

    uint closest = 0;
    float nearest = 0.0;

    //for (uint j = 0; j < 100; j++) {
    while (totalDist < renderDist) {
        vec3 position = pc.pos + step * totalDist;

        
        // algorithm 3
        float dist = 1000000.0;
        // multiplying by 2.0 ''fixes'' a bug
        float radius = (totalDist + 1.0) * thresholdDist;
        for (uint i = 0; i < mutBuf.objCount; i++) {
            distances[i] -= nearest;
            if (distances[i] <= radius + nearest) {
                distances[i] = sphereSDF(position, mutBuf.objs[i]);
            }
            dist = min(dist, distances[i]);
        }
        nearest = max(dist, 0.0);

        totalDist += nearest;
        radius = (totalDist + 1.0) * thresholdDist;
        if (dist <= radius) {
            totalDist -= radius;
            break;
        }
        

        // // algorithm 2: only checks real distance when necessary
        // for (uint i = 0; i < mutBuf.objCount; i++) {
        //     distances[i] -= nearest;
        //     if (distances[i] < distances[closest]) {
        //         closest = i;
        //     }
        // }

        // nearest = distances[closest];
        // totalDist += nearest;
        // distances[closest] = sphereSDF(position, mutBuf.objs[closest]);
        
        // float radius = (totalDist + 1.0) * thresholdDist;
        // if (distances[closest] <= radius) {
        //     totalDist += distances[closest] - radius;
        //     break;
        // }

        
        // algorithm 1: checks real distance for each object every update
        // float dist = sphereSDF(position, mutBuf.objs[0]);
        // for (uint i = 1; i < mutBuf.objCount; i++) {
        //     float newDist = sphereSDF(position, mutBuf.objs[i]);
        //     dist = min(dist, newDist);
        // }

        // totalDist += dist;
        // float radius = (totalDist + 1.0) * thresholdDist;
        // if (dist <= radius) {
        //     totalDist -= radius;
        //     break;
        // }
    }
    imageStore(imgs[pc.iter], ivec2(gl_GlobalInvocationID.xy), vec4(max(totalDist, 0.0)));
}"
    }
}

fn select_physical_device<'a, W>(
    instance: &'a Arc<Instance>,
    surface: &Surface<W>,
    device_extensions: &DeviceExtensions,
) -> (PhysicalDevice<'a>, QueueFamily<'a>) {
    PhysicalDevice::enumerate(instance)
        .filter(|&p| p.supported_extensions().is_superset_of(device_extensions))
        .filter_map(|p| {
            p.queue_families()
                .find(|q| {
                    q.supports_graphics()
                        && q.supports_surface(surface).unwrap_or(false)
                        && q.supports_compute()
                })
                .map(|q| (p, q))
        })
        .min_by_key(|(p, _)| match p.properties().device_type {
            PhysicalDeviceType::DiscreteGpu => 4,
            PhysicalDeviceType::IntegratedGpu => 3,
            PhysicalDeviceType::VirtualGpu => 2,
            PhysicalDeviceType::Cpu => 1,
            PhysicalDeviceType::Other => 0,
        })
        .unwrap()
}

fn get_render_pass(device: Arc<Device>, swapchain: Arc<Swapchain<Window>>) -> Arc<RenderPass> {
    vulkano::single_pass_renderpass!(device.clone(),
        attachments: {
            color: {
                load: DontCare,
                store: Store,
                format: swapchain.image_format(),
                samples: 1,
            }
        },
        pass: {
            color: [color],
            depth_stencil: {}
        }
    )
    .unwrap()
}

fn get_framebuffers(
    images: &[Arc<SwapchainImage<Window>>],
    render_pass: Arc<RenderPass>,
) -> Vec<Arc<Framebuffer>> {
    images
        .iter()
        .map(|i| {
            let image_view = ImageView::new_default(i.clone()).unwrap();
            Framebuffer::new(
                render_pass.clone(),
                FramebufferCreateInfo {
                    attachments: vec![image_view],
                    ..Default::default()
                },
            )
            .unwrap()
        })
        .collect()
}

fn get_graphics_pipeline(
    device: Arc<Device>,
    vertex_shader: Arc<ShaderModule>,
    fragment_shader: Arc<ShaderModule>,
    viewport: Viewport,
    render_pass: Arc<RenderPass>,
) -> Arc<GraphicsPipeline> {
    GraphicsPipeline::start()
        .vertex_input_state(VertexInputState::new())
        .vertex_shader(vertex_shader.entry_point("main").unwrap(), ())
        .input_assembly_state(InputAssemblyState::new()) // might be unnecessary
        .fragment_shader(fragment_shader.entry_point("main").unwrap(), ())
        .viewport_state(ViewportState::viewport_fixed_scissor_irrelevant([viewport]))
        .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
        .build(device.clone())
        .unwrap()
}

fn get_compute_pipeline(
    device: Arc<Device>,
    compute_shader: Arc<ShaderModule>,
) -> Arc<ComputePipeline> {
    ComputePipeline::new(
        device.clone(),
        compute_shader.entry_point("main").unwrap(),
        &(),
        None,
        |_| {},
    )
    .unwrap()
}

fn get_graphics_descriptor_set(
    pipeline: Arc<GraphicsPipeline>,
    image_view: Arc<dyn ImageViewAbstract>,
    immutable_buffer: Arc<dyn BufferAccess>,
    mutable_buffer: Arc<dyn BufferAccess>,
) -> Arc<PersistentDescriptorSet> {
    PersistentDescriptorSet::new(
        pipeline.layout().set_layouts()[0].clone(),
        [
            WriteDescriptorSet::image_view(0, image_view.clone()),
            WriteDescriptorSet::buffer(1, immutable_buffer),
            WriteDescriptorSet::buffer(2, mutable_buffer),
        ],
    )
    .unwrap()
}

fn get_compute_descriptor_set(
    pipeline: Arc<ComputePipeline>,
    image_views: Vec<Arc<dyn ImageViewAbstract>>,
    mutable_buffer: Arc<dyn BufferAccess>,
) -> Arc<PersistentDescriptorSet> {
    PersistentDescriptorSet::new(
        pipeline.layout().set_layouts()[0].clone(),
        [
            WriteDescriptorSet::image_view_array(0, 0, image_views),
            WriteDescriptorSet::buffer(1, mutable_buffer),
        ],
    )
    .unwrap()
}

fn get_compute_images(
    device: Arc<Device>,
    queue: Arc<Queue>,
    image_count: usize,
    max_resolution: [f32; 2],
) -> Vec<Arc<StorageImage>> {
    let ratio = Vec2::from_array(max_resolution) / (4 << image_count) as f32;

    (0..image_count)
        .map(|i| {
            let dims = ((1 << i) as f32 * ratio).ceil().as_uvec2() * 8;

            StorageImage::with_usage(
                device.clone(),
                ImageDimensions::Dim2d {
                    width: dims.x,
                    height: dims.y,
                    array_layers: 1,
                },
                Format::R32_SFLOAT,
                ImageUsage {
                    storage: true,
                    transfer_dst: true,
                    ..ImageUsage::none()
                },
                ImageCreateFlags::none(),
                [queue.family()],
            )
            .unwrap()
        })
        .collect::<Vec<_>>()
}

fn get_compute_image_views(
    device: Arc<Device>,
    queue: Arc<Queue>,
    images: &Vec<Arc<StorageImage>>,
) -> Vec<Arc<dyn ImageViewAbstract>> {
    let mut image_views = images
        .iter()
        .map(|image| ImageView::new_default(image.clone()).unwrap() as _)
        .collect::<Vec<Arc<dyn ImageViewAbstract>>>();

    let null_image = StorageImage::new(
        device.clone(),
        ImageDimensions::Dim2d {
            width: 1,
            height: 1,
            array_layers: 1,
        },
        Format::R32_SFLOAT,
        [queue.family()],
    )
    .unwrap();

    let null_image_view = ImageView::new_default(null_image).unwrap();

    image_views.append(&mut vec![
        null_image_view;
        COMPUTE_IMAGE_COUNT - images.len()
    ]);

    image_views
}

fn get_command_buffer<S>(
    device: Arc<Device>,
    queue: Arc<Queue>,
    graphics_pipeline: Arc<GraphicsPipeline>,
    compute_pipeline: Arc<ComputePipeline>,
    framebuffer: Arc<Framebuffer>,
    compute_images: &Vec<Arc<StorageImage>>,
    compute_image_count: usize,
    push_constants: fragment_shader::ty::PushConstantData,
    graphics_descriptor_set: S,
    compute_descriptor_set: S,
) -> Arc<PrimaryAutoCommandBuffer>
where
    S: DescriptorSetsCollection + Clone,
{
    let mut builder = AutoCommandBufferBuilder::primary(
        device.clone(),
        queue.family(),
        CommandBufferUsage::OneTimeSubmit,
    )
    .unwrap();

    builder
        .bind_pipeline_compute(compute_pipeline.clone())
        .bind_descriptor_sets(
            PipelineBindPoint::Compute,
            compute_pipeline.layout().clone(),
            0,
            compute_descriptor_set.clone(),
        );

    for (image, i) in compute_images.iter().zip(0..compute_image_count) {
        let compute_pc = compute_shader::ty::PushConstantData {
            rot: push_constants.rot,
            pos: push_constants.pos,
            iter: i as u32,
            view: push_constants.view,
            max_img: compute_image_count as u32 - 1,
        };
        let [w, h] = image.dimensions().width_height(); // assumes depth = 1

        builder
            .push_constants(compute_pipeline.layout().clone(), 0, compute_pc)
            .clear_color_image(ClearColorImageInfo::image(image.clone()))
            .unwrap()
            .dispatch([w / 8, h / 8, 1])
            .unwrap();
    }

    builder
        .begin_render_pass(
            RenderPassBeginInfo {
                clear_values: vec![None],
                ..RenderPassBeginInfo::framebuffer(framebuffer.clone())
            },
            SubpassContents::Inline,
        )
        .unwrap()
        .bind_pipeline_graphics(graphics_pipeline.clone())
        .push_constants(graphics_pipeline.layout().clone(), 0, push_constants)
        .bind_descriptor_sets(
            PipelineBindPoint::Graphics,
            graphics_pipeline.layout().clone(),
            0,
            graphics_descriptor_set.clone(),
        )
        .draw(3, 1, 0, 0)
        .unwrap()
        .end_render_pass()
        .unwrap();

    Arc::new(builder.build().unwrap())
}

#[allow(dead_code)]
mod speed {
    pub const MOVEMENT: f32 = 25.0;
    pub const ROTATION: f32 = 1.0;
    pub const MOUSE: f32 = 1.0;
}

#[allow(dead_code)]
mod rotation {
    use glam::Vec3;

    pub const UP: Vec3 = Vec3::new(0.0, 0.0, 1.0);
    pub const FORWARD: Vec3 = Vec3::new(0.0, 1.0, 0.0);
    pub const RIGHT: Vec3 = Vec3::new(1.0, 0.0, 0.0);
}

const COMPUTE_IMAGE_COUNT: usize = 9;

struct Data<W> {
    /// the window surface
    surface: Arc<Surface<W>>,
    window_frozen: bool,
    window_resized: bool,
    recreate_swapchain: bool,
    /// viewport dimensions
    dimensions: Vec2,
    /// change in cursor position
    cursor_delta: Vec2,
    /// change in position relative to the rotation axes
    position: Vec3,
    /// absolute rotation around the x and z axes
    rotation: Vec2,
    /// time since the last update
    last_update: Instant,
    quit: bool,
}

impl<W> Data<W> {
    fn window(&self) -> &W {
        self.surface.window()
    }

    fn delta_time(&self) -> f32 {
        self.last_update.elapsed().as_secs_f32()
    }

    fn rot_time(&self) -> f32 {
        self.delta_time() * speed::ROTATION
    }

    fn mov_time(&self) -> f32 {
        self.delta_time() * speed::MOVEMENT
    }

    fn rotation(&self) -> Quat {
        Quat::from_rotation_z(-self.rotation.x) * Quat::from_rotation_x(self.rotation.y)
    }

    fn position(&self) -> Vec3 {
        let rotation = self.rotation();

        let right = rotation.mul_vec3(rotation::RIGHT);
        let forward = rotation.mul_vec3(rotation::FORWARD);
        let up = rotation.mul_vec3(rotation::UP);

        self.position.x * right + self.position.y * forward + self.position.z * up
    }
}

fn main() {
    let library = VulkanLibrary::new().unwrap();
    let required_extensions = vulkano_win::required_extensions(&library);
    let instance = Instance::new(
        library,
        InstanceCreateInfo {
            enabled_extensions: required_extensions,
            engine_version: Version::V1_3,
            ..Default::default()
        },
    )
    .unwrap();

    let event_loop = EventLoop::new();
    let surface = WindowBuilder::new()
        .build_vk_surface(&event_loop, instance.clone())
        .unwrap();
    // BUG: spamming any key on application startup will make the window invisible
    surface.window().set_cursor_visible(false);
    surface
        .window()
        // WARNING: not supported on Mac, web and mobile platforms
        .set_cursor_grab(CursorGrabMode::Confined)
        .unwrap();

    let device_extensions = DeviceExtensions {
        khr_swapchain: true,
        ..DeviceExtensions::none()
    };

    let (physical, family) = select_physical_device(&instance, &surface, &device_extensions);

    let (device, mut queues) = Device::new(
        physical,
        DeviceCreateInfo {
            queue_create_infos: vec![QueueCreateInfo::family(family)],
            enabled_extensions: device_extensions,
            ..Default::default()
        },
    )
    .unwrap();

    let queue = queues.next().unwrap();

    let capabilities = physical
        .surface_capabilities(&surface, Default::default())
        .unwrap();
    let dimensions = surface.window().inner_size();
    let composite_alpha = capabilities
        .supported_composite_alpha
        .iter()
        .next()
        .unwrap();
    let image_format = Some(
        physical
            .surface_formats(&surface, Default::default())
            .unwrap()
            .iter()
            .max_by_key(|(format, _)| match format {
                Format::R8G8B8A8_SRGB | Format::B8G8R8A8_SRGB => 1,
                _ => 0,
            })
            .unwrap()
            .0,
    );

    let (mut swapchain, swapchain_images) = Swapchain::new(
        device.clone(),
        surface.clone(),
        SwapchainCreateInfo {
            min_image_count: capabilities.min_image_count + 1,
            image_format,
            image_extent: dimensions.into(),
            image_usage: ImageUsage::color_attachment(),
            composite_alpha,
            ..Default::default()
        },
    )
    .unwrap();

    let render_pass = get_render_pass(device.clone(), swapchain.clone());

    let mut framebuffers = get_framebuffers(&swapchain_images, render_pass.clone());

    let vertex_shader = vertex_shader::load(device.clone()).unwrap();
    let fragment_shader = fragment_shader::load(device.clone()).unwrap();

    let mut viewport = Viewport {
        origin: [0.0, 0.0],
        dimensions: surface.window().inner_size().into(),
        depth_range: 0.0..1.0,
    };

    let mut graphics_pipeline = get_graphics_pipeline(
        device.clone(),
        vertex_shader.clone(),
        fragment_shader.clone(),
        viewport.clone(),
        render_pass.clone(),
    );

    const MATERIALS: [fragment_shader::ty::Material; 4] = [
        fragment_shader::ty::Material {
            color: [0.2, 0.2, 1.0],
            diffuse: 1.0,
            specular: 1.0,
            shine: 1.0,
            _dummy0: [0u8; 8],
        },
        fragment_shader::ty::Material {
            color: [0.0, 1.0, 0.0],
            diffuse: 1.0,
            specular: 1.0,
            shine: 1.0,
            _dummy0: [0u8; 8],
        },
        fragment_shader::ty::Material {
            color: [1.0, 1.0, 0.0],
            diffuse: 1.0,
            specular: 1.0,
            shine: 1.0,
            _dummy0: [0u8; 8],
        },
        fragment_shader::ty::Material {
            color: [1.0, 0.0, 0.0],
            diffuse: 1.0,
            specular: 1.0,
            shine: 1.0,
            _dummy0: [0u8; 8],
        },
    ];

    let (immutable_buffer, immutable_future) =
        DeviceLocalBuffer::from_data(MATERIALS, BufferUsage::uniform_buffer(), queue.clone())
            .unwrap();

    let objects = [
        fragment_shader::ty::Object {
            pos: [3.0, 5.0, -1.0],
            size: 3.0,
        },
        fragment_shader::ty::Object {
            pos: [5.0, 4.0, 5.0],
            size: 6.0,
        },
        fragment_shader::ty::Object {
            pos: [-0.0, 3.0, -0.0],
            size: 1.0,
        },
        fragment_shader::ty::Object {
            pos: [4.0, 1.0, 0.0],
            size: 2.0,
        },
    ];

    let mut mutable_data = fragment_shader::ty::MutableData {
        matCount: MATERIALS.len() as u32,
        objCount: objects.len() as u32,
        ..Default::default()
    };
    mutable_data.objs[..objects.len()].copy_from_slice(&objects);
    
    let (mutable_buffer, mutable_future) =
        DeviceLocalBuffer::from_data(mutable_data, BufferUsage::uniform_buffer(), queue.clone())
            .unwrap();

    immutable_future
        .join(mutable_future)
        .then_signal_fence_and_flush()
        .unwrap()
        .wait(None)
        .unwrap();

    let mut push_constants = fragment_shader::ty::PushConstantData {
        rot: [0.0, 0.0, 0.0, 0.0],
        pos: [0.0, 0.0, 0.0],
        view: viewport.dimensions,
        _dummy0: [0u8; 4],
    };

    let compute_shader = compute_shader::load(device.clone()).unwrap();

    let compute_pipeline = get_compute_pipeline(device.clone(), compute_shader.clone());

    let mut compute_image_count = (viewport.dimensions[0] / 8.0).log2() as usize + 1;

    let mut compute_images = get_compute_images(
        device.clone(),
        queue.clone(),
        compute_image_count,
        viewport.dimensions,
    );
    let compute_image_views =
        get_compute_image_views(device.clone(), queue.clone(), &compute_images);

    let mut graphics_descriptor_set = get_graphics_descriptor_set(
        graphics_pipeline.clone(),
        compute_image_views[compute_image_count - 1].clone(),
        immutable_buffer.clone(),
        mutable_buffer.clone(),
    );

    let mut compute_descriptor_set = get_compute_descriptor_set(
        compute_pipeline.clone(),
        compute_image_views.clone(),
        mutable_buffer.clone(),
    );

    let mut command_buffers = vec![None; framebuffers.len()];

    let mut fences: Vec<Option<Arc<FenceSignalFuture<_>>>> = vec![None; swapchain_images.len()];
    let mut previous_fence_index = 0;

    let mut eh = EventHelper::new(Data {
        surface,
        window_frozen: false,
        window_resized: false,
        recreate_swapchain: false,
        dimensions: Vec2::from_array(viewport.dimensions),
        cursor_delta: Vec2::ZERO,
        position: Vec3::ZERO,
        rotation: Vec2::ZERO,
        last_update: Instant::now(),
        quit: false,
    });

    let exit = |data: &mut EventHelper<Data<_>>| data.quit = true;
    eh.close_requested(exit);
    eh.keyboard(VirtualKeyCode::Escape, ElementState::Pressed, exit);

    eh.raw_mouse_delta(|data, (dx, dy)| data.cursor_delta += DVec2::new(dx, dy).as_vec2());

    eh.focused(|data, focused| {
        data.window_frozen = !focused;
        let window = data.window();
        if focused {
            window.set_cursor_grab(CursorGrabMode::Confined).unwrap();
        } else {
            window.set_cursor_grab(CursorGrabMode::None).unwrap();
        }
    });

    eh.resized(|data, mut size| {
        data.window_frozen = size.width == 0 || size.height == 0;
        data.window_resized = true;

        // the shaders are based on the assumption that width is less than height
        if size.width < size.height {
            size.height = size.width;
            data.window().set_inner_size(size);
        }

        data.dimensions = UVec2::new(size.width, size.height).as_vec2();
    });

    eh.keyboard(VirtualKeyCode::F11, ElementState::Pressed, |data| {
        let window = data.window();
        match window.fullscreen() {
            Some(_) => window.set_fullscreen(None),
            None => window.set_fullscreen(Some(Fullscreen::Borderless(None))),
        }
    });

    let mut fps_counter = FPSCounter::new();

    event_loop.run(move |event, _, control_flow| {
        if eh.quit {
            *control_flow = ControlFlow::Exit;
        }

        if !eh.update(&event) || eh.window_frozen {
            return;
        }

        println!("{}", fps_counter.tick());

        let cursor_mov = eh.cursor_delta / eh.dimensions.x * speed::ROTATION * speed::MOUSE;
        eh.rotation += cursor_mov;

        eh.cursor_delta = Vec2::ZERO;

        if eh.key_held(VirtualKeyCode::Left) {
            eh.rotation.x -= eh.rot_time();
        }
        if eh.key_held(VirtualKeyCode::Right) {
            eh.rotation.x += eh.rot_time();
        }
        if eh.key_held(VirtualKeyCode::Up) {
            eh.rotation.y -= eh.rot_time();
        }
        if eh.key_held(VirtualKeyCode::Down) {
            eh.rotation.y += eh.rot_time();
        }

        if eh.key_held(VirtualKeyCode::A) {
            eh.position.x -= eh.mov_time();
        }
        if eh.key_held(VirtualKeyCode::D) {
            eh.position.x += eh.mov_time();
        }
        if eh.key_held(VirtualKeyCode::W) {
            eh.position.y += eh.mov_time();
        }
        if eh.key_held(VirtualKeyCode::S) {
            eh.position.y -= eh.mov_time();
        }
        if eh.key_held(VirtualKeyCode::Q) {
            eh.position.z += eh.mov_time();
        }
        if eh.key_held(VirtualKeyCode::E) {
            eh.position.z -= eh.mov_time();
        }

        eh.rotation.y = eh.rotation.y.clamp(-0.5 * PI, 0.5 * PI);
        push_constants.rot = eh.rotation().to_array();
        push_constants.pos = (Vec3::from(push_constants.pos) + eh.position()).to_array();
        eh.position = Vec3::ZERO;

        eh.last_update = Instant::now();

        // rendering
        if eh.recreate_swapchain || eh.window_resized {
            eh.recreate_swapchain = false;

            let dimensions = eh.window().inner_size();

            let (new_swapchain, images) = match swapchain.recreate(SwapchainCreateInfo {
                image_extent: dimensions.into(),
                ..swapchain.create_info()
            }) {
                Ok(ok) => ok,
                Err(SwapchainCreationError::ImageExtentNotSupported { .. }) => return,
                Err(err) => panic!("{}", err),
            };
            swapchain = new_swapchain;

            framebuffers = get_framebuffers(&images, render_pass.clone());

            if eh.window_resized {
                eh.window_resized = false;

                viewport.dimensions = eh.dimensions.to_array();
                push_constants.view = viewport.dimensions;

                graphics_pipeline = get_graphics_pipeline(
                    device.clone(),
                    vertex_shader.clone(),
                    fragment_shader.clone(),
                    viewport.clone(),
                    render_pass.clone(),
                );

                compute_image_count = (viewport.dimensions[0] / 8.0).log2().ceil() as usize + 1;

                compute_images = get_compute_images(
                    device.clone(),
                    queue.clone(),
                    compute_image_count,
                    viewport.dimensions,
                );
                let compute_image_views =
                    get_compute_image_views(device.clone(), queue.clone(), &compute_images);

                compute_descriptor_set = get_compute_descriptor_set(
                    compute_pipeline.clone(),
                    compute_image_views.clone(),
                    mutable_buffer.clone(),
                );

                graphics_descriptor_set = get_graphics_descriptor_set(
                    graphics_pipeline.clone(),
                    compute_image_views[compute_image_count - 1].clone(),
                    immutable_buffer.clone(),
                    mutable_buffer.clone(),
                );
            }
        }

        let (image_index, suboptimal, image_future) =
            match swapchain::acquire_next_image(swapchain.clone(), None) {
                Ok(ok) => ok,
                Err(AcquireError::OutOfDate) => {
                    return eh.recreate_swapchain = true;
                }
                Err(err) => panic!("{}", err),
            };
        eh.recreate_swapchain |= suboptimal;

        if let Some(image_fence) = &fences[image_index] {
            image_fence.wait(None).unwrap();
        }

        command_buffers[image_index] = Some(get_command_buffer(
            device.clone(),
            queue.clone(),
            graphics_pipeline.clone(),
            compute_pipeline.clone(),
            framebuffers[image_index].clone(),
            &compute_images,
            compute_image_count,
            push_constants.clone(),
            graphics_descriptor_set.clone(),
            compute_descriptor_set.clone(),
        ));

        let previous_future = match fences[previous_fence_index].clone() {
            Some(future) => future.boxed(),
            None => {
                let mut future = sync::now(device.clone());
                future.cleanup_finished();
                future.boxed()
            }
        };

        let future = previous_future
            .join(image_future)
            .then_execute(queue.clone(), command_buffers[image_index].clone().unwrap())
            .unwrap()
            .then_swapchain_present(queue.clone(), swapchain.clone(), image_index)
            .then_signal_fence_and_flush();

        fences[image_index] = match future {
            Ok(ok) => Some(Arc::new(ok)),
            Err(FlushError::OutOfDate) => {
                eh.recreate_swapchain = true;
                None
            }
            Err(err) => {
                eprintln!("{}", err);
                None
            }
        };
        previous_fence_index = image_index;
    })
}
