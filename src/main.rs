use std::{f32::consts::PI, sync::Arc, time::Instant};

use glam::{DVec2, Quat, UVec2, Vec2, Vec3};
use vulkano::{
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
        src: "
#version 460

layout(push_constant) uniform PushConstantData {
    vec4 rot;
    vec3 pos;
    vec2 view;
} pc;

layout(binding = 0, r32f) uniform readonly image2D img;

layout(location = 0) out vec4 fragColor;

// object
const vec4 sphere = vec4(0.0, 4.0, 0.0, 2.0); // w = radius
const vec3 colorMat = vec3(0.2, 0.2, 1.0);
const float specularMat = 1.0;
const float diffuseMat = 1.0;
const float shineMat = 1.0;

// scene
const float ambientScene = 0.2;

// camera
const float camFallOffFactor = 0.001;
const uint maxSteps = 100;
const float minStep = 0.01;

// light
vec4 light = vec4(5.0, 5.0, 0.0, 10.0); // w = strength

vec3 rotate(vec4 q, vec3 v) {
    vec3 t = cross(q.xyz, v) + q.w * v;
    return v + 2.0 * cross(q.xyz, t);
}

// modified iq infinite repetition
// p = point, r = repetition radius, c = repetition center
vec3 repeat(vec3 p, vec3 r, vec3 c) {
    return mod(p - c + 0.5*r, r) - 0.5*r;
}

void main() {
    // maps FragCoord to xy range [-1.0, 1.0]
    vec2 normCoord = gl_FragCoord.xy * 2 / pc.view - 1.0;
    // assuming height <= width, width = 1.0 and height <= 1.0
    normCoord.y *= pc.view.y / pc.view.x;

    vec3 fragPos = vec3(normCoord.x, 1.0, normCoord.y);
    fragPos = rotate(pc.rot, fragPos);
    vec3 step = normalize(fragPos);
    fragPos += pc.pos;

    float totalDist = imageLoad(img, ivec2(gl_FragCoord.xy)).x;

    // fragColor = vec4(totalDist / 100000.0, 0.0, 0.0, 1.0);
    // return;

    for (uint i = 0; i < maxSteps; i++) {
        vec3 position = fragPos + step * totalDist;
        vec3 repPosition = repeat(position, vec3(10.0), sphere.xyz);

        float dist = length(repPosition) - sphere.w;
        if (dist <= 0.0) {
            // light
            vec3 lightDir = normalize(light.xyz - repPosition);

            float lightDist = distance(repPosition, light.xyz);
            float lightDistFallOff = lightDist * lightDist;
            
            // camera
            float camDist = distance(position, fragPos);
            float camDistFallOff = max(camFallOffFactor * (camDist * camDist + 1.0), 1.0);

            // object
            vec3 normal = normalize(repPosition);
            vec3 reflection = reflect(lightDir, normal);

            float diffuse = max(dot(normal, lightDir), 0.0) * light.w * diffuseMat;
            // fix specular
            float specular = pow(max(dot(reflection, -step), 0.0), shineMat) * specularMat;

            float direct = (diffuse + specular) / lightDistFallOff;

            fragColor = vec4((ambientScene + direct) * colorMat / camDistFallOff, 1.0);
            return;
        }
        totalDist += dist + minStep;
    }
    fragColor = vec4(0.0, 0.0, 0.0, 1.0);
}"
    }
}

mod compute_shader {
    vulkano_shaders::shader! {
        ty: "compute",
        src: "
#version 460

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(push_constant) uniform PushConstantData {
    vec4 rot;
    vec3 pos;
    uint iter;
    vec2 view;
    uint img_c; // image count
} pc;

// 9 images allows for a maximum resolution of 2048x2048 pixels
layout(binding = 0, r32f) uniform image2D imgs[9];

// sphere
const vec4 sphere = vec4(0.0, 4.0, 0.0, 2.0); // w = radius

// camera
const uint maxSteps = 1000;

vec3 rotate(vec4 q, vec3 v) {
    vec3 t = cross(q.xyz, v) + q.w * v;
    return v + 2.0 * cross(q.xyz, t);
}

// p = point, r = repetition radius, c = repetition center
vec3 repeat(vec3 p, vec3 r, vec3 c) {
    return mod(p - c + 0.5*r, r) - 0.5*r;
}

void main() {
    vec2 normCoord = ((gl_GlobalInvocationID.xy * 2 + 1) << (pc.img_c - pc.iter - 1)) / pc.view - 1.0;
    normCoord.y *= pc.view.y / pc.view.x;

    float thresholdDist = 1.4142135 / (gl_NumWorkGroups.x * gl_WorkGroupSize.x) * 2.0; // sqrt(2) / width

    vec3 fragPos = vec3(normCoord.x, 1.0, normCoord.y);
    fragPos = rotate(pc.rot, fragPos);
    vec3 step = normalize(fragPos);
    fragPos += pc.pos;

    float totalDist = 0.0;
    if (pc.iter > 0) {
        totalDist = imageLoad(imgs[pc.iter - 1], ivec2(gl_GlobalInvocationID.xy / 2)).r;
    }

    for (uint i = 0; i < maxSteps; i++) {
        vec3 position = fragPos + step * totalDist;

        vec3 repPosition = repeat(position, vec3(10.0), sphere.xyz);

        float dist = length(repPosition) - sphere.w;
        float radius = (totalDist + dist + 1.0) * thresholdDist;
        if (dist < radius) {
            break;
        }
        totalDist += dist;
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
) -> Arc<PersistentDescriptorSet> {
    PersistentDescriptorSet::new(
        pipeline.layout().set_layouts()[0].clone(),
        [WriteDescriptorSet::image_view(0, image_view.clone())],
    )
    .unwrap()
}

fn get_compute_descriptor_set(
    pipeline: Arc<ComputePipeline>,
    image_views: Vec<Arc<dyn ImageViewAbstract>>,
) -> Arc<PersistentDescriptorSet> {
    PersistentDescriptorSet::new(
        pipeline.layout().set_layouts()[0].clone(),
        [WriteDescriptorSet::image_view_array(0, 0, image_views)],
    )
    .unwrap()
}

fn get_compute_images(
    device: Arc<Device>,
    queue: Arc<Queue>,
    image_count: u32,
    max_resolution: [f32; 2],
) -> Vec<Arc<StorageImage>> {
    let ratio = Vec2::from_array(max_resolution) / (4 << image_count) as f32;

    let mut images = (0..image_count)
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
        .collect::<Vec<_>>();

    let empty_images = (image_count..COMPUTE_IMAGE_COUNT).map(|_| {
        StorageImage::with_usage(
            device.clone(),
            ImageDimensions::Dim2d {
                width: 1,
                height: 1,
                array_layers: 1,
            },
            Format::R32_SFLOAT,
            ImageUsage {
                storage: true,
                ..ImageUsage::none()
            },
            ImageCreateFlags::none(),
            [queue.family()],
        )
        .unwrap()
    });

    images.extend(empty_images);

    images
}

fn get_compute_image_views(images: &Vec<Arc<StorageImage>>) -> Vec<Arc<dyn ImageViewAbstract>> {
    images
        .iter()
        .map(|image| ImageView::new_default(image.clone()).unwrap() as _)
        .collect::<Vec<Arc<dyn ImageViewAbstract>>>()
}

fn get_command_buffer<S>(
    device: Arc<Device>,
    queue: Arc<Queue>,
    graphics_pipeline: Arc<GraphicsPipeline>,
    compute_pipeline: Arc<ComputePipeline>,
    framebuffer: Arc<Framebuffer>,
    compute_images: &Vec<Arc<StorageImage>>,
    compute_image_count: u32,
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

    for (image, iter) in compute_images.iter().zip(0..compute_image_count) {
        let compute_pc = compute_shader::ty::PushConstantData {
            rot: push_constants.rot,
            pos: push_constants.pos,
            iter,
            view: push_constants.view,
            img_c: compute_image_count,
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

const COMPUTE_IMAGE_COUNT: u32 = 9;

struct Data<W> {
    surface: Arc<Surface<W>>,
    window_frozen: bool,
    window_resized: bool,
    recreate_swapchain: bool,
    dimensions: Vec2,
    cursor_delta: Vec2,
    position: Vec3,
    rotation: Vec2,
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

    fn up(&self) -> Vec3 {
        self.rotation().mul_vec3(rotation::UP)
    }

    fn forward(&self) -> Vec3 {
        self.rotation().mul_vec3(rotation::FORWARD)
    }

    fn right(&self) -> Vec3 {
        self.rotation().mul_vec3(rotation::RIGHT)
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

    let (mut swapchain, images) = Swapchain::new(
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

    let mut framebuffers = get_framebuffers(&images, render_pass.clone());

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

    let compute_shader = compute_shader::load(device.clone()).unwrap();

    let compute_pipeline = get_compute_pipeline(device.clone(), compute_shader.clone());

    let mut push_constants = fragment_shader::ty::PushConstantData {
        rot: [0.0, 0.0, 0.0, 0.0],
        pos: [0.0, 0.0, 0.0],
        view: viewport.dimensions,
        _dummy0: [0u8; 4],
    };

    let mut compute_image_count = (viewport.dimensions[0] / 4.0).log2() as u32;

    let mut compute_images = get_compute_images(
        device.clone(),
        queue.clone(),
        compute_image_count,
        viewport.dimensions,
    );
    let compute_image_views = get_compute_image_views(&compute_images);

    let mut graphics_descriptor_set = get_graphics_descriptor_set(
        graphics_pipeline.clone(),
        compute_image_views.last().unwrap().clone(),
    );

    let mut compute_descriptor_set =
        get_compute_descriptor_set(compute_pipeline.clone(), compute_image_views.clone());

    let mut command_buffers = vec![None; framebuffers.len()];

    let mut fences: Vec<Option<Arc<FenceSignalFuture<_>>>> = vec![None; images.len()];
    let mut previous_fence_index = 0;

    let mut event_helper = EventHelper::new(Data {
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

    let exit = |data: &mut Data<Window>| data.quit = true;
    event_helper.close_requested(exit);
    event_helper.keyboard(VirtualKeyCode::Escape, ElementState::Pressed, exit);

    event_helper
        .raw_mouse_delta(|data, (dx, dy)| data.cursor_delta += DVec2::new(dx, dy).as_vec2());

    event_helper.focused(|data, focused| {
        data.window_frozen = !focused;
        let window = data.window();
        if focused {
            window.set_cursor_grab(CursorGrabMode::Confined).unwrap();
        } else {
            window.set_cursor_grab(CursorGrabMode::None).unwrap();
        }
    });

    event_helper.resized(|data, size| {
        data.window_frozen = size.width == 0 || size.height == 0;
        data.window_resized = true;
        data.dimensions = UVec2::new(size.width, size.height).as_vec2();
    });

    event_helper.keyboard(VirtualKeyCode::F11, ElementState::Pressed, |data| {
        let window = data.window();
        match window.fullscreen() {
            Some(_) => window.set_fullscreen(None),
            None => window.set_fullscreen(Some(Fullscreen::Borderless(None))),
        }
    });

    event_loop.run(move |event, _, control_flow| {
        if !event_helper.update(&event) || event_helper.window_frozen {
            return;
        }

        if event_helper.quit {
            *control_flow = ControlFlow::Exit;
        }

        let cursor_mov =
            event_helper.cursor_delta / event_helper.dimensions.y * speed::ROTATION * speed::MOUSE;
        event_helper.rotation += cursor_mov;

        event_helper.cursor_delta = Vec2::ZERO;

        let rot_time = event_helper.rot_time();

        if event_helper.key_held(VirtualKeyCode::Left) {
            event_helper.rotation.x += rot_time;
        }
        if event_helper.key_held(VirtualKeyCode::Right) {
            event_helper.rotation.x -= rot_time;
        }
        if event_helper.key_held(VirtualKeyCode::Up) {
            event_helper.rotation.y -= rot_time;
        }
        if event_helper.key_held(VirtualKeyCode::Down) {
            event_helper.rotation.y += rot_time;
        }

        if event_helper.key_held(VirtualKeyCode::A) {
            let change = event_helper.right() * event_helper.mov_time();
            event_helper.position -= change;
        }
        if event_helper.key_held(VirtualKeyCode::D) {
            let change = event_helper.right() * event_helper.mov_time();
            event_helper.position += change;
        }
        if event_helper.key_held(VirtualKeyCode::W) {
            let change = event_helper.forward() * event_helper.mov_time();
            event_helper.position += change;
        }
        if event_helper.key_held(VirtualKeyCode::S) {
            let change = event_helper.forward() * event_helper.mov_time();
            event_helper.position -= change;
        }
        if event_helper.key_held(VirtualKeyCode::Q) {
            let change = event_helper.up() * event_helper.mov_time();
            event_helper.position += change;
        }
        if event_helper.key_held(VirtualKeyCode::E) {
            let change = event_helper.up() * event_helper.mov_time();
            event_helper.position -= change;
        }

        event_helper.rotation.y = event_helper.rotation.y.clamp(-0.5 * PI, 0.5 * PI);
        push_constants.rot = event_helper.rotation().to_array();
        push_constants.pos = event_helper.position.to_array();

        event_helper.last_update = Instant::now();

        // vvv rendering vvv
        if event_helper.recreate_swapchain || event_helper.window_resized {
            event_helper.recreate_swapchain = false;

            let dimensions = event_helper.window().inner_size();

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

            if event_helper.window_resized {
                event_helper.window_resized = false;

                viewport.dimensions = event_helper.dimensions.to_array();
                push_constants.view = viewport.dimensions;

                graphics_pipeline = get_graphics_pipeline(
                    device.clone(),
                    vertex_shader.clone(),
                    fragment_shader.clone(),
                    viewport.clone(),
                    render_pass.clone(),
                );

                compute_image_count = (viewport.dimensions[0] / 4.0).log2().ceil() as u32;

                compute_images = get_compute_images(
                    device.clone(),
                    queue.clone(),
                    compute_image_count,
                    viewport.dimensions,
                );
                let compute_image_views = get_compute_image_views(&compute_images);

                compute_descriptor_set = get_compute_descriptor_set(
                    compute_pipeline.clone(),
                    compute_image_views.clone(),
                );

                graphics_descriptor_set = get_graphics_descriptor_set(
                    graphics_pipeline.clone(),
                    compute_image_views.last().unwrap().clone(),
                );
            }
        }

        let (image_index, suboptimal, image_future) =
            match swapchain::acquire_next_image(swapchain.clone(), None) {
                Ok(ok) => ok,
                Err(AcquireError::OutOfDate) => {
                    return event_helper.recreate_swapchain = true;
                }
                Err(err) => panic!("{}", err),
            };
        event_helper.recreate_swapchain |= suboptimal;

        if let Some(image_fence) = &fences[image_index] {
            image_fence.wait(None).unwrap();
        }

        command_buffers[image_index] = Some(get_command_buffer(
            device.clone(),
            queue.clone(),
            graphics_pipeline.clone(),
            compute_pipeline.clone(),
            framebuffers[image_index].clone(),
            &compute_images, // TODO: add compute_image for each framebuffer
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
                event_helper.recreate_swapchain = true;
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
