use std::{sync::Arc, time::Instant};

use extend::ext;
use glam::{DVec2, Quat, UVec2, Vec2, Vec3};
use vulkano::{
    buffer::{BufferUsage, DeviceLocalBuffer},
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferUsage, PrimaryAutoCommandBuffer,
        RenderPassBeginInfo, SubpassContents,
    },
    descriptor_set::{DescriptorSetsCollection, PersistentDescriptorSet, WriteDescriptorSet},
    device::{
        physical::{PhysicalDevice, PhysicalDeviceType, QueueFamily},
        Device, DeviceCreateInfo, DeviceExtensions, Queue, QueueCreateInfo,
    },
    format::Format,
    image::{view::ImageView, ImageUsage, SwapchainImage},
    instance::{Instance, InstanceCreateInfo},
    pipeline::{
        graphics::{
            input_assembly::InputAssemblyState,
            vertex_input::VertexInputState,
            viewport::{Viewport, ViewportState},
        },
        GraphicsPipeline, Pipeline, PipelineBindPoint,
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
    dpi::PhysicalPosition,
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
} pc;

layout(binding = 0) uniform ViewportBuffer {
    vec2 size;
} view;

layout(location = 0) out vec4 fragColor;

vec3 rotate(vec4 q, vec3 v) {
    vec3 temp = cross(q.xyz, v) + q.w * v;
    return v + 2.0 * cross(q.xyz, temp);
}

void main() {
    vec2 normCoord = gl_FragCoord.xy / view.size - 0.5;
    normCoord.x *= view.size.x / view.size.y;

    vec4 light = vec4(5.0, 5.0, 0.0, 100.0); // w = strength
    vec4 sphere = vec4(0.0, 5.0, 0.0, 2.0); // w = radius

    vec3 colorMat = vec3(0.2, 0.2, 1.0);
    float specularMat = 1.0;
    float diffuseMat = 1.0;
    float shineMat = 1.0;

    float ambientScene = 0.8;

    float nearClipPlane = 1.0;
    uint farClipPlane = 1000;
    float stepLength = 0.01;

    vec3 fragPos = vec3(normCoord.x, 1.0, normCoord.y);
    fragPos = rotate(pc.rot, fragPos);
    vec3 position = pc.pos + fragPos * nearClipPlane;
    vec3 step = normalize(fragPos) * stepLength;

    while (length(position) < farClipPlane) {
        float minDist = distance(position, sphere.xyz);
        if (minDist <= sphere.w) {
            vec3 lightDir = normalize(light.xyz - position);
            
            float lightDist = distance(position, light.xyz);
            float lightDistFallOff = 1.0 / (lightDist * lightDist);
            
            float camDist = distance(position, fragPos);
            float camDistFallOff = 1.0 / (camDist * camDist);

            vec3 normal = normalize(position - sphere.xyz);
            vec3 reflection = reflect(lightDir, normal);

            vec3 ambient = ambientScene * colorMat;
            vec3 diffuse = max(dot(normal, lightDir), 0.0) * light.w * diffuseMat * colorMat * lightDistFallOff;
            vec3 specular = max(pow(dot(reflection, fragPos), shineMat), 0.0) * specularMat * colorMat * lightDistFallOff;

            fragColor = vec4((ambient + diffuse + specular) * camDistFallOff, 1.0);
            return;
        }
        position += step * minDist; // incl treshold: max(step * minDist, minStep)
    }
    fragColor = vec4(0.0, 0.0, 0.0, 1.0);
}"
    }
}

#[ext]
impl Vec2 {
    fn to_pos(self) -> PhysicalPosition<f32> {
        PhysicalPosition::new(self.x, self.y)
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
                .find(|q| q.supports_graphics() && q.supports_surface(surface).unwrap_or(false))
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
                load: Clear,
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

fn get_descriptor_set(
    graphics_pipeline: Arc<GraphicsPipeline>,
    queue: Arc<Queue>,
    viewport: Viewport,
) -> Arc<PersistentDescriptorSet> {
    let (uniform_buffer, _) = DeviceLocalBuffer::from_data(
        viewport.dimensions,
        BufferUsage::uniform_buffer(),
        queue.clone(),
    )
    .unwrap();
    PersistentDescriptorSet::new(
        graphics_pipeline.layout().set_layouts()[0].clone(),
        [WriteDescriptorSet::buffer(0, uniform_buffer.clone())],
    )
    .unwrap()
}

fn get_primary_command_buffer<S>(
    device: Arc<Device>,
    queue: Arc<Queue>,
    graphics_pipeline: Arc<GraphicsPipeline>,
    framebuffer: Arc<Framebuffer>,
    push_constants: fragment_shader::ty::PushConstantData,
    descriptor_set: S,
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
        .begin_render_pass(
            RenderPassBeginInfo {
                clear_values: vec![Some([0.0, 0.0, 0.0, 1.0].into())],
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
            descriptor_set.clone(),
        )
        .draw(3, 1, 0, 0)
        .unwrap()
        .end_render_pass()
        .unwrap();

    Arc::new(builder.build().unwrap())
}

#[allow(dead_code)]
mod speed {
    pub const MOVEMENT: f32 = 4.0;
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

struct Data<W> {
    surface: Arc<Surface<W>>,
    window_frozen: bool,
    window_resized: bool,
    recreate_swapchain: bool,
    dimensions: Vec2,
    cursor_delta: Vec2,
    position: Vec3,
    rotation: Quat,
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

    fn up(&self) -> Vec3 {
        self.rotation.mul_vec3(rotation::UP)
    }

    fn forward(&self) -> Vec3 {
        self.rotation.mul_vec3(rotation::FORWARD)
    }

    fn right(&self) -> Vec3 {
        self.rotation.mul_vec3(rotation::RIGHT)
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
    surface.window().set_cursor_visible(false);
    surface
        .window()
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

    let mut push_constants = fragment_shader::ty::PushConstantData {
        rot: [0.0, 0.0, 0.0, 1.0],
        pos: [0.0, 0.0, 0.0],
    };

    let mut descriptor_set =
        get_descriptor_set(graphics_pipeline.clone(), queue.clone(), viewport.clone());

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
        rotation: Quat::IDENTITY,
        last_update: Instant::now(),
        quit: false,
    });

    let exit = |data: &mut Data<Window>| data.quit = true;
    event_helper.close_requested(exit);
    event_helper.keyboard(VirtualKeyCode::Escape, ElementState::Pressed, exit);

    event_helper
        .raw_mouse_delta(|data, (dx, dy)| data.cursor_delta += DVec2::new(dx, dy).as_vec2());

    event_helper.focused(|data, focused| data.window_frozen = !focused);

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
        if event_helper.quit {
            *control_flow = ControlFlow::Exit;
        }

        if !event_helper.update(&event) || event_helper.window_frozen {
            return;
        }

        event_helper
            .window()
            .set_cursor_position((event_helper.dimensions / 2.0).to_pos())
            .unwrap();

        let mov =
            event_helper.cursor_delta / event_helper.dimensions.y * speed::ROTATION * speed::MOUSE;
        // yaw
        event_helper.rotation *= Quat::from_axis_angle(-rotation::UP, mov.x);
        // pitch
        event_helper.rotation *= Quat::from_axis_angle(rotation::RIGHT, mov.y);

        event_helper.cursor_delta = Vec2::ZERO;

        let rot_time = event_helper.rot_time();

        if event_helper.key_held(VirtualKeyCode::Left) {
            event_helper.rotation *= Quat::from_axis_angle(rotation::UP, rot_time);
        }
        if event_helper.key_held(VirtualKeyCode::Right) {
            event_helper.rotation *= Quat::from_axis_angle(-rotation::UP, rot_time);
        }
        if event_helper.key_held(VirtualKeyCode::Up) {
            event_helper.rotation *= Quat::from_axis_angle(-rotation::RIGHT, rot_time);
        }
        if event_helper.key_held(VirtualKeyCode::Down) {
            event_helper.rotation *= Quat::from_axis_angle(rotation::RIGHT, rot_time);
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

        // neutralizes roll
        event_helper.rotation.y = 0.0;
        event_helper.rotation = event_helper.rotation.normalize();

        push_constants.rot = event_helper.rotation.to_array();
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

                graphics_pipeline = get_graphics_pipeline(
                    device.clone(),
                    vertex_shader.clone(),
                    fragment_shader.clone(),
                    viewport.clone(),
                    render_pass.clone(),
                );

                descriptor_set =
                    get_descriptor_set(graphics_pipeline.clone(), queue.clone(), viewport.clone());
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

        command_buffers[image_index] = Some(get_primary_command_buffer(
            device.clone(),
            queue.clone(),
            graphics_pipeline.clone(),
            framebuffers[image_index].clone(),
            push_constants.clone(),
            descriptor_set.clone(),
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
