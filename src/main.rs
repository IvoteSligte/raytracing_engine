use std::{sync::Arc, time::Instant, ops::Mul};

use extend::ext;
use glam::{Quat, Vec3, UVec2, Vec2};
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
    Version,
    VulkanLibrary,
};
use vulkano_win::VkSurfaceBuild;
use winit::{
    dpi::PhysicalPosition,
    event::{Event, VirtualKeyCode, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::{Fullscreen, Window, WindowBuilder},
};
use winit_input_helper::WinitInputHelper;

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

fn get_window_center(window: &Window) -> Vec2 {
    let size = window.inner_size();
    UVec2::new(size.width / 2, size.height / 2).as_vec2()
}

#[allow(dead_code)]
mod speed {
    pub const MOVEMENT: f32 = 2.0;
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

fn main() {
    let library = VulkanLibrary::new().unwrap();
    let required_extensions = vulkano_win::required_extensions(&library);
    let instance = Instance::new(library, InstanceCreateInfo {
        enabled_extensions: required_extensions,
        engine_version: Version::V1_3,
        ..Default::default()
    })
    .unwrap();

    let event_loop = EventLoop::new();
    let surface = WindowBuilder::new().build_vk_surface(&event_loop, instance.clone()).unwrap();
    surface.window().set_cursor_visible(false);

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

    let mut window_center = get_window_center(surface.window());

    let mut input_helper = WinitInputHelper::new();

    #[cfg(debug_assertions)]
    let mut fps_helper = fps_counter::FPSCounter::new();

    let mut last_update = Instant::now();

    let mut window_resized = false;
    let mut window_frozen = false;
    let mut recreate_swapchain = false;

    // TODO: camera rolls when rotating with mouse
    // TODO: fix rotation when resizing
    event_loop.run(move |event, _, control_flow| {
        if let Event::WindowEvent {
            event: WindowEvent::Focused(focused),
            ..
        } = event
        {
            window_frozen = !focused;
        }

        if !input_helper.update(&event) {
            return;
        }

        if input_helper.quit() || input_helper.key_released(VirtualKeyCode::Escape) {
            *control_flow = ControlFlow::Exit;
        }

        if let Some(size) = input_helper.window_resized() {
            // window minimized
            window_frozen = size.width == 0 || size.height == 0;
            window_resized = true;
            window_center = get_window_center(surface.window());
            viewport.dimensions = window_center.mul(2.0).to_array();
        }

        surface.window().set_cursor_position(window_center.to_pos()).unwrap();
        
        if window_frozen {
            return;
        }

        #[cfg(debug_assertions)]
        println!("{}", fps_helper.tick());

        if input_helper.key_pressed(VirtualKeyCode::F11) {
            let window = surface.window();
            match window.fullscreen() {
                Some(_) => window.set_fullscreen(None),
                None => window.set_fullscreen(Some(Fullscreen::Borderless(None))),
            }
        }

        let delta_time = last_update.elapsed().as_secs_f32();
        last_update = Instant::now();

        let mut rotation = Quat::from_array(push_constants.rot);

        if !window_resized {
            let pos = input_helper
                .mouse()
                .map(|(x, y)| Vec2::new(x, y))
                .unwrap_or(window_center);
    
            let mov = (pos - window_center) / Vec2::from_array(viewport.dimensions) * speed::ROTATION;
            println!("mov: {}, wc: {}, pos: {}", mov, window_center, pos);
            // yaw
            rotation *= Quat::from_axis_angle(-rotation::UP, mov.x);
            // pitch
            rotation *= Quat::from_axis_angle(rotation::RIGHT, mov.y);
        }

        let rot_time = delta_time * speed::ROTATION;

        if input_helper.key_held(VirtualKeyCode::Left) {
            rotation *= Quat::from_axis_angle(rotation::UP, rot_time);
        }
        if input_helper.key_held(VirtualKeyCode::Right) {
            rotation *= Quat::from_axis_angle(-rotation::UP, rot_time);
        }
        if input_helper.key_held(VirtualKeyCode::Down) {
            rotation *= Quat::from_axis_angle(rotation::RIGHT, rot_time);
        }
        if input_helper.key_held(VirtualKeyCode::Up) {
            rotation *= Quat::from_axis_angle(-rotation::RIGHT, rot_time);
        }

        push_constants.rot = rotation.to_array();

        let mov_time = delta_time * speed::MOVEMENT;

        let quaternion = Quat::from_array(push_constants.rot).normalize();
        let forward = quaternion.mul_vec3(rotation::FORWARD);
        let right = quaternion.mul_vec3(rotation::RIGHT);
        let up = quaternion.mul_vec3(rotation::UP);

        let mut position = Vec3::from_array(push_constants.pos);

        if input_helper.key_held(VirtualKeyCode::A) {
            position -= right * mov_time;
        }
        if input_helper.key_held(VirtualKeyCode::D) {
            position += right * mov_time;
        }
        if input_helper.key_held(VirtualKeyCode::W) {
            position += forward * mov_time;
        }
        if input_helper.key_held(VirtualKeyCode::S) {
            position -= forward * mov_time;
        }
        if input_helper.key_held(VirtualKeyCode::Q) {
            position += up * mov_time;
        }
        if input_helper.key_held(VirtualKeyCode::E) {
            position -= up * mov_time;
        }

        push_constants.pos = position.to_array();

        if recreate_swapchain || window_resized {
            recreate_swapchain = false;

            let dimensions = surface.window().inner_size();

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

            if window_resized {
                window_resized = false;

                viewport.dimensions = dimensions.into();

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
                    return recreate_swapchain = true;
                }
                Err(err) => panic!("{}", err),
            };
        recreate_swapchain |= suboptimal;

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
                recreate_swapchain = true;
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
