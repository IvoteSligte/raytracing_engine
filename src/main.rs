use std::sync::Arc;

use vulkano::{
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferUsage, PrimaryAutoCommandBuffer,
        RenderPassBeginInfo, SubpassContents,
    },
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
            viewport::{Viewport, ViewportState}, vertex_input::VertexInputState,
        },
        GraphicsPipeline, Pipeline, PipelineBindPoint,
    },
    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass},
    shader::ShaderModule,
    swapchain::{
        self, AcquireError, Surface, Swapchain, SwapchainCreateInfo, SwapchainCreationError,
    },
    sync::{self, FenceSignalFuture, FlushError, GpuFuture}, descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet}, buffer::{ImmutableBuffer, BufferUsage},
};
use vulkano_win::VkSurfaceBuild;
use winit::{
    event::{ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::{Fullscreen, Window, WindowBuilder},
};

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

layout(binding = 0) uniform ViewportBuffer {
    vec2 viewport;
} ubo;

layout(location = 0) out vec4 fragColor;

void main() {
    vec2 normCoord = gl_FragCoord.xy / ubo.viewport - 0.5;
    normCoord.x *= ubo.viewport.x / ubo.viewport.y;

    vec4 light = vec4(4.0, 3.0, 4.0, 100.0); // w = strength
    vec4 sphere = vec4(0.0, 0.0, 5.0, 2.0); // w = radius

    vec3 colorMat = vec3(0.2, 0.2, 1.0);
    float specularMat = 1.0;
    float diffuseMat = 1.0;
    float shineMat = 100.0;

    float ambientScene = 0.8;

    vec3 fragPos = vec3(normCoord, 1.0);
    vec3 position = fragPos;
    vec3 step = normalize(fragPos);

    for (uint i = 0; i < 100; i++) {
        if (distance(position, sphere.xyz) <= sphere.w) {
            vec3 normal = normalize(position - sphere.xyz);
            vec3 lightDir = normalize(light.xyz - position); // might be light.xyz - fragPos
            vec3 reflection = reflect(lightDir, normal);

            float lightDist = distance(position, light.xyz);
            float lightDistFallOff = 1 / (lightDist * lightDist);

            float camDist = distance(position, fragPos);
            float camDistFallOff = 1 / (camDist * camDist);

            vec3 ambient = ambientScene * colorMat;
            vec3 diffuse = max(dot(normal, lightDir), 0.0) * light.w * diffuseMat * colorMat * lightDistFallOff;
            vec3 specular = max(pow(dot(reflection, fragPos), shineMat), 0.0) * specularMat * colorMat * lightDistFallOff;

            fragColor = vec4((ambient + diffuse + specular) * camDistFallOff, 1.0);
            return;
        }
        position += step;
    }
    fragColor = vec4(0.0, 0.0, 0.0, 1.0);
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
        .input_assembly_state(InputAssemblyState::new())
        .fragment_shader(fragment_shader.entry_point("main").unwrap(), ())
        .viewport_state(ViewportState::viewport_fixed_scissor_irrelevant([viewport]))
        .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
        .build(device.clone())
        .unwrap()
}

fn get_command_buffers(
    device: Arc<Device>,
    queue: Arc<Queue>,
    graphics_pipeline: Arc<GraphicsPipeline>,
    framebuffers: &Vec<Arc<Framebuffer>>,
    viewport: Viewport,
) -> Vec<Arc<PrimaryAutoCommandBuffer>> {
    let (uniform_buffer, _) = ImmutableBuffer::from_data(viewport.dimensions, BufferUsage::uniform_buffer(), queue.clone()).unwrap();
    let descriptor_set = PersistentDescriptorSet::new(graphics_pipeline.layout().set_layouts()[0].clone(), [WriteDescriptorSet::buffer(0, uniform_buffer.clone())]).unwrap();

    framebuffers
        .iter()
        .map(|framebuffer| {
            let mut builder = AutoCommandBufferBuilder::primary(
                device.clone(),
                queue.family(),
                CommandBufferUsage::MultipleSubmit,
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
                .bind_descriptor_sets(PipelineBindPoint::Graphics, graphics_pipeline.layout().clone(), 0, descriptor_set.clone())
                .draw(3, 1, 0, 0)
                .unwrap()
                .end_render_pass()
                .unwrap();

            Arc::new(builder.build().unwrap())
        })
        .collect()
}

fn main() {
    let required_extensions = vulkano_win::required_extensions();
    let instance = Instance::new(InstanceCreateInfo {
        enabled_extensions: required_extensions,
        ..Default::default()
    })
    .unwrap();

    let event_loop = EventLoop::new();
    let surface = WindowBuilder::new()
        .build_vk_surface(&event_loop, instance.clone())
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

    let framebuffers = get_framebuffers(&images, render_pass.clone());

    let vertex_shader = vertex_shader::load(device.clone()).unwrap();
    let fragment_shader = fragment_shader::load(device.clone()).unwrap();

    let mut viewport = Viewport {
        origin: [0.0, 0.0],
        dimensions: surface.window().inner_size().into(),
        depth_range: 0.0..1.0,
    };

    let graphics_pipeline = get_graphics_pipeline(
        device.clone(),
        vertex_shader.clone(),
        fragment_shader.clone(),
        viewport.clone(),
        render_pass.clone(),
    );

    let mut command_buffers = get_command_buffers(
        device.clone(),
        queue.clone(),
        graphics_pipeline,
        &framebuffers,
        viewport.clone(),
    );

    let mut fences: Vec<Option<Arc<FenceSignalFuture<_>>>> = vec![None; images.len()];
    let mut previous_fence_index = 0;

    let mut window_minimized = false;
    let mut window_resized = false;
    let mut recreate_swapchain = false;

    event_loop.run(move |event, _, control_flow| match event {
        Event::WindowEvent {
            event: WindowEvent::CloseRequested,
            ..
        } => {
            *control_flow = ControlFlow::Exit;
        }
        Event::WindowEvent {
            event: WindowEvent::Resized(size),
            ..
        } => {
            window_minimized = size.width == 0 || size.height == 0;
            window_resized = true;
        }
        Event::WindowEvent {
            event:
                WindowEvent::KeyboardInput {
                    input:
                        KeyboardInput {
                            virtual_keycode: Some(VirtualKeyCode::F11),
                            state: ElementState::Pressed,
                            ..
                        },
                    ..
                },
            ..
        } => {
            let window = surface.window();
            match window.fullscreen() {
                Some(_) => window.set_fullscreen(None),
                None => window.set_fullscreen(Some(Fullscreen::Borderless(None))),
            }
        }
        Event::MainEventsCleared if !window_minimized => {
            if recreate_swapchain || recreate_swapchain {
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

                let framebuffers = get_framebuffers(&images, render_pass.clone());

                if window_resized {
                    window_resized = false;

                    viewport.dimensions = dimensions.into();
                    let graphics_pipeline = get_graphics_pipeline(
                        device.clone(),
                        vertex_shader.clone(),
                        fragment_shader.clone(),
                        viewport.clone(),
                        render_pass.clone(),
                    );
                    command_buffers = get_command_buffers(
                        device.clone(),
                        queue.clone(),
                        graphics_pipeline,
                        &framebuffers,
                        viewport.clone(),
                    );
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
                .then_execute(queue.clone(), command_buffers[image_index].clone())
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
                    println!("{}", err);
                    None
                }
            };
            previous_fence_index = image_index;
        }
        _ => (),
    })
}
