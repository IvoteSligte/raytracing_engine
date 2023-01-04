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

// TODO: increase first direct image resolution
mod shaders {
    vulkano_shaders::shader! {
        shaders: {
            Vertex: {
                ty: "vertex",
                path: "shaders/vertex.glsl",
            },
            Fragment: {
                ty: "fragment",
                path: "shaders/fragment.glsl",
            },
            Compute: {
                ty: "compute",
                path: "shaders/compute.glsl",
            },
        },
        types_meta: { #[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)] },
        include: ["shaders/utilities.glsl"],
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
    fragment_spec_consts: shaders::FragmentSpecializationConstants,
) -> Arc<GraphicsPipeline> {
    GraphicsPipeline::start()
        .vertex_input_state(VertexInputState::new())
        .vertex_shader(vertex_shader.entry_point("main").unwrap(), ())
        .input_assembly_state(InputAssemblyState::new()) // might be unnecessary
        .fragment_shader(
            fragment_shader.entry_point("main").unwrap(),
            fragment_spec_consts,
        )
        .viewport_state(ViewportState::viewport_fixed_scissor_irrelevant([viewport]))
        .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
        .build(device.clone())
        .unwrap()
}

fn get_compute_pipeline(
    device: Arc<Device>,
    compute_shader: Arc<ShaderModule>,
    spec_consts: shaders::ComputeSpecializationConstants,
) -> Arc<ComputePipeline> {
    ComputePipeline::new(
        device.clone(),
        compute_shader.entry_point("main").unwrap(),
        &spec_consts,
        None,
        |_| {},
    )
    .unwrap()
}

fn get_graphics_descriptor_set(
    pipeline: Arc<GraphicsPipeline>,
    image_view: Arc<dyn ImageViewAbstract>,
    mutable_buffer: Arc<dyn BufferAccess>,
    constant_buffer: Arc<dyn BufferAccess>,
) -> Arc<PersistentDescriptorSet> {
    PersistentDescriptorSet::new(
        pipeline.layout().set_layouts()[0].clone(),
        [
            WriteDescriptorSet::image_view(0, image_view.clone()),
            WriteDescriptorSet::buffer(1, mutable_buffer),
            WriteDescriptorSet::buffer(2, constant_buffer),
        ],
    )
    .unwrap()
}

fn get_compute_descriptor_set(
    pipeline: Arc<ComputePipeline>,
    image_views: Vec<Arc<dyn ImageViewAbstract>>,
    mutable_buffer: Arc<dyn BufferAccess>,
    constant_buffer: Arc<dyn BufferAccess>,
) -> Arc<PersistentDescriptorSet> {
    PersistentDescriptorSet::new(
        pipeline.layout().set_layouts()[0].clone(),
        [
            WriteDescriptorSet::image_view_array(0, 0, image_views),
            WriteDescriptorSet::buffer(1, mutable_buffer),
            WriteDescriptorSet::buffer(2, constant_buffer),
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
    push_constants: shaders::ty::PushConstantData,
    constants: shaders::ty::ConstantBuffer,
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
        let compute_pc = shaders::ty::PushConstantData {
            iter: i as u32,
            imageSize: ((1 << (compute_image_count - 1 - i)) as f32
                / Vec2::from_array(constants.view))
            .to_array(),
            ..push_constants
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

// distance from the camera at which the ray gets cut off
const RENDER_DIST: f32 = 1000.0;
// field of view
const FOV: f32 = 1.0;

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
    surface.window().set_resizable(false);

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
                // any R8G8B8A8_.NORM variant
                Format::R8G8B8A8_UNORM
                | Format::R8G8B8A8_SNORM
                | Format::B8G8R8A8_UNORM
                | Format::B8G8R8A8_SNORM => 1,
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

    let vertex_shader = shaders::load_Vertex(device.clone()).unwrap();
    let fragment_shader = shaders::load_Fragment(device.clone()).unwrap();

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
        shaders::FragmentSpecializationConstants { RENDER_DIST },
    );

    let materials: [shaders::ty::Material; 4] = [
        shaders::ty::Material {
            color: [0.2, 0.2, 1.0],
            diffuse: 1.0,
            specular: 1.0,
            shine: 1.0,
            ambient: 0.05,
            _dummy0: [0u8; 4],
        },
        shaders::ty::Material {
            color: [0.1, 1.0, 0.1],
            diffuse: 1.0,
            specular: 1.0,
            shine: 10.0,
            ambient: 0.05,
            _dummy0: [0u8; 4],
        },
        shaders::ty::Material {
            color: [1.0, 1.0, 0.1],
            diffuse: 1.0,
            specular: 1.0,
            shine: 1.0,
            ambient: 0.05,
            _dummy0: [0u8; 4],
        },
        shaders::ty::Material {
            color: [1.0, 0.1, 0.1],
            diffuse: 1.0,
            specular: 1.0,
            shine: 1.0,
            ambient: 0.05,
            _dummy0: [0u8; 4],
        },
    ];

    let objects = [
        shaders::ty::Object {
            pos: [5.0, 5.0, -1.0],
            size: 3.0,
        },
        shaders::ty::Object {
            pos: [5.0, 4.0, 10.0],
            size: 6.0,
        },
        shaders::ty::Object {
            pos: [-3.0, 3.0, -3.0],
            size: 1.0,
        },
        shaders::ty::Object {
            pos: [4.0, -1.0, 0.0],
            size: 2.0,
        },
    ];

    let lights = [
        shaders::ty::Light {
            pos: [-1.0, 0.0, -3.0],
            _dummy0: [0u8; 4],
            color: [0.1, 0.5, 0.6],
            _dummy1: [0u8; 4],
        },
        shaders::ty::Light {
            pos: [8.0, -5.0, 10.0],
            _dummy0: [0u8; 4],
            color: [1.2, 0.2, 0.3],
            _dummy1: [0u8; 4],
        },
    ];

    let mut mutable_data = shaders::ty::MutableData {
        matCount: materials.len() as u32,
        objCount: objects.len() as u32,
        lightCount: lights.len() as u32,
        ..Default::default()
    };
    mutable_data.mats[..materials.len()].copy_from_slice(&materials);
    mutable_data.objs[..objects.len()].copy_from_slice(&objects);
    mutable_data.lights[..lights.len()].copy_from_slice(&lights);

    let (mutable_buffer, mutable_future) =
        DeviceLocalBuffer::from_data(mutable_data, BufferUsage::uniform_buffer(), queue.clone())
            .unwrap();

    // near constant data
    let mut constant_data = shaders::ty::ConstantBuffer {
        view: viewport.dimensions,
        ratio: [FOV, FOV * viewport.dimensions[1] / viewport.dimensions[0]],
    };

    let (constant_buffer, constant_future) =
        DeviceLocalBuffer::from_data(constant_data, BufferUsage::uniform_buffer(), queue.clone())
            .unwrap();

    mutable_future
        .join(constant_future)
        .then_signal_fence_and_flush()
        .unwrap()
        .wait(None)
        .unwrap();

    let mut push_constants = shaders::ty::PushConstantData {
        rot: [0.0, 0.0, 0.0, 0.0],
        pos: [0.0, 0.0, 0.0],
        iter: 0,
        imageSize: [1.0; 2],
    };

    let compute_shader = shaders::load_Compute(device.clone()).unwrap();

    let compute_pipeline = get_compute_pipeline(
        device.clone(),
        compute_shader.clone(),
        shaders::ComputeSpecializationConstants { RENDER_DIST },
    );

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
        mutable_buffer.clone(),
        constant_buffer.clone(),
    );

    let mut compute_descriptor_set = get_compute_descriptor_set(
        compute_pipeline.clone(),
        compute_image_views.clone(),
        mutable_buffer.clone(),
        constant_buffer.clone(),
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

        // TODO: make movement independent of framerate
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

                constant_data = shaders::ty::ConstantBuffer {
                    view: viewport.dimensions,
                    ratio: [FOV, FOV * viewport.dimensions[1] / viewport.dimensions[0]],
                };

                // TODO: make this a lot cleaner
                let mut builder = AutoCommandBufferBuilder::primary(
                    device.clone(),
                    queue.family(),
                    CommandBufferUsage::OneTimeSubmit,
                )
                .unwrap();

                builder
                    .update_buffer(Box::new(constant_data), constant_buffer.clone(), 0)
                    .unwrap();
                let command_buffer = builder.build().unwrap();

                match fences[previous_fence_index].clone() {
                    Some(future) => {
                        future
                            .then_signal_fence_and_flush()
                            .unwrap()
                            .wait(None)
                            .unwrap();
                        sync::now(device.clone())
                            .then_execute(queue.clone(), command_buffer)
                            .unwrap()
                            .then_signal_fence_and_flush()
                            .unwrap()
                            .wait(None)
                            .unwrap();
                    }
                    None => (),
                }

                graphics_pipeline = get_graphics_pipeline(
                    device.clone(),
                    vertex_shader.clone(),
                    fragment_shader.clone(),
                    viewport.clone(),
                    render_pass.clone(),
                    shaders::FragmentSpecializationConstants { RENDER_DIST },
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
                    constant_buffer.clone(),
                );

                graphics_descriptor_set = get_graphics_descriptor_set(
                    graphics_pipeline.clone(),
                    compute_image_views[compute_image_count - 1].clone(),
                    mutable_buffer.clone(),
                    constant_buffer.clone(),
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
            constant_data.clone(),
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
