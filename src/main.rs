use std::{
    collections::HashSet,
    ffi::{CStr, CString, NulError},
    sync::mpsc::Receiver,
};

use anyhow::{anyhow, Result};
use ash::{
    extensions::{
        ext::DebugUtils,
        khr::{Surface, Swapchain},
    },
    vk,
    vk::{Handle as _, Rect2D},
    Device, Entry, Instance,
};
use glfw::{ClientApiHint, WindowEvent, WindowHint};
use lazy_static::lazy_static;

const WIDTH: u32 = 800;
const HEIGHT: u32 = 600;

const VALIDATION_LAYERS: [&str; 1] = ["VK_LAYER_KHRONOS_validation"];
const ENABLE_VALIDATION_LAYERS: bool = true;
lazy_static! {
    static ref DEVICE_EXTENSIONS: [&'static CStr; 1] = [ash::extensions::khr::Swapchain::name()];
}

#[derive(Default)]
struct HelloTriangleApplication {
    glfw: Option<glfw::Glfw>,
    window: Option<glfw::Window>,
    events: Option<Receiver<(f64, WindowEvent)>>,
    instance: Option<Instance>,
    entry: Option<Entry>,
    physical_device: Option<vk::PhysicalDevice>,
    device: Option<Device>,
    graphics_queue: Option<vk::Queue>,
    present_queue: Option<vk::Queue>,
    surface: Option<(Surface, vk::SurfaceKHR)>,
    swap_chain: Option<(Swapchain, vk::SwapchainKHR)>,
    swap_chain_images: Vec<vk::Image>,
    swap_chain_image_format: Option<vk::Format>,
    swap_chain_extent: Option<vk::Extent2D>,
    swap_chain_image_views: Vec<vk::ImageView>,
    render_pass: Option<vk::RenderPass>,
    pipeline_layout: Option<vk::PipelineLayout>,
    graphics_pipeline: Option<vk::Pipeline>,
    swap_chain_framebuffers: Vec<vk::Framebuffer>,
    command_pool: Option<vk::CommandPool>,
    command_buffers: Vec<vk::CommandBuffer>,
    image_available_semaphore: Option<vk::Semaphore>,
    render_finished_semaphore: Option<vk::Semaphore>,
}

impl HelloTriangleApplication {
    fn new() -> Self {
        Default::default()
    }

    pub fn run(&mut self) -> Result<()> {
        self.init_window()?;
        self.init_vulkan()?;
        self.main_loop()?;

        Ok(())
    }

    fn init_window(&mut self) -> Result<()> {
        let mut glfw = glfw::init(glfw::FAIL_ON_ERRORS)?;

        glfw.window_hint(WindowHint::Resizable(false));
        glfw.window_hint(WindowHint::ClientApi(ClientApiHint::NoApi));
        let (window, events) = glfw
            .create_window(WIDTH, HEIGHT, "Hello triangle!", glfw::WindowMode::Windowed)
            .ok_or_else(|| anyhow!("Failed to create GLFW window."))?;

        self.glfw = Some(glfw);
        self.window = Some(window);
        self.events = Some(events);
        Ok(())
    }

    fn init_vulkan(&mut self) -> Result<()> {
        self.create_instance()?;
        self.create_surface()?;
        self.pick_physical_device()?;
        self.create_logical_device()?;
        self.create_swap_chain()?;
        self.create_image_views()?;
        self.create_render_pass()?;
        self.create_graphics_pipeline()?;
        self.create_framebuffers()?;
        self.create_command_pool()?;
        self.create_command_buffers()?;
        self.create_semaphores()?;
        Ok(())
    }

    fn create_instance(&mut self) -> Result<()> {
        let glfw = self.glfw.as_ref().ok_or_else(|| anyhow!("no glfw"))?;
        let entry = Entry::linked();

        let app_name = CString::new("Vulkan Application").unwrap();
        let engine_name = CString::new("No Engine").unwrap();

        let app_info = vk::ApplicationInfo::builder()
            .application_name(app_name.as_c_str())
            .application_version(vk::make_api_version(0, 0, 1, 0))
            .engine_name(engine_name.as_c_str())
            .engine_version(vk::make_api_version(0, 0, 1, 0))
            .api_version(vk::API_VERSION_1_0)
            .build();

        let required_extensions = glfw
            .get_required_instance_extensions()
            .ok_or_else(|| anyhow!("could not get instance extensions"))?
            .into_iter()
            .map(CString::new)
            .collect::<std::result::Result<Vec<_>, _>>()?;

        let mut required_extensions: Vec<_> =
            required_extensions.iter().map(|x| x.as_ptr()).collect();

        if ENABLE_VALIDATION_LAYERS {
            required_extensions.push(DebugUtils::name().as_ptr());
        }

        let requested_layers = VALIDATION_LAYERS
            .into_iter()
            .map(|x| Ok(CString::new(x)?))
            .collect::<Result<Vec<_>, NulError>>()?;
        let requested_layer_ptrs = requested_layers
            .iter()
            .map(|x| x.as_ptr())
            .collect::<Vec<_>>();

        let create_info = vk::InstanceCreateInfo::builder()
            .application_info(&app_info)
            .enabled_extension_names(&required_extensions);
        let create_info = if ENABLE_VALIDATION_LAYERS {
            Self::check_validation_layer_support(&entry)?;
            create_info.enabled_layer_names(&requested_layer_ptrs)
        } else {
            create_info
        };
        let instance = unsafe { entry.create_instance(&create_info, None)? };

        self.entry = Some(entry);
        self.instance = Some(instance);

        Ok(())
    }

    fn create_surface(&mut self) -> Result<()> {
        let entry = self.entry.as_ref().ok_or_else(|| anyhow!("no entry"))?;
        let instance = self
            .instance
            .as_ref()
            .ok_or_else(|| anyhow!("no instance"))?;

        let surface = Surface::new(entry, instance);
        let window = self.window.as_ref().ok_or_else(|| anyhow!("no window"))?;
        let mut surface_khr_raw_handle = 0;
        if vk_sys::SUCCESS
            != window.create_window_surface(
                instance.handle().as_raw() as usize,
                std::ptr::null(),
                &mut surface_khr_raw_handle,
            )
        {
            return Err(anyhow!("could not create surface"));
        }

        let surface_khr = vk::Handle::from_raw(surface_khr_raw_handle);
        self.surface = Some((surface, surface_khr));
        Ok(())
    }

    fn pick_physical_device(&mut self) -> Result<()> {
        let (surface, surface_khr) = self
            .surface
            .as_ref()
            .ok_or_else(|| anyhow!("no surface"))?
            .to_owned();
        let instance = self
            .instance
            .as_ref()
            .ok_or_else(|| anyhow!("no instance"))?;
        let devices = unsafe { instance.enumerate_physical_devices()? };
        if devices.is_empty() {
            return Err(anyhow!("no devices found"));
        }
        let is_suitable = |device: &vk::PhysicalDevice| -> bool {
            let indices = if let Ok(indices) =
                QueueFamilyIndices::new(instance, device, &surface, &surface_khr)
            {
                indices
            } else {
                return false;
            };

            let extensions = unsafe {
                match instance.enumerate_device_extension_properties(*device) {
                    Ok(extensions) => extensions,
                    Err(_) => return false,
                }
            };

            let mut required_extensions = DEVICE_EXTENSIONS.into_iter().collect::<HashSet<_>>();

            for ex in extensions {
                required_extensions.remove(unsafe {
                    CString::from_vec_unchecked(
                        ex.extension_name
                            .into_iter()
                            .map(|x| x as u8)
                            .take_while(|x| *x > 0)
                            .collect::<Vec<_>>(),
                    )
                    .as_c_str()
                });
            }
            let extensions_supported = required_extensions.is_empty();

            let swap_chain_adequate = if extensions_supported {
                if let Ok(swap_chain_support) =
                    SwapChainSupportDetails::new(device, &surface, &surface_khr)
                {
                    !swap_chain_support.formats.is_empty()
                        && !swap_chain_support.present_modes.is_empty()
                } else {
                    return false;
                }
            } else {
                false
            };

            indices.is_complete() && extensions_supported && swap_chain_adequate
        };

        let device = devices
            .into_iter()
            .find(is_suitable)
            .ok_or_else(|| anyhow!("no suitable device"))?;
        self.physical_device = Some(device);

        Ok(())
    }

    fn check_validation_layer_support(entry: &Entry) -> Result<()> {
        let available_layers = entry.enumerate_instance_layer_properties()?;

        for requested_layer in VALIDATION_LAYERS.into_iter() {
            let mut layer_found = false;
            for l in available_layers.iter() {
                if CString::new(
                    l.layer_name
                        .into_iter()
                        .map(|x| x as u8)
                        .take_while(|x| *x != 0)
                        .collect::<Vec<_>>(),
                )? == CString::new(requested_layer)?
                {
                    layer_found = true;
                    break;
                }
            }

            if !layer_found {
                return Err(anyhow!(
                    "missing requested validation layer \"{}\"",
                    requested_layer
                ));
            }
        }
        return Ok(());
    }

    fn create_logical_device(&mut self) -> Result<()> {
        let physical_device = self.physical_device.ok_or_else(|| anyhow!("no device"))?;
        let (surface, surface_khr) = self.surface.as_ref().ok_or_else(|| anyhow!("no surface"))?;
        let instance = self
            .instance
            .as_ref()
            .ok_or_else(|| anyhow!("no instance"))?;
        let indices = QueueFamilyIndices::new(instance, &physical_device, &surface, &surface_khr)?;

        let queue_families = HashSet::from([
            indices
                .graphics_family
                .ok_or_else(|| anyhow!("no graphics_family"))?,
            indices
                .present_family
                .ok_or_else(|| anyhow!("no present_family"))?,
        ]);
        let queue_create_infos = queue_families
            .into_iter()
            .map(|queue_family_index| {
                Ok(vk::DeviceQueueCreateInfo {
                    queue_family_index,
                    queue_count: 1,
                    p_queue_priorities: &1.0,
                    ..Default::default()
                })
            })
            .collect::<Result<Vec<_>>>()?;

        let device_features = vk::PhysicalDeviceFeatures::default();
        let extensions = DEVICE_EXTENSIONS
            .into_iter()
            .map(CStr::as_ptr)
            .collect::<Vec<_>>();
        let create_info = vk::DeviceCreateInfo {
            p_queue_create_infos: queue_create_infos.as_ptr(),
            queue_create_info_count: queue_create_infos.len() as u32,
            p_enabled_features: &device_features,
            enabled_extension_count: extensions.len() as u32,
            pp_enabled_extension_names: extensions.as_ptr(),
            ..Default::default()
        };
        let device = unsafe { instance.create_device(physical_device, &create_info, None)? };

        let (graphics_queue, present_queue) = unsafe {
            let graphics_queue = device.get_device_queue(
                indices
                    .graphics_family
                    .ok_or_else(|| anyhow!("no graphics queue"))?,
                0,
            );
            let present_queue = device.get_device_queue(
                indices
                    .present_family
                    .ok_or_else(|| anyhow!("no graphics queue"))?,
                0,
            );
            (graphics_queue, present_queue)
        };

        self.device = Some(device);
        self.graphics_queue = Some(graphics_queue);
        self.present_queue = Some(present_queue);

        Ok(())
    }

    fn create_swap_chain(&mut self) -> Result<()> {
        let device = self.device.as_ref().ok_or_else(|| anyhow!("no device"))?;
        let physical_device = self
            .physical_device
            .as_ref()
            .ok_or_else(|| anyhow!("no device"))?;
        let instance = self
            .instance
            .as_ref()
            .ok_or_else(|| anyhow!("no instance"))?;
        let (surface, surface_khr) = self.surface.as_ref().ok_or_else(|| anyhow!("no surface"))?;
        let swap_chain_support =
            SwapChainSupportDetails::new(physical_device, &surface, &surface_khr)?;

        let surface_format = Self::choose_swap_surface_format(&swap_chain_support.formats);
        let image_format = surface_format.format;
        let present_mode = Self::choose_swap_present_mode(&swap_chain_support.present_modes);
        let image_extent = self.choose_swap_extent(&swap_chain_support.capabilities)?;

        let min_image_count = if swap_chain_support.capabilities.max_image_count > 0 {
            swap_chain_support
                .capabilities
                .max_image_count
                .min(swap_chain_support.capabilities.min_image_count + 1)
        } else {
            swap_chain_support.capabilities.min_image_count + 1
        };

        let mut create_info = vk::SwapchainCreateInfoKHR {
            surface: *surface_khr,
            min_image_count,
            image_format,
            image_color_space: surface_format.color_space,
            image_extent,
            image_array_layers: 1,
            image_usage: vk::ImageUsageFlags::COLOR_ATTACHMENT,
            image_sharing_mode: vk::SharingMode::EXCLUSIVE,
            pre_transform: swap_chain_support.capabilities.current_transform,
            composite_alpha: vk::CompositeAlphaFlagsKHR::OPAQUE,
            present_mode,
            clipped: vk::TRUE,
            ..Default::default()
        };

        let indices = QueueFamilyIndices::new(instance, physical_device, surface, surface_khr)?;
        let queue_familie_indices = [
            indices.graphics_family.unwrap(),
            indices.present_family.unwrap(),
        ];
        if indices.graphics_family != indices.present_family {
            create_info.image_sharing_mode = vk::SharingMode::CONCURRENT;
            create_info.queue_family_index_count = 2;
            create_info.p_queue_family_indices = queue_familie_indices.as_ptr();
        }

        let swap_chain = Swapchain::new(instance, device);
        let swap_chain_khr = unsafe { swap_chain.create_swapchain(&create_info, None)? };
        self.swap_chain_images = unsafe { swap_chain.get_swapchain_images(swap_chain_khr)? };
        self.swap_chain = Some((swap_chain, swap_chain_khr));
        self.swap_chain_extent = Some(image_extent);
        self.swap_chain_image_format = Some(image_format);

        Ok(())
    }

    fn choose_swap_surface_format(
        available_formats: &[vk::SurfaceFormatKHR],
    ) -> vk::SurfaceFormatKHR {
        for format in available_formats {
            if format.format == vk::Format::B8G8R8A8_SRGB
                && format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
            {
                return *format;
            }
        }
        available_formats[0]
    }

    fn choose_swap_present_mode(available_modes: &[vk::PresentModeKHR]) -> vk::PresentModeKHR {
        if available_modes.contains(&vk::PresentModeKHR::MAILBOX) {
            vk::PresentModeKHR::MAILBOX
        } else {
            vk::PresentModeKHR::FIFO
        }
    }

    fn choose_swap_extent(
        &self,
        capabilities: &vk::SurfaceCapabilitiesKHR,
    ) -> Result<vk::Extent2D> {
        let window = self.window.as_ref().ok_or_else(|| anyhow!("no window"))?;
        if capabilities.current_extent.width != u32::MAX {
            Ok(capabilities.current_extent)
        } else {
            let (width, height) = window.get_framebuffer_size();
            let width = width as u32;
            let height = height as u32;
            Ok(vk::Extent2D {
                width: width.clamp(
                    capabilities.min_image_extent.width,
                    capabilities.max_image_extent.width,
                ),
                height: height.clamp(
                    capabilities.min_image_extent.height,
                    capabilities.max_image_extent.height,
                ),
            })
        }
    }

    fn create_image_views(&mut self) -> Result<()> {
        let device = self.device.as_ref().ok_or_else(|| anyhow!("no device"))?;
        let format = self
            .swap_chain_image_format
            .ok_or_else(|| anyhow!("no image format"))?;
        self.swap_chain_image_views = self
            .swap_chain_images
            .iter()
            .copied()
            .map(|image| {
                let create_info = vk::ImageViewCreateInfo {
                    image,
                    view_type: vk::ImageViewType::TYPE_2D,
                    format,
                    components: Default::default(),
                    // vk::ComponentMapping {
                    // r: vk::ComponentSwizzle::IDENTITY,
                    // g: vk::ComponentSwizzle::IDENTITY,
                    // b: vk::ComponentSwizzle::IDENTITY,
                    // a: vk::ComponentSwizzle::IDENTITY,
                    // },
                    subresource_range: vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    },
                    ..Default::default()
                };
                unsafe { device.create_image_view(&create_info, None) }
            })
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(())
    }

    fn create_render_pass(&mut self) -> Result<()> {
        let device = self.device.as_ref().ok_or_else(|| anyhow!("no device"))?;
        let format = self
            .swap_chain_image_format
            .ok_or_else(|| anyhow!("no image format"))?;
        let color_attachment = vk::AttachmentDescription {
            format,
            samples: vk::SampleCountFlags::TYPE_1,
            load_op: vk::AttachmentLoadOp::CLEAR,
            store_op: vk::AttachmentStoreOp::STORE,
            stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
            stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
            initial_layout: vk::ImageLayout::UNDEFINED,
            final_layout: vk::ImageLayout::PRESENT_SRC_KHR,

            ..Default::default()
        };

        let color_attachment_ref = vk::AttachmentReference {
            attachment: 0,
            layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        };

        let subpass = vk::SubpassDescription {
            pipeline_bind_point: vk::PipelineBindPoint::GRAPHICS,
            color_attachment_count: 1,
            p_color_attachments: &color_attachment_ref,
            ..Default::default()
        };

        let dependency = vk::SubpassDependency {
            src_subpass: vk::SUBPASS_EXTERNAL,
            src_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            src_access_mask: vk::AccessFlags::default(),

            dst_subpass: 0,
            dst_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            dst_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,

            ..Default::default()
        };

        let render_pass_info = vk::RenderPassCreateInfo {
            attachment_count: 1,
            p_attachments: &color_attachment,
            subpass_count: 1,
            p_subpasses: &subpass,
            dependency_count: 1,
            p_dependencies: &dependency,

            ..Default::default()
        };

        let render_pass = unsafe { device.create_render_pass(&render_pass_info, None)? };

        self.render_pass = Some(render_pass);
        Ok(())
    }

    fn create_graphics_pipeline(&mut self) -> Result<()> {
        let device = self.device.as_ref().ok_or_else(|| anyhow!("no device"))?;
        let extent = self.swap_chain_extent.ok_or_else(|| anyhow!("no extent"))?;
        let render_pass = self.render_pass.ok_or_else(|| anyhow!("no render_pass"))?;

        let vert_shader_code = std::fs::read("shaders/vert.spv")?;
        let frag_shader_code = std::fs::read("shaders/frag.spv")?;

        let vert_shader_module = self.create_shader_module(&vert_shader_code)?;
        let frag_shader_module = self.create_shader_module(&frag_shader_code)?;

        let name = CString::new("main")?;
        let vert_shader_stage_info = vk::PipelineShaderStageCreateInfo {
            stage: vk::ShaderStageFlags::VERTEX,
            module: vert_shader_module,
            p_name: name.as_ptr(),
            ..Default::default()
        };
        let frag_shader_stage_info = vk::PipelineShaderStageCreateInfo {
            stage: vk::ShaderStageFlags::FRAGMENT,
            module: frag_shader_module,
            p_name: name.as_ptr(),
            ..Default::default()
        };

        let shader_stages = [vert_shader_stage_info, frag_shader_stage_info];

        let vertex_inupt_info = vk::PipelineVertexInputStateCreateInfo {
            // vertexInputInfo.vertexBindingDescriptionCount = 0;
            // vertexInputInfo.pVertexBindingDescriptions = nullptr; // Optional
            // vertexInputInfo.vertexAttributeDescriptionCount = 0;
            // vertexInputInfo.pVertexAttributeDescriptions = nullptr; // Optional
            ..Default::default()
        };

        let input_assemply_info = vk::PipelineInputAssemblyStateCreateInfo {
            topology: vk::PrimitiveTopology::TRIANGLE_LIST,
            primitive_restart_enable: vk::FALSE,
            ..Default::default()
        };

        let viewport = vk::Viewport {
            x: 0.,
            y: 0.,
            width: extent.width as f32,
            height: extent.height as f32,
            min_depth: 0.,
            max_depth: 0.,
        };

        let scissor = vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent,
        };

        let vieport_state_info = vk::PipelineViewportStateCreateInfo {
            viewport_count: 1,
            p_viewports: &viewport,
            scissor_count: 1,
            p_scissors: &scissor,
            ..Default::default()
        };

        let rasterizer_info = vk::PipelineRasterizationStateCreateInfo {
            depth_clamp_enable: vk::FALSE,
            rasterizer_discard_enable: vk::FALSE,
            polygon_mode: vk::PolygonMode::FILL,

            line_width: 1.,

            cull_mode: vk::CullModeFlags::BACK,
            front_face: vk::FrontFace::CLOCKWISE,

            depth_bias_enable: vk::FALSE,
            depth_bias_constant_factor: 0.,
            depth_bias_clamp: 0.,
            depth_bias_slope_factor: 0.,

            ..Default::default()
        };

        let multisampling_info = vk::PipelineMultisampleStateCreateInfo {
            sample_shading_enable: vk::FALSE,
            rasterization_samples: vk::SampleCountFlags::TYPE_1,
            // multisampling.minSampleShading = 1.0f; // Optional
            // multisampling.pSampleMask = nullptr; // Optional
            // multisampling.alphaToCoverageEnable = VK_FALSE; // Optional
            // multisampling.alphaToOneEnable = VK_FALSE; // Optional
            ..Default::default()
        };

        let color_blend_attachment = vk::PipelineColorBlendAttachmentState {
            color_write_mask: vk::ColorComponentFlags::R
                | vk::ColorComponentFlags::G
                | vk::ColorComponentFlags::B
                | vk::ColorComponentFlags::A,
            blend_enable: vk::FALSE,
            // src_color_blend_factor: vk::BlendFactor::ONE,
            // dst_color_blend_factor: vk::BlendFactor::ZERO,
            // color_blend_op: vk::BlendOp::ADD,
            // src_alpha_blend_factor: vk::BlendFactor::ONE,
            // dst_alpha_blend_factor: vk::BlendFactor::ZERO,
            // alpha_blend_op: vk::BlendOp::ADD,
            ..Default::default()
        };

        let color_blending_info = vk::PipelineColorBlendStateCreateInfo {
            logic_op_enable: vk::FALSE,
            // logic_op: vk::LogicOp::COPY,
            attachment_count: 1,
            p_attachments: &color_blend_attachment,
            // blend_constants: [0.; 4],
            ..Default::default()
        };

        let pipeline_layout_info = vk::PipelineLayoutCreateInfo {
            // set_layout_count: 0,
            // p_set_layouts: std::ptr::null(),
            // push_constant_range_count: 0,
            // p_push_constant_ranges: std::ptr::null(),
            ..Default::default()
        };

        let pipeline_layout =
            unsafe { device.create_pipeline_layout(&pipeline_layout_info, None)? };

        let pipeline_info = vk::GraphicsPipelineCreateInfo {
            stage_count: 2,
            p_stages: &shader_stages as *const _,

            p_vertex_input_state: &vertex_inupt_info,
            p_input_assembly_state: &input_assemply_info,
            p_viewport_state: &vieport_state_info,
            p_rasterization_state: &rasterizer_info,
            p_multisample_state: &multisampling_info,
            // p_depth_stencil_state: std::ptr::null(),
            p_color_blend_state: &color_blending_info,
            //p_dynamic_state: std::ptr::null(),
            layout: pipeline_layout,
            render_pass,
            subpass: 0,

            base_pipeline_handle: vk::Pipeline::null(),
            base_pipeline_index: -1,

            ..Default::default()
        };

        let graphics_pipeline = unsafe {
            match device.create_graphics_pipelines(
                vk::PipelineCache::null(),
                &[pipeline_info][..],
                None,
            ) {
                Ok(pipeline) => pipeline[0],
                Err((pipelines, e)) => {
                    for pipeline in pipelines {
                        device.destroy_pipeline(pipeline, None);
                    }
                    return Err(e.into());
                }
            }
        };

        unsafe { device.destroy_shader_module(vert_shader_module, None) };
        unsafe { device.destroy_shader_module(frag_shader_module, None) };
        self.graphics_pipeline = Some(graphics_pipeline);
        self.pipeline_layout = Some(pipeline_layout);
        Ok(())
    }

    fn create_shader_module(&self, code: &[u8]) -> Result<vk::ShaderModule> {
        let device = self.device.as_ref().ok_or_else(|| anyhow!("no device"))?;
        let create_info = vk::ShaderModuleCreateInfo {
            code_size: code.len(),
            // make sure this is aligned properly
            p_code: code.as_ptr() as *const _,
            ..Default::default()
        };
        let shader_module = unsafe { device.create_shader_module(&create_info, None)? };
        Ok(shader_module)
    }

    fn create_framebuffers(&mut self) -> Result<()> {
        let device = self.device.as_ref().ok_or_else(|| anyhow!("no device"))?;
        let render_pass = self.render_pass.ok_or_else(|| anyhow!("no render_pass"))?;
        let swap_chain_extent = self
            .swap_chain_extent
            .ok_or_else(|| anyhow!("no swap_chain_extent"))?;
        self.swap_chain_framebuffers = self
            .swap_chain_image_views
            .iter()
            .map(|image_view| {
                let attachments = [*image_view];

                let framebuffer_info = vk::FramebufferCreateInfo {
                    render_pass,
                    attachment_count: attachments.len() as u32,
                    p_attachments: &attachments as *const _,
                    width: swap_chain_extent.width,
                    height: swap_chain_extent.height,
                    layers: 1,

                    ..Default::default()
                };
                let framebuffer = unsafe { device.create_framebuffer(&framebuffer_info, None)? };
                Ok(framebuffer)
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(())
    }

    fn create_command_pool(&mut self) -> Result<()> {
        let instance = self
            .instance
            .as_ref()
            .ok_or_else(|| anyhow!("no instance"))?;
        let physical_device = self
            .physical_device
            .as_ref()
            .ok_or_else(|| anyhow!("no device"))?;
        let device = self.device.as_ref().ok_or_else(|| anyhow!("no device"))?;
        let (surface, surface_khr) = self.surface.as_ref().ok_or_else(|| anyhow!("no surface"))?;

        let indices = QueueFamilyIndices::new(instance, physical_device, surface, surface_khr)?;
        let pool_info = vk::CommandPoolCreateInfo {
            queue_family_index: indices.graphics_family.unwrap(),
            ..Default::default()
        };

        let command_pool = unsafe { device.create_command_pool(&pool_info, None)? };
        self.command_pool = Some(command_pool);
        Ok(())
    }

    fn create_command_buffers(&mut self) -> Result<()> {
        let device = self.device.as_ref().ok_or_else(|| anyhow!("no device"))?;
        let command_pool = self
            .command_pool
            .ok_or_else(|| anyhow!("no command_pool"))?;
        let render_pass = self.render_pass.ok_or_else(|| anyhow!("no render_pass"))?;
        let extent = self.swap_chain_extent.ok_or_else(|| anyhow!("no extent"))?;
        let graphics_pipeline = self
            .graphics_pipeline
            .ok_or_else(|| anyhow!("no graphics_pipeline"))?;

        let alloc_info = vk::CommandBufferAllocateInfo {
            command_pool,
            level: vk::CommandBufferLevel::PRIMARY,
            command_buffer_count: self.swap_chain_framebuffers.len() as u32,
            ..Default::default()
        };

        self.command_buffers = unsafe { device.allocate_command_buffers(&alloc_info) }?;

        for i in 0..self.command_buffers.len() {
            let begin_info = vk::CommandBufferBeginInfo {
                ..Default::default()
            };

            unsafe { device.begin_command_buffer(self.command_buffers[i], &begin_info) }?;

            let clear_color = vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0., 0., 0., 1.],
                },
            };

            let render_pass_info = vk::RenderPassBeginInfo {
                render_pass,
                framebuffer: self.swap_chain_framebuffers[i],

                render_area: Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent,
                },
                clear_value_count: 1,
                p_clear_values: &clear_color,
                ..Default::default()
            };

            unsafe {
                device.cmd_begin_render_pass(
                    self.command_buffers[i],
                    &render_pass_info,
                    vk::SubpassContents::INLINE,
                );

                device.cmd_bind_pipeline(
                    self.command_buffers[i],
                    vk::PipelineBindPoint::GRAPHICS,
                    graphics_pipeline,
                );
                device.cmd_draw(self.command_buffers[i], 3, 1, 0, 0);
                device.cmd_end_render_pass(self.command_buffers[i]);

                device.end_command_buffer(self.command_buffers[i])?;
            }
        }
        Ok(())
    }

    fn create_semaphores(&mut self) -> Result<()> {
        let device = self.device.as_ref().ok_or_else(|| anyhow!("no device"))?;
        let semaphore_info = vk::SemaphoreCreateInfo {
            ..Default::default()
        };

        unsafe {
            self.image_available_semaphore = Some(device.create_semaphore(&semaphore_info, None)?);
            self.render_finished_semaphore = Some(device.create_semaphore(&semaphore_info, None)?);
        }

        Ok(())
    }

    fn main_loop(&mut self) -> Result<()> {
        let mut glfw = self
            .glfw
            .as_ref()
            .ok_or_else(|| anyhow!("no glfw"))?
            .clone();
        let window = self.window.as_ref().ok_or_else(|| anyhow!("no window"))?;
        while !window.should_close() {
            glfw.poll_events();
            self.draw_frame()?;
        }

        unsafe { self.device.as_ref().unwrap().device_wait_idle()? }
        Ok(())
    }

    fn draw_frame(&self) -> Result<()> {
        let device = self.device.as_ref().ok_or_else(|| anyhow!("no device"))?;
        let (swap_chain, swap_chain_khr) = self
            .swap_chain
            .as_ref()
            .ok_or_else(|| anyhow!("no swap chain"))?;
        let image_available_semaphore = self
            .image_available_semaphore
            .ok_or_else(|| anyhow!("no image available semaphore"))?;
        let render_finished_semaphore = self
            .render_finished_semaphore
            .ok_or_else(|| anyhow!("no render finished semaphore"))?;
        let graphics_queue = self
            .graphics_queue
            .ok_or_else(|| anyhow!("no graphics queue"))?;
        let present_queue = self
            .present_queue
            .ok_or_else(|| anyhow!("no present queue"))?;

        let image_index = unsafe {
            swap_chain
                .acquire_next_image(
                    *swap_chain_khr,
                    u64::MAX,
                    image_available_semaphore,
                    vk::Fence::null(),
                )?
                .0
        };

        let wait_semaphores = [image_available_semaphore];
        let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let signal_semaphores = [render_finished_semaphore];
        let submit_info = [vk::SubmitInfo {
            wait_semaphore_count: 1,
            p_wait_semaphores: wait_semaphores.as_ptr(),
            p_wait_dst_stage_mask: wait_stages.as_ptr(),
            command_buffer_count: 1,
            p_command_buffers: &self.command_buffers[image_index as usize],
            signal_semaphore_count: 1,
            p_signal_semaphores: signal_semaphores.as_ptr(),

            ..Default::default()
        }];
        unsafe {
            device.queue_submit(graphics_queue, &submit_info, vk::Fence::null())?;
        }

        let swap_chains = [*swap_chain_khr];
        let present_info = vk::PresentInfoKHR {
            wait_semaphore_count: 1,
            p_wait_semaphores: signal_semaphores.as_ptr(),

            swapchain_count: 1,
            p_swapchains: swap_chains.as_ptr(),
            p_image_indices: &image_index,
            ..Default::default()
        };

        unsafe {
            swap_chain.queue_present(present_queue, &present_info)?;
        }

        Ok(())
    }
}

impl Drop for HelloTriangleApplication {
    fn drop(&mut self) {
        unsafe {
            if let Some(dev) = self.device.take() {
                if let Some(sem) = self.render_finished_semaphore.take() {
                    dev.destroy_semaphore(sem, None)
                }
                if let Some(sem) = self.image_available_semaphore.take() {
                    dev.destroy_semaphore(sem, None)
                }
                if let Some(command_pool) = self.command_pool.take() {
                    dev.destroy_command_pool(command_pool, None)
                }
                for framebuffer in self.swap_chain_framebuffers.drain(..) {
                    dev.destroy_framebuffer(framebuffer, None)
                }
                if let Some(pipeline) = self.graphics_pipeline.take() {
                    dev.destroy_pipeline(pipeline, None)
                }
                if let Some(layout) = self.pipeline_layout.take() {
                    dev.destroy_pipeline_layout(layout, None)
                }
                if let Some(render_pass) = self.render_pass.take() {
                    dev.destroy_render_pass(render_pass, None)
                }
                for image_view in self.swap_chain_image_views.drain(..) {
                    dev.destroy_image_view(image_view, None)
                }
                if let Some((swap_chain, swap_chain_khr)) = self.swap_chain.take() {
                    swap_chain.destroy_swapchain(swap_chain_khr, None)
                }
                dev.destroy_device(None)
            }
            if let Some((surface, surface_khr)) = self.surface.take() {
                surface.destroy_surface(surface_khr, None)
            }
            if let Some(inst) = self.instance.take() {
                inst.destroy_instance(None)
            }
        }
    }
}

#[derive(Default)]
struct QueueFamilyIndices {
    graphics_family: Option<u32>,
    present_family: Option<u32>,
}

impl QueueFamilyIndices {
    fn new(
        instance: &Instance,
        device: &vk::PhysicalDevice,
        surface: &Surface,
        surface_khr: &vk::SurfaceKHR,
    ) -> Result<QueueFamilyIndices> {
        let mut indices = QueueFamilyIndices::default();

        let queue_families =
            unsafe { instance.get_physical_device_queue_family_properties(*device) };

        for (i, family) in queue_families.iter().enumerate() {
            let i = i as u32;
            if family.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
                indices.graphics_family = Some(i);
            }
            unsafe {
                if surface.get_physical_device_surface_support(*device, i, *surface_khr)? {
                    indices.present_family = Some(i);
                }
            }
            if indices.is_complete() {
                break;
            }
        }
        Ok(indices)
    }

    fn is_complete(&self) -> bool {
        self.graphics_family.is_some() && self.present_family.is_some()
    }
}

struct SwapChainSupportDetails {
    capabilities: vk::SurfaceCapabilitiesKHR,
    formats: Vec<vk::SurfaceFormatKHR>,
    present_modes: Vec<vk::PresentModeKHR>,
}

impl SwapChainSupportDetails {
    fn new(
        device: &vk::PhysicalDevice,
        surface: &Surface,
        surface_khr: &vk::SurfaceKHR,
    ) -> Result<Self> {
        let capabilities =
            unsafe { surface.get_physical_device_surface_capabilities(*device, *surface_khr)? };
        let formats =
            unsafe { surface.get_physical_device_surface_formats(*device, *surface_khr)? };
        let present_modes =
            unsafe { surface.get_physical_device_surface_present_modes(*device, *surface_khr)? };
        Ok(Self {
            capabilities,
            formats,
            present_modes,
        })
    }
}
fn main() -> Result<()> {
    let mut app = HelloTriangleApplication::new();

    if let Err(e) = app.run() {
        return Err(e);
    }

    Ok(())
}
