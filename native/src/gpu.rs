use anyhow::{Context, Result};
use ash::{vk, Device};
use std::{fs::File, io::BufReader, mem, path::Path};

pub struct Gpu {
    pub _entry: ash::Entry,
    pub instance: ash::Instance,
    pub physical: vk::PhysicalDevice,
    pub device: Device,
    pub queue: vk::Queue,
    pub family: u32,
    /// VK_KHR_pipeline_executable_properties, where supported.
    pub stats: Option<ash::khr::pipeline_executable_properties::Device>,
}

impl Drop for Gpu {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

#[derive(Clone, Copy)]
pub enum Resource {
    Buffer(vk::Buffer),
    Image(vk::Image),
    View(vk::ImageView),
    Memory(vk::DeviceMemory),
    Mapped(vk::DeviceMemory),
    Pass(vk::RenderPass),
    Framebuffer(vk::Framebuffer),
    Pipeline(vk::Pipeline),
    Layout(vk::PipelineLayout),
    SetLayout(vk::DescriptorSetLayout),
    Pool(vk::DescriptorPool),
    Sampler(vk::Sampler),
    Commands(vk::CommandPool),
    Fence(vk::Fence),
    Queries(vk::QueryPool),
}

pub struct Resources<'a> {
    pub gpu: &'a Gpu,
    objects: Vec<Resource>,
}

impl Drop for Resources<'_> {
    fn drop(&mut self) {
        use Resource::*;
        let d = &self.gpu.device;
        unsafe {
            let _ = d.device_wait_idle();
            for &resource in self.objects.iter().rev() {
                match resource {
                    Buffer(x) => d.destroy_buffer(x, None),
                    Image(x) => d.destroy_image(x, None),
                    View(x) => d.destroy_image_view(x, None),
                    Memory(x) => d.free_memory(x, None),
                    Mapped(x) => d.unmap_memory(x),
                    Pass(x) => d.destroy_render_pass(x, None),
                    Framebuffer(x) => d.destroy_framebuffer(x, None),
                    Pipeline(x) => d.destroy_pipeline(x, None),
                    Layout(x) => d.destroy_pipeline_layout(x, None),
                    SetLayout(x) => d.destroy_descriptor_set_layout(x, None),
                    Pool(x) => d.destroy_descriptor_pool(x, None),
                    Sampler(x) => d.destroy_sampler(x, None),
                    Commands(x) => d.destroy_command_pool(x, None),
                    Fence(x) => d.destroy_fence(x, None),
                    Queries(x) => d.destroy_query_pool(x, None),
                }
            }
        }
    }
}

impl<'a> Resources<'a> {
    pub fn new(gpu: &'a Gpu) -> Self {
        Self {
            gpu,
            objects: Vec::new(),
        }
    }

    pub fn keep<T: Copy>(&mut self, object: T, resource: fn(T) -> Resource) -> T {
        self.objects.push(resource(object));
        object
    }

    fn memory(
        &mut self,
        requirements: vk::MemoryRequirements,
        flags: vk::MemoryPropertyFlags,
    ) -> Result<vk::DeviceMemory> {
        let properties = unsafe {
            self.gpu
                .instance
                .get_physical_device_memory_properties(self.gpu.physical)
        };
        let index = (0..properties.memory_type_count)
            .find(|&i| {
                requirements.memory_type_bits & (1 << i) != 0
                    && properties.memory_types[i as usize]
                        .property_flags
                        .contains(flags)
            })
            .context("No compatible Vulkan memory type")?;
        let info = vk::MemoryAllocateInfo::default()
            .allocation_size(requirements.size)
            .memory_type_index(index);
        Ok(self.keep(
            unsafe { self.gpu.device.allocate_memory(&info, None)? },
            Resource::Memory,
        ))
    }

    pub fn buffer<T: Copy>(
        &mut self,
        data: &[T],
        usage: vk::BufferUsageFlags,
    ) -> Result<(vk::Buffer, *mut T)> {
        let d = &self.gpu.device;
        let size = mem::size_of_val(data) as u64;
        let info = vk::BufferCreateInfo::default().size(size).usage(usage);
        let buffer = unsafe { d.create_buffer(&info, None)? };
        let memory = self.memory(
            unsafe { d.get_buffer_memory_requirements(buffer) },
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        );
        self.keep(buffer, Resource::Buffer);
        let memory = memory?;
        unsafe {
            d.bind_buffer_memory(buffer, memory, 0)?;
            let mapped = d
                .map_memory(memory, 0, size, vk::MemoryMapFlags::empty())?
                .cast::<T>();
            self.keep(memory, Resource::Mapped);
            mapped.copy_from_nonoverlapping(data.as_ptr(), data.len());
            Ok((buffer, mapped))
        }
    }

    pub fn view(
        &mut self,
        image: vk::Image,
        format: vk::Format,
        aspect: vk::ImageAspectFlags,
    ) -> Result<vk::ImageView> {
        self.view_levels(image, format, aspect, 1)
    }

    /// A view of mip levels 0..levels.
    pub fn view_levels(
        &mut self,
        image: vk::Image,
        format: vk::Format,
        aspect: vk::ImageAspectFlags,
        levels: u32,
    ) -> Result<vk::ImageView> {
        let info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format)
            .subresource_range(image_range(aspect).level_count(levels));
        Ok(self.keep(
            unsafe { self.gpu.device.create_image_view(&info, None)? },
            Resource::View,
        ))
    }

    pub fn image(
        &mut self,
        size: vk::Extent2D,
        format: vk::Format,
        usage: vk::ImageUsageFlags,
    ) -> Result<(vk::Image, vk::ImageView)> {
        self.image_levels(size, format, usage, 1)
    }

    /// An image with `levels` mip levels; the view shows level 0 only.
    pub fn image_levels(
        &mut self,
        size: vk::Extent2D,
        format: vk::Format,
        usage: vk::ImageUsageFlags,
        levels: u32,
    ) -> Result<(vk::Image, vk::ImageView)> {
        let d = &self.gpu.device;
        let info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(vk::Extent3D {
                width: size.width,
                height: size.height,
                depth: 1,
            })
            .mip_levels(levels)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .usage(usage);
        let image = unsafe { d.create_image(&info, None)? };
        let memory = self.memory(
            unsafe { d.get_image_memory_requirements(image) },
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        );
        self.keep(image, Resource::Image);
        unsafe { d.bind_image_memory(image, memory?, 0)? };
        let aspect = if usage.contains(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT) {
            vk::ImageAspectFlags::DEPTH
        } else {
            vk::ImageAspectFlags::COLOR
        };
        Ok((image, self.view(image, format, aspect)?))
    }

    pub fn framebuffer(
        &mut self,
        pass: vk::RenderPass,
        size: vk::Extent2D,
        views: &[vk::ImageView],
    ) -> Result<vk::Framebuffer> {
        let info = vk::FramebufferCreateInfo::default()
            .render_pass(pass)
            .attachments(views)
            .width(size.width)
            .height(size.height)
            .layers(1);
        Ok(self.keep(
            unsafe { self.gpu.device.create_framebuffer(&info, None)? },
            Resource::Framebuffer,
        ))
    }
}

pub fn image_range(aspect: vk::ImageAspectFlags) -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange::default()
        .aspect_mask(aspect)
        .level_count(1)
        .layer_count(1)
}

pub struct Shader<'a> {
    pub handle: vk::ShaderModule,
    device: &'a Device,
}

impl<'a> Shader<'a> {
    pub fn load(device: &'a Device, dir: &Path, name: &str) -> Result<Self> {
        let path = dir.join(name);
        let mut file = BufReader::new(
            File::open(&path).with_context(|| format!("Opening {}", path.display()))?,
        );
        let code = ash::util::read_spv(&mut file)?;
        let info = vk::ShaderModuleCreateInfo::default().code(&code);
        Ok(Self {
            handle: unsafe { device.create_shader_module(&info, None)? },
            device,
        })
    }

    pub fn stage(&self, stage: vk::ShaderStageFlags) -> vk::PipelineShaderStageCreateInfo<'_> {
        vk::PipelineShaderStageCreateInfo::default()
            .module(self.handle)
            .stage(stage)
            .name(c"main")
    }
}

impl Drop for Shader<'_> {
    fn drop(&mut self) {
        unsafe { self.device.destroy_shader_module(self.handle, None) };
    }
}
