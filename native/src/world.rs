//! Per-eye neural-world state: reprojection splat, resolve + CA, display.
//!
//! State per eye and parity (ping-pong), both RGBA16F:
//!   A = (r, g, b, inverse depth)
//!   B = (world velocity xyz, certainty)
//!   C = (h0, h1, h2, h3), D = (h4, h5, h6, spare): hidden state
//! Splat accumulation per eye, RGBA16F with additive blending (4 targets, the
//! portable Vulkan minimum):
//!   acc0 = (rgb*w, w), acc1 = (invDepth, velocity)*w,
//!   acc2 = (certainty, h0, h1, h2)*w, acc3 = (h3, h4, h5, h6)*w

use crate::gpu::{image_range, Resource, Resources, Shader};
use crate::rule::{RuleFrame, READOUT_VEC4S, RULE_VEC4S};
use anyhow::Result;
use ash::{vk, Device};
use glam::{Mat4, Vec3};
use std::{mem::size_of, path::Path};

pub const EYES: usize = 2;
const STATE: vk::Format = vk::Format::R16G16B16A16_SFLOAT;
/// State images per eye (A, B, C, D) and splat accumulators per eye.
const STATES: usize = 4;
const ACCUMULATORS: usize = 4;
/// prev state (2 eyes x 4), accumulators (2 x 4), new state (2 x 4)
const SAMPLED_BINDINGS: u32 = (EYES * (2 * STATES + ACCUMULATORS)) as u32;

#[repr(C, align(16))]
#[derive(Clone, Copy)]
struct Uniforms {
    view_proj: [[f32; 16]; EYES],
    prev_inv_view_proj: [[f32; 16]; EYES],
    inv_view_proj: [[f32; 16]; EYES],
    eye: [[f32; 4]; EYES],
    prev_eye: [[f32; 4]; EYES],
    info: [f32; 4],
    step: [f32; 4],
    rule: [[f32; 4]; RULE_VEC4S],
    rule_b: [[f32; 4]; RULE_VEC4S],
    readout: [[f32; 4]; READOUT_VEC4S],
    nca: [f32; 4],
}
const _: () = assert!(size_of::<Uniforms>() == 480 + (2 * RULE_VEC4S + READOUT_VEC4S + 1) * 16);

#[repr(C)]
#[derive(Clone, Copy)]
struct Push {
    target: i32,
    source: i32,
    mode: i32,
}

impl Push {
    fn bytes(self) -> [u8; 12] {
        let mut out = [0; 12];
        for (i, v) in [self.target, self.source, self.mode].iter().enumerate() {
            out[i * 4..i * 4 + 4].copy_from_slice(&v.to_ne_bytes());
        }
        out
    }
}

/// Camera of one eye for one frame.
#[derive(Clone, Copy)]
pub struct EyeCamera {
    pub view_proj: Mat4,
    pub position: Vec3,
}

pub struct World {
    size: vk::Extent2D,
    splat_pass: vk::RenderPass,
    ca_pass: vk::RenderPass,
    display_pass: vk::RenderPass,
    splat: vk::Pipeline,
    ca: vk::Pipeline,
    display: vk::Pipeline,
    layout: vk::PipelineLayout,
    sets: [vk::DescriptorSet; 2],
    state_images: Vec<vk::Image>,
    acc_targets: [vk::Framebuffer; EYES],
    /// ca_targets[eye][parity] writes state[eye][1 - parity].
    ca_targets: [[vk::Framebuffer; 2]; EYES],
    outputs: [Vec<vk::Framebuffer>; EYES],
    mapped: *mut Uniforms,
    previous: Option<([EyeCamera; EYES], f32)>,
    parity: usize,
    frame: u32,
    pub mode: i32,
}

fn render_pass(
    r: &mut Resources<'_>,
    formats: &[vk::Format],
    load: vk::AttachmentLoadOp,
    final_layout: vk::ImageLayout,
    external: bool,
) -> Result<vk::RenderPass> {
    use vk::{AccessFlags as A, ImageLayout as L, PipelineStageFlags as S};
    let attachments: Vec<_> = formats
        .iter()
        .map(|&format| {
            vk::AttachmentDescription::default()
                .format(format)
                .samples(vk::SampleCountFlags::TYPE_1)
                .load_op(load)
                .store_op(vk::AttachmentStoreOp::STORE)
                .final_layout(final_layout)
        })
        .collect();
    let colors: Vec<_> = (0..formats.len() as u32)
        .map(|attachment| vk::AttachmentReference {
            attachment,
            layout: L::COLOR_ATTACHMENT_OPTIMAL,
        })
        .collect();
    let subpasses = [vk::SubpassDescription::default()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(&colors)];
    let shaders = S::VERTEX_SHADER | S::FRAGMENT_SHADER;
    let dependencies = [
        vk::SubpassDependency::default()
            .src_subpass(vk::SUBPASS_EXTERNAL)
            .dst_subpass(0)
            .src_stage_mask(shaders | S::COLOR_ATTACHMENT_OUTPUT)
            .dst_stage_mask(S::COLOR_ATTACHMENT_OUTPUT)
            .src_access_mask(A::SHADER_READ | A::COLOR_ATTACHMENT_WRITE)
            .dst_access_mask(A::COLOR_ATTACHMENT_WRITE | A::COLOR_ATTACHMENT_READ),
        vk::SubpassDependency::default()
            .src_subpass(0)
            .dst_subpass(vk::SUBPASS_EXTERNAL)
            .src_stage_mask(S::COLOR_ATTACHMENT_OUTPUT)
            .dst_stage_mask(shaders)
            .src_access_mask(A::COLOR_ATTACHMENT_WRITE)
            .dst_access_mask(A::SHADER_READ),
    ];
    let info = vk::RenderPassCreateInfo::default()
        .attachments(&attachments)
        .subpasses(&subpasses)
        .dependencies(if external { &dependencies } else { &[] });
    Ok(r.keep(
        unsafe { r.gpu.device.create_render_pass(&info, None)? },
        Resource::Pass,
    ))
}

fn pipeline(
    d: &Device,
    dir: &Path,
    (vert, frag): (&str, &str),
    pass: vk::RenderPass,
    layout: vk::PipelineLayout,
    targets: usize,
    additive: bool,
) -> Result<vk::Pipeline> {
    let vertex = Shader::load(d, dir, vert)?;
    let fragment = Shader::load(d, dir, frag)?;
    let stages = [
        vertex.stage(vk::ShaderStageFlags::VERTEX),
        fragment.stage(vk::ShaderStageFlags::FRAGMENT),
    ];
    let input = vk::PipelineVertexInputStateCreateInfo::default();
    let assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
    let viewport = vk::PipelineViewportStateCreateInfo::default()
        .viewport_count(1)
        .scissor_count(1);
    let raster = vk::PipelineRasterizationStateCreateInfo::default()
        .line_width(1.)
        .cull_mode(vk::CullModeFlags::NONE);
    let sampling = vk::PipelineMultisampleStateCreateInfo::default()
        .rasterization_samples(vk::SampleCountFlags::TYPE_1);
    let depth = vk::PipelineDepthStencilStateCreateInfo::default();
    let blend = vk::PipelineColorBlendAttachmentState::default()
        .color_write_mask(vk::ColorComponentFlags::RGBA)
        .blend_enable(additive)
        .src_color_blend_factor(vk::BlendFactor::ONE)
        .dst_color_blend_factor(vk::BlendFactor::ONE)
        .color_blend_op(vk::BlendOp::ADD)
        .src_alpha_blend_factor(vk::BlendFactor::ONE)
        .dst_alpha_blend_factor(vk::BlendFactor::ONE)
        .alpha_blend_op(vk::BlendOp::ADD);
    let attachments = vec![blend; targets];
    let blending = vk::PipelineColorBlendStateCreateInfo::default().attachments(&attachments);
    let states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dynamic = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&states);
    let info = [vk::GraphicsPipelineCreateInfo::default()
        .stages(&stages)
        .vertex_input_state(&input)
        .input_assembly_state(&assembly)
        .viewport_state(&viewport)
        .rasterization_state(&raster)
        .multisample_state(&sampling)
        .depth_stencil_state(&depth)
        .color_blend_state(&blending)
        .dynamic_state(&dynamic)
        .layout(layout)
        .render_pass(pass)];
    let pipelines = unsafe { d.create_graphics_pipelines(vk::PipelineCache::null(), &info, None) }
        .map_err(|(_, error)| error)?;
    Ok(pipelines[0])
}

impl World {
    pub fn new(
        r: &mut Resources<'_>,
        shaders: &Path,
        size: vk::Extent2D,
        output_format: vk::Format,
        swapchains: [&[vk::Image]; EYES],
    ) -> Result<Self> {
        use vk::ImageUsageFlags as Usage;
        let d = &r.gpu.device;
        let stages = vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT;

        let bindings: Vec<_> = (0..=SAMPLED_BINDINGS)
            .map(|i| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(i)
                    .descriptor_count(1)
                    .stage_flags(stages)
                    .descriptor_type(if i == 0 {
                        vk::DescriptorType::UNIFORM_BUFFER
                    } else {
                        vk::DescriptorType::COMBINED_IMAGE_SAMPLER
                    })
            })
            .collect();
        let set_layout = r.keep(
            unsafe {
                d.create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                    None,
                )?
            },
            Resource::SetLayout,
        );
        let set_layouts = [set_layout];
        let push = [vk::PushConstantRange::default()
            .stage_flags(stages)
            .size(size_of::<Push>() as u32)];
        let layout = r.keep(
            unsafe {
                d.create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::default()
                        .set_layouts(&set_layouts)
                        .push_constant_ranges(&push),
                    None,
                )?
            },
            Resource::Layout,
        );
        let pool_sizes = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(2),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(2 * SAMPLED_BINDINGS),
        ];
        let pool = r.keep(
            unsafe {
                d.create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo::default()
                        .max_sets(2)
                        .pool_sizes(&pool_sizes),
                    None,
                )?
            },
            Resource::Pool,
        );
        let nearest = r.keep(
            unsafe {
                d.create_sampler(
                    &vk::SamplerCreateInfo::default()
                        .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                        .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                        .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE),
                    None,
                )?
            },
            Resource::Sampler,
        );

        let read_layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
        let splat_pass = render_pass(r, &[STATE; ACCUMULATORS], vk::AttachmentLoadOp::CLEAR, read_layout, true)?;
        let ca_pass = render_pass(r, &[STATE; STATES], vk::AttachmentLoadOp::DONT_CARE, read_layout, true)?;
        let display_pass = render_pass(
            r,
            &[output_format],
            vk::AttachmentLoadOp::DONT_CARE,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            false,
        )?;
        let d = &r.gpu.device;
        let splat = pipeline(d, shaders, ("splat.vert.spv", "splat.frag.spv"), splat_pass, layout, ACCUMULATORS, true)?;
        let splat = r.keep(splat, Resource::Pipeline);
        let d = &r.gpu.device;
        let ca = pipeline(d, shaders, ("fullscreen.vert.spv", "ca.frag.spv"), ca_pass, layout, STATES, false)?;
        let ca = r.keep(ca, Resource::Pipeline);
        let d = &r.gpu.device;
        let display = pipeline(
            d,
            shaders,
            ("fullscreen.vert.spv", "display.frag.spv"),
            display_pass,
            layout,
            1,
            false,
        )?;
        let display = r.keep(display, Resource::Pipeline);

        // state[eye][parity] = (A, B, C, D)
        let state_usage = Usage::COLOR_ATTACHMENT | Usage::SAMPLED | Usage::TRANSFER_DST;
        let mut state_images = Vec::new();
        let mut state = [[[vk::ImageView::null(); STATES]; 2]; EYES];
        for eye in &mut state {
            for parity in eye.iter_mut() {
                for view in parity.iter_mut() {
                    let (image, v) = r.image(size, STATE, state_usage)?;
                    state_images.push(image);
                    *view = v;
                }
            }
        }
        let mut acc = [[vk::ImageView::null(); ACCUMULATORS]; EYES];
        for eye in &mut acc {
            for view in eye.iter_mut() {
                *view = r.image(size, STATE, Usage::COLOR_ATTACHMENT | Usage::SAMPLED)?.1;
            }
        }
        let acc_targets = [
            r.framebuffer(splat_pass, size, &acc[0])?,
            r.framebuffer(splat_pass, size, &acc[1])?,
        ];
        let mut ca_targets = [[vk::Framebuffer::null(); 2]; EYES];
        for eye in 0..EYES {
            for parity in 0..2 {
                ca_targets[eye][parity] = r.framebuffer(ca_pass, size, &state[eye][1 - parity])?;
            }
        }
        let mut outputs: [Vec<vk::Framebuffer>; EYES] = Default::default();
        for eye in 0..EYES {
            for &image in swapchains[eye] {
                let view = r.view(image, output_format, vk::ImageAspectFlags::COLOR)?;
                outputs[eye].push(r.framebuffer(display_pass, size, &[view])?);
            }
        }

        // Zeroed initial contents; every field is plain f32 data.
        let initial: Uniforms = unsafe { std::mem::zeroed() };
        let (uniforms, mapped) = r.buffer(&[initial], vk::BufferUsageFlags::UNIFORM_BUFFER)?;
        let d = &r.gpu.device;
        let layouts = [set_layout; 2];
        let allocated = unsafe {
            d.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(pool)
                    .set_layouts(&layouts),
            )?
        };
        for (parity, &set) in allocated.iter().enumerate() {
            let next = 1 - parity;
            let uniform = [vk::DescriptorBufferInfo::default()
                .buffer(uniforms)
                .range(size_of::<Uniforms>() as u64)];
            // Bindings 1.. : prev state of both eyes, accumulators of both eyes, new state of both eyes.
            let views: Vec<_> = (0..EYES)
                .flat_map(|e| state[e][parity])
                .chain((0..EYES).flat_map(|e| acc[e]))
                .chain((0..EYES).flat_map(|e| state[e][next]))
                .collect();
            let views: Vec<_> = views
                .into_iter()
                .map(|view| {
                    [vk::DescriptorImageInfo::default()
                        .image_view(view)
                        .sampler(nearest)
                        .image_layout(read_layout)]
                })
                .collect();
            let mut writes = vec![vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(&uniform)];
            writes.extend(views.iter().enumerate().map(|(i, info)| {
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(i as u32 + 1)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(info)
            }));
            unsafe { d.update_descriptor_sets(&writes, &[]) };
        }

        Ok(Self {
            size,
            splat_pass,
            ca_pass,
            display_pass,
            splat,
            ca,
            display,
            layout,
            sets: [allocated[0], allocated[1]],
            state_images,
            acc_targets,
            ca_targets,
            outputs,
            mapped,
            previous: None,
            parity: 0,
            frame: 0,
            mode: 0,
        })
    }

    pub fn size(&self) -> vk::Extent2D {
        self.size
    }

    /// Recreate the three pipelines from the (re)compiled SPIR-V in `shaders`.
    /// The caller must make sure the GPU is idle. On error the old pipelines stay.
    /// Old pipelines are only destroyed at exit (they stay in `Resources`).
    pub fn reload(&mut self, r: &mut Resources<'_>, shaders: &Path) -> Result<()> {
        let d = &r.gpu.device;
        let splat = pipeline(d, shaders, ("splat.vert.spv", "splat.frag.spv"), self.splat_pass, self.layout, ACCUMULATORS, true)?;
        let ca = pipeline(d, shaders, ("fullscreen.vert.spv", "ca.frag.spv"), self.ca_pass, self.layout, STATES, false);
        let display = pipeline(d, shaders, ("fullscreen.vert.spv", "display.frag.spv"), self.display_pass, self.layout, 1, false);
        self.splat = r.keep(splat, Resource::Pipeline);
        self.ca = r.keep(ca?, Resource::Pipeline);
        self.display = r.keep(display?, Resource::Pipeline);
        Ok(())
    }

    /// Record one frame for both eyes: splat, resolve + CA, display.
    pub fn draw(
        &mut self,
        d: &Device,
        cmd: vk::CommandBuffer,
        cameras: [EyeCamera; EYES],
        images: [usize; EYES],
        time: f32,
        rule: &RuleFrame,
    ) {
        if self.previous.is_none() {
            clear_to_read(d, cmd, &self.state_images);
        }
        let (previous, last_time) = self.previous.replace((cameras, time)).unwrap_or((cameras, time));
        let dt = (time - last_time).clamp(0., 0.1);
        let uniforms = Uniforms {
            view_proj: cameras.map(|c| c.view_proj.to_cols_array()),
            prev_inv_view_proj: previous.map(|c| c.view_proj.inverse().to_cols_array()),
            inv_view_proj: cameras.map(|c| c.view_proj.inverse().to_cols_array()),
            eye: cameras.map(|c| c.position.extend(1.).to_array()),
            prev_eye: previous.map(|c| c.position.extend(1.).to_array()),
            info: [
                self.size.width as f32,
                self.size.height as f32,
                time,
                (self.frame % 65536) as f32,
            ],
            step: [dt, 0., 0., 0.],
            rule: rule.rule,
            rule_b: rule.rule_b,
            readout: rule.readout,
            nca: rule.params,
        };
        let area = vk::Rect2D::default().extent(self.size);
        let viewport = vk::Viewport::default()
            .width(self.size.width as f32)
            .height(self.size.height as f32)
            .max_depth(1.);
        let zero = [vk::ClearValue {
            color: vk::ClearColorValue { float32: [0.; 4] },
        }; STATES];
        let stages = vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT;
        let pixels = self.size.width * self.size.height;
        unsafe {
            self.mapped.write(uniforms);
            d.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.layout,
                0,
                &[self.sets[self.parity]],
                &[],
            );
            let begin = |pass, framebuffer| {
                let info = vk::RenderPassBeginInfo::default()
                    .render_pass(pass)
                    .framebuffer(framebuffer)
                    .render_area(area)
                    .clear_values(&zero);
                d.cmd_begin_render_pass(cmd, &info, vk::SubpassContents::INLINE);
                d.cmd_set_viewport(cmd, 0, &[viewport]);
                d.cmd_set_scissor(cmd, 0, &[area]);
            };
            let push = |target: usize, source: usize, mode: i32| {
                let p = Push {
                    target: target as i32,
                    source: source as i32,
                    mode,
                };
                d.cmd_push_constants(cmd, self.layout, stages, 0, &p.bytes());
            };
            for eye in 0..EYES {
                begin(self.splat_pass, self.acc_targets[eye]);
                d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.splat);
                // Own previous state, then the other eye's (cross-eye splat).
                for source in [eye, 1 - eye] {
                    push(eye, source, 0);
                    d.cmd_draw(cmd, 6, pixels, 0, 0);
                }
                d.cmd_end_render_pass(cmd);
            }
            for eye in 0..EYES {
                begin(self.ca_pass, self.ca_targets[eye][self.parity]);
                d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.ca);
                push(eye, eye, 0);
                d.cmd_draw(cmd, 3, 1, 0, 0);
                d.cmd_end_render_pass(cmd);
            }
            for eye in 0..EYES {
                begin(self.display_pass, self.outputs[eye][images[eye]]);
                d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.display);
                push(eye, eye, self.mode);
                d.cmd_draw(cmd, 3, 1, 0, 0);
                d.cmd_end_render_pass(cmd);
            }
        }
        self.parity = 1 - self.parity;
        self.frame += 1;
    }
}

/// Zero all state images (no depth = nothing to splat, certainty 0) and make them readable.
fn clear_to_read(d: &Device, cmd: vk::CommandBuffer, images: &[vk::Image]) {
    use vk::{AccessFlags as A, ImageLayout as L, PipelineStageFlags as S};
    let range = image_range(vk::ImageAspectFlags::COLOR);
    let barriers = |old, new, src, dst| -> Vec<_> {
        images
            .iter()
            .map(|&image| {
                vk::ImageMemoryBarrier::default()
                    .image(image)
                    .subresource_range(range)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .old_layout(old)
                    .new_layout(new)
                    .src_access_mask(src)
                    .dst_access_mask(dst)
            })
            .collect()
    };
    unsafe {
        let to_transfer = barriers(L::UNDEFINED, L::TRANSFER_DST_OPTIMAL, A::empty(), A::TRANSFER_WRITE);
        d.cmd_pipeline_barrier(cmd, S::TOP_OF_PIPE, S::TRANSFER, vk::DependencyFlags::empty(), &[], &[], &to_transfer);
        for &image in images {
            d.cmd_clear_color_image(
                cmd,
                image,
                L::TRANSFER_DST_OPTIMAL,
                &vk::ClearColorValue { float32: [0.; 4] },
                &[range],
            );
        }
        let to_read = barriers(
            L::TRANSFER_DST_OPTIMAL,
            L::SHADER_READ_ONLY_OPTIMAL,
            A::TRANSFER_WRITE,
            A::SHADER_READ,
        );
        d.cmd_pipeline_barrier(
            cmd,
            S::TRANSFER,
            S::VERTEX_SHADER | S::FRAGMENT_SHADER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &to_read,
        );
    }
}
