//! Per-eye neural-world state: reprojection splat, resolve + CA, display.
//!
//! Frame order per eye: scene (skyscraper G-buffer) -> splat -> resolve + CA
//! (composites the scene before the CA step) -> display.
//!
//! State per eye and parity (ping-pong), all RGBA16F:
//!   A = (r, g, b, inverse depth)
//!   B = (world velocity xyz, certainty)
//!   C = (h0, h1, h2, h3), D = (h4, h5, h6, spare): hidden state; with one hidden
//!   image (3 channels) only C = (h0, h1, h2, 0) exists
//! Splat accumulation per eye, RGBA16F with additive blending (at most 4 targets,
//! the portable Vulkan minimum):
//!   acc0 = (rgb*w, w), acc1 = (invDepth, velocity)*w,
//!   acc2 = (certainty, h0, h1, h2)*w, acc3 = (h3, h4, h5, h6)*w (2 hidden images)
//! Scene G-buffer per eye: (sun shade, inverse distance, vertical velocity, id).

use crate::gpu::{image_range, Resource, Resources, Shader};
use crate::rule::{RuleFrame, READOUT_VEC4S, RULE_VEC4S};
use anyhow::Result;
use ash::{vk, Device};
use glam::{Mat4, Vec3};
use std::{mem::size_of, path::Path};

pub const EYES: usize = 2;
const STATE: vk::Format = vk::Format::R16G16B16A16_SFLOAT;
const DEPTH: vk::Format = vk::Format::D32_SFLOAT;
/// Maximum state images per eye (A, B, C, D); accumulators match the state images.
const STATES: usize = 4;
/// prev state (2 eyes x 4), accumulators (2 x 4), new state (2 x 4), scene (2)
const SAMPLED_BINDINGS: u32 = (EYES * (3 * STATES + 1)) as u32;

/// Live settings (av-controls faders).
#[derive(Clone, Copy)]
pub struct Settings {
    /// Active perception rings (1..4) and the fraction of taps per ring.
    pub rings: usize,
    pub taps: f32,
    /// Scene alpha for colour, for the building identity (hidden), for depth + velocity.
    pub color: f32,
    pub identity: f32,
    pub depth: f32,
    /// Multipliers for the hidden-state noise and the seed frequency.
    pub noise: f32,
    pub seeds: f32,
    /// Blocks per side, cell size (m), maximum height (m), gaussian radius (m).
    pub blocks: u32,
    pub cell: f32,
    pub height: f32,
    /// Depth range (m) of the CA's own geometry; it is pushed out of `near`.
    pub near: f32,
    pub far: f32,
    pub radius: f32,
}

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
    lookup: [f32; 4],
    city: [f32; 4],
    mixing: [f32; 4],
    player: [f32; 4],
    tone: [[f32; 4]; 3],
}
const _: () = assert!(size_of::<Uniforms>() == 480 + (2 * RULE_VEC4S + READOUT_VEC4S + 8) * 16);

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
    /// Hidden images (1 or 2); state images per eye = 2 + hidden_images.
    hidden_images: usize,
    scene_pass: vk::RenderPass,
    splat_pass: vk::RenderPass,
    ca_pass: vk::RenderPass,
    display_pass: vk::RenderPass,
    scene: vk::Pipeline,
    splat: vk::Pipeline,
    ca: vk::Pipeline,
    display: vk::Pipeline,
    layout: vk::PipelineLayout,
    sets: [vk::DescriptorSet; 2],
    state_images: Vec<vk::Image>,
    scene_targets: [vk::Framebuffer; EYES],
    acc_targets: [vk::Framebuffer; EYES],
    /// ca_targets[eye][parity] writes state[eye][1 - parity].
    ca_targets: [[vk::Framebuffer; 2]; EYES],
    outputs: [Vec<vk::Framebuffer>; EYES],
    mapped: *mut Uniforms,
    /// Host-visible copy of the left eye's state A (RGBA16F), filled on request.
    readback: (vk::Buffer, *mut [u16; 4]),
    /// Record a copy of the new state A into `readback` in the next `draw`.
    pub capture: bool,
    /// Display exposure per channel (r, g, b): black point, white point, gamma.
    tone: [[f32; 3]; 3],
    previous: Option<([EyeCamera; EYES], f32)>,
    parity: usize,
    frame: u32,
    pub mode: i32,
}

/// Colour attachments plus an optional cleared, discarded depth attachment.
fn render_pass(
    r: &mut Resources<'_>,
    formats: &[vk::Format],
    depth: bool,
    load: vk::AttachmentLoadOp,
    final_layout: vk::ImageLayout,
    external: bool,
) -> Result<vk::RenderPass> {
    use vk::{AccessFlags as A, ImageLayout as L, PipelineStageFlags as S};
    let mut attachments: Vec<_> = formats
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
    if depth {
        attachments.push(
            vk::AttachmentDescription::default()
                .format(DEPTH)
                .samples(vk::SampleCountFlags::TYPE_1)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::DONT_CARE)
                .final_layout(L::DEPTH_STENCIL_ATTACHMENT_OPTIMAL),
        );
    }
    let colors: Vec<_> = (0..formats.len() as u32)
        .map(|attachment| vk::AttachmentReference {
            attachment,
            layout: L::COLOR_ATTACHMENT_OPTIMAL,
        })
        .collect();
    let depth_reference = vk::AttachmentReference {
        attachment: formats.len() as u32,
        layout: L::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
    };
    let mut subpass = vk::SubpassDescription::default()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(&colors);
    if depth {
        subpass = subpass.depth_stencil_attachment(&depth_reference);
    }
    let subpasses = [subpass];
    let shaders = S::VERTEX_SHADER | S::FRAGMENT_SHADER;
    let tests = S::EARLY_FRAGMENT_TESTS | S::LATE_FRAGMENT_TESTS;
    let dependencies = [
        vk::SubpassDependency::default()
            .src_subpass(vk::SUBPASS_EXTERNAL)
            .dst_subpass(0)
            .src_stage_mask(shaders | S::COLOR_ATTACHMENT_OUTPUT | tests)
            .dst_stage_mask(S::COLOR_ATTACHMENT_OUTPUT | tests)
            .src_access_mask(A::SHADER_READ | A::COLOR_ATTACHMENT_WRITE | A::DEPTH_STENCIL_ATTACHMENT_WRITE)
            .dst_access_mask(
                A::COLOR_ATTACHMENT_WRITE | A::COLOR_ATTACHMENT_READ | A::DEPTH_STENCIL_ATTACHMENT_WRITE,
            ),
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
    depth_test: bool,
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
    let depth = vk::PipelineDepthStencilStateCreateInfo::default()
        .depth_test_enable(depth_test)
        .depth_write_enable(depth_test)
        .depth_compare_op(vk::CompareOp::LESS);
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
        hidden_images: usize,
    ) -> Result<Self> {
        let images = 2 + hidden_images;
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
        let clear = vk::AttachmentLoadOp::CLEAR;
        let scene_pass = render_pass(r, &[STATE], true, clear, read_layout, true)?;
        let splat_pass = render_pass(r, &vec![STATE; images], false, clear, read_layout, true)?;
        let ca_pass = render_pass(r, &vec![STATE; images], false, vk::AttachmentLoadOp::DONT_CARE, read_layout, true)?;
        let display_pass = render_pass(
            r,
            &[output_format],
            false,
            vk::AttachmentLoadOp::DONT_CARE,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            false,
        )?;
        let mut pipelines = Self::pipelines(&r.gpu.device, shaders, [scene_pass, splat_pass, ca_pass, display_pass], layout, images)?
            .into_iter()
            .map(|p| r.keep(p, Resource::Pipeline));
        let [scene, splat, ca, display] = std::array::from_fn(|_| pipelines.next().unwrap());

        // state[eye][parity] = (A, B, C, D); absent images (1 hidden image) are
        // bound as their predecessor in the descriptor set and never read.
        let state_usage = Usage::COLOR_ATTACHMENT | Usage::SAMPLED | Usage::TRANSFER_DST | Usage::TRANSFER_SRC;
        let mut state_images = Vec::new();
        let mut state = [[[vk::ImageView::null(); STATES]; 2]; EYES];
        for eye in &mut state {
            for parity in eye.iter_mut() {
                for i in 0..STATES {
                    parity[i] = if i < images {
                        let (image, view) = r.image(size, STATE, state_usage)?;
                        state_images.push(image);
                        view
                    } else {
                        parity[i - 1]
                    };
                }
            }
        }
        let mut acc = [[vk::ImageView::null(); STATES]; EYES];
        for eye in &mut acc {
            for i in 0..STATES {
                eye[i] = if i < images {
                    r.image(size, STATE, Usage::COLOR_ATTACHMENT | Usage::SAMPLED)?.1
                } else {
                    eye[i - 1]
                };
            }
        }
        let mut gbuffer = [vk::ImageView::null(); EYES];
        let mut scene_targets = [vk::Framebuffer::null(); EYES];
        let mut acc_targets = [vk::Framebuffer::null(); EYES];
        let mut ca_targets = [[vk::Framebuffer::null(); 2]; EYES];
        for eye in 0..EYES {
            gbuffer[eye] = r.image(size, STATE, Usage::COLOR_ATTACHMENT | Usage::SAMPLED)?.1;
            let depth = r.image(size, DEPTH, Usage::DEPTH_STENCIL_ATTACHMENT)?.1;
            scene_targets[eye] = r.framebuffer(scene_pass, size, &[gbuffer[eye], depth])?;
            acc_targets[eye] = r.framebuffer(splat_pass, size, &acc[eye][..images])?;
            for parity in 0..2 {
                ca_targets[eye][parity] = r.framebuffer(ca_pass, size, &state[eye][1 - parity][..images])?;
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
        let pixels = vec![[0u16; 4]; (size.width * size.height) as usize];
        let readback = r.buffer(&pixels, vk::BufferUsageFlags::TRANSFER_DST)?;
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
            // Bindings 1.. : prev state of both eyes, accumulators of both eyes, new state
            // of both eyes, scene G-buffer of both eyes.
            let views: Vec<_> = (0..EYES)
                .flat_map(|e| state[e][parity])
                .chain((0..EYES).flat_map(|e| acc[e]))
                .chain((0..EYES).flat_map(|e| state[e][next]))
                .chain(gbuffer)
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
            hidden_images,
            scene_pass,
            scene,
            scene_targets,
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
            readback,
            capture: false,
            tone: [[0., 1., 1.]; 3],
            previous: None,
            parity: 0,
            frame: 0,
            mode: 0,
        })
    }

    pub fn size(&self) -> vk::Extent2D {
        self.size
    }

    /// The four pipelines (scene, splat, CA, display) from the SPIR-V in `shaders`.
    fn pipelines(
        d: &Device,
        shaders: &Path,
        [scene, splat, ca, display]: [vk::RenderPass; 4],
        layout: vk::PipelineLayout,
        images: usize,
    ) -> Result<Vec<vk::Pipeline>> {
        Ok(vec![
            pipeline(d, shaders, ("scene.vert.spv", "scene.frag.spv"), scene, layout, 1, false, true)?,
            pipeline(d, shaders, ("splat.vert.spv", "splat.frag.spv"), splat, layout, images, true, false)?,
            pipeline(d, shaders, ("fullscreen.vert.spv", "ca.frag.spv"), ca, layout, images, false, false)?,
            pipeline(d, shaders, ("fullscreen.vert.spv", "display.frag.spv"), display, layout, 1, false, false)?,
        ])
    }

    /// Recreate the pipelines from the (re)compiled SPIR-V in `shaders`.
    /// The caller must make sure the GPU is idle. On error the old pipelines stay
    /// (a partial success leaks the new ones until exit, like replaced ones).
    /// Old pipelines are only destroyed at exit (they stay in `Resources`).
    pub fn reload(&mut self, r: &mut Resources<'_>, shaders: &Path) -> Result<()> {
        let passes = [self.scene_pass, self.splat_pass, self.ca_pass, self.display_pass];
        let new = Self::pipelines(&r.gpu.device, shaders, passes, self.layout, 2 + self.hidden_images)?;
        let mut new = new.into_iter().map(|p| r.keep(p, Resource::Pipeline));
        [self.scene, self.splat, self.ca, self.display] = std::array::from_fn(|_| new.next().unwrap());
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
        settings: Settings,
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
            step: [dt, settings.near, settings.far, 0.],
            rule: rule.rule,
            rule_b: rule.rule_b,
            readout: rule.readout,
            nca: rule.params,
            lookup: [settings.rings as f32, settings.taps, self.hidden_images as f32, settings.seeds],
            city: [settings.cell, settings.height, settings.radius, settings.blocks as f32],
            mixing: [settings.color, settings.identity, settings.depth, settings.noise],
            player: {
                let ground = |c: [EyeCamera; EYES]| (c[0].position + c[1].position) * 0.5;
                let (now, before) = (ground(cameras), ground(previous));
                [now.x, now.z, before.x, before.z]
            },
            tone: self.tone_uniform(),
        };
        let area = vk::Rect2D::default().extent(self.size);
        let viewport = vk::Viewport::default()
            .width(self.size.width as f32)
            .height(self.size.height as f32)
            .max_depth(1.);
        let zero = [vk::ClearValue {
            color: vk::ClearColorValue { float32: [0.; 4] },
        }; STATES];
        let scene_clear = [
            zero[0],
            vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue { depth: 1., stencil: 0 },
            },
        ];
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
                let clears: &[vk::ClearValue] = if pass == self.scene_pass { &scene_clear } else { &zero };
                let info = vk::RenderPassBeginInfo::default()
                    .render_pass(pass)
                    .framebuffer(framebuffer)
                    .render_area(area)
                    .clear_values(clears);
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
            // Scene first: rendered before the reprojection, composited by the CA.
            for eye in 0..EYES {
                begin(self.scene_pass, self.scene_targets[eye]);
                d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.scene);
                push(eye, eye, 0);
                d.cmd_draw(cmd, 30, settings.blocks * settings.blocks, 0, 0);
                d.cmd_end_render_pass(cmd);
            }
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
        if self.capture {
            // The new state A of the left eye: state[0][1 - parity], first image.
            let image = self.state_images[(1 - self.parity) * (2 + self.hidden_images)];
            copy_to_buffer(d, cmd, image, self.size, self.readback.0);
        }
        self.parity = 1 - self.parity;
        self.frame += 1;
    }

    fn captured(&self) -> &[[u16; 4]] {
        let (w, h) = (self.size.width as usize, self.size.height as usize);
        unsafe { std::slice::from_raw_parts(self.readback.1, w * h) }
    }

    /// Display mapping per channel `pow(max(c * scale + offset, 0), gamma)`.
    fn tone_uniform(&self) -> [[f32; 4]; 3] {
        let mut out = [[0f32; 4]; 3];
        for (c, &[black, white, gamma]) in self.tone.iter().enumerate() {
            let scale = 1. / (white - black).max(1e-3);
            out[0][c] = scale;
            out[1][c] = -black * scale;
            out[2][c] = gamma;
        }
        out
    }

    /// Exposure per channel as (black point, white point, gamma), for logging.
    pub fn tone(&self) -> [[f32; 3]; 3] {
        self.tone
    }

    /// Automatic exposure from the last captured frame (call after its commands
    /// finished), smoothed over successive calls. Each channel is mapped on its own,
    /// so no channel dominates (fewer pure primaries):
    ///   its darkest `black` fraction becomes 0, its brightest `clip` fraction clips,
    ///   and a gamma puts its 80th percentile at `mid` (dark, with highlights).
    /// `tint` then pulls the three mappings toward their mean (0 = independent,
    /// 1 = shared), so a scene's colour cast partly survives.
    pub fn calibrate(&mut self, mid: f32, clip: f32, black: f32, tint: f32) {
        let pixels: Vec<[f32; 3]> = self
            .captured()
            .iter()
            .step_by(7)
            .map(|p| [0, 1, 2].map(|c| half_to_f32(p[c])))
            .filter(|p| p.iter().all(|v| v.is_finite()))
            .collect();
        if pixels.len() < 16 {
            return;
        }
        let target: [[f32; 3]; 3] = [0, 1, 2].map(|c| {
            let mut value: Vec<f32> = pixels.iter().map(|p| p[c]).collect();
            let mut percentile = |q: f32| {
                let k = ((value.len() - 1) as f32 * q.clamp(0., 1.)) as usize;
                *value.select_nth_unstable_by(k, f32::total_cmp).1
            };
            let low = percentile(black);
            let p80 = percentile(0.8);
            let high = percentile(1. - clip).max(low + 1e-2);
            let at80 = ((p80 - low) / (high - low)).clamp(1e-3, 0.999);
            [low, high, (mid.clamp(0.01, 0.99).ln() / at80.ln()).clamp(0.25, 10.)]
        });
        for (c, tone) in self.tone.iter_mut().enumerate() {
            for i in 0..3 {
                let mean = (target[0][i] + target[1][i] + target[2][i]) / 3.;
                let goal = target[c][i] + (mean - target[c][i]) * tint;
                tone[i] += (goal - tone[i]) * 0.15;
            }
        }
    }

    /// The left eye's colour from the last captured frame, exposed like the display,
    /// as a binary PPM.
    pub fn snapshot_ppm(&self) -> Vec<u8> {
        let (w, h) = (self.size.width, self.size.height);
        let mut out = format!("P6\n{w} {h}\n255\n").into_bytes();
        let [scale, offset, gamma] = self.tone_uniform();
        for pixel in self.captured() {
            for c in 0..3 {
                let value = (half_to_f32(pixel[c]) * scale[c] + offset[c]).max(0.).powf(gamma[c]);
                out.push((value.clamp(0., 1.) * 255. + 0.5) as u8);
            }
        }
        out
    }
}

fn half_to_f32(h: u16) -> f32 {
    let sign = if h & 0x8000 != 0 { -1. } else { 1. };
    let exponent = ((h >> 10) & 0x1f) as i32;
    let mantissa = (h & 0x3ff) as f32;
    sign * match exponent {
        0 => mantissa * 2f32.powi(-24),
        31 => f32::NAN,
        _ => (1. + mantissa / 1024.) * 2f32.powi(exponent - 15),
    }
}

/// Copy a shader-readable colour image into a buffer, leaving it shader-readable.
fn copy_to_buffer(d: &Device, cmd: vk::CommandBuffer, image: vk::Image, size: vk::Extent2D, buffer: vk::Buffer) {
    use vk::{AccessFlags as A, ImageLayout as L, PipelineStageFlags as S};
    let barrier = |old, new, src, dst| {
        vk::ImageMemoryBarrier::default()
            .image(image)
            .subresource_range(image_range(vk::ImageAspectFlags::COLOR))
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .old_layout(old)
            .new_layout(new)
            .src_access_mask(src)
            .dst_access_mask(dst)
    };
    let region = vk::BufferImageCopy::default()
        .image_subresource(vk::ImageSubresourceLayers {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            mip_level: 0,
            base_array_layer: 0,
            layer_count: 1,
        })
        .image_extent(vk::Extent3D { width: size.width, height: size.height, depth: 1 });
    let none = vk::DependencyFlags::empty();
    unsafe {
        let to_copy = [barrier(L::SHADER_READ_ONLY_OPTIMAL, L::TRANSFER_SRC_OPTIMAL, A::COLOR_ATTACHMENT_WRITE | A::SHADER_READ, A::TRANSFER_READ)];
        d.cmd_pipeline_barrier(cmd, S::COLOR_ATTACHMENT_OUTPUT | S::FRAGMENT_SHADER, S::TRANSFER, none, &[], &[], &to_copy);
        d.cmd_copy_image_to_buffer(cmd, image, L::TRANSFER_SRC_OPTIMAL, buffer, &[region]);
        let back = [barrier(L::TRANSFER_SRC_OPTIMAL, L::SHADER_READ_ONLY_OPTIMAL, A::TRANSFER_READ, A::SHADER_READ)];
        d.cmd_pipeline_barrier(cmd, S::TRANSFER, S::VERTEX_SHADER | S::FRAGMENT_SHADER, none, &[], &[], &back);
        let host = [vk::BufferMemoryBarrier::default()
            .buffer(buffer)
            .size(vk::WHOLE_SIZE)
            .src_access_mask(A::TRANSFER_WRITE)
            .dst_access_mask(A::HOST_READ)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)];
        d.cmd_pipeline_barrier(cmd, S::TRANSFER, S::HOST, none, &[], &host, &[]);
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
