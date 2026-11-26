//! Per-eye neural-world state: reprojection splat, resolve, trained net, display.
//! The splat, resolve and net match train/sunk/model.py (the net is trained on them).
//!
//! Frame order per eye: scene (skyscraper G-buffer) -> splat -> resolve (normalise,
//! composite the scene) -> mips (hole-aware pyramid) -> net -> display.
//!
//! State per eye and parity (ping-pong), all RGBA16F:
//!   A = (r, g, b, inverse distance; 0 = no data)
//!   B = (world velocity xyz, certainty)
//!   C = (h0, h1, h2, h3)
//! Splat accumulation per eye, additive blending:
//!   acc0 = (rgb, inv)*w, acc1 = (velocity, certainty)*w, acc2 = h*w, acc3 = w (R16F)
//! Resolved state per eye (net input): A = (rgb, inv), B = (velocity, coverage), C = h,
//! holes all 0, with mip levels (2x2 averages = the hole-aware pyramid), plus D: the
//! certainty (reproject-v1, D.x) or (screen motion, weight) (reproject-v2; then A, B, C,
//! D are premultiplied by the splat weight, see shaders/resolve.frag), with mip levels.
//! Scene G-buffer per eye: (sun shade, inverse distance, vertical velocity, id).
//!
//! cube-v1 (net layout 3, shaders/cube.glsl): the state is one head-centred sphere fixed
//! to the world instead, an equi-angular cubemap in a 3 x 2 atlas of F x F faces. On the
//! frames where the net runs: scene -> cube_forward (where each texel moves) + mips ->
//! cube_pull (pull reprojection + scene blend, premultiplied like v2's resolve) + mips ->
//! net (LAYOUT 3) -> per eye cube_display (a direction lookup). On the other frames only
//! the display runs: the store is fixed to the world, so head rotation needs no update.

use crate::gpu::{image_range, Resource, Resources, Shader};
use crate::rule::{MAX_LAYERS, MAX_TAPS, RuleFrame};
use anyhow::Result;
use ash::{vk, Device};
use glam::{Mat4, Vec2, Vec3};
use std::{mem::size_of, path::Path};

pub const EYES: usize = 2;
const STATE: vk::Format = vk::Format::R16G16B16A16_SFLOAT;
/// cube-v1 splat accumulators (cube_splat.frag): float32, the nearness-weighted sums of
/// HDR colours would overflow half floats.
const ACC: vk::Format = vk::Format::R32G32B32A32_SFLOAT;
const ACC_SINGLE: vk::Format = vk::Format::R32_SFLOAT;
/// cube-v1 splat accumulators: four sums (cube.glsl cubeAccA..D) and the coverage.
const CUBE_ACCS: usize = 5;
const SINGLE: vk::Format = vk::Format::R16_SFLOAT;
const DEPTH: vk::Format = vk::Format::D32_SFLOAT;
/// Vertices per scene instance: a crystal (scene.vert: 6 sides x 6 rings x 6 + 6 x 3 tip).
const SCENE_VERTICES: u32 = 6 * 6 * 6 + 18;
/// State images per eye (A, B, C); the resolved state matches them.
const STATES: usize = 3;
/// Splat accumulators per eye (the three state images + the weight).
const ACCS: usize = 4;
/// Scene images per eye: (colour, inverse distance), (world velocity, 0).
const SCENES: usize = 2;
/// prev state, accumulators, resolved state, resolved D, new state, scene (per eye)
const SAMPLED_BINDINGS: u32 = (EYES * (STATES + ACCS + STATES + 1 + STATES + SCENES)) as u32;
const WEIGHTS_BINDING: u32 = SAMPLED_BINDINGS + 1;
/// The creatures' vertices (creatures.rs Vertex, std430), read by creatures.vert.
const CREATURES_BINDING: u32 = WEIGHTS_BINDING + 1;
/// The weights uniform buffer (state.glsl Weights, rule::pack): float16, 64 KiB, the
/// smallest guaranteed maxUniformBufferRange on mobile GPUs. Uniform (not storage) so the
/// weights are read through the constant path (AMD scalar cache, Adreno constant RAM).
pub const WEIGHT_HALVES: usize = 32768;
/// cube-v1 images (cube.glsl): prev A, B, C, next A, B, C, M, mid A..D, the splat
/// accumulators A..D, W at bindings CUBE_BINDING.. (all with the linear sampler).
const CUBE_BINDING: u32 = CREATURES_BINDING + 1;
const CUBE_IMAGES: u32 = 13 + CUBE_ACCS as u32;
const LAST_BINDING: u32 = CUBE_BINDING + CUBE_IMAGES - 1;
/// Pipelines capture the driver's statistics (VK_KHR_pipeline_executable_properties on).
pub static CAPTURE_STATS: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// How fast the display exposure follows its measurement (1/s).
const EXPOSURE_RATE: f32 = 1.5;
/// cube-v1 face frames (cube.glsl FACE_FORWARD, FACE_UP).
const CUBE_FORWARD: [Vec3; 6] = [Vec3::X, Vec3::NEG_X, Vec3::Y, Vec3::NEG_Y, Vec3::Z, Vec3::NEG_Z];
const CUBE_UP: [Vec3; 6] = [Vec3::Y, Vec3::Y, Vec3::Z, Vec3::NEG_Z, Vec3::Y, Vec3::Y];
/// cube-v1: texels per face edge of the sphere's depth probe (collisions).
const CUBE_PROBE: u32 = 24;
/// Width (px) of the depth probe read back each frame for collisions.
const PROBE: u32 = 32;
/// GPU timestamps per frame: start, after scene, splat, resolve, net, display.
pub const STAMPS: u32 = 6;
/// Split updates (World::split): the fold (or pull) and the net in this many bands of rows,
/// a stage each (small stages pack the frames' GPU budget well).
const FOLD_BANDS: u32 = 2;
const NET_BANDS: u32 = 6;

/// Live settings (av-controls faders).
#[derive(Clone, Copy)]
pub struct Settings {
    /// Trained -> random weights (0..1); net ticks this frame (0 = fill holes only).
    pub wild: f32,
    /// Certainty gate of the net's update (net.frag, netRun.y).
    pub gate: f32,
    pub tau: f32,
    /// How fast the state blends toward the scene (per second): colour, and depth with
    /// velocity and certainty.
    pub color: f32,
    pub depth: f32,
    /// Distance (m) of the sky, where the scene has no geometry.
    pub sky: f32,
    /// Extra scene blend toward the view centre ("looking uncovers detail"), 0 = off.
    pub focus: f32,
    /// The planet's day phase 0..1 (0.25 noon, 0.75 midnight); None = fixed sun.
    pub day: Option<f32>,
    /// Where the eyes look (world point), if tracked; else the focus is the view centre.
    pub gaze: Option<Vec3>,
    /// Blocks per side, cell size (m), maximum height (m).
    pub blocks: u32,
    pub cell: f32,
    pub height: f32,
    /// Fraction of cells drawn as twisted crystals (scene.vert).
    pub crystals: f32,
    /// Draw the blocks (the ground plate stays).
    pub city: bool,
    /// Particle size of the creature bodies (m).
    pub dot: f32,
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
    net: [f32; 4],
    net_run: [f32; 4],
    net_layers: [[f32; 4]; MAX_LAYERS],
    net_taps: [[f32; 4]; MAX_TAPS],
    city: [f32; 4],
    mixing: [f32; 4],
    player: [f32; 4],
    tone: [[f32; 4]; 3],
    gaze: [f32; 4],
    cube: [f32; 4],
    head: [f32; 4],
    head_move: [f32; 4],
}
const _: () = assert!(size_of::<Uniforms>() == 480 + (2 + MAX_LAYERS + MAX_TAPS + 10) * 16);

/// panorama.frag's push constants (floats): camera right, up, back (vec4 each), view
/// (attention, dome angle, kind, mode), size (width, height, square offset, side).
pub const PANORAMA_PUSH: usize = 20;

/// DESKTOP's whole-sphere view (panorama.frag) into an image of `format`.
struct Panorama {
    pass: vk::RenderPass,
    pipeline: vk::Pipeline,
}

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
    scene_pass: vk::RenderPass,
    splat_pass: vk::RenderPass,
    resolve_pass: vk::RenderPass,
    net_pass: vk::RenderPass,
    display_pass: vk::RenderPass,
    scene: vk::Pipeline,
    /// Particle bodies (swarm.rs; creatures.vert/frag, points) in the scene pass.
    creatures: vk::Pipeline,
    creature_vertices: *mut crate::creatures::Vertex,
    creature_count: u32,
    /// splat[target][source] (state.glsl TARGET, SOURCE).
    splat: [[vk::Pipeline; EYES]; EYES],
    /// The fullscreen passes per eye (state.glsl TARGET).
    resolve: [vk::Pipeline; EYES],
    net: [vk::Pipeline; EYES],
    display: [vk::Pipeline; EYES],
    layout: vk::PipelineLayout,
    sets: [vk::DescriptorSet; 2],
    state_images: Vec<vk::Image>,
    /// Resolved A, B, C per eye and D per eye, all with mip levels.
    mid_images: [[vk::Image; STATES]; EYES],
    d_images: [vk::Image; EYES],
    levels: u32,
    /// Specialization constants of the net pipeline (the net's layout).
    spec: [i32; 8],
    weights: *mut u16,
    weight_count: usize,
    /// Host-visible copy of eye 0's resolved state (A, B, C, D) and new state.
    dump: (vk::Buffer, *mut u8),
    pub dumping: bool,
    panorama: Option<Panorama>,
    scene_targets: [vk::Framebuffer; EYES],
    acc_targets: [vk::Framebuffer; EYES],
    resolve_targets: [vk::Framebuffer; EYES],
    /// net_targets[eye][parity] writes state[eye][1 - parity].
    net_targets: [[vk::Framebuffer; 2]; EYES],
    outputs: [Vec<vk::Framebuffer>; EYES],
    mapped: *mut Uniforms,
    /// Host-visible copy of the left eye's state A (RGBA16F), filled on request.
    readback: (vk::Buffer, *mut [u16; 4]),
    /// Record a copy of the new state A into `readback` in the next `draw`.
    pub capture: bool,
    /// Host-visible copy of a small mip level of both eyes' resolved A and B (every
    /// frame), with the cameras it was seen from: the image's geometry for collisions.
    probe: (vk::Buffer, *mut [u16; 4]),
    probe_level: u32,
    probe_cameras: [EyeCamera; EYES],
    /// Display exposure per channel (r, g, b): black point, white point, gamma.
    tone: [[f32; 3]; 3],
    /// The exposure the last measurement asks for; `tone` glides toward it every frame.
    tone_goal: [[f32; 3]; 3],
    previous: Option<([EyeCamera; EYES], f32)>,
    parity: usize,
    frame: u32,
    pub mode: i32,
    /// cube-v1: the forward pass (one target), pipelines, the images (resizable).
    forward_pass: vk::RenderPass,
    cube_forward: vk::Pipeline,
    cube_pull: vk::Pipeline,
    cube_net: vk::Pipeline,
    cube_display: [vk::Pipeline; EYES],
    /// Splat reprojection: its pass (CUBE_ACCS additive targets), the points and the fold.
    cube_splat_pass: vk::RenderPass,
    cube_splat: vk::Pipeline,
    cube_fold: vk::Pipeline,
    /// The net's reprojection: Some(nearness) = splat (net.json reproject), None = pull.
    pub splat_nearness: Option<f32>,
    /// Split cube updates (the desktop): `draw` records only the scene and the splat (or
    /// forward motion); `advance` records the rest, a stage per frame (fold, then the net in
    /// NET_BANDS bands), so the window keeps its frame rate while the cube updates slowly.
    pub split: bool,
    /// The split update in flight: its next stage (1.. fold / pull bands, then net bands).
    stage: Option<u32>,
    /// The net and fold passes that keep what they don't draw (a band of rows).
    cube_band_pass: vk::RenderPass,
    cube_fold_band_pass: vk::RenderPass,
    cube: Cube,
    /// Clear the cube's images before the next frame (new or resized).
    cube_clear: bool,
    /// The store's centre: the head at the last update (None: no update yet).
    cube_centre: Option<Vec3>,
    /// The pyramid sampler (the cube's descriptors use it for all its images).
    linear: vk::Sampler,
    dump_capacity: usize,
}

/// cube-v1's images at one face size (World::set_cube_face makes new ones).
struct Cube {
    face: u32,
    /// Texels around each face in the atlas (one texel of the coarsest mip level, cube.glsl).
    gutter: u32,
    atlas: vk::Extent2D,
    /// state[parity] = (A, B, C); M and mid A..D with mip levels.
    state: [[vk::Image; STATES]; 2],
    m: vk::Image,
    mid: [vk::Image; 4],
    state_views: [[vk::ImageView; STATES]; 2],
    m_view: vk::ImageView,
    mid_views: [vk::ImageView; 4],
    forward_target: vk::Framebuffer,
    pull_target: vk::Framebuffer,
    /// Splat reprojection (net.json reproject "splat"): the accumulators, cleared by the
    /// splat pass and folded into mid A..D by cube_fold.
    acc_views: [vk::ImageView; CUBE_ACCS],
    splat_target: vk::Framebuffer,
    /// Collisions: mid A and B at a coarse mip level (about CUBE_PROBE texels per face),
    /// read back after every update (World::surroundings).
    probe: (vk::Buffer, *mut [u16; 4]),
    probe_level: u32,
    /// net_targets[parity] writes state[1 - parity].
    net_targets: [vk::Framebuffer; 2],
    /// Split updates: the state before the latest (A, B), copied when an update finishes,
    /// so the display can blend into the latest while the next update overwrites it.
    old: [vk::Image; 2],
    old_views: [vk::ImageView; 2],
}

impl Cube {
    /// passes: forward (M), pull (mid A..D; the resolve pass), net (A, B, C), splat (accumulators).
    fn new(r: &mut Resources<'_>, face: u32, levels: u32, [forward, pull, net, splat]: [vk::RenderPass; 4]) -> Result<Self> {
        use vk::ImageUsageFlags as Usage;
        anyhow::ensure!(face % (1 << (levels - 1)) == 0, "the cube face size must be divisible by 2^(levels - 1)");
        let gutter = 1 << (levels - 1);
        let cell = face + 2 * gutter;
        let atlas = vk::Extent2D { width: 3 * cell, height: 2 * cell };
        let state_usage = Usage::COLOR_ATTACHMENT | Usage::SAMPLED | Usage::TRANSFER_DST | Usage::TRANSFER_SRC;
        let mid_usage = Usage::COLOR_ATTACHMENT | Usage::SAMPLED | Usage::TRANSFER_SRC | Usage::TRANSFER_DST;
        let mut state = [[vk::Image::null(); STATES]; 2];
        let mut state_views = [[vk::ImageView::null(); STATES]; 2];
        for parity in 0..2 {
            for i in 0..STATES {
                (state[parity][i], state_views[parity][i]) = r.image(atlas, STATE, state_usage)?;
            }
        }
        let (m, m_target) = r.image_levels(atlas, STATE, mid_usage, levels)?;
        let m_view = r.view_levels(m, STATE, vk::ImageAspectFlags::COLOR, levels)?;
        let mut mid = [vk::Image::null(); 4];
        let mut mid_targets = [vk::ImageView::null(); 4];
        let mut mid_views = [vk::ImageView::null(); 4];
        for i in 0..4 {
            (mid[i], mid_targets[i]) = r.image_levels(atlas, STATE, mid_usage, levels)?;
            mid_views[i] = r.view_levels(mid[i], STATE, vk::ImageAspectFlags::COLOR, levels)?;
        }
        let acc_usage = Usage::COLOR_ATTACHMENT | Usage::SAMPLED;
        let (mut old, mut old_views) = ([vk::Image::null(); 2], [vk::ImageView::null(); 2]);
        for i in 0..2 {
            (old[i], old_views[i]) = r.image(atlas, STATE, Usage::SAMPLED | Usage::TRANSFER_DST)?;
        }
        let probe_level = (face / CUBE_PROBE).max(1).ilog2().min(levels - 1);
        let probe_texels = mip_extent(atlas, probe_level);
        let probe = r.buffer(&vec![[0u16; 4]; 2 * (probe_texels.width * probe_texels.height) as usize], vk::BufferUsageFlags::TRANSFER_DST)?;
        let mut acc_views = [vk::ImageView::null(); CUBE_ACCS];
        for (i, view) in acc_views.iter_mut().enumerate() {
            *view = r.image(atlas, if i < 4 { ACC } else { ACC_SINGLE }, acc_usage)?.1;
        }
        Ok(Self {
            face,
            gutter,
            atlas,
            state,
            m,
            mid,
            state_views,
            m_view,
            mid_views,
            forward_target: r.framebuffer(forward, atlas, &[m_target])?,
            pull_target: r.framebuffer(pull, atlas, &mid_targets)?,
            splat_target: r.framebuffer(splat, atlas, &acc_views)?,
            probe,
            probe_level,
            acc_views,
            net_targets: [r.framebuffer(net, atlas, &state_views[1])?, r.framebuffer(net, atlas, &state_views[0])?],
            old,
            old_views,
        })
    }

    /// Point both descriptor sets' cube bindings at these images: prev, next (by parity), M, mid A..D.
    fn bind(&self, d: &Device, sets: [vk::DescriptorSet; 2], sampler: vk::Sampler) {
        for (parity, &set) in sets.iter().enumerate() {
            let infos: Vec<_> = self.state_views[parity]
                .into_iter()
                .chain(self.state_views[1 - parity])
                .chain([self.m_view])
                .chain(self.mid_views)
                .chain(self.acc_views)
                .chain(self.old_views)
                .map(|view| {
                    [vk::DescriptorImageInfo::default()
                        .image_view(view)
                        .sampler(sampler)
                        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)]
                })
                .collect();
            assert_eq!(infos.len(), CUBE_IMAGES as usize);
            let writes: Vec<_> = infos
                .iter()
                .enumerate()
                .map(|(i, info)| {
                    vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(CUBE_BINDING + i as u32)
                        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .image_info(info)
                })
                .collect();
            unsafe { d.update_descriptor_sets(&writes, &[]) };
        }
    }

    /// Every image with its mip level count.
    fn images(&self, levels: u32) -> Vec<(vk::Image, u32)> {
        self.state.as_flattened().iter().chain(&self.old).map(|&i| (i, 1)).chain([self.m].into_iter().chain(self.mid).map(|i| (i, levels))).collect()
    }
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
                .initial_layout(if load == vk::AttachmentLoadOp::LOAD { final_layout } else { vk::ImageLayout::UNDEFINED })
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
    points: bool,
    spec: &[i32],
) -> Result<vk::Pipeline> {
    let vertex = Shader::load(d, dir, vert)?;
    let fragment = Shader::load(d, dir, frag)?;
    // Specialization constants 0.. (both stages): the i32 values of `spec`.
    let entries: Vec<_> = (0..spec.len() as u32)
        .map(|i| vk::SpecializationMapEntry { constant_id: i, offset: 4 * i, size: 4 })
        .collect();
    let data: Vec<u8> = spec.iter().flat_map(|v| v.to_ne_bytes()).collect();
    let specialization = vk::SpecializationInfo::default().map_entries(&entries).data(&data);
    let stages = [
        vertex.stage(vk::ShaderStageFlags::VERTEX).specialization_info(&specialization),
        fragment.stage(vk::ShaderStageFlags::FRAGMENT).specialization_info(&specialization),
    ];
    let input = vk::PipelineVertexInputStateCreateInfo::default();
    let assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(if points { vk::PrimitiveTopology::POINT_LIST } else { vk::PrimitiveTopology::TRIANGLE_LIST });
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
    // Statistics for main.rs print_stats, where the device has them enabled.
    let flags = match CAPTURE_STATS.load(std::sync::atomic::Ordering::Relaxed) {
        true => vk::PipelineCreateFlags::CAPTURE_STATISTICS_KHR,
        false => vk::PipelineCreateFlags::empty(),
    };
    let info = [vk::GraphicsPipelineCreateInfo::default()
        .flags(flags)
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
    /// `spec`: the net's layout (rule.rs specialization: taps, levels = mip levels of
    /// the resolved state, layers, units); `weight_count`: weights (halves) in the weights buffer.
    pub fn new(
        r: &mut Resources<'_>,
        shaders: &Path,
        size: vk::Extent2D,
        output_format: vk::Format,
        swapchains: [&[vk::Image]; EYES],
        spec: [i32; 8],
        weight_count: usize,
        cube_face: u32,
    ) -> Result<Self> {
        let levels = spec[1] as u32;
        use vk::ImageUsageFlags as Usage;
        let d = &r.gpu.device;
        let stages = vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT;

        let bindings: Vec<_> = (0..=LAST_BINDING)
            .map(|i| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(i)
                    .descriptor_count(1)
                    .stage_flags(stages)
                    .descriptor_type(match i {
                        0 => vk::DescriptorType::UNIFORM_BUFFER,
                        WEIGHTS_BINDING => vk::DescriptorType::UNIFORM_BUFFER,
                        CREATURES_BINDING => vk::DescriptorType::STORAGE_BUFFER,
                        _ => vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
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
        // Room for the panorama's push (PANORAMA_PUSH floats); the others use the first 12 bytes.
        let push = [vk::PushConstantRange::default()
            .stage_flags(stages)
            .size((4 * PANORAMA_PUSH) as u32)];
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
                .descriptor_count(2 * (SAMPLED_BINDINGS + CUBE_IMAGES)),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(2),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(2),
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
        // The pyramid: bilinear within a level, exact levels (textureLod with integers).
        let linear = r.keep(
            unsafe {
                d.create_sampler(
                    &vk::SamplerCreateInfo::default()
                        .mag_filter(vk::Filter::LINEAR)
                        .min_filter(vk::Filter::LINEAR)
                        .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
                        .max_lod(vk::LOD_CLAMP_NONE)
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
        let scene_pass = render_pass(r, &[STATE; SCENES], true, clear, read_layout, true)?;
        let four = [STATE, STATE, STATE, SINGLE];
        let splat_pass = render_pass(r, &four, false, clear, read_layout, true)?;
        let dont_care = vk::AttachmentLoadOp::DONT_CARE;
        let resolve_pass = render_pass(r, &[STATE; 4], false, dont_care, read_layout, true)?;
        let net_pass = render_pass(r, &[STATE; STATES], false, dont_care, read_layout, true)?;
        let display_pass = render_pass(
            r,
            &[output_format],
            false,
            vk::AttachmentLoadOp::DONT_CARE,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            false,
        )?;
        let forward_pass = render_pass(r, &[STATE], false, dont_care, read_layout, true)?;
        let cube_splat_pass = render_pass(r, &[ACC, ACC, ACC, ACC, ACC_SINGLE], false, clear, read_layout, true)?;
        let cube_band_pass = render_pass(r, &[STATE; STATES], false, vk::AttachmentLoadOp::LOAD, read_layout, true)?;
        let cube_fold_band_pass = render_pass(r, &[STATE; 4], false, vk::AttachmentLoadOp::LOAD, read_layout, true)?;
        let passes = [scene_pass, splat_pass, resolve_pass, net_pass, display_pass, forward_pass, cube_splat_pass];
        let mut pipelines = Self::pipelines(&r.gpu.device, shaders, passes, layout, &spec)?
            .into_iter()
            .map(|p| r.keep(p, Resource::Pipeline));
        let mut next = || pipelines.next().unwrap();
        let scene = next();
        let creatures = next();
        let splat = std::array::from_fn(|_| [next(), next()]);
        let [resolve, net, display] = std::array::from_fn(|_| [next(), next()]);
        let (cube_forward, cube_pull, cube_net) = (next(), next(), next());
        let cube_display = [next(), next()];
        let (cube_splat, cube_fold) = (next(), next());

        // state[eye][parity] = (A, B, C)
        let state_usage = Usage::COLOR_ATTACHMENT | Usage::SAMPLED | Usage::TRANSFER_DST | Usage::TRANSFER_SRC;
        let target_usage = Usage::COLOR_ATTACHMENT | Usage::SAMPLED;
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
        let mut acc = [[vk::ImageView::null(); ACCS]; EYES];
        for eye in &mut acc {
            for (i, view) in eye.iter_mut().enumerate() {
                *view = r.image(size, if i < STATES { STATE } else { SINGLE }, target_usage)?.1;
            }
        }
        // Resolved state: written at level 0 (`mid_target`), sampled with all levels.
        let mid_usage = target_usage | Usage::TRANSFER_SRC | Usage::TRANSFER_DST;
        let mut mid = [[vk::ImageView::null(); STATES]; EYES];
        let mut mid_target = [[vk::ImageView::null(); STATES]; EYES];
        let mut mid_images = [[vk::Image::null(); STATES]; EYES];
        for eye in 0..EYES {
            for i in 0..STATES {
                let (image, level0) = r.image_levels(size, STATE, mid_usage, levels)?;
                mid_images[eye][i] = image;
                mid_target[eye][i] = level0;
                mid[eye][i] = r.view_levels(image, STATE, vk::ImageAspectFlags::COLOR, levels)?;
            }
        }
        let mut d_view = [vk::ImageView::null(); EYES];
        let mut d_target = [vk::ImageView::null(); EYES];
        let mut d_images = [vk::Image::null(); EYES];
        for eye in 0..EYES {
            let (image, level0) = r.image_levels(size, STATE, mid_usage, levels)?;
            (d_images[eye], d_target[eye]) = (image, level0);
            d_view[eye] = r.view_levels(image, STATE, vk::ImageAspectFlags::COLOR, levels)?;
        }
        let mut gbuffer = [[vk::ImageView::null(); SCENES]; EYES];
        let mut scene_targets = [vk::Framebuffer::null(); EYES];
        let mut acc_targets = [vk::Framebuffer::null(); EYES];
        let mut resolve_targets = [vk::Framebuffer::null(); EYES];
        let mut net_targets = [[vk::Framebuffer::null(); 2]; EYES];
        for eye in 0..EYES {
            for view in &mut gbuffer[eye] {
                *view = r.image(size, STATE, target_usage)?.1;
            }
            let depth = r.image(size, DEPTH, Usage::DEPTH_STENCIL_ATTACHMENT)?.1;
            scene_targets[eye] = r.framebuffer(scene_pass, size, &[gbuffer[eye][0], gbuffer[eye][1], depth])?;
            acc_targets[eye] = r.framebuffer(splat_pass, size, &acc[eye])?;
            let [a, b, c] = mid_target[eye];
            resolve_targets[eye] = r.framebuffer(resolve_pass, size, &[a, b, c, d_target[eye]])?;
            for parity in 0..2 {
                net_targets[eye][parity] = r.framebuffer(net_pass, size, &state[eye][1 - parity])?;
            }
        }
        let mut outputs: [Vec<vk::Framebuffer>; EYES] = Default::default();
        for eye in 0..EYES {
            for &image in swapchains[eye] {
                let view = r.view(image, output_format, vk::ImageAspectFlags::COLOR)?;
                outputs[eye].push(r.framebuffer(display_pass, size, &[view])?);
            }
        }

        let cube = Cube::new(r, cube_face, levels, [forward_pass, resolve_pass, net_pass, cube_splat_pass])?;

        // Zeroed initial contents; every field is plain f32 data.
        let initial: Uniforms = unsafe { std::mem::zeroed() };
        let (uniforms, mapped) = r.buffer(&[initial], vk::BufferUsageFlags::UNIFORM_BUFFER)?;
        let pixels = vec![[0u16; 4]; (size.width * size.height) as usize];
        let readback = r.buffer(&pixels, vk::BufferUsageFlags::TRANSFER_DST)?;
        // Room for eye 0's images or the cube atlas (cube-v1), whichever is larger (a
        // larger cube later gets a larger buffer).
        let dump_capacity = dump_bytes(size).max(dump_bytes(cube.atlas));
        let dump = r.buffer(&vec![0u8; dump_capacity], vk::BufferUsageFlags::TRANSFER_DST)?;
        // The level about PROBE pixels wide (within the pyramid).
        let probe_level = (size.width / PROBE).max(1).ilog2().min(levels - 1);
        let probe_pixels = mip_extent(size, probe_level);
        let probe = r.buffer(
            &vec![[0u16; 4]; EYES * 2 * (probe_pixels.width * probe_pixels.height) as usize],
            vk::BufferUsageFlags::TRANSFER_DST,
        )?;
        anyhow::ensure!(weight_count <= WEIGHT_HALVES, "the net has {weight_count} weights, at most {WEIGHT_HALVES} fit");
        let (weight_buffer, weights) =
            r.buffer(&vec![0u16; WEIGHT_HALVES], vk::BufferUsageFlags::UNIFORM_BUFFER)?;
        let (creature_buffer, creature_vertices) = r.buffer(
            &vec![crate::creatures::Vertex::default(); crate::creatures::MAX_VERTICES],
            vk::BufferUsageFlags::STORAGE_BUFFER,
        )?;
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
            // Bindings 1.. : prev state, accumulators, resolved state (pyramid sampler),
            // D (pyramid sampler), new state (each of both eyes), scene G-buffer of both eyes.
            let views: Vec<_> = (0..EYES)
                .flat_map(|e| state[e][parity])
                .chain(acc.into_iter().flatten())
                .map(|v| (v, nearest))
                .chain(mid.into_iter().flatten().chain(d_view).map(|v| (v, linear)))
                .chain((0..EYES).flat_map(|e| state[e][next]).chain(gbuffer.into_iter().flatten()).map(|v| (v, nearest)))
                .collect();
            assert_eq!(views.len(), SAMPLED_BINDINGS as usize);
            let views: Vec<_> = views
                .into_iter()
                .map(|(view, sampler)| {
                    [vk::DescriptorImageInfo::default()
                        .image_view(view)
                        .sampler(sampler)
                        .image_layout(read_layout)]
                })
                .collect();
            let storage = [vk::DescriptorBufferInfo::default().buffer(weight_buffer).range(vk::WHOLE_SIZE)];
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
            writes.push(
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(WEIGHTS_BINDING)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                    .buffer_info(&storage),
            );
            let creature_info = [vk::DescriptorBufferInfo::default().buffer(creature_buffer).range(vk::WHOLE_SIZE)];
            writes.push(
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(CREATURES_BINDING)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&creature_info),
            );
            unsafe { d.update_descriptor_sets(&writes, &[]) };
        }
        cube.bind(d, [allocated[0], allocated[1]], linear);

        Ok(Self {
            size,
            scene_pass,
            scene,
            creatures,
            creature_vertices,
            creature_count: 0,
            scene_targets,
            splat_pass,
            resolve_pass,
            net_pass,
            display_pass,
            splat,
            resolve,
            net,
            display,
            layout,
            sets: [allocated[0], allocated[1]],
            state_images,
            mid_images,
            d_images,
            levels,
            spec,
            weights,
            weight_count,
            dump,
            dumping: false,
            panorama: None,
            acc_targets,
            resolve_targets,
            net_targets,
            outputs,
            mapped,
            readback,
            capture: false,
            probe,
            probe_level,
            probe_cameras: [EyeCamera { view_proj: Mat4::IDENTITY, position: Vec3::ZERO }; EYES],
            tone: [[0., 1., 1.]; 3],
            tone_goal: [[0., 1., 1.]; 3],
            previous: None,
            parity: 0,
            frame: 0,
            mode: 0,
            forward_pass,
            cube_forward,
            cube_pull,
            cube_net,
            cube_display,
            cube_splat_pass,
            split: false,
            stage: None,
            cube_band_pass,
            cube_fold_band_pass,
            cube_splat,
            cube_fold,
            splat_nearness: None,
            cube,
            cube_clear: true,
            cube_centre: None,
            linear,
            dump_capacity,
        })
    }

    /// This frame's creature triangles (world space), drawn in the scene pass. Call before
    /// `draw`, after the previous frame's commands finished (the buffer is not doubled).
    pub fn upload_creatures(&mut self, vertices: &[crate::creatures::Vertex]) {
        let count = vertices.len().min(crate::creatures::MAX_VERTICES);
        unsafe { self.creature_vertices.copy_from_nonoverlapping(vertices.as_ptr(), count) };
        self.creature_count = count as u32;
    }

    pub fn size(&self) -> vk::Extent2D {
        self.size
    }

    /// The pipelines (scene, splat per target and source eye, then resolve, net, display
    /// per eye) from the SPIR-V in `shaders`. They get the net's layout as constants 0..6,
    /// their eye as 7 and the splat's source as 8 (state.glsl TARGET, SOURCE), so the
    /// per-eye image picks compile away.
    fn pipelines(
        d: &Device,
        shaders: &Path,
        [scene, splat, resolve, net, display, forward, cube_splat]: [vk::RenderPass; 7],
        layout: vk::PipelineLayout,
        spec: &[i32],
    ) -> Result<Vec<vk::Pipeline>> {
        let with = |extra: &[i32]| -> Vec<i32> { spec.iter().chain(extra).copied().collect() };
        [
            pipeline(d, shaders, ("scene.vert.spv", "scene.frag.spv"), scene, layout, SCENES, false, true, false, &[]),
            pipeline(d, shaders, ("creatures.vert.spv", "creatures.frag.spv"), scene, layout, SCENES, false, true, true, &[]),
        ]
        .into_iter()
        .chain((0..EYES as i32 * EYES as i32).map(|i| {
            let spec = with(&[i / EYES as i32, i % EYES as i32]);
            pipeline(d, shaders, ("splat.vert.spv", "splat.frag.spv"), splat, layout, ACCS, true, false, true, &spec)
        }))
        .chain(
            [("resolve.frag.spv", resolve, ACCS), ("net.frag.spv", net, STATES), ("display.frag.spv", display, 1)]
                .into_iter()
                .flat_map(|(fragment, pass, targets)| {
                    (0..EYES as i32).map(move |eye| {
                        let spec = with(&[eye]);
                        pipeline(d, shaders, ("fullscreen.vert.spv", fragment), pass, layout, targets, false, false, false, &spec)
                    })
                }),
        )
        // cube-v1: forward, pull, net (net.frag, LAYOUT from the net), display per eye.
        .chain([
            ("cube_forward.frag.spv", forward, 1, 0),
            ("cube_pull.frag.spv", resolve, ACCS, 0),
            ("net.frag.spv", net, STATES, 0),
            ("cube_display.frag.spv", display, 1, 0),
            ("cube_display.frag.spv", display, 1, 1),
        ]
        .into_iter()
        .map(|(fragment, pass, targets, eye)| {
            pipeline(d, shaders, ("fullscreen.vert.spv", fragment), pass, layout, targets, false, false, false, &with(&[eye]))
        }))
        // cube-v1 splat reprojection: the points (additive), the fold into mid A..D.
        .chain([
            pipeline(d, shaders, ("cube_splat.vert.spv", "cube_splat.frag.spv"), cube_splat, layout, CUBE_ACCS, true, false, true, &with(&[0])),
            pipeline(d, shaders, ("fullscreen.vert.spv", "cube_fold.frag.spv"), resolve, layout, ACCS, false, false, false, &with(&[0])),
        ])
        .collect()
    }

    pub fn named_pipelines(&self) -> [(&'static str, vk::Pipeline); 5] {
        [("scene", self.scene), ("splat", self.splat[0][0]), ("resolve", self.resolve[0]), ("net", self.net[0]), ("display", self.display[0])]
    }

    /// Recreate the pipelines from the (re)compiled SPIR-V in `shaders`.
    /// The caller must make sure the GPU is idle. On error the old pipelines stay
    /// (a partial success leaks the new ones until exit, like replaced ones).
    /// Old pipelines are only destroyed at exit (they stay in `Resources`).
    pub fn reload(&mut self, r: &mut Resources<'_>, shaders: &Path) -> Result<()> {
        let passes = [self.scene_pass, self.splat_pass, self.resolve_pass, self.net_pass, self.display_pass, self.forward_pass, self.cube_splat_pass];
        let new = Self::pipelines(&r.gpu.device, shaders, passes, self.layout, &self.spec)?;
        let mut new = new.into_iter().map(|p| r.keep(p, Resource::Pipeline));
        let mut next = || new.next().unwrap();
        self.scene = next();
        self.creatures = next();
        self.splat = std::array::from_fn(|_| [next(), next()]);
        [self.resolve, self.net, self.display] = std::array::from_fn(|_| [next(), next()]);
        (self.cube_forward, self.cube_pull, self.cube_net) = (next(), next(), next());
        self.cube_display = [next(), next()];
        (self.cube_splat, self.cube_fold) = (next(), next());
        if let Some(panorama) = &self.panorama {
            let pipeline = self.panorama_pipeline(r, shaders, panorama.pass)?;
            self.panorama.as_mut().unwrap().pipeline = pipeline;
        }
        Ok(())
    }

    fn panorama_pipeline(&self, r: &mut Resources<'_>, shaders: &Path, pass: vk::RenderPass) -> Result<vk::Pipeline> {
        let spec: Vec<i32> = self.spec.iter().chain(&[0]).copied().collect();
        let pipeline = pipeline(&r.gpu.device, shaders, ("fullscreen.vert.spv", "panorama.frag.spv"), pass, self.layout, 1, false, false, false, &spec)?;
        Ok(r.keep(pipeline, Resource::Pipeline))
    }

    /// DESKTOP: the whole-sphere view (panorama.frag) into images of `format`; returns
    /// the render pass for their framebuffers (they end in COLOR_ATTACHMENT_OPTIMAL).
    pub fn enable_panorama(&mut self, r: &mut Resources<'_>, shaders: &Path, format: vk::Format) -> Result<vk::RenderPass> {
        let dont_care = vk::AttachmentLoadOp::DONT_CARE;
        let pass = render_pass(r, &[format], false, dont_care, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL, false)?;
        let pipeline = self.panorama_pipeline(r, shaders, pass)?;
        self.panorama = Some(Panorama { pass, pipeline });
        Ok(pass)
    }

    /// Record the panorama (after `draw` in the same command buffer: its uniforms, the
    /// latest state) into `framebuffer` (from `enable_panorama`'s pass).
    pub fn draw_panorama(&self, d: &Device, cmd: vk::CommandBuffer, framebuffer: vk::Framebuffer, extent: vk::Extent2D, push: [f32; PANORAMA_PUSH]) {
        let panorama = self.panorama.as_ref().expect("enable_panorama first");
        let stages = vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT;
        let area = vk::Rect2D::default().extent(extent);
        let viewport = vk::Viewport::default().width(extent.width as f32).height(extent.height as f32).max_depth(1.);
        let bytes: Vec<u8> = push.iter().flat_map(|v| v.to_ne_bytes()).collect();
        unsafe {
            // After draw the parity has flipped on an update: this set's prev is the latest state.
            d.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::GRAPHICS, self.layout, 0, &[self.sets[self.parity]], &[]);
            let info = vk::RenderPassBeginInfo::default().render_pass(panorama.pass).framebuffer(framebuffer).render_area(area);
            d.cmd_begin_render_pass(cmd, &info, vk::SubpassContents::INLINE);
            d.cmd_set_viewport(cmd, 0, &[viewport]);
            d.cmd_set_scissor(cmd, 0, &[area]);
            d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, panorama.pipeline);
            d.cmd_push_constants(cmd, self.layout, stages, 0, &bytes);
            d.cmd_draw(cmd, 3, 1, 0, 0);
            d.cmd_end_render_pass(cmd);
        }
    }

    /// Switch to another net live (main.rs NET_FILE): its specialization constants and
    /// weight count, then the pipelines rebuilt as in `reload`. The pyramid depth is fixed
    /// by the images, so the net must read as many mip levels. The caller makes sure the
    /// GPU is idle; on error nothing changes.
    pub fn set_net(&mut self, r: &mut Resources<'_>, shaders: &Path, spec: [i32; 8], weight_count: usize) -> Result<()> {
        anyhow::ensure!(spec[1] as u32 == self.levels, "the net reads {} mip levels, the running app has {} (restart for it)", spec[1], self.levels);
        anyhow::ensure!(weight_count <= WEIGHT_HALVES, "the net has {weight_count} weights, at most {WEIGHT_HALVES} fit");
        let old = self.spec;
        self.spec = spec;
        if let Err(error) = self.reload(r, shaders) {
            self.spec = old;
            return Err(error);
        }
        self.weight_count = weight_count;
        if (old[7] == 3) != (spec[7] == 3) {
            self.cube_centre = None; // a new store starts where the head is
        }
        Ok(())
    }

    /// Give the sphere store `face` texels per face: new, cleared images (the old ones stay
    /// in `Resources` until exit). The caller makes sure the GPU is idle.
    pub fn set_cube_face(&mut self, r: &mut Resources<'_>, face: u32) -> Result<()> {
        if face == self.cube.face {
            return Ok(());
        }
        let cube = Cube::new(r, face, self.levels, [self.forward_pass, self.resolve_pass, self.net_pass, self.cube_splat_pass])?;
        if dump_bytes(cube.atlas) > self.dump_capacity {
            self.dump_capacity = dump_bytes(cube.atlas);
            self.dump = r.buffer(&vec![0u8; self.dump_capacity], vk::BufferUsageFlags::TRANSFER_DST)?;
        }
        cube.bind(&r.gpu.device, self.sets, self.linear);
        self.cube = cube;
        self.cube_clear = true;
        self.cube_centre = None;
        Ok(())
    }

    /// Record one frame for both eyes: scene, splat, resolve, net, display. Writes
    /// timestamps 1..STAMPS-1 into `queries` after each stage (0 is the caller's).
    pub fn draw(
        &mut self,
        d: &Device,
        cmd: vk::CommandBuffer,
        queries: vk::QueryPool,
        cameras: [EyeCamera; EYES],
        images: [usize; EYES],
        time: f32,
        rule: &RuleFrame,
        weights: &[u16],
        settings: Settings,
    ) {
        if self.previous.is_none() {
            clear_to_read(d, cmd, &self.state_images, 1);
            // Pyramids too: a layout's passes never write the other layout's images, and
            // the descriptors expect every image readable.
            let pyramids: Vec<_> = self.mid_images.as_flattened().iter().chain(&self.d_images).copied().collect();
            clear_to_read(d, cmd, &pyramids, self.levels);
        }
        if std::mem::take(&mut self.cube_clear) {
            for (image, levels) in self.cube.images(self.levels) {
                clear_to_read(d, cmd, &[image], levels);
            }
        }
        // cube-v1: the store updates on the frames where the net runs; it moves with the head.
        let cube = self.spec[7] == 3;
        let update = cube && settings.tau > 0.;
        let head = (cameras[0].position + cameras[1].position) * 0.5;
        let head_move = match (update, self.cube_centre) {
            (true, Some(centre)) => head - centre,
            _ => Vec3::ZERO,
        };
        if update {
            self.cube_centre = Some(head);
        }
        let centre = self.cube_centre.unwrap_or(head);
        let texel_angle = std::f32::consts::FRAC_PI_2 / self.cube.face as f32;
        let arcmin = (1f32 / 60.).to_radians();
        let (previous, last_time) = self.previous.replace((cameras, time)).unwrap_or((cameras, time));
        let dt = (time - last_time).clamp(0., 0.1);
        // Exposure glides toward the last measurement every frame (no steps when measured).
        let follow = 1. - (-EXPOSURE_RATE * dt).exp();
        for (tone, goal) in self.tone.iter_mut().zip(self.tone_goal) {
            for i in 0..3 {
                tone[i] += (goal[i] - tone[i]) * follow;
            }
        }
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
            step: [dt, settings.tau, settings.day.unwrap_or(-1.), self.splat_nearness.unwrap_or(0.)],
            net: rule.net,
            net_run: [rule.run[0], settings.gate, rule.run[2], 0.],
            net_layers: rule.layers,
            net_taps: rule.taps,
            city: [settings.cell, settings.height, settings.crystals, settings.blocks as f32],
            // Per-second rates as this frame's blend weights (frame-rate independent).
            mixing: [1. - (-settings.color * dt).exp(), if settings.sky > 0. { 1. / settings.sky.max(1.) } else { 0. }, 1. - (-settings.depth * dt).exp(), settings.focus],
            player: {
                let ground = |c: [EyeCamera; EYES]| (c[0].position + c[1].position) * 0.5;
                let (now, before) = (ground(cameras), ground(previous));
                [now.x, now.z, before.x, before.z]
            },
            tone: self.tone_uniform(),
            gaze: {
                // Per eye: the gaze point in centred view coordinates (x right, y down, -1..1).
                let at = |camera: &EyeCamera| {
                    settings.gaze.map_or([0., 0.], |point| {
                        let clip = camera.view_proj * point.extend(1.);
                        if clip.w <= 1e-4 { return [0., 0.] }
                        [(clip.x / clip.w).clamp(-1.5, 1.5), (-clip.y / clip.w).clamp(-1.5, 1.5)]
                    })
                };
                let [l, r] = [at(&cameras[0]), at(&cameras[1])];
                [l[0], l[1], r[0], r[1]]
            },
            // Net time of this update (tau ticks at the net's rate), the scale input
            // (model.py scale_code), the texel angle.
            cube: [self.cube.face as f32, settings.tau / rule.run[2].max(1e-3), (texel_angle / arcmin).log2() / 4., texel_angle],
            head: centre.extend(self.cube.gutter as f32).to_array(),
            head_move: head_move.extend(0.).to_array(),
        };
        let zero = [vk::ClearValue {
            color: vk::ClearColorValue { float32: [0.; 4] },
        }; CUBE_ACCS]; // enough for every pass that clears (extra values are ignored)
        let scene_clear = [
            zero[0],
            zero[1],
            vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue { depth: 1., stencil: 0 },
            },
        ];
        let stages = vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT;
        let pixels = self.size.width * self.size.height;
        unsafe {
            self.mapped.write(uniforms);
            let count = weights.len().min(self.weight_count);
            self.weights.copy_from_nonoverlapping(weights.as_ptr(), count);
            d.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.layout,
                0,
                &[self.sets[self.parity]],
                &[],
            );
            let begin_sized = |pass, framebuffer, extent: vk::Extent2D| {
                let clears: &[vk::ClearValue] = if pass == self.scene_pass { &scene_clear } else { &zero };
                let area = vk::Rect2D::default().extent(extent);
                let viewport = vk::Viewport::default()
                    .width(extent.width as f32)
                    .height(extent.height as f32)
                    .max_depth(1.);
                let info = vk::RenderPassBeginInfo::default()
                    .render_pass(pass)
                    .framebuffer(framebuffer)
                    .render_area(area)
                    .clear_values(clears);
                d.cmd_begin_render_pass(cmd, &info, vk::SubpassContents::INLINE);
                d.cmd_set_viewport(cmd, 0, &[viewport]);
                d.cmd_set_scissor(cmd, 0, &[area]);
            };
            let begin = |pass, framebuffer| begin_sized(pass, framebuffer, self.size);
            let push = |target: usize, source: usize, mode: i32| {
                let p = Push {
                    target: target as i32,
                    source: source as i32,
                    mode,
                };
                d.cmd_push_constants(cmd, self.layout, stages, 0, &p.bytes());
            };
            let stamp = |index: u32| {
                d.cmd_write_timestamp(cmd, vk::PipelineStageFlags::BOTTOM_OF_PIPE, queries, index)
            };
            // Scene first: rendered before the reprojection, composited by resolve (cube:
            // by cube_pull and net on update frames; the display's scene view every frame).
            for eye in 0..EYES {
                begin(self.scene_pass, self.scene_targets[eye]);
                d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.scene);
                push(eye, eye, 0);
                // The blocks, and one more instance: the ground plate.
                let blocks = settings.blocks * settings.blocks;
                if settings.city {
                    d.cmd_draw(cmd, SCENE_VERTICES, blocks + 1, 0, 0);
                } else {
                    d.cmd_draw(cmd, SCENE_VERTICES, 1, 0, blocks);
                }
                if self.creature_count > 0 {
                    d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.creatures);
                    push(eye, eye, (settings.dot * 1000.).round() as i32);
                    d.cmd_draw(cmd, self.creature_count, 1, 0, 0);
                }
                d.cmd_end_render_pass(cmd);
            }
            stamp(1);
            if cube {
                let fullscreen = |pass, target, pipeline| {
                    begin_sized(pass, target, self.cube.atlas);
                    d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline);
                    push(0, 0, 0);
                    d.cmd_draw(cmd, 3, 1, 0, 0);
                    d.cmd_end_render_pass(cmd);
                };
                let splat = self.splat_nearness.is_some();
                if update && splat {
                    // Splat: every face texel a point into the accumulators, then the fold.
                    begin_sized(self.cube_splat_pass, self.cube.splat_target, self.cube.atlas);
                    d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.cube_splat);
                    push(0, 0, 0);
                    d.cmd_draw(cmd, 6 * self.cube.face * self.cube.face, 1, 0, 0);
                    d.cmd_end_render_pass(cmd);
                    stamp(2);
                } else if update {
                    fullscreen(self.forward_pass, self.cube.forward_target, self.cube_forward);
                    build_mips(d, cmd, &[self.cube.m], self.cube.atlas, self.levels);
                    stamp(2);
                } else {
                    stamp(2);
                }
                if self.split {
                    // The rest in `advance` (the caller's stamps from here on).
                    self.stage = update.then_some(1);
                    self.frame += 1;
                    return;
                }
                if update {
                    fullscreen(self.resolve_pass, self.cube.pull_target, if splat { self.cube_fold } else { self.cube_pull });
                    build_mips(d, cmd, &self.cube.mid, self.cube.atlas, self.levels);
                    stamp(3);
                    fullscreen(self.net_pass, self.cube.net_targets[self.parity], self.cube_net);
                    stamp(4);
                } else {
                    stamp(3);
                    stamp(4);
                }
                for eye in 0..EYES {
                    begin(self.display_pass, self.outputs[eye][images[eye]]);
                    d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.cube_display[eye]);
                    push(eye, update as usize, self.mode); // source 1: this update's state (next)
                    d.cmd_draw(cmd, 3, 1, 0, 0);
                    d.cmd_end_render_pass(cmd);
                }
                stamp(5);
                if update {
                    self.finish_update(d, cmd);
                } else if self.capture && self.capture_fits() {
                    copy_to_buffer(d, cmd, self.cube.state[self.parity][0], self.size, 0, self.readback.0, 0);
                }
                self.frame += 1;
                return;
            }
            for eye in 0..EYES {
                begin(self.splat_pass, self.acc_targets[eye]);
                // Own previous state, then the other eye's (cross-eye splat).
                for source in [eye, 1 - eye] {
                    d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.splat[eye][source]);
                    push(eye, source, 0);
                    d.cmd_draw(cmd, pixels, 1, 0, 0);
                }
                d.cmd_end_render_pass(cmd);
            }
            stamp(2);
            for (pass, pipeline, targets, index) in [
                (self.resolve_pass, self.resolve, self.resolve_targets, 3),
                (self.net_pass, self.net, self.net_targets.map(|t| t[self.parity]), 4),
            ] {
                for eye in 0..EYES {
                    begin(pass, targets[eye]);
                    d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline[eye]);
                    push(eye, eye, 0);
                    d.cmd_draw(cmd, 3, 1, 0, 0);
                    d.cmd_end_render_pass(cmd);
                }
                if index == 3 {
                    let mids: Vec<_> = self.mid_images.as_flattened().iter().chain(&self.d_images).copied().collect();
                    build_mips(d, cmd, &mids, self.size, self.levels);
                }
                stamp(index);
            }
            for eye in 0..EYES {
                begin(self.display_pass, self.outputs[eye][images[eye]]);
                d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.display[eye]);
                push(eye, eye, self.mode);
                d.cmd_draw(cmd, 3, 1, 0, 0);
                d.cmd_end_render_pass(cmd);
            }
            stamp(5);
        }
        if self.capture {
            // The new state A of the left eye: state[0][1 - parity], first image.
            let image = self.state_images[(1 - self.parity) * STATES];
            copy_to_buffer(d, cmd, image, self.size, 0, self.readback.0, 0);
        }
        if self.dumping {
            // Eye 0: resolved A, B, C, D, then the new A, B, C.
            let state = 8 * (self.size.width * self.size.height) as u64;
            let mut offset = 0;
            for image in self.mid_images[0].into_iter().chain([self.d_images[0]]) {
                copy_to_buffer(d, cmd, image, self.size, 0, self.dump.0, offset);
                offset += state;
            }
            for i in 0..STATES {
                let image = self.state_images[(1 - self.parity) * STATES + i];
                copy_to_buffer(d, cmd, image, self.size, 0, self.dump.0, offset);
                offset += state;
            }
        }
        // The probe: both eyes' resolved A and B at a small mip level.
        let probe = mip_extent(self.size, self.probe_level);
        let bytes = 8 * (probe.width * probe.height) as u64;
        for (i, image) in self.mid_images.iter().flat_map(|eye| &eye[..2]).enumerate() {
            copy_to_buffer(d, cmd, *image, self.size, self.probe_level, self.probe.0, i as u64 * bytes);
        }
        self.probe_cameras = cameras;
        self.parity = 1 - self.parity;
        self.frame += 1;
    }

    /// Whether a split update is in flight (`advance` has stages left).
    pub fn updating(&self) -> bool {
        self.stage.is_some()
    }

    /// Split updates: record the next stage of the update in flight: the fold (or pull) in
    /// FOLD_BANDS bands of rows (the last builds the net's mips), then the net in NET_BANDS. True when this stage
    /// finished the update (the parity has flipped: the panorama shows the new state).
    /// Several stages may follow each other in one command buffer, but no `draw` after
    /// them (it rewrites the uniforms they read).
    pub fn advance(&mut self, d: &Device, cmd: vk::CommandBuffer) -> bool {
        let Some(stage) = self.stage else { return false };
        let stages = vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT;
        let atlas = self.cube.atlas;
        // A fullscreen pass over the rows `rows` (the viewport stays the whole atlas).
        let pass = |pass, framebuffer, pipeline, rows: (u32, u32)| unsafe {
            let area = vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: rows.0 as i32 },
                extent: vk::Extent2D { width: atlas.width, height: rows.1 - rows.0 },
            };
            let viewport = vk::Viewport::default().width(atlas.width as f32).height(atlas.height as f32).max_depth(1.);
            let info = vk::RenderPassBeginInfo::default().render_pass(pass).framebuffer(framebuffer).render_area(area);
            d.cmd_begin_render_pass(cmd, &info, vk::SubpassContents::INLINE);
            d.cmd_set_viewport(cmd, 0, &[viewport]);
            d.cmd_set_scissor(cmd, 0, &[area]);
            d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline);
            d.cmd_draw(cmd, 3, 1, 0, 0);
            d.cmd_end_render_pass(cmd);
        };
        unsafe {
            d.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::GRAPHICS, self.layout, 0, &[self.sets[self.parity]], &[]);
            d.cmd_push_constants(cmd, self.layout, stages, 0, &Push { target: 0, source: 0, mode: 0 }.bytes());
        }
        let bands = |band: u32, count: u32| (atlas.height * band / count, atlas.height * (band + 1) / count);
        if stage <= FOLD_BANDS {
            // Level 0 only: the fold's pass keeps what it doesn't draw, like the net's.
            let pipeline = if self.splat_nearness.is_some() { self.cube_fold } else { self.cube_pull };
            pass(self.cube_fold_band_pass, self.cube.pull_target, pipeline, bands(stage - 1, FOLD_BANDS));
            if stage == FOLD_BANDS {
                build_mips(d, cmd, &self.cube.mid, atlas, self.levels);
            }
            self.stage = Some(stage + 1);
            return false;
        }
        let band = stage - 1 - FOLD_BANDS;
        pass(self.cube_band_pass, self.cube.net_targets[self.parity], self.cube_net, bands(band, NET_BANDS));
        if band + 1 < NET_BANDS {
            self.stage = Some(stage + 1);
            return false;
        }
        self.stage = None;
        self.finish_update(d, cmd);
        true
    }

    fn capture_fits(&self) -> bool {
        self.cube.atlas.width >= self.size.width && self.cube.atlas.height >= self.size.height
    }

    /// After a cube update's net: the exposure capture, dump and probe read back, the state
    /// before (split: kept as `old` for the display's blend), and the parity flipped.
    fn finish_update(&mut self, d: &Device, cmd: vk::CommandBuffer) {
        let latest = 1 - self.parity;
        if self.capture && self.capture_fits() {
            // Exposure: the top-left eye-sized part of the atlas (faces +x, -x, +y).
            copy_to_buffer(d, cmd, self.cube.state[latest][0], self.size, 0, self.readback.0, 0);
        }
        if self.dumping {
            // The net's input (mid A..D) and output (the new A, B, C) on the atlas.
            let bytes = 8 * (self.cube.atlas.width * self.cube.atlas.height) as u64;
            let images = self.cube.mid.iter().chain(&self.cube.state[latest]);
            for (i, &image) in images.enumerate() {
                copy_to_buffer(d, cmd, image, self.cube.atlas, 0, self.dump.0, i as u64 * bytes);
            }
        }
        // The probe: this update's pulled/splatted state (scene blended in) at a coarse level.
        let level = self.cube.probe_level;
        let texels = mip_extent(self.cube.atlas, level);
        let bytes = 8 * (texels.width * texels.height) as u64;
        for (i, &image) in self.cube.mid[..2].iter().enumerate() {
            copy_to_buffer(d, cmd, image, self.cube.atlas, level, self.cube.probe.0, i as u64 * bytes);
        }
        if self.split {
            for i in 0..2 {
                copy_image(d, cmd, self.cube.state[self.parity][i], self.cube.old[i], self.cube.atlas);
            }
        }
        self.parity = latest;
    }

    /// The dump of the last frame with `dumping` set (after its commands finished):
    /// eye 0's resolved A, B, C, D, new A, B, C (all RGBA16F).
    pub fn dumped(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.dump.1, dump_bytes(self.dump_extent())) }
    }

    /// cube-v1: texels per face and the atlas's gutter around each face.
    pub fn cube_layout(&self) -> (u32, u32) {
        (self.cube.face, self.cube.gutter)
    }

    /// The dumped images' size: eye 0's, or the cube atlas (cube-v1: 3 x 2 faces).
    pub fn dump_extent(&self) -> vk::Extent2D {
        if self.spec[7] == 3 { self.cube.atlas } else { self.size }
    }

    /// World points of the image's geometry around both eyes, from the last frame's
    /// probe (call after its commands finished): one per covered probe pixel.
    pub fn surroundings(&self) -> Vec<Vec3> {
        if self.spec[7] == 3 {
            return self.cube_surroundings();
        }
        let size = mip_extent(self.size, self.probe_level);
        let (w, h) = (size.width as usize, size.height as usize);
        let probe = unsafe { std::slice::from_raw_parts(self.probe.1, EYES * 2 * w * h) };
        let mut points = Vec::new();
        for (eye, camera) in self.probe_cameras.iter().enumerate() {
            let inverse = camera.view_proj.inverse();
            let (a, b) = (&probe[2 * eye * w * h..][..w * h], &probe[(2 * eye + 1) * w * h..][..w * h]);
            for (i, (a, b)) in a.iter().zip(b).enumerate() {
                let coverage = half_to_f32(b[3]);
                let inv = half_to_f32(a[3]) / coverage;
                if !(coverage > 0.5 && inv > 0.) {
                    continue;
                }
                // common.glsl ndcFromPixel: y points down in the image.
                let ndc = Vec2::new(((i % w) as f32 + 0.5) / w as f32 * 2. - 1., 1. - ((i / w) as f32 + 0.5) / h as f32 * 2.);
                let far = inverse * ndc.extend(0.).extend(1.);
                let dir = (far.truncate() / far.w - camera.position).normalize_or_zero();
                points.push(camera.position + dir / inv);
            }
        }
        points
    }

    /// cube-v1: the sphere probe (the last update's state, seen from the store's centre)
    /// as world points, one per face texel of the probe level with weight >= 0.3.
    fn cube_surroundings(&self) -> Vec<Vec3> {
        let Some(centre) = self.cube_centre else { return Vec::new() };
        let cube = &self.cube;
        let level = cube.probe_level;
        let texels = mip_extent(cube.atlas, level);
        let (w, h) = (texels.width as usize, texels.height as usize);
        let probe = unsafe { std::slice::from_raw_parts(cube.probe.1, 2 * w * h) };
        let (a, b) = probe.split_at(w * h);
        let step = 1u32 << level;
        let (cell, gutter, face) = ((cube.face + 2 * cube.gutter) / step, cube.gutter / step, cube.face / step);
        let angle = std::f32::consts::FRAC_PI_2 / cube.face as f32;
        let mut points = Vec::new();
        for (i, (a, b)) in a.iter().zip(b).enumerate() {
            let (x, y) = ((i % w) as u32, (i / w) as u32);
            let (lx, ly) = ((x % cell) as i32 - gutter as i32, (y % cell) as i32 - gutter as i32);
            if lx < 0 || ly < 0 || lx >= face as i32 || ly >= face as i32 {
                continue; // gutter
            }
            let weight = half_to_f32(b[3]);
            let inv = half_to_f32(a[3]) / weight;
            if !(weight >= 0.3 && inv > 0.) {
                continue;
            }
            let f = (x / cell + 3 * (y / cell)) as usize;
            // cube.glsl faceDir at the coarse texel's centre in fine texel coords.
            let fine = |l: i32| (l as f32 + 0.5) * step as f32 - 0.5;
            let t = Vec2::new(fine(lx), fine(ly)).map(|c| ((c + 0.5) * angle - std::f32::consts::FRAC_PI_4).tan());
            let (forward, up) = (CUBE_FORWARD[f], CUBE_UP[f]);
            let dir = (forward.cross(up) * t.x - up * t.y + forward).normalize();
            points.push(centre + dir / inv);
        }
        points
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
    /// Show the state's colour 1:1 (black 0, white 1, gamma 1), gliding there.
    pub fn expose_plainly(&mut self) {
        self.tone_goal = [[0., 1., 1.]; 3];
    }

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
        for (c, goal) in self.tone_goal.iter_mut().enumerate() {
            for i in 0..3 {
                let mean = (target[0][i] + target[1][i] + target[2][i]) / 3.;
                goal[i] = target[c][i] + (mean - target[c][i]) * tint;
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

fn dump_bytes(size: vk::Extent2D) -> usize {
    let pixels = (size.width * size.height) as usize;
    pixels * 8 * (2 * STATES + 1)
}

/// Fill mip levels 1.. of the resolved images (level 0 written by the resolve pass,
/// shader-readable) with 2x2 averages (linear blits halve exactly), leaving all
/// levels shader-readable.
fn build_mips(d: &Device, cmd: vk::CommandBuffer, images: &[vk::Image], size: vk::Extent2D, levels: u32) {
    use vk::{AccessFlags as A, ImageLayout as L, PipelineStageFlags as S};
    let barriers = |level: u32, count: u32, old, new, src, dst| -> Vec<_> {
        images
            .iter()
            .map(|&image| {
                vk::ImageMemoryBarrier::default()
                    .image(image)
                    .subresource_range(image_range(vk::ImageAspectFlags::COLOR).base_mip_level(level).level_count(count))
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .old_layout(old)
                    .new_layout(new)
                    .src_access_mask(src)
                    .dst_access_mask(dst)
            })
            .collect()
    };
    let none = vk::DependencyFlags::empty();
    let extent = |level: u32| vk::Offset3D {
        x: (size.width >> level).max(1) as i32,
        y: (size.height >> level).max(1) as i32,
        z: 1,
    };
    unsafe {
        let b = barriers(0, 1, L::SHADER_READ_ONLY_OPTIMAL, L::TRANSFER_SRC_OPTIMAL, A::COLOR_ATTACHMENT_WRITE, A::TRANSFER_READ);
        d.cmd_pipeline_barrier(cmd, S::COLOR_ATTACHMENT_OUTPUT, S::TRANSFER, none, &[], &[], &b);
        for level in 1..levels {
            let b = barriers(level, 1, L::UNDEFINED, L::TRANSFER_DST_OPTIMAL, A::empty(), A::TRANSFER_WRITE);
            d.cmd_pipeline_barrier(cmd, S::TOP_OF_PIPE, S::TRANSFER, none, &[], &[], &b);
            let layers = |mip_level| vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level,
                base_array_layer: 0,
                layer_count: 1,
            };
            let region = vk::ImageBlit::default()
                .src_subresource(layers(level - 1))
                .src_offsets([vk::Offset3D::default(), extent(level - 1)])
                .dst_subresource(layers(level))
                .dst_offsets([vk::Offset3D::default(), extent(level)]);
            for &image in images {
                d.cmd_blit_image(cmd, image, L::TRANSFER_SRC_OPTIMAL, image, L::TRANSFER_DST_OPTIMAL, &[region], vk::Filter::LINEAR);
            }
            let b = barriers(level, 1, L::TRANSFER_DST_OPTIMAL, L::TRANSFER_SRC_OPTIMAL, A::TRANSFER_WRITE, A::TRANSFER_READ);
            d.cmd_pipeline_barrier(cmd, S::TRANSFER, S::TRANSFER, none, &[], &[], &b);
        }
        let b = barriers(0, levels, L::TRANSFER_SRC_OPTIMAL, L::SHADER_READ_ONLY_OPTIMAL, A::TRANSFER_READ | A::TRANSFER_WRITE, A::SHADER_READ);
        d.cmd_pipeline_barrier(cmd, S::TRANSFER, S::FRAGMENT_SHADER, none, &[], &[], &b);
    }
}

/// Copy a shader-readable colour image (level 0) into a buffer at `offset` bytes,
/// leaving it shader-readable.
/// Copy a whole single-level colour image into another of the same size; both are
/// shader-readable before and after.
fn copy_image(d: &Device, cmd: vk::CommandBuffer, from: vk::Image, to: vk::Image, size: vk::Extent2D) {
    use vk::{AccessFlags as A, ImageLayout as L, PipelineStageFlags as S};
    let barrier = |image, old, new, src, dst| {
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
    let read = L::SHADER_READ_ONLY_OPTIMAL;
    let layers = vk::ImageSubresourceLayers { aspect_mask: vk::ImageAspectFlags::COLOR, mip_level: 0, base_array_layer: 0, layer_count: 1 };
    let region = vk::ImageCopy::default()
        .src_subresource(layers)
        .dst_subresource(layers)
        .extent(vk::Extent3D { width: size.width, height: size.height, depth: 1 });
    let none = vk::DependencyFlags::empty();
    let shaders = S::VERTEX_SHADER | S::FRAGMENT_SHADER;
    unsafe {
        let before = [
            barrier(from, read, L::TRANSFER_SRC_OPTIMAL, A::COLOR_ATTACHMENT_WRITE | A::SHADER_READ, A::TRANSFER_READ),
            barrier(to, read, L::TRANSFER_DST_OPTIMAL, A::SHADER_READ, A::TRANSFER_WRITE),
        ];
        d.cmd_pipeline_barrier(cmd, shaders | S::COLOR_ATTACHMENT_OUTPUT, S::TRANSFER, none, &[], &[], &before);
        d.cmd_copy_image(cmd, from, L::TRANSFER_SRC_OPTIMAL, to, L::TRANSFER_DST_OPTIMAL, &[region]);
        let after = [
            barrier(from, L::TRANSFER_SRC_OPTIMAL, read, A::TRANSFER_READ, A::SHADER_READ),
            barrier(to, L::TRANSFER_DST_OPTIMAL, read, A::TRANSFER_WRITE, A::SHADER_READ),
        ];
        d.cmd_pipeline_barrier(cmd, S::TRANSFER, shaders, none, &[], &[], &after);
    }
}

fn copy_to_buffer(
    d: &Device,
    cmd: vk::CommandBuffer,
    image: vk::Image,
    size: vk::Extent2D,
    level: u32,
    buffer: vk::Buffer,
    offset: u64,
) {
    use vk::{AccessFlags as A, ImageLayout as L, PipelineStageFlags as S};
    let barrier = |old, new, src, dst| {
        vk::ImageMemoryBarrier::default()
            .image(image)
            .subresource_range(image_range(vk::ImageAspectFlags::COLOR).base_mip_level(level))
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .old_layout(old)
            .new_layout(new)
            .src_access_mask(src)
            .dst_access_mask(dst)
    };
    let region = vk::BufferImageCopy::default()
        .buffer_offset(offset)
        .image_subresource(vk::ImageSubresourceLayers {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            mip_level: level,
            base_array_layer: 0,
            layer_count: 1,
        })
        .image_extent({
            let size = mip_extent(size, level);
            vk::Extent3D { width: size.width, height: size.height, depth: 1 }
        });
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

fn mip_extent(size: vk::Extent2D, level: u32) -> vk::Extent2D {
    vk::Extent2D { width: (size.width >> level).max(1), height: (size.height >> level).max(1) }
}

/// Zero all state images (no depth = nothing to splat, certainty 0) and make them readable.
fn clear_to_read(d: &Device, cmd: vk::CommandBuffer, images: &[vk::Image], levels: u32) {
    use vk::{AccessFlags as A, ImageLayout as L, PipelineStageFlags as S};
    let range = image_range(vk::ImageAspectFlags::COLOR).level_count(levels);
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
