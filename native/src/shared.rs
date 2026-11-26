//! What the VR loop (main.rs) and the desktop loop (desktop.rs) share: env settings, the
//! faders turned into the world's settings, the city, the creatures and the exposure, the
//! cube's size, and the per-frame Vulkan objects.

use crate::gpu::{Resource, Resources};
use crate::world::{self, Settings, World};
use crate::{city, per_second, rule, swarm};
use anyhow::Result;
use ash::vk;
use av_controls::Controls;
use glam::{Vec2, Vec3};
use std::{
    env,
    str::FromStr,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
};

/// An env var parsed as `T`, else `default` (unset or unparsable).
pub fn env_or<T: FromStr>(name: &str, default: T) -> T {
    env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

/// Set by Ctrl+C / SIGTERM: the loops end cleanly instead of dying mid-frame.
pub fn quit_flag() -> Result<Arc<AtomicBool>> {
    let quit = Arc::new(AtomicBool::new(false));
    let flag = quit.clone();
    ctrlc::set_handler(move || flag.store(true, Ordering::SeqCst))?;
    Ok(quit)
}

pub fn print_net(rule: &rule::Rule) {
    println!(
        "Net: {} {} ({} weights, {} mip levels, {} reprojection)",
        if rule.trained() { "trained" } else { "random" },
        rule.layout_name(),
        rule.len(),
        rule.levels(),
        if rule.splat.is_some() { "splat" } else { "pull" }
    );
}

/// cube-v1's texels per face for eye pixels of `pixel` radians at the view centre: a texel
/// about a pixel (the net's finest taps, radius 1 texel, then reach about 1 px), divisible
/// by the coarsest mip level's step.
pub fn cube_face(pixel: f32, levels: u32) -> u32 {
    let step = 1u32 << (levels - 1);
    ((std::f32::consts::FRAC_PI_2 / pixel / step as f32).round() as u32).max(1) * step
}

/// The city tab's blocks around `player` (x, z).
pub fn city(controls: &Controls, player: Vec2) -> city::City {
    city::City {
        cell: controls.get("cell"),
        height: controls.get("height"),
        blocks: controls.get("blocks").round().max(1.) as u32,
        player,
    }
}

/// The world's settings from the faders; `tau` the net's step (0: no update), `gaze` a
/// world point looked at.
pub fn settings(controls: &Controls, seconds: f32, tau: f32, gaze: Option<Vec3>) -> Settings {
    Settings {
        wild: controls.get("wild"),
        gate: controls.get("gate"),
        tau,
        color: per_second(controls.get("color")),
        depth: per_second(controls.get("depth")),
        sky: controls.get("sky"),
        focus: controls.get("focus"),
        day: (controls.get("day") > 0.).then(|| (seconds / controls.get("day") + 0.25).fract()),
        gaze,
        blocks: controls.get("blocks").round().max(1.) as u32,
        cell: controls.get("cell"),
        height: controls.get("height"),
        crystals: controls.get("crystals"),
        city: controls.on("city"),
        dot: controls.get("dot"),
    }
}

/// The creatures tab: the swarm steps around `player` and goes to the GPU (none when off).
pub fn update_creatures(controls: &Controls, swarm: &mut swarm::Swarm, world: &mut World, seconds: f32, dt: f32, player: Vec3) {
    if !controls.on("creatures") {
        world.upload_creatures(&[]);
        return;
    }
    let params = swarm::Params {
        count: controls.get("count").round().max(1.) as usize,
        scale: controls.get("scale"),
        hold: controls.get("hold"),
        link: controls.get("link"),
        drag: controls.get("damp"),
        flow: controls.get("flow"),
        swirl: controls.get("swirl"),
        follow: 10.,
    };
    swarm.update(params, seconds, dt, player);
    world.upload_creatures(&swarm.out);
}

/// The display tab's exposure from the last capture (call after its commands finished).
pub fn expose(controls: &Controls, world: &mut World) {
    if controls.on("exposure") {
        world.calibrate(controls.get("mid"), controls.get("clip"), controls.get("black"), controls.get("tint"));
    } else {
        world.expose_plainly();
    }
}

/// `N` resettable command buffers from a new pool.
pub fn command_buffers<const N: usize>(r: &mut Resources<'_>) -> Result<[vk::CommandBuffer; N]> {
    let d = &r.gpu.device;
    let info = vk::CommandPoolCreateInfo::default()
        .queue_family_index(r.gpu.family)
        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
    let pool = unsafe { d.create_command_pool(&info, None)? };
    let pool = r.keep(pool, Resource::Commands);
    let d = &r.gpu.device;
    let buffers = unsafe { d.allocate_command_buffers(&vk::CommandBufferAllocateInfo::default().command_pool(pool).command_buffer_count(N as u32))? };
    Ok(buffers.try_into().unwrap())
}

pub fn fence(r: &mut Resources<'_>) -> Result<vk::Fence> {
    let fence = unsafe { r.gpu.device.create_fence(&vk::FenceCreateInfo::default(), None)? };
    Ok(r.keep(fence, Resource::Fence))
}

/// The frame's GPU timestamps (world::STAMPS) and the GPU's nanoseconds per tick.
pub fn timestamps(r: &mut Resources<'_>) -> Result<(vk::QueryPool, f64)> {
    let info = vk::QueryPoolCreateInfo::default().query_type(vk::QueryType::TIMESTAMP).query_count(world::STAMPS);
    let queries = unsafe { r.gpu.device.create_query_pool(&info, None)? };
    let tick_ns = unsafe { r.gpu.instance.get_physical_device_properties(r.gpu.physical).limits.timestamp_period } as f64;
    Ok((r.keep(queries, Resource::Queries), tick_ns))
}
