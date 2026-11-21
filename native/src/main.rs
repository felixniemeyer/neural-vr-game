mod gpu;
mod rule;
mod world;

use anyhow::{ensure, Context, Result};
use ash::vk::{self, Handle};
use glam::{Mat4, Quat, Vec3, Vec4};
use gpu::{Gpu, Resource, Resources};
use openxr as xr;
use std::{
    env,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    ffi::CStr,
    path::{Path, PathBuf},
    thread,
    time::Duration,
};
use world::{EyeCamera, Settings, World, EYES};

const STEREO: xr::ViewConfigurationType = xr::ViewConfigurationType::PRIMARY_STEREO;
const BLEND: xr::EnvironmentBlendMode = xr::EnvironmentBlendMode::OPAQUE;

fn init_gpu(instance: &xr::Instance, system: xr::SystemId) -> Result<Gpu> {
    let requirements = instance.graphics_requirements::<xr::Vulkan>(system)?;
    let version = xr::Version::new(1, 1, 0);
    ensure!(
        version >= requirements.min_api_version_supported
            && version <= requirements.max_api_version_supported,
        "OpenXR runtime must support Vulkan 1.1"
    );
    // OpenXR creates both Vulkan handles so the runtime can supply its required extensions.
    unsafe {
        let entry = ash::Entry::load().context("Loading Vulkan")?;
        let get_proc: xr::sys::platform::VkGetInstanceProcAddr =
            std::mem::transmute(entry.static_fn().get_instance_proc_addr);
        let application = vk::ApplicationInfo::default()
            .application_name(c"Sunk City")
            .api_version(vk::API_VERSION_1_1);
        let info = vk::InstanceCreateInfo::default().application_info(&application);
        let raw = instance
            .create_vulkan_instance(
                system,
                get_proc,
                (&info as *const vk::InstanceCreateInfo).cast(),
            )?
            .map_err(vk::Result::from_raw)?;
        let vulkan = ash::Instance::load(entry.static_fn(), vk::Instance::from_raw(raw as u64));
        let setup = (|| -> Result<_> {
            let physical = instance.vulkan_graphics_device(system, raw)?;
            let physical_handle = vk::PhysicalDevice::from_raw(physical as u64);
            let family = vulkan
                .get_physical_device_queue_family_properties(physical_handle)
                .iter()
                .position(|p| {
                    p.queue_count > 0
                        && p.queue_flags
                            .contains(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE)
                })
                .context("No Vulkan graphics and compute queue")? as u32;
            let queues = [vk::DeviceQueueCreateInfo::default()
                .queue_family_index(family)
                .queue_priorities(&[1.])];
            let info = vk::DeviceCreateInfo::default().queue_create_infos(&queues);
            let raw = instance
                .create_vulkan_device(
                    system,
                    get_proc,
                    physical,
                    (&info as *const vk::DeviceCreateInfo).cast(),
                )?
                .map_err(vk::Result::from_raw)?;
            let device = ash::Device::load(vulkan.fp_v1_0(), vk::Device::from_raw(raw as u64));
            let queue = device.get_device_queue(family, 0);
            let properties = vulkan.get_physical_device_properties(physical_handle);
            println!(
                "Vulkan GPU: {}",
                CStr::from_ptr(properties.device_name.as_ptr()).to_string_lossy()
            );
            Ok((physical_handle, device, queue, family))
        })();
        match setup {
            Ok((physical, device, queue, family)) => Ok(Gpu {
                _entry: entry,
                instance: vulkan,
                physical,
                device,
                queue,
                family,
            }),
            Err(error) => {
                vulkan.destroy_instance(None);
                Err(error)
            }
        }
    }
}

/// Newest modification time of the compiled shaders (for live reload).
fn shaders_stamp(dir: &Path) -> Option<std::time::SystemTime> {
    std::fs::read_dir(dir)
        .ok()?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().is_some_and(|x| x == "spv"))
        .filter_map(|e| e.metadata().ok()?.modified().ok())
        .max()
}

fn vector(v: xr::Vector3f) -> Vec3 {
    Vec3::new(v.x, v.y, v.z)
}
fn quaternion(q: xr::Quaternionf) -> Quat {
    Quat::from_xyzw(q.x, q.y, q.z, q.w)
}

/// Widen a field of view by `factor` in tangent space (overscan). The compositor's
/// late timewarp then has a margin instead of showing black edges, and the margin
/// is already filled by the CA before it becomes visible.
fn widen(fov: xr::Fovf, factor: f32) -> xr::Fovf {
    let w = |angle: f32| (angle.tan() * factor).atan();
    xr::Fovf {
        angle_left: w(fov.angle_left),
        angle_right: w(fov.angle_right),
        angle_up: w(fov.angle_up),
        angle_down: w(fov.angle_down),
    }
}

/// Camera for one eye. `head` is subtracted from the tracked position: pass the head
/// centre for rotation-only viewing (only the stereo eye offset remains), or zero to
/// keep head translation. `origin` places the player in the world. OpenGL-style clip
/// space, y up.
fn camera(view: &xr::View, head: Vec3, origin: Vec3) -> EyeCamera {
    let (l, r, d, u) = (
        view.fov.angle_left.tan(),
        view.fov.angle_right.tan(),
        view.fov.angle_down.tan(),
        view.fov.angle_up.tan(),
    );
    let (near, far) = (0.05, 2000.);
    let projection = Mat4::from_cols(
        Vec4::new(2. / (r - l), 0., 0., 0.),
        Vec4::new(0., 2. / (u - d), 0., 0.),
        Vec4::new(
            (r + l) / (r - l),
            (u + d) / (u - d),
            -(far + near) / (far - near),
            -1.,
        ),
        Vec4::new(0., 0., -2. * far * near / (far - near), 0.),
    );
    let position = vector(view.pose.position) - head + origin;
    let pose = Mat4::from_rotation_translation(quaternion(view.pose.orientation), position);
    EyeCamera {
        view_proj: projection * pose.inverse(),
        position,
    }
}

fn run(shaders: &Path) -> Result<()> {
    // OPENXR_LOADER overrides; on the Steam Frame SteamVR ships its own loader.
    let frame_loader = Path::new("/opt/steamvr/bin/linuxarm64/libopenxr_loader.so");
    let loader = env::var_os("OPENXR_LOADER")
        .map(PathBuf::from)
        .or_else(|| frame_loader.is_file().then(|| frame_loader.to_path_buf()));
    let entry = unsafe {
        match loader {
            Some(path) => xr::Entry::load_from(&path),
            None => xr::Entry::load(),
        }
    }
    .context("Loading OpenXR")?;
    let mut extensions = xr::ExtensionSet::default();
    extensions.khr_vulkan_enable2 = true;
    let instance = entry.create_instance(
        &xr::ApplicationInfo {
            application_name: "Sunk City",
            application_version: 0,
            engine_name: "Native Vulkan",
            engine_version: 0,
            api_version: xr::Version::new(1, 0, 0),
        },
        &extensions,
        &[],
    )?;
    let system = instance.system(xr::FormFactor::HEAD_MOUNTED_DISPLAY)?;
    let gpu = init_gpu(&instance, system)?;
    let d = &gpu.device;
    let (session, mut waiter, mut stream) = unsafe {
        instance.create_session::<xr::Vulkan>(
            system,
            &xr::vulkan::SessionCreateInfo {
                instance: gpu.instance.handle().as_raw() as _,
                physical_device: gpu.physical.as_raw() as _,
                device: d.handle().as_raw() as _,
                queue_family_index: gpu.family,
                queue_index: 0,
            },
        )?
    };
    let space =
        session.create_reference_space(xr::ReferenceSpaceType::LOCAL, xr::Posef::IDENTITY)?;

    let formats = session.enumerate_swapchain_formats()?;
    let format = [
        vk::Format::R8G8B8A8_UNORM,
        vk::Format::B8G8R8A8_UNORM,
        vk::Format::R8G8B8A8_SRGB,
        vk::Format::B8G8R8A8_SRGB,
    ]
    .into_iter()
    .find(|f| formats.contains(&(f.as_raw() as u32)))
    .context("No RGB OpenXR swapchain format")?;
    let views = instance.enumerate_view_configuration_views(system, STEREO)?;
    ensure!(views.len() == EYES, "Expected two stereo views");
    // Square per-eye resolution; EYE_SIZE (default 256) is clamped to the runtime maximum.
    let edge = env::var("EYE_SIZE")
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
        .unwrap_or(256)
        .min(views[0].max_image_rect_width)
        .min(views[0].max_image_rect_height);
    let size = vk::Extent2D {
        width: edge,
        height: edge,
    };
    let mut swapchains = Vec::with_capacity(EYES);
    for _ in 0..EYES {
        swapchains.push(session.create_swapchain(&xr::SwapchainCreateInfo {
            create_flags: xr::SwapchainCreateFlags::EMPTY,
            usage_flags: xr::SwapchainUsageFlags::COLOR_ATTACHMENT
                | xr::SwapchainUsageFlags::SAMPLED,
            format: format.as_raw() as u32,
            sample_count: 1,
            width: edge,
            height: edge,
            face_count: 1,
            array_size: 1,
            mip_count: 1,
        })?);
    }
    let images: Vec<Vec<vk::Image>> = swapchains
        .iter()
        .map(|s| Ok(s.enumerate_images()?.into_iter().map(vk::Image::from_raw).collect()))
        .collect::<Result<_>>()?;

    // These resources are dropped before the OpenXR images and session.
    let mut resources = Resources::new(&gpu);
    // HIDDEN=3 (one hidden image, default) or 7 (two hidden images).
    let hidden_images = if env::var("HIDDEN").is_ok_and(|v| v == "7") { 2 } else { 1 };
    let mut world = World::new(&mut resources, shaders, size, format, [&images[0], &images[1]], hidden_images)?;
    world.mode = match env::var("VIEW").as_deref() {
        Ok("certainty") => 1,
        Ok("depth") => 2,
        Ok("velocity") => 3,
        Ok("hidden") => 4,
        Ok("hidden2") => 5,
        Ok("scene") => 6,
        _ => 0,
    };
    let pool_info = vk::CommandPoolCreateInfo::default()
        .queue_family_index(gpu.family)
        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
    let pool = resources.keep(
        unsafe { d.create_command_pool(&pool_info, None)? },
        Resource::Commands,
    );
    let cmd = unsafe {
        d.allocate_command_buffers(
            &vk::CommandBufferAllocateInfo::default()
                .command_pool(pool)
                .command_buffer_count(1),
        )?
    }[0];
    let fence = resources.keep(
        unsafe { d.create_fence(&vk::FenceCreateInfo::default(), None)? },
        Resource::Fence,
    );
    let queries = resources.keep(
        unsafe {
            d.create_query_pool(
                &vk::QueryPoolCreateInfo::default()
                    .query_type(vk::QueryType::TIMESTAMP)
                    .query_count(2),
                None,
            )?
        },
        Resource::Queries,
    );
    let tick_ns = unsafe {
        gpu.instance
            .get_physical_device_properties(gpu.physical)
            .limits
            .timestamp_period
    } as f64;

    // Ctrl+C / SIGTERM: end the OpenXR session cleanly instead of dying mid-frame,
    // which leaves Monado waiting on a dead client.
    let quit = Arc::new(AtomicBool::new(false));
    {
        let quit = quit.clone();
        ctrlc::set_handler(move || quit.store(true, Ordering::SeqCst))?;
    }
    let mut exit_requested = false;
    let mut stamp = shaders_stamp(shaders);
    let mut events = xr::EventDataBuffer::new();
    let mut running = false;
    let mut frames = 0u64;
    let mut gpu_ms = 0.;
    let mut start = None;
    let rule = rule::Rule::random();
    // Live settings from the av-controls broker (CONTROLS, default wss://localhost:8080);
    // without a broker the defaults below stay.
    let controls = av_controls::Panel::new("sunk-city")
        .fader("color", 0., 1., 0.)
        .fader("depth", 0., 1., 0.1)
        .fader("identity", 0., 1., 0.1)
        .fader("noise", 0., 4., 1.)
        .fader("seeds", 0., 4., 1.)
        .fader_with("rings", 1., 4., 4., 0)
        .fader("taps", 0.1, 1., 1.)
        .fader_with("blocks", 4., 64., 24., 0)
        .fader_with("cell", 1., 10., 4., 1)
        .fader_with("height", 1., 25., 7., 1)
        .fader_with("radius", 3., 100., 20., 1)
        .fader_with("altitude", 0., 25., 7.5, 1)
        .fader_with("near", 1., 40., 10., 1)
        .fader_with("far", 10., 300., 80., 0)
        .fader("mid", 0., 1., 0.5)
        .fader_with("clip", 0., 0.2, 0.03, 3)
        .fader_with("black", 0., 0.3, 0.05, 3)
        .fader("tint", 0., 1., 0.3)
        .connect(&env::var("CONTROLS").unwrap_or_else(|_| "wss://localhost:8080".into()));
    // MOVE scales head translation: 0 rotation only (default), 1 real scale, more =
    // giant steps (e.g. to cross the city with Monado's slow qwerty WASD).
    let stride: f32 = env::var("MOVE").ok().and_then(|v| v.parse().ok()).unwrap_or(0.);
    let overscan = env::var("OVERSCAN")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .unwrap_or(1.15);
    // SNAPSHOT=file.ppm: write the left eye's colour every 300 frames (debugging).
    let snapshot = env::var_os("SNAPSHOT").map(PathBuf::from);
    println!("Sunk City session created ({edge}x{edge} per eye)");

    loop {
        if quit.load(Ordering::SeqCst) && !exit_requested {
            exit_requested = true;
            if running {
                session.request_exit()?;
            } else {
                return Ok(());
            }
        }
        while let Some(event) = instance.poll_event(&mut events)? {
            match event {
                xr::Event::SessionStateChanged(state) => match state.state() {
                    xr::SessionState::READY => {
                        session.begin(STEREO)?;
                        running = true;
                    }
                    xr::SessionState::STOPPING => {
                        session.end()?;
                        running = false;
                    }
                    xr::SessionState::EXITING | xr::SessionState::LOSS_PENDING => return Ok(()),
                    _ => {}
                },
                xr::Event::InstanceLossPending(_) => return Ok(()),
                _ => {}
            }
        }
        if !running {
            thread::sleep(Duration::from_millis(15));
            continue;
        }
        // Live reload: recompiled shaders are picked up without restarting.
        if frames % 45 == 0 {
            let now = shaders_stamp(shaders);
            if now != stamp {
                stamp = now;
                thread::sleep(Duration::from_millis(200)); // let the compiler finish writing
                unsafe { d.device_wait_idle()? };
                match world.reload(&mut resources, shaders) {
                    Ok(()) => println!("Shaders reloaded"),
                    Err(error) => println!("Shader reload failed, keeping the old ones: {error:#}"),
                }
            }
        }
        let state = waiter.wait()?;
        stream.begin()?;
        let time = state.predicted_display_time;
        if !state.should_render {
            stream.end(time, BLEND, &[])?;
            continue;
        }
        let (flags, mut views) = session.locate_views(STEREO, time, &space)?;
        for view in &mut views {
            view.fov = widen(view.fov, overscan);
        }
        if views.len() != EYES || !flags.contains(xr::ViewStateFlags::ORIENTATION_VALID) {
            stream.end(time, BLEND, &[])?;
            continue;
        }
        let seconds = (time - *start.get_or_insert(time)).as_nanos() as f32 / 1e9;
        // Only the head centre is scaled; the stereo eye offset stays real.
        let head = (vector(views[0].pose.position) + vector(views[1].pose.position)) * 0.5 * (1. - stride);
        let origin = Vec3::new(0., controls.get("altitude"), 0.);
        let cameras = [camera(&views[0], head, origin), camera(&views[1], head, origin)];
        let settings = Settings {
            rings: controls.get("rings").round().clamp(1., 4.) as usize,
            taps: controls.get("taps"),
            color: controls.get("color"),
            identity: controls.get("identity"),
            depth: controls.get("depth"),
            noise: controls.get("noise"),
            seeds: controls.get("seeds"),
            blocks: controls.get("blocks").round().max(1.) as u32,
            cell: controls.get("cell"),
            height: controls.get("height"),
            radius: controls.get("radius"),
            near: controls.get("near"),
            far: controls.get("far"),
        };
        let mut indices = [0; EYES];
        for (swapchain, index) in swapchains.iter_mut().zip(&mut indices) {
            *index = swapchain.acquire_image()? as usize;
            swapchain.wait_image(xr::Duration::INFINITE)?;
        }
        unsafe {
            d.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())?;
            d.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default())?;
            d.cmd_reset_query_pool(cmd, queries, 0, 2);
            d.cmd_write_timestamp(cmd, vk::PipelineStageFlags::TOP_OF_PIPE, queries, 0);
            let frame_rule = rule.at(seconds, hidden_images, settings.rings);
            // Every 8th frame: read back the left eye for the automatic exposure.
            world.capture = frames % 8 == 0;
            world.draw(d, cmd, cameras, indices, seconds, &frame_rule, settings);
            d.cmd_write_timestamp(cmd, vk::PipelineStageFlags::BOTTOM_OF_PIPE, queries, 1);
            d.end_command_buffer(cmd)?;
            d.reset_fences(&[fence])?;
            let commands = [cmd];
            d.queue_submit(
                gpu.queue,
                &[vk::SubmitInfo::default().command_buffers(&commands)],
                fence,
            )?;
            d.wait_for_fences(&[fence], true, u64::MAX)?;
            let mut stamps = [0u64; 2];
            if d
                .get_query_pool_results(queries, 0, &mut stamps, vk::QueryResultFlags::TYPE_64)
                .is_ok()
            {
                gpu_ms += stamps[1].wrapping_sub(stamps[0]) as f64 * tick_ns / 1e6;
            }
        }
        if world.capture {
            world.calibrate(
                controls.get("mid"),
                controls.get("clip"),
                controls.get("black"),
                controls.get("tint"),
            );
            if let (true, Some(path)) = (frames % 296 == 0, &snapshot) {
                std::fs::write(path, world.snapshot_ppm())?;
            }
        }
        for swapchain in &mut swapchains {
            swapchain.release_image()?;
        }
        let layers: Vec<_> = (0..EYES)
            .map(|i| {
                xr::CompositionLayerProjectionView::new()
                    .pose(views[i].pose)
                    .fov(views[i].fov)
                    .sub_image(
                        xr::SwapchainSubImage::new()
                            .swapchain(&swapchains[i])
                            .image_rect(xr::Rect2Di {
                                offset: xr::Offset2Di { x: 0, y: 0 },
                                extent: xr::Extent2Di {
                                    width: world.size().width as i32,
                                    height: world.size().height as i32,
                                },
                            }),
                    )
            })
            .collect();
        stream.end(
            time,
            BLEND,
            &[&xr::CompositionLayerProjection::new()
                .space(&space)
                .views(&layers)],
        )?;
        frames += 1;
        if frames % 120 == 0 {
            let forward = quaternion(views[0].pose.orientation) * Vec3::NEG_Z;
            let [r, g, b] = world.tone().map(|[k, w, y]| format!("{k:.2}..{w:.2}^{y:.2}"));
            println!(
                "{frames} frames, GPU {:.3} ms/frame (both eyes), looking ({:.2}, {:.2}, {:.2}), exposure r {r} g {g} b {b}",
                gpu_ms / 120.,
                forward.x,
                forward.y,
                forward.z
            );
            gpu_ms = 0.;
        }
    }
}

fn main() -> Result<()> {
    let executable = env::current_exe()?;
    let shaders = env::args_os().nth(1).map(PathBuf::from).unwrap_or_else(|| {
        executable
            .parent()
            .expect("Executable has a parent directory")
            .to_path_buf()
    });
    run(&shaders)
}
