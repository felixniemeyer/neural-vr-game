//! DESKTOP=1: Sunk City without OpenXR (and without Monado's emulated headset), in a window.
//! Plain Vulkan, two virtual eyes that feed the cube-v1 sphere store as in VR, and the store
//! itself as the picture: panorama.frag draws the whole sphere on the GPU every frame,
//! seen from its centre, as a domemaster (DOME=1: a square azimuthal-equidistant fisheye)
//! or an equirectangular panorama, into an offscreen image that is blitted (letterboxed)
//! into the window's swapchain. Both follow the camera; in the dome, the camera's forward
//! sits the attention fader's angle from the zenith, towards the bottom edge (as
//! fabric-artwork's camera: 0 = the camera looks at the zenith, 90 = at the bottom rim).
//!
//! Env (besides the app's WEIGHTS, CUBE_SIZE, FADERS, CONTROLS, EYE_SIZE):
//!   DOME=1              domemaster (else equirectangular); Tab in the window switches
//!   DOME_FOV            the dome's angle across, degrees (default 180)
//!   PANORAMA_WIDTH      the picture's pixels across (default 1024; equirect height = width / 2)
//!   VIDEO=file.mp4      also the picture after every net update into a video via ffmpeg
//!                       (netfps frames per second; the simulation then runs on a fixed 90 Hz
//!                       clock, else on the wall clock)
//!   FRAMES=n            stop after n frames (default: until the window closes, Esc or Ctrl+C)
//!   FOV                 the virtual eyes' field of view, degrees (default 90)
//! Controls (av-controls, tab "camera"): the camera, a Player3D flown from the controller
//! (WASD, mouse, Q/E roll) on top of the autopilot (tab "motion": auto, off by default, speed; with auto off
//! the camera alone moves), and attention. The eyes look where the camera does.

use crate::gpu::{Gpu, Resource, Resources};
use crate::world::{self, Settings, World, EYES, PANORAMA_PUSH};
use crate::{camera, city, panel, per_second, rule, swarm, NetClock, Zip, START_HEIGHT};
use anyhow::{bail, ensure, Context, Result};
use ash::vk;
use glam::{Mat3, Quat, Vec3};
use openxr as xr;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle, RawDisplayHandle};
use std::{
    env,
    io::Write,
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};
use winit::{
    application::ApplicationHandler,
    event::{ElementState, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{Key, NamedKey},
    platform::pump_events::{EventLoopExtPumpEvents, PumpStatus},
    window::{Window, WindowId},
};

const DT: f32 = 1. / 90.;
const IPD: f32 = 0.064;
const PICTURE: vk::Format = vk::Format::R8G8B8A8_UNORM;

/// Vulkan without OpenXR, able to present to `display`'s windows: the first discrete GPU,
/// else any.
fn plain_gpu(display: RawDisplayHandle) -> Result<Gpu> {
    unsafe {
        let entry = ash::Entry::load().context("Loading Vulkan")?;
        let application = vk::ApplicationInfo::default()
            .application_name(c"Sunk City")
            .api_version(vk::API_VERSION_1_1);
        let extensions = ash_window::enumerate_required_extensions(display)?;
        let info = vk::InstanceCreateInfo::default().application_info(&application).enabled_extension_names(extensions);
        let instance = entry.create_instance(&info, None)?;
        let physical = instance
            .enumerate_physical_devices()?
            .into_iter()
            .max_by_key(|&p| (instance.get_physical_device_properties(p).device_type == vk::PhysicalDeviceType::DISCRETE_GPU) as u8)
            .context("No Vulkan device")?;
        let swapchain = [ash::khr::swapchain::NAME];
        match crate::make_device(&instance, physical, &swapchain, |info| Ok(instance.create_device(physical, info, None)?)) {
            Ok((device, queue, family, stats)) => Ok(Gpu { stats, _entry: entry, instance, physical, device, queue, family }),
            Err(error) => {
                instance.destroy_instance(None);
                Err(error)
            }
        }
    }
}

/// The window's surface and swapchain (recreated on resize), destroyed before the device.
struct Screen {
    device: ash::Device,
    surfaces: ash::khr::surface::Instance,
    swapchains: ash::khr::swapchain::Device,
    surface: vk::SurfaceKHR,
    swapchain: vk::SwapchainKHR,
    images: Vec<vk::Image>,
    extent: vk::Extent2D,
    acquired: vk::Semaphore,
    /// Per swapchain image: its commands finished (present waits on it).
    rendered: Vec<vk::Semaphore>,
}

impl Screen {
    fn new(gpu: &Gpu, display: RawDisplayHandle, window: &Window) -> Result<Self> {
        let surfaces = ash::khr::surface::Instance::new(&gpu._entry, &gpu.instance);
        let surface = unsafe {
            ash_window::create_surface(&gpu._entry, &gpu.instance, display, window.window_handle()?.as_raw(), None)?
        };
        ensure!(
            unsafe { surfaces.get_physical_device_surface_support(gpu.physical, gpu.family, surface)? },
            "the GPU's queue cannot present to the window"
        );
        let acquired = unsafe { gpu.device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None)? };
        let mut screen = Self {
            device: gpu.device.clone(),
            surfaces,
            swapchains: ash::khr::swapchain::Device::new(&gpu.instance, &gpu.device),
            surface,
            swapchain: vk::SwapchainKHR::null(),
            images: Vec::new(),
            extent: vk::Extent2D::default(),
            acquired,
            rendered: Vec::new(),
        };
        let size = window.inner_size();
        screen.rebuild(gpu.physical, vk::Extent2D { width: size.width, height: size.height })?;
        Ok(screen)
    }

    /// A swapchain for the window's current size (the device must be idle); a zero
    /// extent (minimised) keeps none.
    fn rebuild(&mut self, physical: vk::PhysicalDevice, size: vk::Extent2D) -> Result<()> {
        unsafe {
            let caps = self.surfaces.get_physical_device_surface_capabilities(physical, self.surface)?;
            let extent = if caps.current_extent.width != u32::MAX {
                caps.current_extent
            } else {
                vk::Extent2D {
                    width: size.width.clamp(caps.min_image_extent.width, caps.max_image_extent.width),
                    height: size.height.clamp(caps.min_image_extent.height, caps.max_image_extent.height),
                }
            };
            for semaphore in self.rendered.drain(..) {
                self.device.destroy_semaphore(semaphore, None);
            }
            let old = std::mem::replace(&mut self.swapchain, vk::SwapchainKHR::null());
            self.images.clear();
            self.extent = extent;
            if extent.width == 0 || extent.height == 0 {
                if old != vk::SwapchainKHR::null() {
                    self.swapchains.destroy_swapchain(old, None);
                }
                return Ok(());
            }
            ensure!(caps.supported_usage_flags.contains(vk::ImageUsageFlags::TRANSFER_DST), "the window's images cannot be blitted to");
            let formats = self.surfaces.get_physical_device_surface_formats(physical, self.surface)?;
            // UNORM: the picture is already tone-mapped for display (as the video).
            let format = formats
                .iter()
                .find(|f| matches!(f.format, vk::Format::B8G8R8A8_UNORM | vk::Format::R8G8B8A8_UNORM))
                .or(formats.first())
                .copied()
                .context("the window has no surface formats")?;
            // Never wait for the compositor: FIFO blocks forever while the window is hidden
            // (Wayland), and the simulation runs as fast as the GPU anyway.
            let modes = self.surfaces.get_physical_device_surface_present_modes(physical, self.surface)?;
            let mode = [vk::PresentModeKHR::MAILBOX, vk::PresentModeKHR::IMMEDIATE]
                .into_iter()
                .find(|m| modes.contains(m))
                .unwrap_or(vk::PresentModeKHR::FIFO);
            let count = if caps.max_image_count > 0 { (caps.min_image_count + 1).min(caps.max_image_count) } else { caps.min_image_count + 1 };
            let info = vk::SwapchainCreateInfoKHR::default()
                .surface(self.surface)
                .min_image_count(count)
                .image_format(format.format)
                .image_color_space(format.color_space)
                .image_extent(extent)
                .image_array_layers(1)
                .image_usage(vk::ImageUsageFlags::TRANSFER_DST)
                .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
                .pre_transform(caps.current_transform)
                .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
                .present_mode(mode)
                .clipped(true)
                .old_swapchain(old);
            self.swapchain = self.swapchains.create_swapchain(&info, None)?;
            if old != vk::SwapchainKHR::null() {
                self.swapchains.destroy_swapchain(old, None);
            }
            self.images = self.swapchains.get_swapchain_images(self.swapchain)?;
            for _ in &self.images {
                self.rendered.push(self.device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None)?);
            }
        }
        Ok(())
    }
}

impl Drop for Screen {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            for &semaphore in self.rendered.iter().chain([&self.acquired]) {
                self.device.destroy_semaphore(semaphore, None);
            }
            if self.swapchain != vk::SwapchainKHR::null() {
                self.swapchains.destroy_swapchain(self.swapchain, None);
            }
            self.surfaces.destroy_surface(self.surface, None);
        }
    }
}

/// The window's events since the last pump.
#[derive(Default)]
struct Events {
    close: bool,
    resized: bool,
    /// Tab: dome <-> equirect.
    switch: bool,
}

impl ApplicationHandler for Events {
    fn resumed(&mut self, _: &ActiveEventLoop) {}

    fn window_event(&mut self, _: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => self.close = true,
            WindowEvent::Resized(_) | WindowEvent::ScaleFactorChanged { .. } => self.resized = true,
            WindowEvent::KeyboardInput { event, .. } if event.state == ElementState::Pressed && !event.repeat => match event.logical_key {
                Key::Named(NamedKey::Escape) => self.close = true,
                Key::Named(NamedKey::Tab) => self.switch = true,
                _ => {}
            },
            _ => {}
        }
    }
}

fn video(path: &Path, width: u32, height: u32, fps: f32) -> Result<Child> {
    Command::new("ffmpeg")
        .args(["-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", &format!("{width}x{height}")])
        .args(["-r", &format!("{fps}"), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18"])
        .arg(path)
        .stdin(Stdio::piped())
        .spawn()
        .context("Starting ffmpeg (VIDEO)")
}

fn barrier(d: &ash::Device, cmd: vk::CommandBuffer, image: vk::Image, (from, to): (vk::ImageLayout, vk::ImageLayout), (src, dst): (vk::AccessFlags, vk::AccessFlags), (after, before): (vk::PipelineStageFlags, vk::PipelineStageFlags)) {
    let range = vk::ImageSubresourceRange::default().aspect_mask(vk::ImageAspectFlags::COLOR).level_count(1).layer_count(1);
    let barrier = vk::ImageMemoryBarrier::default()
        .image(image)
        .old_layout(from)
        .new_layout(to)
        .src_access_mask(src)
        .dst_access_mask(dst)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .subresource_range(range);
    unsafe { d.cmd_pipeline_barrier(cmd, after, before, vk::DependencyFlags::empty(), &[], &[], &[barrier]) };
}

pub fn run(shaders: &Path) -> Result<()> {
    let mut event_loop = EventLoop::new().context("Opening the window system (Wayland/X11)")?;
    let display = event_loop.display_handle()?.as_raw();
    let gpu = plain_gpu(display)?;
    let d = &gpu.device;
    let mut resources = Resources::new(&gpu);
    let rule = rule::Rule::load(env::var_os("WEIGHTS").map(PathBuf::from).as_deref())?;
    if rule.layout != 3 {
        bail!("DESKTOP renders the cube-v1 sphere store: WEIGHTS must be a cube-v1 net (this is {})", rule.layout_name());
    }
    println!("Net: {} {} ({} weights, {} mip levels, {} reprojection)", if rule.trained() { "trained" } else { "random" }, rule.layout_name(), rule.len(), rule.levels(), if rule.splat.is_some() { "splat" } else { "pull" });
    let edge: u32 = env::var("EYE_SIZE").ok().and_then(|v| v.parse().ok()).unwrap_or(256);
    ensure!(edge % (1 << (rule.levels() - 1)) == 0, "EYE_SIZE must be divisible by 2^(levels - 1)");
    let size = vk::Extent2D { width: edge, height: edge };
    let fov = env::var("FOV").ok().and_then(|v| v.parse::<f32>().ok()).unwrap_or(90.).to_radians();
    let format = vk::Format::R8G8B8A8_UNORM;
    let usage = vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_SRC;
    let eyes: Vec<Vec<vk::Image>> = (0..EYES).map(|_| Ok(vec![resources.image(size, format, usage)?.0])).collect::<Result<_>>()?;
    // A texel = an eye pixel at the view centre (as main.rs from the located views).
    let step = 1u32 << (rule.levels() - 1);
    let pixel = 2. * (fov / 2.).tan() / edge as f32;
    let face = env::var("CUBE_SIZE").ok().and_then(|v| v.parse::<u32>().ok())
        .unwrap_or(((std::f32::consts::FRAC_PI_2 / pixel / step as f32).round() as u32).max(1) * step);
    let mut world = World::new(&mut resources, shaders, size, format, [&eyes[0], &eyes[1]], rule.specialization(), rule.len(), face)?;
    world.splat_nearness = rule.splat;
    println!("Cube: {face} texels per face ({:.2} arcmin); eyes {edge} px, {:.0} deg", (std::f32::consts::FRAC_PI_2 / face as f32).to_degrees() * 60., fov.to_degrees());

    // The picture: room for the dome's square; the equirect uses its top half.
    let width: u32 = env::var("PANORAMA_WIDTH").ok().and_then(|v| v.parse().ok()).unwrap_or(1024);
    let picture_size = vk::Extent2D { width, height: width };
    let picture_usage = vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC;
    let (picture, picture_view) = resources.image(picture_size, PICTURE, picture_usage)?;
    let picture_pass = world.enable_panorama(&mut resources, shaders, PICTURE)?;
    let picture_target = resources.framebuffer(picture_pass, picture_size, &[picture_view])?;
    let degrees = |name: &str, default: f32| env::var(name).ok().and_then(|v| v.parse::<f32>().ok()).unwrap_or(default).to_radians();
    let dome_angle = degrees("DOME_FOV", 180.);
    let mut dome = env::var("DOME").is_ok_and(|v| v != "0");

    let pool = resources.keep(
        unsafe { d.create_command_pool(&vk::CommandPoolCreateInfo::default().queue_family_index(gpu.family).flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER), None)? },
        Resource::Commands,
    );
    // Two command buffers: the world and its picture (never waits for the window), then
    // the picture blitted into a window image (waits for that image only).
    let [cmd, show]: [vk::CommandBuffer; 2] = unsafe { d.allocate_command_buffers(&vk::CommandBufferAllocateInfo::default().command_pool(pool).command_buffer_count(2))? }
        .try_into()
        .unwrap();
    let fence = resources.keep(unsafe { d.create_fence(&vk::FenceCreateInfo::default(), None)? }, Resource::Fence);
    let shown = resources.keep(unsafe { d.create_fence(&vk::FenceCreateInfo::default(), None)? }, Resource::Fence);
    let queries = resources.keep(
        unsafe { d.create_query_pool(&vk::QueryPoolCreateInfo::default().query_type(vk::QueryType::TIMESTAMP).query_count(world::STAMPS), None)? },
        Resource::Queries,
    );

    let quit = Arc::new(AtomicBool::new(false));
    {
        let quit = quit.clone();
        ctrlc::set_handler(move || quit.store(true, Ordering::SeqCst))?;
    }
    let controls = panel(&rule, true);
    let mut ffmpeg = match env::var_os("VIDEO") {
        Some(path) => Some(video(Path::new(&path), width, if dome { width } else { width / 2 }, controls.get("netfps").round().max(1.))?),
        None => None,
    };
    let (video_buffer, video_pixels) = resources.buffer(&vec![0u8; 4 * (width * width) as usize], vk::BufferUsageFlags::TRANSFER_DST)?;

    #[allow(deprecated)] // winit 0.30: windows from the pumped loop, no ApplicationHandler::resumed dance
    let window = event_loop.create_window(
        Window::default_attributes()
            .with_title("Sunk City")
            .with_inner_size(winit::dpi::LogicalSize::new(width, if dome { width } else { width / 2 })),
    )?;
    let mut screen = Screen::new(&gpu, display, &window)?;
    let mut events = Events::default();

    let limit: Option<u64> = env::var("FRAMES").ok().and_then(|v| v.parse().ok());
    let mut zip = Zip { position: Vec3::new(0., START_HEIGHT, 0.), ..Zip::default() };
    let mut clock = NetClock::default();
    let mut swarm = swarm::Swarm::new()?;
    let (mut weights, mut packed) = (vec![0f32; rule.len()], vec![0u16; rule.len()]);
    let (mut frames, mut updates, mut since) = (0u64, 0u64, Instant::now());
    let mut gpu_ms = [0f64; world::STAMPS as usize - 1]; // per stage (world.rs stamps)
    let tick_ns = unsafe { gpu.instance.get_physical_device_properties(gpu.physical).limits.timestamp_period } as f64;
    let start = Instant::now();
    let mut last = 0f32;
    println!("Desktop: a window ({}; Tab switches, Esc closes){}", if dome { "domemaster" } else { "equirect" }, env::var("VIDEO").map(|v| format!(", video to {v}")).unwrap_or_default());

    while !quit.load(Ordering::SeqCst) && limit.is_none_or(|n| frames < n) {
        let frame_start = Instant::now();
        if let PumpStatus::Exit(_) = event_loop.pump_app_events(Some(Duration::ZERO), &mut events) {
            break;
        }
        if std::mem::take(&mut events.close) {
            break;
        }
        if std::mem::take(&mut events.switch) && ffmpeg.is_none() {
            dome = !dome; // not while recording: the video's size is fixed
        }
        if std::mem::take(&mut events.resized) {
            unsafe { d.device_wait_idle()? };
            let size = window.inner_size();
            screen.rebuild(gpu.physical, vk::Extent2D { width: size.width, height: size.height })?;
        }
        if screen.images.is_empty() {
            std::thread::sleep(Duration::from_millis(50)); // minimised
            events.resized = true;
            continue;
        }

        // Time: the wall clock live, a fixed 90 Hz clock for a video (whatever the GPU's speed).
        let seconds = if ffmpeg.is_some() { frames as f32 * DT } else { start.elapsed().as_secs_f32() };
        let dt = (seconds - std::mem::replace(&mut last, seconds)).clamp(0., 0.1);
        let city = city::City {
            cell: controls.get("cell"),
            height: controls.get("height"),
            blocks: controls.get("blocks").round().max(1.) as u32,
            player: glam::Vec2::new(zip.position.x, zip.position.z),
        };
        if controls.on("auto") {
            zip.fly(dt, Vec3::ZERO, &city, controls.get("speed"), controls.get("body"));
        }
        // The camera (controller) on top of the autopilot: its position in the autopilot's
        // heading, its orientation as a head's.
        let pose = controls.pose("camera");
        let [x, y, z, w] = pose.rotation;
        let look = Quat::from_xyzw(x, y, z, w);
        let look = if look.length_squared() > 1e-6 { look.normalize() } else { Quat::IDENTITY };
        let turn = zip.turn();
        let origin = zip.position + turn * Vec3::from(pose.position);
        let half = fov / 2.;
        let views: [xr::View; EYES] = std::array::from_fn(|i| xr::View {
            pose: xr::Posef {
                orientation: xr::Quaternionf { x: look.x, y: look.y, z: look.z, w: look.w },
                position: {
                    let eye = look * Vec3::new((i as f32 - 0.5) * IPD, 0., 0.);
                    xr::Vector3f { x: eye.x, y: eye.y, z: eye.z }
                },
            },
            fov: xr::Fovf { angle_left: -half, angle_right: half, angle_up: half, angle_down: -half },
        });
        let cameras = [camera(&views[0], Vec3::ZERO, origin, turn), camera(&views[1], Vec3::ZERO, origin, turn)];
        let settings = Settings {
            wild: controls.get("wild"),
            gate: controls.get("gate"),
            tau: clock.tick(seconds, controls.get("timescale"), controls.get("netfps"), &rule, false),
            color: per_second(controls.get("color")),
            depth: per_second(controls.get("depth")),
            sky: controls.get("sky"),
            focus: controls.get("focus"),
            day: (controls.get("day") > 0.).then(|| (seconds / controls.get("day") + 0.25).fract()),
            gaze: None,
            blocks: city.blocks,
            cell: city.cell,
            height: city.height,
            crystals: controls.get("crystals"),
            city: controls.on("city"),
            dot: controls.get("dot"),
        };
        if controls.on("creatures") {
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
            swarm.update(params, seconds, dt, zip.position);
            world.upload_creatures(&swarm.out);
        } else {
            world.upload_creatures(&[]);
        }

        // The picture: the camera's axes in the world, the dome's attention.
        let axes = Mat3::from_quat(turn * look);
        let shape = if dome { vk::Extent2D { width, height: width } } else { vk::Extent2D { width, height: width / 2 } };
        let mut push = [0f32; PANORAMA_PUSH];
        push[0..3].copy_from_slice(&axes.x_axis.to_array());
        push[4..7].copy_from_slice(&axes.y_axis.to_array());
        push[8..11].copy_from_slice(&axes.z_axis.to_array());
        push[12..16].copy_from_slice(&[controls.get("attention").to_radians(), dome_angle, dome as u8 as f32, 0.]);
        push[16..20].copy_from_slice(&[shape.width as f32, shape.height as f32, 0., width as f32]);
        // Letterboxed into the window.
        let fit = (screen.extent.width as f32 / shape.width as f32).min(screen.extent.height as f32 / shape.height as f32);
        let (w, h) = ((shape.width as f32 * fit) as i32, (shape.height as f32 * fit) as i32);
        let (x0, y0) = ((screen.extent.width as i32 - w) / 2, (screen.extent.height as i32 - h) / 2);
        let record_video = ffmpeg.is_some() && settings.tau > 0.;

        unsafe {
            use vk::{AccessFlags as A, ImageLayout as L, PipelineStageFlags as S};
            d.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())?;
            d.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default())?;
            d.cmd_reset_query_pool(cmd, queries, 0, world::STAMPS);
            d.cmd_write_timestamp(cmd, S::TOP_OF_PIPE, queries, 0);
            let frame_rule = rule.frame();
            rule.weights(seconds, settings.wild, &mut weights);
            rule::pack(&rule.layers, &weights, &mut packed);
            world.capture = frames % 8 == 0;
            world.draw(d, cmd, queries, cameras, [0, 0], seconds, &frame_rule, &packed, settings);
            world.draw_panorama(d, cmd, picture_target, shape, push);
            let read = L::TRANSFER_SRC_OPTIMAL;
            barrier(d, cmd, picture, (L::COLOR_ATTACHMENT_OPTIMAL, read), (A::COLOR_ATTACHMENT_WRITE, A::TRANSFER_READ), (S::COLOR_ATTACHMENT_OUTPUT, S::TRANSFER));
            let layers = vk::ImageSubresourceLayers::default().aspect_mask(vk::ImageAspectFlags::COLOR).layer_count(1);
            if record_video {
                let copy = vk::BufferImageCopy::default()
                    .image_subresource(layers)
                    .image_extent(vk::Extent3D { width: shape.width, height: shape.height, depth: 1 });
                d.cmd_copy_image_to_buffer(cmd, picture, read, video_buffer, &[copy]);
            }
            d.end_command_buffer(cmd)?;
            d.reset_fences(&[fence])?;
            let commands = [cmd];
            d.queue_submit(gpu.queue, &[vk::SubmitInfo::default().command_buffers(&commands)], fence)?;
            d.wait_for_fences(&[fence], true, u64::MAX)?;

            // Into the window, if it has an image free now (hidden or minimised: skipped).
            match screen.swapchains.acquire_next_image(screen.swapchain, 0, screen.acquired, vk::Fence::null()) {
                Ok((image, _)) => {
                    let target = screen.images[image as usize];
                    d.reset_command_buffer(show, vk::CommandBufferResetFlags::empty())?;
                    d.begin_command_buffer(show, &vk::CommandBufferBeginInfo::default())?;
                    let write = L::TRANSFER_DST_OPTIMAL;
                    barrier(d, show, target, (L::UNDEFINED, write), (A::empty(), A::TRANSFER_WRITE), (S::TRANSFER, S::TRANSFER));
                    let range = vk::ImageSubresourceRange::default().aspect_mask(vk::ImageAspectFlags::COLOR).level_count(1).layer_count(1);
                    d.cmd_clear_color_image(show, target, write, &vk::ClearColorValue { float32: [0., 0., 0., 1.] }, &[range]);
                    barrier(d, show, target, (write, write), (A::TRANSFER_WRITE, A::TRANSFER_WRITE), (S::TRANSFER, S::TRANSFER));
                    let blit = vk::ImageBlit::default()
                        .src_subresource(layers)
                        .src_offsets([vk::Offset3D::default(), vk::Offset3D { x: shape.width as i32, y: shape.height as i32, z: 1 }])
                        .dst_subresource(layers)
                        .dst_offsets([vk::Offset3D { x: x0, y: y0, z: 0 }, vk::Offset3D { x: x0 + w, y: y0 + h, z: 1 }]);
                    d.cmd_blit_image(show, picture, read, target, write, &[blit], vk::Filter::LINEAR);
                    barrier(d, show, target, (write, L::PRESENT_SRC_KHR), (A::TRANSFER_WRITE, A::empty()), (S::TRANSFER, S::BOTTOM_OF_PIPE));
                    d.end_command_buffer(show)?;
                    d.reset_fences(&[shown])?;
                    let waits = [screen.acquired];
                    let stages = [S::TRANSFER];
                    let signals = [screen.rendered[image as usize]];
                    let commands = [show];
                    let submit = vk::SubmitInfo::default().wait_semaphores(&waits).wait_dst_stage_mask(&stages).command_buffers(&commands).signal_semaphores(&signals);
                    d.queue_submit(gpu.queue, &[submit], shown)?;
                    let swapchains = [screen.swapchain];
                    let indices = [image];
                    let present = vk::PresentInfoKHR::default().wait_semaphores(&signals).swapchains(&swapchains).image_indices(&indices);
                    match screen.swapchains.queue_present(gpu.queue, &present) {
                        Ok(false) => {}
                        Ok(true) | Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => events.resized = true,
                        Err(error) => return Err(error.into()),
                    }
                    // The next picture overwrites what this blit reads.
                    d.wait_for_fences(&[shown], true, u64::MAX)?;
                }
                Err(vk::Result::TIMEOUT | vk::Result::NOT_READY) => {}
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => events.resized = true,
                Err(error) => return Err(error.into()),
            }
            let mut stamps = [0u64; world::STAMPS as usize];
            if d.get_query_pool_results(queries, 0, &mut stamps, vk::QueryResultFlags::TYPE_64).is_ok() {
                for (i, ms) in gpu_ms.iter_mut().enumerate() {
                    *ms += stamps[i + 1].wrapping_sub(stamps[i]) as f64 * tick_ns / 1e6;
                }
            }
        }
        if world.capture {
            if controls.on("exposure") {
                world.calibrate(controls.get("mid"), controls.get("clip"), controls.get("black"), controls.get("tint"));
            } else {
                world.expose_plainly();
            }
        }
        if settings.tau > 0. {
            updates += 1;
        }
        if let (true, Some(child)) = (record_video, &mut ffmpeg) {
            let bytes = unsafe { std::slice::from_raw_parts(video_pixels, 4 * (shape.width * shape.height) as usize) };
            child.stdin.as_mut().context("ffmpeg stdin")?.write_all(bytes)?;
        }
        // Live: paced to 90 Hz (as the headset), not as fast as the GPU goes (the whole
        // system would lag); a video renders as fast as it can on its fixed clock.
        if ffmpeg.is_none() {
            std::thread::sleep(Duration::from_secs_f32(DT).saturating_sub(frame_start.elapsed()));
        }
        frames += 1;
        if frames % 90 == 0 {
            let rate = 90. / std::mem::replace(&mut since, Instant::now()).elapsed().as_secs_f64();
            let forward = turn * look * Vec3::NEG_Z;
            let [scene, forward_ms, pull, net, display] = gpu_ms.map(|ms| ms / 90.);
            let total = scene + forward_ms + pull + net + display;
            println!(
                "{frames} frames ({seconds:.1} s, {rate:.1} frames/s, {updates} updates): GPU {total:.2} ms/frame (scene {scene:.2} forward {forward_ms:.2} pull {pull:.2} net {net:.2} display {display:.2}), at ({:.1}, {:.1}, {:.1}) looking ({:.2}, {:.2}, {:.2})",
                origin.x, origin.y, origin.z, forward.x, forward.y, forward.z
            );
            window.set_title(&format!("Sunk City: {} {rate:.0} fps, GPU {total:.1} ms", if dome { "dome" } else { "equirect" }));
            gpu_ms = [0.; world::STAMPS as usize - 1];
        }
    }
    unsafe { d.device_wait_idle()? };
    drop(screen);
    if let Some(mut child) = ffmpeg {
        drop(child.stdin.take());
        child.wait()?;
    }
    Ok(())
}
