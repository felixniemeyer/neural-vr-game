//! DESKTOP=1: Sunk City without OpenXR (and without Monado's emulated headset). Plain
//! Vulkan, two virtual eyes on the autopilot (zip.rs's wander through the city, a fixed
//! 90 Hz clock: the result does not depend on how fast the GPU is), and the cube-v1 sphere
//! store itself as the output: after each net update its colour is read back and drawn as
//! an equirectangular panorama (the whole sphere, world-fixed: -z at the centre, y up), or
//! a domemaster (DOME=1: a square fisheye that follows the camera, as fabric-artwork's: the
//! camera's forward sits the attention fader's angle from the dome's zenith, towards the
//! image's bottom edge; 0 = the camera looks at the zenith, 90 = at the bottom rim).
//!
//! Env (besides the app's WEIGHTS, CUBE_SIZE, FADERS, CONTROLS, EYE_SIZE):
//!   PANORAMA=file.ppm   the latest panorama, rewritten every update (default
//!                       build/desktop/panorama.ppm; an image viewer that reloads shows it live)
//!   PANORAMA_WIDTH      pixels (default 1024; height = width / 2, the dome's = width)
//!   DOME=1              a domemaster instead (azimuthal equidistant)
//!   DOME_FOV            the dome's angle across, degrees (default 180)
//!   VIDEO=file.mp4      also every panorama into a video via ffmpeg (netfps frames per second)
//!   FRAMES=n            stop after n frames (default: until Ctrl+C)
//!   FOV                 the virtual eyes' field of view, degrees (default 90)
//! Controls (av-controls, tab "camera"): the camera, a Player3D flown from the controller
//! (WASD, mouse, Q/E roll) on top of the autopilot (tab "motion": auto, speed; with auto off
//! the camera alone moves), and attention. The eyes look where the camera does.

use crate::gpu::{Gpu, Resource, Resources};
use crate::world::{self, Settings, World, EYES};
use crate::{camera, city, panel, per_second, rule, swarm, NetClock, Zip, START_HEIGHT};
use anyhow::{bail, ensure, Context, Result};
use ash::vk;
use glam::{Quat, Vec3};
use openxr as xr;
use std::{
    env,
    io::Write,
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
};

const DT: f32 = 1. / 90.;
const IPD: f32 = 0.064;

/// Vulkan without OpenXR: the first discrete GPU, else any.
fn plain_gpu() -> Result<Gpu> {
    unsafe {
        let entry = ash::Entry::load().context("Loading Vulkan")?;
        let application = vk::ApplicationInfo::default()
            .application_name(c"Sunk City")
            .api_version(vk::API_VERSION_1_1);
        let instance = entry.create_instance(&vk::InstanceCreateInfo::default().application_info(&application), None)?;
        let physical = instance
            .enumerate_physical_devices()?
            .into_iter()
            .max_by_key(|&p| (instance.get_physical_device_properties(p).device_type == vk::PhysicalDeviceType::DISCRETE_GPU) as u8)
            .context("No Vulkan device")?;
        match crate::make_device(&instance, physical, |info| Ok(instance.create_device(physical, info, None)?)) {
            Ok((device, queue, family, stats)) => Ok(Gpu { stats, _entry: entry, instance, physical, device, queue, family }),
            Err(error) => {
                instance.destroy_instance(None);
                Err(error)
            }
        }
    }
}

/// The cube's faces (cube.glsl): forward and up per face, world coordinates.
const FORWARD: [Vec3; 6] = [Vec3::X, Vec3::NEG_X, Vec3::Y, Vec3::NEG_Y, Vec3::Z, Vec3::NEG_Z];
const UP: [Vec3; 6] = [Vec3::Y, Vec3::Y, Vec3::Z, Vec3::NEG_Z, Vec3::Y, Vec3::Y];

fn threads() -> usize {
    std::thread::available_parallelism().map_or(4, |n| n.get())
}

fn face_of(d: Vec3) -> usize {
    let a = d.abs();
    if a.x >= a.y && a.x >= a.z {
        if d.x > 0. { 0 } else { 1 }
    } else if a.y >= a.z {
        if d.y > 0. { 2 } else { 3 }
    } else if d.z > 0. {
        4
    } else {
        5
    }
}

/// The equirectangular panorama (rgb24) of the dumped state A (cube.glsl's atlas):
/// bilinear within each face, tone-mapped as cube_display.frag. Where each pixel reads
/// (4 atlas texels and the bilinear weights) is worked out once per face size.
struct Panorama {
    width: usize,
    /// DOME: the angle across, radians.
    dome: Option<f32>,
    /// The camera's orientation and the attention (radians) the dome's table was built for.
    view: (Quat, f32),
    layout: (u32, u32),
    /// Per pixel: atlas index of the top-left texel, the steps to the right and down
    /// neighbours (0 at a face's last texel), the fractions.
    table: Vec<(u32, u8, u32, f32, f32)>,
    state: Vec<[f32; 3]>,
}

impl Panorama {
    fn new(width: usize, dome: Option<f32>) -> Self {
        Self { width, dome, view: (Quat::IDENTITY, 0.), layout: (0, 0), table: Vec::new(), state: Vec::new() }
    }

    fn height(&self) -> usize {
        if self.dome.is_some() { self.width } else { self.width / 2 }
    }

    /// The world direction of a pixel; None outside the dome's circle.
    fn direction(&self, column: usize, row: usize) -> Option<Vec3> {
        let (width, height) = (self.width as f32, self.height() as f32);
        let (column, row) = (column as f32 + 0.5, row as f32 + 0.5);
        let Some(fov) = self.dome else {
            let lat = std::f32::consts::FRAC_PI_2 - row / height * std::f32::consts::PI;
            let lon = column / width * std::f32::consts::TAU - std::f32::consts::PI;
            return Some(Vec3::new(lon.sin() * lat.cos(), lat.sin(), -lon.cos() * lat.cos()));
        };
        // The dome's frame: x image right, y image up, z the zenith.
        let (u, v) = (2. * column / width - 1., 1. - 2. * row / height);
        let r = u.hypot(v);
        if r > 1. {
            return None;
        }
        let theta = r * fov / 2.;
        let (sin, cos) = theta.sin_cos();
        let dome = if r > 0. { Vec3::new(sin * u / r, sin * v / r, cos) } else { Vec3::Z };
        // The camera's forward (-z) is the attention's angle from the zenith towards the
        // bottom, its right is the image's right (camera.ts' playerToDomemaster).
        let (orientation, attention) = self.view;
        let (s, c) = attention.sin_cos();
        let forward = dome.dot(Vec3::new(0., -s, c));
        let up = dome.dot(Vec3::new(0., c, s));
        Some(orientation * Vec3::new(dome.x, up, -forward))
    }

    fn build(&mut self, (face, gutter): (u32, u32), aw: usize) {
        self.layout = (face, gutter);
        let n = face as f32;
        let angle = std::f32::consts::FRAC_PI_2 / n;
        let cell = (face + 2 * gutter) as usize;
        let (width, height) = (self.width, self.height());
        // Per pixel, on all cores in bands of rows (the dome rebuilds it as the camera turns).
        let mut table = std::mem::take(&mut self.table);
        table.resize(width * height, (0, 0, 0, 0., 0.));
        let band = height.div_ceil(threads()) * width;
        let this = &*self;
        std::thread::scope(|scope| {
            for (b, chunk) in table.chunks_mut(band).enumerate() {
                scope.spawn(move || {
                    for (i, entry) in chunk.iter_mut().enumerate() {
                        let at = b * band + i;
                        *entry = this.entry(at % width, at / width, (n, angle, cell, gutter as usize), aw);
                    }
                });
            }
        });
        self.table = table;
    }

    /// Where pixel (column, row) reads the atlas (see `table`).
    fn entry(&self, column: usize, row: usize, (n, angle, cell, gutter): (f32, f32, usize, usize), aw: usize) -> (u32, u8, u32, f32, f32) {
        let Some(d) = self.direction(column, row) else {
            return (u32::MAX, 0, 0, 0., 0.);
        };
        let f = face_of(d);
        let (forward, up) = (FORWARD[f], UP[f]);
        let right = forward.cross(up);
        let z = d.dot(forward);
        let x = ((d.dot(right).atan2(z) + std::f32::consts::FRAC_PI_4) / angle - 0.5).clamp(0., n - 1.);
        let y = ((-d.dot(up)).atan2(z) + std::f32::consts::FRAC_PI_4) / angle - 0.5;
        let y = y.clamp(0., n - 1.);
        let (x0, y0) = (x.floor(), y.floor());
        let base = (f % 3 * cell + gutter + x0 as usize, f / 3 * cell + gutter + y0 as usize);
        let right_step = u8::from(x0 < n - 1.);
        let down_step = if y0 < n - 1. { aw as u32 } else { 0 };
        ((base.1 * aw + base.0) as u32, right_step, down_step, x - x0, y - y0)
    }

    /// `view`: the camera's orientation and the attention (radians), for the dome.
    fn draw(&mut self, world: &World, view: (Quat, f32), out: &mut Vec<u8>) {
        let atlas = world.dump_extent();
        let (aw, ah) = (atlas.width as usize, atlas.height as usize);
        if world.cube_layout() != self.layout || (self.dome.is_some() && view != self.view) {
            self.view = view;
            self.build(world.cube_layout(), aw);
        }
        // State A (after mid A..D), copied out in one go: the mapped dump is uncached
        // memory, slow to read texel by texel.
        let start = 4 * 8 * aw * ah;
        let mut raw = vec![0u8; 8 * aw * ah];
        raw.copy_from_slice(&world.dumped()[start..start + 8 * aw * ah]);
        self.state.clear();
        self.state.extend(raw.chunks_exact(8).map(|t| -> [f32; 3] {
            std::array::from_fn(|c| half::f16::from_le_bytes([t[2 * c], t[2 * c + 1]]).to_f32())
        }));
        let tone = world.tone().map(|[black, white, gamma]| {
            let scale = 1. / (white - black).max(1e-3);
            (scale, -black * scale, gamma)
        });
        out.clear();
        out.resize(3 * self.table.len(), 0);
        let band = self.table.len().div_ceil(threads());
        let state = &self.state;
        std::thread::scope(|scope| {
            for (entries, pixels) in self.table.chunks(band).zip(out.chunks_mut(3 * band)) {
                scope.spawn(move || {
                    for (&(at, right, down, tx, ty), pixel) in entries.iter().zip(pixels.chunks_exact_mut(3)) {
                        if at == u32::MAX {
                            continue; // outside the dome: black
                        }
                        let at = at as usize;
                        let (p, b) = (state[at], state[at + right as usize]);
                        let (c, e) = (state[at + down as usize], state[at + down as usize + right as usize]);
                        for ch in 0..3 {
                            let value = (p[ch] * (1. - tx) + b[ch] * tx) * (1. - ty) + (c[ch] * (1. - tx) + e[ch] * tx) * ty;
                            let (scale, offset, gamma) = tone[ch];
                            let v = (value * scale + offset).max(0.);
                            let v = if (gamma - 1.).abs() < 1e-3 { v } else { v.powf(gamma) };
                            pixel[ch] = if v.is_finite() { (v.min(1.) * 255. + 0.5) as u8 } else { 0 };
                        }
                    }
                });
            }
        });
    }
}

fn video(path: &Path, width: usize, height: usize, fps: f32) -> Result<Child> {
    Command::new("ffmpeg")
        .args(["-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", &format!("{width}x{height}")])
        .args(["-r", &format!("{fps}"), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18"])
        .arg(path)
        .stdin(Stdio::piped())
        .spawn()
        .context("Starting ffmpeg (VIDEO)")
}

pub fn run(shaders: &Path) -> Result<()> {
    let gpu = plain_gpu()?;
    let d = &gpu.device;
    let mut resources = Resources::new(&gpu);
    let rule = rule::Rule::load(env::var_os("WEIGHTS").map(PathBuf::from).as_deref())?;
    if rule.layout != 3 {
        bail!("DESKTOP renders the cube-v1 sphere store: WEIGHTS must be a cube-v1 net (this is {})", rule.layout_name());
    }
    println!("Net: {} {} ({} weights, {} mip levels)", if rule.trained() { "trained" } else { "random" }, rule.layout_name(), rule.len(), rule.levels());
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
    println!("Cube: {face} texels per face ({:.2} arcmin); eyes {edge} px, {:.0} deg", (std::f32::consts::FRAC_PI_2 / face as f32).to_degrees() * 60., fov.to_degrees());

    let pool = resources.keep(
        unsafe { d.create_command_pool(&vk::CommandPoolCreateInfo::default().queue_family_index(gpu.family).flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER), None)? },
        Resource::Commands,
    );
    let cmd = unsafe { d.allocate_command_buffers(&vk::CommandBufferAllocateInfo::default().command_pool(pool).command_buffer_count(1))? }[0];
    let fence = resources.keep(unsafe { d.create_fence(&vk::FenceCreateInfo::default(), None)? }, Resource::Fence);
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
    let width: usize = env::var("PANORAMA_WIDTH").ok().and_then(|v| v.parse().ok()).unwrap_or(1024);
    let still = env::var_os("PANORAMA").map(PathBuf::from).unwrap_or_else(|| PathBuf::from("build/desktop/panorama.ppm"));
    if let Some(dir) = still.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let degrees = |name: &str, default: f32| env::var(name).ok().and_then(|v| v.parse::<f32>().ok()).unwrap_or(default).to_radians();
    let dome = env::var("DOME").is_ok_and(|v| v != "0").then(|| degrees("DOME_FOV", 180.));
    let mut panorama = Panorama::new(width, dome);
    let height = panorama.height();
    let mut ffmpeg = match env::var_os("VIDEO") {
        Some(path) => Some(video(Path::new(&path), width, height, controls.get("netfps").round().max(1.))?),
        None => None,
    };
    let limit: Option<u64> = env::var("FRAMES").ok().and_then(|v| v.parse().ok());
    let mut zip = Zip { position: Vec3::new(0., START_HEIGHT, 0.), ..Zip::default() };
    let mut clock = NetClock::default();
    let mut swarm = swarm::Swarm::new()?;
    let (mut weights, mut packed) = (vec![0f32; rule.len()], vec![0u16; rule.len()]);
    let mut pixels = Vec::with_capacity(width * height * 3);
    let (mut frames, mut updates, mut since) = (0u64, 0u64, std::time::Instant::now());
    let (mut gpu_ms, mut read_ms) = (0f64, 0f64);
    let tick_ns = unsafe { gpu.instance.get_physical_device_properties(gpu.physical).limits.timestamp_period } as f64;
    println!("Desktop: panoramas to {}{}", still.display(), env::var("VIDEO").map(|v| format!(" and {v}")).unwrap_or_default());

    while !quit.load(Ordering::SeqCst) && limit.is_none_or(|n| frames < n) {
        let seconds = frames as f32 * DT;
        let city = city::City {
            cell: controls.get("cell"),
            height: controls.get("height"),
            blocks: controls.get("blocks").round().max(1.) as u32,
            player: glam::Vec2::new(zip.position.x, zip.position.z),
        };
        if controls.on("auto") {
            zip.fly(DT, Vec3::ZERO, &city, controls.get("speed"), controls.get("body"));
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
            swarm.update(params, seconds, DT, zip.position);
            world.upload_creatures(&swarm.out);
        } else {
            world.upload_creatures(&[]);
        }
        unsafe {
            d.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())?;
            d.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default())?;
            d.cmd_reset_query_pool(cmd, queries, 0, world::STAMPS);
            d.cmd_write_timestamp(cmd, vk::PipelineStageFlags::TOP_OF_PIPE, queries, 0);
            let frame_rule = rule.frame();
            rule.weights(seconds, settings.wild, &mut weights);
            rule::pack(&rule.layers, &weights, &mut packed);
            world.capture = frames % 8 == 0;
            world.dumping = true; // the cube copies its new state on update frames only
            world.draw(d, cmd, queries, cameras, [0, 0], seconds, &frame_rule, &packed, settings);
            d.end_command_buffer(cmd)?;
            d.reset_fences(&[fence])?;
            d.queue_submit(gpu.queue, &[vk::SubmitInfo::default().command_buffers(&[cmd])], fence)?;
            d.wait_for_fences(&[fence], true, u64::MAX)?;
            let mut stamps = [0u64; world::STAMPS as usize];
            if d.get_query_pool_results(queries, 0, &mut stamps, vk::QueryResultFlags::TYPE_64).is_ok() {
                gpu_ms += stamps[world::STAMPS as usize - 1].wrapping_sub(stamps[0]) as f64 * tick_ns / 1e6;
            }
        }
        let read = std::time::Instant::now();
        if world.capture {
            if controls.on("exposure") {
                world.calibrate(controls.get("mid"), controls.get("clip"), controls.get("black"), controls.get("tint"));
            } else {
                world.expose_plainly();
            }
        }
        if settings.tau > 0. {
            updates += 1;
            panorama.draw(&world, (turn * look, controls.get("attention").to_radians()), &mut pixels);
            let mut file = Vec::with_capacity(pixels.len() + 32);
            write!(file, "P6\n{width} {height}\n255\n")?;
            file.extend_from_slice(&pixels);
            let temporary = still.with_extension("tmp");
            std::fs::write(&temporary, &file)?;
            std::fs::rename(&temporary, &still)?; // whole files only, for viewers
            if let Some(child) = &mut ffmpeg {
                child.stdin.as_mut().context("ffmpeg stdin")?.write_all(&pixels)?;
            }
        }
        read_ms += read.elapsed().as_secs_f64() * 1e3;
        frames += 1;
        if frames % 90 == 0 {
            let rate = 90. / std::mem::replace(&mut since, std::time::Instant::now()).elapsed().as_secs_f64();
            let forward = turn * look * Vec3::NEG_Z;
            println!(
                "{frames} frames ({:.1} s simulated, {rate:.1} frames/s: GPU {:.2} ms/frame, panoramas {:.1} ms/frame), {updates} updates, at ({:.1}, {:.1}, {:.1}) looking ({:.2}, {:.2})",
                seconds, gpu_ms / 90., read_ms / 90., origin.x, origin.y, origin.z, forward.x, forward.z
            );
            (gpu_ms, read_ms) = (0., 0.);
        }
    }
    if let Some(mut child) = ffmpeg {
        drop(child.stdin.take());
        child.wait()?;
    }
    unsafe { d.device_wait_idle()? };
    Ok(())
}
