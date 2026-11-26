mod creatures;
mod desktop;
mod swarm;
mod city;
mod gpu;
mod swing;
mod timestop;
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
/// Where the player starts (m above the ground); only the zip moves them.
const START_HEIGHT: f32 = 7.5;
/// ZIP_ANYWHERE: how far along the ray the anchor goes when it misses the blocks (m).
const MISS_DISTANCE: f32 = 40.;
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
            let (device, queue, family, stats) = make_device(&vulkan, physical_handle, |info| {
                let raw = instance
                    .create_vulkan_device(system, get_proc, physical, (info as *const vk::DeviceCreateInfo).cast())?
                    .map_err(vk::Result::from_raw)?;
                Ok(ash::Device::load(vulkan.fp_v1_0(), vk::Device::from_raw(raw as u64)))
            })?;
            Ok((physical_handle, device, queue, family, stats))
        })();
        match setup {
            Ok((physical, device, queue, family, stats)) => Ok(Gpu {
                stats,
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

/// The logical device on `physical`: a graphics + compute queue, shaderFloat16, large
/// points, the pipeline statistics where offered. `create` makes the device from the info
/// (OpenXR's create_vulkan_device, or plain Vulkan in desktop.rs).
unsafe fn make_device(
    vulkan: &ash::Instance,
    physical: vk::PhysicalDevice,
    create: impl FnOnce(&vk::DeviceCreateInfo) -> Result<ash::Device>,
) -> Result<(ash::Device, vk::Queue, u32, Option<ash::khr::pipeline_executable_properties::Device>)> {
    unsafe {
        let family = vulkan
            .get_physical_device_queue_family_properties(physical)
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
        // shaderFloat16: the net runs in float16 (net.frag, packed math).
        let mut half = vk::PhysicalDeviceShaderFloat16Int8Features::default();
        vulkan.get_physical_device_features2(
            physical,
            &mut vk::PhysicalDeviceFeatures2::default().push_next(&mut half),
        );
        ensure!(half.shader_float16 == vk::TRUE, "The GPU lacks shaderFloat16");
        let mut half = vk::PhysicalDeviceShaderFloat16Int8Features::default().shader_float16(true);
        let mut extensions = vec![ash::khr::shader_float16_int8::NAME.as_ptr()];
        // The driver's statistics (registers, instructions) per pipeline, if it has them.
        let stats = vulkan
            .enumerate_device_extension_properties(physical)?
            .iter()
            .any(|e| e.extension_name_as_c_str() == Ok(ash::khr::pipeline_executable_properties::NAME));
        let mut executable = vk::PhysicalDevicePipelineExecutablePropertiesFeaturesKHR::default()
            .pipeline_executable_info(true);
        if stats {
            extensions.push(ash::khr::pipeline_executable_properties::NAME.as_ptr());
        }
        // largePoints: the splat draws each previous pixel as a 2x2 point sprite.
        let features = vk::PhysicalDeviceFeatures::default().large_points(true);
        let info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queues)
            .enabled_extension_names(&extensions)
            .enabled_features(&features)
            .push_next(&mut half);
        let info = if stats { info.push_next(&mut executable) } else { info };
        let device = create(&info)?;
        let queue = device.get_device_queue(family, 0);
        let properties = vulkan.get_physical_device_properties(physical);
        println!(
            "Vulkan GPU: {}",
            CStr::from_ptr(properties.device_name.as_ptr()).to_string_lossy()
        );
        world::CAPTURE_STATS.store(stats, std::sync::atomic::Ordering::Relaxed);
        let stats = stats.then(|| ash::khr::pipeline_executable_properties::Device::new(&vulkan, &device));
        Ok((device, queue, family, stats))
    }
}

/// The driver's statistics of each pipeline's shaders, one line per shader.
fn print_stats(gpu: &Gpu, world: &World) {
    let Some(stats) = &gpu.stats else { return };
    for (name, pipeline) in world.named_pipelines() {
        let info = vk::PipelineInfoKHR::default().pipeline(pipeline);
        let Ok(executables) = (unsafe { stats.get_pipeline_executable_properties(&info) }) else { continue };
        for (index, executable) in executables.iter().enumerate() {
            let info = vk::PipelineExecutableInfoKHR::default().pipeline(pipeline).executable_index(index as u32);
            let Ok(values) = (unsafe { stats.get_pipeline_executable_statistics(&info) }) else { continue };
            let values: Vec<String> = values
                .iter()
                .map(|v| {
                    let name = v.name_as_c_str().map(|n| n.to_string_lossy()).unwrap_or_default();
                    let value = unsafe {
                        match v.format {
                            vk::PipelineExecutableStatisticFormatKHR::BOOL32 => v.value.b32.to_string(),
                            vk::PipelineExecutableStatisticFormatKHR::INT64 => v.value.i64.to_string(),
                            vk::PipelineExecutableStatisticFormatKHR::UINT64 => v.value.u64.to_string(),
                            _ => format!("{:.2}", v.value.f64),
                        }
                    };
                    format!("{name} {value}")
                })
                .collect();
            let stage = executable.name_as_c_str().map(|n| n.to_string_lossy()).unwrap_or_default();
            println!("stats {name} {stage}: {}", values.join(", "));
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
/// keep head translation. `origin` places the player in the world, turned by `turn`
/// (the autopilot's look direction). OpenGL-style clip space, y up.
fn camera(view: &xr::View, head: Vec3, origin: Vec3, turn: Quat) -> EyeCamera {
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
    let position = origin + turn * (vector(view.pose.position) - head);
    let pose = Mat4::from_rotation_translation(turn * quaternion(view.pose.orientation), position);
    EyeCamera {
        view_proj: projection * pose.inverse(),
        position,
    }
}

/// The controllers: trigger (zip), aim pose and haptics per hand.
struct Hands {
    set: xr::ActionSet,
    paths: [xr::Path; 2],
    zip: xr::Action<bool>,
    /// Grip: personal time-stop (timestop.rs).
    stop: xr::Action<bool>,
    buzz: xr::Action<xr::Haptic>,
    aims: [xr::Space; 2],
    /// XR_EXT_eye_gaze_interaction: the gaze pose (action kept alive with its space).
    gaze: Option<(xr::Action<xr::Posef>, xr::Space)>,
}

/// One hand this frame: trigger held, and its aim ray in tracking space.
struct Held {
    pressed: bool,
    /// Grip held: time-stop.
    stop: bool,
    ray: Option<(Vec3, Vec3)>,
}

impl Hands {
    fn new(instance: &xr::Instance, session: &xr::Session<xr::Vulkan>, eye_gaze: bool) -> Result<Self> {
        let set = instance.create_action_set("sunk", "Sunk City", 0)?;
        let paths = [instance.string_to_path("/user/hand/left")?, instance.string_to_path("/user/hand/right")?];
        let zip = set.create_action::<bool>("zip", "Zip toward a block", &paths)?;
        let aim = set.create_action::<xr::Posef>("aim", "Aim", &paths)?;
        let buzz = set.create_action::<xr::Haptic>("buzz", "Hit", &paths)?;
        let stop = set.create_action::<bool>("stop", "Stop time", &paths)?;
        // The runtime picks a matching profile (or remaps one, as SteamVR does).
        for (profile, trigger, grip) in [
            ("khr/simple_controller", "select/click", "menu/click"),
            ("valve/index_controller", "trigger/click", "squeeze/value"),
            ("oculus/touch_controller", "trigger/value", "squeeze/value"),
        ] {
            let mut bindings = Vec::new();
            for hand in ["left", "right"] {
                let path = |p: &str| instance.string_to_path(&format!("/user/hand/{hand}/{p}"));
                bindings.push(xr::Binding::new(&zip, path(&format!("input/{trigger}"))?));
                bindings.push(xr::Binding::new(&aim, path("input/aim/pose")?));
                bindings.push(xr::Binding::new(&buzz, path("output/haptic")?));
                bindings.push(xr::Binding::new(&stop, path(&format!("input/{grip}"))?));
            }
            let profile = instance.string_to_path(&format!("/interaction_profiles/{profile}"))?;
            if let Err(error) = instance.suggest_interaction_profile_bindings(profile, &bindings) {
                println!("Controller bindings for {profile:?} not accepted: {error}");
            }
        }
        let gaze = if eye_gaze {
            let action = set.create_action::<xr::Posef>("gaze", "Eye gaze", &[])?;
            let binding = [xr::Binding::new(&action, instance.string_to_path("/user/eyes_ext/input/gaze_ext/pose")?)];
            let profile = instance.string_to_path("/interaction_profiles/ext/eye_gaze_interaction")?;
            match instance.suggest_interaction_profile_bindings(profile, &binding) {
                Ok(()) => Some(action),
                Err(error) => {
                    println!("Eye gaze binding not accepted: {error}");
                    None
                }
            }
        } else {
            None
        };
        session.attach_action_sets(&[&set])?;
        let gaze = match gaze {
            Some(action) => {
                let space = action.create_space(session.clone(), xr::Path::NULL, xr::Posef::IDENTITY)?;
                Some((action, space))
            }
            None => None,
        };
        let aims = [
            aim.create_space(session.clone(), paths[0], xr::Posef::IDENTITY)?,
            aim.create_space(session.clone(), paths[1], xr::Posef::IDENTITY)?,
        ];
        Ok(Self { set, paths, zip, stop, buzz, aims, gaze })
    }

    fn sync(&self, session: &xr::Session<xr::Vulkan>, space: &xr::Space, time: xr::Time) -> Result<[Held; 2]> {
        session.sync_actions(&[xr::ActiveActionSet::new(&self.set)])?;
        let mut held = [0, 1].map(|_| Held { pressed: false, stop: false, ray: None });
        for (i, out) in held.iter_mut().enumerate() {
            out.pressed = self.zip.state(session, self.paths[i])?.current_state;
            out.stop = self.stop.state(session, self.paths[i])?.current_state;
            let location = self.aims[i].locate(space, time)?;
            let valid = xr::SpaceLocationFlags::POSITION_VALID | xr::SpaceLocationFlags::ORIENTATION_VALID;
            if location.location_flags.contains(valid) {
                let pose = location.pose;
                out.ray = Some((vector(pose.position), quaternion(pose.orientation) * Vec3::NEG_Z));
            }
        }
        Ok(held)
    }

    /// The gaze ray (origin, direction) in tracking space, after sync, where tracked.
    fn gaze(&self, space: &xr::Space, time: xr::Time) -> Option<(Vec3, Vec3)> {
        let (_, gaze) = self.gaze.as_ref()?;
        let location = gaze.locate(space, time).ok()?;
        let valid = xr::SpaceLocationFlags::POSITION_VALID | xr::SpaceLocationFlags::ORIENTATION_VALID;
        let pose = location.pose;
        location.location_flags.contains(valid).then(|| (vector(pose.position), quaternion(pose.orientation) * Vec3::NEG_Z))
    }

    fn buzz(&self, session: &xr::Session<xr::Vulkan>, hand: usize) {
        let pulse = xr::HapticVibration::new()
            .amplitude(0.6)
            .frequency(xr::FREQUENCY_UNSPECIFIED)
            .duration(xr::Duration::from_nanos(40_000_000));
        let _ = self.buzz.apply_feedback(session, self.paths[hand], &pulse);
    }
}

/// A blend weight per 90 Hz frame as a rate per second: 1 - (1 - w)^(90 dt) = 1 - exp(-rate dt).
fn per_second(per_frame: f32) -> f32 {
    -(1. - per_frame.min(0.999)).ln() * 90.
}

/// When the net updates: every 1/netfps seconds of (time-scaled) time, by the ticks
/// that passed (tau = elapsed * rate, at most the largest trained step); the frames in
/// between only reproject and fill holes (tau 0). The display stays at the full rate.
#[derive(Default)]
struct NetClock {
    /// Real and time-scaled seconds since the last update.
    real: f32,
    scaled: f32,
    last: Option<f32>,
}

impl NetClock {
    fn tick(&mut self, seconds: f32, timescale: f32, netfps: f32, rule: &rule::Rule, dump: bool) -> f32 {
        let dt = (seconds - self.last.replace(seconds).unwrap_or(seconds)).clamp(0., 0.1);
        self.real += dt;
        self.scaled += dt * timescale;
        if dump {
            // One update of DUMP_TAU (default 1) ticks, for train/parity.py.
            (self.real, self.scaled) = (0., 0.);
            return env::var("DUMP_TAU").ok().and_then(|v| v.parse().ok()).unwrap_or(1.);
        }
        let period = 1. / netfps.max(1.);
        if self.real < period {
            return 0.;
        }
        self.real = (self.real - period).min(period);
        (std::mem::take(&mut self.scaled) * rule.rate).min(rule.max_tau)
    }
}

/// The player. Zip toward blocks: holding a trigger anchors the controller's ray on a
/// block and accelerates the player toward it; drag damps the velocity, so releasing
/// glides to a stop. Autopilot (switch `auto`): wanders through the city at about 0.9
/// of the block height (+- 0.9 m), looking around. `position` is the world origin of
/// the tracking space and `yaw` its turn (camera() `origin`, `turn`).
#[derive(Default)]
struct Zip {
    position: Vec3,
    velocity: Vec3,
    yaw: f32,
    anchors: [Option<Vec3>; 2],
    /// Swing mode: each anchor's rope length (m), and each hand's last position relative
    /// to the head (world orientation) for arm pulls.
    ropes: [f32; 2],
    hands: [Option<Vec3>; 2],
    was_pressed: [bool; 2],
    last: Option<f32>,
    anywhere: bool,
    /// Autopilot: its time and heading.
    flown: f32,
    heading: f32,
}

/// How quickly the autopilot steers, climbs and looks (1/s).
const AUTO_RESPONSE: f32 = 1.5;

impl Zip {
    fn turn(&self) -> Quat {
        Quat::from_rotation_y(self.yaw)
    }

    /// Steer the velocity along a wandering heading at ~0.9 of the block height, and turn
    /// the view toward the heading, glancing around.
    fn autopilot(&mut self, dt: f32, eye: Vec3, height: f32, speed: f32) {
        self.flown += dt;
        let t = self.flown;
        self.heading += dt * (0.25 * (0.07 * t).sin() + 0.15 * (0.031 * t + 1.).sin());
        let ahead = Vec3::new(-self.heading.sin(), 0., -self.heading.cos());
        let altitude = 0.9 * height + 0.9 * (0.13 * t).sin();
        let climb = (altitude - (self.position.y + eye.y)) * AUTO_RESPONSE;
        let follow = 1. - (-AUTO_RESPONSE * dt).exp();
        self.velocity += (ahead * speed + Vec3::Y * climb - self.velocity) * follow;
        let look = self.heading + 0.6 * (0.09 * t).sin();
        self.yaw += (look - self.yaw) * follow;
    }

    /// pull (m/s²), drag (1/s), body: the collision sphere's radius (m); `eye` the tracked eye centre after the head
    /// scaling (tracking space).
    #[allow(clippy::too_many_arguments)]
    fn update(
        &mut self,
        held: &[Held; 2],
        hands: &Hands,
        session: &xr::Session<xr::Vulkan>,
        seconds: f32,
        head: Vec3,
        eye: Vec3,
        city: &city::City,
        [pull, drag, body, brake]: [f32; 4],
        walls: &[Vec3],
        auto: Option<f32>,
        swing: Option<f32>,
    ) {
        let dt = (seconds - self.last.replace(seconds).unwrap_or(seconds)).clamp(0., 0.1);
        let eye = self.turn() * eye;
        if let Some(speed) = auto {
            self.autopilot(dt, eye, city.height, speed);
        }
        for (i, hand) in held.iter().enumerate() {
            if hand.pressed && !self.was_pressed[i] {
                self.anchors[i] = hand.ray.and_then(|(from, dir)| {
                    let (from, dir) = (self.position + self.turn() * (from - head), self.turn() * dir);
                    // ZIP_ANYWHERE: missing the blocks anchors in the air, MISS_DISTANCE along the ray.
                    city.hit(from, dir, 150.).or(self.anywhere.then(|| from + dir * MISS_DISTANCE))
                });
                if let Some(anchor) = self.anchors[i] {
                    self.ropes[i] = anchor.distance(self.position + eye);
                    println!("Zip ({}): anchored {:.1} m away", ["left", "right"][i], anchor.distance(self.position + eye));
                    hands.buzz(session, i);
                }
            }
            if !hand.pressed {
                self.anchors[i] = None;
            }
            let at = hand.ray.map(|(from, _)| self.turn() * (from - head));
            self.hands[i] = if self.anchors[i].is_some() { self.hands[i].or(at) } else { None };
            self.was_pressed[i] = hand.pressed;
        }
        let player = self.position + eye;
        match swing {
            // Swing: gravity, and the ropes reel in at `reel` m/s (applied after the move).
            Some(reel) if auto.is_none() => {
                self.velocity.y -= swing::GRAVITY * dt;
                for i in 0..2 {
                    let Some(anchor) = self.anchors[i] else { continue };
                    // Arm pulls: the hand moving away from the anchor (relative to the
                    // head) shortens the rope by ARM times that, and a taut rope pulls
                    // the player in at that speed.
                    let hand = held[i].ray.map(|(from, _)| self.turn() * (from - head));
                    let pulled = match (hand, self.hands[i]) {
                        (Some(now), Some(last)) => swing::pulled(last, now, anchor - player),
                        _ => 0.,
                    };
                    let shorter = reel * dt + swing::ARM * pulled;
                    if player.distance(anchor) >= self.ropes[i] - 0.05 && dt > 0. {
                        self.velocity += (anchor - player).normalize_or_zero() * (swing::ARM * pulled / dt).min(swing::MAX_PULL);
                    }
                    self.ropes[i] = (self.ropes[i] - shorter).max(swing::SHORTEST);
                    self.hands[i] = hand;
                }
            }
            _ => {
                for anchor in self.anchors.iter().flatten() {
                    let to = *anchor - player;
                    let distance = to.length();
                    // Fade out over the last 1.5 m, so the pull does not end in the wall.
                    self.velocity += to / distance.max(1e-3) * pull * (distance / 1.5).min(1.) * dt;
                }
            }
        }
        if auto.is_none() {
            self.velocity *= (-drag * dt).exp();
        }
        // The image's own geometry (last frame's depth probe): within `brake` of the body,
        // the velocity toward a point fades out (fully at the body's surface).
        for point in walls {
            let to = *point - player;
            let distance = to.length();
            let reach = body + brake;
            if brake > 0. && distance < reach && distance > 1e-3 {
                let toward = to / distance;
                let closing = self.velocity.dot(toward).max(0.);
                self.velocity -= toward * closing * ((reach - distance) / brake).min(1.);
            }
        }
        self.position += self.velocity * dt;
        if swing.is_some() && auto.is_none() {
            for (anchor, &length) in self.anchors.iter().zip(&self.ropes) {
                if let Some(anchor) = anchor {
                    let (head, velocity) = swing::rope(self.position + eye, self.velocity, *anchor, length);
                    (self.position, self.velocity) = (head - eye, velocity);
                }
            }
        }
        self.keep_out(city, eye, body);
    }

    /// Keep the head out of the blocks: push it out, and drop the velocity into the walls.
    fn keep_out(&mut self, city: &city::City, eye: Vec3, body: f32) {
        let (free, normals) = city.collide(self.position + eye, body);
        self.position = free - eye;
        for normal in normals {
            self.velocity -= normal * self.velocity.dot(normal).min(0.);
        }
        let floor = 1. - eye.y;
        if self.position.y < floor {
            self.position.y = floor;
            self.velocity.y = self.velocity.y.max(0.);
        }
    }

    /// The autopilot alone (no controllers): one step of `dt` seconds (desktop.rs).
    fn fly(&mut self, dt: f32, eye: Vec3, city: &city::City, speed: f32, body: f32) {
        let eye = self.turn() * eye;
        self.autopilot(dt, eye, city.height, speed);
        self.position += self.velocity * dt;
        self.keep_out(city, eye, body);
    }
}

/// The live faders (av-controls).
fn panel(rule: &rule::Rule) -> av_controls::Controls {
    // Live settings from the av-controls broker (CONTROLS, default wss://localhost:8080);
    // without a broker the defaults below stay.
    // FADERS="wild=0.5,color=0.3": other initial fader values (e.g. without a broker).
    let overrides = env::var("FADERS").unwrap_or_default();
    let initial = |name: &str, default: f32| -> f32 {
        overrides
            .split(',')
            .filter_map(|pair| pair.split_once('='))
            .find(|(key, _)| key.trim() == name)
            .and_then(|(_, value)| value.trim().parse().ok())
            .unwrap_or(default)
    };
    av_controls::Panel::new("sunk-city")
        .section("net")
        .fader("wild", 0., 1., initial("wild", 0.))
        // Certainty gate: the net's change per pixel times 1 - gate * certainty, so
        // well-observed pixels drift less (train net.gate; 1 = full gate). Starts at the
        // value the net trained with (random weights: 1); reproject-v2 has no gate.
        .fader_with("gate", 0., 1., initial("gate", rule.gate.unwrap_or(1.)), 2)
        .fader("timescale", 0., 4., initial("timescale", 1.))
        .fader_with("netfps", 5., 90., initial("netfps", 24.), 0)
        // How much of the scene's colour and depth goes into the state per frame (at
        // 90 Hz; applied per second, so other frame rates look the same).
        .section("scene")
        .fader_with("color", 0., 0.1, initial("color", 0.01), 3)
        .fader_with("depth", 0., 0.3, initial("depth", 0.05), 3)
        // Where nothing is rendered: the sky's distance (m), blended in like depth; 0 = off.
        .fader_with("sky", 0., 1000., initial("sky", 150.), 0)
        // Looking uncovers detail: the scene blends in up to 1 + focus times faster at the view centre.
        .fader_with("focus", 0., 20., initial("focus", 0.), 1)
        // The planet spins fast: seconds per day (0 = fixed sun); night is another world.
        .fader_with("day", 0., 300., initial("day", 0.), 0)
        // Scenes, one tab each, switched on and off independently.
        .section("city")
        .switch("city", initial("city", 1.) > 0.5)
        .fader_with("blocks", 4., 64., initial("blocks", 24.), 0)
        .fader_with("cell", 1., 10., initial("cell", 4.), 1)
        .fader_with("height", 1., 25., initial("height", 7.), 1)
        // Twisted crystals: the fraction of cells whose block is a twisted hexagonal prism.
        .fader_with("crystals", 0., 1., initial("crystals", 0.), 2)
        // Sea creatures (David's Aquarium) as particle soft bodies (swarm.rs).
        .section("creatures")
        .switch("creatures", initial("creatures", 0.) > 0.5)
        .fader_with("count", 1., 100., initial("count", 8.), 0)
        .fader_with("scale", 0.1, 4., initial("scale", 1.), 2)
        .fader_with("hold", 0., 40., initial("hold", 4.), 1)
        .fader_with("link", 0., 400., initial("link", 120.), 0)
        .fader_with("damp", 0., 10., initial("damp", 1.5), 2)
        .fader_with("flow", 0., 20., initial("flow", 2.), 1)
        .fader_with("swirl", 0.5, 30., initial("swirl", 6.), 1)
        .fader_with("dot", 0.01, 0.5, initial("dot", 0.08), 2)
        .section("motion")
        .switch("auto", initial("auto", 0.) > 0.5)
        .fader_with("speed", 0., 15., initial("speed", 3.), 1)
        .fader_with("pull", 0., 60., initial("pull", 25.), 1)
        .fader_with("drag", 0., 6., initial("drag", 2.5), 2)
        // Swing instead of zip: the trigger holds a rope (gravity on, `reel` m/s reeling in).
        .switch("swing", initial("swing", 0.) > 0.5)
        .fader_with("reel", 0., 10., initial("reel", 2.), 1)
        // The player's collision sphere around the head (radius, m).
        .fader_with("body", 0.05, 5., initial("body", 0.3), 2)
        // Braking distance (m) to the net's own geometry (the image's depth), 0 = off.
        .fader_with("brake", 0., 5., initial("brake", 1.5), 2)
        // exposure off: the state's colour shown 1:1 (a trained net stays in the video's
        // range); on: automatic per-channel exposure (mid, clip, black, tint).
        .section("display")
        .switch("exposure", initial("exposure", 0.) > 0.5)
        .fader("mid", 0., 1., initial("mid", 0.5))
        .fader_with("clip", 0., 0.2, initial("clip", 0.03), 3)
        .fader_with("black", 0., 0.3, initial("black", 0.05), 3)
        .fader("tint", 0., 1., initial("tint", 0.3))
        .connect(&env::var("CONTROLS").unwrap_or_else(|_| "wss://localhost:8080".into()))
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
    // Eye tracking (Steam Frame): the focus fader follows the gaze where the runtime has it.
    let eye_gaze = entry.enumerate_extensions()?.ext_eye_gaze_interaction;
    extensions.ext_eye_gaze_interaction = eye_gaze;
    println!("Eye gaze: {}", if eye_gaze { "available" } else { "not offered by the runtime (focus at the view centre)" });
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
    let hands = Hands::new(&instance, &session, eye_gaze)?;

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
    // WEIGHTS=dir: a trained net (train/runs/<name>/net); without it random weights.
    let mut rule = rule::Rule::load(env::var_os("WEIGHTS").map(PathBuf::from).as_deref())?;
    println!(
        "Net: {} {} ({} weights, {} mip levels)",
        if rule.trained() { "trained" } else { "random" },
        rule.layout_name(),
        rule.len(),
        rule.levels()
    );
    ensure!(edge % (1 << (rule.levels() - 1)) == 0, "EYE_SIZE must be divisible by 2^(levels - 1)");
    // CUBE_SIZE: texels per face of cube-v1's sphere store (world.rs). Unset: chosen from
    // the first located views so that a texel is about an eye pixel at the view centre (the
    // net's finest taps, radius 1 texel, then reach about 1 px); until then a small one.
    let pinned_cube = env::var("CUBE_SIZE").ok().and_then(|v| v.parse::<u32>().ok());
    let cube_face = pinned_cube.unwrap_or(1 << (rule.levels() - 1));
    let mut cube_sized = pinned_cube.is_some();
    let mut weights = vec![0f32; rule.len()];
    let mut packed = vec![0u16; rule.len()];
    let mut world = World::new(
        &mut resources,
        shaders,
        size,
        format,
        [&images[0], &images[1]],
        rule.specialization(),
        rule.len(),
        cube_face,
    )?;
    print_stats(&gpu, &world);
    world.mode = match env::var("VIEW").as_deref() {
        Ok("certainty") => 1,
        Ok("depth") => 2,
        Ok("velocity") => 3,
        Ok("hidden") => 4,
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
                    .query_count(world::STAMPS),
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
    // NET_FILE=file: holds a net directory (dev.sh writes it: picker, new exports of the
    // followed run). When the file changes, that net is swapped in live, without a restart
    // (it must read as many mip levels); a net that fails to load leaves the current one.
    let net_file = env::var_os("NET_FILE").map(PathBuf::from);
    let net_stamp_of = |file: &Option<PathBuf>| file.as_ref().and_then(|f| f.metadata().ok()?.modified().ok());
    let mut net_stamp = net_stamp_of(&net_file);
    let mut events = xr::EventDataBuffer::new();
    let mut running = false;
    let mut frames = 0u64;
    let mut gpu_ms = [0f64; world::STAMPS as usize - 1];
    // The net pass on frames where it runs (tau > 0): total ms and count; and the wall clock.
    let (mut net_runs, mut net_run_ms, mut since) = (0u32, 0f64, std::time::Instant::now());
    let mut start = None;
    // ZIP_ANYWHERE=0: zip only toward blocks the ray hits (default 1: also into the air).
    let mut clock = NetClock::default();
    let mut swarm = swarm::Swarm::new()?;
    // Grip: personal time-stop, and the previous frame's time for its dt.
    let (mut time_stop, mut last_seconds) = (timestop::TimeStop::default(), None::<f32>);
    // The image's geometry around the player, from the last frame (for the brake).
    let mut walls: Vec<Vec3> = Vec::new();
    let mut zip = Zip { position: Vec3::new(0., START_HEIGHT, 0.), anywhere: env::var("ZIP_ANYWHERE").map_or(true, |v| v != "0"), ..Zip::default() };
    let controls = panel(&rule);
    // MOVE scales head translation: 0 rotation only (default), 1 real scale, more =
    // giant steps (e.g. to cross the city with Monado's slow qwerty WASD).
    let stride: f32 = env::var("MOVE").ok().and_then(|v| v.parse().ok()).unwrap_or(0.);
    let overscan = env::var("OVERSCAN")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .unwrap_or(1.15);
    // SNAPSHOT=file.ppm: write the left eye's colour every 300 frames (debugging).
    let snapshot = env::var_os("SNAPSHOT").map(PathBuf::from);
    // DUMP=dir: after DUMP_FRAME (60) frames write eye 0's resolved and new state once (dump.bin +
    // dump.json) for train/parity.py, which runs the torch net on the same input.
    let dump = env::var_os("DUMP").map(PathBuf::from);
    let dump_frame: u64 = env::var("DUMP_FRAME").ok().and_then(|v| v.parse().ok()).unwrap_or(60);
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
                    Ok(()) => {
                        println!("Shaders reloaded");
                        print_stats(&gpu, &world);
                    }
                    Err(error) => println!("Shader reload failed, keeping the old ones: {error:#}"),
                }
            }
            let now = net_stamp_of(&net_file);
            if now != net_stamp {
                net_stamp = now;
                thread::sleep(Duration::from_millis(200)); // let the export finish writing
                if let Some(text) = net_file.as_ref().and_then(|f| std::fs::read_to_string(f).ok()) {
                    let dir = PathBuf::from(text.trim());
                    let switched = rule::Rule::load(Some(&dir)).and_then(|next| {
                        unsafe { d.device_wait_idle()? };
                        world.set_net(&mut resources, shaders, next.specialization(), next.len())?;
                        Ok(next)
                    });
                    match switched {
                        Ok(next) => {
                            rule = next;
                            weights = vec![0f32; rule.len()];
                            packed = vec![0u16; rule.len()];
                            println!("Net: {} ({}, {} weights), switched live", dir.display(), rule.layout_name(), rule.len());
                        }
                        Err(error) => println!("Net switch to {} failed, keeping the current one: {error:#}", dir.display()),
                    }
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
        if !cube_sized {
            cube_sized = true;
            // Radians per pixel at the view centre (d angle = d tan there), the finest of
            // both eyes and axes, overscan included.
            let pixel = views
                .iter()
                .flat_map(|v| [v.fov.angle_right.tan() - v.fov.angle_left.tan(), v.fov.angle_up.tan() - v.fov.angle_down.tan()])
                .fold(f32::MAX, f32::min)
                / edge as f32;
            let step = 1u32 << (rule.levels() - 1);
            let face = ((std::f32::consts::FRAC_PI_2 / pixel / step as f32).round() as u32).max(1) * step;
            unsafe { d.device_wait_idle()? };
            world.set_cube_face(&mut resources, face)?;
            let arcmin = |radians: f32| radians.to_degrees() * 60.;
            println!(
                "Cube: {face} texels per face ({:.2} arcmin; eye pixel at the view centre {:.2} arcmin; CUBE_SIZE overrides)",
                arcmin(std::f32::consts::FRAC_PI_2 / face as f32),
                arcmin(pixel)
            );
        }
        let seconds = (time - *start.get_or_insert(time)).as_nanos() as f32 / 1e9;
        // Only the head centre is scaled; the stereo eye offset stays real.
        let head = (vector(views[0].pose.position) + vector(views[1].pose.position)) * 0.5 * (1. - stride);
        // The tracked eye centre that remains after the head scaling (MOVE).
        let eye = (vector(views[0].pose.position) + vector(views[1].pose.position)) * 0.5 - head;
        let city = city::City {
            cell: controls.get("cell"),
            height: controls.get("height"),
            blocks: controls.get("blocks").round().max(1.) as u32,
            player: {
                let p = zip.position + zip.turn() * eye;
                glam::Vec2::new(p.x, p.z)
            },
        };
        let held = hands.sync(&session, &space, time)?;
        let frame_dt = (seconds - last_seconds.replace(seconds).unwrap_or(seconds)).clamp(0., 0.1);
        let stopped = time_stop.update(held.iter().any(|h| h.stop), frame_dt);
        let motion = [controls.get("pull"), controls.get("drag"), controls.get("body"), controls.get("brake")];
        zip.update(&held, &hands, &session, seconds, head, eye, &city, motion, &walls, controls.on("auto").then(|| controls.get("speed")),
            controls.on("swing").then(|| controls.get("reel")));
        let (origin, turn) = (zip.position, zip.turn());
        let cameras = [camera(&views[0], head, origin, turn), camera(&views[1], head, origin, turn)];
        let dumping = dump.is_some() && frames == dump_frame;
        let settings = Settings {
            wild: controls.get("wild"),
            gate: controls.get("gate"),
            tau: clock.tick(seconds, controls.get("timescale") * stopped, controls.get("netfps"), &rule, dumping),
            color: per_second(controls.get("color")),
            depth: per_second(controls.get("depth")),
            sky: controls.get("sky"),
            focus: controls.get("focus"),
            day: (controls.get("day") > 0.).then(|| (seconds / controls.get("day") + 0.25).fract()),
            // A point 50 m along the gaze in the world (as the hands' rays, zip.update).
            gaze: hands.gaze(&space, time).map(|(from, dir)| origin + turn * (from - head) + turn * dir * 50.),
            blocks: controls.get("blocks").round().max(1.) as u32,
            cell: controls.get("cell"),
            height: controls.get("height"),
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
            swarm.update(params, seconds, frame_dt.max(1e-3), zip.position + zip.turn() * eye);
            world.upload_creatures(&swarm.out);
        } else {
            world.upload_creatures(&[]);
        }
        let mut indices = [0; EYES];
        for (swapchain, index) in swapchains.iter_mut().zip(&mut indices) {
            *index = swapchain.acquire_image()? as usize;
            swapchain.wait_image(xr::Duration::INFINITE)?;
        }
        unsafe {
            d.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())?;
            d.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default())?;
            d.cmd_reset_query_pool(cmd, queries, 0, world::STAMPS);
            d.cmd_write_timestamp(cmd, vk::PipelineStageFlags::TOP_OF_PIPE, queries, 0);
            let frame_rule = rule.frame();
            rule.weights(seconds, settings.wild, &mut weights);
            rule::pack(&rule.layers, &weights, &mut packed);
            // Every 8th frame: read back the left eye for the automatic exposure.
            world.capture = frames % 8 == 0;
            world.dumping = dumping;
            world.draw(d, cmd, queries, cameras, indices, seconds, &frame_rule, &packed, settings);
            d.end_command_buffer(cmd)?;
            d.reset_fences(&[fence])?;
            let commands = [cmd];
            d.queue_submit(
                gpu.queue,
                &[vk::SubmitInfo::default().command_buffers(&commands)],
                fence,
            )?;
            d.wait_for_fences(&[fence], true, u64::MAX)?;
            let mut stamps = [0u64; world::STAMPS as usize];
            if d
                .get_query_pool_results(queries, 0, &mut stamps, vk::QueryResultFlags::TYPE_64)
                .is_ok()
            {
                for (ms, pair) in gpu_ms.iter_mut().zip(stamps.windows(2)) {
                    *ms += pair[1].wrapping_sub(pair[0]) as f64 * tick_ns / 1e6;
                }
                if settings.tau > 0. {
                    net_runs += 1;
                    net_run_ms += stamps[4].wrapping_sub(stamps[3]) as f64 * tick_ns / 1e6;
                }
            }
        }
        if let (true, Some(dir)) = (world.dumping, &dump) {
            std::fs::create_dir_all(dir)?;
            std::fs::write(dir.join("dump.bin"), world.dumped())?;
            std::fs::write(
                dir.join("dump.json"),
                format!(
                    "{{\"size\": {edge}, \"width\": {}, \"height\": {}, \"face\": {}, \"gutter\": {}, \"tau\": {}, \"wild\": {}, \"net_layout\": {}, \"layout\": \"mid A, B, C, D (rgba16f), next A, B, C (rgba16f)\"}}\n",
                    world.dump_extent().width, world.dump_extent().height, world.cube_layout().0, world.cube_layout().1, settings.tau, settings.wild, rule.layout
                ),
            )?;
            println!("Dumped to {}", dir.display());
        }
        walls = world.surroundings();
        if world.capture {
            if controls.on("exposure") {
                world.calibrate(controls.get("mid"), controls.get("clip"), controls.get("black"), controls.get("tint"));
            } else {
                world.expose_plainly();
            }
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
            let [scene, splat, resolve, net, display] = gpu_ms.map(|ms| ms / 120.);
            let fps = 120. / std::mem::replace(&mut since, std::time::Instant::now()).elapsed().as_secs_f64();
            let net_run = net_run_ms / net_runs.max(1) as f64;
            let net_fill = (net * 120. - net_run_ms) / (120 - net_runs).max(1) as f64;
            println!(
                "{frames} frames, GPU {:.3} ms/frame at {fps:.0} fps (both eyes: scene {scene:.2} splat {splat:.2} resolve {resolve:.2} net {net:.2} [{net_runs} runs {net_run:.2}, fill {net_fill:.2}] display {display:.2}), looking ({:.2}, {:.2}, {:.2}), exposure r {r} g {g} b {b}, image geometry {} points, nearest {:.1} m",
                scene + splat + resolve + net + display,
                forward.x,
                forward.y,
                forward.z,
                walls.len(),
                walls.iter().map(|p| p.distance(cameras[0].position)).fold(f32::INFINITY, f32::min),
            );
            gpu_ms = [0.; world::STAMPS as usize - 1];
            (net_runs, net_run_ms) = (0, 0.);
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
    // DESKTOP=1: no OpenXR (desktop.rs): virtual eyes, the cube rendered as panoramas.
    if env::var_os("DESKTOP").is_some() {
        return desktop::run(&shaders);
    }
    run(&shaders)
}
