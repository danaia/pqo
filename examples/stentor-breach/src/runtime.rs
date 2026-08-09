use std::{ffi::c_void, os::raw::c_char};

const ABI_VERSION: u32 = 1;
const MAX_OVERRIDES: usize = 32;
const NAME_CAPACITY: usize = 96;

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct ProjectEventV1 {
    _kind: u32,
    _pressed: u32,
    _key: u32,
    _reserved: u32,
    _x: f32,
    _y: f32,
    _delta: f32,
    viewport_width: f32,
    viewport_height: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct ProjectFrameContextV1 {
    viewport_width: f32,
    viewport_height: f32,
    frames_per_second: f32,
    gpu_memory_mb: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ProjectControlV1 {
    name: [u8; NAME_CAPACITY],
    value: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ProjectF32OverrideV1 {
    name: [u8; NAME_CAPACITY],
    value: f32,
}

impl Default for ProjectF32OverrideV1 {
    fn default() -> Self { Self { name: [0; NAME_CAPACITY], value: 0.0 } }
}

#[repr(C)]
pub struct ProjectFrameOutputV1 {
    override_count: u32,
    request_redraw: u32,
    overrides: [ProjectF32OverrideV1; MAX_OVERRIDES],
}

struct State {
    viewport: [f32; 2],
    playing: bool,
    elapsed: f32,
    particle_density: f32,
    membrane_tension: f32,
    brownian_motion: f32,
    chemotaxis: f32,
    repair_rate: f32,
    reset_pulse: bool,
}

impl Default for State {
    fn default() -> Self {
        Self {
            viewport: [1280.0, 800.0],
            playing: true,
            elapsed: 0.0,
            particle_density: 0.25,
            membrane_tension: 0.72,
            brownian_motion: 0.42,
            chemotaxis: 0.68,
            repair_rate: 0.58,
            reset_pulse: false,
        }
    }
}

impl State {
    fn phase(&self) -> f32 { (self.elapsed / 8.0).floor().clamp(0.0, 4.0) }

    fn control(&mut self, name: &str, value: f32) -> bool {
        match name {
            "interaction.playing" => self.playing = value > 0.5,
            "interaction.phase" => self.elapsed = value.round().clamp(0.0, 4.0) * 8.0,
            "interaction.particle_density" => self.particle_density = value.clamp(0.16, 1.0),
            "interaction.membrane_tension" => self.membrane_tension = value.clamp(0.0, 1.0),
            "interaction.brownian_motion" => self.brownian_motion = value.clamp(0.0, 1.0),
            "interaction.chemotaxis" => self.chemotaxis = value.clamp(0.0, 1.0),
            "interaction.repair_rate" => self.repair_rate = value.clamp(0.0, 1.0),
            "interaction.reset_scene" => {
                if value > 0.5 { self.reset_pulse = true; }
            }
            _ => return false,
        }
        true
    }

    fn frame(&mut self, context: ProjectFrameContextV1, output: &mut ProjectFrameOutputV1) {
        if context.viewport_width > 0.0 && context.viewport_height > 0.0 {
            self.viewport = [context.viewport_width, context.viewport_height];
        }
        if self.playing {
            let frame_dt = if context.frames_per_second > 1.0 {
                1.0 / context.frames_per_second
            } else { 1.0 / 60.0 };
            self.elapsed = (self.elapsed + frame_dt) % 40.0;
        }
        let reset = if self.reset_pulse { 1.0 } else { 0.0 };
        self.reset_pulse = false;
        for (name, value) in [
            ("interaction.playing", if self.playing { 1.0 } else { 0.0 }),
            ("interaction.phase", self.phase()),
            ("interaction.particle_density", self.particle_density),
            ("interaction.membrane_tension", self.membrane_tension),
            ("interaction.brownian_motion", self.brownian_motion),
            ("interaction.chemotaxis", self.chemotaxis),
            ("interaction.repair_rate", self.repair_rate),
            ("interaction.reset_scene", reset),
            ("interaction.hud_fps", context.frames_per_second),
            ("interaction.hud_gpu_mb", context.gpu_memory_mb),
            ("interaction.hud_gpu_frame_ms", 1000.0 / context.frames_per_second.max(1.0)),
            ("camera.aspect", self.viewport[0] / self.viewport[1].max(1.0)),
        ] {
            push(output, name, value);
        }
    }
}

fn push(output: &mut ProjectFrameOutputV1, name: &str, value: f32) {
    let index = output.override_count as usize;
    if index < MAX_OVERRIDES && name.len() < NAME_CAPACITY {
        output.overrides[index].name[..name.len()].copy_from_slice(name.as_bytes());
        output.overrides[index].value = value;
        output.override_count += 1;
    }
}

#[no_mangle]
pub extern "C" fn pqo_project_abi_version_v1() -> u32 { ABI_VERSION }

#[no_mangle]
pub extern "C" fn pqo_project_title_v1() -> *const c_char {
    b"Pqo - Stentor Breach\0".as_ptr() as *const c_char
}

#[no_mangle]
pub extern "C" fn pqo_project_help_v1() -> *const c_char {
    b"Holophrya pierces a Stentor; use the companion panel to inspect the five phases and tune the biophysics.\0".as_ptr() as *const c_char
}

#[no_mangle]
pub extern "C" fn pqo_project_create_v1() -> *mut c_void {
    Box::into_raw(Box::new(State::default())) as *mut c_void
}

#[no_mangle]
pub unsafe extern "C" fn pqo_project_destroy_v1(state: *mut c_void) {
    if !state.is_null() { drop(Box::from_raw(state as *mut State)); }
}

#[no_mangle]
pub unsafe extern "C" fn pqo_project_event_v1(
    state: *mut c_void,
    event: *const ProjectEventV1,
) -> u32 {
    if state.is_null() || event.is_null() { return 0; }
    let event = *event;
    if event.viewport_width > 0.0 && event.viewport_height > 0.0 {
        (*(state as *mut State)).viewport = [event.viewport_width, event.viewport_height];
    }
    1
}

#[no_mangle]
pub unsafe extern "C" fn pqo_project_control_v1(
    state: *mut c_void,
    control: *const ProjectControlV1,
) -> u32 {
    if state.is_null() || control.is_null() { return 0; }
    let control = &*control;
    let length = control.name.iter().position(|byte| *byte == 0).unwrap_or(NAME_CAPACITY);
    let Ok(name) = std::str::from_utf8(&control.name[..length]) else { return 0; };
    u32::from((*(state as *mut State)).control(name, control.value))
}

#[no_mangle]
pub unsafe extern "C" fn pqo_project_frame_v1(
    state: *mut c_void,
    context: *const ProjectFrameContextV1,
    output: *mut ProjectFrameOutputV1,
) -> u32 {
    if state.is_null() || context.is_null() || output.is_null() { return 0; }
    let output = &mut *output;
    output.override_count = 0;
    output.request_redraw = 0;
    output.overrides = [ProjectF32OverrideV1::default(); MAX_OVERRIDES];
    (*(state as *mut State)).frame(*context, output);
    1
}
