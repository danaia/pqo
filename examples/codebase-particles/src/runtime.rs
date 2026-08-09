use std::{ffi::c_void, os::raw::c_char};

const ABI_VERSION: u32 = 1;
const MAX_OVERRIDES: usize = 32;
const NAME_CAPACITY: usize = 96;
const EVENT_CURSOR_MOVED: u32 = 1;
const EVENT_LEFT_MOUSE: u32 = 2;
const EVENT_SCROLL: u32 = 3;

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct ProjectEventV1 {
    kind: u32,
    pressed: u32,
    key: u32,
    _reserved: u32,
    x: f32,
    y: f32,
    delta: f32,
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
    fn default() -> Self {
        Self { name: [0; NAME_CAPACITY], value: 0.0 }
    }
}

#[repr(C)]
pub struct ProjectFrameOutputV1 {
    override_count: u32,
    request_redraw: u32,
    overrides: [ProjectF32OverrideV1; MAX_OVERRIDES],
}

struct State {
    cursor: [f32; 2],
    previous_cursor: [f32; 2],
    viewport: [f32; 2],
    center: [f32; 2],
    zoom: f32,
    orbit_angle: f32,
    orbit_selected: bool,
    pointer_down: bool,
    click_generation: f32,
    fps: f32,
    gpu_memory_mb: f32,
}

impl Default for State {
    fn default() -> Self {
        Self {
            cursor: [480.0, 360.0],
            previous_cursor: [480.0, 360.0],
            viewport: [960.0, 720.0],
            center: [0.0, 0.0],
            zoom: 1.0,
            orbit_angle: 0.0,
            orbit_selected: false,
            pointer_down: false,
            click_generation: 0.0,
            fps: 0.0,
            gpu_memory_mb: 0.0,
        }
    }
}

impl State {
    fn aspect(&self) -> f32 {
        self.viewport[0] / self.viewport[1].max(1.0)
    }

    fn pointer_ndc(&self) -> [f32; 2] {
        [
            self.cursor[0] / self.viewport[0].max(1.0) * 2.0 - 1.0,
            1.0 - self.cursor[1] / self.viewport[1].max(1.0) * 2.0,
        ]
    }

    fn update_viewport(&mut self, event: ProjectEventV1) {
        if event.viewport_width > 0.0 && event.viewport_height > 0.0 {
            self.viewport = [event.viewport_width, event.viewport_height];
        }
    }

    fn zoom_at_cursor(&mut self, delta: f32) {
        let pointer = self.pointer_ndc();
        let scaled_pointer = [pointer[0] * self.aspect(), pointer[1]];
        let world_before = [
            self.center[0] + scaled_pointer[0] / self.zoom,
            self.center[1] + scaled_pointer[1] / self.zoom,
        ];
        self.zoom = (self.zoom * (-delta * 0.12).exp()).clamp(0.55, 6.0);
        if !self.orbit_selected {
            self.center = [
                world_before[0] - scaled_pointer[0] / self.zoom,
                world_before[1] - scaled_pointer[1] / self.zoom,
            ];
        }
    }

    fn event(&mut self, event: ProjectEventV1) {
        self.update_viewport(event);
        match event.kind {
            EVENT_CURSOR_MOVED => {
                self.previous_cursor = self.cursor;
                self.cursor = [event.x, event.y];
                if self.pointer_down && self.orbit_selected {
                    let dx = self.cursor[0] - self.previous_cursor[0];
                    self.orbit_angle += dx * 0.008;
                }
            }
            EVENT_LEFT_MOUSE => {
                self.previous_cursor = [event.x, event.y];
                self.cursor = [event.x, event.y];
                self.pointer_down = event.pressed != 0;
                if self.pointer_down {
                    self.click_generation += 1.0;
                }
            }
            EVENT_SCROLL => {
                self.cursor = [event.x, event.y];
                self.zoom_at_cursor(event.delta);
            }
            _ => {}
        }
    }

    fn control(&mut self, name: &str, value: f32) -> bool {
        match name {
            "interaction.orbit_selected" => self.orbit_selected = value > 0.5,
            "camera.orbit_angle" => self.orbit_angle = value,
            "camera.zoom" => self.zoom = value.clamp(0.55, 6.0),
            "camera.reset" => {
                self.center = [0.0, 0.0];
                self.zoom = 1.0;
                self.orbit_angle = 0.0;
                self.orbit_selected = false;
            }
            _ => return false,
        }
        true
    }

    fn frame(&mut self, context: ProjectFrameContextV1, output: &mut ProjectFrameOutputV1) {
        if context.viewport_width > 0.0 && context.viewport_height > 0.0 {
            self.viewport = [context.viewport_width, context.viewport_height];
        }
        self.fps = context.frames_per_second;
        self.gpu_memory_mb = context.gpu_memory_mb;
        let pointer = self.pointer_ndc();
        for (name, value) in [
            ("interaction.pointer_x", pointer[0]),
            ("interaction.pointer_y", pointer[1]),
            ("interaction.click_generation", self.click_generation),
            ("camera.center_x", self.center[0]),
            ("camera.center_y", self.center[1]),
            ("camera.zoom", self.zoom),
            ("camera.orbit_angle", self.orbit_angle),
            ("camera.orbit_selected", if self.orbit_selected { 1.0 } else { 0.0 }),
            ("camera.aspect", self.aspect()),
            ("interaction.hud_fps", self.fps),
            ("interaction.hud_gpu_mb", self.gpu_memory_mb),
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
    b"Pqo - Codebase Particles\0".as_ptr() as *const c_char
}

#[no_mangle]
pub extern "C" fn pqo_project_help_v1() -> *const c_char {
    b"click a file to select; scroll to zoom at the pointer; enable Orbit selection and drag horizontally\0".as_ptr() as *const c_char
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
    (*(state as *mut State)).event(*event);
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

#[cfg(test)]
mod tests {
    use super::{ProjectEventV1, State, EVENT_SCROLL};

    #[test]
    fn zoom_preserves_the_world_point_under_the_cursor() {
        let mut state = State::default();
        state.cursor = [720.0, 180.0];
        let pointer = state.pointer_ndc();
        let before = [
            state.center[0] + pointer[0] * state.aspect() / state.zoom,
            state.center[1] + pointer[1] / state.zoom,
        ];
        state.event(ProjectEventV1 {
            kind: EVENT_SCROLL,
            x: 720.0,
            y: 180.0,
            delta: -1.0,
            viewport_width: 960.0,
            viewport_height: 720.0,
            ..ProjectEventV1::default()
        });
        let after = [
            state.center[0] + pointer[0] * state.aspect() / state.zoom,
            state.center[1] + pointer[1] / state.zoom,
        ];
        assert!((before[0] - after[0]).abs() < 1e-5);
        assert!((before[1] - after[1]).abs() < 1e-5);
    }
}
