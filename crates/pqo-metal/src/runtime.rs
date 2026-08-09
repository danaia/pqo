use std::{
    borrow::Cow,
    collections::{BTreeMap, VecDeque},
    fs,
    path::{Component, Path, PathBuf},
    process::Command,
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
        mpsc::{self, Receiver, Sender},
    },
    time::{Duration, Instant},
};

use block::ConcreteBlock;
use core_graphics_types::geometry::CGSize;
use metal::{
    Buffer, CommandQueue, CompileOptions, ComputePipelineState, Device, MTLBlendFactor,
    MTLClearColor, MTLCommandBufferStatus, MTLLoadAction, MTLPixelFormat, MTLPrimitiveType,
    MTLResourceOptions, MTLSize, MTLStorageMode, MTLStoreAction, MTLTextureUsage, MetalLayer,
    RenderPassDescriptor, RenderPipelineDescriptor, RenderPipelineState, TextureDescriptor,
};
use objc::{msg_send, rc::autoreleasepool, runtime::YES, sel, sel_impl};
use pqo_core::{
    DataType, DispatchDomain, Literal, PassId, ResourceId, ScalarType, ScheduleItemId, StreamId,
    StreamInitializer, StreamLength, ValueId, ValueKind, ViewId,
};
use pqo_validator::{
    CompletionEnforcement, CompletionRequirement, ExecutionSchedule, PlannedPass, PlannedView,
    ValidatedModuleGraph, Validator,
};
use pqo_windowing::{SnapManager, SnapUpdate, WindowFrame, WindowLayoutConfig};
use winit::{
    dpi::{LogicalSize, PhysicalSize},
    event::{
        ElementState, Event, KeyboardInput, ModifiersState, MouseButton, MouseScrollDelta,
        VirtualKeyCode, WindowEvent,
    },
    event_loop::{ControlFlow, EventLoop},
    platform::macos::WindowExtMacOS,
    window::WindowBuilder,
};

use crate::{
    BenchmarkConfig, BenchmarkMode, BenchmarkResult, BenchmarkRunner, PacingResult,
    PipelineIdentity, PresentationResult, ResourceMetrics, RuntimeDiagnostic,
    RuntimeDiagnosticCode, RuntimeFingerprint, ShaderIdentity, ViewportSize,
    display_link::{DisplayLinkDriver, DisplayUpdate},
    panel::{PanelBridge, PanelCommand, ProjectUi},
    project::{
        EVENT_CURSOR_MOVED, EVENT_KEY, EVENT_LEFT_MOUSE, EVENT_RESIZED, EVENT_SCROLL, KEY_A, KEY_D,
        KEY_DOWN, KEY_G, KEY_LEFT, KEY_RIGHT, KEY_S, KEY_SHIFT, KEY_UP, KEY_W, MODIFIER_COMMAND,
        ProjectEventV1, ProjectExtension, ProjectFrameContextV1,
    },
    sha256, summarize,
};

const INTEGRATE_SOURCE: &str = include_str!("../../../kernels/euler_integrate.metal");
const CONTACT_SOURCE: &str = include_str!("../../../kernels/ground_contact.metal");
const PARTICLE_SOURCE: &str = include_str!("../../../shaders/particle.metal");
const NEON_FLOCK_SOURCE: &str = include_str!("../../../kernels/neon_flock.metal");
const NEON_FLOCK_RENDER_SOURCE: &str = include_str!("../../../shaders/neon_flock.metal");
const CRYSTAL_SOURCE: &str = include_str!("../../../kernels/crystal.metal");
const CRYSTAL_RENDER_SOURCE: &str = include_str!("../../../shaders/crystal.metal");
const INDIRECT_ARGUMENT_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void pqo_prepare_compute_dispatch(
    device const uint* active_count [[buffer(0)]],
    device uint* arguments [[buffer(1)]],
    constant uint& threadgroup_width [[buffer(2)]],
    constant uint& maximum_count [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid != 0) return;
    const uint count = min(active_count[0], maximum_count);
    arguments[0] = (count + threadgroup_width - 1) / threadgroup_width;
    arguments[1] = 1;
    arguments[2] = 1;
}

kernel void pqo_prepare_draw(
    device const uint* active_count [[buffer(0)]],
    device uint* arguments [[buffer(1)]],
    constant uint& maximum_count [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid != 0) return;
    arguments[0] = 6;
    arguments[1] = min(active_count[0], maximum_count);
    arguments[2] = 0;
    arguments[3] = 0;
}
"#;

fn project_key(key: VirtualKeyCode) -> Option<u32> {
    match key {
        VirtualKeyCode::W => Some(KEY_W),
        VirtualKeyCode::A => Some(KEY_A),
        VirtualKeyCode::S => Some(KEY_S),
        VirtualKeyCode::D => Some(KEY_D),
        VirtualKeyCode::Up => Some(KEY_UP),
        VirtualKeyCode::Left => Some(KEY_LEFT),
        VirtualKeyCode::Down => Some(KEY_DOWN),
        VirtualKeyCode::Right => Some(KEY_RIGHT),
        VirtualKeyCode::G => Some(KEY_G),
        VirtualKeyCode::LShift | VirtualKeyCode::RShift => Some(KEY_SHIFT),
        _ => None,
    }
}

pub struct MetalRuntime;

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ScenarioEventRecord {
    pub tick: u64,
    pub pass: String,
    pub value_overrides: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ScenarioRunResult {
    pub scenario: String,
    pub executed_ticks: u64,
    pub events: Vec<ScenarioEventRecord>,
    pub runtime_fingerprint: String,
}

impl MetalRuntime {
    pub fn run(validated: ValidatedModuleGraph) -> Result<(), RuntimeDiagnostic> {
        Self::run_project(validated, None, None)
    }

    pub fn run_project(
        validated: ValidatedModuleGraph,
        project_root: Option<PathBuf>,
        extension_path: Option<PathBuf>,
    ) -> Result<(), RuntimeDiagnostic> {
        Self::run_project_with_ui(validated, project_root, extension_path, None)
    }

    pub fn run_project_with_ui(
        validated: ValidatedModuleGraph,
        project_root: Option<PathBuf>,
        extension_path: Option<PathBuf>,
        project_ui: Option<ProjectUi>,
    ) -> Result<(), RuntimeDiagnostic> {
        let mut project_extension = extension_path
            .as_deref()
            .map(ProjectExtension::load)
            .transpose()?;
        let interactive_crystal = validated.graph().name == "hello_crystal";
        let interactive_worm = validated.graph().name == "hello_worm";
        let title = if let Some(extension) = project_extension.as_ref() {
            extension.title().to_owned()
        } else if interactive_crystal {
            "Pqo — Crystal — left: slice/orbit · scroll: zoom · self-healing".to_owned()
        } else if interactive_worm {
            "Pqo — Scent Weaver — click: feed · drag: orbit · scroll: zoom".to_owned()
        } else {
            format!("Pqo — {}", display_name(validated.graph().name.as_str()))
        };
        let window_layout = project_root
            .as_deref()
            .map(WindowLayoutConfig::load)
            .transpose()
            .map_err(|message| {
                RuntimeDiagnostic::new(RuntimeDiagnosticCode::ProjectPanelFailed, message)
            })?
            .unwrap_or_default();
        let event_loop = EventLoop::new();
        let window = WindowBuilder::new()
            .with_inner_size(LogicalSize::new(
                window_layout.viewer_width,
                window_layout.viewer_height,
            ))
            .with_title(title)
            .build(&event_loop)
            .map_err(|error| {
                RuntimeDiagnostic::new(
                    RuntimeDiagnosticCode::WindowCreationFailed,
                    error.to_string(),
                )
            })?;
        let mut project_panel = project_ui.map(PanelBridge::launch).transpose()?;
        let mut panel_snap = project_panel.as_ref().map(|panel| {
            let layout = panel.window_layout().clone();
            SnapManager::new(layout.clone(), layout.viewer_panel.clone())
        });
        let mut panel_frame = None::<WindowFrame>;
        let device = Device::system_default().ok_or_else(|| {
            RuntimeDiagnostic::new(
                RuntimeDiagnosticCode::DeviceUnavailable,
                "Metal has no system-default device",
            )
        })?;
        let layer = MetalLayer::new();
        layer.set_device(&device);
        layer.set_pixel_format(MTLPixelFormat::BGRA8Unorm);
        layer.set_presents_with_transaction(false);
        layer.set_maximum_drawable_count(2);
        attach_layer(&window, &layer);
        resize_layer(&window, &layer);

        let mut state =
            RuntimeState::new_with_project_root(validated, device, layer, project_root.clone())?;
        println!(
            "runtime_fingerprint:\n{}",
            serde_json::to_string_pretty(&state.fingerprint)
                .expect("runtime fingerprint serialization")
        );
        if let Some(extension) = project_extension.as_ref() {
            if !extension.help().is_empty() {
                println!("interaction: {}", extension.help());
            }
        } else if interactive_crystal {
            println!(
                "interaction: left-drag crystal to slice; left-drag space to orbit; \
                 scroll to zoom; cuts self-heal automatically"
            );
        } else if interactive_worm {
            println!(
                "interaction: click the plane to drop food; left-drag to orbit; \
                 scroll to zoom; the Scent Weaver will smell, pursue, and eat"
            );
        }

        #[derive(Clone, Copy)]
        enum CrystalGesture {
            Slice((f64, f64)),
            Orbit((f64, f64)),
        }

        #[derive(Clone, Copy)]
        struct WormGesture {
            previous: (f64, f64),
            dragged: bool,
        }

        let mut tick_interval = Duration::from_nanos(1_000_000_000 / state.rate_hz as u64);
        let mut next_tick = Instant::now();
        let mut cursor = (0.0_f64, 0.0_f64);
        let mut modifiers = ModifiersState::empty();
        let mut crystal_gesture = None::<CrystalGesture>;
        let mut worm_gesture = None::<WormGesture>;
        event_loop.run(move |event, _, control_flow| {
            *control_flow = ControlFlow::WaitUntil(next_tick);
            autoreleasepool(|| match event {
                Event::WindowEvent { event, .. } => match event {
                    WindowEvent::ModifiersChanged(next) => {
                        modifiers = next;
                    }
                    WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,
                    WindowEvent::Moved(_) => {
                        follow_linked_panel(
                            &window,
                            panel_frame,
                            panel_snap.as_ref(),
                            project_panel.as_mut(),
                        );
                        publish_viewer_frame(&window, project_panel.as_mut());
                    }
                    WindowEvent::Resized(_) | WindowEvent::ScaleFactorChanged { .. } => {
                        resize_layer(&window, &state.layer);
                        follow_linked_panel(
                            &window,
                            panel_frame,
                            panel_snap.as_ref(),
                            project_panel.as_mut(),
                        );
                        publish_viewer_frame(&window, project_panel.as_mut());
                        if let Some(extension) = project_extension.as_mut() {
                            let size = window.inner_size();
                            if extension.event(ProjectEventV1 {
                                kind: EVENT_RESIZED,
                                viewport_width: size.width as f32,
                                viewport_height: size.height as f32,
                                ..ProjectEventV1::default()
                            }) {
                                window.request_redraw();
                            }
                        }
                    }
                    WindowEvent::CursorMoved { position, .. } => {
                        cursor = (position.x, position.y);
                        if let Some(gesture) = crystal_gesture {
                            match gesture {
                                CrystalGesture::Slice(previous) => {
                                    let dx = cursor.0 - previous.0;
                                    let dy = cursor.1 - previous.1;
                                    if dx * dx + dy * dy >= 4.0 {
                                        state.queue_pointer_slice(
                                            previous,
                                            cursor,
                                            window.inner_size(),
                                        );
                                        crystal_gesture = Some(CrystalGesture::Slice(cursor));
                                        window.request_redraw();
                                    }
                                }
                                CrystalGesture::Orbit(previous) => {
                                    let dx = cursor.0 - previous.0;
                                    let dy = cursor.1 - previous.1;
                                    if dx * dx + dy * dy >= 1.0 {
                                        state.queue_pointer_orbit(previous, cursor);
                                        crystal_gesture = Some(CrystalGesture::Orbit(cursor));
                                        window.request_redraw();
                                    }
                                }
                            }
                        }
                        if let Some(mut gesture) = worm_gesture {
                            let dx = cursor.0 - gesture.previous.0;
                            let dy = cursor.1 - gesture.previous.1;
                            if dx * dx + dy * dy >= 1.0 {
                                state.queue_pointer_orbit(gesture.previous, cursor);
                                gesture.previous = cursor;
                                gesture.dragged = true;
                                worm_gesture = Some(gesture);
                                window.request_redraw();
                            }
                        }
                        if let Some(extension) = project_extension.as_mut() {
                            let size = window.inner_size();
                            if extension.event(ProjectEventV1 {
                                kind: EVENT_CURSOR_MOVED,
                                x: cursor.0 as f32,
                                y: cursor.1 as f32,
                                viewport_width: size.width as f32,
                                viewport_height: size.height as f32,
                                ..ProjectEventV1::default()
                            }) {
                                window.request_redraw();
                            }
                        }
                    }
                    WindowEvent::MouseInput {
                        state: ElementState::Pressed,
                        button: MouseButton::Left,
                        ..
                    } if interactive_crystal => {
                        match state.pointer_hits_crystal(cursor, window.inner_size()) {
                            Ok(true) => {
                                crystal_gesture = Some(CrystalGesture::Slice(cursor));
                            }
                            Ok(false) => {
                                crystal_gesture = Some(CrystalGesture::Orbit(cursor));
                            }
                            Err(diagnostic) => {
                                eprintln!(
                                    "{}",
                                    serde_json::to_string_pretty(&diagnostic)
                                        .unwrap_or_else(|_| diagnostic.to_string())
                                );
                                *control_flow = ControlFlow::Exit;
                            }
                        }
                    }
                    WindowEvent::MouseInput {
                        state: ElementState::Pressed,
                        button: MouseButton::Left,
                        ..
                    } if interactive_worm => {
                        worm_gesture = Some(WormGesture {
                            previous: cursor,
                            dragged: false,
                        });
                    }
                    WindowEvent::MouseInput {
                        state: ElementState::Pressed,
                        button: MouseButton::Left,
                        ..
                    } if project_extension.is_some() => {
                        let size = window.inner_size();
                        if project_extension.as_mut().unwrap().event(ProjectEventV1 {
                            kind: EVENT_LEFT_MOUSE,
                            pressed: 1,
                            _reserved: if modifiers.logo() {
                                MODIFIER_COMMAND
                            } else {
                                0
                            },
                            x: cursor.0 as f32,
                            y: cursor.1 as f32,
                            viewport_width: size.width as f32,
                            viewport_height: size.height as f32,
                            ..ProjectEventV1::default()
                        }) {
                            window.request_redraw();
                        }
                    }
                    WindowEvent::MouseInput {
                        state: ElementState::Released,
                        button: MouseButton::Left,
                        ..
                    } => {
                        crystal_gesture = None;
                        if let Some(gesture) = worm_gesture.take()
                            && !gesture.dragged
                        {
                            state.queue_pointer_drop(cursor, window.inner_size());
                            window.request_redraw();
                        }
                        if let Some(extension) = project_extension.as_mut() {
                            let size = window.inner_size();
                            if extension.event(ProjectEventV1 {
                                kind: EVENT_LEFT_MOUSE,
                                pressed: 0,
                                _reserved: if modifiers.logo() {
                                    MODIFIER_COMMAND
                                } else {
                                    0
                                },
                                x: cursor.0 as f32,
                                y: cursor.1 as f32,
                                viewport_width: size.width as f32,
                                viewport_height: size.height as f32,
                                ..ProjectEventV1::default()
                            }) {
                                window.request_redraw();
                            }
                        }
                    }
                    WindowEvent::MouseWheel { delta, .. }
                        if interactive_crystal || interactive_worm =>
                    {
                        let zoom_delta = match delta {
                            MouseScrollDelta::LineDelta(_, vertical) => vertical * 0.12,
                            MouseScrollDelta::PixelDelta(position) => position.y as f32 * 0.003,
                        };
                        state.queue_pointer_zoom(zoom_delta);
                        window.request_redraw();
                    }
                    WindowEvent::MouseWheel { delta, .. } if project_extension.is_some() => {
                        let delta = match delta {
                            MouseScrollDelta::LineDelta(_, vertical) => vertical,
                            MouseScrollDelta::PixelDelta(position) => position.y as f32 * 0.025,
                        };
                        let size = window.inner_size();
                        if project_extension.as_mut().unwrap().event(ProjectEventV1 {
                            kind: EVENT_SCROLL,
                            x: cursor.0 as f32,
                            y: cursor.1 as f32,
                            delta,
                            viewport_width: size.width as f32,
                            viewport_height: size.height as f32,
                            ..ProjectEventV1::default()
                        }) {
                            window.request_redraw();
                        }
                    }
                    WindowEvent::KeyboardInput {
                        input:
                            KeyboardInput {
                                state: key_state,
                                virtual_keycode: Some(key),
                                ..
                            },
                        ..
                    } if project_extension.is_some() => {
                        if let Some(key) = project_key(key) {
                            let size = window.inner_size();
                            if project_extension.as_mut().unwrap().event(ProjectEventV1 {
                                kind: EVENT_KEY,
                                pressed: u32::from(key_state == ElementState::Pressed),
                                key,
                                x: cursor.0 as f32,
                                y: cursor.1 as f32,
                                viewport_width: size.width as f32,
                                viewport_height: size.height as f32,
                                ..ProjectEventV1::default()
                            }) {
                                window.request_redraw();
                            }
                        }
                    }
                    _ => {}
                },
                Event::MainEventsCleared => {
                    if Instant::now() >= next_tick {
                        window.request_redraw();
                    }
                }
                Event::RedrawRequested(_) => {
                    if let Some(panel) = project_panel.as_ref() {
                        let commands = panel.drain_commands().collect::<Vec<_>>();
                        for command in commands {
                            match command {
                                PanelCommand::Set(control) => {
                                    let result = project_extension
                                        .as_mut()
                                        .ok_or_else(|| {
                                            RuntimeDiagnostic::new(
                                                RuntimeDiagnosticCode::ProjectPanelFailed,
                                                "project panel requires a project extension",
                                            )
                                        })
                                        .and_then(|extension| {
                                            extension
                                                .control(&control.name, control.value)
                                                .map(|_| ())
                                        });
                                    if let Err(diagnostic) = result {
                                        eprintln!(
                                            "{}",
                                            serde_json::to_string_pretty(&diagnostic)
                                                .unwrap_or_else(|_| diagnostic.to_string())
                                        );
                                        *control_flow = ControlFlow::Exit;
                                        return;
                                    }
                                }
                                PanelCommand::Reload { generation } => {
                                    let result = project_root
                                        .as_deref()
                                        .ok_or_else(|| {
                                            "the running project has no source root".to_owned()
                                        })
                                        .and_then(|root| {
                                            hot_reload_state(&state, root).map_err(|diagnostic| {
                                                serde_json::to_string_pretty(&diagnostic)
                                                    .unwrap_or_else(|_| diagnostic.to_string())
                                            })
                                        });
                                    let status = match result {
                                        Ok(reloaded) => {
                                            state = reloaded;
                                            tick_interval = Duration::from_nanos(
                                                1_000_000_000 / state.rate_hz as u64,
                                            );
                                            next_tick = Instant::now();
                                            window.request_redraw();
                                            Ok(())
                                        }
                                        Err(message) => Err(message),
                                    };
                                    if let Some(panel) = project_panel.as_mut() {
                                        panel.publish_reload_status(generation, &status);
                                    }
                                }
                                PanelCommand::WindowFrame(frame) => {
                                    panel_frame = Some(frame);
                                    let Some(anchor) = window_frame(&window) else {
                                        continue;
                                    };
                                    let Some(manager) = panel_snap.as_mut() else {
                                        continue;
                                    };
                                    if let SnapUpdate::Move(position) =
                                        manager.observe(anchor, frame)
                                        && let Some(panel) = project_panel.as_mut()
                                    {
                                        panel.set_window_position(position);
                                    }
                                    if let Some(panel) = project_panel.as_mut() {
                                        panel.publish_viewer_frame(anchor);
                                    }
                                }
                                PanelCommand::Quit => *control_flow = ControlFlow::Exit,
                            }
                        }
                    }
                    let mut project_values = if let Some(extension) = project_extension.as_mut() {
                        let size = window.inner_size();
                        match extension.frame(ProjectFrameContextV1 {
                            viewport_width: size.width as f32,
                            viewport_height: size.height as f32,
                            frames_per_second: state.frames_per_second(),
                            gpu_memory_mb: state.gpu_memory_mb(),
                        }) {
                            Ok((values, request_redraw)) => {
                                if request_redraw {
                                    window.request_redraw();
                                }
                                values
                            }
                            Err(diagnostic) => {
                                eprintln!(
                                    "{}",
                                    serde_json::to_string_pretty(&diagnostic)
                                        .unwrap_or_else(|_| diagnostic.to_string())
                                );
                                *control_flow = ControlFlow::Exit;
                                return;
                            }
                        }
                    } else {
                        Vec::new()
                    };
                    state.append_runtime_telemetry(&mut project_values);
                    if let Some(panel) = project_panel.as_mut() {
                        let mut panel_values = project_values.clone();
                        state.append_panel_telemetry(&mut panel_values);
                        panel.publish(&panel_values);
                    }
                    if let Err(diagnostic) = state.draw_tick_with_project_values(&project_values) {
                        eprintln!(
                            "{}",
                            serde_json::to_string_pretty(&diagnostic)
                                .unwrap_or_else(|_| diagnostic.to_string())
                        );
                        *control_flow = ControlFlow::Exit;
                        return;
                    }
                    next_tick += tick_interval;
                    let now = Instant::now();
                    if now.saturating_duration_since(next_tick) > tick_interval * 4 {
                        next_tick = now + tick_interval;
                    }
                }
                _ => {}
            });
        });
    }

    pub fn benchmark(
        validated: ValidatedModuleGraph,
        config: BenchmarkConfig,
    ) -> Result<BenchmarkResult, RuntimeDiagnostic> {
        Self::benchmark_with_project_root(validated, config, None)
    }

    pub fn benchmark_project(
        validated: ValidatedModuleGraph,
        project_root: PathBuf,
        config: BenchmarkConfig,
    ) -> Result<BenchmarkResult, RuntimeDiagnostic> {
        Self::benchmark_with_project_root(validated, config, Some(project_root))
    }

    fn benchmark_with_project_root(
        validated: ValidatedModuleGraph,
        config: BenchmarkConfig,
        project_root: Option<PathBuf>,
    ) -> Result<BenchmarkResult, RuntimeDiagnostic> {
        if config.sample_ticks == 0 && config.sample_seconds.is_none() {
            return Err(RuntimeDiagnostic::new(
                RuntimeDiagnosticCode::UnsupportedGraph,
                "benchmark sample count must be positive",
            ));
        }
        if config.pacing_hz == Some(0) {
            return Err(RuntimeDiagnostic::new(
                RuntimeDiagnosticCode::UnsupportedGraph,
                "benchmark pacing rate must be positive",
            ));
        }
        if let Some(rate_hz) = config.pacing_hz
            && u64::from(config.pacing_lead_microseconds) * u64::from(rate_hz) >= 1_000_000
        {
            return Err(RuntimeDiagnostic::new(
                RuntimeDiagnosticCode::UnsupportedGraph,
                "benchmark pacing lead must be shorter than one tick",
            ));
        }
        let device = Device::system_default().ok_or_else(|| {
            RuntimeDiagnostic::new(
                RuntimeDiagnosticCode::DeviceUnavailable,
                "Metal has no system-default device",
            )
        })?;
        if config.mode == BenchmarkMode::Presented {
            let event_loop = EventLoop::new();
            let window = WindowBuilder::new()
                .with_inner_size(LogicalSize::new(
                    config.viewport_width as f64,
                    config.viewport_height as f64,
                ))
                .with_title("Pqo — Hello Batch Presented Benchmark")
                .build(&event_loop)
                .map_err(|error| {
                    RuntimeDiagnostic::new(
                        RuntimeDiagnosticCode::WindowCreationFailed,
                        error.to_string(),
                    )
                })?;
            let layer = MetalLayer::new();
            layer.set_device(&device);
            layer.set_pixel_format(MTLPixelFormat::BGRA8Unorm);
            layer.set_presents_with_transaction(false);
            layer.set_maximum_drawable_count(3);
            attach_layer(&window, &layer);
            resize_layer(&window, &layer);
            present_benchmark_window(&window);
            let mut state = RuntimeState::new_with_project_root(
                validated,
                device.clone(),
                layer,
                project_root,
            )?;
            return autoreleasepool(|| state.run_benchmark(&device, config));
        }
        let layer = MetalLayer::new();
        layer.set_device(&device);
        layer.set_pixel_format(MTLPixelFormat::BGRA8Unorm);
        let mut state =
            RuntimeState::new_with_project_root(validated, device.clone(), layer, project_root)?;
        state.run_benchmark(&device, config)
    }

    pub fn run_scenario(
        validated: ValidatedModuleGraph,
        scenario_name: &str,
    ) -> Result<ScenarioRunResult, RuntimeDiagnostic> {
        let scenario = validated
            .graph()
            .scenarios
            .iter()
            .find(|scenario| scenario.name == scenario_name)
            .cloned()
            .ok_or_else(|| {
                RuntimeDiagnostic::new(
                    RuntimeDiagnosticCode::UnsupportedGraph,
                    format!("unknown scenario `{scenario_name}`"),
                )
            })?;
        let pqo_core::ScenarioDuration::SimulationTicks(ticks) = scenario.duration;
        let device = Device::system_default().ok_or_else(|| {
            RuntimeDiagnostic::new(
                RuntimeDiagnosticCode::DeviceUnavailable,
                "Metal has no system-default device",
            )
        })?;
        let layer = MetalLayer::new();
        layer.set_device(&device);
        layer.set_pixel_format(MTLPixelFormat::BGRA8Unorm);
        let mut state = RuntimeState::new(validated, device.clone(), layer)?;
        let events = state.execute_scenario(&device, &scenario, ticks)?;
        Ok(ScenarioRunResult {
            scenario: scenario.name,
            executed_ticks: ticks,
            events,
            runtime_fingerprint: state.fingerprint.fingerprint.clone(),
        })
    }
}

struct DirectMetalEncoding {
    fall: PassId,
    bounce: PassId,
    viewport: ViewId,
    position: StreamId,
    velocity: StreamId,
    radius: StreamId,
    restitution: StreamId,
    friction: StreamId,
    color: StreamId,
    gravity: ValueId,
    fixed_dt: ValueId,
    ground_height: ValueId,
}

impl DirectMetalEncoding {
    fn resolve(graph: &pqo_core::ModuleGraph) -> Option<Self> {
        let pass = |name: &str| {
            graph
                .passes
                .iter()
                .find(|item| item.name == name)
                .map(|item| item.id)
        };
        let view = |name: &str| {
            graph
                .views
                .iter()
                .find(|item| item.name == name)
                .map(|item| item.id)
        };
        let stream = |name: &str| {
            graph
                .resources
                .streams
                .iter()
                .find(|item| item.name == name)
                .map(|item| item.id)
        };
        let value = |name: &str| {
            graph
                .resources
                .values
                .iter()
                .find(|item| item.name == name)
                .map(|item| item.id)
        };
        Some(Self {
            fall: pass("fall")?,
            bounce: pass("bounce")?,
            viewport: view("viewport")?,
            position: stream("particles.position")?,
            velocity: stream("particles.velocity")?,
            radius: stream("particles.radius")?,
            restitution: stream("particles.restitution")?,
            friction: stream("particles.friction")?,
            color: stream("particles.color")?,
            gravity: value("world.gravity")?,
            fixed_dt: value("simulation.fixed_dt")?,
            ground_height: value("ground.height")?,
        })
    }
}

struct RuntimeState {
    validated: ValidatedModuleGraph,
    device: Device,
    queue: CommandQueue,
    layer: MetalLayer,
    stream_buffers: Vec<Vec<Buffer>>,
    value_buffers: Vec<Buffer>,
    compute_pipelines: BTreeMap<PassId, ComputePipelineState>,
    render_pipelines: BTreeMap<ViewId, RenderPipelineState>,
    indirect: Option<IndirectSupport>,
    rate_hz: u32,
    fingerprint: RuntimeFingerprint,
    max_in_flight_command_buffers: u32,
    direct_metal: Option<DirectMetalEncoding>,
    pending_pointer_slices: VecDeque<PointerSlice>,
    pending_pointer_orbits: VecDeque<PointerOrbit>,
    pending_pointer_zooms: VecDeque<f32>,
    pending_pointer_drops: VecDeque<PointerDrop>,
    frames_per_second: f32,
    fps_sample_started: Instant,
    presented_frames: u32,
    gpu_timing: Arc<Mutex<GpuTiming>>,
    selected_particle_stream: Option<StreamId>,
    selected_particle_readback: Option<Buffer>,
    selected_particle: Arc<Mutex<Option<f32>>>,
}

#[derive(Default)]
struct GpuTiming {
    frame_time_ms: f32,
    samples: u64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct PointerSlice {
    start: [f32; 2],
    end: [f32; 2],
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct PointerOrbit {
    delta_yaw: f32,
    delta_pitch: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct PointerDrop {
    point: [f32; 2],
}

struct IndirectSupport {
    prepare_compute: ComputePipelineState,
    prepare_draw: ComputePipelineState,
    compute_arguments: BTreeMap<PassId, Buffer>,
    draw_arguments: BTreeMap<ViewId, Buffer>,
}

impl RuntimeState {
    fn new(
        validated: ValidatedModuleGraph,
        device: Device,
        layer: MetalLayer,
    ) -> Result<Self, RuntimeDiagnostic> {
        Self::new_with_project_root(validated, device, layer, None)
    }

    fn new_with_project_root(
        validated: ValidatedModuleGraph,
        device: Device,
        layer: MetalLayer,
        project_root: Option<PathBuf>,
    ) -> Result<Self, RuntimeDiagnostic> {
        if validated.execution_plan().schedules.len() != 1 {
            return Err(RuntimeDiagnostic::new(
                RuntimeDiagnosticCode::UnsupportedGraph,
                "the first Metal slice requires exactly one schedule",
            ));
        }
        let schedule = &validated.execution_plan().schedules[0];
        if schedule
            .resource_versions
            .iter()
            .any(|allocation| allocation.required_versions > 1)
        {
            return Err(RuntimeDiagnostic::new(
                RuntimeDiagnosticCode::UnsupportedGraph,
                "the first Metal slice supports one physical version per stream",
            ));
        }

        let max_in_flight_command_buffers = schedule
            .requested_ticks
            .max(schedule.requested_render_frames)
            .max(1);
        let queue = device
            .new_command_queue_with_max_command_buffer_count(max_in_flight_command_buffers as u64);
        let stream_buffers = allocate_streams(&validated, &device, &queue)?;
        let value_buffers = allocate_values(&validated, &device)?;
        let (compute_pipelines, render_pipelines, indirect, shader_identities, pipeline_identities) =
            build_pipelines(&validated, &device, project_root.as_deref())?;
        let rate_hz = match validated.graph().schedules[0].timing {
            pqo_core::ScheduleTiming::Fixed { rate_hz, .. } => rate_hz,
        };
        let fingerprint =
            make_fingerprint(&validated, &device, shader_identities, pipeline_identities);
        let direct_metal = DirectMetalEncoding::resolve(validated.graph());
        let selected_particle_stream = validated
            .graph()
            .resources
            .streams
            .iter()
            .find(|stream| {
                stream.name == "interaction.selected"
                    && stream.element_type == DataType::Scalar(ScalarType::F32)
                    && stream.capacity == 1
            })
            .map(|stream| stream.id);
        let selected_particle_readback = selected_particle_stream.map(|_| {
            device.new_buffer(
                std::mem::size_of::<f32>() as u64,
                MTLResourceOptions::StorageModeShared,
            )
        });

        Ok(Self {
            validated,
            device,
            queue,
            layer,
            stream_buffers,
            value_buffers,
            compute_pipelines,
            render_pipelines,
            indirect,
            rate_hz,
            fingerprint,
            max_in_flight_command_buffers,
            direct_metal,
            pending_pointer_slices: VecDeque::new(),
            pending_pointer_orbits: VecDeque::new(),
            pending_pointer_zooms: VecDeque::new(),
            pending_pointer_drops: VecDeque::new(),
            frames_per_second: 0.0,
            fps_sample_started: Instant::now(),
            presented_frames: 0,
            gpu_timing: Arc::new(Mutex::new(GpuTiming::default())),
            selected_particle_stream,
            selected_particle_readback,
            selected_particle: Arc::new(Mutex::new(None)),
        })
    }

    fn frames_per_second(&self) -> f32 {
        self.frames_per_second
    }

    fn gpu_memory_mb(&self) -> f32 {
        self.device.current_allocated_size() as f32 / (1024.0 * 1024.0)
    }

    fn gpu_frame_time_ms(&self) -> f32 {
        self.gpu_timing
            .lock()
            .map(|timing| timing.frame_time_ms)
            .unwrap_or_default()
    }

    fn append_runtime_telemetry(&self, values: &mut Vec<(String, f32)>) {
        let gpu_frame_time_ms = self.gpu_frame_time_ms();
        let gpu_budget_ms = 1_000.0 / self.rate_hz.max(1) as f32;
        let gpu_pressure = if gpu_frame_time_ms > 0.0 {
            gpu_frame_time_ms / gpu_budget_ms * 100.0
        } else {
            0.0
        };

        for (name, value) in [
            ("interaction.hud_gpu_frame_ms", gpu_frame_time_ms),
            ("interaction.hud_gpu_budget_ms", gpu_budget_ms),
            ("interaction.hud_gpu_pressure", gpu_pressure),
        ] {
            if !self
                .validated
                .graph()
                .resources
                .values
                .iter()
                .any(|candidate| candidate.name == name)
            {
                continue;
            }
            if let Some((_, existing)) = values.iter_mut().find(|(key, _)| key == name) {
                *existing = value;
            } else {
                values.push((name.to_owned(), value));
            }
        }
    }

    fn append_panel_telemetry(&self, values: &mut Vec<(String, f32)>) {
        if let Ok(selected) = self.selected_particle.lock()
            && let Some(selected) = *selected
        {
            values.push(("interaction.selected".to_owned(), selected));
        }
    }

    fn record_presented_frame(&mut self) {
        self.presented_frames += 1;
        let elapsed = self.fps_sample_started.elapsed();
        if elapsed >= Duration::from_millis(500) {
            self.frames_per_second = self.presented_frames as f32 / elapsed.as_secs_f32();
            self.presented_frames = 0;
            self.fps_sample_started = Instant::now();
        }
    }

    fn queue_pointer_slice(
        &mut self,
        start: (f64, f64),
        end: (f64, f64),
        viewport: PhysicalSize<u32>,
    ) {
        let supports_slicing = self
            .validated
            .execution_plan()
            .intervention_passes
            .iter()
            .any(|pass| self.validated.graph().pass(pass.pass).unwrap().name == "slice_material");
        if !supports_slicing || viewport.width == 0 || viewport.height == 0 {
            return;
        }
        self.pending_pointer_slices.push_back(PointerSlice {
            start: pointer_ndc(start, viewport),
            end: pointer_ndc(end, viewport),
        });
        while self.pending_pointer_slices.len() > 64 {
            self.pending_pointer_slices.pop_front();
        }
    }

    fn pointer_slice_overrides(
        &self,
        slice: PointerSlice,
    ) -> Result<BTreeMap<ValueId, Buffer>, RuntimeDiagnostic> {
        self.f32_value_overrides([
            ("interaction.slice_start_x", slice.start[0]),
            ("interaction.slice_start_y", slice.start[1]),
            ("interaction.slice_end_x", slice.end[0]),
            ("interaction.slice_end_y", slice.end[1]),
        ])
    }

    fn queue_pointer_orbit(&mut self, start: (f64, f64), end: (f64, f64)) {
        self.pending_pointer_orbits.push_back(PointerOrbit {
            delta_yaw: ((end.0 - start.0) * 0.008) as f32,
            delta_pitch: ((end.1 - start.1) * 0.008) as f32,
        });
        while self.pending_pointer_orbits.len() > 64 {
            self.pending_pointer_orbits.pop_front();
        }
    }

    fn queue_pointer_zoom(&mut self, delta: f32) {
        if delta.abs() < f32::EPSILON {
            return;
        }
        self.pending_pointer_zooms.push_back(delta.clamp(-0.5, 0.5));
        while self.pending_pointer_zooms.len() > 64 {
            self.pending_pointer_zooms.pop_front();
        }
    }

    fn queue_pointer_drop(&mut self, point: (f64, f64), viewport: PhysicalSize<u32>) {
        let supports_food = self
            .validated
            .execution_plan()
            .intervention_passes
            .iter()
            .any(|pass| self.validated.graph().pass(pass.pass).unwrap().name == "drop_food");
        if !supports_food || viewport.width == 0 || viewport.height == 0 {
            return;
        }
        self.pending_pointer_drops.push_back(PointerDrop {
            point: pointer_ndc(point, viewport),
        });
        while self.pending_pointer_drops.len() > 32 {
            self.pending_pointer_drops.pop_front();
        }
    }

    fn f32_value_overrides<'a>(
        &self,
        values: impl IntoIterator<Item = (&'a str, f32)>,
    ) -> Result<BTreeMap<ValueId, Buffer>, RuntimeDiagnostic> {
        let mut overrides = BTreeMap::new();
        for (name, value) in values {
            let resource = self
                .validated
                .graph()
                .resources
                .values
                .iter()
                .find(|candidate| candidate.name == name)
                .ok_or_else(|| {
                    RuntimeDiagnostic::new(
                        RuntimeDiagnosticCode::UnsupportedGraph,
                        format!("pointer interaction requires value `{name}`"),
                    )
                })?;
            let bytes = value.to_le_bytes();
            overrides.insert(
                resource.id,
                self.device.new_buffer_with_data(
                    bytes.as_ptr().cast(),
                    bytes.len() as u64,
                    MTLResourceOptions::StorageModeShared,
                ),
            );
        }
        Ok(overrides)
    }

    fn pointer_hits_crystal(
        &self,
        point: (f64, f64),
        viewport: PhysicalSize<u32>,
    ) -> Result<bool, RuntimeDiagnostic> {
        if viewport.width == 0 || viewport.height == 0 {
            return Ok(false);
        }
        let find_pass = |name: &str| {
            self.validated
                .execution_plan()
                .intervention_passes
                .iter()
                .find(|pass| self.validated.graph().pass(pass.pass).unwrap().name == name)
        };
        let clear_pass = find_pass("clear_pointer_pick").ok_or_else(|| {
            RuntimeDiagnostic::new(
                RuntimeDiagnosticCode::UnsupportedGraph,
                "pointer hit-testing requires intervention pass `clear_pointer_pick`",
            )
        })?;
        let pick_pass = find_pass("pick_crystal").ok_or_else(|| {
            RuntimeDiagnostic::new(
                RuntimeDiagnosticCode::UnsupportedGraph,
                "pointer hit-testing requires intervention pass `pick_crystal`",
            )
        })?;
        let pick_stream = self
            .validated
            .graph()
            .resources
            .streams
            .iter()
            .find(|stream| stream.name == "interaction.pick_hit")
            .ok_or_else(|| {
                RuntimeDiagnostic::new(
                    RuntimeDiagnosticCode::UnsupportedGraph,
                    "pointer hit-testing requires stream `interaction.pick_hit`",
                )
            })?
            .id;
        let point = pointer_ndc(point, viewport);
        let overrides = self.f32_value_overrides([
            ("interaction.pick_x", point[0]),
            ("interaction.pick_y", point[1]),
        ])?;
        let readback = self.device.new_buffer(
            std::mem::size_of::<u32>() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let command_buffer = self.queue.new_command_buffer();
        command_buffer.set_label("pqo.pointer-pick");
        self.encode_compute(command_buffer, clear_pass)?;
        self.encode_compute_with_overrides(command_buffer, pick_pass, Some(&overrides))?;
        let blit = command_buffer.new_blit_command_encoder();
        blit.copy_from_buffer(
            &self.stream_buffers[pick_stream.0 as usize][0],
            0,
            &readback,
            0,
            std::mem::size_of::<u32>() as u64,
        );
        blit.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        if command_buffer.status() == MTLCommandBufferStatus::Error {
            return Err(RuntimeDiagnostic::new(
                RuntimeDiagnosticCode::CommandBufferFailed,
                "Metal reported an error while hit-testing the crystal",
            ));
        }
        Ok(unsafe { *(readback.contents().cast::<u32>()) != 0 })
    }

    #[cfg(test)]
    fn draw_tick(&mut self) -> Result<(), RuntimeDiagnostic> {
        self.draw_tick_with_project_values(&[])
    }

    fn draw_tick_with_project_values(
        &mut self,
        project_values: &[(String, f32)],
    ) -> Result<(), RuntimeDiagnostic> {
        let schedule = &self.validated.execution_plan().schedules[0];
        let command_buffer = self.queue.new_command_buffer();
        command_buffer.set_label("pqo.tick");
        let drawable = self.layer.next_drawable();
        let presented_frame = drawable.is_some();
        let mut wait_before_next_tick = false;
        let pending_zooms = self.pending_pointer_zooms.drain(..).collect::<Vec<_>>();
        let pending_orbits = self.pending_pointer_orbits.drain(..).collect::<Vec<_>>();
        let pending_slices = self.pending_pointer_slices.drain(..).collect::<Vec<_>>();
        let pending_drops = self.pending_pointer_drops.drain(..).collect::<Vec<_>>();
        let mut interaction_buffers = Vec::with_capacity(
            pending_zooms.len()
                + pending_orbits.len()
                + pending_slices.len()
                + pending_drops.len()
                + usize::from(!project_values.is_empty()),
        );
        let project_overrides = if project_values.is_empty() {
            None
        } else {
            Some(
                self.f32_value_overrides(
                    project_values
                        .iter()
                        .map(|(name, value)| (name.as_str(), *value)),
                )?,
            )
        };
        if !pending_zooms.is_empty()
            && let Some(zoom_pass) = self
                .validated
                .execution_plan()
                .intervention_passes
                .iter()
                .find(|pass| self.validated.graph().pass(pass.pass).unwrap().name == "zoom_camera")
        {
            for zoom in pending_zooms {
                let overrides = self.f32_value_overrides([("interaction.zoom_delta", zoom)])?;
                self.encode_compute_with_overrides(command_buffer, zoom_pass, Some(&overrides))?;
                interaction_buffers.push(overrides);
            }
        }
        if !pending_orbits.is_empty()
            && let Some(orbit_pass) = self
                .validated
                .execution_plan()
                .intervention_passes
                .iter()
                .find(|pass| self.validated.graph().pass(pass.pass).unwrap().name == "orbit_camera")
        {
            for orbit in pending_orbits {
                let overrides = self.f32_value_overrides([
                    ("interaction.orbit_delta_yaw", orbit.delta_yaw),
                    ("interaction.orbit_delta_pitch", orbit.delta_pitch),
                ])?;
                self.encode_compute_with_overrides(command_buffer, orbit_pass, Some(&overrides))?;
                interaction_buffers.push(overrides);
            }
        }
        if !pending_slices.is_empty()
            && let Some(slice_pass) = self
                .validated
                .execution_plan()
                .intervention_passes
                .iter()
                .find(|pass| {
                    self.validated.graph().pass(pass.pass).unwrap().name == "slice_material"
                })
        {
            for slice in pending_slices {
                let overrides = self.pointer_slice_overrides(slice)?;
                self.encode_compute_with_overrides(command_buffer, slice_pass, Some(&overrides))?;
                interaction_buffers.push(overrides);
            }
        }
        if !pending_drops.is_empty()
            && let Some(drop_pass) = self
                .validated
                .execution_plan()
                .intervention_passes
                .iter()
                .find(|pass| self.validated.graph().pass(pass.pass).unwrap().name == "drop_food")
        {
            for drop in pending_drops {
                let overrides = self.f32_value_overrides([
                    ("interaction.drop_x", drop.point[0]),
                    ("interaction.drop_y", drop.point[1]),
                ])?;
                self.encode_compute_with_overrides(command_buffer, drop_pass, Some(&overrides))?;
                interaction_buffers.push(overrides);
            }
        }
        for item in &schedule.order {
            match item {
                ScheduleItemId::Pass(pass_id) => {
                    let pass = schedule
                        .passes
                        .iter()
                        .find(|pass| pass.pass == *pass_id)
                        .expect("validated plan pass");
                    if project_overrides.is_some() {
                        self.encode_compute_with_overrides(
                            command_buffer,
                            pass,
                            project_overrides.as_ref(),
                        )?;
                    } else {
                        self.encode_compute(command_buffer, pass)?;
                    }
                    wait_before_next_tick |= requires_wait_after(schedule, *item);
                }
                ScheduleItemId::View(view_id) => {
                    if let Some(drawable) = drawable {
                        let view = schedule
                            .views
                            .iter()
                            .find(|view| view.view == *view_id)
                            .expect("validated plan view");
                        self.encode_render(command_buffer, view, drawable.texture())?;
                        command_buffer.present_drawable(drawable);
                        wait_before_next_tick |= requires_wait_after(schedule, *item);
                    }
                }
            }
        }

        if let Some(overrides) = project_overrides {
            interaction_buffers.push(overrides);
        }

        if let (Some(stream), Some(readback)) = (
            self.selected_particle_stream,
            self.selected_particle_readback.as_ref(),
        ) {
            let blit = command_buffer.new_blit_command_encoder();
            blit.copy_from_buffer(
                &self.stream_buffers[stream.0 as usize][0],
                0,
                readback,
                0,
                std::mem::size_of::<f32>() as u64,
            );
            blit.end_encoding();
        }

        let gpu_timing = Arc::clone(&self.gpu_timing);
        let selected_particle = Arc::clone(&self.selected_particle);
        let selected_particle_readback = self.selected_particle_readback.clone();
        let handler = ConcreteBlock::new(move |completed: &metal::CommandBufferRef| {
            if let Some(readback) = selected_particle_readback.as_ref() {
                let value = unsafe { *(readback.contents().cast::<f32>()) };
                if value.is_finite()
                    && let Ok(mut selected) = selected_particle.lock()
                {
                    *selected = Some(value);
                }
            }
            let gpu_start: f64 = unsafe { msg_send![completed, GPUStartTime] };
            let gpu_end: f64 = unsafe { msg_send![completed, GPUEndTime] };
            let frame_time_ms = ((gpu_end - gpu_start) * 1_000.0) as f32;
            if !frame_time_ms.is_finite() || frame_time_ms < 0.0 {
                return;
            }
            if let Ok(mut timing) = gpu_timing.lock() {
                timing.frame_time_ms = if timing.samples == 0 {
                    frame_time_ms
                } else {
                    timing.frame_time_ms + (frame_time_ms - timing.frame_time_ms) * 0.15
                };
                timing.samples += 1;
            }
        })
        .copy();
        command_buffer.add_completed_handler(&handler);
        command_buffer.commit();
        if wait_before_next_tick {
            command_buffer.wait_until_completed();
            if command_buffer.status() == MTLCommandBufferStatus::Error {
                return Err(RuntimeDiagnostic::new(
                    RuntimeDiagnosticCode::CommandBufferFailed,
                    "Metal reported a command-buffer execution error",
                )
                .at("schedules.simulation"));
            }
        }
        if presented_frame {
            self.record_presented_frame();
        }
        drop(interaction_buffers);
        Ok(())
    }

    fn execute_scenario(
        &mut self,
        device: &Device,
        scenario: &pqo_core::ScenarioNode,
        ticks: u64,
    ) -> Result<Vec<ScenarioEventRecord>, RuntimeDiagnostic> {
        self.execute_scenario_range(device, scenario, 0, ticks)
    }

    fn execute_scenario_range(
        &mut self,
        device: &Device,
        scenario: &pqo_core::ScenarioNode,
        start_tick: u64,
        ticks: u64,
    ) -> Result<Vec<ScenarioEventRecord>, RuntimeDiagnostic> {
        let schedule = &self.validated.execution_plan().schedules[0];
        if schedule.schedule != scenario.schedule {
            return Err(RuntimeDiagnostic::new(
                RuntimeDiagnosticCode::UnsupportedGraph,
                "scenario schedule is not the runtime schedule",
            ));
        }
        let mut records = Vec::new();
        let end_tick = start_tick.checked_add(ticks).ok_or_else(|| {
            RuntimeDiagnostic::new(
                RuntimeDiagnosticCode::UnsupportedGraph,
                "scenario tick range exceeds u64",
            )
        })?;
        for tick in start_tick..end_tick {
            let command_buffer = self.queue.new_command_buffer();
            command_buffer.set_label("pqo.scenario.tick");
            let mut override_buffers = Vec::<BTreeMap<ValueId, Buffer>>::new();
            for intervention in scenario
                .interventions
                .iter()
                .filter(|intervention| intervention.tick == tick)
            {
                let pass = self
                    .validated
                    .execution_plan()
                    .intervention_passes
                    .iter()
                    .find(|pass| pass.pass == intervention.pass)
                    .expect("validated intervention pass");
                let mut overrides = BTreeMap::new();
                let mut override_names = Vec::new();
                for override_ in &intervention.value_overrides {
                    let value = self.validated.graph().value(override_.value).unwrap();
                    let bytes = encode_literal(&value.data_type, &override_.literal)?;
                    let buffer = device.new_buffer_with_data(
                        bytes.as_ptr().cast(),
                        bytes.len() as u64,
                        MTLResourceOptions::StorageModeShared,
                    );
                    overrides.insert(override_.value, buffer);
                    override_names.push(value.name.clone());
                }
                override_names.sort();
                override_buffers.push(overrides);
                let current = override_buffers.last().expect("just inserted");
                self.encode_compute_with_overrides(command_buffer, pass, Some(current))?;
                records.push(ScenarioEventRecord {
                    tick,
                    pass: self
                        .validated
                        .graph()
                        .pass(intervention.pass)
                        .unwrap()
                        .name
                        .clone(),
                    value_overrides: override_names,
                });
            }
            for item in &schedule.order {
                if let ScheduleItemId::Pass(pass_id) = item {
                    let pass = schedule
                        .passes
                        .iter()
                        .find(|pass| pass.pass == *pass_id)
                        .expect("validated schedule pass");
                    self.encode_compute(command_buffer, pass)?;
                }
            }
            command_buffer.commit();
            command_buffer.wait_until_completed();
            if command_buffer.status() == MTLCommandBufferStatus::Error {
                return Err(RuntimeDiagnostic::new(
                    RuntimeDiagnosticCode::CommandBufferFailed,
                    format!("scenario tick {tick} failed"),
                ));
            }
        }
        Ok(records)
    }

    #[cfg(test)]
    fn fork_checkpoint(&self, device: &Device) -> Result<Self, RuntimeDiagnostic> {
        let layer = MetalLayer::new();
        layer.set_device(device);
        layer.set_pixel_format(MTLPixelFormat::BGRA8Unorm);
        let fork = Self::new(self.validated.clone(), device.clone(), layer)?;
        let command_buffer = fork.queue.new_command_buffer();
        command_buffer.set_label("pqo.checkpoint.fork");
        let blit = command_buffer.new_blit_command_encoder();
        for (source_versions, destination_versions) in
            self.stream_buffers.iter().zip(&fork.stream_buffers)
        {
            for (source, destination) in source_versions.iter().zip(destination_versions) {
                let length = source.length();
                debug_assert_eq!(length, destination.length());
                blit.copy_from_buffer(source, 0, destination, 0, length);
            }
        }
        blit.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        if command_buffer.status() == MTLCommandBufferStatus::Error {
            return Err(RuntimeDiagnostic::new(
                RuntimeDiagnosticCode::CommandBufferFailed,
                "Metal failed to fork the committed checkpoint",
            )
            .at("runtime.checkpoint"));
        }
        Ok(fork)
    }

    fn run_benchmark(
        &mut self,
        device: &Device,
        config: BenchmarkConfig,
    ) -> Result<BenchmarkResult, RuntimeDiagnostic> {
        if config.pacing_hz.is_some() {
            let status = unsafe { pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0) };
            if status != 0 {
                return Err(RuntimeDiagnostic::new(
                    RuntimeDiagnosticCode::UnsupportedGraph,
                    format!("failed to request interactive pacing QoS: errno {status}"),
                ));
            }
        }
        if config.mode == BenchmarkMode::Presented {
            return self.run_presented_benchmark(config);
        }
        let render_target = (config.mode == BenchmarkMode::Rendered).then(|| {
            let descriptor = TextureDescriptor::new();
            descriptor.set_width(config.viewport_width as u64);
            descriptor.set_height(config.viewport_height as u64);
            descriptor.set_pixel_format(MTLPixelFormat::BGRA8Unorm);
            descriptor.set_storage_mode(MTLStorageMode::Private);
            descriptor.set_usage(MTLTextureUsage::RenderTarget);
            device.new_texture(&descriptor)
        });

        let (warmup_receiver, warmup_ticks) = self.submit_benchmark_phase(
            render_target.as_deref(),
            config.warmup_ticks,
            config.warmup_seconds,
            &config,
        )?;
        self.drain_benchmark_ticks(warmup_receiver, warmup_ticks)?;

        let sample_start = Instant::now();
        let (sample_receiver, sample_ticks) = self.submit_benchmark_phase(
            render_target.as_deref(),
            config.sample_ticks,
            config.sample_seconds,
            &config,
        )?;
        let timings = self.drain_benchmark_ticks(sample_receiver, sample_ticks)?;
        let sample_wall_time_seconds = sample_start.elapsed().as_secs_f64();
        let gpu_samples = timings
            .iter()
            .map(|timing| timing.gpu_ms)
            .collect::<Vec<_>>();
        let cpu_samples = timings
            .iter()
            .map(|timing| timing.cpu_orchestration_ms)
            .collect::<Vec<_>>();
        let latency_samples = timings
            .iter()
            .map(|timing| timing.end_to_end_tick_ms)
            .collect::<Vec<_>>();

        let gpu_ms = summarize(&gpu_samples);
        let cpu_orchestration_ms = summarize(&cpu_samples);
        let end_to_end_tick_ms = summarize(&latency_samples);
        let pacing = config.pacing_hz.map(|target_hz| {
            let deadline_misses = timings
                .iter()
                .filter(|timing| timing.deadline_lateness_ms > 0.0)
                .count() as u32;
            PacingResult {
                target_hz,
                tick_budget_ms: 1_000.0 / target_hz as f64,
                submission_lead_ms: config.pacing_lead_microseconds as f64 / 1_000.0,
                deadline_misses,
                deadline_miss_rate: deadline_misses as f64 / sample_ticks as f64,
                maximum_lateness_ms: timings
                    .iter()
                    .map(|timing| timing.deadline_lateness_ms)
                    .fold(0.0, f64::max),
            }
        });
        let particle_count = benchmark_dispatch_count(
            self.validated.graph(),
            &self.validated.execution_plan().schedules[0],
        )?;
        Ok(BenchmarkResult {
            experiment: self.validated.graph().name.clone(),
            particle_count,
            mode: config.mode,
            runner: config.runner,
            viewport: (config.mode != BenchmarkMode::Headless).then_some(ViewportSize {
                width: config.viewport_width,
                height: config.viewport_height,
            }),
            warmup_ticks,
            sample_ticks,
            requested_warmup_seconds: config.warmup_seconds,
            requested_sample_seconds: config.sample_seconds,
            gpu_p95_below_8_33_ms: gpu_ms.p95 < 8.33,
            gpu_ms,
            cpu_orchestration_ms,
            end_to_end_tick_ms,
            sample_wall_time_seconds,
            submitted_ticks_per_second: sample_ticks as f64 / sample_wall_time_seconds,
            synchronized_each_tick: false,
            max_in_flight_command_buffers: self.max_in_flight_command_buffers,
            pacing,
            presentation: None,
            resources: self.resource_metrics(),
            runtime: self.fingerprint.clone(),
        })
    }

    fn run_presented_benchmark(
        &self,
        config: BenchmarkConfig,
    ) -> Result<BenchmarkResult, RuntimeDiagnostic> {
        if config.pacing_hz.is_some_and(|rate| rate != self.rate_hz) {
            return Err(RuntimeDiagnostic::new(
                RuntimeDiagnosticCode::UnsupportedGraph,
                format!(
                    "presented benchmark simulation rate is fixed by the graph at {} Hz",
                    self.rate_hz
                ),
            ));
        }
        let display_link = DisplayLinkDriver::start(&self.layer)?;
        let warmup = self.submit_presented_phase(
            &display_link,
            config.warmup_ticks,
            config.warmup_seconds,
            &config,
        )?;
        self.drain_benchmark_ticks(warmup.simulation, warmup.simulation_ticks)?;
        self.drain_benchmark_ticks(warmup.presentation, warmup.presented_frames)?;
        display_link.discard_pending();

        let sample_start = Instant::now();
        let sample = self.submit_presented_phase(
            &display_link,
            config.sample_ticks,
            config.sample_seconds,
            &config,
        )?;
        let simulation = self.drain_benchmark_ticks(sample.simulation, sample.simulation_ticks)?;
        let presentation =
            self.drain_benchmark_ticks(sample.presentation, sample.presented_frames)?;
        let sample_wall_time_seconds = sample_start.elapsed().as_secs_f64();

        let gpu_ms = summarize_field(&simulation, |timing| timing.gpu_ms)?;
        let cpu_orchestration_ms =
            summarize_field(&simulation, |timing| timing.cpu_orchestration_ms)?;
        let end_to_end_tick_ms = summarize_field(&simulation, |timing| timing.end_to_end_tick_ms)?;
        let simulation_deadline_misses = simulation
            .iter()
            .filter(|timing| timing.deadline_lateness_ms > 0.0)
            .count() as u32;
        let render_gpu_ms = summarize_field(&presentation, |timing| timing.gpu_ms)?;
        let render_cpu_orchestration_ms =
            summarize_field(&presentation, |timing| timing.cpu_orchestration_ms)?;
        let render_end_to_end_ms =
            summarize_field(&presentation, |timing| timing.end_to_end_tick_ms)?;
        let display_target_lead_ms =
            summarize_optional_field(&presentation, |timing| timing.display_target_lead_ms)?;
        let presentation_lateness_ms =
            summarize_optional_field(&presentation, |timing| timing.presentation_lateness_ms)?;
        let gpu_deadline_misses = presentation
            .iter()
            .filter(|timing| timing.gpu_deadline_missed)
            .count() as u32;
        let presentation_deadline_misses = presentation
            .iter()
            .filter(|timing| timing.presentation_deadline_missed)
            .count() as u32;
        let skipped_presentations = presentation
            .iter()
            .filter(|timing| timing.presentation_skipped)
            .count() as u32;
        let particle_count = benchmark_dispatch_count(
            self.validated.graph(),
            &self.validated.execution_plan().schedules[0],
        )?;
        Ok(BenchmarkResult {
            experiment: self.validated.graph().name.clone(),
            particle_count,
            mode: config.mode,
            runner: config.runner,
            viewport: Some(ViewportSize {
                width: config.viewport_width,
                height: config.viewport_height,
            }),
            warmup_ticks: warmup.simulation_ticks,
            sample_ticks: sample.simulation_ticks,
            requested_warmup_seconds: config.warmup_seconds,
            requested_sample_seconds: config.sample_seconds,
            gpu_p95_below_8_33_ms: gpu_ms.p95.max(render_gpu_ms.p95) < 8.33,
            gpu_ms,
            cpu_orchestration_ms,
            end_to_end_tick_ms,
            sample_wall_time_seconds,
            submitted_ticks_per_second: sample.simulation_ticks as f64 / sample_wall_time_seconds,
            synchronized_each_tick: false,
            max_in_flight_command_buffers: self.max_in_flight_command_buffers,
            pacing: Some(PacingResult {
                target_hz: self.rate_hz,
                tick_budget_ms: 1_000.0 / self.rate_hz as f64,
                submission_lead_ms: config.pacing_lead_microseconds as f64 / 1_000.0,
                deadline_misses: simulation_deadline_misses,
                deadline_miss_rate: simulation_deadline_misses as f64
                    / sample.simulation_ticks as f64,
                maximum_lateness_ms: simulation
                    .iter()
                    .map(|timing| timing.deadline_lateness_ms)
                    .fold(0.0, f64::max),
            }),
            presentation: Some(PresentationResult {
                display_link_driven: true,
                presented_frames: sample.presented_frames,
                drawable_starvation_events: sample.drawable_starvation_events,
                gpu_deadline_misses,
                presentation_deadline_misses,
                skipped_presentations,
                lateness_tolerance_ms: 0.5,
                render_gpu_ms,
                render_cpu_orchestration_ms,
                render_end_to_end_ms,
                display_target_lead_ms,
                presentation_lateness_ms,
            }),
            resources: self.resource_metrics(),
            runtime: self.fingerprint.clone(),
        })
    }

    fn submit_presented_phase(
        &self,
        display_link: &DisplayLinkDriver,
        ticks: u32,
        seconds: Option<u32>,
        config: &BenchmarkConfig,
    ) -> Result<PresentedPhase, RuntimeDiagnostic> {
        let (simulation_sender, simulation) = mpsc::channel();
        let (presentation_sender, presentation) = mpsc::channel();
        let mut simulation_ticks = 0_u32;
        let mut presented_frames = 0_u32;
        let mut drawable_starvation_events = 0_u32;
        let mut starvation_active = false;
        let mut last_display_update = Instant::now();
        let phase_start = Instant::now();
        loop {
            let phase_complete = if let Some(seconds) = seconds {
                pacing_offset(simulation_ticks, self.rate_hz) >= Duration::from_secs(seconds as u64)
            } else {
                simulation_ticks >= ticks
            };
            if phase_complete {
                break;
            }

            let nominal_admission = phase_start + pacing_offset(simulation_ticks, self.rate_hz);
            let lead = Duration::from_micros(u64::from(config.pacing_lead_microseconds));
            let admission = nominal_admission.checked_sub(lead).unwrap_or(phase_start);
            let now = Instant::now();
            if now >= admission {
                let deadline = phase_start + pacing_offset(simulation_ticks + 1, self.rate_hz);
                self.submit_benchmark_tick(
                    None,
                    simulation_sender.clone(),
                    config.runner,
                    Some(deadline),
                )?;
                simulation_ticks += 1;
                continue;
            }

            let timeout = admission
                .saturating_duration_since(now)
                .min(Duration::from_millis(20));
            if let Some(update) = display_link.next(timeout)? {
                last_display_update = Instant::now();
                starvation_active = false;
                self.submit_presented_frame(update, presentation_sender.clone(), config.runner)?;
                presented_frames += 1;
            } else if last_display_update.elapsed() > Duration::from_millis(25)
                && !starvation_active
            {
                drawable_starvation_events += 1;
                starvation_active = true;
            }
        }
        drop(simulation_sender);
        drop(presentation_sender);
        Ok(PresentedPhase {
            simulation,
            simulation_ticks,
            presentation,
            presented_frames,
            drawable_starvation_events,
        })
    }

    fn submit_benchmark_phase(
        &self,
        render_target: Option<&metal::TextureRef>,
        ticks: u32,
        seconds: Option<u32>,
        config: &BenchmarkConfig,
    ) -> Result<(Receiver<RawTickTiming>, u32), RuntimeDiagnostic> {
        let (sender, receiver) = mpsc::channel();
        let mut submitted = 0_u32;
        let phase_start = Instant::now();
        loop {
            if let Some(seconds) = seconds {
                let duration = Duration::from_secs(seconds as u64);
                let phase_complete = config.pacing_hz.map_or_else(
                    || submitted > 0 && phase_start.elapsed() >= duration,
                    |hz| pacing_offset(submitted, hz) >= duration,
                );
                if phase_complete {
                    break;
                }
            } else if submitted >= ticks {
                break;
            }
            let deadline = config.pacing_hz.map(|hz| {
                let nominal_admission = phase_start + pacing_offset(submitted, hz);
                let lead = Duration::from_micros(u64::from(config.pacing_lead_microseconds));
                let admission = nominal_admission.checked_sub(lead).unwrap_or(phase_start);
                wait_until(admission);
                phase_start + pacing_offset(submitted + 1, hz)
            });
            self.submit_benchmark_tick(render_target, sender.clone(), config.runner, deadline)?;
            submitted = submitted.checked_add(1).ok_or_else(|| {
                RuntimeDiagnostic::new(
                    RuntimeDiagnosticCode::UnsupportedGraph,
                    "benchmark phase submitted more ticks than can be reported",
                )
            })?;
        }
        drop(sender);
        Ok((receiver, submitted))
    }

    fn submit_benchmark_tick(
        &self,
        render_target: Option<&metal::TextureRef>,
        sender: Sender<RawTickTiming>,
        runner: BenchmarkRunner,
        deadline: Option<Instant>,
    ) -> Result<(), RuntimeDiagnostic> {
        let schedule = &self.validated.execution_plan().schedules[0];
        let cpu_start = Instant::now();
        let command_buffer = self.queue.new_command_buffer();
        command_buffer.set_label("pqo.benchmark.tick");

        match runner {
            BenchmarkRunner::PqoPlan => {
                for item in &schedule.order {
                    match item {
                        ScheduleItemId::Pass(pass_id) => {
                            let pass = schedule
                                .passes
                                .iter()
                                .find(|pass| pass.pass == *pass_id)
                                .expect("validated plan pass");
                            self.encode_compute(command_buffer, pass)?;
                        }
                        ScheduleItemId::View(view_id) => {
                            if let Some(texture) = render_target {
                                let view = schedule
                                    .views
                                    .iter()
                                    .find(|view| view.view == *view_id)
                                    .expect("validated plan view");
                                self.encode_render(command_buffer, view, texture)?;
                            }
                        }
                    }
                }
            }
            BenchmarkRunner::DirectMetalEncoding => {
                self.encode_direct_metal(command_buffer, render_target)?;
            }
        }

        let cpu_orchestration_bits = Arc::new(AtomicU64::new(u64::MAX));
        let callback_cpu_bits = Arc::clone(&cpu_orchestration_bits);
        let callback_start = cpu_start;
        let handler = ConcreteBlock::new(move |completed: &metal::CommandBufferRef| {
            let gpu_start: f64 = unsafe { msg_send![completed, GPUStartTime] };
            let gpu_end: f64 = unsafe { msg_send![completed, GPUEndTime] };
            let mut cpu_bits = callback_cpu_bits.load(Ordering::Acquire);
            while cpu_bits == u64::MAX {
                std::hint::spin_loop();
                cpu_bits = callback_cpu_bits.load(Ordering::Acquire);
            }
            let _ = sender.send(RawTickTiming {
                status: completed.status(),
                gpu_start,
                gpu_end,
                cpu_orchestration_ms: f64::from_bits(cpu_bits),
                end_to_end_tick_ms: callback_start.elapsed().as_secs_f64() * 1_000.0,
                deadline_lateness_ms: deadline
                    .and_then(|deadline| Instant::now().checked_duration_since(deadline))
                    .map_or(0.0, |lateness| lateness.as_secs_f64() * 1_000.0),
                presentation: None,
            });
        })
        .copy();
        command_buffer.add_completed_handler(&handler);
        command_buffer.commit();
        let cpu_orchestration_ms = cpu_start.elapsed().as_secs_f64() * 1_000.0;
        cpu_orchestration_bits.store(cpu_orchestration_ms.to_bits(), Ordering::Release);
        Ok(())
    }

    fn submit_presented_frame(
        &self,
        update: DisplayUpdate,
        sender: Sender<RawTickTiming>,
        runner: BenchmarkRunner,
    ) -> Result<(), RuntimeDiagnostic> {
        let schedule = &self.validated.execution_plan().schedules[0];
        let cpu_start = Instant::now();
        let command_buffer = self.queue.new_command_buffer();
        command_buffer.set_label("pqo.benchmark.presented-frame");
        match runner {
            BenchmarkRunner::PqoPlan => {
                for item in &schedule.order {
                    if let ScheduleItemId::View(view_id) = item {
                        let view = schedule
                            .views
                            .iter()
                            .find(|view| view.view == *view_id)
                            .expect("validated plan view");
                        self.encode_render(command_buffer, view, update.drawable.texture())?;
                    }
                }
            }
            BenchmarkRunner::DirectMetalEncoding => {
                self.encode_direct_render(command_buffer, update.drawable.texture())?;
            }
        }

        let (presented_sender, presented_time) = mpsc::channel();
        let presented_handler = ConcreteBlock::new(move |drawable: &metal::DrawableRef| {
            let _ = presented_sender.send(drawable.presented_time());
        })
        .copy();
        update.drawable.add_presented_handler(&presented_handler);
        command_buffer.present_drawable(&update.drawable);

        let cpu_orchestration_bits = Arc::new(AtomicU64::new(u64::MAX));
        let callback_cpu_bits = Arc::clone(&cpu_orchestration_bits);
        let callback_start = cpu_start;
        let presentation = Arc::new(Mutex::new(Some(PendingPresentation {
            presented_time,
            target_timestamp: update.target_timestamp,
            target_presentation_timestamp: update.target_presentation_timestamp,
        })));
        let callback_presentation = Arc::clone(&presentation);
        let handler = ConcreteBlock::new(move |completed: &metal::CommandBufferRef| {
            let gpu_start: f64 = unsafe { msg_send![completed, GPUStartTime] };
            let gpu_end: f64 = unsafe { msg_send![completed, GPUEndTime] };
            let mut cpu_bits = callback_cpu_bits.load(Ordering::Acquire);
            while cpu_bits == u64::MAX {
                std::hint::spin_loop();
                cpu_bits = callback_cpu_bits.load(Ordering::Acquire);
            }
            let _ = sender.send(RawTickTiming {
                status: completed.status(),
                gpu_start,
                gpu_end,
                cpu_orchestration_ms: f64::from_bits(cpu_bits),
                end_to_end_tick_ms: callback_start.elapsed().as_secs_f64() * 1_000.0,
                deadline_lateness_ms: 0.0,
                presentation: callback_presentation
                    .lock()
                    .expect("presentation callback state")
                    .take(),
            });
        })
        .copy();
        command_buffer.add_completed_handler(&handler);
        command_buffer.commit();
        let cpu_orchestration_ms = cpu_start.elapsed().as_secs_f64() * 1_000.0;
        cpu_orchestration_bits.store(cpu_orchestration_ms.to_bits(), Ordering::Release);
        Ok(())
    }

    fn drain_benchmark_ticks(
        &self,
        receiver: Receiver<RawTickTiming>,
        count: u32,
    ) -> Result<Vec<TickTiming>, RuntimeDiagnostic> {
        (0..count)
            .map(|_| {
                receiver
                    .recv()
                    .map_err(|_| {
                        RuntimeDiagnostic::new(
                            RuntimeDiagnosticCode::CommandBufferFailed,
                            "Metal stopped asynchronous benchmark timestamp delivery",
                        )
                    })?
                    .finish()
            })
            .collect()
    }

    fn encode_compute(
        &self,
        command_buffer: &metal::CommandBufferRef,
        pass: &PlannedPass,
    ) -> Result<(), RuntimeDiagnostic> {
        self.encode_compute_with_overrides(command_buffer, pass, None)
    }

    fn encode_compute_with_overrides(
        &self,
        command_buffer: &metal::CommandBufferRef,
        pass: &PlannedPass,
        value_overrides: Option<&BTreeMap<ValueId, Buffer>>,
    ) -> Result<(), RuntimeDiagnostic> {
        let pipeline = &self.compute_pipelines[&pass.pass];
        let dynamic_count = match pass.dispatch {
            DispatchDomain::OverStream(stream) => {
                match self.validated.graph().stream(stream).unwrap().length {
                    StreamLength::Dynamic(count) => Some(count),
                    StreamLength::Fixed(_) => None,
                }
            }
            DispatchDomain::Fixed(_) => None,
        };
        let width = match pass.threads_per_threadgroup {
            Some(requested)
                if u64::from(requested) <= pipeline.max_total_threads_per_threadgroup() =>
            {
                u64::from(requested)
            }
            Some(requested) => {
                return Err(RuntimeDiagnostic::new(
                    RuntimeDiagnosticCode::UnsupportedGraph,
                    format!(
                        "pass requests {requested} threads per threadgroup but its Metal pipeline supports at most {}",
                        pipeline.max_total_threads_per_threadgroup()
                    ),
                ));
            }
            None => pipeline
                .thread_execution_width()
                .min(pipeline.max_total_threads_per_threadgroup())
                .max(1),
        };
        if let Some(count) = dynamic_count {
            let indirect = self.indirect.as_ref().ok_or_else(|| {
                RuntimeDiagnostic::new(
                    RuntimeDiagnosticCode::UnsupportedGraph,
                    "dynamic dispatch has no indirect lowering support",
                )
            })?;
            let arguments = &indirect.compute_arguments[&pass.pass];
            let prepare = command_buffer.new_compute_command_encoder();
            prepare.set_compute_pipeline_state(&indirect.prepare_compute);
            prepare.set_buffer(0, Some(&self.stream_buffers[count.0 as usize][0]), 0);
            prepare.set_buffer(1, Some(arguments), 0);
            let width_u32 = width as u32;
            prepare.set_bytes(
                2,
                std::mem::size_of::<u32>() as u64,
                (&width_u32 as *const u32).cast(),
            );
            let maximum_count = match pass.dispatch {
                DispatchDomain::OverStream(stream) => {
                    self.validated.graph().stream(stream).unwrap().capacity
                }
                DispatchDomain::Fixed(_) => unreachable!(),
            };
            prepare.set_bytes(
                3,
                std::mem::size_of::<u32>() as u64,
                (&maximum_count as *const u32).cast(),
            );
            prepare.dispatch_threads(MTLSize::new(1, 1, 1), MTLSize::new(1, 1, 1));
            prepare.end_encoding();
        }

        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        for (index, slot) in pass.abi.binding_order.iter().enumerate() {
            let binding = pass
                .bindings
                .iter()
                .find(|binding| binding.slot == *slot)
                .expect("validated binding");
            match binding.resource {
                ResourceId::Stream(stream) => {
                    encoder.set_buffer(
                        index as u64,
                        Some(&self.stream_buffers[stream.0 as usize][0]),
                        0,
                    );
                }
                ResourceId::Value(value) => {
                    let buffer = value_overrides
                        .and_then(|overrides| overrides.get(&value))
                        .unwrap_or(&self.value_buffers[value.0 as usize]);
                    encoder.set_buffer(index as u64, Some(buffer), 0);
                }
            }
        }
        if dynamic_count.is_some() {
            let arguments = &self
                .indirect
                .as_ref()
                .expect("dynamic dispatch support")
                .compute_arguments[&pass.pass];
            encoder.dispatch_thread_groups_indirect(arguments, 0, MTLSize::new(width, 1, 1));
        } else {
            let count = dispatch_count(self.validated.graph(), &pass.dispatch)?;
            encoder.dispatch_threads(MTLSize::new(count, 1, 1), MTLSize::new(width, 1, 1));
        }
        encoder.end_encoding();
        Ok(())
    }

    fn encode_render(
        &self,
        command_buffer: &metal::CommandBufferRef,
        view: &PlannedView,
        texture: &metal::TextureRef,
    ) -> Result<(), RuntimeDiagnostic> {
        let dynamic_count = dynamic_count_for_view(self.validated.graph(), view)?;
        if let Some(count) = dynamic_count {
            let indirect = self.indirect.as_ref().ok_or_else(|| {
                RuntimeDiagnostic::new(
                    RuntimeDiagnosticCode::UnsupportedGraph,
                    "dynamic rendering has no indirect lowering support",
                )
            })?;
            let arguments = &indirect.draw_arguments[&view.view];
            let prepare = command_buffer.new_compute_command_encoder();
            prepare.set_compute_pipeline_state(&indirect.prepare_draw);
            prepare.set_buffer(0, Some(&self.stream_buffers[count.0 as usize][0]), 0);
            prepare.set_buffer(1, Some(arguments), 0);
            let maximum_count = view
                .reads
                .first()
                .map(|read| self.validated.graph().stream(read.stream).unwrap().capacity)
                .unwrap_or(0);
            prepare.set_bytes(
                2,
                std::mem::size_of::<u32>() as u64,
                (&maximum_count as *const u32).cast(),
            );
            prepare.dispatch_threads(MTLSize::new(1, 1, 1), MTLSize::new(1, 1, 1));
            prepare.end_encoding();
        }

        let descriptor = RenderPassDescriptor::new();
        let attachment = descriptor.color_attachments().object_at(0).unwrap();
        attachment.set_texture(Some(texture));
        attachment.set_load_action(MTLLoadAction::Clear);
        let clear = match self.validated.graph().name.as_str() {
            "hello_worm" => MTLClearColor::new(0.008, 0.018, 0.014, 1.0),
            "neon_flock" | "quantum_field" => MTLClearColor::new(0.003, 0.005, 0.015, 1.0),
            _ => MTLClearColor::new(0.025, 0.03, 0.055, 1.0),
        };
        attachment.set_clear_color(clear);
        attachment.set_store_action(MTLStoreAction::Store);

        let encoder = command_buffer.new_render_command_encoder(descriptor);
        encoder.set_render_pipeline_state(&self.render_pipelines[&view.view]);
        for (index, read) in view.reads.iter().enumerate() {
            encoder.set_vertex_buffer(
                index as u64,
                Some(&self.stream_buffers[read.stream.0 as usize][0]),
                0,
            );
        }
        if dynamic_count.is_some() {
            let arguments = &self
                .indirect
                .as_ref()
                .expect("dynamic rendering support")
                .draw_arguments[&view.view];
            encoder.draw_primitives_indirect(MTLPrimitiveType::Triangle, arguments, 0);
        } else {
            let instances = view_instance_count(self.validated.graph(), view)?;
            encoder.draw_primitives_instanced(MTLPrimitiveType::Triangle, 0, 6, instances);
        }
        encoder.end_encoding();
        Ok(())
    }

    fn encode_direct_metal(
        &self,
        command_buffer: &metal::CommandBufferRef,
        render_target: Option<&metal::TextureRef>,
    ) -> Result<(), RuntimeDiagnostic> {
        let direct = self.direct_metal.as_ref().ok_or_else(|| {
            RuntimeDiagnostic::new(
                RuntimeDiagnosticCode::UnsupportedGraph,
                "direct-Metal encoding is available only for the Hello Batch resource layout",
            )
        })?;
        let count = dispatch_count(
            self.validated.graph(),
            &DispatchDomain::OverStream(direct.position),
        )?;

        let fall_pipeline = &self.compute_pipelines[&direct.fall];
        let fall = command_buffer.new_compute_command_encoder();
        fall.set_compute_pipeline_state(fall_pipeline);
        fall.set_buffer(
            0,
            Some(&self.stream_buffers[direct.position.0 as usize][0]),
            0,
        );
        fall.set_buffer(
            1,
            Some(&self.stream_buffers[direct.velocity.0 as usize][0]),
            0,
        );
        fall.set_buffer(2, Some(&self.value_buffers[direct.gravity.0 as usize]), 0);
        fall.set_buffer(3, Some(&self.value_buffers[direct.fixed_dt.0 as usize]), 0);
        let width = fall_pipeline
            .thread_execution_width()
            .min(fall_pipeline.max_total_threads_per_threadgroup())
            .max(1);
        fall.dispatch_threads(MTLSize::new(count, 1, 1), MTLSize::new(width, 1, 1));
        fall.end_encoding();

        let bounce_pipeline = &self.compute_pipelines[&direct.bounce];
        let bounce = command_buffer.new_compute_command_encoder();
        bounce.set_compute_pipeline_state(bounce_pipeline);
        for (index, stream) in [
            direct.position,
            direct.velocity,
            direct.radius,
            direct.restitution,
            direct.friction,
        ]
        .iter()
        .enumerate()
        {
            bounce.set_buffer(
                index as u64,
                Some(&self.stream_buffers[stream.0 as usize][0]),
                0,
            );
        }
        bounce.set_buffer(
            5,
            Some(&self.value_buffers[direct.ground_height.0 as usize]),
            0,
        );
        let width = bounce_pipeline
            .thread_execution_width()
            .min(bounce_pipeline.max_total_threads_per_threadgroup())
            .max(1);
        bounce.dispatch_threads(MTLSize::new(count, 1, 1), MTLSize::new(width, 1, 1));
        bounce.end_encoding();

        if let Some(texture) = render_target {
            let descriptor = RenderPassDescriptor::new();
            let attachment = descriptor.color_attachments().object_at(0).unwrap();
            attachment.set_texture(Some(texture));
            attachment.set_load_action(MTLLoadAction::Clear);
            attachment.set_clear_color(MTLClearColor::new(0.025, 0.03, 0.055, 1.0));
            attachment.set_store_action(MTLStoreAction::Store);
            let render = command_buffer.new_render_command_encoder(descriptor);
            render.set_render_pipeline_state(&self.render_pipelines[&direct.viewport]);
            for (index, stream) in [direct.color, direct.position, direct.radius]
                .iter()
                .enumerate()
            {
                render.set_vertex_buffer(
                    index as u64,
                    Some(&self.stream_buffers[stream.0 as usize][0]),
                    0,
                );
            }
            render.draw_primitives_instanced(MTLPrimitiveType::Triangle, 0, 6, count);
            render.end_encoding();
        }
        Ok(())
    }

    fn encode_direct_render(
        &self,
        command_buffer: &metal::CommandBufferRef,
        texture: &metal::TextureRef,
    ) -> Result<(), RuntimeDiagnostic> {
        let direct = self.direct_metal.as_ref().ok_or_else(|| {
            RuntimeDiagnostic::new(
                RuntimeDiagnosticCode::UnsupportedGraph,
                "direct-Metal encoding is available only for the Hello Batch resource layout",
            )
        })?;
        let count = dispatch_count(
            self.validated.graph(),
            &DispatchDomain::OverStream(direct.position),
        )?;
        let descriptor = RenderPassDescriptor::new();
        let attachment = descriptor.color_attachments().object_at(0).unwrap();
        attachment.set_texture(Some(texture));
        attachment.set_load_action(MTLLoadAction::Clear);
        attachment.set_clear_color(MTLClearColor::new(0.025, 0.03, 0.055, 1.0));
        attachment.set_store_action(MTLStoreAction::Store);
        let render = command_buffer.new_render_command_encoder(descriptor);
        render.set_render_pipeline_state(&self.render_pipelines[&direct.viewport]);
        for (index, stream) in [direct.color, direct.position, direct.radius]
            .iter()
            .enumerate()
        {
            render.set_vertex_buffer(
                index as u64,
                Some(&self.stream_buffers[stream.0 as usize][0]),
                0,
            );
        }
        render.draw_primitives_instanced(MTLPrimitiveType::Triangle, 0, 6, count);
        render.end_encoding();
        Ok(())
    }

    fn resource_metrics(&self) -> ResourceMetrics {
        ResourceMetrics {
            gpu_stream_buffer_bytes: self
                .stream_buffers
                .iter()
                .flatten()
                .map(|buffer| buffer.length())
                .sum(),
            gpu_value_buffer_bytes: self
                .value_buffers
                .iter()
                .map(|buffer| buffer.length())
                .sum(),
            gpu_indirect_buffer_bytes: self.indirect.as_ref().map_or(0, |indirect| {
                indirect
                    .compute_arguments
                    .values()
                    .chain(indirect.draw_arguments.values())
                    .map(|buffer| buffer.length())
                    .sum()
            }),
            initialization_application_blits: self
                .stream_buffers
                .iter()
                .map(|versions| versions.len() as u64)
                .sum(),
            steady_state_application_copies_per_tick: 0,
            steady_state_application_blits_per_tick: 0,
            steady_state_heap_allocations_per_tick: None,
            peak_resident_set_bytes: peak_resident_set_bytes(),
        }
    }
}

struct PresentedPhase {
    simulation: Receiver<RawTickTiming>,
    simulation_ticks: u32,
    presentation: Receiver<RawTickTiming>,
    presented_frames: u32,
    drawable_starvation_events: u32,
}

struct TickTiming {
    gpu_ms: f64,
    cpu_orchestration_ms: f64,
    end_to_end_tick_ms: f64,
    deadline_lateness_ms: f64,
    gpu_deadline_missed: bool,
    presentation_lateness_ms: Option<f64>,
    presentation_deadline_missed: bool,
    presentation_skipped: bool,
    display_target_lead_ms: Option<f64>,
}

fn summarize_field(
    timings: &[TickTiming],
    field: impl Fn(&TickTiming) -> f64,
) -> Result<crate::TimingSummary, RuntimeDiagnostic> {
    if timings.is_empty() {
        return Err(RuntimeDiagnostic::new(
            RuntimeDiagnosticCode::CommandBufferFailed,
            "benchmark phase produced no timing samples",
        ));
    }
    Ok(summarize(&timings.iter().map(field).collect::<Vec<_>>()))
}

fn summarize_optional_field(
    timings: &[TickTiming],
    field: impl Fn(&TickTiming) -> Option<f64>,
) -> Result<crate::TimingSummary, RuntimeDiagnostic> {
    let samples = timings.iter().filter_map(field).collect::<Vec<_>>();
    if samples.is_empty() {
        return Err(RuntimeDiagnostic::new(
            RuntimeDiagnosticCode::CommandBufferFailed,
            "presented benchmark produced no presentation timing samples",
        ));
    }
    Ok(summarize(&samples))
}

struct RawTickTiming {
    status: MTLCommandBufferStatus,
    gpu_start: f64,
    gpu_end: f64,
    cpu_orchestration_ms: f64,
    end_to_end_tick_ms: f64,
    deadline_lateness_ms: f64,
    presentation: Option<PendingPresentation>,
}

struct PendingPresentation {
    presented_time: Receiver<f64>,
    target_timestamp: f64,
    target_presentation_timestamp: f64,
}

impl RawTickTiming {
    fn finish(self) -> Result<TickTiming, RuntimeDiagnostic> {
        if self.status == MTLCommandBufferStatus::Error {
            return Err(RuntimeDiagnostic::new(
                RuntimeDiagnosticCode::CommandBufferFailed,
                "Metal reported a benchmark command-buffer error",
            ));
        }
        if self.gpu_start <= 0.0 || self.gpu_end < self.gpu_start {
            return Err(RuntimeDiagnostic::new(
                RuntimeDiagnosticCode::CommandBufferFailed,
                "Metal did not provide valid command-buffer GPU timestamps",
            ));
        }
        let gpu_deadline_missed = self
            .presentation
            .as_ref()
            .is_some_and(|presentation| self.gpu_end > presentation.target_presentation_timestamp);
        let display_target_lead_ms = self.presentation.as_ref().map(|presentation| {
            (presentation.target_presentation_timestamp - presentation.target_timestamp) * 1_000.0
        });
        let (presentation_lateness_ms, presentation_deadline_missed, presentation_skipped) =
            if let Some(presentation) = self.presentation {
                let presented_time = presentation
                    .presented_time
                    .recv_timeout(Duration::from_millis(250))
                    .map_err(|_| {
                        RuntimeDiagnostic::new(
                            RuntimeDiagnosticCode::CommandBufferFailed,
                            "drawable presented handler did not report a presentation",
                        )
                    })?;
                let skipped = presented_time <= 0.0;
                let lateness = ((presented_time - presentation.target_presentation_timestamp)
                    * 1_000.0)
                    .max(0.0);
                (Some(lateness), skipped || lateness > 0.5, skipped)
            } else {
                (None, false, false)
            };
        Ok(TickTiming {
            gpu_ms: (self.gpu_end - self.gpu_start) * 1_000.0,
            cpu_orchestration_ms: self.cpu_orchestration_ms,
            end_to_end_tick_ms: self.end_to_end_tick_ms,
            deadline_lateness_ms: self.deadline_lateness_ms,
            gpu_deadline_missed,
            presentation_lateness_ms,
            presentation_deadline_missed,
            presentation_skipped,
            display_target_lead_ms,
        })
    }
}

fn display_name(module_name: &str) -> String {
    module_name
        .split('_')
        .map(|word| {
            let mut characters = word.chars();
            match characters.next() {
                Some(first) => first.to_uppercase().chain(characters).collect(),
                None => String::new(),
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn peak_resident_set_bytes() -> Option<u64> {
    let mut usage = std::mem::MaybeUninit::<libc::rusage>::zeroed();
    let status = unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) };
    (status == 0).then(|| unsafe { usage.assume_init().ru_maxrss as u64 })
}

fn wait_until(target: Instant) {
    if let Some(remaining) = target.checked_duration_since(Instant::now()) {
        let (numerator, denominator) = mach_timebase();
        let ticks =
            remaining.as_nanos().saturating_mul(u128::from(denominator)) / u128::from(numerator);
        let deadline = unsafe { mach_absolute_time() }.saturating_add(ticks as u64);
        unsafe {
            mach_wait_until(deadline);
        }
    }
    while Instant::now() < target {
        std::hint::spin_loop();
    }
}

fn pacing_offset(tick: u32, rate_hz: u32) -> Duration {
    let nanoseconds = (u128::from(tick) * 1_000_000_000_u128) / u128::from(rate_hz);
    Duration::from_nanos(nanoseconds as u64)
}

fn mach_timebase() -> (u32, u32) {
    static TIMEBASE: std::sync::OnceLock<(u32, u32)> = std::sync::OnceLock::new();
    *TIMEBASE.get_or_init(|| {
        let mut info = std::mem::MaybeUninit::<MachTimebaseInfo>::zeroed();
        let status = unsafe { mach_timebase_info(info.as_mut_ptr()) };
        assert_eq!(status, 0, "mach_timebase_info failed");
        let info = unsafe { info.assume_init() };
        (info.numer, info.denom)
    })
}

#[repr(C)]
struct MachTimebaseInfo {
    numer: u32,
    denom: u32,
}

const QOS_CLASS_USER_INTERACTIVE: u32 = 0x21;

unsafe extern "C" {
    fn mach_absolute_time() -> u64;
    fn mach_timebase_info(info: *mut MachTimebaseInfo) -> libc::c_int;
    fn mach_wait_until(deadline: u64) -> libc::c_int;
    fn pthread_set_qos_class_self_np(qos_class: u32, relative_priority: libc::c_int)
    -> libc::c_int;
}

fn view_instance_count(
    graph: &pqo_core::ModuleGraph,
    view: &PlannedView,
) -> Result<u64, RuntimeDiagnostic> {
    let mut count = None;
    for read in &view.reads {
        let length = match graph.stream(read.stream).unwrap().length {
            StreamLength::Fixed(length) => length,
            StreamLength::Dynamic(_) => {
                return Err(RuntimeDiagnostic::new(
                    RuntimeDiagnosticCode::UnsupportedGraph,
                    "dynamic view lengths are not in the first Metal slice",
                ));
            }
        };
        if count.is_some_and(|expected| expected != length) {
            return Err(RuntimeDiagnostic::new(
                RuntimeDiagnosticCode::UnsupportedGraph,
                "view streams have incompatible logical lengths",
            )
            .at(format!("views.{}", view.view.0)));
        }
        count = Some(length);
    }
    Ok(count.unwrap_or(0) as u64)
}

fn dynamic_count_for_view(
    graph: &pqo_core::ModuleGraph,
    view: &PlannedView,
) -> Result<Option<StreamId>, RuntimeDiagnostic> {
    let mut dynamic_count = None;
    let mut fixed_length = None;
    for read in &view.reads {
        match graph.stream(read.stream).unwrap().length {
            StreamLength::Fixed(length) => {
                if dynamic_count.is_some()
                    || fixed_length.is_some_and(|expected| expected != length)
                {
                    return Err(RuntimeDiagnostic::new(
                        RuntimeDiagnosticCode::UnsupportedGraph,
                        "view streams have incompatible logical lengths",
                    )
                    .at(format!("views.{}", view.view.0)));
                }
                fixed_length = Some(length);
            }
            StreamLength::Dynamic(count) => {
                if fixed_length.is_some() || dynamic_count.is_some_and(|expected| expected != count)
                {
                    return Err(RuntimeDiagnostic::new(
                        RuntimeDiagnosticCode::UnsupportedGraph,
                        "view streams have incompatible dynamic count domains",
                    )
                    .at(format!("views.{}", view.view.0)));
                }
                dynamic_count = Some(count);
            }
        }
    }
    Ok(dynamic_count)
}

fn requires_wait_after(schedule: &ExecutionSchedule, submitted: ScheduleItemId) -> bool {
    schedule.completion_requirements.iter().any(|requirement| {
        matches!(
            requirement,
            CompletionRequirement::BeforeNextTick {
                after,
                enforcement: CompletionEnforcement::HostWait,
                ..
            } if *after == submitted
        )
    })
}

fn attach_layer(window: &winit::window::Window, layer: &MetalLayer) {
    unsafe {
        let view = window.ns_view() as *mut objc::runtime::Object;
        let _: () = msg_send![view, setWantsLayer: YES];
        let layer_object =
            layer.as_ref() as *const metal::MetalLayerRef as *mut objc::runtime::Object;
        let _: () = msg_send![view, setLayer: layer_object];
    }
}

fn window_frame(window: &winit::window::Window) -> Option<WindowFrame> {
    let position = window.outer_position().ok()?;
    let size = window.outer_size();
    Some(WindowFrame {
        x: position.x,
        y: position.y,
        width: size.width,
        height: size.height,
        scale_factor: window.scale_factor(),
    })
}

fn follow_linked_panel(
    window: &winit::window::Window,
    panel_frame: Option<WindowFrame>,
    manager: Option<&SnapManager>,
    panel: Option<&mut PanelBridge>,
) {
    let (Some(anchor), Some(moving), Some(manager), Some(panel)) =
        (window_frame(window), panel_frame, manager, panel)
    else {
        return;
    };
    if let Some(position) = manager.follow(anchor, moving) {
        panel.set_window_position(position);
    }
}

fn publish_viewer_frame(window: &winit::window::Window, panel: Option<&mut PanelBridge>) {
    if let (Some(frame), Some(panel)) = (window_frame(window), panel) {
        panel.publish_viewer_frame(frame);
    }
}

fn present_benchmark_window(window: &winit::window::Window) {
    unsafe {
        let application: *mut objc::runtime::Object =
            msg_send![objc::class!(NSApplication), sharedApplication];
        let _: bool = msg_send![application, setActivationPolicy: 0_isize];
        let _: () = msg_send![application, finishLaunching];
        let _: () = msg_send![application, activateIgnoringOtherApps: YES];
        let native_window = window.ns_window() as *mut objc::runtime::Object;
        let _: () = msg_send![
            native_window,
            makeKeyAndOrderFront: std::ptr::null_mut::<objc::runtime::Object>()
        ];
        let _: () = msg_send![native_window, display];
    }
}

fn resize_layer(window: &winit::window::Window, layer: &MetalLayer) {
    let size = window.inner_size();
    layer.set_drawable_size(CGSize::new(size.width as f64, size.height as f64));
}

fn pointer_ndc(point: (f64, f64), viewport: PhysicalSize<u32>) -> [f32; 2] {
    [
        (2.0 * point.0 / f64::from(viewport.width) - 1.0) as f32,
        (1.0 - 2.0 * point.1 / f64::from(viewport.height)) as f32,
    ]
}

fn allocate_streams(
    validated: &ValidatedModuleGraph,
    device: &Device,
    queue: &CommandQueue,
) -> Result<Vec<Vec<Buffer>>, RuntimeDiagnostic> {
    let command_buffer = queue.new_command_buffer();
    command_buffer.set_label("pqo.initialize");
    let blit = command_buffer.new_blit_command_encoder();
    let mut staging = Vec::new();
    let mut result = Vec::new();

    for stream in &validated.graph().resources.streams {
        let length = element_size(&stream.element_type)?
            .checked_mul(stream.capacity as usize)
            .ok_or_else(|| {
                RuntimeDiagnostic::new(
                    RuntimeDiagnosticCode::ResourceAllocationFailed,
                    "stream allocation size overflow",
                )
                .at(format!("streams.{}", stream.name))
            })?;
        let mut initial = encode_stream_initial(stream)?;
        initial.resize(length, 0);
        let upload = device.new_buffer_with_data(
            initial.as_ptr().cast(),
            length as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let mut versions = Vec::new();
        for _ in 0..stream.buffering {
            let buffer = device.new_buffer(length as u64, MTLResourceOptions::StorageModePrivate);
            blit.copy_from_buffer(&upload, 0, &buffer, 0, length as u64);
            versions.push(buffer);
        }
        staging.push(upload);
        result.push(versions);
    }
    blit.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();
    if command_buffer.status() == MTLCommandBufferStatus::Error {
        return Err(RuntimeDiagnostic::new(
            RuntimeDiagnosticCode::CommandBufferFailed,
            "initial resource upload failed",
        ));
    }
    drop(staging);
    Ok(result)
}

fn allocate_values(
    validated: &ValidatedModuleGraph,
    device: &Device,
) -> Result<Vec<Buffer>, RuntimeDiagnostic> {
    let bytes = validated
        .graph()
        .resources
        .values
        .iter()
        .map(|value| match &value.kind {
            ValueKind::Constant(literal) => encode_literal(&value.data_type, literal),
            ValueKind::ScheduleFixedDt { schedule } => {
                let rate_hz = match validated.graph().schedule(*schedule).unwrap().timing {
                    pqo_core::ScheduleTiming::Fixed { rate_hz, .. } => rate_hz,
                };
                Ok((1.0_f32 / rate_hz as f32).to_le_bytes().to_vec())
            }
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(bytes
        .iter()
        .map(|bytes| {
            device.new_buffer_with_data(
                bytes.as_ptr().cast(),
                bytes.len() as u64,
                MTLResourceOptions::StorageModeShared,
            )
        })
        .collect())
}

fn encode_stream_initial(stream: &pqo_core::StreamNode) -> Result<Vec<u8>, RuntimeDiagnostic> {
    match &stream.initial {
        Some(StreamInitializer::Explicit(Literal::Array(items))) => {
            let mut bytes = Vec::new();
            for item in items {
                bytes.extend(encode_literal(&stream.element_type, item)?);
            }
            Ok(bytes)
        }
        Some(StreamInitializer::Repeat { value, count }) => {
            let element = encode_literal(&stream.element_type, value)?;
            let mut bytes = Vec::with_capacity(element.len() * *count as usize);
            for _ in 0..*count {
                bytes.extend_from_slice(&element);
            }
            Ok(bytes)
        }
        Some(StreamInitializer::Linear { start, step, count }) => {
            encode_f32_initializer(&stream.element_type, start, step, None, *count, *count)
        }
        Some(StreamInitializer::Grid2D {
            origin,
            column_step,
            row_step,
            columns,
            count,
        }) => encode_f32_initializer(
            &stream.element_type,
            origin,
            column_step,
            Some(row_step),
            *columns,
            *count,
        ),
        Some(_) => Err(RuntimeDiagnostic::new(
            RuntimeDiagnosticCode::UnsupportedGraph,
            "validated stream initializer cannot be lowered by Metal v0",
        )
        .at(format!("streams.{}.initial", stream.name))),
        None => Ok(Vec::new()),
    }
}

fn encode_f32_initializer(
    data_type: &DataType,
    origin: &Literal,
    column_step: &Literal,
    row_step: Option<&Literal>,
    columns: u32,
    count: u32,
) -> Result<Vec<u8>, RuntimeDiagnostic> {
    let origin = f32_components(data_type, origin)?;
    let column_step = f32_components(data_type, column_step)?;
    let row_step = row_step
        .map(|literal| f32_components(data_type, literal))
        .transpose()?;
    let mut bytes = Vec::with_capacity(origin.len() * count as usize * std::mem::size_of::<f32>());
    for index in 0..count {
        let column = if row_step.is_some() {
            index % columns
        } else {
            index
        } as f32;
        let row = if row_step.is_some() {
            index / columns
        } else {
            0
        } as f32;
        for lane in 0..origin.len() {
            let value = origin[lane]
                + column_step[lane] * column
                + row_step.as_ref().map_or(0.0, |step| step[lane] * row);
            bytes.extend_from_slice(&value.to_bits().to_le_bytes());
        }
    }
    Ok(bytes)
}

fn f32_components(data_type: &DataType, literal: &Literal) -> Result<Vec<f32>, RuntimeDiagnostic> {
    match (data_type, literal) {
        (DataType::Scalar(ScalarType::F32), Literal::F32Bits(value)) => {
            Ok(vec![f32::from_bits(*value)])
        }
        (
            DataType::Vector {
                scalar: ScalarType::F32,
                lanes,
            },
            Literal::Vector(items),
        ) if items.len() == *lanes as usize => items
            .iter()
            .map(|item| match item {
                Literal::F32Bits(value) => Ok(f32::from_bits(*value)),
                _ => Err(RuntimeDiagnostic::new(
                    RuntimeDiagnosticCode::UnsupportedGraph,
                    "f32 initializer vector contains a non-f32 lane",
                )),
            })
            .collect(),
        _ => Err(RuntimeDiagnostic::new(
            RuntimeDiagnosticCode::UnsupportedGraph,
            "linear and grid initializers require f32 scalar or vector data",
        )),
    }
}

fn encode_literal(data_type: &DataType, literal: &Literal) -> Result<Vec<u8>, RuntimeDiagnostic> {
    match (data_type, literal) {
        (DataType::Scalar(ScalarType::Bool), Literal::Bool(value)) => Ok(vec![u8::from(*value)]),
        (DataType::Scalar(ScalarType::I32), Literal::I32(value)) => {
            Ok(value.to_le_bytes().to_vec())
        }
        (DataType::Scalar(ScalarType::U32), Literal::U32(value)) => {
            Ok(value.to_le_bytes().to_vec())
        }
        (DataType::Scalar(ScalarType::F16), Literal::F16Bits(value)) => {
            Ok(value.to_le_bytes().to_vec())
        }
        (DataType::Scalar(ScalarType::F32), Literal::F32Bits(value)) => {
            Ok(value.to_le_bytes().to_vec())
        }
        (DataType::Vector { scalar, lanes }, Literal::Vector(items))
            if items.len() == *lanes as usize =>
        {
            let mut bytes = Vec::new();
            for item in items {
                bytes.extend(encode_literal(&DataType::Scalar(scalar.clone()), item)?);
            }
            Ok(bytes)
        }
        _ => Err(RuntimeDiagnostic::new(
            RuntimeDiagnosticCode::UnsupportedGraph,
            format!("Metal v0 cannot encode literal {literal:?} as {data_type:?}"),
        )),
    }
}

fn element_size(data_type: &DataType) -> Result<usize, RuntimeDiagnostic> {
    match data_type {
        DataType::Scalar(ScalarType::Bool) => Ok(1),
        DataType::Scalar(ScalarType::F16) => Ok(2),
        DataType::Scalar(_) => Ok(4),
        DataType::Vector { scalar, lanes } => {
            element_size(&DataType::Scalar(scalar.clone())).map(|size| size * *lanes as usize)
        }
        _ => Err(RuntimeDiagnostic::new(
            RuntimeDiagnosticCode::UnsupportedGraph,
            format!("Metal v0 has no storage layout for {data_type:?}"),
        )),
    }
}

type Pipelines = (
    BTreeMap<PassId, ComputePipelineState>,
    BTreeMap<ViewId, RenderPipelineState>,
    Option<IndirectSupport>,
    Vec<ShaderIdentity>,
    Vec<PipelineIdentity>,
);

fn build_pipelines(
    validated: &ValidatedModuleGraph,
    device: &Device,
    project_root: Option<&Path>,
) -> Result<Pipelines, RuntimeDiagnostic> {
    let schedule = &validated.execution_plan().schedules[0];
    let options = CompileOptions::new();
    let mut compute = BTreeMap::new();
    let mut render = BTreeMap::new();
    let mut shaders = BTreeMap::<String, String>::new();
    let mut pipeline_identities = Vec::new();

    for pass in schedule
        .passes
        .iter()
        .chain(validated.execution_plan().intervention_passes.iter())
    {
        let source = shader_source(&pass.implementation, project_root)?;
        let library = device
            .new_library_with_source(source.as_ref(), &options)
            .map_err(|error| {
                RuntimeDiagnostic::new(RuntimeDiagnosticCode::ShaderCompilationFailed, error)
                    .at(format!("passes.{}", pass.pass.0))
            })?;
        let function = library
            .get_function(&pass.implementation.entry, None)
            .map_err(|error| {
                RuntimeDiagnostic::new(RuntimeDiagnosticCode::PipelineCreationFailed, error)
                    .at(format!("passes.{}", pass.pass.0))
            })?;
        let pipeline = device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|error| {
                RuntimeDiagnostic::new(RuntimeDiagnosticCode::PipelineCreationFailed, error)
                    .at(format!("passes.{}", pass.pass.0))
            })?;
        pipeline_identities.push(PipelineIdentity {
            entry: pass.implementation.entry.clone(),
            kind: "compute".to_owned(),
            source_sha256: sha256(source.as_bytes()),
            thread_execution_width: Some(pipeline.thread_execution_width()),
            max_threads_per_threadgroup: Some(pipeline.max_total_threads_per_threadgroup()),
        });
        shaders.insert(
            pass.implementation.source.clone(),
            sha256(source.as_bytes()),
        );
        compute.insert(pass.pass, pipeline);
    }

    for view in &schedule.views {
        let source = shader_source(&view.implementation, project_root)?;
        let library = device
            .new_library_with_source(source.as_ref(), &options)
            .map_err(|error| {
                RuntimeDiagnostic::new(RuntimeDiagnosticCode::ShaderCompilationFailed, error)
                    .at(format!("views.{}", view.view.0))
            })?;
        let vertex_name = format!("{}_vertex", view.implementation.entry);
        let fragment_name = format!("{}_fragment", view.implementation.entry);
        let vertex = library.get_function(&vertex_name, None).map_err(|error| {
            RuntimeDiagnostic::new(RuntimeDiagnosticCode::PipelineCreationFailed, error)
                .at(format!("views.{}", view.view.0))
        })?;
        let fragment = library
            .get_function(&fragment_name, None)
            .map_err(|error| {
                RuntimeDiagnostic::new(RuntimeDiagnosticCode::PipelineCreationFailed, error)
                    .at(format!("views.{}", view.view.0))
            })?;
        let descriptor = RenderPipelineDescriptor::new();
        descriptor.set_vertex_function(Some(&vertex));
        descriptor.set_fragment_function(Some(&fragment));
        descriptor
            .color_attachments()
            .object_at(0)
            .unwrap()
            .set_pixel_format(MTLPixelFormat::BGRA8Unorm);
        if matches!(
            view.implementation.entry.as_str(),
            "neon_flock_pipeline" | "quantum_field_pipeline"
        ) {
            let attachment = descriptor.color_attachments().object_at(0).unwrap();
            attachment.set_blending_enabled(true);
            attachment.set_source_rgb_blend_factor(MTLBlendFactor::SourceAlpha);
            attachment.set_destination_rgb_blend_factor(MTLBlendFactor::One);
            attachment.set_source_alpha_blend_factor(MTLBlendFactor::One);
            attachment.set_destination_alpha_blend_factor(MTLBlendFactor::One);
        } else if project_root.is_some() {
            let attachment = descriptor.color_attachments().object_at(0).unwrap();
            attachment.set_blending_enabled(true);
            attachment.set_source_rgb_blend_factor(MTLBlendFactor::SourceAlpha);
            attachment.set_destination_rgb_blend_factor(MTLBlendFactor::OneMinusSourceAlpha);
            attachment.set_source_alpha_blend_factor(MTLBlendFactor::One);
            attachment.set_destination_alpha_blend_factor(MTLBlendFactor::OneMinusSourceAlpha);
        }
        let pipeline = device
            .new_render_pipeline_state(&descriptor)
            .map_err(|error| {
                RuntimeDiagnostic::new(RuntimeDiagnosticCode::PipelineCreationFailed, error)
                    .at(format!("views.{}", view.view.0))
            })?;
        pipeline_identities.push(PipelineIdentity {
            entry: view.implementation.entry.clone(),
            kind: "render".to_owned(),
            source_sha256: sha256(source.as_bytes()),
            thread_execution_width: None,
            max_threads_per_threadgroup: None,
        });
        shaders.insert(
            view.implementation.source.clone(),
            sha256(source.as_bytes()),
        );
        render.insert(view.view, pipeline);
    }

    let dynamic_passes = schedule
        .passes
        .iter()
        .chain(validated.execution_plan().intervention_passes.iter())
        .filter_map(|pass| match pass.dispatch {
            DispatchDomain::OverStream(stream)
                if matches!(
                    validated.graph().stream(stream).unwrap().length,
                    StreamLength::Dynamic(_)
                ) =>
            {
                Some(pass.pass)
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    let dynamic_views = schedule
        .views
        .iter()
        .filter_map(|view| {
            dynamic_count_for_view(validated.graph(), view)
                .ok()
                .flatten()
                .map(|_| view.view)
        })
        .collect::<Vec<_>>();
    let indirect = if dynamic_passes.is_empty() && dynamic_views.is_empty() {
        None
    } else {
        let library = device
            .new_library_with_source(INDIRECT_ARGUMENT_SOURCE, &options)
            .map_err(|error| {
                RuntimeDiagnostic::new(RuntimeDiagnosticCode::ShaderCompilationFailed, error)
                    .at("runtime.indirect_arguments")
            })?;
        let compute_function = library
            .get_function("pqo_prepare_compute_dispatch", None)
            .map_err(|error| {
                RuntimeDiagnostic::new(RuntimeDiagnosticCode::PipelineCreationFailed, error)
                    .at("runtime.indirect_arguments.compute")
            })?;
        let draw_function = library
            .get_function("pqo_prepare_draw", None)
            .map_err(|error| {
                RuntimeDiagnostic::new(RuntimeDiagnosticCode::PipelineCreationFailed, error)
                    .at("runtime.indirect_arguments.draw")
            })?;
        let prepare_compute = device
            .new_compute_pipeline_state_with_function(&compute_function)
            .map_err(|error| {
                RuntimeDiagnostic::new(RuntimeDiagnosticCode::PipelineCreationFailed, error)
                    .at("runtime.indirect_arguments.compute")
            })?;
        let prepare_draw = device
            .new_compute_pipeline_state_with_function(&draw_function)
            .map_err(|error| {
                RuntimeDiagnostic::new(RuntimeDiagnosticCode::PipelineCreationFailed, error)
                    .at("runtime.indirect_arguments.draw")
            })?;
        let compute_arguments = dynamic_passes
            .into_iter()
            .map(|pass| {
                (
                    pass,
                    device.new_buffer(
                        (3 * std::mem::size_of::<u32>()) as u64,
                        MTLResourceOptions::StorageModePrivate,
                    ),
                )
            })
            .collect();
        let draw_arguments = dynamic_views
            .into_iter()
            .map(|view| {
                (
                    view,
                    device.new_buffer(
                        (4 * std::mem::size_of::<u32>()) as u64,
                        MTLResourceOptions::StorageModePrivate,
                    ),
                )
            })
            .collect();
        let internal_sha = sha256(INDIRECT_ARGUMENT_SOURCE.as_bytes());
        shaders.insert(
            "pqo://runtime/indirect_arguments.metal".to_owned(),
            internal_sha.clone(),
        );
        pipeline_identities.push(PipelineIdentity {
            entry: "pqo_prepare_compute_dispatch".to_owned(),
            kind: "compute-lowering".to_owned(),
            source_sha256: internal_sha.clone(),
            thread_execution_width: Some(prepare_compute.thread_execution_width()),
            max_threads_per_threadgroup: Some(prepare_compute.max_total_threads_per_threadgroup()),
        });
        pipeline_identities.push(PipelineIdentity {
            entry: "pqo_prepare_draw".to_owned(),
            kind: "compute-lowering".to_owned(),
            source_sha256: internal_sha,
            thread_execution_width: Some(prepare_draw.thread_execution_width()),
            max_threads_per_threadgroup: Some(prepare_draw.max_total_threads_per_threadgroup()),
        });
        Some(IndirectSupport {
            prepare_compute,
            prepare_draw,
            compute_arguments,
            draw_arguments,
        })
    };

    let shaders = shaders
        .into_iter()
        .map(|(source_path, sha256)| ShaderIdentity {
            source_path,
            sha256,
        })
        .collect();
    pipeline_identities
        .sort_by(|left, right| (&left.kind, &left.entry).cmp(&(&right.kind, &right.entry)));
    Ok((compute, render, indirect, shaders, pipeline_identities))
}

fn shader_source<'a>(
    implementation: &'a pqo_core::BackendImplementation,
    project_root: Option<&Path>,
) -> Result<Cow<'a, str>, RuntimeDiagnostic> {
    if let Some(source) = implementation.source_text.as_deref() {
        return Ok(Cow::Borrowed(source));
    }
    if let Some(project_root) = project_root {
        let relative = Path::new(&implementation.source);
        if relative.is_absolute()
            || relative
                .components()
                .any(|component| !matches!(component, Component::Normal(_) | Component::CurDir))
        {
            return Err(RuntimeDiagnostic::new(
                RuntimeDiagnosticCode::UnsupportedGraph,
                format!(
                    "project Metal source must stay inside the package: `{}`",
                    implementation.source
                ),
            ));
        }
        let path = project_root.join(relative);
        match fs::read_to_string(&path) {
            Ok(source) => return Ok(Cow::Owned(source)),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => {
                return Err(RuntimeDiagnostic::new(
                    RuntimeDiagnosticCode::UnsupportedGraph,
                    format!(
                        "could not read project-local Metal source `{}`: {error}",
                        path.display()
                    ),
                ));
            }
        }
    }
    match implementation.source.as_str() {
        "kernels/euler_integrate.metal" => Ok(Cow::Borrowed(INTEGRATE_SOURCE)),
        "kernels/ground_contact.metal" => Ok(Cow::Borrowed(CONTACT_SOURCE)),
        "shaders/particle.metal" => Ok(Cow::Borrowed(PARTICLE_SOURCE)),
        "kernels/neon_flock.metal" => Ok(Cow::Borrowed(NEON_FLOCK_SOURCE)),
        "shaders/neon_flock.metal" => Ok(Cow::Borrowed(NEON_FLOCK_RENDER_SOURCE)),
        "kernels/crystal.metal" => Ok(Cow::Borrowed(CRYSTAL_SOURCE)),
        "shaders/crystal.metal" => Ok(Cow::Borrowed(CRYSTAL_RENDER_SOURCE)),
        _ => Err(RuntimeDiagnostic::new(
            RuntimeDiagnosticCode::UnsupportedGraph,
            format!("no packaged Metal source for `{}`", implementation.source),
        )),
    }
}

fn hot_reload_state(
    current: &RuntimeState,
    project_root: &Path,
) -> Result<RuntimeState, RuntimeDiagnostic> {
    let module_name = current.validated.graph().name.as_str();
    let preferred = project_root.join(format!("{module_name}.pqo"));
    let entry = if preferred.is_file() {
        preferred
    } else {
        let mut candidates = fs::read_dir(project_root)
            .map_err(|error| {
                RuntimeDiagnostic::new(
                    RuntimeDiagnosticCode::UnsupportedGraph,
                    format!(
                        "could not inspect hot-reload project root `{}`: {error}",
                        project_root.display()
                    ),
                )
            })?
            .filter_map(Result::ok)
            .map(|entry| entry.path())
            .filter(|path| {
                path.is_file()
                    && path.extension().and_then(|extension| extension.to_str()) == Some("pqo")
            })
            .collect::<Vec<_>>();
        candidates.sort();
        candidates.into_iter().next().ok_or_else(|| {
            RuntimeDiagnostic::new(
                RuntimeDiagnosticCode::UnsupportedGraph,
                format!(
                    "hot reload could not find a .pqo entry in `{}`",
                    project_root.display()
                ),
            )
        })?
    };
    let source = fs::read_to_string(&entry).map_err(|error| {
        RuntimeDiagnostic::new(
            RuntimeDiagnosticCode::UnsupportedGraph,
            format!(
                "could not read hot-reload entry `{}`: {error}",
                entry.display()
            ),
        )
    })?;
    let graph = pqo_syntax::parse(&source).map_err(|diagnostics| {
        RuntimeDiagnostic::new(
            RuntimeDiagnosticCode::UnsupportedGraph,
            format!("hot-reload parse failed:\n{diagnostics:#?}"),
        )
    })?;
    let report = Validator::validate(&graph);
    let validated = report.validated.ok_or_else(|| {
        RuntimeDiagnostic::new(
            RuntimeDiagnosticCode::UnsupportedGraph,
            format!(
                "hot-reload validation failed:\n{}",
                serde_json::to_string_pretty(&report.diagnostics)
                    .unwrap_or_else(|_| format!("{:#?}", report.diagnostics))
            ),
        )
    })?;
    RuntimeState::new_with_project_root(
        validated,
        current.device.clone(),
        current.layer.clone(),
        Some(project_root.to_owned()),
    )
}

fn dispatch_count(
    graph: &pqo_core::ModuleGraph,
    domain: &DispatchDomain,
) -> Result<u64, RuntimeDiagnostic> {
    match domain {
        DispatchDomain::Fixed(count) => Ok(*count as u64),
        DispatchDomain::OverStream(stream) => match graph.stream(*stream).unwrap().length {
            StreamLength::Fixed(length) => Ok(length as u64),
            StreamLength::Dynamic(_) => Err(RuntimeDiagnostic::new(
                RuntimeDiagnosticCode::UnsupportedGraph,
                "dynamic dispatch requires GPU indirect encoding",
            )),
        },
    }
}

fn benchmark_dispatch_count(
    graph: &pqo_core::ModuleGraph,
    schedule: &ExecutionSchedule,
) -> Result<u32, RuntimeDiagnostic> {
    let pass = schedule
        .passes
        .iter()
        .find(|pass| {
            matches!(
                pass.dispatch,
                DispatchDomain::OverStream(stream)
                    if matches!(
                        graph.stream(stream).unwrap().length,
                        StreamLength::Dynamic(_)
                    )
            )
        })
        .or_else(|| {
            schedule
                .passes
                .iter()
                .max_by_key(|pass| match pass.dispatch {
                    DispatchDomain::Fixed(count) => u64::from(count),
                    DispatchDomain::OverStream(stream) => {
                        match graph.stream(stream).unwrap().length {
                            StreamLength::Fixed(length) => u64::from(length),
                            StreamLength::Dynamic(_) => 0,
                        }
                    }
                })
        })
        .ok_or_else(|| {
            RuntimeDiagnostic::new(
                RuntimeDiagnosticCode::UnsupportedGraph,
                "benchmark schedule has no compute dispatch domain",
            )
        })?;
    let declared_count = match pass.dispatch {
        DispatchDomain::OverStream(stream) => {
            let stream = graph.stream(stream).unwrap();
            match stream.length {
                StreamLength::Fixed(length) => u64::from(length),
                StreamLength::Dynamic(_) => u64::from(stream.capacity),
            }
        }
        DispatchDomain::Fixed(count) => u64::from(count),
    };
    u32::try_from(declared_count).map_err(|_| {
        RuntimeDiagnostic::new(
            RuntimeDiagnosticCode::UnsupportedGraph,
            "benchmark dispatch domain exceeds the supported count",
        )
    })
}

fn make_fingerprint(
    validated: &ValidatedModuleGraph,
    device: &Device,
    shader_hashes: Vec<ShaderIdentity>,
    pipelines: Vec<PipelineIdentity>,
) -> RuntimeFingerprint {
    #[derive(serde::Serialize)]
    struct Identity<'a> {
        artifact: &'a str,
        device: &'a str,
        operating_system: &'a str,
        host_profile: &'a str,
        host_executable_sha256: &'a str,
        source_revision: &'a str,
        source_dirty: Option<bool>,
        rust_compiler: &'a str,
        metal_sdk: &'a str,
        shader_hashes: &'a [ShaderIdentity],
        pipelines: &'a [PipelineIdentity],
    }

    let device_name = device.name().to_owned();
    let operating_system = command_output("sw_vers", &["-productVersion"])
        .unwrap_or_else(|| "macOS unknown".to_owned());
    let host_profile = if cfg!(debug_assertions) {
        "debug".to_owned()
    } else {
        "release".to_owned()
    };
    let host_executable_sha256 = std::env::current_exe()
        .ok()
        .and_then(|path| std::fs::read(path).ok())
        .map(sha256)
        .unwrap_or_else(|| "unknown".to_owned());
    let source_revision =
        command_output("git", &["rev-parse", "HEAD"]).unwrap_or_else(|| "unknown".to_owned());
    let source_dirty = Command::new("git")
        .args(["status", "--porcelain"])
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map(|output| !output.stdout.is_empty());
    let rust_compiler =
        command_output("rustc", &["--version"]).unwrap_or_else(|| "unknown".to_owned());
    let metal_sdk = command_output("xcrun", &["--sdk", "macosx", "--show-sdk-version"])
        .unwrap_or_else(|| "unknown".to_owned());
    let identity = Identity {
        artifact: validated.artifact_fingerprint(),
        device: &device_name,
        operating_system: &operating_system,
        host_profile: &host_profile,
        host_executable_sha256: &host_executable_sha256,
        source_revision: &source_revision,
        source_dirty,
        rust_compiler: &rust_compiler,
        metal_sdk: &metal_sdk,
        shader_hashes: &shader_hashes,
        pipelines: &pipelines,
    };
    let bytes = serde_json::to_vec(&identity).expect("runtime identity serialization");
    RuntimeFingerprint {
        artifact: validated.artifact_fingerprint().to_owned(),
        device: device_name,
        operating_system,
        host_profile,
        host_executable_sha256,
        source_revision,
        source_dirty,
        rust_compiler,
        metal_sdk,
        shader_hashes,
        pipelines,
        fingerprint: sha256(bytes),
    }
}

fn command_output(program: &str, arguments: &[&str]) -> Option<String> {
    Command::new(program)
        .args(arguments)
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_owned())
        .filter(|output| !output.is_empty())
}

#[cfg(test)]
mod tests {
    use super::{MetalRuntime, RuntimeState, encode_f32_initializer, pacing_offset, project_key};
    use crate::fingerprint::sha256;
    use crate::project::KEY_SHIFT;
    use crate::{BenchmarkConfig, BenchmarkMode, BenchmarkRunner};
    use pqo_core::{
        DataType, HelloCrystalConfig, HelloOrganismConfig, Literal, StreamInitializer,
        conformance::{hello_field_builder, hello_population_builder},
        hello_crystal_builder_with_config, hello_organism_builder,
        hello_organism_builder_with_config,
    };
    use pqo_syntax::parse;
    use pqo_validator::Validator;
    use std::time::Duration;
    use winit::dpi::PhysicalSize;

    const CRYSTAL_LANGUAGE_SOURCE: &str =
        include_str!("../../../examples/hello-crystal/crystal.pqo");

    #[test]
    fn both_shift_keys_forward_the_project_shift_code() {
        assert_eq!(
            project_key(winit::event::VirtualKeyCode::LShift),
            Some(KEY_SHIFT)
        );
        assert_eq!(
            project_key(winit::event::VirtualKeyCode::RShift),
            Some(KEY_SHIFT)
        );
    }

    #[test]
    fn rational_pacing_has_no_one_second_drift() {
        assert_eq!(pacing_offset(120, 120), Duration::from_secs(1));
        assert_eq!(pacing_offset(240, 120), Duration::from_secs(2));
    }

    #[test]
    fn linear_initializer_expands_to_expected_bytes() {
        let bytes = encode_f32_initializer(
            &DataType::f32(),
            &Literal::f32(1.0),
            &Literal::f32(0.5),
            None,
            3,
            3,
        )
        .unwrap();
        assert_eq!(decode_f32(&bytes), vec![1.0, 1.5, 2.0]);
    }

    #[test]
    fn grid_initializer_expands_in_row_major_order() {
        let vector =
            |values: &[f32]| Literal::Vector(values.iter().copied().map(Literal::f32).collect());
        let bytes = encode_f32_initializer(
            &DataType::Vector {
                scalar: pqo_core::ScalarType::F32,
                lanes: 2,
            },
            &vector(&[10.0, 20.0]),
            &vector(&[1.0, 0.0]),
            Some(&vector(&[0.0, 2.0])),
            2,
            4,
        )
        .unwrap();
        assert_eq!(
            decode_f32(&bytes),
            vec![10.0, 20.0, 11.0, 20.0, 10.0, 22.0, 11.0, 22.0]
        );
    }

    #[test]
    fn packaged_dynamic_population_executes_through_gpu_indirect_dispatch() {
        let graph = hello_population_builder(1024, 7).build().unwrap();
        let validated = Validator::validate(&graph)
            .validated
            .expect("dynamic population graph must validate");
        let result = MetalRuntime::benchmark(
            validated,
            BenchmarkConfig {
                mode: BenchmarkMode::Headless,
                runner: BenchmarkRunner::PqoPlan,
                warmup_ticks: 0,
                sample_ticks: 1,
                ..BenchmarkConfig::default()
            },
        )
        .expect("dynamic population must execute through Metal");

        assert_eq!(result.sample_ticks, 1);
        assert_eq!(result.particle_count, 1024);
        assert!(
            result
                .runtime
                .shader_hashes
                .iter()
                .any(|shader| { shader.source_path == "pqo://runtime/indirect_arguments.metal" })
        );
    }

    #[test]
    fn scenario_intervention_executes_at_the_recorded_tick() {
        let graph = hello_population_builder(32, 3).build().unwrap();
        let validated = Validator::validate(&graph)
            .validated
            .expect("population scenario graph must validate");
        let result = MetalRuntime::run_scenario(validated, "recorded_reset")
            .expect("recorded intervention must execute");

        assert_eq!(result.executed_ticks, 2);
        assert_eq!(result.events.len(), 1);
        assert_eq!(result.events[0].tick, 1);
        assert_eq!(result.events[0].pass, "reset_population_age");
        assert_eq!(
            result.events[0].value_overrides,
            vec!["population.reset_age"]
        );
    }

    #[test]
    fn packaged_reflective_field_executes_as_ordered_gpu_passes() {
        let graph = hello_field_builder().build().unwrap();
        let validated = Validator::validate(&graph)
            .validated
            .expect("field graph must validate");
        let result = MetalRuntime::benchmark(
            validated,
            BenchmarkConfig {
                mode: BenchmarkMode::Headless,
                runner: BenchmarkRunner::PqoPlan,
                warmup_ticks: 0,
                sample_ticks: 2,
                ..BenchmarkConfig::default()
            },
        )
        .expect("field specimen must execute through Metal");

        assert_eq!(result.sample_ticks, 2);
        assert!(
            result
                .runtime
                .shader_hashes
                .iter()
                .any(|shader| { shader.source_path == "pqo://specimens/hello_field.metal" })
        );
    }

    #[test]
    fn packaged_organism_executes_coupled_population_and_field_tick() {
        let graph = hello_organism_builder(16_384).build().unwrap();
        let count = graph
            .resources
            .streams
            .iter()
            .find(|stream| stream.name == "cells.active_count")
            .unwrap()
            .id;
        let validated = Validator::validate(&graph)
            .validated
            .expect("organism graph must validate");
        let device = metal::Device::system_default().expect("Metal device");
        let layer = metal::MetalLayer::new();
        layer.set_device(&device);
        layer.set_pixel_format(metal::MTLPixelFormat::BGRA8Unorm);
        let mut state = RuntimeState::new(validated, device.clone(), layer).unwrap();
        let result = state
            .run_benchmark(
                &device,
                BenchmarkConfig {
                    mode: BenchmarkMode::Headless,
                    runner: BenchmarkRunner::PqoPlan,
                    warmup_ticks: 0,
                    sample_ticks: 300,
                    ..BenchmarkConfig::default()
                },
            )
            .expect("organism specimen must execute through Metal");

        assert_eq!(result.sample_ticks, 300);
        assert_eq!(result.particle_count, 16_384);
        assert!(
            result
                .runtime
                .shader_hashes
                .iter()
                .any(|shader| { shader.source_path == "kernels/organism.metal" })
        );

        let readback = device.new_buffer(
            std::mem::size_of::<u32>() as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let command_buffer = state.queue.new_command_buffer();
        let blit = command_buffer.new_blit_command_encoder();
        blit.copy_from_buffer(
            &state.stream_buffers[count.0 as usize][0],
            0,
            &readback,
            0,
            std::mem::size_of::<u32>() as u64,
        );
        blit.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        let active_count = unsafe { *(readback.contents().cast::<u32>()) };
        assert!(
            active_count > 1,
            "the organizer should have produced at least one daughter"
        );
    }

    #[test]
    fn packaged_crystal_orbits_zooms_slices_and_self_heals() {
        let graph = hello_crystal_builder_with_config(HelloCrystalConfig {
            cell_count: 32 * 32 * 32,
        })
        .build()
        .unwrap();
        let metrics = graph
            .resources
            .streams
            .iter()
            .find(|stream| stream.name == "metrics.snapshot")
            .unwrap()
            .id;
        let camera_yaw = graph
            .resources
            .streams
            .iter()
            .find(|stream| stream.name == "interaction.camera_yaw")
            .unwrap()
            .id;
        let camera_pitch = graph
            .resources
            .streams
            .iter()
            .find(|stream| stream.name == "interaction.camera_pitch")
            .unwrap()
            .id;
        let camera_zoom = graph
            .resources
            .streams
            .iter()
            .find(|stream| stream.name == "interaction.camera_zoom")
            .unwrap()
            .id;
        let validated = Validator::validate(&graph)
            .validated
            .expect("crystal graph must validate");
        let device = metal::Device::system_default().expect("Metal device");
        let layer = metal::MetalLayer::new();
        layer.set_device(&device);
        layer.set_pixel_format(metal::MTLPixelFormat::BGRA8Unorm);
        let mut state = RuntimeState::new(validated, device.clone(), layer).unwrap();
        state
            .run_benchmark(
                &device,
                BenchmarkConfig {
                    mode: BenchmarkMode::Headless,
                    runner: BenchmarkRunner::PqoPlan,
                    warmup_ticks: 0,
                    sample_ticks: 100,
                    ..BenchmarkConfig::default()
                },
            )
            .expect("crystal specimen must execute through Metal");

        let byte_count = (16 * std::mem::size_of::<u32>()) as u64;
        let read_metrics = |state: &RuntimeState| {
            let readback =
                device.new_buffer(byte_count, metal::MTLResourceOptions::StorageModeShared);
            let command_buffer = state.queue.new_command_buffer();
            let blit = command_buffer.new_blit_command_encoder();
            blit.copy_from_buffer(
                &state.stream_buffers[metrics.0 as usize][0],
                0,
                &readback,
                0,
                byte_count,
            );
            blit.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();
            unsafe { std::slice::from_raw_parts(readback.contents().cast::<u32>(), 16).to_vec() }
        };
        let before = read_metrics(&state);
        assert!(before[0] > 56, "solid phase must grow beyond the seed");
        assert!(before[2] > 0, "the grown crystal must expose a surface");
        assert_eq!(before[3], 0, "autonomous growth must never damage crystal");
        assert_eq!(before[8], 0, "no slice intervention has occurred");
        assert!(
            state
                .pointer_hits_crystal((480.0, 360.0), PhysicalSize::new(960, 720))
                .unwrap(),
            "the center of the grown crystal must pick as material"
        );
        assert!(
            !state
                .pointer_hits_crystal((30.0, 30.0), PhysicalSize::new(960, 720))
                .unwrap(),
            "the black corner must pick as orbit space"
        );

        let read_camera = |state: &RuntimeState| {
            let readback = device.new_buffer(12, metal::MTLResourceOptions::StorageModeShared);
            let command_buffer = state.queue.new_command_buffer();
            let blit = command_buffer.new_blit_command_encoder();
            blit.copy_from_buffer(
                &state.stream_buffers[camera_yaw.0 as usize][0],
                0,
                &readback,
                0,
                4,
            );
            blit.copy_from_buffer(
                &state.stream_buffers[camera_pitch.0 as usize][0],
                0,
                &readback,
                4,
                4,
            );
            blit.copy_from_buffer(
                &state.stream_buffers[camera_zoom.0 as usize][0],
                0,
                &readback,
                8,
                4,
            );
            blit.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();
            unsafe {
                let values = std::slice::from_raw_parts(readback.contents().cast::<f32>(), 3);
                [values[0], values[1], values[2]]
            }
        };
        state.queue_pointer_orbit((30.0, 30.0), (90.0, 55.0));
        state.draw_tick().unwrap();
        let camera = read_camera(&state);
        assert!((camera[0] - 0.48).abs() < 0.001);
        assert!((camera[1] - 0.20).abs() < 0.001);
        assert!((camera[2] - 1.0).abs() < 0.001);
        state.queue_pointer_zoom(1.5_f32.ln());
        state.draw_tick().unwrap();
        let camera = read_camera(&state);
        assert!((camera[2] - 1.5).abs() < 0.001);
        let after_orbit = read_metrics(&state);
        assert_eq!(
            after_orbit[3], 0,
            "orbiting through black space must not damage material"
        );
        assert_eq!(
            after_orbit[8], 0,
            "orbiting must not record a slice intervention"
        );

        let texture_descriptor = metal::TextureDescriptor::new();
        texture_descriptor.set_width(256);
        texture_descriptor.set_height(256);
        texture_descriptor.set_pixel_format(metal::MTLPixelFormat::BGRA8Unorm);
        texture_descriptor.set_storage_mode(metal::MTLStorageMode::Private);
        texture_descriptor.set_usage(metal::MTLTextureUsage::RenderTarget);
        let texture = device.new_texture(&texture_descriptor);
        let render_readback =
            device.new_buffer(256 * 256 * 4, metal::MTLResourceOptions::StorageModeShared);
        let command_buffer = state.queue.new_command_buffer();
        let view = &state.validated.execution_plan().schedules[0].views[0];
        state.encode_render(command_buffer, view, &texture).unwrap();
        let blit = command_buffer.new_blit_command_encoder();
        blit.copy_from_texture_to_buffer(
            &texture,
            0,
            0,
            metal::MTLOrigin::default(),
            metal::MTLSize::new(256, 256, 1),
            &render_readback,
            0,
            256 * 4,
            256 * 256 * 4,
            metal::MTLBlitOption::None,
        );
        blit.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        let pixels = unsafe {
            std::slice::from_raw_parts(render_readback.contents().cast::<u8>(), 256 * 256 * 4)
        };
        assert!(
            pixels.chunks_exact(4).any(|pixel| pixel[0] > 64),
            "the crystal renderer must emit visible blue surface pixels"
        );

        state.queue_pointer_slice((250.0, 360.0), (710.0, 360.0), PhysicalSize::new(960, 720));
        for _ in 0..20 {
            state.draw_tick().unwrap();
        }
        let after = read_metrics(&state);
        assert!(after[3] > 0, "the pointer slice must remove material");
        assert_eq!(after[8], 1, "one drag segment must record one slice");

        for _ in 0..40 {
            state.draw_tick().unwrap();
        }
        let healed = read_metrics(&state);
        assert_eq!(
            healed[3], 0,
            "cut cells must heal automatically without another pointer gesture"
        );
        assert_eq!(
            healed[8], 1,
            "healing must preserve the historical slice count"
        );
    }

    #[test]
    fn language_crystal_orbits_zooms_slices_and_self_heals() {
        let graph = parse(CRYSTAL_LANGUAGE_SOURCE).expect("crystal source must parse");
        let stream = |name: &str| {
            graph
                .resources
                .streams
                .iter()
                .find(|stream| stream.name == name)
                .unwrap_or_else(|| panic!("missing crystal stream `{name}`"))
                .id
        };
        let damage = stream("material.damage");
        let render_color = stream("render.color");
        let damage_cell_count = graph
            .resources
            .streams
            .iter()
            .find(|stream| stream.name == "material.damage")
            .expect("material.damage stream")
            .capacity as usize;
        let slice_count = stream("interaction.slice_count");
        let camera_yaw = stream("interaction.camera_yaw");
        let camera_pitch = stream("interaction.camera_pitch");
        let camera_zoom = stream("interaction.camera_zoom");
        let validated = Validator::validate(&graph)
            .validated
            .expect("crystal source graph must validate");
        let device = metal::Device::system_default().expect("Metal device");
        let layer = metal::MetalLayer::new();
        layer.set_device(&device);
        layer.set_pixel_format(metal::MTLPixelFormat::BGRA8Unorm);
        let mut state = RuntimeState::new(validated, device.clone(), layer).unwrap();
        state
            .run_benchmark(
                &device,
                BenchmarkConfig {
                    mode: BenchmarkMode::Headless,
                    runner: BenchmarkRunner::PqoPlan,
                    warmup_ticks: 0,
                    sample_ticks: 100,
                    ..BenchmarkConfig::default()
                },
            )
            .expect("language crystal must execute through Metal");

        let read_bytes = |state: &RuntimeState, stream: pqo_core::StreamId, byte_count: u64| {
            let readback =
                device.new_buffer(byte_count, metal::MTLResourceOptions::StorageModeShared);
            let command_buffer = state.queue.new_command_buffer();
            let blit = command_buffer.new_blit_command_encoder();
            blit.copy_from_buffer(
                &state.stream_buffers[stream.0 as usize][0],
                0,
                &readback,
                0,
                byte_count,
            );
            blit.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();
            readback
        };
        let read_scalar_f32 = |state: &RuntimeState, stream| {
            let readback = read_bytes(state, stream, std::mem::size_of::<f32>() as u64);
            unsafe { *(readback.contents().cast::<f32>()) }
        };
        let read_scalar_u32 = |state: &RuntimeState, stream| {
            let readback = read_bytes(state, stream, std::mem::size_of::<u32>() as u64);
            unsafe { *(readback.contents().cast::<u32>()) }
        };
        let damaged_cells = |state: &RuntimeState| {
            let count = damage_cell_count;
            let readback = read_bytes(state, damage, (count * std::mem::size_of::<f32>()) as u64);
            let damage =
                unsafe { std::slice::from_raw_parts(readback.contents().cast::<f32>(), count) };
            damage.iter().filter(|value| **value > 0.0).count()
        };
        let red_damage_cells = |state: &RuntimeState| {
            let component_count = damage_cell_count * 4;
            let readback = read_bytes(
                state,
                render_color,
                (component_count * std::mem::size_of::<f32>()) as u64,
            );
            let colors = unsafe {
                std::slice::from_raw_parts(readback.contents().cast::<f32>(), component_count)
            };
            colors
                .chunks_exact(4)
                .filter(|color| color[0] > 0.45 && color[0] > color[1] * 1.5)
                .count()
        };

        assert!(
            state
                .pointer_hits_crystal((480.0, 360.0), PhysicalSize::new(960, 720))
                .unwrap(),
            "the language crystal must support pointer hit-testing"
        );
        assert!(
            !state
                .pointer_hits_crystal((30.0, 30.0), PhysicalSize::new(960, 720))
                .unwrap(),
            "background space must remain available for orbiting"
        );

        state.queue_pointer_orbit((30.0, 30.0), (90.0, 55.0));
        state.queue_pointer_zoom(1.5_f32.ln());
        state.draw_tick().unwrap();
        assert!((read_scalar_f32(&state, camera_yaw) - 0.48).abs() < 0.001);
        assert!((read_scalar_f32(&state, camera_pitch) - 0.20).abs() < 0.001);
        assert!((read_scalar_f32(&state, camera_zoom) - 1.5).abs() < 0.001);

        state.queue_pointer_slice((250.0, 360.0), (710.0, 360.0), PhysicalSize::new(960, 720));
        state.draw_tick().unwrap();
        assert!(damaged_cells(&state) > 0, "mouse slicing must damage cells");
        assert!(
            red_damage_cells(&state) > 0,
            "the exposed cut must render with a red damage color"
        );
        assert_eq!(read_scalar_u32(&state, slice_count), 1);

        for _ in 0..60 {
            state.draw_tick().unwrap();
        }
        assert_eq!(
            damaged_cells(&state),
            0,
            "the language crystal must heal without another pointer gesture"
        );
        assert_eq!(read_scalar_u32(&state, slice_count), 1);
    }

    #[test]
    fn parallel_population_compacts_and_allocates_one_thousand_parents() {
        const INITIAL_COUNT: u32 = 1_024;
        const CAPACITY: u32 = 4_096;

        let mut graph = hello_organism_builder(CAPACITY).build().unwrap();
        let repeat = |value, count| StreamInitializer::Repeat { value, count };
        for stream in &mut graph.resources.streams {
            stream.initial = match stream.name.as_str() {
                "cells.active_count" => Some(StreamInitializer::Explicit(Literal::Array(vec![
                    Literal::U32(INITIAL_COUNT),
                ]))),
                "cells.next_stable_id" => Some(StreamInitializer::Explicit(Literal::Array(vec![
                    Literal::U32(INITIAL_COUNT + 1),
                ]))),
                "cells.stable_id" => Some(StreamInitializer::Explicit(Literal::Array(
                    (1..=INITIAL_COUNT).rev().map(Literal::U32).collect(),
                ))),
                "cells.position" => Some(StreamInitializer::Grid2D {
                    origin: Literal::Vector(vec![
                        Literal::f32(-0.93),
                        Literal::f32(-0.93),
                        Literal::f32(0.0),
                    ]),
                    column_step: Literal::Vector(vec![
                        Literal::f32(0.06),
                        Literal::f32(0.0),
                        Literal::f32(0.0),
                    ]),
                    row_step: Literal::Vector(vec![
                        Literal::f32(0.0),
                        Literal::f32(0.06),
                        Literal::f32(0.0),
                    ]),
                    columns: 32,
                    count: INITIAL_COUNT,
                }),
                "cells.radius" => Some(repeat(Literal::f32(0.01), INITIAL_COUNT)),
                "cells.energy" => Some(repeat(Literal::f32(4.0), INITIAL_COUNT)),
                "cells.age" => Some(repeat(Literal::U32(240), INITIAL_COUNT)),
                "cells.fate" | "cells.previous_fate" => {
                    Some(repeat(Literal::U32(1), INITIAL_COUNT))
                }
                "cells.phase" => Some(repeat(Literal::U32(1), INITIAL_COUNT)),
                "cells.health" => Some(StreamInitializer::Explicit(Literal::Array(
                    (0..INITIAL_COUNT)
                        .map(|index| Literal::U32(u32::from(index % 4 == 0) * 2))
                        .collect(),
                ))),
                "cells.fate_confidence" | "cells.time_in_fate" => {
                    Some(repeat(Literal::U32(100), INITIAL_COUNT))
                }
                "cells.parent_id" => Some(repeat(Literal::U32(0), INITIAL_COUNT)),
                "cells.color" => Some(repeat(
                    Literal::Vector(vec![
                        Literal::f32(0.8),
                        Literal::f32(0.8),
                        Literal::f32(0.9),
                        Literal::f32(1.0),
                    ]),
                    INITIAL_COUNT,
                )),
                "field.activator" => Some(repeat(Literal::f32(1.0), 256 * 256)),
                _ => stream.initial.clone(),
            };
        }

        let stream_id = |name: &str| {
            graph
                .resources
                .streams
                .iter()
                .find(|stream| stream.name == name)
                .unwrap()
                .id
        };
        let active_count_id = stream_id("cells.active_count");
        let stable_id = stream_id("cells.stable_id");
        let overflow_id = stream_id("population.neighbor_overflow");
        let report = Validator::validate(&graph);
        assert!(
            report.is_valid(),
            "seeded organism diagnostics: {:#?}",
            report.diagnostics
        );
        let validated = report.validated.unwrap();
        let device = metal::Device::system_default().expect("Metal device");
        let layer = metal::MetalLayer::new();
        layer.set_device(&device);
        layer.set_pixel_format(metal::MTLPixelFormat::BGRA8Unorm);
        let mut state = RuntimeState::new(validated, device.clone(), layer).unwrap();
        state
            .run_benchmark(
                &device,
                BenchmarkConfig {
                    mode: BenchmarkMode::Headless,
                    runner: BenchmarkRunner::PqoPlan,
                    warmup_ticks: 0,
                    sample_ticks: 1,
                    ..BenchmarkConfig::default()
                },
            )
            .expect("parallel population tick must execute");

        let scalar_readback = device.new_buffer(8, metal::MTLResourceOptions::StorageModeShared);
        let command_buffer = state.queue.new_command_buffer();
        let blit = command_buffer.new_blit_command_encoder();
        blit.copy_from_buffer(
            &state.stream_buffers[active_count_id.0 as usize][0],
            0,
            &scalar_readback,
            0,
            4,
        );
        blit.copy_from_buffer(
            &state.stream_buffers[overflow_id.0 as usize][0],
            0,
            &scalar_readback,
            4,
            4,
        );
        blit.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        let scalars =
            unsafe { std::slice::from_raw_parts(scalar_readback.contents().cast::<u32>(), 2) };
        let active_count = scalars[0];
        assert!(
            active_count > INITIAL_COUNT,
            "parallel allocation should accept daughters"
        );
        assert_eq!(scalars[1], 0, "reference density must not overflow bins");

        let id_readback = device.new_buffer(
            u64::from(active_count) * 8,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let command_buffer = state.queue.new_command_buffer();
        let blit = command_buffer.new_blit_command_encoder();
        blit.copy_from_buffer(
            &state.stream_buffers[stable_id.0 as usize][0],
            0,
            &id_readback,
            0,
            u64::from(active_count) * 4,
        );
        blit.copy_from_buffer(
            &state.stream_buffers[stable_id.0 as usize][0],
            0,
            &id_readback,
            u64::from(active_count) * 4,
            u64::from(active_count) * 4,
        );
        blit.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        let ids = unsafe {
            std::slice::from_raw_parts(id_readback.contents().cast::<u32>(), active_count as usize)
        };
        let parents = unsafe {
            std::slice::from_raw_parts(
                id_readback
                    .contents()
                    .cast::<u32>()
                    .add(active_count as usize),
                active_count as usize,
            )
        };
        assert!(
            ids.windows(2).all(|pair| pair[0] < pair[1]),
            "compaction and allocation must preserve canonical stable-ID order"
        );
        assert!(
            ids.iter()
                .filter(|id| **id <= INITIAL_COUNT)
                .all(|id| (INITIAL_COUNT - *id) % 4 != 0),
            "cells with accepted death intents must be removed during compaction"
        );
        let child_parents = ids
            .iter()
            .zip(parents)
            .filter_map(|(id, parent)| (*id > INITIAL_COUNT).then_some(*parent))
            .collect::<Vec<_>>();
        assert!(
            child_parents.windows(2).all(|pair| pair[0] < pair[1]),
            "birth allocation must follow stable parent ID, not input storage order"
        );
    }

    #[test]
    fn developmental_neighborhoods_produce_connected_differentiated_metrics() {
        const INITIAL_COUNT: u32 = 256;
        const CAPACITY: u32 = 1_024;

        let mut graph = hello_organism_builder(CAPACITY).build().unwrap();
        let repeat = |value, count| StreamInitializer::Repeat { value, count };
        for stream in &mut graph.resources.streams {
            stream.initial = match stream.name.as_str() {
                "cells.active_count" => Some(StreamInitializer::Explicit(Literal::Array(vec![
                    Literal::U32(INITIAL_COUNT),
                ]))),
                "cells.next_stable_id" => Some(StreamInitializer::Explicit(Literal::Array(vec![
                    Literal::U32(INITIAL_COUNT + 1),
                ]))),
                "cells.stable_id" => Some(StreamInitializer::Explicit(Literal::Array(
                    (1..=INITIAL_COUNT).rev().map(Literal::U32).collect(),
                ))),
                "cells.position" => Some(StreamInitializer::Grid2D {
                    origin: Literal::Vector(vec![
                        Literal::f32(-0.15),
                        Literal::f32(-0.15),
                        Literal::f32(0.0),
                    ]),
                    column_step: Literal::Vector(vec![
                        Literal::f32(0.02),
                        Literal::f32(0.0),
                        Literal::f32(0.0),
                    ]),
                    row_step: Literal::Vector(vec![
                        Literal::f32(0.0),
                        Literal::f32(0.02),
                        Literal::f32(0.0),
                    ]),
                    columns: 16,
                    count: INITIAL_COUNT,
                }),
                "cells.radius" => Some(repeat(Literal::f32(0.01), INITIAL_COUNT)),
                "cells.energy" => Some(repeat(Literal::f32(4.0), INITIAL_COUNT)),
                "cells.age" => Some(repeat(Literal::U32(0), INITIAL_COUNT)),
                "cells.fate" | "cells.previous_fate" => {
                    Some(repeat(Literal::U32(1), INITIAL_COUNT))
                }
                "cells.phase" => Some(repeat(Literal::U32(2), INITIAL_COUNT)),
                "cells.health" => Some(repeat(Literal::U32(0), INITIAL_COUNT)),
                "cells.fate_confidence" | "cells.time_in_fate" => {
                    Some(repeat(Literal::U32(120), INITIAL_COUNT))
                }
                "cells.recent_surface_exposure" => Some(repeat(Literal::U32(4095), INITIAL_COUNT)),
                "cells.parent_id" | "cells.recent_activator" | "cells.recent_inhibitor" => {
                    Some(repeat(Literal::U32(0), INITIAL_COUNT))
                }
                "cells.color" => Some(repeat(
                    Literal::Vector(vec![
                        Literal::f32(0.8),
                        Literal::f32(0.8),
                        Literal::f32(0.9),
                        Literal::f32(1.0),
                    ]),
                    INITIAL_COUNT,
                )),
                _ => stream.initial.clone(),
            };
        }
        let stream_id = |name: &str| {
            graph
                .resources
                .streams
                .iter()
                .find(|stream| stream.name == name)
                .unwrap()
                .id
        };
        let metric_names = [
            "morphology.population",
            "morphology.component_count",
            "morphology.component_unresolved",
            "morphology.boundary_count",
            "morphology.interior_count",
            "morphology.area_q16",
            "morphology.perimeter_q16",
            "morphology.compactness_q16",
            "population.physical_neighbor_overflow",
            "population.perception_truncation",
        ];
        let radial_id = stream_id("morphology.radial_density");
        let metric_ids = metric_names.map(stream_id);
        let report = Validator::validate(&graph);
        assert!(
            report.is_valid(),
            "developmental metric diagnostics: {:#?}",
            report.diagnostics
        );
        let device = metal::Device::system_default().expect("Metal device");
        let layer = metal::MetalLayer::new();
        layer.set_device(&device);
        layer.set_pixel_format(metal::MTLPixelFormat::BGRA8Unorm);
        let mut state =
            RuntimeState::new(report.validated.unwrap(), device.clone(), layer).unwrap();
        state
            .run_benchmark(
                &device,
                BenchmarkConfig {
                    mode: BenchmarkMode::Headless,
                    runner: BenchmarkRunner::PqoPlan,
                    warmup_ticks: 0,
                    sample_ticks: 2,
                    ..BenchmarkConfig::default()
                },
            )
            .expect("developmental neighborhood ticks must execute");

        let readback = device.new_buffer(
            ((metric_ids.len() + 8) * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let command_buffer = state.queue.new_command_buffer();
        let blit = command_buffer.new_blit_command_encoder();
        for (index, metric) in metric_ids.iter().enumerate() {
            blit.copy_from_buffer(
                &state.stream_buffers[metric.0 as usize][0],
                0,
                &readback,
                (index * 4) as u64,
                4,
            );
        }
        blit.copy_from_buffer(
            &state.stream_buffers[radial_id.0 as usize][0],
            0,
            &readback,
            (metric_ids.len() * 4) as u64,
            8 * 4,
        );
        blit.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        let values = unsafe {
            std::slice::from_raw_parts(readback.contents().cast::<u32>(), metric_ids.len() + 8)
        };
        assert_eq!(values[0], INITIAL_COUNT);
        assert_eq!(values[1], 1, "contact graph should form one component");
        assert_eq!(values[2], 0, "component propagation should converge");
        assert!(
            values[3] > 0,
            "exposed cells should differentiate as boundary: {values:?}"
        );
        assert!(
            values[4] > 0,
            "enclosed cells should differentiate as interior: {values:?}"
        );
        assert!(values[5] > 0 && values[6] > 0 && values[7] > 0);
        assert_eq!(values[8], 0, "physical neighborhood must remain bounded");
        assert_eq!(values[9], 0, "perception must not truncate");
        assert_eq!(
            values[metric_ids.len()..].iter().sum::<u32>(),
            INITIAL_COUNT,
            "radial density bins must account for every active cell"
        );
    }

    #[derive(Debug, PartialEq, Eq)]
    struct DevelopmentProbe {
        metrics: Vec<u32>,
        radial_density: Vec<u32>,
        ledger_bits: Vec<u32>,
        homeostasis_u32: Vec<u32>,
        homeostasis_f32_bits: Vec<u32>,
        logical_state: Vec<u32>,
    }

    fn run_development_probe(config: HelloOrganismConfig, ticks: u32) -> DevelopmentProbe {
        let graph = hello_organism_builder_with_config(config).build().unwrap();
        let stream_id = |name: &str| {
            graph
                .resources
                .streams
                .iter()
                .find(|stream| stream.name == name)
                .unwrap()
                .id
        };
        let metric_ids = [
            "cells.active_count",
            "morphology.population",
            "morphology.component_count",
            "morphology.component_unresolved",
            "morphology.organizer_count",
            "morphology.undifferentiated_count",
            "morphology.boundary_count",
            "morphology.interior_count",
            "morphology.area_q16",
            "morphology.perimeter_q16",
            "morphology.compactness_q16",
            "population.neighbor_overflow",
            "population.physical_neighbor_overflow",
            "population.perception_truncation",
            "deposit.saturation_count",
        ]
        .map(stream_id);
        let radial_id = stream_id("morphology.radial_density");
        let ledger_ids = [
            "ledger.previous_total",
            "ledger.absorbed",
            "ledger.maintenance",
            "ledger.decisions",
            "ledger.motion",
            "ledger.signaling",
            "ledger.division",
            "ledger.environmental_death_loss",
            "ledger.current_total",
            "ledger.residual",
            "ledger.cumulative_residual",
            "environment.nutrient_supply",
            "simulation.tick",
        ]
        .map(stream_id);
        let logical_ids = [
            "cells.stable_id",
            "cells.parent_id",
            "cells.fate",
            "cells.phase",
            "cells.health",
            "cells.event_hash",
            "perception.inhibitor_bin",
        ]
        .map(stream_id);
        let homeostasis_scalar_ids = [
            "homeostasis.reference_samples",
            "homeostasis.validation_samples",
            "homeostasis.validation_violations",
            "homeostasis.invariant_violations",
        ]
        .map(stream_id);
        let homeostasis_metric_min_id = stream_id("homeostasis.metric_min");
        let homeostasis_metric_max_id = stream_id("homeostasis.metric_max");
        let homeostasis_metric_sum_id = stream_id("homeostasis.metric_sum");
        let homeostasis_metric_sum_sq_id = stream_id("homeostasis.metric_sum_sq");
        let homeostasis_energy_ids = [
            "homeostasis.energy_min",
            "homeostasis.energy_max",
            "homeostasis.energy_sum",
            "homeostasis.energy_sum_sq",
            "homeostasis.perturbation_energy_min",
        ]
        .map(stream_id);
        let report = Validator::validate(&graph);
        assert!(
            report.is_valid(),
            "Hello Organism diagnostics: {:#?}",
            report.diagnostics
        );
        let device = metal::Device::system_default().expect("Metal device");
        let layer = metal::MetalLayer::new();
        layer.set_device(&device);
        layer.set_pixel_format(metal::MTLPixelFormat::BGRA8Unorm);
        let mut state =
            RuntimeState::new(report.validated.unwrap(), device.clone(), layer).unwrap();
        state
            .run_benchmark(
                &device,
                BenchmarkConfig {
                    mode: BenchmarkMode::Headless,
                    runner: BenchmarkRunner::PqoPlan,
                    warmup_ticks: 0,
                    sample_ticks: ticks,
                    ..BenchmarkConfig::default()
                },
            )
            .expect("one organizer must execute the developmental program");

        let homeostasis_u32_words = homeostasis_scalar_ids.len() + 2 * 16;
        let homeostasis_f32_words = 2 * 16 + homeostasis_energy_ids.len();
        let metric_words =
            metric_ids.len() + 8 + ledger_ids.len() + homeostasis_u32_words + homeostasis_f32_words;
        let logical_words = logical_ids.len() * config.capacity as usize;
        let readback = device.new_buffer(
            ((metric_words + logical_words) * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let command_buffer = state.queue.new_command_buffer();
        let blit = command_buffer.new_blit_command_encoder();
        for (index, metric) in metric_ids.iter().enumerate() {
            blit.copy_from_buffer(
                &state.stream_buffers[metric.0 as usize][0],
                0,
                &readback,
                (index * 4) as u64,
                4,
            );
        }
        blit.copy_from_buffer(
            &state.stream_buffers[radial_id.0 as usize][0],
            0,
            &readback,
            (metric_ids.len() * 4) as u64,
            8 * 4,
        );
        for (index, ledger) in ledger_ids.iter().enumerate() {
            blit.copy_from_buffer(
                &state.stream_buffers[ledger.0 as usize][0],
                0,
                &readback,
                ((metric_ids.len() + 8 + index) * 4) as u64,
                4,
            );
        }
        let homeostasis_u32_offset = metric_ids.len() + 8 + ledger_ids.len();
        for (index, scalar) in homeostasis_scalar_ids.iter().enumerate() {
            blit.copy_from_buffer(
                &state.stream_buffers[scalar.0 as usize][0],
                0,
                &readback,
                ((homeostasis_u32_offset + index) * 4) as u64,
                4,
            );
        }
        let metric_min_offset = homeostasis_u32_offset + homeostasis_scalar_ids.len();
        blit.copy_from_buffer(
            &state.stream_buffers[homeostasis_metric_min_id.0 as usize][0],
            0,
            &readback,
            (metric_min_offset * 4) as u64,
            16 * 4,
        );
        let metric_max_offset = metric_min_offset + 16;
        blit.copy_from_buffer(
            &state.stream_buffers[homeostasis_metric_max_id.0 as usize][0],
            0,
            &readback,
            (metric_max_offset * 4) as u64,
            16 * 4,
        );
        let homeostasis_f32_offset = homeostasis_u32_offset + homeostasis_u32_words;
        blit.copy_from_buffer(
            &state.stream_buffers[homeostasis_metric_sum_id.0 as usize][0],
            0,
            &readback,
            (homeostasis_f32_offset * 4) as u64,
            16 * 4,
        );
        blit.copy_from_buffer(
            &state.stream_buffers[homeostasis_metric_sum_sq_id.0 as usize][0],
            0,
            &readback,
            ((homeostasis_f32_offset + 16) * 4) as u64,
            16 * 4,
        );
        for (index, energy) in homeostasis_energy_ids.iter().enumerate() {
            blit.copy_from_buffer(
                &state.stream_buffers[energy.0 as usize][0],
                0,
                &readback,
                ((homeostasis_f32_offset + 32 + index) * 4) as u64,
                4,
            );
        }
        for (index, stream) in logical_ids.iter().enumerate() {
            blit.copy_from_buffer(
                &state.stream_buffers[stream.0 as usize][0],
                0,
                &readback,
                ((metric_words + index * config.capacity as usize) * 4) as u64,
                u64::from(config.capacity) * 4,
            );
        }
        blit.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        let words = unsafe {
            std::slice::from_raw_parts(
                readback.contents().cast::<u32>(),
                metric_words + logical_words,
            )
        };
        let active_count = words[0] as usize;
        let logical_state = (0..logical_ids.len())
            .flat_map(|stream| {
                let start = metric_words + stream * config.capacity as usize;
                words[start..start + active_count].iter().copied()
            })
            .collect();
        DevelopmentProbe {
            metrics: words[..metric_ids.len()].to_vec(),
            radial_density: words[metric_ids.len()..metric_ids.len() + 8].to_vec(),
            ledger_bits: words[metric_ids.len() + 8..metric_ids.len() + 8 + ledger_ids.len()]
                .to_vec(),
            homeostasis_u32: words
                [homeostasis_u32_offset..homeostasis_u32_offset + homeostasis_u32_words]
                .to_vec(),
            homeostasis_f32_bits: words
                [homeostasis_f32_offset..homeostasis_f32_offset + homeostasis_f32_words]
                .to_vec(),
            logical_state,
        }
    }

    #[test]
    fn one_organizer_constructs_a_connected_differentiated_body() {
        const CAPACITY: u32 = 1_024;
        const DEADLINE: u32 = 3_200;
        let probe = run_development_probe(HelloOrganismConfig::reference(CAPACITY), DEADLINE);
        let metrics = &probe.metrics;
        assert!(
            (24..=48).contains(&metrics[1]),
            "development must reach the declared broad envelope: {probe:?}"
        );
        assert_eq!(metrics[2], 1, "the constructed contact graph must connect");
        assert_eq!(metrics[3], 0, "component propagation must converge");
        assert_eq!(metrics[4], 1, "organizer fate must remain unique");
        assert!(metrics[6] > 0, "boundary tissue must differentiate");
        assert!(metrics[7] > 0, "interior tissue must differentiate");
        assert_eq!(
            (metrics[1], metrics[6], metrics[7]),
            (39, 16, 22),
            "the deterministic reference morphology changed: {probe:?}"
        );
        assert!(metrics[8] > 0 && metrics[9] > 0 && metrics[10] > 0);
        assert_eq!(
            probe.radial_density.iter().sum::<u32>(),
            metrics[1],
            "radial bins must account for the observed body"
        );
        assert_eq!(
            &metrics[11..],
            &[0, 0, 0, 0],
            "reference development must remain within all declared bounds"
        );
        let ledger = probe.ledger_bits[..12]
            .iter()
            .map(|bits| f32::from_bits(*bits))
            .collect::<Vec<_>>();
        assert_eq!(probe.ledger_bits[12], DEADLINE);
        assert!(
            ledger[9].abs() <= 0.001 && ledger[10].abs() <= 0.1,
            "energy accounting residuals must remain bounded: {ledger:?}"
        );
        assert!(
            (metrics[1] as f32..=metrics[1] as f32 * 5.0).contains(&ledger[8]),
            "total energy must remain locally bounded: {ledger:?}"
        );
    }

    #[test]
    fn developmental_fields_are_causal_and_logical_replay_is_exact() {
        const CAPACITY: u32 = 1_024;
        const DEADLINE: u32 = 3_200;
        let reference = run_development_probe(HelloOrganismConfig::reference(CAPACITY), DEADLINE);
        let replay = run_development_probe(HelloOrganismConfig::reference(CAPACITY), DEADLINE);
        assert_eq!(
            replay, reference,
            "the complete logical and integer morphology state must replay exactly"
        );

        let without_activator = run_development_probe(
            HelloOrganismConfig {
                capacity: CAPACITY,
                activator_transport: false,
                inhibitor_transport: true,
            },
            DEADLINE,
        );
        assert!(
            without_activator.metrics[1] <= 8 && without_activator.metrics[7] == 0,
            "without activator transport the differentiated body must not form: \
             {without_activator:?}"
        );

        let without_inhibitor = run_development_probe(
            HelloOrganismConfig {
                capacity: CAPACITY,
                activator_transport: true,
                inhibitor_transport: false,
            },
            DEADLINE,
        );
        assert!(
            without_inhibitor.metrics[1] > 48,
            "without inhibitor transport growth must leave the reference envelope: \
             reference={reference:?}, inhibitor_off={without_inhibitor:?}"
        );
    }

    #[test]
    fn organism_sustains_a_bounded_homeostatic_state() {
        const CAPACITY: u32 = 1_024;
        const DEADLINE: u32 = 30_000;
        let probe = run_development_probe(HelloOrganismConfig::reference(CAPACITY), DEADLINE);
        let ledger = probe.ledger_bits[..12]
            .iter()
            .map(|bits| f32::from_bits(*bits))
            .collect::<Vec<_>>();
        assert_eq!(
            (probe.metrics[1], probe.metrics[2], probe.metrics[4]),
            (39, 1, 1),
            "homeostasis must retain the reference connected organism: {probe:?}"
        );
        assert_eq!((probe.metrics[6], probe.metrics[7]), (16, 22));
        assert_eq!(&probe.metrics[11..], &[0, 0, 0, 0]);
        assert_eq!(probe.ledger_bits[12], DEADLINE);
        assert_eq!(
            &probe.homeostasis_u32[..4],
            &[1_000, 1_000, 0, 0],
            "the reference and validation windows must both pass completely: {probe:?}"
        );
        let metric_min = &probe.homeostasis_u32[4..20];
        let metric_max = &probe.homeostasis_u32[20..36];
        assert!(
            metric_min
                .iter()
                .zip(metric_max)
                .all(|(minimum, maximum)| minimum <= maximum),
            "every morphology envelope must be initialized: {probe:?}"
        );
        let homeostasis = probe
            .homeostasis_f32_bits
            .iter()
            .map(|bits| f32::from_bits(*bits))
            .collect::<Vec<_>>();
        let energy_min = homeostasis[32];
        let energy_max = homeostasis[33];
        let energy_mean = homeostasis[34] / 1_000.0;
        assert!(
            energy_min.is_finite()
                && energy_max.is_finite()
                && energy_min <= energy_mean + 0.01
                && energy_mean <= energy_max + 0.01,
            "the reference energy envelope must be finite and ordered: {homeostasis:?}"
        );
        assert!(
            ledger[9].abs() <= 0.001 && (ledger[10] / DEADLINE as f32).abs() <= 0.000_02,
            "per-tick and mean cumulative accounting residuals must remain bounded: {ledger:?}"
        );
        assert!(
            (probe.metrics[1] as f32..=probe.metrics[1] as f32 * 5.0).contains(&ledger[8]),
            "energy must remain locally bounded without a global corrective clamp: {ledger:?}"
        );
    }

    #[test]
    fn organism_returns_to_its_reference_envelope_after_nutrient_perturbation() {
        const CAPACITY: u32 = 1_024;
        const DEADLINE: u64 = 30_000;
        let graph = hello_organism_builder_with_config(HelloOrganismConfig::reference(CAPACITY))
            .build()
            .unwrap();
        let stream_id = |name: &str| {
            graph
                .resources
                .streams
                .iter()
                .find(|stream| stream.name == name)
                .unwrap()
                .id
        };
        let u32_ids = [
            "simulation.tick",
            "morphology.population",
            "morphology.component_count",
            "morphology.component_unresolved",
            "morphology.organizer_count",
            "morphology.boundary_count",
            "morphology.interior_count",
            "population.neighbor_overflow",
            "population.physical_neighbor_overflow",
            "population.perception_truncation",
            "deposit.saturation_count",
            "homeostasis.reference_samples",
            "homeostasis.validation_samples",
            "homeostasis.validation_violations",
            "homeostasis.invariant_violations",
        ]
        .map(stream_id);
        let f32_ids = [
            "environment.nutrient_supply",
            "ledger.current_total",
            "ledger.residual",
            "ledger.cumulative_residual",
            "homeostasis.energy_min",
            "homeostasis.perturbation_energy_min",
        ]
        .map(stream_id);
        let scenario = graph
            .scenarios
            .iter()
            .find(|scenario| scenario.name == "homeostasis_perturbation")
            .cloned()
            .unwrap();
        let report = Validator::validate(&graph);
        assert!(
            report.is_valid(),
            "Gate 4 perturbation diagnostics: {:#?}",
            report.diagnostics
        );
        let device = metal::Device::system_default().expect("Metal device");
        let layer = metal::MetalLayer::new();
        layer.set_device(&device);
        layer.set_pixel_format(metal::MTLPixelFormat::BGRA8Unorm);
        let mut state =
            RuntimeState::new(report.validated.unwrap(), device.clone(), layer).unwrap();
        let events = state
            .execute_scenario(&device, &scenario, DEADLINE)
            .expect("recorded nutrient perturbation must execute");
        assert_eq!(events.len(), 2);
        assert_eq!((events[0].tick, events[1].tick), (12_000, 14_000));
        assert!(
            events
                .iter()
                .all(|event| event.pass == "set_nutrient_supply"
                    && event.value_overrides == vec!["environment.reference_nutrient_supply"]),
            "both environmental changes must be canonical recorded interventions: {events:?}"
        );

        let word_count = u32_ids.len() + f32_ids.len();
        let readback = device.new_buffer(
            (word_count * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let command_buffer = state.queue.new_command_buffer();
        let blit = command_buffer.new_blit_command_encoder();
        for (index, stream) in u32_ids.iter().chain(f32_ids.iter()).enumerate() {
            blit.copy_from_buffer(
                &state.stream_buffers[stream.0 as usize][0],
                0,
                &readback,
                (index * 4) as u64,
                4,
            );
        }
        blit.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        let words =
            unsafe { std::slice::from_raw_parts(readback.contents().cast::<u32>(), word_count) };
        let floats = words[u32_ids.len()..]
            .iter()
            .map(|bits| f32::from_bits(*bits))
            .collect::<Vec<_>>();
        assert_eq!(
            &words[..u32_ids.len()],
            &[30_000, 39, 1, 0, 1, 16, 22, 0, 0, 0, 0, 1_000, 1_000, 0, 0,],
            "the organism must recover its connected differentiated reference envelope"
        );
        assert_eq!(floats[0], 1.0, "the nutrient supply must be restored");
        assert!(
            (39.0..=195.0).contains(&floats[1])
                && floats[2].abs() <= 0.001
                && (floats[3] / DEADLINE as f32).abs() <= 0.000_02
                && floats[5] < floats[4],
            "the recovered organism must retain bounded, auditable energy: {floats:?}"
        );
    }

    #[derive(Debug)]
    struct RegenerationProbe {
        scalars: Vec<u32>,
        removed_ids: Vec<u32>,
    }

    fn regeneration_probe(
        graph: &pqo_core::ModuleGraph,
        state: &RuntimeState,
        device: &metal::Device,
    ) -> RegenerationProbe {
        let stream_id = |name: &str| {
            graph
                .resources
                .streams
                .iter()
                .find(|stream| stream.name == name)
                .unwrap()
                .id
        };
        let scalar_ids = [
            "simulation.tick",
            "morphology.population",
            "morphology.component_count",
            "morphology.component_unresolved",
            "morphology.organizer_count",
            "morphology.boundary_count",
            "morphology.interior_count",
            "morphology.area_q16",
            "morphology.compactness_q16",
            "lesion.removed_count",
            "lesion.damaged_count",
            "lesion.region_occupancy",
            "regeneration.injury_total_q16",
            "regeneration.post_lesion_peak_q16",
            "regeneration.consecutive_ticks",
            "regeneration.success_tick",
            "environment.injury_transport",
            "environment.repair_enabled",
            "ledger.residual",
            "lesion.removed_energy",
            "ledger.current_total",
        ]
        .map(stream_id);
        let removed_ids = stream_id("lesion.removed_ids");
        let word_count = scalar_ids.len() + 64;
        let readback = device.new_buffer(
            (word_count * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let command_buffer = state.queue.new_command_buffer();
        let blit = command_buffer.new_blit_command_encoder();
        for (index, stream) in scalar_ids.iter().enumerate() {
            blit.copy_from_buffer(
                &state.stream_buffers[stream.0 as usize][0],
                0,
                &readback,
                (index * 4) as u64,
                4,
            );
        }
        blit.copy_from_buffer(
            &state.stream_buffers[removed_ids.0 as usize][0],
            0,
            &readback,
            (scalar_ids.len() * 4) as u64,
            64 * 4,
        );
        blit.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        let words =
            unsafe { std::slice::from_raw_parts(readback.contents().cast::<u32>(), word_count) };
        RegenerationProbe {
            scalars: words[..scalar_ids.len()].to_vec(),
            removed_ids: words[scalar_ids.len()..].to_vec(),
        }
    }

    fn checkpoint_signature(state: &RuntimeState, device: &metal::Device) -> Vec<u8> {
        let length = state
            .stream_buffers
            .iter()
            .flatten()
            .map(|buffer| buffer.length())
            .sum::<u64>();
        let readback = device.new_buffer(length, metal::MTLResourceOptions::StorageModeShared);
        let command_buffer = state.queue.new_command_buffer();
        let blit = command_buffer.new_blit_command_encoder();
        let mut offset = 0;
        for buffer in state.stream_buffers.iter().flatten() {
            blit.copy_from_buffer(buffer, 0, &readback, offset, buffer.length());
            offset += buffer.length();
        }
        blit.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        unsafe {
            std::slice::from_raw_parts(readback.contents().cast::<u8>(), length as usize).to_vec()
        }
    }

    #[test]
    fn committed_homeostatic_checkpoint_branches_into_causal_regeneration_proofs() {
        const CAPACITY: u32 = 1_024;
        const CHECKPOINT_TICK: u64 = 30_000;
        const RECOVERY_TICKS: u64 = 20_000;
        let graph = hello_organism_builder(CAPACITY).build().unwrap();
        let scenario = |name: &str| {
            graph
                .scenarios
                .iter()
                .find(|scenario| scenario.name == name)
                .cloned()
                .unwrap()
        };
        let control_scenario = scenario("regeneration_control");
        let lesion_scenario = scenario("structural_regeneration");
        let no_injury_scenario = scenario("regeneration_without_injury");
        let no_repair_scenario = scenario("regeneration_without_repair");
        let report = Validator::validate(&graph);
        assert!(
            report.is_valid(),
            "Gate 5 diagnostics: {:#?}",
            report.diagnostics
        );
        let device = metal::Device::system_default().expect("Metal device");
        let layer = metal::MetalLayer::new();
        layer.set_device(&device);
        layer.set_pixel_format(metal::MTLPixelFormat::BGRA8Unorm);
        let mut control =
            RuntimeState::new(report.validated.unwrap(), device.clone(), layer).unwrap();
        control
            .run_benchmark(
                &device,
                BenchmarkConfig {
                    mode: BenchmarkMode::Headless,
                    runner: BenchmarkRunner::PqoPlan,
                    warmup_ticks: 0,
                    sample_ticks: CHECKPOINT_TICK as u32,
                    ..BenchmarkConfig::default()
                },
            )
            .expect("the homeostatic checkpoint must execute");

        let mut lesion = control.fork_checkpoint(&device).unwrap();
        let mut no_injury = control.fork_checkpoint(&device).unwrap();
        let mut no_repair = control.fork_checkpoint(&device).unwrap();
        let checkpoint = checkpoint_signature(&control, &device);
        eprintln!("Gate 5 checkpoint_sha256={}", sha256(&checkpoint));
        assert_eq!(checkpoint_signature(&lesion, &device), checkpoint);
        assert_eq!(checkpoint_signature(&no_injury, &device), checkpoint);
        assert_eq!(checkpoint_signature(&no_repair, &device), checkpoint);

        let control_events = control
            .execute_scenario_range(&device, &control_scenario, CHECKPOINT_TICK, RECOVERY_TICKS)
            .unwrap();
        let lesion_events = lesion
            .execute_scenario_range(&device, &lesion_scenario, CHECKPOINT_TICK, RECOVERY_TICKS)
            .unwrap();
        let no_injury_events = no_injury
            .execute_scenario_range(
                &device,
                &no_injury_scenario,
                CHECKPOINT_TICK,
                RECOVERY_TICKS,
            )
            .unwrap();
        let no_repair_events = no_repair
            .execute_scenario_range(
                &device,
                &no_repair_scenario,
                CHECKPOINT_TICK,
                RECOVERY_TICKS,
            )
            .unwrap();
        assert!(control_events.is_empty());
        assert_eq!(lesion_events.len(), 2);
        assert_eq!(no_injury_events.len(), 3);
        assert_eq!(no_repair_events.len(), 3);
        eprintln!(
            "Gate 5 lesion_events_sha256={} no_injury_events_sha256={} \
             no_repair_events_sha256={}",
            sha256(serde_json::to_vec(&lesion_events).unwrap()),
            sha256(serde_json::to_vec(&no_injury_events).unwrap()),
            sha256(serde_json::to_vec(&no_repair_events).unwrap())
        );
        assert!(
            lesion_events
                .iter()
                .all(|event| event.tick == CHECKPOINT_TICK),
            "the lesion must be a canonical tick-addressed event: {lesion_events:?}"
        );

        let control = regeneration_probe(&graph, &control, &device);
        let lesion = regeneration_probe(&graph, &lesion, &device);
        let no_injury = regeneration_probe(&graph, &no_injury, &device);
        let no_repair = regeneration_probe(&graph, &no_repair, &device);
        eprintln!(
            "Gate 5 control={:?} lesion={:?} no_injury={:?} no_repair={:?}",
            control.scalars, lesion.scalars, no_injury.scalars, no_repair.scalars
        );
        assert_eq!(&control.scalars[..7], &[50_000, 39, 1, 0, 1, 16, 22]);
        assert_eq!(control.scalars[9], 0);
        assert!(
            (8..=12).contains(&lesion.scalars[9]),
            "the reference lesion must remove 20–30% of the body: {lesion:?}"
        );
        assert!(lesion.scalars[10] > 0, "the wound shell must be damaged");
        assert!(
            lesion.removed_ids[..lesion.scalars[9] as usize]
                .windows(2)
                .all(|ids| ids[0] < ids[1]),
            "removed stable IDs must be canonical: {lesion:?}"
        );
        assert!(
            f32::from_bits(lesion.scalars[19]) > 0.0,
            "lesion energy must be recorded as environmental loss"
        );
        assert_eq!(no_injury.scalars[9], lesion.scalars[9]);
        assert_eq!(no_repair.scalars[9], lesion.scalars[9]);
        let removed = lesion.scalars[9] as usize;
        assert_eq!(
            &no_injury.removed_ids[..removed],
            &lesion.removed_ids[..removed]
        );
        assert_eq!(
            &no_repair.removed_ids[..removed],
            &lesion.removed_ids[..removed]
        );
        assert_eq!(f32::from_bits(lesion.scalars[16]), 1.0);
        assert_eq!(lesion.scalars[17], 1);
        assert_eq!(f32::from_bits(no_injury.scalars[16]), 0.0);
        assert_eq!(no_injury.scalars[17], 1);
        assert_eq!(f32::from_bits(no_repair.scalars[16]), 1.0);
        assert_eq!(no_repair.scalars[17], 0);
        assert_eq!(
            no_injury.scalars[13], 0,
            "disabling injury transport must eliminate the distributed injury signal"
        );
        assert!(
            (36..=43).contains(&lesion.scalars[1])
                && lesion.scalars[2] == 1
                && lesion.scalars[3] == 0
                && lesion.scalars[4] == 1,
            "the repaired body must recover population, connectivity, and organizer identity: \
             {lesion:?}"
        );
        assert!(
            lesion.scalars[11] * 10 >= lesion.scalars[9] * 9,
            "the original wound region must close: {lesion:?}"
        );
        assert!(
            lesion.scalars[15] > CHECKPOINT_TICK as u32
                && lesion.scalars[15] <= 50_000
                && lesion.scalars[14] >= 500,
            "the reference branch must sustain its recovery envelope: {lesion:?}"
        );
        assert!(
            lesion.scalars[12] * 20 <= lesion.scalars[13]
                && f32::from_bits(lesion.scalars[18]).abs() <= 0.001,
            "injury and energy accounting must resolve: {lesion:?}"
        );
        assert_eq!(
            no_injury.scalars[15], 0,
            "without injury transport the causal regeneration criterion must fail: \
             {no_injury:?}"
        );
        assert_eq!(
            no_repair.scalars[15], 0,
            "without repair behavior sensing alone must not regenerate: {no_repair:?}"
        );
    }

    fn decode_f32(bytes: &[u8]) -> Vec<f32> {
        bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect()
    }
}
