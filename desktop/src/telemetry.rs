//! Opt-in local telemetry for diagnosing hangs and slowdowns of the main thread.
//!
//! Enabled by the `RUFFLE_TELEMETRY` environment variable: `1` writes to a new directory
//! under `<cache>/telemetry/`, any other value is used as the directory. It contains:
//!
//! - `metrics.jsonl`: one line per second with timings of the main loop and memory usage.
//! - `stalls.log`: whenever the main thread hasn't returned to the event loop for
//!   [`STALL_THRESHOLD`], what it's busy with, native stacks of the main thread
//!   (taken with `eu-stack`) and, if it's running ActionScript, the AVM2 call stack.
//! - `avm2_profile.txt`: a sampling profile of the ActionScript running on the main thread,
//!   rewritten every [`PROFILE_WRITE_INTERVAL`].
//! - `orphan_classes.txt`: the classes of the display objects in the orphan list,
//!   rewritten every [`CLASS_CENSUS_INTERVAL`].

use ruffle_core::Player;
use ruffle_core::telemetry::{DisplayCensus, GotoStats, TickStats};
use std::collections::HashMap;
use std::fmt::Write as _;
use std::fs::File;
use std::io::{LineWriter, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

/// How long the main thread may stay in a handler before it's reported as stalled.
const STALL_THRESHOLD: Duration = Duration::from_secs(1);

/// How long into a stall to take stack samples.
const SAMPLE_AT_SECS: [u64; 8] = [1, 2, 4, 7, 11, 16, 25, 40];

/// How long the main thread may stay outside of our handlers before it's reported.
/// While a movie is playing, it wakes up at least once per frame.
const IDLE_THRESHOLD: Duration = Duration::from_secs(3);

/// How often the watchdog checks on the main thread, and samples the AVM2 call stack while it's busy.
const POLL_INTERVAL: Duration = Duration::from_millis(10);

/// Samples taken this long into a handler are also counted in the "slow handlers" profile.
const SLOW_HANDLER: Duration = Duration::from_millis(100);

const PROFILE_WRITE_INTERVAL: Duration = Duration::from_secs(30);

/// How many innermost frames of a call stack are kept in the profile.
const PROFILE_DEPTH: usize = 16;

const METRICS_INTERVAL: Duration = Duration::from_secs(1);

/// How often to count the objects on the display list; this walks all of it.
const CENSUS_INTERVAL: Duration = Duration::from_secs(5);

const CLASS_CENSUS_INTERVAL: Duration = Duration::from_secs(30);

/// What the main thread is doing.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum Phase {
    Idle,
    NewEvents,
    WindowEvent,
    Redraw,
    Render,
    Gui,
    UserEvent,
    Task,
    AboutToWait,
    Tick,
    IdleGc,
}

impl Phase {
    const ALL: [Phase; 11] = [
        Phase::Idle,
        Phase::NewEvents,
        Phase::WindowEvent,
        Phase::Redraw,
        Phase::Render,
        Phase::Gui,
        Phase::UserEvent,
        Phase::Task,
        Phase::AboutToWait,
        Phase::Tick,
        Phase::IdleGc,
    ];

    fn name(self) -> &'static str {
        match self {
            Phase::Idle => "idle",
            Phase::NewEvents => "new_events",
            Phase::WindowEvent => "window_event",
            Phase::Redraw => "redraw",
            Phase::Render => "render",
            Phase::Gui => "gui",
            Phase::UserEvent => "user_event",
            Phase::Task => "task",
            Phase::AboutToWait => "about_to_wait",
            Phase::Tick => "tick",
            Phase::IdleGc => "idle_gc",
        }
    }
}

#[derive(Clone, Copy, Default)]
struct PhaseStats {
    count: u32,
    total: Duration,
    max: Duration,
}

impl PhaseStats {
    fn record(&mut self, elapsed: Duration) {
        self.count += 1;
        self.total += elapsed;
        self.max = self.max.max(elapsed);
    }
}

#[derive(Default)]
struct TickParts {
    frames: u32,
    run_frame: Duration,
    sockets: Duration,
    net_connections: Duration,
    timers: Duration,
    streams: Duration,
    audio: Duration,
}

impl TickParts {
    fn record(&mut self, stats: TickStats) {
        self.frames += stats.frames;
        self.run_frame = self.run_frame.max(stats.run_frame);
        self.sockets = self.sockets.max(stats.sockets);
        self.net_connections = self.net_connections.max(stats.net_connections);
        self.timers = self.timers.max(stats.timers);
        self.streams = self.streams.max(stats.streams);
        self.audio = self.audio.max(stats.audio);
    }
}

/// Aggregates of the current metrics interval, only touched by the main thread.
struct Metrics {
    since: Instant,
    phases: [PhaseStats; Phase::ALL.len()],
    tick: TickParts,
    gc_objects: usize,
    census: DisplayCensus,
    census_taken: Option<Instant>,
    class_census_taken: Option<Instant>,
    writer: LineWriter<File>,
}

struct Telemetry {
    start: Instant,
    dir: PathBuf,
    pid: u32,
    /// Microseconds since `start` at which the main thread entered its outermost handler,
    /// plus one; zero while it's idle.
    busy_since: AtomicU64,
    /// Microseconds since `start` at which the main thread last left its outermost handler.
    idle_since: AtomicU64,
    phase: AtomicU8,
    metrics: Mutex<Metrics>,
    stalls: Mutex<LineWriter<File>>,
}

static TELEMETRY: OnceLock<Telemetry> = OnceLock::new();

fn ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

impl Telemetry {
    fn now_us(&self) -> u64 {
        self.start.elapsed().as_micros() as u64
    }

    fn secs(&self) -> f64 {
        self.start.elapsed().as_secs_f64()
    }

    fn log_stall(&self, text: &str) {
        if let Ok(mut stalls) = self.stalls.lock() {
            let _ = stalls.write_all(text.as_bytes());
        }
    }
}

/// Starts telemetry if `RUFFLE_TELEMETRY` is set.
pub fn init(cache_directory: &Path) {
    let Some(value) = std::env::var_os("RUFFLE_TELEMETRY") else {
        return;
    };
    let dir = if value == "1" {
        cache_directory
            .join("telemetry")
            .join(chrono::Local::now().format("%F_%H-%M-%S").to_string())
    } else {
        PathBuf::from(value)
    };

    if let Err(err) = start(&dir) {
        tracing::error!("Failed to start telemetry in {}: {err}", dir.display());
    }
}

fn start(dir: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(dir)?;
    let metrics = LineWriter::new(File::create(dir.join("metrics.jsonl"))?);
    let mut stalls = LineWriter::new(File::create(dir.join("stalls.log"))?);

    let pid = std::process::id();
    let args: Vec<String> = std::env::args().collect();
    writeln!(
        stalls,
        "Ruffle {} telemetry, pid {pid}, started {}\nargs: {args:?}\n",
        env!("CARGO_PKG_VERSION"),
        chrono::Local::now().to_rfc3339(),
    )?;

    let start = Instant::now();
    let telemetry = Telemetry {
        start,
        dir: dir.to_owned(),
        pid,
        busy_since: AtomicU64::new(0),
        idle_since: AtomicU64::new(0),
        phase: AtomicU8::new(Phase::Idle as u8),
        metrics: Mutex::new(Metrics {
            since: start,
            phases: Default::default(),
            tick: Default::default(),
            gc_objects: 0,
            census: Default::default(),
            census_taken: None,
            class_census_taken: None,
            writer: metrics,
        }),
        stalls: Mutex::new(stalls),
    };
    if TELEMETRY.set(telemetry).is_err() {
        return Ok(());
    }

    std::thread::Builder::new()
        .name("telemetry-watchdog".to_string())
        .spawn(watchdog)?;

    tracing::info!("Telemetry enabled, writing to {}", dir.display());
    Ok(())
}

/// Marks the main thread as busy with `phase` until the returned guard is dropped.
#[must_use]
pub fn scope(phase: Phase) -> Option<Scope> {
    let telemetry = TELEMETRY.get()?;
    let prev = Phase::ALL[telemetry.phase.swap(phase as u8, Ordering::Relaxed) as usize];
    if prev == Phase::Idle {
        telemetry
            .busy_since
            .store(telemetry.now_us() + 1, Ordering::Release);
    }
    Some(Scope {
        telemetry,
        phase,
        prev,
        entered: Instant::now(),
    })
}

pub struct Scope {
    telemetry: &'static Telemetry,
    phase: Phase,
    prev: Phase,
    entered: Instant,
}

impl Drop for Scope {
    fn drop(&mut self) {
        let telemetry = self.telemetry;
        let elapsed = self.entered.elapsed();
        telemetry.phase.store(self.prev as u8, Ordering::Relaxed);
        if let Ok(mut metrics) = telemetry.metrics.lock() {
            metrics.phases[self.phase as usize].record(elapsed);
            // Only between handlers, so that each one is counted in the interval it ends in.
            if self.prev == Phase::Idle && metrics.since.elapsed() >= METRICS_INTERVAL {
                metrics.write(telemetry);
            }
        }

        if self.prev == Phase::Idle {
            telemetry
                .idle_since
                .store(telemetry.now_us(), Ordering::Relaxed);
            telemetry.busy_since.store(0, Ordering::Release);

            if elapsed >= STALL_THRESHOLD {
                telemetry.log_stall(&format!(
                    "[{:.3}s] main thread back in the event loop after {:.0} ms in {}\n\n",
                    telemetry.secs(),
                    ms(elapsed),
                    self.phase.name(),
                ));
            }
        }
    }
}

/// Records the last tick of `player`.
pub fn record_tick(player: &mut Player) {
    let Some(telemetry) = TELEMETRY.get() else {
        return;
    };
    let Ok(mut metrics) = telemetry.metrics.lock() else {
        return;
    };
    metrics.tick.record(player.last_tick_stats());
    metrics.gc_objects = player.gc_object_count();

    let due = |taken: Option<Instant>, interval| taken.is_none_or(|t| t.elapsed() >= interval);
    if due(metrics.census_taken, CENSUS_INTERVAL) {
        let with_classes = due(metrics.class_census_taken, CLASS_CENSUS_INTERVAL);
        let start = Instant::now();
        metrics.census = player.display_census(with_classes);
        metrics.census_taken = Some(Instant::now());
        if with_classes {
            metrics.class_census_taken = metrics.census_taken;
            write_orphan_classes(telemetry, &metrics.census, start.elapsed());
        }
    }
}

fn write_orphan_classes(telemetry: &Telemetry, census: &DisplayCensus, took: Duration) {
    let mut out = format!(
        "at {:.3}s (census took {:.1} ms): {} objects on stage, {} orphan entries, \
         {} live orphans with {} objects in total\n\n",
        telemetry.secs(),
        ms(took),
        census.stage_objects,
        census.orphan_entries,
        census.live_orphans,
        census.orphan_objects,
    );
    for (class, count) in census.orphan_classes.iter().take(60) {
        let _ = writeln!(out, "{count:8}  {class}");
    }
    let path = telemetry.dir.join("orphan_classes.txt");
    if let Err(err) = std::fs::write(&path, out) {
        tracing::warn!("Failed to write {}: {err}", path.display());
    }
}

impl Metrics {
    /// Writes the metrics of the current interval and starts the next one.
    fn write(&mut self, telemetry: &Telemetry) {
        let interval = self.since.elapsed();
        let mut line = format!(
            "{{\"t\":{:.3},\"interval_ms\":{:.1},\"fps\":{:.1}",
            telemetry.secs(),
            ms(interval),
            f64::from(self.tick.frames) / interval.as_secs_f64(),
        );
        for (phase, stats) in Phase::ALL.iter().zip(&self.phases) {
            if stats.count > 0 {
                let _ = write!(
                    line,
                    ",\"{}\":{{\"n\":{},\"total_ms\":{:.2},\"max_ms\":{:.2}}}",
                    phase.name(),
                    stats.count,
                    ms(stats.total),
                    ms(stats.max),
                );
            }
        }
        let tick = &self.tick;
        let _ = write!(
            line,
            ",\"tick_max_ms\":{{\"run_frame\":{:.2},\"sockets\":{:.2},\"net_connections\":{:.2},\"timers\":{:.2},\"streams\":{:.2},\"audio\":{:.2}}}",
            ms(tick.run_frame),
            ms(tick.sockets),
            ms(tick.net_connections),
            ms(tick.timers),
            ms(tick.streams),
            ms(tick.audio),
        );
        let gotos = ruffle_core::telemetry::take_goto_stats();
        let _ = write!(
            line,
            ",\"goto\":{{\"n\":{},\"settled_skips\":{},\"settled_skip_violations\":{}",
            gotos.count, gotos.settled_skips, gotos.settled_skip_violations
        );
        for (name, part) in GotoStats::PART_NAMES.iter().zip(gotos.parts) {
            let _ = write!(line, ",\"{name}_ms\":{:.2}", ms(part));
        }
        let census = &self.census;
        let _ = writeln!(
            line,
            "}},\"stage_objects\":{},\"orphan_entries\":{},\"live_orphans\":{},\"orphan_objects\":{},\"gc_objects\":{},\"rss_mb\":{:.1}}}",
            census.stage_objects,
            census.orphan_entries,
            census.live_orphans,
            census.orphan_objects,
            self.gc_objects,
            rss_bytes().unwrap_or(0) as f64 / (1024.0 * 1024.0),
        );

        let _ = self.writer.write_all(line.as_bytes());
        self.since = Instant::now();
        self.phases = Default::default();
        self.tick = Default::default();
    }
}

fn rss_bytes() -> Option<u64> {
    let statm = std::fs::read_to_string("/proc/self/statm").ok()?;
    let resident_pages: u64 = statm.split_whitespace().nth(1)?.parse().ok()?;
    Some(resident_pages * 4096)
}

fn watchdog() {
    let Some(telemetry) = TELEMETRY.get() else {
        return;
    };
    // The `busy_since` of the stall being sampled, and how many samples were taken.
    let mut stall = (0, 0);
    // The `idle_since` that was last reported.
    let mut reported_idle = u64::MAX;
    let mut profile = Profile::default();
    let mut profile_written = Instant::now();
    // Whether the AVM2 call stack was requested by the last poll.
    let mut requested = false;

    loop {
        std::thread::sleep(POLL_INTERVAL);
        if profile_written.elapsed() >= PROFILE_WRITE_INTERVAL {
            profile.write(&telemetry.dir.join("avm2_profile.txt"));
            profile_written = Instant::now();
        }

        let now = telemetry.now_us();
        let busy_since = telemetry.busy_since.load(Ordering::Acquire);

        if busy_since != 0 {
            let busy_for = Duration::from_micros(now.saturating_sub(busy_since - 1));
            let phase = Phase::ALL[telemetry.phase.load(Ordering::Relaxed) as usize];

            if requested {
                let stack = ruffle_core::telemetry::take_avm2_stack();
                profile.record(phase, busy_for >= SLOW_HANDLER, stack.as_deref());
            }
            ruffle_core::telemetry::request_avm2_stack();
            requested = true;

            if stall.0 != busy_since {
                stall = (busy_since, 0);
            }
            let Some(&sample_at) = SAMPLE_AT_SECS.get(stall.1) else {
                continue;
            };
            if busy_for < Duration::from_secs(sample_at) {
                continue;
            }
            // `sample` answers the pending request itself.
            requested = false;
            if stall.1 == 0 {
                tracing::warn!(
                    "Telemetry: main thread busy for {:.0} ms in {}",
                    ms(busy_for),
                    phase.name()
                );
                telemetry.log_stall(&format!(
                    "[{:.3}s] STALL: main thread busy for {:.0} ms, currently in {}\n",
                    telemetry.secs(),
                    ms(busy_for),
                    phase.name(),
                ));
            }
            sample(
                telemetry,
                &format!("+{:.0} ms, in {}", ms(busy_for), phase.name()),
            );
            stall.1 += 1;
        } else {
            requested = false;
            let idle_since = telemetry.idle_since.load(Ordering::Relaxed);
            let idle_for = Duration::from_micros(now.saturating_sub(idle_since));
            if idle_since != 0 && idle_since != reported_idle && idle_for >= IDLE_THRESHOLD {
                reported_idle = idle_since;
                telemetry.log_stall(&format!(
                    "[{:.3}s] main thread hasn't entered a Ruffle handler for {:.0} ms\n",
                    telemetry.secs(),
                    ms(idle_for),
                ));
                sample(telemetry, &format!("idle for {:.0} ms", ms(idle_for)));
            }
        }
    }
}

/// Counts of sampled AVM2 call stacks, keyed by the phase and the innermost frames.
#[derive(Default)]
struct Profile {
    all: HashMap<String, u32>,
    /// Only the samples taken in handlers that ran for at least [`SLOW_HANDLER`].
    slow: HashMap<String, u32>,
}

impl Profile {
    fn record(&mut self, phase: Phase, slow: bool, stack: Option<&str>) {
        let mut key = format!("[{}]", phase.name());
        match stack {
            // `CallStack` prints the innermost frame first, each as "\n\tat <function>".
            Some(stack) => {
                for frame in stack
                    .split("\n\tat ")
                    .filter(|f| !f.is_empty())
                    .take(PROFILE_DEPTH)
                {
                    key.push('\n');
                    key.push_str(frame);
                }
            }
            None => key.push_str("\n(no ActionScript: native code)"),
        }
        if slow {
            *self.slow.entry(key.clone()).or_default() += 1;
        }
        *self.all.entry(key).or_default() += 1;
    }

    fn write(&self, path: &Path) {
        let mut out = String::new();
        let _ = writeln!(
            out,
            "Sampled every {} ms while the main thread is busy. 'self' counts the innermost frame,\n\
             'incl' counts every function on the stack once. Lines starting with [phase] are\n\
             samples that didn't land in ActionScript.\n",
            POLL_INTERVAL.as_millis()
        );
        Self::write_section(&mut out, "SLOW HANDLERS (>= 100 ms)", &self.slow);
        Self::write_section(&mut out, "ALL BUSY TIME", &self.all);
        if let Err(err) = std::fs::write(path, out) {
            tracing::warn!("Failed to write {}: {err}", path.display());
        }
    }

    fn write_section(out: &mut String, title: &str, stacks: &HashMap<String, u32>) {
        let total: u32 = stacks.values().sum();
        let _ = writeln!(out, "===== {title}: {total} samples =====");
        if total == 0 {
            out.push('\n');
            return;
        }

        let mut self_counts: HashMap<&str, u32> = HashMap::new();
        let mut incl_counts: HashMap<&str, u32> = HashMap::new();
        for (key, &count) in stacks {
            let mut frames = key.split('\n');
            let phase = frames.next().unwrap_or_default();
            let frames: Vec<&str> = frames.collect();
            let leaf = match frames.first() {
                Some(&"(no ActionScript: native code)") | None => phase,
                Some(leaf) => leaf,
            };
            *self_counts.entry(leaf).or_default() += count;
            let mut seen: Vec<&str> = Vec::with_capacity(frames.len());
            for &frame in &frames {
                if !seen.contains(&frame) {
                    seen.push(frame);
                    *incl_counts.entry(frame).or_default() += count;
                }
            }
        }

        let percent = |count: u32| f64::from(count) * 100.0 / f64::from(total);
        for (label, counts) in [("self", &self_counts), ("incl", &incl_counts)] {
            let mut sorted: Vec<_> = counts.iter().collect();
            sorted.sort_by(|a, b| b.1.cmp(a.1));
            let _ = writeln!(out, "--- top functions by {label} ---");
            for (function, &count) in sorted.into_iter().take(40) {
                let _ = writeln!(out, "{:6.2}% {count:7}  {function}", percent(count));
            }
            out.push('\n');
        }

        let mut sorted: Vec<_> = stacks.iter().collect();
        sorted.sort_by(|a, b| b.1.cmp(a.1));
        let _ = writeln!(out, "--- top stacks ---");
        for (stack, &count) in sorted.into_iter().take(25) {
            let _ = writeln!(
                out,
                "{:6.2}% {count:7}  {}\n",
                percent(count),
                stack.replace('\n', "\n                 ")
            );
        }
        out.push('\n');
    }
}

/// Logs the native stack of the main thread and, if it's running ActionScript, the AVM2 call stack.
fn sample(telemetry: &Telemetry, label: &str) {
    // Discard a stack captured for an earlier request that came too late.
    let _ = ruffle_core::telemetry::take_avm2_stack();
    ruffle_core::telemetry::request_avm2_stack();

    let native = native_stack(telemetry.pid);
    // The interpreter answers within its next 10000 instructions.
    std::thread::sleep(Duration::from_millis(20));
    let avm2 = ruffle_core::telemetry::take_avm2_stack().unwrap_or_else(|| {
        "\n\t(none: the main thread isn't executing ActionScript bytecode)".to_string()
    });

    telemetry.log_stall(&format!(
        "--- sample at {:.3}s ({label}) ---\nAVM2 call stack:{avm2}\nnative stack of the main thread:\n{native}\n",
        telemetry.secs(),
    ));
}

fn native_stack(pid: u32) -> String {
    // The main thread's TID is the PID; `-1` limits `eu-stack` to it.
    let output = Command::new("timeout")
        .args(["10", "eu-stack", "-1", "-n", "120", "-p"])
        .arg(pid.to_string())
        // Don't wait for symbol downloads while the player is stuck.
        .env("DEBUGINFOD_URLS", "")
        .output();
    match output {
        Ok(output) => {
            let mut text = demangle_lines(&String::from_utf8_lossy(&output.stdout));
            if !output.status.success() {
                let _ = write!(
                    text,
                    "(eu-stack failed, {}: {})\n",
                    output.status,
                    String::from_utf8_lossy(&output.stderr).trim()
                );
            }
            text
        }
        Err(err) => format!("(couldn't run eu-stack: {err})\n"),
    }
}

/// Demangles the Rust symbols that `eu-stack` prints as the last word of each frame.
fn demangle_lines(stack: &str) -> String {
    let mut out = String::with_capacity(stack.len());
    for line in stack.lines() {
        match line.rsplit_once(' ') {
            Some((head, symbol)) if symbol.starts_with("_R") || symbol.starts_with("_ZN") => {
                let _ = writeln!(out, "{head} {:#}", rustc_demangle::demangle(symbol));
            }
            _ => {
                out.push_str(line);
                out.push('\n');
            }
        }
    }
    out
}
