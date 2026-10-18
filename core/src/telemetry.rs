//! Hooks that let a frontend diagnose a player that is slow or has stopped responding.
//!
//! The player runs on a single thread, so a frontend watchdog can't inspect it directly.
//! Instead it raises a flag that the AVM2 interpreter checks during its periodic timeout
//! check, and the interpreter answers with its current call stack.

use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Duration;

static AVM2_STACK_REQUESTED: AtomicBool = AtomicBool::new(false);
static AVM2_STACK: Mutex<Option<String>> = Mutex::new(None);

/// How long each part of the last `Player::tick` took.
#[derive(Clone, Copy, Debug, Default)]
pub struct TickStats {
    /// How many SWF frames were run.
    pub frames: u32,
    pub run_frame: Duration,
    pub sockets: Duration,
    pub net_connections: Duration,
    pub timers: Duration,
    pub streams: Duration,
    pub audio: Duration,
}

/// Asks the AVM2 interpreter to capture its call stack the next time it checks for timeouts.
///
/// Can be called from any thread.
pub fn request_avm2_stack() {
    AVM2_STACK_REQUESTED.store(true, Ordering::Relaxed);
}

/// Takes the call stack captured since the last [`request_avm2_stack`], if any.
pub fn take_avm2_stack() -> Option<String> {
    AVM2_STACK.lock().ok()?.take()
}

/// Whether a call stack was requested; clears the request.
#[inline]
pub(crate) fn take_avm2_stack_request() -> bool {
    AVM2_STACK_REQUESTED.load(Ordering::Relaxed)
        && AVM2_STACK_REQUESTED.swap(false, Ordering::Relaxed)
}

pub(crate) fn store_avm2_stack(stack: String) {
    if let Ok(mut slot) = AVM2_STACK.lock() {
        *slot = Some(stack);
    }
}

/// Parts of an ActionScript-triggered goto, which runs frame phases for the entire stage.
#[derive(Clone, Copy, Debug)]
pub enum GotoPart {
    OrphanConstruct,
    StageConstruct,
    FrameConstructed,
    StageScripts,
    OrphanScripts,
    FrameExited,
    Cleanup,
}

const GOTO_PARTS: usize = 7;

static GOTO_COUNT: AtomicU64 = AtomicU64::new(0);
static GOTO_NANOS: [AtomicU64; GOTO_PARTS] = [const { AtomicU64::new(0) }; GOTO_PARTS];

/// Time spent in gotos since the last [`take_goto_stats`].
/// `settled_skips` is only counted on the calling thread.
#[derive(Clone, Copy, Debug, Default)]
pub struct GotoStats {
    pub count: u64,
    /// Subtrees skipped because they had no frame work (counted since startup).
    pub settled_skips: u64,
    /// Skipped subtrees that turned out to have frame work (counted since startup).
    pub settled_skip_violations: u64,
    /// Indexed by [`GotoPart`]; nested gotos are counted in their parent's parts too.
    pub parts: [Duration; GOTO_PARTS],
}

impl GotoStats {
    pub const PART_NAMES: [&str; GOTO_PARTS] = [
        "orphan_construct",
        "stage_construct",
        "frame_constructed",
        "stage_scripts",
        "orphan_scripts",
        "frame_exited",
        "cleanup",
    ];
}

pub fn take_goto_stats() -> GotoStats {
    let mut stats = GotoStats {
        count: GOTO_COUNT.swap(0, Ordering::Relaxed),
        settled_skips: SETTLED_SKIPS.get(),
        settled_skip_violations: SETTLED_SKIP_VIOLATIONS.load(Ordering::Relaxed),
        ..Default::default()
    };
    for (part, nanos) in stats.parts.iter_mut().zip(&GOTO_NANOS) {
        *part = Duration::from_nanos(nanos.swap(0, Ordering::Relaxed));
    }
    stats
}

thread_local! {
    // Not atomic: this is counted for every skipped object, and only read on the player's thread.
    static SETTLED_SKIPS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}
static SETTLED_SKIP_VIOLATIONS: AtomicU64 = AtomicU64::new(0);

/// Counts a subtree skipped by a goto because it has no frame work; returns the previous count.
#[inline]
pub(crate) fn record_settled_skip() -> u64 {
    let skips = SETTLED_SKIPS.get();
    SETTLED_SKIPS.set(skips.wrapping_add(1));
    skips
}

/// Counts a skipped subtree that turned out to have frame work.
pub(crate) fn record_settled_skip_violation() {
    SETTLED_SKIP_VIOLATIONS.fetch_add(1, Ordering::Relaxed);
}

pub(crate) fn record_goto() {
    GOTO_COUNT.fetch_add(1, Ordering::Relaxed);
}

/// Runs `f`, adding its duration to `part` of the goto stats.
#[inline]
pub(crate) fn time_goto_part<R>(part: GotoPart, f: impl FnOnce() -> R) -> R {
    let start = web_time::Instant::now();
    let result = f();
    GOTO_NANOS[part as usize].fetch_add(start.elapsed().as_nanos() as u64, Ordering::Relaxed);
    result
}

/// Sizes of the display list, to spot objects that pile up.
#[derive(Clone, Debug, Default)]
pub struct DisplayCensus {
    /// The stage and all its descendants.
    pub stage_objects: usize,
    /// Entries in the orphan list, including ones that are dead or were re-parented.
    pub orphan_entries: usize,
    pub live_orphans: usize,
    /// Live orphans and all their descendants.
    pub orphan_objects: usize,
    /// The most common classes of live orphans, if requested.
    pub orphan_classes: Vec<(String, usize)>,
}
