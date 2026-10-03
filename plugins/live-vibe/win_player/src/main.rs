//! The sidecar's speaker on Windows, for WSL: the sidecar (winplayer.py) starts this through WSL interop and streams
//! the synthesized speech over its stdin, so playback goes straight to WASAPI instead of through WSLg's RDP audio,
//! which crackles. A native binary, so there is no interpreter or package cold start between launch and `hello`.
//! A port of win_player.py (which stays as the fallback when this binary is missing); the wire is the same.
//!
//! stdin, binary frames: a header "<cI" (kind, payload length), then the payload.
//!   J  JSON control: {"op":"open","rate":24000}  {"op":"cancel","id":n}  {"op":"ping","t":x}  {"op":"quit"}
//!   A  audio: "<III" (clip id, clip length, offset of this chunk; all in samples), then float32 mono samples
//! stdout, one JSON object per line ("t" names it):
//!   hello   the process runs: pid, player (this binary's name and version)
//!   opening progress of an open: stage (received, device), ms since the open arrived, device and note once picked
//!   open    the stream: device, hostapi, rate, device_rate, latency (s), how (wasapi-convert, wasapi, default)
//!   played  id, at, n, dac: samples at..at+n of clip id reach the speaker at perf-counter time dac
//!   end     id, n: the clip's last sample went to the device
//!   cut     id, played, ms: a cancel took effect after `played` samples of clip id; ms to drop the device buffer
//!   pong    t0 (the sidecar's t), w (this process's perf counter on receipt)
//!   xrun    total device underflows so far
//!   error   text; bye  why, frames, xruns
//! Times are QueryPerformanceCounter seconds on Windows (Python's time.perf_counter() there): the sidecar maps them
//! to its clock with ping and pong. The process exits on stdin EOF (the sidecar closed it, exited or crashed), on
//! quit, and when no frame arrives for --watchdog seconds (the sidecar pings every second from `hello` on), so it
//! cannot outlive the sidecar. The watchdog pauses while the device opens; the open has its own limit.
//!
//!   win_player.exe [--speaker NAME] [--latency S] [--watchdog S] [--list] [--fake]
//! --fake drives the same engine from a timer instead of a device: the sidecar's unit checks run it on Linux.

mod engine;
mod fake;
mod out;
#[cfg(windows)]
mod wasapi;

use std::io::{ErrorKind, Read};
use std::process;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::thread;
use std::time::{Duration, Instant};

use serde_json::{json, Value};

use engine::{Engine, Sink};
use fake::FakeStream;
use out::{perf_counter, round, Out};

const MAX_PAYLOAD: u32 = 8 << 20;
const AUDIO_HEADER: usize = 12;
#[cfg(windows)]
const OPEN_TIMEOUT: Duration = Duration::from_secs(45);
const USAGE: &str = "usage: win_player [--speaker NAME] [--latency S] [--watchdog S] [--list] [--fake]";

struct Args {
    speaker: String,
    latency: f64,
    watchdog: f64,
    list: bool,
    fake: bool,
}

fn parse_args() -> Result<Args, String> {
    let mut a = Args { speaker: String::new(), latency: 0.05, watchdog: 15.0, list: false, fake: false };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        let (name, inline) = match arg.split_once('=') {
            Some((n, v)) if n.starts_with("--") => (n.to_string(), Some(v.to_string())),
            _ => (arg.clone(), None),
        };
        let mut value = |what: &str| -> Result<String, String> {
            inline.clone().or_else(|| it.next()).ok_or_else(|| format!("{what} needs a value"))
        };
        let num = |v: String, what: &str| v.parse::<f64>().map_err(|_| format!("{what}: not a number: {v}"));
        match name.as_str() {
            "--speaker" => a.speaker = value("--speaker")?,
            "--latency" => a.latency = num(value("--latency")?, "--latency")?,
            "--watchdog" => a.watchdog = num(value("--watchdog")?, "--watchdog")?,
            "--list" => a.list = true,
            "--fake" => a.fake = true,
            "-h" | "--help" => {
                println!("{USAGE}");
                process::exit(0);
            }
            other => return Err(format!("unrecognized argument: {other}")),
        }
    }
    Ok(a)
}

enum Stream {
    Fake(FakeStream),
    #[cfg(windows)]
    Wasapi(wasapi::WasapiStream),
}

impl Stream {
    fn abort(&self) {
        match self {
            Stream::Fake(s) => s.abort(),
            #[cfg(windows)]
            Stream::Wasapi(s) => s.abort(),
        }
    }

    /// Drops what the device still buffers, as the local player does: the cut is immediate.
    fn restart(&self) -> Result<(), String> {
        match self {
            Stream::Fake(_) => Ok(()),
            #[cfg(windows)]
            Stream::Wasapi(s) => s.restart(),
        }
    }
}

struct Shared {
    out: Arc<Out>,
    engine: OnceLock<Arc<Engine>>,
    stream: Mutex<Option<Stream>>,
    last_rx: Mutex<Instant>,
    opening: AtomicBool,
    ending: AtomicBool,
}

impl Shared {
    fn touch(&self) {
        *self.last_rx.lock().unwrap_or_else(|e| e.into_inner()) = Instant::now();
    }

    fn quiet_for(&self) -> f64 {
        self.last_rx.lock().unwrap_or_else(|e| e.into_inner()).elapsed().as_secs_f64()
    }

    fn error(&self, text: impl Into<String>) {
        self.out.send(&[("t", json!("error")), ("text", json!(text.into()))]);
    }

    /// Stops the device, says bye and exits; from whichever thread ends it first.
    fn finish(&self, why: &str) -> ! {
        if self.ending.swap(true, Ordering::SeqCst) {
            loop {
                thread::sleep(Duration::from_secs(1)); // another thread is ending the process
            }
        }
        if let Some(s) = self.stream.lock().unwrap_or_else(|e| e.into_inner()).take() {
            s.abort();
        }
        let (frames, xruns) = self.engine.get().map(|e| e.counts()).unwrap_or((0, 0));
        self.out.send(&[("t", json!("bye")), ("why", json!(why)), ("frames", json!(frames)), ("xruns", json!(xruns))]);
        self.out.close();
        process::exit(0);
    }
}

fn list() -> i32 {
    #[cfg(windows)]
    {
        match wasapi::list() {
            Ok(rows) => {
                let rows: Vec<Value> = rows
                    .into_iter()
                    .enumerate()
                    .map(|(i, (name, rate))| json!({"index": i, "name": name, "rate": rate}))
                    .collect();
                println!("{}", Value::Array(rows));
                0
            }
            Err(e) => {
                eprintln!("cannot list the outputs: {e}");
                1
            }
        }
    }
    #[cfg(not(windows))]
    {
        println!("[]"); // no WASAPI here
        0
    }
}

fn main() {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("{USAGE}\nwin_player: error: {e}");
            process::exit(2);
        }
    };
    if args.list {
        process::exit(list());
    }
    let out = Arc::new(Out::start());
    out.send(&[
        ("t", json!("hello")),
        ("pid", json!(process::id())),
        ("player", json!(concat!("win_player ", env!("CARGO_PKG_VERSION"), " (rust)"))),
    ]);
    let sh = Arc::new(Shared {
        out,
        engine: OnceLock::new(),
        stream: Mutex::new(None),
        last_rx: Mutex::new(Instant::now()),
        opening: AtomicBool::new(false),
        ending: AtomicBool::new(false),
    });
    {
        let sh = sh.clone();
        let limit = args.watchdog;
        thread::Builder::new()
            .name("watchdog".into())
            .spawn(move || loop {
                thread::sleep(Duration::from_millis(500));
                if sh.opening.load(Ordering::SeqCst) {
                    continue; // the open has its own limit
                }
                if sh.quiet_for() > limit {
                    sh.finish("watchdog: the sidecar went quiet");
                }
            })
            .expect("spawn the watchdog");
    }
    let why = run(&sh, &args);
    sh.finish(&why);
}

enum Read5 {
    Frame(u8, Vec<u8>),
    End(String),
}

fn read_frame(inp: &mut impl Read) -> Result<Read5, std::io::Error> {
    let mut head = [0u8; 5];
    match inp.read_exact(&mut head) {
        Ok(()) => {}
        Err(e) if e.kind() == ErrorKind::UnexpectedEof => return Ok(Read5::End("stdin closed".into())),
        Err(e) => return Err(e),
    }
    let size = u32::from_le_bytes([head[1], head[2], head[3], head[4]]);
    if size > MAX_PAYLOAD {
        return Ok(Read5::End(format!("frame too large ({size})")));
    }
    let mut body = vec![0u8; size as usize];
    match inp.read_exact(&mut body) {
        Ok(()) => Ok(Read5::Frame(head[0], body)),
        Err(e) if e.kind() == ErrorKind::UnexpectedEof => Ok(Read5::End("stdin closed".into())),
        Err(e) => Err(e),
    }
}

/// The frame loop; returns why it ended.
fn run(sh: &Arc<Shared>, args: &Args) -> String {
    let stdin = std::io::stdin();
    let mut inp = stdin.lock();
    loop {
        let (kind, body) = match read_frame(&mut inp) {
            Ok(Read5::Frame(k, b)) => (k, b),
            Ok(Read5::End(why)) => return why,
            Err(e) => {
                sh.error(format!("stdin: {e}"));
                return "error".into();
            }
        };
        sh.touch();
        if kind == b'A' {
            let Some(engine) = sh.engine.get() else { continue };
            if body.len() < AUDIO_HEADER || !(body.len() - AUDIO_HEADER).is_multiple_of(4) {
                sh.error(format!("bad audio frame ({} bytes)", body.len()));
                return "error".into();
            }
            let u = |i: usize| u32::from_le_bytes([body[i], body[i + 1], body[i + 2], body[i + 3]]);
            let (cid, total, offset) = (u(0), u(4), u(8));
            let samples: Vec<f32> =
                body[AUDIO_HEADER..].chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
            engine.add(cid, total, offset, samples);
            continue;
        }
        let msg: Value = match serde_json::from_slice(&body) {
            Ok(Value::Object(m)) => Value::Object(m),
            Ok(other) => {
                sh.error(format!("control frame is not an object: {other}"));
                return "error".into();
            }
            Err(e) => {
                sh.error(format!("JSONDecodeError: {e}"));
                return "error".into();
            }
        };
        match msg.get("op").and_then(Value::as_str) {
            Some("ping") => {
                let t0 = msg.get("t").cloned().unwrap_or(Value::Null);
                sh.out.send(&[("t", json!("pong")), ("t0", t0), ("w", json!(perf_counter()))]);
            }
            Some("open") if sh.engine.get().is_none() => {
                let Some(rate) = msg.get("rate").and_then(|r| r.as_u64().or_else(|| r.as_f64().map(|f| f as u64)))
                else {
                    sh.error("KeyError: 'rate'");
                    return "error".into();
                };
                let engine = Arc::new(Engine::new(sh.out.clone() as Arc<dyn Sink>, rate as u32));
                let _ = sh.engine.set(engine.clone());
                if args.fake {
                    *sh.stream.lock().unwrap_or_else(|e| e.into_inner()) =
                        Some(Stream::Fake(FakeStream::start(engine)));
                    sh.out.send(&[
                        ("t", json!("open")),
                        ("device", json!("fake")),
                        ("hostapi", json!("none")),
                        ("rate", json!(rate)),
                        ("device_rate", json!(rate)),
                        ("latency", json!(fake::LATENCY)),
                        ("how", json!("fake")),
                        ("note", json!("")),
                        ("tried", json!([])),
                    ]);
                } else if let Err(why) = open_device(sh, engine, args) {
                    return why;
                }
            }
            Some("cancel") => {
                let Some(engine) = sh.engine.get() else { continue };
                let Some(cid) = msg.get("id").and_then(|v| v.as_u64().or_else(|| v.as_f64().map(|f| f as u64))) else {
                    sh.error("KeyError: 'id'");
                    return "error".into();
                };
                let t = perf_counter();
                let played = engine.cancel(cid as u32);
                if played > 0 && !args.fake {
                    if let Some(s) = sh.stream.lock().unwrap_or_else(|e| e.into_inner()).as_ref() {
                        if let Err(e) = s.restart() {
                            sh.error(format!("restart after cancel: {e}"));
                        }
                    }
                }
                sh.out.send(&[
                    ("t", json!("cut")),
                    ("id", json!(cid)),
                    ("played", json!(played)),
                    ("ms", json!(round(1000.0 * (perf_counter() - t), 1))),
                ]);
            }
            Some("quit") => return "quit".into(),
            _ => {}
        }
    }
}

/// Opens the device on a thread, so frames (pings, a quit, EOF) are still read while Windows takes its time; the
/// watchdog pauses meanwhile and the open has its own limit. Err: why the process ends now.
#[cfg(windows)]
fn open_device(sh: &Arc<Shared>, engine: Arc<Engine>, args: &Args) -> Result<(), String> {
    let t0 = perf_counter();
    sh.opening.store(true, Ordering::SeqCst);
    sh.touch();
    sh.out.send(&[
        ("t", json!("opening")),
        ("stage", json!("received")),
        ("rate", json!(engine.rate)),
        ("ms", json!(0.0)),
    ]);
    let (sh, spec, latency) = (sh.clone(), args.speaker.clone(), args.latency);
    thread::Builder::new()
        .name("open".into())
        .spawn(move || {
            let sink = sh.out.clone() as Arc<dyn Sink>;
            match wasapi::open(engine.clone(), sink, spec, latency, t0, OPEN_TIMEOUT) {
                Ok((stream, info)) => {
                    *sh.stream.lock().unwrap_or_else(|e| e.into_inner()) = Some(Stream::Wasapi(stream));
                    sh.out.send(&[
                        ("t", json!("open")),
                        ("device", json!(info.device)),
                        ("hostapi", json!(wasapi::HOSTAPI)),
                        ("rate", json!(engine.rate)),
                        ("device_rate", json!(info.device_rate)),
                        ("latency", json!(round(info.latency, 4))),
                        ("how", json!(info.how)),
                        ("note", json!(info.note)),
                        ("tried", json!(info.tried)),
                        ("ms", json!(round(1000.0 * (perf_counter() - t0), 1))),
                    ]);
                    sh.touch();
                    sh.opening.store(false, Ordering::SeqCst);
                }
                Err(e) => {
                    sh.error(format!("cannot open the speaker: {e}"));
                    sh.finish("no speaker");
                }
            }
        })
        .expect("spawn the open thread");
    Ok(())
}

#[cfg(not(windows))]
fn open_device(sh: &Arc<Shared>, _engine: Arc<Engine>, _args: &Args) -> Result<(), String> {
    sh.error("cannot open the speaker: no WASAPI on this OS (use --fake)");
    Err("no speaker".into())
}
