//! WASAPI shared mode, the way win_player.py drove it through PortAudio: the clip rate with Windows converting to
//! the mix format (AUTOCONVERTPCM, "wasapi-convert"); then plain WASAPI on a device whose mix rate is the clip rate
//! ("wasapi", channels and sample format converted here); last, the Windows default output with Windows converting
//! and a timer in place of the device event ("default"). No resampling here, so the sample positions this reports
//! are the clip's own.
//!
//! One thread owns every COM object: it opens the device, renders on the device's event, and takes the restart and
//! quit commands (woken by a second event), so nothing COM crosses threads.

use std::sync::mpsc::{channel, sync_channel, Receiver, RecvTimeoutError, Sender, SyncSender};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use serde_json::json;
use windows::core::{GUID, PCWSTR};
use windows::Win32::Foundation::{CloseHandle, HANDLE, PROPERTYKEY};
use windows::Win32::Media::Audio::{
    eConsole, eRender, IAudioClient, IAudioClock, IAudioRenderClient, IMMDevice, IMMDeviceEnumerator,
    MMDeviceEnumerator, AUDCLNT_SHAREMODE_SHARED, AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM,
    AUDCLNT_STREAMFLAGS_EVENTCALLBACK, AUDCLNT_STREAMFLAGS_SRC_DEFAULT_QUALITY, DEVICE_STATE_ACTIVE, WAVEFORMATEX,
    WAVEFORMATEXTENSIBLE, WAVEFORMATEXTENSIBLE_0,
};
use windows::Win32::System::Com::StructuredStorage::{PropVariantClear, PropVariantToStringAlloc};
use windows::Win32::System::Com::{
    CoCreateInstance, CoInitializeEx, CoTaskMemFree, CoUninitialize, CLSCTX_ALL, COINIT_MULTITHREADED, STGM_READ,
};
use windows::Win32::System::Threading::{CreateEventW, SetEvent, WaitForMultipleObjects, WaitForSingleObject};

use crate::engine::{Engine, Sink};
use crate::out::{perf_counter, round};

const PKEY_DEVICE_FRIENDLY_NAME: PROPERTYKEY =
    PROPERTYKEY { fmtid: GUID::from_u128(0xa45c254e_df1c_4efd_8020_67d146a850e0), pid: 14 };
const SUBTYPE_IEEE_FLOAT: GUID = GUID::from_u128(0x00000003_0000_0010_8000_00aa00389b71);
const SUBTYPE_PCM: GUID = GUID::from_u128(0x00000001_0000_0010_8000_00aa00389b71);
const WAVE_FORMAT_PCM: u16 = 1;
const WAVE_FORMAT_IEEE_FLOAT: u16 = 3;
const WAVE_FORMAT_EXTENSIBLE: u16 = 0xFFFE;
const SPEAKER_FRONT_CENTER: u32 = 0x4;
const POLL_MS: u32 = 5; // the "default" path's timer

pub const HOSTAPI: &str = "Windows WASAPI";

/// A windows::core::Error as text: the system message when there is one, the AUDCLNT name otherwise, and the code.
fn we(e: &windows::core::Error) -> String {
    let code = e.code().0 as u32;
    let name = match code {
        0x8889_0001 => "AUDCLNT_E_NOT_INITIALIZED",
        0x8889_0002 => "AUDCLNT_E_ALREADY_INITIALIZED",
        0x8889_0004 => "AUDCLNT_E_DEVICE_INVALIDATED",
        0x8889_0008 => "AUDCLNT_E_UNSUPPORTED_FORMAT",
        0x8889_000A => "AUDCLNT_E_DEVICE_IN_USE",
        0x8889_000F => "AUDCLNT_E_ENDPOINT_CREATE_FAILED",
        0x8889_0010 => "AUDCLNT_E_SERVICE_NOT_RUNNING",
        0x8889_0016 => "AUDCLNT_E_BUFFER_SIZE_ERROR",
        0x8889_0019 => "AUDCLNT_E_BUFFER_SIZE_NOT_ALIGNED",
        0x8889_0020 => "AUDCLNT_E_INVALID_DEVICE_PERIOD",
        0x8889_0021 => "AUDCLNT_E_INVALID_STREAM_FLAG",
        0x8007_0005 => "E_ACCESSDENIED",
        0x8007_0057 => "E_INVALIDARG",
        0x8007_0490 => "E_NOTFOUND",
        _ => "",
    };
    let msg = e.message();
    let msg = msg.trim();
    match (msg.is_empty(), name.is_empty()) {
        (false, _) => format!("{msg} (0x{code:08X})"),
        (true, false) => format!("{name} (0x{code:08X})"),
        (true, true) => format!("HRESULT 0x{code:08X}"),
    }
}

struct Com;

impl Com {
    fn init() -> Result<Com, String> {
        unsafe { CoInitializeEx(None, COINIT_MULTITHREADED) }.ok().map_err(|e| format!("COM: {}", we(&e)))?;
        Ok(Com)
    }
}

impl Drop for Com {
    fn drop(&mut self) {
        unsafe { CoUninitialize() };
    }
}

fn enumerator() -> Result<IMMDeviceEnumerator, String> {
    unsafe { CoCreateInstance(&MMDeviceEnumerator, None, CLSCTX_ALL) }
        .map_err(|e| format!("MMDeviceEnumerator: {}", we(&e)))
}

fn friendly_name(dev: &IMMDevice) -> String {
    unsafe {
        let Ok(store) = dev.OpenPropertyStore(STGM_READ) else { return "?".into() };
        let Ok(mut pv) = store.GetValue(&PKEY_DEVICE_FRIENDLY_NAME) else { return "?".into() };
        let name = match PropVariantToStringAlloc(&pv) {
            Ok(p) => {
                let s = p.to_string().unwrap_or_else(|_| "?".into());
                CoTaskMemFree(Some(p.0 as _));
                s
            }
            Err(_) => "?".into(),
        };
        let _ = PropVariantClear(&mut pv);
        name
    }
}

fn outputs(en: &IMMDeviceEnumerator) -> Result<Vec<(IMMDevice, String)>, String> {
    unsafe {
        let all = en.EnumAudioEndpoints(eRender, DEVICE_STATE_ACTIVE).map_err(|e| format!("endpoints: {}", we(&e)))?;
        let n = all.GetCount().map_err(|e| format!("endpoints: {}", we(&e)))?;
        let mut v = Vec::new();
        for i in 0..n {
            if let Ok(d) = all.Item(i) {
                let name = friendly_name(&d);
                v.push((d, name));
            }
        }
        Ok(v)
    }
}

fn mix_rate(dev: &IMMDevice) -> Option<u32> {
    unsafe {
        let client: IAudioClient = dev.Activate(CLSCTX_ALL, None).ok()?;
        let mix = client.GetMixFormat().ok()?;
        let rate = (*mix).nSamplesPerSec;
        CoTaskMemFree(Some(mix as _));
        Some(rate)
    }
}

/// What --list prints: the active output endpoints with their mix rates.
pub fn list() -> Result<Vec<(String, f64)>, String> {
    let _com = Com::init()?;
    let en = enumerator()?;
    Ok(outputs(&en)?.into_iter().map(|(d, name)| (name, mix_rate(&d).unwrap_or(0) as f64)).collect())
}

/// Python's repr() of a str, for the notes: 'Stealth'.
fn py_repr(s: &str) -> String {
    if s.contains('\'') && !s.contains('"') {
        format!("\"{s}\"")
    } else {
        format!("'{}'", s.replace('\\', "\\\\").replace('\'', "\\'"))
    }
}

/// An output whose name contains `spec`, else Windows' default output.
fn pick(en: &IMMDeviceEnumerator, spec: &str) -> (Option<(IMMDevice, String)>, String) {
    if !spec.is_empty() {
        let want = spec.to_lowercase();
        if let Ok(all) = outputs(en) {
            if let Some(found) = all.into_iter().find(|(_, name)| name.to_lowercase().contains(&want)) {
                return (Some(found), format!("matches {}", py_repr(spec)));
            }
        }
    }
    let note = if spec.is_empty() {
        "Windows default".to_string()
    } else {
        format!("no WASAPI output matches {}; Windows default", py_repr(spec))
    };
    match unsafe { en.GetDefaultAudioEndpoint(eRender, eConsole) } {
        Ok(d) => {
            let name = friendly_name(&d);
            (Some((d, name)), note)
        }
        Err(e) => (None, format!("no default output: {}", we(&e))),
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Kind {
    F32,
    I16,
    I32,
}

#[derive(Clone, Copy)]
enum Mode {
    /// The clip's own format (mono float32 at the clip rate); Windows converts. `event`: the device's event drives
    /// rendering, else a timer.
    Convert { event: bool },
    /// The mix format as is, which only works when its rate is the clip rate.
    Native,
}

struct Client {
    client: IAudioClient,
    render: IAudioRenderClient,
    clock: Option<(IAudioClock, f64)>,
    event: Option<HANDLE>,
    rate: f64,
    buffer_frames: u32,
    channels: usize,
    kind: Kind,
    device_rate: u32,
    stream_latency: f64,
    written: u64, // frames written since the stream (re)started: the next block's first frame
    primed: bool, // serviced once since the start: an empty buffer from here on is an underflow
    scratch: Vec<f32>,
}

fn mix_kind(mix: *const WAVEFORMATEX) -> Option<Kind> {
    unsafe {
        let f = &*mix;
        let float = match f.wFormatTag {
            WAVE_FORMAT_IEEE_FLOAT => true,
            WAVE_FORMAT_PCM => false,
            WAVE_FORMAT_EXTENSIBLE if f.cbSize >= 22 => {
                let sub = (*(mix as *const WAVEFORMATEXTENSIBLE)).SubFormat;
                if sub == SUBTYPE_IEEE_FLOAT {
                    true
                } else if sub == SUBTYPE_PCM {
                    false
                } else {
                    return None;
                }
            }
            _ => return None,
        };
        let block = f.nBlockAlign as usize / f.nChannels.max(1) as usize;
        match (float, f.wBitsPerSample, block) {
            (true, 32, 4) => Some(Kind::F32),
            (false, 16, 2) => Some(Kind::I16),
            (false, 32, 4) => Some(Kind::I32),
            _ => None,
        }
    }
}

impl Client {
    fn init(dev: &IMMDevice, rate: u32, latency: f64, mode: Mode) -> Result<Client, String> {
        unsafe {
            let client: IAudioClient = dev.Activate(CLSCTX_ALL, None).map_err(|e| format!("activate: {}", we(&e)))?;
            let mix = client.GetMixFormat().map_err(|e| format!("mix format: {}", we(&e)))?;
            let device_rate = (*mix).nSamplesPerSec;
            let hns = (latency * 1e7) as i64;
            let result = (|| {
                let (fmt, flags, channels, kind, event) = match mode {
                    Mode::Convert { event } => {
                        let mut flags = AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM | AUDCLNT_STREAMFLAGS_SRC_DEFAULT_QUALITY;
                        if event {
                            flags |= AUDCLNT_STREAMFLAGS_EVENTCALLBACK;
                        }
                        (None, flags, 1usize, Kind::F32, event)
                    }
                    Mode::Native => {
                        if device_rate != rate {
                            return Err(format!("the device mixes at {device_rate} Hz, not the clip's {rate} Hz"));
                        }
                        let kind = mix_kind(mix).ok_or("an unsupported mix format")?;
                        (
                            Some(mix as *const WAVEFORMATEX),
                            AUDCLNT_STREAMFLAGS_EVENTCALLBACK,
                            (*mix).nChannels as usize,
                            kind,
                            true,
                        )
                    }
                };
                let mono = WAVEFORMATEXTENSIBLE {
                    Format: WAVEFORMATEX {
                        wFormatTag: WAVE_FORMAT_EXTENSIBLE,
                        nChannels: 1,
                        nSamplesPerSec: rate,
                        nAvgBytesPerSec: rate * 4,
                        nBlockAlign: 4,
                        wBitsPerSample: 32,
                        cbSize: 22,
                    },
                    Samples: WAVEFORMATEXTENSIBLE_0 { wValidBitsPerSample: 32 },
                    dwChannelMask: SPEAKER_FRONT_CENTER,
                    SubFormat: SUBTYPE_IEEE_FLOAT,
                };
                let pfmt = fmt.unwrap_or(&mono as *const WAVEFORMATEXTENSIBLE as *const WAVEFORMATEX);
                client
                    .Initialize(AUDCLNT_SHAREMODE_SHARED, flags, hns, 0, pfmt, None)
                    .map_err(|e| format!("initialize: {}", we(&e)))?;
                Ok((flags, channels, kind, event))
            })();
            CoTaskMemFree(Some(mix as _));
            let (_flags, channels, kind, event) = result?;
            let event = if event {
                let h = CreateEventW(None, false, false, PCWSTR::null()).map_err(|e| format!("event: {}", we(&e)))?;
                client.SetEventHandle(h).map_err(|e| format!("event handle: {}", we(&e)))?;
                Some(h)
            } else {
                None
            };
            let buffer_frames = client.GetBufferSize().map_err(|e| format!("buffer size: {}", we(&e)))?;
            let render: IAudioRenderClient = client.GetService().map_err(|e| format!("render client: {}", we(&e)))?;
            let clock = client
                .GetService::<IAudioClock>()
                .ok()
                .and_then(|c| c.GetFrequency().ok().filter(|&f| f > 0).map(|f| (c, f as f64)));
            let stream_latency = client.GetStreamLatency().map(|h| h as f64 / 1e7).unwrap_or(0.0);
            Ok(Client {
                client,
                render,
                clock,
                event,
                rate: rate as f64,
                buffer_frames,
                channels,
                kind,
                device_rate,
                stream_latency,
                written: 0,
                primed: false,
                scratch: Vec::new(),
            })
        }
    }

    /// The output latency reported in `open`: the buffer plus the stream's own.
    fn latency(&self) -> f64 {
        self.buffer_frames as f64 / self.rate + self.stream_latency
    }

    /// The perf-counter time the next block's first frame reaches the DAC: the device clock's position against
    /// the frames written, else the time now plus what is queued plus the stream latency.
    fn dac_time(&self, padding: u32) -> f64 {
        let now = perf_counter();
        if let Some((clock, freq)) = &self.clock {
            let (mut pos, mut qpc) = (0u64, 0u64);
            if unsafe { clock.GetPosition(&mut pos, Some(&mut qpc)) }.is_ok() && qpc > 0 {
                let dac = qpc as f64 * 1e-7 + (self.written as f64 / self.rate - pos as f64 / freq);
                let ahead = dac - now;
                if (0.0..2.0).contains(&ahead) {
                    return now + ahead;
                }
            }
        }
        now + padding as f64 / self.rate + self.stream_latency
    }

    /// Fills `frames` of the device buffer from the engine.
    fn write(&mut self, engine: &Engine, frames: u32, dac: f64, underflow: bool) -> Result<(), String> {
        unsafe {
            let data = self.render.GetBuffer(frames).map_err(|e| format!("get buffer: {}", we(&e)))?;
            let n = frames as usize;
            if self.channels == 1 && self.kind == Kind::F32 {
                engine.fill(std::slice::from_raw_parts_mut(data as *mut f32, n), dac, underflow);
            } else {
                self.scratch.resize(n, 0.0);
                engine.fill(&mut self.scratch, dac, underflow);
                let ch = self.channels;
                match self.kind {
                    Kind::F32 => {
                        let dst = std::slice::from_raw_parts_mut(data as *mut f32, n * ch);
                        for (i, &s) in self.scratch.iter().enumerate() {
                            dst[i * ch..(i + 1) * ch].fill(s);
                        }
                    }
                    Kind::I16 => {
                        let dst = std::slice::from_raw_parts_mut(data as *mut i16, n * ch);
                        for (i, &s) in self.scratch.iter().enumerate() {
                            dst[i * ch..(i + 1) * ch].fill((s.clamp(-1.0, 1.0) * 32767.0) as i16);
                        }
                    }
                    Kind::I32 => {
                        let dst = std::slice::from_raw_parts_mut(data as *mut i32, n * ch);
                        for (i, &s) in self.scratch.iter().enumerate() {
                            dst[i * ch..(i + 1) * ch].fill((s.clamp(-1.0, 1.0) as f64 * 2147483647.0) as i32);
                        }
                    }
                }
            }
            self.render.ReleaseBuffer(frames, 0).map_err(|e| format!("release buffer: {}", we(&e)))?;
        }
        self.written += frames as u64;
        Ok(())
    }

    /// Fills the whole (empty) buffer and starts the stream, so the first device period has data.
    fn start(&mut self, engine: &Engine) -> Result<(), String> {
        self.written = 0;
        self.primed = false;
        let dac = perf_counter() + self.stream_latency;
        self.write(engine, self.buffer_frames, dac, false)?;
        unsafe { self.client.Start() }.map_err(|e| format!("start: {}", we(&e)))
    }

    /// Drops what the device still buffers (stop, reset, start): the cut is immediate.
    fn restart(&mut self, engine: &Engine) -> Result<(), String> {
        unsafe {
            self.client.Stop().map_err(|e| format!("stop: {}", we(&e)))?;
            self.client.Reset().map_err(|e| format!("reset: {}", we(&e)))?;
        }
        self.start(engine)
    }

    /// One device period: top the buffer up.
    fn service(&mut self, engine: &Engine) -> Result<(), String> {
        let padding = unsafe { self.client.GetCurrentPadding() }.map_err(|e| format!("padding: {}", we(&e)))?;
        let avail = self.buffer_frames.saturating_sub(padding);
        if avail == 0 {
            return Ok(());
        }
        let underflow = padding == 0 && self.primed;
        self.primed = true;
        let dac = self.dac_time(padding);
        self.write(engine, avail, dac, underflow)
    }

    fn stop(&self) {
        unsafe {
            let _ = self.client.Stop();
        }
    }
}

impl Drop for Client {
    fn drop(&mut self) {
        if let Some(h) = self.event.take() {
            unsafe {
                let _ = CloseHandle(h);
            }
        }
    }
}

enum Cmd {
    Restart(SyncSender<Result<(), String>>),
    Quit(SyncSender<()>),
}

/// What `open` reports.
pub struct Info {
    pub device: String,
    pub device_rate: u32,
    pub latency: f64,
    pub how: &'static str,
    pub note: String,
    pub tried: Vec<String>,
}

pub struct WasapiStream {
    cmd: Sender<Cmd>,
    wake: usize, // the HANDLE of the event that wakes the device thread for a command
}

impl WasapiStream {
    fn poke(&self, c: Cmd) -> bool {
        self.cmd.send(c).is_ok() && unsafe { SetEvent(HANDLE(self.wake as _)) }.is_ok()
    }

    pub fn restart(&self) -> Result<(), String> {
        let (tx, rx) = sync_channel(1);
        if !self.poke(Cmd::Restart(tx)) {
            return Err("the device thread is gone".into());
        }
        rx.recv_timeout(Duration::from_secs(1)).map_err(|_| "no answer from the device thread".to_string())?
    }

    pub fn abort(&self) {
        let (tx, rx) = sync_channel(1);
        if self.poke(Cmd::Quit(tx)) {
            let _ = rx.recv_timeout(Duration::from_millis(300));
        }
    }
}

/// Opens the device on its own thread and waits up to `timeout` for it. `t0`: when the open request arrived (perf
/// counter), for the progress line's ms.
pub fn open(
    engine: Arc<Engine>,
    sink: Arc<dyn Sink>,
    spec: String,
    latency: f64,
    t0: f64,
    timeout: Duration,
) -> Result<(WasapiStream, Info), String> {
    let wake = unsafe { CreateEventW(None, false, false, PCWSTR::null()) }.map_err(|e| format!("event: {}", we(&e)))?;
    let wake_raw = wake.0 as usize;
    let (cmd_tx, cmd_rx) = channel::<Cmd>();
    let (reply_tx, reply_rx) = sync_channel::<Result<Info, String>>(1);
    thread::Builder::new()
        .name("wasapi".into())
        .spawn(move || device_thread(engine, sink, spec, latency, t0, reply_tx, cmd_rx, wake_raw))
        .map_err(|e| format!("device thread: {e}"))?;
    match reply_rx.recv_timeout(timeout) {
        Ok(Ok(info)) => Ok((WasapiStream { cmd: cmd_tx, wake: wake_raw }, info)),
        Ok(Err(e)) => Err(e),
        Err(RecvTimeoutError::Timeout) => {
            Err(format!("Windows audio did not answer within {:.0}s", timeout.as_secs_f64()))
        }
        Err(RecvTimeoutError::Disconnected) => Err("the device thread died".into()),
    }
}

#[allow(clippy::too_many_arguments)]
fn device_thread(
    engine: Arc<Engine>,
    sink: Arc<dyn Sink>,
    spec: String,
    latency: f64,
    t0: f64,
    reply: SyncSender<Result<Info, String>>,
    cmd: Receiver<Cmd>,
    wake_raw: usize,
) {
    let wake = HANDLE(wake_raw as _);
    let _com = match Com::init() {
        Ok(c) => c,
        Err(e) => {
            let _ = reply.send(Err(e));
            return;
        }
    };
    let opened = (|| -> Result<(Client, Info), String> {
        let en = enumerator()?;
        let (dev, note) = pick(&en, &spec);
        sink.send(&[
            ("t", json!("opening")),
            ("stage", json!("device")),
            ("device", json!(dev.as_ref().map(|(_, n)| n.as_str()))),
            ("note", json!(note)),
            ("ms", json!(round(1000.0 * (perf_counter() - t0), 1))),
        ]);
        let mut tries: Vec<(&'static str, IMMDevice, String, Mode)> = Vec::new();
        if let Some((d, name)) = dev {
            tries.push(("wasapi-convert", d.clone(), name.clone(), Mode::Convert { event: true }));
            tries.push(("wasapi", d, name, Mode::Native));
        }
        let mut tried = Vec::new();
        let fallback = unsafe { en.GetDefaultAudioEndpoint(eRender, eConsole) };
        match fallback {
            Ok(d) => {
                let name = friendly_name(&d);
                tries.push(("default", d, name, Mode::Convert { event: false }));
            }
            Err(e) => tried.push(format!("default: {}", we(&e))),
        }
        for (how, d, name, mode) in tries {
            let attempt = Client::init(&d, engine.rate, latency, mode).and_then(|mut c| c.start(&engine).map(|_| c));
            match attempt {
                Ok(c) => {
                    let info = Info {
                        device: name,
                        device_rate: c.device_rate,
                        latency: c.latency(),
                        how,
                        note: note.clone(),
                        tried: tried.clone(),
                    };
                    return Ok((c, info));
                }
                Err(e) => tried.push(format!("{how}: {e}")),
            }
        }
        Err(tried.join("; "))
    })();
    let mut client = match opened {
        Ok((c, info)) => {
            if reply.send(Ok(info)).is_err() {
                c.stop(); // the opener gave up waiting
                return;
            }
            c
        }
        Err(e) => {
            let _ = reply.send(Err(e));
            return;
        }
    };
    let mut dead = false; // a device error: the sidecar sees no more reports and falls back
    loop {
        unsafe {
            match client.event {
                Some(ev) if !dead => {
                    WaitForMultipleObjects(&[ev, wake], false, 200);
                }
                _ => {
                    WaitForSingleObject(wake, if dead { 200 } else { POLL_MS });
                }
            }
        }
        loop {
            match cmd.try_recv() {
                Ok(Cmd::Restart(tx)) => {
                    let r = if dead { Err("the device failed earlier".into()) } else { client.restart(&engine) };
                    let _ = tx.send(r);
                }
                Ok(Cmd::Quit(tx)) => {
                    client.stop();
                    let _ = tx.send(());
                    return;
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => break,
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    client.stop();
                    return;
                }
            }
        }
        if dead {
            continue;
        }
        if let Err(e) = client.service(&engine) {
            sink.send(&[("t", json!("error")), ("text", json!(format!("device: {e}")))]);
            client.stop();
            dead = true;
        }
    }
}
