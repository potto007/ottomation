//! JSON lines to stdout from one writer thread (the device thread only queues), and the perf-counter clock the
//! lines carry.

use std::io::Write;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::Mutex;
use std::thread;
use std::time::Duration;

use serde_json::Value;

/// One stdout line: `{"t":..., ...}` with the keys in the order given.
pub fn line(pairs: &[(&str, Value)]) -> String {
    let mut s = String::with_capacity(96);
    s.push('{');
    for (i, (k, v)) in pairs.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        s.push_str(&Value::from(*k).to_string());
        s.push(':');
        s.push_str(&v.to_string());
    }
    s.push_str("}\n");
    s
}

/// Rounds like Python's round(x, digits) for the values this sends (DAC times, latencies, milliseconds).
pub fn round(x: f64, digits: i32) -> f64 {
    let m = 10f64.powi(digits);
    (x * m).round() / m
}

pub struct Out {
    tx: Sender<Option<String>>,
    done: Mutex<Option<Receiver<()>>>,
}

impl Out {
    pub fn start() -> Out {
        let (tx, rx) = channel::<Option<String>>();
        let (done_tx, done_rx) = channel::<()>();
        thread::Builder::new()
            .name("out".into())
            .spawn(move || {
                let stdout = std::io::stdout();
                while let Ok(Some(text)) = rx.recv() {
                    let mut w = stdout.lock();
                    if w.write_all(text.as_bytes()).and_then(|_| w.flush()).is_err() {
                        std::process::exit(0); // the sidecar is gone
                    }
                }
                let _ = done_tx.send(());
            })
            .expect("spawn the stdout writer");
        Out { tx, done: Mutex::new(Some(done_rx)) }
    }

    pub fn send(&self, pairs: &[(&str, Value)]) {
        let _ = self.tx.send(Some(line(pairs)));
    }

    /// Flushes what is queued (up to a second) and stops the writer.
    pub fn close(&self) {
        let _ = self.tx.send(None);
        if let Some(done) = self.done.lock().ok().and_then(|mut d| d.take()) {
            let _ = done.recv_timeout(Duration::from_secs(1));
        }
    }
}

/// Seconds on the performance counter: QueryPerformanceCounter on Windows (what Python's time.perf_counter() reads
/// there, and the clock IAudioClock's QPC positions are on), a monotonic clock elsewhere.
#[cfg(windows)]
pub fn perf_counter() -> f64 {
    use std::sync::OnceLock;
    use windows::Win32::System::Performance::{QueryPerformanceCounter, QueryPerformanceFrequency};
    static FREQ: OnceLock<f64> = OnceLock::new();
    let freq = *FREQ.get_or_init(|| {
        let mut f = 0i64;
        unsafe {
            let _ = QueryPerformanceFrequency(&mut f);
        }
        if f > 0 {
            f as f64
        } else {
            1e7
        }
    });
    let mut c = 0i64;
    unsafe {
        let _ = QueryPerformanceCounter(&mut c);
    }
    c as f64 / freq
}

#[cfg(not(windows))]
pub fn perf_counter() -> f64 {
    use std::sync::OnceLock;
    use std::time::Instant;
    static ORIGIN: OnceLock<Instant> = OnceLock::new();
    ORIGIN.get_or_init(Instant::now).elapsed().as_secs_f64()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn line_keeps_key_order() {
        let l = line(&[("t", json!("played")), ("id", json!(1)), ("dac", json!(round(5.020000001, 6)))]);
        assert_eq!(l, "{\"t\":\"played\",\"id\":1,\"dac\":5.02}\n");
    }
}
