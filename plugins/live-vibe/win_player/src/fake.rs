//! A timer in place of a device: 10 ms blocks, DAC time 30 ms ahead. The sidecar's unit checks run it on Linux.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use crate::engine::Engine;
use crate::out::perf_counter;

pub const LATENCY: f64 = 0.03;

pub struct FakeStream {
    quit: Arc<AtomicBool>,
}

impl FakeStream {
    pub fn start(engine: Arc<Engine>) -> FakeStream {
        let quit = Arc::new(AtomicBool::new(false));
        let q = quit.clone();
        thread::Builder::new()
            .name("fake-device".into())
            .spawn(move || {
                let block = (engine.rate / 100).max(1) as usize;
                let mut buf = vec![0f32; block];
                let mut nxt = perf_counter();
                while !q.load(Ordering::Relaxed) {
                    engine.fill(&mut buf, perf_counter() + LATENCY, false);
                    nxt += block as f64 / engine.rate as f64;
                    let wait = nxt - perf_counter();
                    if wait > 0.0 {
                        thread::sleep(Duration::from_secs_f64(wait));
                    }
                }
            })
            .expect("spawn the fake device");
        FakeStream { quit }
    }

    pub fn abort(&self) {
        self.quit.store(true, Ordering::Relaxed);
    }
}
