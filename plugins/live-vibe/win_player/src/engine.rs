//! Clips in order; the device thread takes from the head. A clip whose next samples have not arrived yet plays
//! silence until they do (the sidecar sends a whole clip at once, so that only happens on a slow pipe).

use std::collections::{HashSet, VecDeque};
use std::sync::{Arc, Mutex};

use serde_json::{json, Value};

use crate::out::{round, Out};

pub trait Sink: Send + Sync {
    fn send(&self, pairs: &[(&str, Value)]);
}

impl Sink for Out {
    fn send(&self, pairs: &[(&str, Value)]) {
        Out::send(self, pairs)
    }
}

struct Clip {
    id: u32,
    total: u64,
    chunks: VecDeque<Vec<f32>>,
    head: usize,   // samples of chunks[0] already taken
    consumed: u64, // samples handed to the device
}

struct Inner {
    clips: VecDeque<Clip>,
    dropped: HashSet<u32>, // cancelled ids: late chunks of them are ignored
    frames: u64,           // device frames so far
    xruns: u64,
    t: Option<f64>, // perf-counter DAC time of frame `frames`, kept continuous
}

pub struct Engine {
    pub rate: u32,
    out: Arc<dyn Sink>,
    inner: Mutex<Inner>,
}

impl Engine {
    pub const FOLLOW: f64 = 0.02;
    pub const JUMP_S: f64 = 0.02;

    pub fn new(out: Arc<dyn Sink>, rate: u32) -> Engine {
        Engine {
            rate,
            out,
            inner: Mutex::new(Inner { clips: VecDeque::new(), dropped: HashSet::new(), frames: 0, xruns: 0, t: None }),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Inner> {
        self.inner.lock().unwrap_or_else(|e| e.into_inner())
    }

    pub fn add(&self, cid: u32, total: u32, _offset: u32, samples: Vec<f32>) {
        let mut g = self.lock();
        if g.dropped.contains(&cid) {
            return;
        }
        let idx = match g.clips.iter().position(|c| c.id == cid) {
            Some(i) => i,
            None => {
                g.clips.push_back(Clip { id: cid, total: total as u64, chunks: VecDeque::new(), head: 0, consumed: 0 });
                g.clips.len() - 1
            }
        };
        if !samples.is_empty() {
            g.clips[idx].chunks.push_back(samples);
        }
    }

    /// Drops every clip up to and including cid; returns how much of cid was played.
    pub fn cancel(&self, cid: u32) -> u64 {
        let mut g = self.lock();
        let mut played = 0;
        g.dropped.insert(cid);
        while g.clips.front().is_some_and(|c| c.id <= cid) {
            let c = g.clips.pop_front().expect("front checked");
            if c.id == cid {
                played = c.consumed;
            }
        }
        played
    }

    pub fn counts(&self) -> (u64, u64) {
        let g = self.lock();
        (g.frames, g.xruns)
    }

    /// The DAC time of this block's first frame: anchored, advanced by frame count, following the report slowly
    /// and re-anchoring on a jump (a restarted stream), so consecutive blocks never overlap or leave gaps.
    fn clock(g: &mut Inner, reported: f64) -> f64 {
        let t = match g.t {
            Some(t) if (reported - t).abs() <= Self::JUMP_S => t + Self::FOLLOW * (reported - t),
            _ => reported,
        };
        g.t = Some(t);
        t
    }

    pub fn fill(&self, out: &mut [f32], dac: f64, underflow: bool) {
        let frames = out.len();
        let rate = self.rate as f64;
        let mut g = self.lock();
        let start = Self::clock(&mut g, dac);
        g.t = Some(start + frames as f64 / rate);
        g.frames += frames as u64;
        if underflow {
            g.xruns += 1;
            self.out.send(&[("t", json!("xrun")), ("total", json!(g.xruns))]);
        }
        let mut n = 0usize;
        while n < frames {
            let Some(c) = g.clips.front_mut() else { break };
            if c.chunks.is_empty() {
                if c.consumed >= c.total {
                    let (id, consumed) = (c.id, c.consumed);
                    g.clips.pop_front();
                    self.out.send(&[("t", json!("end")), ("id", json!(id)), ("n", json!(consumed))]);
                    continue;
                }
                break; // the rest of this clip has not arrived
            }
            let chunk = &c.chunks[0][c.head..];
            let k = (frames - n).min(chunk.len());
            out[n..n + k].copy_from_slice(&chunk[..k]);
            self.out.send(&[
                ("t", json!("played")),
                ("id", json!(c.id)),
                ("at", json!(c.consumed)),
                ("n", json!(k)),
                ("dac", json!(round(start + n as f64 / rate, 6))),
            ]);
            c.consumed += k as u64;
            n += k;
            if k == chunk.len() {
                c.chunks.pop_front();
                c.head = 0;
            } else {
                c.head += k;
            }
            if c.chunks.is_empty() && c.consumed >= c.total {
                let (id, consumed) = (c.id, c.consumed);
                g.clips.pop_front();
                self.out.send(&[("t", json!("end")), ("id", json!(id)), ("n", json!(consumed))]);
            }
        }
        out[n..].fill(0.0);
    }

    #[cfg(test)]
    fn clip_count(&self) -> usize {
        self.lock().clips.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Default)]
    struct Sent(Mutex<Vec<Value>>);

    impl Sink for Sent {
        fn send(&self, pairs: &[(&str, Value)]) {
            let m: serde_json::Map<String, Value> = pairs.iter().map(|(k, v)| (k.to_string(), v.clone())).collect();
            self.0.lock().unwrap().push(Value::Object(m));
        }
    }

    impl Sent {
        fn take(&self, t: &str) -> Vec<Value> {
            self.0.lock().unwrap().iter().filter(|m| m["t"] == t).cloned().collect()
        }
        fn clear(&self) {
            self.0.lock().unwrap().clear();
        }
    }

    fn rig() -> (Arc<Sent>, Engine) {
        let sent = Arc::new(Sent::default());
        let eng = Engine::new(sent.clone(), 1000);
        (sent, eng)
    }

    #[test]
    fn clips_back_to_back_with_dac_times() {
        let (sent, eng) = rig();
        eng.add(1, 30, 0, vec![0.1; 20]);
        eng.add(1, 30, 20, vec![0.2; 10]);
        eng.add(2, 15, 0, vec![0.3; 15]);
        let mut buf = vec![1.0f32; 40];
        eng.fill(&mut buf, 5.0, false);
        let played: Vec<(u64, u64, u64, f64)> = sent
            .take("played")
            .iter()
            .map(|m| {
                (
                    m["id"].as_u64().unwrap(),
                    m["at"].as_u64().unwrap(),
                    m["n"].as_u64().unwrap(),
                    m["dac"].as_f64().unwrap(),
                )
            })
            .collect();
        assert_eq!(played, vec![(1, 0, 20, 5.0), (1, 20, 10, 5.02), (2, 0, 10, 5.03)]);
        let ends: Vec<(u64, u64)> =
            sent.take("end").iter().map(|m| (m["id"].as_u64().unwrap(), m["n"].as_u64().unwrap())).collect();
        assert_eq!(ends, vec![(1, 30)]);
        assert!(buf[..20].iter().all(|&x| x == 0.1f32) && buf[30..].iter().all(|&x| x == 0.3f32));

        // 0.1 ms off the running clock: followed, not re-anchored; silence after the last clip; underflow counted
        sent.clear();
        eng.fill(&mut buf, 5.0401, true);
        let first = &sent.take("played")[0];
        assert!((first["dac"].as_f64().unwrap() - 5.04).abs() < 1e-4);
        assert!(buf[5..].iter().all(|&x| x == 0.0));
        assert_eq!(sent.take("end").len(), 1);
        assert_eq!(sent.take("xrun")[0]["total"], 1);
        assert_eq!(eng.counts(), (80, 1));
    }

    #[test]
    fn cancel_reports_played_and_drops_late_chunks() {
        let (_sent, eng) = rig();
        eng.add(3, 100, 0, vec![0.4; 50]);
        eng.fill(&mut [0.0; 10], 6.0, false);
        let cut = eng.cancel(3);
        eng.add(3, 100, 50, vec![0.4; 50]); // a late chunk of the cancelled clip
        let mut tail = [1.0f32; 10];
        eng.fill(&mut tail, 6.01, false);
        assert_eq!(cut, 10);
        assert_eq!(eng.clip_count(), 0);
        assert!(tail.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn missing_chunks_play_silence_then_resume() {
        let (sent, eng) = rig();
        eng.add(1, 20, 0, vec![0.5; 5]);
        let mut buf = [1.0f32; 10];
        eng.fill(&mut buf, 1.0, false);
        assert!(buf[..5].iter().all(|&x| x == 0.5) && buf[5..].iter().all(|&x| x == 0.0));
        eng.add(1, 20, 5, vec![0.6; 15]);
        eng.fill(&mut buf, 1.01, false);
        eng.fill(&mut buf, 1.02, false);
        assert_eq!(sent.take("end")[0]["n"], 20);
    }

    #[test]
    fn a_jump_re_anchors_the_clock() {
        let (sent, eng) = rig();
        eng.add(1, 100, 0, vec![0.1; 100]);
        eng.fill(&mut [0.0; 10], 1.0, false);
        eng.fill(&mut [0.0; 10], 3.0, false); // a restarted stream
        let p = sent.take("played");
        assert_eq!(p[1]["dac"].as_f64().unwrap(), 3.0);
    }
}
