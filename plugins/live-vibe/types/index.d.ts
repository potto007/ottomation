export type LiveState = 'off' | 'loading' | 'listening' | 'user_speaking' | 'transcribing' | 'thinking' | 'speaking'

/** live: speak to Claude directly. livevibe: speak to a voice front that delegates to Claude as the vibe director. */
export type LiveMode = 'live' | 'livevibe'

export type Live = {
  isOn: boolean
  mode: LiveMode
  state: LiveState
  /** The sidecar's HTTP port on 127.0.0.1; 0 until it reports ready. */
  port: number
  /** The main loop's running turn, so a spoken interruption can steer it; null when idle. */
  turnId: string | null
  /** Vibe mode as it was before /livevibe turned it on, restored when live vibe ends. */
  vibeBefore: boolean
  /** Bumped per sidecar start, so a stale sidecar's loop leaves the current one's state alone. */
  gen: number
}

declare module 'claude-code' {
  interface PluginState {
    'live-vibe': { live: Live; isVibe: boolean }
  }
}
