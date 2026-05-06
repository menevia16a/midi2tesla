"""midi2tesla.py  –  MIDI → Tesla-coil square-wave WAV converter

Usage:
    python midi2tesla.py <input.mid>

Output:
    <input>.wav  (16-bit PCM, mono, 44100 Hz)

Design goals
────────────
• Monophonic output: only the single most prominent note plays at any time.
  Tesla coils cannot produce overlapping tones cleanly; polyphony produces
  static and obscures the melody.

• Fixed duty cycle: all notes are rendered at the same pulse width fraction
  regardless of original velocity, so perceived volume is consistent.
  Velocity is used only to rank note importance during overlap resolution.

• Intelligent note priority: when multiple notes overlap, the winner is chosen
  by (1) highest velocity, (2) melodic range bonus (MIDI 60–84), (3) highest
  pitch.  When the winner changes mid-note, a seamless switch is emitted so
  the melody stays in front and the song remains recognisable.

• Full tempo-map support: all set_tempo events across all tracks are honoured,
  including mid-song tempo changes.

• Drum channel (9) is excluded from output.

• Pitch-bend messages are tracked per channel and applied to the active note's
  frequency.

• Rolling duty-cycle safety limiter prevents coil overheating during
  dense passages.
"""

import sys
import os
import time as _time
import numpy as np
import mido
import soundfile as sf

# ── Tuning parameters ──────────────────────────────────────────────────────────
SAMPLE_RATE    = 44100      # Hz
A3_REF_HZ      = 220.0      # Hz for MIDI note 57 (A3).
                            # Combined with the /2 period trick below, this
                            # produces the correct pitch (tested empirically).

DUTY_FRACTION  = 0.04       # Fraction of one period that the pulse is HIGH.
                            # 4 % gives a clean, audible tone.  Raise for more
                            # apparent volume; lower for less coil stress.
MIN_PULSE_SAMP = 2          # Hard floor: pulse never shorter than this many samples
MAX_PULSE_SAMP = 12         # Hard ceiling: pulse never longer than this many samples

# Rolling coil-safety limiter ─────────────────────────────────────────────────
# If the rolling-average duty cycle in any SAFETY_WINDOW-sample window exceeds
# SAFETY_DUTY_MAX, the output is zeroed for those samples.  This prevents the
# coil from drawing excessive power during very dense passages.
SAFETY_DUTY_MAX = 0.45      # 45 % rolling-average ceiling
SAFETY_WINDOW   = 1000      # rolling window length in samples

DRUM_CHANNEL = 9            # 0-indexed MIDI drum channel – excluded from output

# ── Frequency / period helpers ─────────────────────────────────────────────────

def note_to_period(note: int, pitch_bend_semitones: float = 0.0) -> int:
    """
    Return the half-period (samples) for a MIDI note with optional pitch bend.

    The /2 in the denominator is intentional: it compensates for A3_REF_HZ
    being 220 (one octave below A4=440) while MIDI note 69 should map to A4.
    Together they produce the correct perceived pitch.
    """
    freq = A3_REF_HZ * 2.0 ** ((note + pitch_bend_semitones - 69) / 12.0)
    return max(2, int(SAMPLE_RATE / freq / 2))


def pulse_width_for_period(period: int) -> int:
    """Compute on-time samples for a given period, clamped to safe bounds."""
    return max(MIN_PULSE_SAMP, min(MAX_PULSE_SAMP, int(period * DUTY_FRACTION)))


# ── MIDI parsing ───────────────────────────────────────────────────────────────

def build_tempo_map(mid: mido.MidiFile) -> list:
    """
    Scan every track for set_tempo events and build a sorted tempo map.

    Returns a list of (abs_tick, tempo_µs_per_beat) tuples, starting at tick 0.
    If multiple set_tempo events land on the same tick (e.g. in different
    tracks), the last one wins (standard behaviour).
    """
    events: dict = {0: 500_000}   # default: 120 BPM
    for track in mid.tracks:
        abs_tick = 0
        for msg in track:
            abs_tick += msg.time
            if msg.type == 'set_tempo':
                events[abs_tick] = msg.tempo
    return sorted(events.items())


def tick_to_sample(abs_tick: int, ticks_per_beat: int, tempo_map: list) -> int:
    """
    Convert an absolute MIDI tick to an absolute audio sample index.

    Correctly accounts for all tempo changes up to abs_tick.
    """
    sample = 0.0
    prev_tick, prev_tempo = 0, 500_000
    for map_tick, map_tempo in tempo_map:
        if abs_tick <= map_tick:
            break
        sample += (
            mido.tick2second(map_tick - prev_tick, ticks_per_beat, prev_tempo)
            * SAMPLE_RATE
        )
        prev_tick, prev_tempo = map_tick, map_tempo
    sample += (
        mido.tick2second(abs_tick - prev_tick, ticks_per_beat, prev_tempo)
        * SAMPLE_RATE
    )
    return int(sample)


def note_priority(note: int, velocity: int) -> int:
    """
    Score a note for prominence when resolving polyphony.  Higher = keep.

    Priority order:
      1. Velocity (loudest note is most important).
      2. Melodic range: MIDI 60–84 (middle C to C5) gets a bonus since the
         melody almost always lives here.  Notes above 84 get a smaller bonus.
      3. Pitch as tiebreaker: higher pitch wins (melody sits above bass).
    """
    if 60 <= note <= 84:
        range_bonus = 200
    elif note > 84:
        range_bonus = 100
    else:
        range_bonus = 0
    return velocity * 10_000 + range_bonus + note


def collect_mono_events(mid: mido.MidiFile) -> tuple:
    """
    Parse all MIDI tracks, resolve polyphony, and return a flat list of

        (sample_index, 'on' | 'off', note_number)

    describing a *strictly monophonic* note sequence.

    Algorithm
    ─────────
    All events from all tracks are merged into a single timeline sorted by
    absolute tick.  A set of "active notes" is maintained; at every event the
    highest-priority active note is re-evaluated.  If the winner changes, an
    'off' for the outgoing note and an 'on' for the incoming note are emitted
    at that exact sample.  This ensures seamless melody-over-accompaniment
    behaviour and clean note transitions.

    Pitch-bend messages are tracked per channel; the active note's period is
    adjusted accordingly during synthesis.
    """
    tpb       = mid.ticks_per_beat
    tempo_map = build_tempo_map(mid)

    # ── Collect all relevant messages with absolute tick times ────────────────
    raw: list = []
    for track in mid.tracks:
        abs_tick = 0
        for msg in track:
            abs_tick += msg.time
            if msg.type in ('note_on', 'note_off', 'pitchwheel'):
                raw.append((abs_tick, msg))
    raw.sort(key=lambda x: x[0])

    # ── Simulate polyphony and emit monophonic transitions ───────────────────
    active: dict = {}        # note → {'velocity': int, 'channel': int}
    pitch_bends: dict = {}   # channel → semitone offset (float)
    current_note = None      # the note currently outputting
    mono: list = []          # result: (sample, 'on'|'off', note)
    pitch_bend_map: dict = {}  # note → semitone bend at time of 'on' event

    def best_note():
        if not active:
            return None
        return max(active, key=lambda n: note_priority(n, active[n]['velocity']))

    def emit_transition(sample: int, new_note) -> None:
        nonlocal current_note
        if new_note == current_note:
            return
        if current_note is not None:
            mono.append((sample, 'off', current_note))
        if new_note is not None:
            bend = pitch_bends.get(active[new_note]['channel'], 0.0)
            pitch_bend_map[new_note] = bend
            mono.append((sample, 'on', new_note))
        current_note = new_note

    for abs_tick, msg in raw:
        if msg.channel == DRUM_CHANNEL:
            continue

        sample = tick_to_sample(abs_tick, tpb, tempo_map)

        if msg.type == 'note_on':
            if msg.velocity == 0:
                # Velocity-0 note_on is equivalent to note_off (MIDI spec)
                active.pop(msg.note, None)
            else:
                active[msg.note] = {'velocity': msg.velocity, 'channel': msg.channel}
            emit_transition(sample, best_note())

        elif msg.type == 'note_off':
            active.pop(msg.note, None)
            emit_transition(sample, best_note())

        elif msg.type == 'pitchwheel':
            # pitch value: −8192 to +8191; map to ±2 semitones
            semitones = msg.pitch / 4096.0
            pitch_bends[msg.channel] = semitones
            # If the currently playing note is on this channel, update its bend
            if current_note is not None and current_note in active:
                if active[current_note]['channel'] == msg.channel:
                    pitch_bend_map[current_note] = semitones

    # Emit final note-off at the tail
    if current_note is not None and raw:
        tail = tick_to_sample(raw[-1][0], tpb, tempo_map) + SAMPLE_RATE // 4
        mono.append((tail, 'off', current_note))

    return mono, pitch_bend_map


# ── Waveform synthesis ─────────────────────────────────────────────────────────

def synthesize(mono_events: list, pitch_bend_map: dict) -> np.ndarray:
    """
    Render the monophonic note event list into a float32 square-wave buffer.

    Every note is rendered at a fixed duty cycle (DUTY_FRACTION of its period),
    independent of the original MIDI velocity.  This gives consistent perceived
    volume across all notes and eliminates the static caused by mixing
    different pulse widths.

    Pitch-bend offsets stored in pitch_bend_map are applied per note.
    """
    if not mono_events:
        return np.zeros(0, dtype=np.float32)

    total_samples = mono_events[-1][0] + SAFETY_WINDOW + SAMPLE_RATE // 4
    music = np.zeros(total_samples, dtype=np.float32)

    # Pair on/off events into (start_sample, end_sample, note) segments
    pending: dict = {}
    segments: list = []
    for sample, etype, note in mono_events:
        if etype == 'on':
            pending[note] = sample
        elif etype == 'off' and note in pending:
            segments.append((pending.pop(note), sample, note))
    for note, start in pending.items():
        segments.append((start, total_samples, note))

    for start, end, note in segments:
        length = end - start
        if length <= 0:
            continue
        bend   = pitch_bend_map.get(note, 0.0)
        period = note_to_period(note, bend)
        pw     = pulse_width_for_period(period)

        # Build one period then tile it to cover the full segment length
        one_period = np.zeros(period, dtype=np.float32)
        one_period[:pw] = 1.0
        repeats = length // period + 2
        tiled   = np.tile(one_period, repeats)
        music[start:end] = tiled[:length]

    return music


# ── Post-processing ────────────────────────────────────────────────────────────

def safety_limiter(music: np.ndarray) -> np.ndarray:
    """
    Rolling duty-cycle safety limiter.

    Zeros out any samples in windows where the rolling-average duty cycle
    exceeds SAFETY_DUTY_MAX.  This protects the coil from overheating during
    passages where many short high-frequency notes would push average power up.
    """
    kernel = np.ones(SAFETY_WINDOW, dtype=np.float32) / SAFETY_WINDOW
    avg    = np.convolve(music, kernel, mode='valid')   # length = N − W + 1
    mask   = (avg < SAFETY_DUTY_MAX).astype(np.float32)
    return music[:len(mask)] * mask


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python midi2tesla.py <input.mid>")
        sys.exit(1)

    input_path  = sys.argv[1]
    output_path = os.path.splitext(input_path)[0] + ".wav"

    t0 = _time.monotonic()

    print(f"Loading   {input_path}")
    mid = mido.MidiFile(input_path)
    print(f"  ticks/beat : {mid.ticks_per_beat}")
    print(f"  tracks     : {len(mid.tracks)}")

    print("Building monophonic timeline …")
    mono, pitch_bend_map = collect_mono_events(mid)
    note_segments = sum(1 for _, e, _ in mono if e == 'on')
    print(f"  {note_segments} note segments after polyphony reduction")

    if not mono:
        print("No playable notes found.  Exiting.")
        sys.exit(1)

    print("Synthesising square-wave pulses …")
    music = synthesize(mono, pitch_bend_map)
    duration = len(music) / SAMPLE_RATE
    print(f"  {duration:.2f} s  ({len(music)} samples)")

    print("Applying safety duty-cycle limiter …")
    music = safety_limiter(music)

    print(f"Saving    {output_path}")
    sf.write(output_path, music, SAMPLE_RATE, subtype='PCM_16')

    elapsed = _time.monotonic() - t0
    print(f"Done in {elapsed:.1f} s.")


if __name__ == "__main__":
    main()
