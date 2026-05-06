"""midi2tesla.py  –  MIDI → Tesla-coil square-wave MP3 converter

Usage:
    python midi2tesla.py <input.mid> [output.mp3]

Output:
    <input>.mp3  (mono, 44100 Hz, 128 kbps)

Design goals
────────────
• Monophonic output: only the single most prominent note plays at any time.
  Tesla coils produce static with polyphony; the melody note always wins.

• 50% duty-cycle square wave: produces a proper audible tone on any speaker
  or audio device, and drives a Tesla coil interrupter at the correct firing
  rate.  All notes play at maximum amplitude (velocity is ignored for volume).

• Full tempo-map support: all set_tempo events are honoured.

• Pitch-bend and sustain-pedal (CC 64) are tracked per channel.

• Note articulation gap: a short silence is trimmed from the tail of each
  note so that repeated notes sound distinct rather than blurring together.
"""

import sys
import os
import time as _time
import collections
import numpy as np
import mido
import lameenc

# ── Tuning parameters ──────────────────────────────────────────────────────────
SAMPLE_RATE   = 44100   # Hz
DUTY_FRACTION = 0.5     # 50% duty cycle – proper square wave, full amplitude
MP3_BITRATE   = 128     # kbps

# Melody priority range – notes here always beat bass/sub-bass when resolving
# polyphony.  Covers the human-voice / lead-instrument zone.
MELODY_LO = 55          # G3 (~196 Hz)
MELODY_HI = 84          # C6 (~1047 Hz)

# Note articulation gap: samples trimmed from the end of every rendered note
# so that consecutive identical notes sound distinct.
NOTE_GAP_SAMP = 220     # ~5 ms at 44100 Hz


# ── Frequency helpers ──────────────────────────────────────────────────────────

def note_to_freq(note: int, bend: float = 0.0) -> float:
    """Standard equal-temperament frequency for a MIDI note (A4 = 440 Hz)."""
    return 440.0 * 2.0 ** ((note + bend - 69) / 12.0)


def note_to_period(note: int, bend: float = 0.0) -> int:
    """Period in samples for a MIDI note. Returns 0 for sub-20 Hz notes."""
    freq = note_to_freq(note, bend)
    if freq < 20.0:
        return 0
    return max(2, int(SAMPLE_RATE / freq))


# ── Note priority ──────────────────────────────────────────────────────────────

def note_priority(note: int, velocity: int, track_idx: int) -> int:
    """
    Score a note for prominence when resolving polyphony.  Higher = keep.

    Melody-range notes (MELODY_LO–MELODY_HI) always beat bass notes.
    Within a band, higher pitch wins; velocity and track index break ties.
    """
    if MELODY_LO <= note <= MELODY_HI:
        band = 1_000_000
    elif note > MELODY_HI:
        band = 700_000 - (note - MELODY_HI) * 1_000
    else:
        band = 200_000 + note * 5_000
    return band + note * 500 + velocity * 100 + max(0, (16 - min(track_idx, 16))) * 50


# ── MIDI parsing ───────────────────────────────────────────────────────────────

def build_tempo_map(mid: mido.MidiFile) -> list:
    """Build a sorted list of (abs_tick, tempo_µs) from all tracks."""
    events: dict = {0: 500_000}
    for track in mid.tracks:
        abs_tick = 0
        for msg in track:
            abs_tick += msg.time
            if msg.type == 'set_tempo':
                events[abs_tick] = msg.tempo
    return sorted(events.items())


def tick_to_sample(abs_tick: int, tpb: int, tempo_map: list) -> int:
    """Convert an absolute MIDI tick to an audio sample index."""
    sample = 0.0
    prev_tick, prev_tempo = 0, 500_000
    for map_tick, map_tempo in tempo_map:
        if abs_tick <= map_tick:
            break
        sample += mido.tick2second(map_tick - prev_tick, tpb, prev_tempo) * SAMPLE_RATE
        prev_tick, prev_tempo = map_tick, map_tempo
    sample += mido.tick2second(abs_tick - prev_tick, tpb, prev_tempo) * SAMPLE_RATE
    return int(sample)


def collect_mono_events(mid: mido.MidiFile) -> tuple:
    """
    Merge all MIDI tracks into a monophonic timeline of
    (sample_index, 'on'|'off', note_number) events.

    All channels are included.  Polyphony is resolved at every event by
    keeping the highest-priority note.  Sustain pedal and pitch bend are
    tracked per channel.
    """
    tpb       = mid.ticks_per_beat
    tempo_map = build_tempo_map(mid)

    raw: list = []
    for track_idx, track in enumerate(mid.tracks):
        abs_tick = 0
        for msg in track:
            abs_tick += msg.time
            if msg.type in ('note_on', 'note_off', 'pitchwheel', 'control_change'):
                raw.append((abs_tick, msg, track_idx))
    raw.sort(key=lambda x: x[0])

    active: dict      = {}   # note → {velocity, channel, track_idx}
    sustained: dict   = {}   # channel → set of notes deferred by pedal
    pedal_down: dict  = {}   # channel → bool
    pitch_bends: dict = {}   # channel → semitones
    pitch_bend_map: dict = {}
    current_note      = None
    mono: list        = []

    def best():
        if not active:
            return None
        return max(active, key=lambda n: note_priority(
            n, active[n]['velocity'], active[n]['track_idx']))

    def emit(sample: int, new_note) -> None:
        nonlocal current_note
        if new_note == current_note:
            return
        if current_note is not None:
            mono.append((sample, 'off', current_note))
        if new_note is not None:
            pitch_bend_map[new_note] = pitch_bends.get(active[new_note]['channel'], 0.0)
            mono.append((sample, 'on', new_note))
        current_note = new_note

    def handle_note_off(note: int, channel: int, sample: int) -> None:
        if pedal_down.get(channel, False):
            if note in active:
                sustained.setdefault(channel, set()).add(note)
        else:
            active.pop(note, None)
            sustained.get(channel, set()).discard(note)
        emit(sample, best())

    for abs_tick, msg, track_idx in raw:
        sample = tick_to_sample(abs_tick, tpb, tempo_map)

        if msg.type == 'note_on':
            if msg.velocity == 0:
                handle_note_off(msg.note, msg.channel, sample)
            else:
                if note_to_freq(msg.note) >= 20.0:
                    active[msg.note] = {
                        'velocity' : msg.velocity,
                        'channel'  : msg.channel,
                        'track_idx': track_idx,
                    }
                emit(sample, best())

        elif msg.type == 'note_off':
            handle_note_off(msg.note, msg.channel, sample)

        elif msg.type == 'control_change' and msg.control == 64:
            ch = msg.channel
            if msg.value >= 64:
                pedal_down[ch] = True
            else:
                pedal_down[ch] = False
                for note in list(sustained.get(ch, set())):
                    active.pop(note, None)
                sustained.pop(ch, None)
                emit(sample, best())

        elif msg.type == 'pitchwheel':
            semitones = msg.pitch / 4096.0
            pitch_bends[msg.channel] = semitones
            if (current_note is not None and current_note in active
                    and active[current_note]['channel'] == msg.channel):
                pitch_bend_map[current_note] = semitones

    if current_note is not None and raw:
        tail = tick_to_sample(raw[-1][0], tpb, tempo_map) + SAMPLE_RATE // 4
        mono.append((tail, 'off', current_note))

    return mono, pitch_bend_map


# ── Waveform synthesis ─────────────────────────────────────────────────────────

def synthesize(mono_events: list, pitch_bend_map: dict) -> np.ndarray:
    """
    Render the monophonic note event list as 50% duty-cycle square waves.

    Every note plays at maximum amplitude regardless of original velocity.
    A NOTE_GAP_SAMP silence is trimmed from the end of each note segment
    so that consecutive identical notes sound distinct.
    """
    if not mono_events:
        return np.zeros(0, dtype=np.float32)

    total_samples = mono_events[-1][0] + SAMPLE_RATE
    music = np.zeros(total_samples, dtype=np.float32)

    pending: dict  = {}
    segments: list = []
    for sample, etype, note in mono_events:
        if etype == 'on':
            pending[note] = sample
        elif etype == 'off' and note in pending:
            segments.append((pending.pop(note), sample, note))
    for note, start in pending.items():
        segments.append((start, total_samples, note))

    for start, end, note in segments:
        play_end = max(start, end - NOTE_GAP_SAMP)
        length   = play_end - start
        if length <= 0:
            continue

        period = note_to_period(note, pitch_bend_map.get(note, 0.0))
        if period == 0:
            continue
        pw = max(1, int(period * DUTY_FRACTION))

        one_period       = np.zeros(period, dtype=np.float32)
        one_period[:pw]  = 1.0
        tiled            = np.tile(one_period, length // period + 2)
        music[start:play_end] = tiled[:length]

    return music


# ── MP3 output ─────────────────────────────────────────────────────────────────

def save_mp3(music: np.ndarray, path: str) -> None:
    """Normalise to full scale and write an MP3 file via lameenc."""
    peak = float(np.max(np.abs(music)))
    if peak > 0:
        music = music / peak

    samples_int16 = (music * 32767.0).astype(np.int16)

    enc = lameenc.Encoder()
    enc.set_bit_rate(MP3_BITRATE)
    enc.set_in_sample_rate(SAMPLE_RATE)
    enc.set_channels(1)
    enc.set_quality(2)   # 2 = highest quality

    mp3_data  = enc.encode(samples_int16.tobytes())
    mp3_data += enc.flush()

    with open(path, 'wb') as fh:
        fh.write(mp3_data)


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python midi2tesla.py <input.mid> [output.mp3]")
        sys.exit(1)

    input_path  = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) >= 3 else \
                  os.path.splitext(input_path)[0] + ".mp3"

    t0 = _time.monotonic()

    print(f"Loading   {input_path}")
    mid = mido.MidiFile(input_path)
    print(f"  type={mid.type}  ticks/beat={mid.ticks_per_beat}  tracks={len(mid.tracks)}")

    ch_stats: dict = collections.defaultdict(lambda: {'notes': 0, 'vel': [], 'pitches': []})
    ch_track: dict = {}
    for ti, track in enumerate(mid.tracks):
        for msg in track:
            if msg.type == 'note_on' and msg.velocity > 0:
                ch = msg.channel
                ch_stats[ch]['notes']    += 1
                ch_stats[ch]['vel'].append(msg.velocity)
                ch_stats[ch]['pitches'].append(msg.note)
                if ch not in ch_track:
                    ch_track[ch] = (ti, track.name.strip())
    print("  Channel summary:")
    for ch in sorted(ch_stats.keys()):
        s = ch_stats[ch]
        ti, tname = ch_track.get(ch, (0, ''))
        v, p = s['vel'], s['pitches']
        mel_pct = sum(1 for x in p if MELODY_LO <= x <= MELODY_HI) * 100 // len(p)
        print(f"    ch{ch:2d}  track[{ti:2d}] {tname[:12]:12s}  "
              f"{s['notes']:5d} notes  vel {min(v)}-{max(v)}  "
              f"pitch {min(p)}-{max(p)} "
              f"({note_to_freq(min(p)):.0f}-{note_to_freq(max(p)):.0f} Hz)  "
              f"melody-range {mel_pct}%")
    print()

    print("Building monophonic timeline …")
    mono, pitch_bend_map = collect_mono_events(mid)
    note_segments = sum(1 for _, e, _ in mono if e == 'on')
    print(f"  {note_segments} note segments after polyphony reduction")

    if mono:
        selected = [n for _, e, n in mono if e == 'on']
        if selected:
            print(f"  Selected pitch range: MIDI {min(selected)}-{max(selected)}  "
                  f"({note_to_freq(min(selected)):.0f}-{note_to_freq(max(selected)):.0f} Hz)")

    if not mono:
        print("No playable notes found.  Exiting.")
        sys.exit(1)

    print("Synthesising square-wave audio …")
    music = synthesize(mono, pitch_bend_map)
    duration    = len(music) / SAMPLE_RATE
    actual_duty = float(np.mean(music))
    print(f"  {duration:.2f} s  ({len(music)} samples)  avg duty={actual_duty*100:.1f}%")

    print(f"Encoding  {output_path}  ({MP3_BITRATE} kbps) …")
    save_mp3(music, output_path)

    elapsed = _time.monotonic() - t0
    print(f"Done in {elapsed:.1f} s.")


if __name__ == "__main__":
    main()
