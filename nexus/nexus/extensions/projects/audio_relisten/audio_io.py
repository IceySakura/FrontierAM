"""Audio I/O helpers for the audio_relisten task.

Loads audio at its **native sample rate** (no resampling, no normalisation
beyond float32 in ``[-1, 1]``) and re-encodes mono segments to 16-bit PCM
WAV bytes — exactly the on-the-wire shape vLLM / OpenAI expect for the
``input_audio`` multimodal content part:

    {"type": "input_audio",
     "input_audio": {"data": <base64 wav bytes>, "format": "wav"}}

vLLM decodes that data URI back through ``librosa.load(BytesIO, sr=None)``,
so the round-trip ``load_wav → encode_wav_bytes → vLLM`` preserves the
original sample rate.
"""

from __future__ import annotations

import io
import wave

import numpy as np


def load_wav(path: str) -> tuple[np.ndarray, int]:
    """Load an audio file as ``(samples_float32, sample_rate)``.

    Thin wrapper over ``librosa.load(path, sr=None, mono=True)``:

    - ``sr=None`` keeps the file's **native** sample rate — no resampling,
      so the returned waveform is the original signal (this is the same
      call vLLM uses inside ``MediaConnector.fetch_audio``).
    - ``mono=True`` mixes multi-channel input down to a single channel,
      matching what ``encode_wav_bytes`` and audio-LLM feature extractors
      expect.
    - librosa transparently handles every WAV variant the stdlib ``wave``
      module rejects (``WAVE_FORMAT_IEEE_FLOAT``, ``WAVE_FORMAT_EXTENSIBLE``)
      plus mp3 / flac / ogg via libsndfile / audioread.
    """
    try:
        import librosa
    except ImportError as exc:  # pragma: no cover - dependency check
        raise ImportError(
            "load_wav requires `librosa`. Install it in the runtime image (e.g. `pip install 'librosa>=0.10'`)."
        ) from exc

    samples, sample_rate = librosa.load(path, sr=None, mono=True)
    return np.ascontiguousarray(samples, dtype=np.float32), int(sample_rate)


def encode_wav_bytes(samples: np.ndarray, sample_rate: int) -> bytes:
    """Encode mono float32 ``samples`` ``∈ [-1, 1]`` to a 16-bit PCM WAV bytestring."""
    if samples.ndim != 1:
        samples = samples.reshape(-1)
    samples = np.clip(samples, -1.0, 1.0)
    pcm = (samples * 32767.0).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


def slice_audio(
    samples: np.ndarray,
    sample_rate: int,
    start: float,
    end: float,
) -> tuple[np.ndarray, float, float]:
    """Slice ``samples[start*sr : end*sr]``.

    Clamps to valid ranges. Returns the sliced waveform plus the *effective*
    ``(start_clamped, end_clamped)`` so the caller can report what was
    actually returned.
    """
    if sample_rate <= 0:
        raise ValueError(f'sample_rate must be positive, got {sample_rate}')

    duration = len(samples) / float(sample_rate)
    start_c = max(0.0, float(start))
    end_c = min(float(duration), float(end))
    if end_c <= start_c:
        # Caller should treat this as an error — return an empty array.
        return samples[0:0], start_c, end_c

    i0 = int(round(start_c * sample_rate))
    i1 = int(round(end_c * sample_rate))
    return samples[i0:i1], start_c, end_c
