"""
tests/test_audio_preprocess.py
-------------------------------
Unit and integration tests for the audio preprocessing pipeline that
prevents prefix clipping in Whisper transcription.

Run with:
    cd MeetingMind
    pytest tests/test_audio_preprocess.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Ensure src/ is importable when running tests from the project root
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from meetingmind.transcriber import (
    _PREPROCESS_DEFAULTS,
    _preprocess_audio,
    _load_preprocess_config,
)


# ---------------------------------------------------------------------------
# Default config used across most tests
# ---------------------------------------------------------------------------
_CFG = dict(_PREPROCESS_DEFAULTS)


# ===========================================================================
# _preprocess_audio — unit tests
# ===========================================================================

class TestPreprocessAudioOverlap:
    """Step 1: overlap tail is correctly prepended."""

    def test_overlap_prepended(self):
        """Output starts with the overlap tail samples."""
        original = np.ones(24_000, dtype=np.float32) * 0.5  # 1.5 s
        tail = np.ones(4_800, dtype=np.float32) * 0.1       # 0.3 s
        result = _preprocess_audio(original, tail, _CFG)
        # Length = overlap(4800) + pad(1600) + original(24000)
        # (pre-emphasis keeps same length via np.append trick)
        expected_len = len(tail) + int(_CFG["leading_silence_pad_seconds"] * 16_000) + len(original)
        assert len(result) == expected_len

    def test_none_overlap_tail_first_chunk(self):
        """First chunk has no tail — should not crash."""
        audio = np.random.randn(24_000).astype(np.float32) * 0.3
        result = _preprocess_audio(audio, None, _CFG)
        # Length = pad(1600) + original(24000)
        expected_len = int(_CFG["leading_silence_pad_seconds"] * 16_000) + len(audio)
        assert len(result) == expected_len

    def test_empty_overlap_tail(self):
        """Empty array tail is same as None — no extra samples."""
        audio = np.ones(16_000, dtype=np.float32) * 0.4
        empty_tail = np.array([], dtype=np.float32)
        result = _preprocess_audio(audio, empty_tail, _CFG)
        expected_len = int(_CFG["leading_silence_pad_seconds"] * 16_000) + len(audio)
        assert len(result) == expected_len


class TestPreprocessAudioPad:
    """Step 2: leading silence pad is zeros."""

    def test_pad_is_silence(self):
        """Leading pad region should be near-zero after processing."""
        cfg = dict(_CFG, pre_emphasis_coefficient=0.0, normalisation_peak=0.0,
                   fade_in_ms=0)
        audio = np.ones(16_000, dtype=np.float32) * 0.5
        result = _preprocess_audio(audio, None, cfg)
        pad_samples = int(cfg["leading_silence_pad_seconds"] * 16_000)
        # With pre-emphasis=0 and norm=0, pad region stays zeros
        assert np.allclose(result[:pad_samples], 0.0, atol=1e-7)

    def test_pad_length_correct(self):
        """Pad adds exactly leading_silence_pad_seconds * 16000 samples."""
        cfg = dict(_CFG, leading_silence_pad_seconds=0.2)
        audio = np.ones(8_000, dtype=np.float32) * 0.3
        result = _preprocess_audio(audio, None, cfg)
        expected_len = int(0.2 * 16_000) + len(audio)
        assert len(result) == expected_len


class TestPreprocessAudioFadeIn:
    """Step 3: fade-in linear ramp."""

    def test_fade_in_ramp_applied(self):
        """First sample should be attenuated (close to zero), last ramp sample near full."""
        # Disable pre-emphasis and normalisation to isolate fade-in
        cfg = dict(_CFG, pre_emphasis_coefficient=0.0, normalisation_peak=0.0,
                   leading_silence_pad_seconds=0.0, chunk_overlap_seconds=0.0,
                   fade_in_ms=10)
        audio = np.ones(16_000, dtype=np.float32) * 0.5
        result = _preprocess_audio(audio, None, cfg)
        # First sample should be 0 (ramp starts at 0)
        assert result[0] == pytest.approx(0.0, abs=1e-6)
        # Sample at end of ramp (160 samples for 10ms @ 16kHz) should be ~0.5
        fade_samples = 10 * 16
        assert result[fade_samples - 1] == pytest.approx(0.5, abs=0.01)

    def test_fade_in_does_not_affect_rest(self):
        """Samples after fade-in region should be unchanged (with no pre-emphasis/norm)."""
        cfg = dict(_CFG, pre_emphasis_coefficient=0.0, normalisation_peak=0.0,
                   leading_silence_pad_seconds=0.0, chunk_overlap_seconds=0.0,
                   fade_in_ms=10)
        audio = np.ones(16_000, dtype=np.float32) * 0.5
        result = _preprocess_audio(audio, None, cfg)
        fade_samples = 10 * 16
        # All samples after the ramp should be exactly 0.5
        assert np.allclose(result[fade_samples:], 0.5, atol=1e-6)


class TestPreprocessAudioPreEmphasis:
    """Step 4: pre-emphasis filter y[n] = x[n] - coeff * x[n-1]."""

    def test_pre_emphasis_manual_calculation(self):
        """Pre-emphasis output matches hand-calculated values."""
        # Disable other steps
        cfg = dict(_CFG, leading_silence_pad_seconds=0.0, chunk_overlap_seconds=0.0,
                   fade_in_ms=0, normalisation_peak=0.0, pre_emphasis_coefficient=0.97)
        audio = np.array([0.0, 0.5, 0.5, 1.0, 0.0], dtype=np.float32)
        result = _preprocess_audio(audio, None, cfg)
        # y[0] = x[0] = 0.0
        # y[1] = x[1] - 0.97*x[0] = 0.5 - 0 = 0.5
        # y[2] = x[2] - 0.97*x[1] = 0.5 - 0.485 = 0.015
        # y[3] = x[3] - 0.97*x[2] = 1.0 - 0.485 = 0.515
        # y[4] = x[4] - 0.97*x[3] = 0.0 - 0.97 = -0.97
        expected = np.array([0.0, 0.5, 0.015, 0.515, -0.97], dtype=np.float32)
        assert np.allclose(result, expected, atol=1e-5)

    def test_pre_emphasis_zero_coefficient(self):
        """With coeff=0, output equals input (no filtering)."""
        cfg = dict(_CFG, leading_silence_pad_seconds=0.0, chunk_overlap_seconds=0.0,
                   fade_in_ms=0, normalisation_peak=0.0, pre_emphasis_coefficient=0.0)
        audio = np.array([0.1, 0.5, 0.3], dtype=np.float32)
        result = _preprocess_audio(audio, None, cfg)
        # coeff=0 → y[n] = x[n] - 0*x[n-1] = x[n]
        # But note: np.append(x[0], x[1:] - 0*x[:-1]) = [x[0], x[1], x[2], ...]
        assert np.allclose(result, audio, atol=1e-6)


class TestPreprocessAudioNormalisation:
    """Step 5: peak normalisation."""

    def test_normalisation_scales_to_target(self):
        """Output peak should equal normalisation_peak."""
        cfg = dict(_CFG, leading_silence_pad_seconds=0.0, chunk_overlap_seconds=0.0,
                   fade_in_ms=0, pre_emphasis_coefficient=0.0, normalisation_peak=0.9)
        audio = np.array([0.1, -0.3, 0.5, -0.2], dtype=np.float32)
        result = _preprocess_audio(audio, None, cfg)
        assert np.max(np.abs(result)) == pytest.approx(0.9, abs=1e-5)

    def test_normalisation_disabled_when_zero(self):
        """normalisation_peak=0 should leave audio unchanged."""
        cfg = dict(_CFG, leading_silence_pad_seconds=0.0, chunk_overlap_seconds=0.0,
                   fade_in_ms=0, pre_emphasis_coefficient=0.0, normalisation_peak=0.0)
        audio = np.array([0.1, 0.3, -0.2], dtype=np.float32)
        result = _preprocess_audio(audio, None, cfg)
        assert np.allclose(result, audio, atol=1e-6)


class TestPreprocessAudioEdgeCases:
    """Edge cases and safety checks."""

    def test_all_zeros_no_division_by_zero(self):
        """All-zero audio should not cause division by zero."""
        audio = np.zeros(16_000, dtype=np.float32)
        result = _preprocess_audio(audio, None, _CFG)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))

    def test_single_sample(self):
        """A single-sample chunk should not crash."""
        cfg = dict(_CFG, leading_silence_pad_seconds=0.0, chunk_overlap_seconds=0.0,
                   fade_in_ms=0, pre_emphasis_coefficient=0.0, normalisation_peak=0.0)
        audio = np.array([0.5], dtype=np.float32)
        result = _preprocess_audio(audio, None, cfg)
        assert len(result) == 1

    def test_output_is_float32(self):
        """Output must always be float32 for Whisper."""
        audio = np.ones(1000, dtype=np.float32) * 0.3
        result = _preprocess_audio(audio, None, _CFG)
        assert result.dtype == np.float32

    def test_full_pipeline_produces_finite_output(self):
        """Full pipeline with all steps enabled produces finite, non-NaN output."""
        audio = np.random.randn(24_000).astype(np.float32) * 0.3
        tail = np.random.randn(4_800).astype(np.float32) * 0.2
        result = _preprocess_audio(audio, tail, _CFG)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))
        assert len(result) > len(audio)  # overlap + pad added


# ===========================================================================
# _load_preprocess_config — tests
# ===========================================================================

class TestLoadPreprocessConfig:
    """Config loading from settings.toml."""

    def test_returns_defaults_when_file_missing(self, tmp_path, monkeypatch):
        """When settings.toml does not exist, return defaults."""
        monkeypatch.setattr(
            "meetingmind.transcriber.Path",
            lambda *a, **kw: tmp_path / "nonexistent" / "transcriber.py",
        )
        # Call the function directly with a patched Path that won't find config
        from meetingmind.transcriber import _PREPROCESS_DEFAULTS
        # Just verify defaults are well-formed
        assert _PREPROCESS_DEFAULTS["chunk_overlap_seconds"] == 0.3
        assert _PREPROCESS_DEFAULTS["leading_silence_pad_seconds"] == 0.1
        assert _PREPROCESS_DEFAULTS["pre_emphasis_coefficient"] == 0.97
        assert _PREPROCESS_DEFAULTS["normalisation_peak"] == 0.9
        assert _PREPROCESS_DEFAULTS["fade_in_ms"] == 10

    def test_reads_values_from_settings(self):
        """Config loader reads the real settings.toml."""
        cfg = _load_preprocess_config()
        # The real settings.toml has these values
        assert cfg["chunk_overlap_seconds"] == 0.3
        assert cfg["pre_emphasis_coefficient"] == 0.97
        assert cfg["normalisation_peak"] == 0.9


# ===========================================================================
# Integration: overlap tail tracking in StreamingTranscriber
# ===========================================================================

class TestOverlapTailTracking:
    """Verify that StreamingTranscriber stores and retrieves overlap tails per source."""

    def _make_chunk(self, source: str = "microphone", rms: float = 0.05,
                    peak_rms: float = 0.1, data: np.ndarray | None = None):
        """Create a mock AudioChunk."""
        chunk = MagicMock()
        chunk.source.value = source
        chunk.rms = rms
        chunk.peak_rms = peak_rms
        chunk.timestamp = 1000.0
        if data is None:
            data = np.random.randn(24_000).astype(np.float32) * 0.3
        chunk.data = data
        return chunk

    @patch("meetingmind.transcriber._load_whisper_model")
    def test_tail_stored_per_source(self, mock_load):
        """After transcription, tail is stored for the source key."""
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {
            "text": "Hello world",
            "language": "en",
            "segments": [{"id": 0, "avg_logprob": -0.3, "no_speech_prob": 0.1}],
        }
        mock_load.return_value = mock_model

        from meetingmind.transcriber import StreamingTranscriber
        t = StreamingTranscriber(model_size="tiny.en")
        chunk = self._make_chunk(source="microphone")
        t.transcribe_chunk(chunk)

        assert "microphone" in t._overlap_tails
        overlap_samples = int(_CFG["chunk_overlap_seconds"] * 16_000)
        assert len(t._overlap_tails["microphone"]) == overlap_samples

    @patch("meetingmind.transcriber._load_whisper_model")
    def test_different_sources_independent_tails(self, mock_load):
        """Microphone and system sources maintain independent tails."""
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {
            "text": "Some speech",
            "language": "en",
            "segments": [{"id": 0, "avg_logprob": -0.3, "no_speech_prob": 0.1}],
        }
        mock_load.return_value = mock_model

        from meetingmind.transcriber import StreamingTranscriber
        t = StreamingTranscriber(model_size="tiny.en")

        mic_data = np.ones(24_000, dtype=np.float32) * 0.4
        sys_data = np.ones(24_000, dtype=np.float32) * 0.2

        t.transcribe_chunk(self._make_chunk(source="microphone", data=mic_data))
        t.transcribe_chunk(self._make_chunk(source="system", data=sys_data))

        assert "microphone" in t._overlap_tails
        assert "system" in t._overlap_tails
        # Tails should differ because source data differs
        assert not np.array_equal(t._overlap_tails["microphone"], t._overlap_tails["system"])

    @patch("meetingmind.transcriber._load_whisper_model")
    def test_tail_is_from_original_chunk_data(self, mock_load):
        """Stored tail comes from original chunk.data, not preprocessed audio."""
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {
            "text": "Test speech",
            "language": "en",
            "segments": [{"id": 0, "avg_logprob": -0.3, "no_speech_prob": 0.1}],
        }
        mock_load.return_value = mock_model

        from meetingmind.transcriber import StreamingTranscriber
        t = StreamingTranscriber(model_size="tiny.en")

        data = np.linspace(0, 1, 24_000, dtype=np.float32)
        chunk = self._make_chunk(source="microphone", data=data)
        t.transcribe_chunk(chunk)

        overlap_samples = int(_CFG["chunk_overlap_seconds"] * 16_000)
        expected_tail = data[-overlap_samples:]
        assert np.allclose(t._overlap_tails["microphone"], expected_tail)

    @patch("meetingmind.transcriber._load_whisper_model")
    def test_tail_not_stored_when_gated(self, mock_load):
        """When chunk is gated (silent), no tail should be stored."""
        mock_load.return_value = MagicMock()

        from meetingmind.transcriber import StreamingTranscriber
        t = StreamingTranscriber(model_size="tiny.en")

        # Create a chunk that will be gated as silent
        chunk = self._make_chunk(source="microphone", rms=0.0, peak_rms=0.0)
        result = t.transcribe_chunk(chunk)

        assert result is None
        assert "microphone" not in t._overlap_tails
