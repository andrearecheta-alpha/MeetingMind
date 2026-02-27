"""
tests/test_device_detector.py
------------------------------
Unit tests for the device_detector module.

All PyAudio calls are mocked so the tests run without real audio hardware.
Each test scenario mirrors a common real-world setup.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from meetingmind.device_detector import (
    DetectedDevice,
    DetectionResult,
    DeviceCategory,
    detect_devices,
)


# ---------------------------------------------------------------------------
# Helper — build a dict that matches PyAudio's get_device_info_by_index output
# ---------------------------------------------------------------------------

def make_device(index: int, name: str, in_ch: int, sample_rate: float = 44100.0, host_api: int = 0) -> dict:
    return {
        "index":              index,
        "name":               name,
        "maxInputChannels":   in_ch,
        "maxOutputChannels":  2 if in_ch == 0 else 0,
        "defaultSampleRate":  sample_rate,
        "hostApi":            host_api,
    }


def _build_mock_pa(
    devices: list[dict],
    host_apis: dict[int, str] | None = None,
) -> MagicMock:
    """Return a MagicMock that behaves like pyaudio.PyAudio() for our detector.

    host_apis maps hostApi integer → host-API name string.
    Defaults to {0: "Windows WASAPI"} so existing single-API tests are unchanged.
    """
    pa = MagicMock()
    pa.get_device_count.return_value = len(devices)
    pa.get_device_info_by_index.side_effect = lambda i: devices[i]
    _apis = host_apis if host_apis is not None else {0: "Windows WASAPI"}
    pa.get_host_api_info_by_index.side_effect = (
        lambda i: {"name": _apis.get(i, "Windows WASAPI")}
    )
    return pa


# ---------------------------------------------------------------------------
# Scenario 1 — Typical Windows laptop
# ---------------------------------------------------------------------------

class TestWindowsLaptop:
    """Intel array mic + Camo virtual cam + Stereo Mix loopback."""

    DEVICES = [
        make_device(0, "Microphone Array (Intel® Smart Sound Technology)", 2),
        make_device(1, "Camo Microphone",                                   1),
        make_device(2, "Stereo Mix (Realtek HD Audio)",                     2),
        make_device(3, "Speakers (Realtek HD Audio)",                       0),  # output only
    ]

    def _run(self) -> DetectionResult:
        pa_mock = _build_mock_pa(self.DEVICES)
        with patch("pyaudio.PyAudio", return_value=pa_mock):
            return detect_devices()

    def test_recommended_mic_is_intel_array(self):
        result = self._run()
        assert result.recommended_mic is not None
        assert "Intel" in result.recommended_mic["name"] or "Array" in result.recommended_mic["name"]

    def test_recommended_mic_score_is_10(self):
        result = self._run()
        assert result.recommended_mic["score"] == 10

    def test_recommended_system_is_stereo_mix(self):
        result = self._run()
        assert result.recommended_system is not None
        assert "Stereo Mix" in result.recommended_system["name"]

    def test_camo_generates_warning(self):
        result = self._run()
        assert any("Camo" in w or "camo" in w.lower() for w in result.warnings)

    def test_output_only_device_excluded(self):
        result = self._run()
        indices = [d.index for d in result.all_devices]
        assert 3 not in indices   # Speakers is output-only


# ---------------------------------------------------------------------------
# Scenario 2 — Mac developer setup
# ---------------------------------------------------------------------------

class TestMacSetup:
    """Built-in Microphone + BlackHole (system audio) + Soundflower (virtual)."""

    DEVICES = [
        make_device(0, "Built-in Microphone", 1),
        make_device(1, "BlackHole 2ch",       2),
        make_device(2, "Soundflower (2ch)",   2),
    ]

    def _run(self) -> DetectionResult:
        pa_mock = _build_mock_pa(self.DEVICES)
        with patch("pyaudio.PyAudio", return_value=pa_mock):
            return detect_devices()

    def test_recommended_mic_is_builtin(self):
        result = self._run()
        assert result.recommended_mic is not None
        assert "Built-in" in result.recommended_mic["name"]

    def test_recommended_mic_score_is_10(self):
        result = self._run()
        assert result.recommended_mic["score"] == 10

    def test_recommended_system_is_blackhole(self):
        result = self._run()
        assert result.recommended_system is not None
        assert "BlackHole" in result.recommended_system["name"]

    def test_soundflower_generates_warning(self):
        result = self._run()
        assert any("Soundflower" in w or "soundflower" in w.lower() for w in result.warnings)

    def test_soundflower_category_is_virtual(self):
        result = self._run()
        virtual = [d for d in result.all_devices if d.category == DeviceCategory.VIRTUAL_MIC]
        names = [d.name for d in virtual]
        assert any("Soundflower" in n for n in names)


# ---------------------------------------------------------------------------
# Scenario 3 — No Stereo Mix available
# ---------------------------------------------------------------------------

class TestNoStereoMix:
    """Intel Array mic + Camo only — no system loopback."""

    DEVICES = [
        make_device(0, "Microphone Array (Intel® Smart Sound Technology)", 2),
        make_device(1, "Camo Microphone",                                   1),
    ]

    def _run(self) -> DetectionResult:
        pa_mock = _build_mock_pa(self.DEVICES)
        with patch("pyaudio.PyAudio", return_value=pa_mock):
            return detect_devices()

    def test_recommended_mic_is_intel(self):
        result = self._run()
        assert result.recommended_mic is not None
        assert "Intel" in result.recommended_mic["name"] or "Array" in result.recommended_mic["name"]

    def test_recommended_system_is_none(self):
        result = self._run()
        assert result.recommended_system is None

    def test_camo_warning_present(self):
        result = self._run()
        assert len(result.warnings) >= 1

    def test_two_devices_found(self):
        result = self._run()
        assert len(result.all_devices) == 2


# ---------------------------------------------------------------------------
# Scenario 4 — Only virtual mics available
# ---------------------------------------------------------------------------

class TestOnlyVirtualMics:
    """OBS Virtual Mic + Camo — no real mic detected."""

    DEVICES = [
        make_device(0, "OBS Virtual Microphone", 1),
        make_device(1, "Camo Microphone",         1),
    ]

    def _run(self) -> DetectionResult:
        pa_mock = _build_mock_pa(self.DEVICES)
        with patch("pyaudio.PyAudio", return_value=pa_mock):
            return detect_devices()

    def test_recommended_mic_is_not_none(self):
        """Falls back to best virtual mic rather than returning None."""
        result = self._run()
        assert result.recommended_mic is not None

    def test_recommended_mic_score_is_2(self):
        """Virtual mics get score 2."""
        result = self._run()
        assert result.recommended_mic["score"] == 2

    def test_two_warnings_generated(self):
        result = self._run()
        assert len(result.warnings) == 2

    def test_recommended_system_is_none(self):
        result = self._run()
        assert result.recommended_system is None

    def test_both_devices_categorised_virtual(self):
        result = self._run()
        categories = {d.category for d in result.all_devices}
        assert categories == {DeviceCategory.VIRTUAL_MIC}


# ---------------------------------------------------------------------------
# Scenario 5 — No mic at all (output-only system)
# ---------------------------------------------------------------------------

class TestNoMicAtAll:
    """Only output devices — headphones and speakers."""

    DEVICES = [
        make_device(0, "Headphones (Realtek HD Audio)", 0),
        make_device(1, "Speakers (Realtek HD Audio)",   0),
    ]

    def _run(self) -> DetectionResult:
        pa_mock = _build_mock_pa(self.DEVICES)
        with patch("pyaudio.PyAudio", return_value=pa_mock):
            return detect_devices()

    def test_does_not_crash(self):
        """Should return gracefully, not raise."""
        result = self._run()
        assert isinstance(result, DetectionResult)

    def test_recommended_mic_is_none(self):
        result = self._run()
        assert result.recommended_mic is None

    def test_recommended_system_is_none(self):
        result = self._run()
        assert result.recommended_system is None

    def test_all_devices_empty(self):
        result = self._run()
        assert result.all_devices == []

    def test_no_warnings(self):
        result = self._run()
        assert result.warnings == []


# ---------------------------------------------------------------------------
# Scenario 6 — Deduplication across host APIs
# ---------------------------------------------------------------------------

class TestDeduplication:
    """
    Intel mic + Camo virtual each appear across MME, DirectSound, and WASAPI.
    After dedup, each physical device should collapse to a single WASAPI entry.
    """

    # Four host API indices used in this scenario
    HOST_APIS = {
        0: "MME",
        1: "Windows DirectSound",
        2: "Windows WASAPI",
        3: "Windows WDM-KS",
    }

    # Three host-API copies of Intel mic, three of Camo, one Stereo Mix
    DEVICES = [
        make_device(0, "Intel Smart Sound Mic",  2, host_api=0),  # MME
        make_device(1, "Intel Smart Sound Mic",  2, host_api=1),  # DirectSound
        make_device(2, "Intel Smart Sound Mic",  2, host_api=2),  # WASAPI  ← winner
        make_device(3, "Microphone (Camo)",       1, host_api=0),  # MME
        make_device(4, "Microphone (Camo)",       1, host_api=1),  # DirectSound
        make_device(5, "Microphone (Camo)",       1, host_api=2),  # WASAPI  ← winner
        make_device(6, "Stereo Mix",              2, host_api=3),  # WDM-KS
    ]

    def _run(self) -> DetectionResult:
        pa_mock = _build_mock_pa(self.DEVICES, self.HOST_APIS)
        with patch("pyaudio.PyAudio", return_value=pa_mock):
            return detect_devices()

    def test_intel_deduplicated_to_one_entry(self):
        result = self._run()
        intel = [d for d in result.all_devices if "Intel" in d.name]
        assert len(intel) == 1

    def test_intel_winner_is_wasapi(self):
        result = self._run()
        intel = next(d for d in result.all_devices if "Intel" in d.name)
        assert "WASAPI" in intel.host_api

    def test_intel_duplicate_count_is_3(self):
        result = self._run()
        intel = next(d for d in result.all_devices if "Intel" in d.name)
        assert intel.duplicate_count == 3

    def test_camo_deduplicated_to_one_entry(self):
        result = self._run()
        camo = [d for d in result.all_devices if "Camo" in d.name]
        assert len(camo) == 1

    def test_camo_winner_is_wasapi(self):
        result = self._run()
        camo = next(d for d in result.all_devices if "Camo" in d.name)
        assert "WASAPI" in camo.host_api

    def test_camo_duplicate_count_is_3(self):
        result = self._run()
        camo = next(d for d in result.all_devices if "Camo" in d.name)
        assert camo.duplicate_count == 3

    def test_exactly_one_warning_for_camo(self):
        """Three Camo instances → one merged warning, not three."""
        result = self._run()
        assert len(result.warnings) == 1

    def test_warning_mentions_instance_count(self):
        result = self._run()
        assert "3" in result.warnings[0]

    def test_total_three_devices_after_dedup(self):
        """7 raw entries collapse to 3 (Intel, Camo, Stereo Mix)."""
        result = self._run()
        assert len(result.all_devices) == 3

    def test_recommended_mic_is_intel_wasapi(self):
        result = self._run()
        assert result.recommended_mic is not None
        assert "Intel" in result.recommended_mic["name"]
        assert "WASAPI" in result.recommended_mic["host_api"]
