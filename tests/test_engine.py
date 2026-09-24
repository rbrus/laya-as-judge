"""Unit tests for engine abstractions and calibration math."""

import numpy as np
import pytest
from laya_as_judge.engine import (
    BaseDecisionEngine,
    EmulatorBackend,
    confidence_from_probs,
    get_engine,
    render_options,
    validate_question_spec,
)


def test_confidence_from_probs():
    # Uniform distribution of 4 options -> maximum entropy -> 0.0 confidence
    uniform = np.array([0.25, 0.25, 0.25, 0.25])
    conf_uniform = confidence_from_probs(uniform, 4)
    assert conf_uniform == pytest.approx(0.0, abs=1e-3)

    # Completely certain distribution -> zero entropy -> 1.0 confidence
    one_hot = np.array([1.0, 0.0, 0.0, 0.0])
    conf_certain = confidence_from_probs(one_hot, 4)
    assert conf_certain == pytest.approx(1.0, abs=1e-3)

    # Boundary: single option -> confidence 1.0
    assert confidence_from_probs(np.array([1.0]), 1) == 1.0


def test_render_options():
    choice_q = {
        "type": "choice",
        "instructions": "Test",
        "criteria": {"a": "option A", "b": "option B"},
    }
    opts = render_options(choice_q)
    assert opts == ["a: option A", "b: option B"]

    score_q = {
        "type": "score",
        "instructions": "Test",
        "criteria": ["low", "med", "high"],
    }
    opts_score = render_options(score_q)
    assert opts_score == ["level 0: low", "level 1: med", "level 2: high"]

    noul_q = {
        "type": "noul",
        "instructions": "Test",
        "criteria": {"false": "no", "true": "yes"},
    }
    opts_noul = render_options(noul_q)
    assert opts_noul == ["false: no", "true: yes"]


def test_validate_question_spec():
    valid = validate_question_spec("test", {
        "type": "score",
        "instructions": "Rate this",
        "criteria": ["bad", "good"],
    })
    assert valid["type"] == "score"

    # Missing instructions
    with pytest.raises(ValueError):
        validate_question_spec("invalid", {"type": "choice", "criteria": ["a", "b"]})

    # Unknown type
    with pytest.raises(ValueError):
        validate_question_spec("invalid", {"type": "unknown", "instructions": "foo"})

    # Invalid score criteria count
    with pytest.raises(ValueError):
        validate_question_spec("invalid", {"type": "score", "instructions": "foo", "criteria": ["one"]})


def test_emulator_backend():
    engine = EmulatorBackend()
    questions = {
        "is_safe": {
            "type": "noul",
            "instructions": "Is `text` safe?",
        },
        "quality": {
            "type": "score",
            "instructions": "Rate quality",
            "criteria": ["poor", "acceptable", "great"],
        },
    }
    res = engine.predict("A completely normal safe query.", questions)
    assert "answers" in res
    assert "is_safe" in res["answers"]
    assert "quality" in res["answers"]
    assert res["usage"]["output_tokens"] == 0
    assert res["answers"]["is_safe"]["noul"] >= 0.5


def test_factory_get_engine():
    engine = get_engine("emulator")
    assert isinstance(engine, EmulatorBackend)

    # auto backend falls back safely when mlx/torch not on system
    auto_engine = get_engine("auto")
    assert isinstance(auto_engine, BaseDecisionEngine)


def test_auto_backend_falls_back_to_emulator_without_mlx():
    # On non-Apple-Silicon hosts (CI), "auto" must resolve to the emulator and must
    # never silently pick the incomplete TorchBackend.
    import platform

    if platform.system() == "Darwin" and platform.machine() == "arm64":
        pytest.skip("auto may select MLX on Apple Silicon")
    assert isinstance(get_engine("auto"), EmulatorBackend)
