import json

import pytest

from deploy_ultralytics import target

SOURCE_AND_TOOLCHAIN = {
    "source": {"store_url": "https://example.invalid", "version": "1.0.0", "vendor": "X"},
    "toolchain": {"image": "x-toolchain:1.0.0", "input": "model", "converted_model_extension": "bin"},
}


@pytest.fixture
def write_target(tmp_path, monkeypatch):
    monkeypatch.setattr(target, "TARGETS_DIR", tmp_path)

    def write(**fields):
        (tmp_path / "x").mkdir()
        (tmp_path / "x" / "target.json").write_text(json.dumps({**SOURCE_AND_TOOLCHAIN, **fields}))
        return target.load_target("x")

    return write


def test_a_target_that_only_names_its_toolchain_takes_the_exported_models_own_input(write_target):
    config = write_target()

    assert config.element_type == "float32"
    assert config.layout == "NCHW"
    assert config.mean == (0, 0, 0)
    assert config.std == (255, 255, 255)
    assert config.iou_threshold == 0.9


def test_a_target_overrides_the_defaults_it_states(write_target):
    config = write_target(
        element_type="uint8",
        layout="NHWC",
        normalization={"mean": [1, 2, 3], "std": [4, 5, 6]},
        iou_threshold=0.5,
    )

    assert config.element_type == "uint8"
    assert config.layout == "NHWC"
    assert config.mean == (1, 2, 3)
    assert config.std == (4, 5, 6)
    assert config.iou_threshold == 0.5


def test_a_target_states_whether_its_toolchain_takes_the_exported_model_or_a_bundle(write_target):
    config = write_target(toolchain={**SOURCE_AND_TOOLCHAIN["toolchain"], "input": "bundle"})

    assert config.toolchain_input == "bundle"


def test_a_target_with_another_toolchain_input_is_refused(write_target):
    with pytest.raises(target.TargetError, match="'model' or 'bundle'"):
        write_target(toolchain={**SOURCE_AND_TOOLCHAIN["toolchain"], "input": "tarball"})
