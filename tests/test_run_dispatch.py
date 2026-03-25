"""Routing from Hydra ``cfg.name`` to experiment classes."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra


def test_quick_synthetic_ope_dispatches_to_synthetic_ope_experiment() -> None:
    GlobalHydra.instance().clear()
    conf_dir = Path(__file__).resolve().parents[1] / "src" / "opfv" / "conf"
    with initialize_config_dir(version_base=None, config_dir=str(conf_dir)):
        cfg = compose(
            config_name="config",
            overrides=["experiment=quick_synthetic_ope"],
        )
    assert cfg.name == "quick_synthetic_ope"

    with patch("opfv.run.SyntheticOPEExperiment", autospec=True) as mock_cls:
        mock_inst = MagicMock()
        mock_cls.return_value = mock_inst
        from opfv.run import _dispatch

        _dispatch(cfg)

    mock_cls.assert_called_once_with(cfg)
    mock_inst.run.assert_called_once()
