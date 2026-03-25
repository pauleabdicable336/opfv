from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra


def test_compose_default_config() -> None:
    GlobalHydra.instance().clear()
    conf_dir = Path(__file__).resolve().parents[1] / "src" / "opfv" / "conf"
    with initialize_config_dir(version_base=None, config_dir=str(conf_dir)):
        cfg = compose(config_name="config")
    assert cfg.name is not None
    assert cfg.sweep is not None
    assert cfg.synthetic.n_actions == 10
