from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from opfv.experiments.synthetic_ope import SyntheticOPEExperiment
from opfv.logging_config import configure_logging


@pytest.mark.slow
def test_quick_synthetic_ope_smoke() -> None:
    GlobalHydra.instance().clear()
    conf_dir = Path(__file__).resolve().parents[1] / "src" / "opfv" / "conf"
    with initialize_config_dir(version_base=None, config_dir=str(conf_dir)):
        cfg = compose(
            config_name="config",
            overrides=[
                "domain=synthetic_ope_base",
                "experiment=quick_synthetic_ope",
            ],
        )
    configure_logging("WARNING")
    SyntheticOPEExperiment(cfg).run()
    out = Path(cfg.paths.output_subdir) / cfg.name / "df" / "result_df.csv"
    assert out.is_file()
