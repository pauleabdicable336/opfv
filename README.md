<h1 align="center"><b>OPFV</b><br>Future off-policy evaluation & learning under non-stationarity</h1>

<p align="center">
  <img src="https://img.shields.io/badge/python-%3E%3D3.12-blue" alt="Python" />
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-526EAF.svg?logo=opensourceinitiative&logoColor=white" alt="License: MIT" /></a>
  <a href="https://github.com/st-tech/zr-obp"><img src="https://img.shields.io/badge/stack-OBP-111111.svg" alt="Open Bandit Pipeline" /></a>
  <a href="https://dl.acm.org/doi/abs/10.1145/3690624.3709237"><img src="https://img.shields.io/badge/paper-ACM-CC4141.svg" alt="ACM Digital Library" /></a>
  <a href="https://kdd2025.kdd.org/"><img src="https://img.shields.io/badge/KDD-2025-00629B.svg" alt="KDD 2025" /></a>
</p>

Research code for **[Off-Policy Evaluation and Learning for the Future under Non-Stationarity](https://dl.acm.org/doi/abs/10.1145/3690624.3709237)** ([KDD 2025](https://kdd2025.kdd.org/)). **OPFV** (Off-Policy Estimator for the Future Value) targets policy value at a **future** timestamp when the world **drifts**; the paper also develops a policy-gradient extension (**OPFV-PG**) for future-oriented learning.

**Maintainer note.** I’m **[Tatsuhiro Shimizu](https://ss1.xrea.com/tshimizu.s203.xrea.com/works/index.html)**. I was the **primary implementer** of the original experimental stack during collaboration with **Sony Group Corporation**. This repo is **my maintained distribution**: same algorithms and experiments as in the paper, with clearer attribution and onboarding. The **original software is Sony’s** under the MIT License; see [Acknowledgement](#acknowledgement) and [LICENSE](LICENSE).

---

## Documentation

| Resource | What it is |
|----------|------------|
| [`pyproject.toml`](pyproject.toml) / [`uv.lock`](uv.lock) | Dependencies & reproducible resolve (`uv`) |
| [`src/opfv/conf/`](src/opfv/conf/) | **Hydra** defaults: `domain/*.yaml`, `experiment/*.yaml` |
| [`src/opfv/run.py`](src/opfv/run.py) | CLI entry: `python -m opfv.run` |
| [`src/opfv/domain/`](src/opfv/domain/) | Shared estimators, synthetic bandit simulator, OBP-compat helpers |
| [`src/opfv/experiments/`](src/opfv/experiments/) | **All** Hydra experiments (synthetic OPE/OPL and real KuaiRec) |
| [`src/opfv/synthetic_fopl/`](src/opfv/synthetic_fopl/) | Synthetic **OPFV-PG** learners + `SyntheticFOPLSettings` (explicit Hydra → dataclass; no global `conf`) |
| [`src/opfv/kuairec_fopl/`](src/opfv/kuairec_fopl/) | **KuaiRec** F-OPL stack (same package tier as `synthetic_fopl/`); defaults in `conf.py`, overrides from Hydra `kuairec.*` |

---

## Installation & quick start ([uv](https://docs.astral.sh/uv/))

```bash
git clone https://github.com/<your-github>/OPFV.git
cd OPFV
uv sync --all-extras
```

**Run an experiment** (from repo root; Hydra `chdir`s to its output dir):

```bash
uv run python -m opfv.run experiment=synthetic_ope_target_time
# CI / smoke:
uv run python -m opfv.run experiment=quick_synthetic_ope
```

**Override config** (examples):

```bash
uv run python -m opfv.run domain=synthetic_opl_base experiment=synthetic_opl_time_at_eval
uv run python -m opfv.run domain=kuairec_base experiment=real_kuairec_tune_phi kuairec.root=/path/to/KuaiRec/
```

**Tests & lint:**

```bash
uv run pytest
uv run pytest -m slow   # short end-to-end synthetic OPE
uv run ruff check src/opfv tests && uv run mypy src/opfv
```

---

## Docker

```bash
docker build -t opfv:local .
docker run --rm -v "$(pwd)/results:/app/results" opfv:local
```

Override the Hydra experiment by appending arguments after the image name. See [`docker-compose.yml`](docker-compose.yml) for a volume layout example.

---

## Experiment index (synthetic)

| Question | Hydra `experiment=` |
|----------|---------------------|
| OPFV vs target future time | `synthetic_ope_target_time` |
| Time-feature strength (λ) | `synthetic_ope_lambda` |
| Number of time features | `synthetic_ope_num_time_feature` |
| Logged data size | `synthetic_ope_n_trains` |
| OPFV-PG vs evaluation time | `synthetic_opl_time_at_eval` |
| OPFV-PG vs training size | `synthetic_opl_n_trains` |

**Extra F-OPL sweeps:** `synthetic_opl_lambda`, `synthetic_opl_num_time_feature` (see `src/opfv/conf/experiment/`).

---

## Real data (KuaiRec)

Download **[KuaiRec](https://kuairec.com/)** and place files under `KuaiRec/data/`. Run with Hydra (set `kuairec.root` or `KUAIREC_ROOT`) and `experiment=real_kuairec_tune_phi` via `python -m opfv.run`. Optional exploratory notebook: [`notebooks/kuairec_F-OPL_main.ipynb`](notebooks/kuairec_F-OPL_main.ipynb) (imports `opfv.kuairec_fopl`; run Jupyter from repo root with `PYTHONPATH=src` or an editable install).

---

## Repository layout (mental model)

```
src/opfv/           # installable package
├── experiments/    # Hydra runners (synthetic + KuaiRec)
├── synthetic_fopl/
├── kuairec_fopl/
├── domain/, pipelines/, conf/, …
notebooks/          # optional KuaiRec Jupyter workflow (not required for CLI)
```

**Stacking:** [Open Bandit Pipeline (OBP)](https://github.com/st-tech/zr-obp) supplies bandit feedback types and standard OPE building blocks; this codebase adds **time-structured** futures, OPFV weights, and experiment-specific learners. Install with `uv sync` (editable local package).

---

## Results

Figures and tables in the paper map to the experiments above. CSV summaries are written under `results/<experiment_name>/df/` (relative to Hydra’s run directory unless overridden). This README does not duplicate numeric results.

---

## Paper authors

- Tatsuhiro Shimizu (Yale University / Hanjuku-kaso Co., Ltd.)
- Kazuki Kawamura (Sony Group Corporation)
- Takanori Muroi (Sony Group Corporation)
- Yusuke Narita (Hanjuku-kaso Co., Ltd. / Yale University)
- Kei Tateno (Sony Group Corporation)
- Takuma Udagawa (Sony Group Corporation)
- Yuta Saito (Cornell University / Hanjuku-kaso Co., Ltd.)

---

## Acknowledgement

Work originated at **Sony Group Corporation**; Sony **open-sourced the implementation under the MIT License**. I built and integrated the original codebase as **lead engineer** on the project. **Sony** retains copyright in the original software; **modifications** in this fork are noted in [LICENSE](LICENSE) (**Modifications Copyright (c) 2026 Tatsuhiro Shimizu**). Thank you to Sony and my co-authors for the collaboration and for releasing the code.

---

## Related work

- [Open Bandit Pipeline](https://github.com/st-tech/zr-obp) — logging, OPE estimators, datasets.
- [KuaiRec](https://kuairec.com/) — fully observed recommender logs used in §4.

---

## Citation

```bibtex
@inproceedings{shimizu2025offpolicy,
  title     = {Off-Policy Evaluation and Learning for the Future under Non-Stationarity},
  author    = {Shimizu, Tatsuhiro and Kawamura, Kazuki and Muroi, Takanori and Narita, Yusuke and Tateno, Kei and Udagawa, Takuma and Saito, Yuta},
  booktitle = {Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  year      = {2025},
  url       = {https://dl.acm.org/doi/abs/10.1145/3690624.3709237},
  doi       = {10.1145/3690624.3709237}
}
```

---

## License

[MIT License](LICENSE) — **Copyright (c) 2025 Sony Group Corporation**; modifications **Copyright (c) 2026 Tatsuhiro Shimizu**.
