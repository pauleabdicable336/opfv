<h1 align="center"><b>OPFV</b><br>Future off-policy evaluation & learning under non-stationarity</h1>

<p align="center">
  <img src="https://img.shields.io/badge/python-%3E%3D3.8-blue" alt="Python" />
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
| [`requirements.txt`](requirements.txt) | Runtime dependencies (`numpy`, `pandas`, `obp`, `scikit-learn`, …) |
| [`src/synthetic/F-OPE/conf.py`](src/synthetic/F-OPE/conf.py) | F-OPE synthetic hyperparameters & estimator flags |
| [`src/synthetic/F-OPL/conf.py`](src/synthetic/F-OPL/conf.py) | F-OPL synthetic hyperparameters & learner flags |
| [`src/real/F-OPL/conf.py`](src/real/F-OPL/conf.py) | KuaiRec / real-data experiment settings |
| Notebooks under `src/synthetic/{F-OPE,F-OPL}/` | Paper §4 synthetic sweeps |
| [`src/real/F-OPL/main.ipynb`](src/real/F-OPL/main.ipynb), [`main-opfv-tune-phi.ipynb`](src/real/F-OPL/main-opfv-tune-phi.ipynb) | Real-data runs (KuaiRec) |

---

## Installation & quick start

```bash
git clone https://github.com/<your-github>/OPFV.git   # replace with your public URL
cd OPFV
python3.8 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install jupyter
```

**Smoke check:** open any notebook under `src/synthetic/F-OPE/`, run the first few cells that build data and estimators. Imports assume the notebook’s **directory** is on the import path (open Jupyter with that folder as cwd, or `cd` there before `jupyter notebook`).

---

## Notebook workflow (no CLI entrypoint)

Unlike Hydra-driven repos, experiments are **notebook-first**: each directory is a small “lab” with `conf.py` + local modules (`ope.py`, `opl.py`, `estimators_time.py`, …).

**Convention**

1. `cd` to the notebook’s directory (e.g. `src/synthetic/F-OPE`).
2. Start Jupyter from that directory so `import conf` resolves.
3. Toggle flags and lists in `conf.py` for sweeps; notebooks orchestrate loops and plots.

There is **no** bundled `Dockerfile` in this tree; reproduce via the pinned stack in `requirements.txt` and the Python version above.

---

## Experiment index (synthetic)

| Question | Notebook |
|----------|----------|
| OPFV vs target future time | [`src/synthetic/F-OPE/main_target_time.ipynb`](src/synthetic/F-OPE/main_target_time.ipynb) |
| Time-feature strength (λ) | [`src/synthetic/F-OPE/main_lambda.ipynb`](src/synthetic/F-OPE/main_lambda.ipynb) |
| Number of time features | [`src/synthetic/F-OPE/main_num_time_feature.ipynb`](src/synthetic/F-OPE/main_num_time_feature.ipynb) |
| Logged data size | [`src/synthetic/F-OPE/main_n_trains.ipynb`](src/synthetic/F-OPE/main_n_trains.ipynb) |
| OPFV-PG vs evaluation time | [`src/synthetic/F-OPL/main_time_at_evaluation.ipynb`](src/synthetic/F-OPL/main_time_at_evaluation.ipynb) |
| OPFV-PG vs training size | [`src/synthetic/F-OPL/main_n_trains.ipynb`](src/synthetic/F-OPL/main_n_trains.ipynb) |

**Extra F-OPL sweeps:** [`main_lambda_.ipynb`](src/synthetic/F-OPL/main_lambda_.ipynb), [`main_num_time_feature.ipynb`](src/synthetic/F-OPL/main_num_time_feature.ipynb).

---

## Real data (KuaiRec)

Download **[KuaiRec](https://kuairec.com/)** and place files under `KuaiRec/data/` (e.g. `big_matrix.csv`, `small_matrix.csv`, `user_features.csv`, plus the other CSVs from the dataset). Then run [`src/real/F-OPL/main.ipynb`](src/real/F-OPL/main.ipynb) and [`main-opfv-tune-phi.ipynb`](src/real/F-OPL/main-opfv-tune-phi.ipynb), adjusting paths inside the notebooks if your layout differs.

---

## Repository layout (mental model)

```
src/
├── synthetic/
│   ├── F-OPE/     # future OPE, baselines vs OPFV
│   └── F-OPL/     # future OPL / policy-gradient variants
└── real/
    └── F-OPL/     # KuaiRec experiments, φ tuning
```

**Stacking:** [Open Bandit Pipeline (OBP)](https://github.com/st-tech/zr-obp) supplies bandit feedback types and standard OPE building blocks; this codebase adds **time-structured** futures, OPFV weights, and experiment-specific learners. No separate `pip install opfv` package—run from source as above.

---

## Results

Figures and tables in the paper map to the notebooks listed here. This README does not duplicate numeric results; regenerate plots from the corresponding notebooks after installs.

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
