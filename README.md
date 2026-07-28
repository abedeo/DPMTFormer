# DPMTFormer

Reference implementation for **"Dual-Scale Multi-Task Anomaly Detection for Water-Level Time Series"**.

DPMTFormer is a dual-scale patch-based multi-task Transformer for water-level time-series anomaly detection. It jointly learns (1) 3-hour future forecasting from 1-hour patches, (2) conditional denoising reconstruction from 30-minute patches, and (3) 30-minute patch-level anomaly discrimination. Water level is the primary input; precipitation and periodic temporal features are auxiliary inputs.

## Repository contents

| Path | Purpose |
| --- | --- |
| `model.py` | DPMTFormer architecture and its forecasting, reconstruction, and discrimination branches. |
| `train.py` | Training, validation, checkpointing, and early stopping. |
| `test.py` | Patch-level test evaluation and JSON metric export. |
| `dataset.py`, `loader.py` | Data loading, continuous-segment handling, normalization, windowing, and batching. |
| `anomaly_injector.py` | Controlled injection of spikes, local oscillations, and ramp/step drifts. |
| `prepare_test_anomaly_dataset.py` | Creates a labeled test copy with deterministic injected anomalies. |
| `generate_synthetic_demo_data.py` | Maintainer utility for generating synthetic data profiles from authorized source data. |
| `loss.py`, `eval.py` | Multi-task losses and validation metrics. |
| `requirements.txt` | Python package requirements. |

## Data availability and confidentiality

The water-level records used in this study were obtained from the Zhejiang Provincial Hydrological Telemetry System, an internal database maintained by the Zhejiang Provincial Hydrology Bureau (official website: http://www.zjsw.cn/).

The raw records are not externally accessible because of data-management restrictions and security-sensitive operational information. To support code execution and workflow verification, a synthetic demonstration dataset generated from coarse statistical characteristics of the original records is publicly available at https://doi.org/10.5281/zenodo.21465817.


## Requirements

The experiments reported in the manuscript used Windows 11, Python 3.11.14, PyTorch 2.10.0, 32 GB RAM, and an NVIDIA GeForce RTX 5070 Ti GPU.

Install the dependencies in a virtual environment from the project root:

```test
pip install -r requirements.txt
```

## Data layout

Training and validation data must follow this layout:

```text
data/
  train/
    <station_id>/
      clean.csv
  eval/
    <station_id>/
      clean.csv
  test/
    <station_id>/
      clean.csv
```

## Usage

Prepare the labeled test data by injecting controlled anomalies into a copy of `data/test`:

```bash
python prepare_test_anomaly_dataset.py --overwrite
```

Train the model:

```bash
python train.py --batch_size 256 --num_workers 4 --ckpt_every 1000 --fast_eval_every 500 --full_eval_every 1000 --recon_k 4
```

Evaluate a trained checkpoint. Select the threshold on the validation set and keep it fixed for testing:

```bash
python test.py --ckpt ckpt/experiment_01/best_step_xxxxxx --batch_size 256 --stride 24 --threshold <validation_threshold>
```

## Methodology

DPMTFormer performs patch-level anomaly detection through three jointly optimized tasks: future forecasting, conditional denoising reconstruction, and anomaly discrimination. Each input window contains 12 hours of historical observations and 3 hours of future observations sampled at 5-minute intervals. The forecasting branch models long-term water-level dynamics using 1-hour patches, whereas the reconstruction and discrimination branches use 30-minute patches to preserve local variations.

Water level is used as the primary input, with precipitation and periodic temporal features as auxiliary inputs. To account for differences among monitoring stations, global normalization and instance normalization are applied. Anomaly labels are defined at the 30-minute patch level: a patch is considered anomalous when it contains anomalous observations. Training and evaluation samples include threshold-labeled abnormal segments and controlled injections of spikes, local oscillations, and ramp/step drifts.

## Citation

If you use this code, please cite:

```text
Chen, Z., Wang, S., Yuan, Z., and Kong, X. Dual-Scale Multi-Task Anomaly
Detection for Water-Level Time Series. Manuscript submitted to PeerJ Computer
Science.
```

Update this entry with the final journal citation and DOI after publication.

## License and contributions

This code is released under the MIT License. See the `LICENSE` file for details.
