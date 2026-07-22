# DPMTFormer

## Environment

Install the dependencies from the project root:

```bash
pip install -r requirements.txt
```

## Prepare the Dataset

The default dataset directories are `data/train`, `data/eval`, and `data/test`. Run the following command to inject anomalies into a copy of `data/test` and create `data_anom/test`:

```bash
python prepare_test_anomaly_dataset.py --overwrite
```

## Train

The following command trains on the default dataset and saves model checkpoints and logs to `ckpt/experiment_01`:

```bash
python train.py --batch_size 256 --num_workers 4 --ckpt_every 1000 --fast_eval_every 500 --full_eval_every 1000 --recon_k 4
```

## Test

Replace the checkpoint path with the directory produced by training. The threshold below is selected on the validation set:

```bash
python test.py --ckpt ckpt/experiment_01/best_step_xxxxxx --batch_size 256 --stride 24 --threshold 1.92604
```
