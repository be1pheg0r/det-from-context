# Synthetic regression dataset

- `provider.py` — dataset protocol, deterministic generator and DataLoader build.
- `config.yaml` — data distribution and train/validation split parameters.
- `artifacts/` — generated `train.pt` and `validation.pt` for the latest run.

The folder is loaded through `data.component_path`; it is not coupled to a
particular model or experiment.
