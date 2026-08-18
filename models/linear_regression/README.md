# Linear regression model

- `provider.py` — model protocol and `nn.Module` implementation.
- `config.yaml` — architecture parameters.
- `artifacts/` — trained checkpoints produced by experiments.

The folder is loaded through `detector.component_path`; it is independent of
the dataset and training loop.
