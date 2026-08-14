"""Единая точка входа. Человек 5.

Одна команда вместо пяти скриптов:
    python tools/cli.py train    configs/memory_ema.json
    python tools/cli.py eval     configs/memory_ema.json --ckpt ...
    python tools/cli.py smoke    configs/baseline.json
    python tools/cli.py showbatch configs/baseline.json
    python tools/cli.py profile  configs/memory_stream.json
    python tools/cli.py grid     configs/*.json        # сетка режимов отказа
"""

from __future__ import annotations

# TODO(чел.5): argparse с подкомандами, --set key=value для ablation-прогонов.
# smoke: 10 шагов обучения + 1 eval, должен падать быстро и громко.
# grid: прогоняет все ветки и печатает одну таблицу ветка × режим отказа —
#       это и есть DoD проекта.


def main() -> None:
    raise NotImplementedError("Человек 5")


if __name__ == "__main__":
    main()
