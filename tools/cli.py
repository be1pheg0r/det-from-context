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

import argparse

# TODO(чел.5): остальные подкоманды и --set key=value для ablation-прогонов.
# smoke: 10 шагов обучения + 1 eval, должен падать быстро и громко.
# grid: прогоняет все ветки и печатает одну таблицу ветка × режим отказа —
#       это и есть DoD проекта.


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    smoke = subparsers.add_parser("smoke")
    smoke.add_argument("config", nargs="?", default=None)
    for name in ("train", "eval", "showbatch", "profile", "grid"):
        subparsers.add_parser(name)
    args = parser.parse_args()

    if args.command == "smoke":
        from smoke_memot import run

        run(args.config)
        return
    raise NotImplementedError(
        f"подкоманда {args.command!r} ещё не реализована (Человек 5)"
    )


if __name__ == "__main__":
    main()
