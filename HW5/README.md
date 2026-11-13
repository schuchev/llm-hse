# HW5: RL для вызова функций

Дообучение `Qwen/Qwen3.5-2B` алгоритмом GRPO на `Team-ACE/ToolACE`. Итоговая оценка — supplied BFCL из 350 примеров. Использованы только RL и ToolACE.

Seed: `42`. Версия модели зафиксирована в `train.py`; версии библиотек и параметры запусков сохраняются в `results/*/config.json` и `summary.json`.

## Команды

```bash
# подготовка ToolACE, EDA и dialog-level split
python3 prepare_data.py

# baseline
python3 evaluate.py Qwen/Qwen3.5-2B --output results/base

# RL-запуски
python3 train.py --experiment rl_exact
python3 train.py --experiment rl_shaped
python3 train.py --experiment rl_shaped_g8
python3 train.py --experiment rl_shaped_temp1
python3 train.py --experiment rl_shaped_temp1_multicall

# BFCL-оценка модели
python3 evaluate.py results/rl_shaped_temp1/model \
  --output results/rl_shaped_temp1/bfcl

# графики
python3 plot_results.py
```

## Содержимое

- `prepare_data.py` - подготовка данных;
- `rewards.py` - функция награды;
- `train.py` - GRPO-конфигурации и экспорт модели;
- `evaluate.py` - BFCL-оценка;
- `results/` - метрики, predictions и графики;
- `report.pdf` - итоговый отчёт.

В Git не хранятся данные, веса моделей и checkpoints. Лучший запуск: `rl_shaped_temp1`, `overall = 0.731`.
