from copy import deepcopy
from pathlib import Path


MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
SEED = 42

DATASET_NAME = "Den4ikAI/ru_sberquad_long_answers"
DATA_DIR = Path("data")
RESULTS_DIR = Path("results")

NUM_TRAIN = 10_000
NUM_VALIDATION = 1_000
NUM_TEST = 1_000

MAX_LENGTH = 1_024

SYSTEM_PROMPT = (
    "Ответь на вопрос, используя только приведённый контекст. "
    "Дай полный ответ без рассуждений, приветствий и дополнительной информации."
)

COMMON_TRAINING = {
    "num_train_epochs": 2,
    "max_steps": 900,
    "per_device_train_batch_size": 8,
    "per_device_eval_batch_size": 4,
    "gradient_accumulation_steps": 2,
    "learning_rate": 2e-4,
    "warmup_ratio": 0.03,
    "lr_scheduler_type": "cosine",
    "logging_steps": 20,
}

EXPERIMENTS = {
    "full_ft": {
        "method": "full_ft",
        "learning_rate": 1e-5,
    },
    "lora_baseline": {
        "method": "lora",
        "r": 16,
        "alpha": 32,
        "dropout": 0.05,
        "target_modules": ["q_proj", "v_proj"],
    },
    "lora_r8": {
        "method": "lora",
        "r": 8,
        "alpha": 32,
        "dropout": 0.05,
        "target_modules": ["q_proj", "v_proj"],
    },
    "lora_alpha16": {
        "method": "lora",
        "r": 16,
        "alpha": 16,
        "dropout": 0.05,
        "target_modules": ["q_proj", "v_proj"],
    },
    "lora_dropout10": {
        "method": "lora",
        "r": 16,
        "alpha": 32,
        "dropout": 0.10,
        "target_modules": ["q_proj", "v_proj"],
    },
    "lora_attention_all": {
        "method": "lora",
        "r": 16,
        "alpha": 32,
        "dropout": 0.05,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    },
}


def get_experiment(name):
    if name not in EXPERIMENTS:
        available = ", ".join(EXPERIMENTS)
        raise ValueError(f"Unknown experiment: {name}. Available: {available}")
    config = deepcopy(COMMON_TRAINING)
    config.update(deepcopy(EXPERIMENTS[name]))
    config["name"] = name
    return config
