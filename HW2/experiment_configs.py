BATCH_PRESETS = {
    "b32_ga2": {"per_device_train_batch_size": 32, "gradient_accumulation_steps": 2},
    "b32_ga1": {"per_device_train_batch_size": 32, "gradient_accumulation_steps": 1},
    "b64_ga1": {"per_device_train_batch_size": 64, "gradient_accumulation_steps": 1},
    "b128_ga1": {"per_device_train_batch_size": 128, "gradient_accumulation_steps": 1},
    "b16_ga4": {"per_device_train_batch_size": 16, "gradient_accumulation_steps": 4},
    "b32_ga4": {"per_device_train_batch_size": 32, "gradient_accumulation_steps": 4},
    "b48_ga1": {"per_device_train_batch_size": 48, "gradient_accumulation_steps": 1},
}


OPTIMIZATION_PRESETS = {
    "plain": {},
    "flash2": {"attention_implementation": "flash_attention_2"},
    "compile": {"training_overrides": {"torch_compile": True}},
    "liger": {"training_overrides": {"use_liger_kernel": True}},
    "checkpoint": {"training_overrides": {"gradient_checkpointing": True}},
    "offload": {"training_overrides": {"activation_offloading": True}},
    "flash2_padding_free": {
        "attention_implementation": "flash_attention_2",
        "training_overrides": {"padding_free": True},
    },
    "flash2_packing": {
        "attention_implementation": "flash_attention_2",
        "training_overrides": {"packing": True},
    },
    "flash2_liger": {
        "attention_implementation": "flash_attention_2",
        "training_overrides": {"use_liger_kernel": True},
    },
    "flash2_padding_free_liger": {
        "attention_implementation": "flash_attention_2",
        "training_overrides": {"padding_free": True, "use_liger_kernel": True},
    },
    "flash2_compile": {
        "attention_implementation": "flash_attention_2",
        "training_overrides": {"torch_compile": True},
    },
    "flash2_padding_free_compile": {
        "attention_implementation": "flash_attention_2",
        "training_overrides": {"padding_free": True, "torch_compile": True},
    },
}
