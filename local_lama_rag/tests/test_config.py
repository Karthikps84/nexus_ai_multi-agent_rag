import config


def test_model_defaults_are_set():
    assert config.NVIDIA_BASE_URL == "https://integrate.api.nvidia.com/v1"
    for name in ("LLM_MODEL", "FAST_LLM_MODEL", "VISION_MODEL", "EMBEDDING_MODEL"):
        value = getattr(config, name)
        assert isinstance(value, str) and value, f"{name} must be a non-empty string"
