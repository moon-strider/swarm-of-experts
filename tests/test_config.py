import pytest
from src.config.swarm_configs import (
    SwarmConfig, GeneratorConfig, get_swarm_config,
    get_all_swarm_configs, SWARM_CONFIGS,
)
from src.config.settings import Settings


class TestSwarmConfig:
    def test_has_merger_true(self, multi_generator_config):
        assert multi_generator_config.has_merger is True

    def test_has_merger_false(self, single_generator_config):
        assert single_generator_config.has_merger is False

    def test_has_taskmaster_true(self, taskmaster_config):
        assert taskmaster_config.has_taskmaster is True

    def test_has_taskmaster_false(self, single_generator_config):
        assert single_generator_config.has_taskmaster is False

    def test_generator_config_defaults(self):
        gen = GeneratorConfig(provider="openai", model="gpt-4.1")
        assert gen.temperature == 0.7

    def test_swarm_config_defaults(self):
        config = SwarmConfig(name="test")
        assert config.generators == []
        assert config.merger is None
        assert config.taskmaster is None
        assert config.max_context_tokens == 128000


class TestGetSwarmConfig:
    def test_valid_config(self):
        config = get_swarm_config("basic")
        assert config.name == "basic"
        assert len(config.generators) == 1

    def test_invalid_config_raises(self):
        with pytest.raises(ValueError, match="Unknown swarm config"):
            get_swarm_config("nonexistent-config")

    def test_all_predefined_configs_valid(self):
        for name in SWARM_CONFIGS:
            config = get_swarm_config(name)
            assert config.name == name
            assert len(config.generators) > 0

    def test_get_all_returns_dict(self):
        all_configs = get_all_swarm_configs()
        assert isinstance(all_configs, dict)
        assert "basic" in all_configs
        assert "groq-swarm" in all_configs

    def test_groq_swarm_has_3_generators(self):
        config = get_swarm_config("groq-swarm")
        assert len(config.generators) == 3
        assert config.has_merger is True

    def test_groq_taskmaster_config(self):
        config = get_swarm_config("groq-taskmaster")
        assert config.has_taskmaster is True
        assert config.has_merger is True
        assert len(config.generators) == 3


class TestSettings:
    def test_no_keys_validation_fails(self):
        s = Settings()
        is_valid, msg = s.validate()
        assert is_valid is False
        assert "No API keys" in msg

    def test_with_keys_validation_passes(self, fake_env):
        s = Settings()
        is_valid, msg = s.validate()
        assert is_valid is True
        assert msg == ""

    def test_get_api_key_valid_provider(self, fake_env):
        s = Settings()
        assert s.get_api_key_for_provider("openai") == "sk-test-openai-key"
        assert s.get_api_key_for_provider("groq") == "gsk-test-groq-key"

    def test_get_api_key_unknown_provider(self, fake_env):
        s = Settings()
        with pytest.raises(ValueError, match="Unknown provider"):
            s.get_api_key_for_provider("unknown-provider")

    def test_get_api_key_missing_key(self):
        s = Settings()
        with pytest.raises(ValueError, match="API key not found"):
            s.get_api_key_for_provider("openai")

    def test_env_overrides(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test")
        monkeypatch.setenv("DEFAULT_MODEL", "custom-model")
        monkeypatch.setenv("TEMPERATURE", "0.9")
        monkeypatch.setenv("MAX_TOKENS", "4096")
        monkeypatch.setenv("SERVER_HOST", "127.0.0.1")
        monkeypatch.setenv("SERVER_PORT", "9000")

        s = Settings()
        assert s.default_model == "custom-model"
        assert s.temperature == 0.9
        assert s.max_tokens == 4096
        assert s.server_host == "127.0.0.1"
        assert s.server_port == 9000

    def test_case_insensitive_provider(self, fake_env):
        s = Settings()
        assert s.get_api_key_for_provider("OpenAI") == "sk-test-openai-key"
        assert s.get_api_key_for_provider("GROQ") == "gsk-test-groq-key"

    def test_get_swarm_config_from_settings(self, fake_env):
        s = Settings()
        s.swarm_config_name = "basic"
        config = s.get_swarm_config()
        assert config.name == "basic"
