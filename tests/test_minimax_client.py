"""Unit tests for MiniMax client."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from letta.llm_api.minimax_client import MiniMaxClient
from letta.schemas.enums import AgentType
from letta.schemas.llm_config import LLMConfig

MINIMAX_GLOBAL_BASE_URL = "https://api.minimax.io/anthropic"
MINIMAX_CN_BASE_URL = "https://api.minimaxi.com/anthropic"


def _make_llm_config(
    model_name: str = "MiniMax-M3",
    model_endpoint: str = MINIMAX_GLOBAL_BASE_URL,
    *,
    enable_reasoner: bool = True,
    temperature: float = 0.7,
) -> LLMConfig:
    context_window = 1000000 if model_name == "MiniMax-M3" else 204800
    return LLMConfig(
        model=model_name,
        model_endpoint_type="minimax",
        model_endpoint=model_endpoint,
        context_window=context_window,
        enable_reasoner=enable_reasoner,
        temperature=temperature,
    )


class TestMiniMaxClient:
    """Tests for MiniMaxClient."""

    def setup_method(self):
        self.client = MiniMaxClient(put_inner_thoughts_first=True)
        self.llm_config = _make_llm_config(enable_reasoner=False)

    def test_is_reasoning_model_supported_modes(self):
        assert self.client.is_reasoning_model(_make_llm_config("MiniMax-M3")) is True
        assert self.client.is_reasoning_model(_make_llm_config("MiniMax-M2.7")) is True

    def test_requires_auto_tool_choice(self):
        assert self.client.requires_auto_tool_choice(self.llm_config) is False

    def test_supports_structured_output(self):
        assert self.client.supports_structured_output(self.llm_config) is False

    @pytest.mark.parametrize("model_endpoint", [MINIMAX_GLOBAL_BASE_URL, MINIMAX_CN_BASE_URL])
    @patch("letta.llm_api.minimax_client.model_settings")
    def test_get_anthropic_client_with_api_key(self, mock_settings, model_endpoint):
        mock_settings.minimax_api_key = "test-api-key"
        mock_settings.anthropic_max_retries = 7

        with patch("letta.llm_api.minimax_client.anthropic") as mock_anthropic:
            mock_anthropic.Anthropic.return_value = MagicMock()

            self.client.get_byok_overrides = MagicMock(return_value=(None, None, None))
            self.client._get_anthropic_client(_make_llm_config(model_endpoint=model_endpoint), async_client=False)

            mock_anthropic.Anthropic.assert_called_once_with(
                api_key="test-api-key",
                base_url=model_endpoint,
                max_retries=7,
            )

    @pytest.mark.parametrize("model_endpoint", [MINIMAX_GLOBAL_BASE_URL, MINIMAX_CN_BASE_URL])
    @pytest.mark.asyncio
    @patch("letta.llm_api.minimax_client.model_settings")
    async def test_get_anthropic_client_async(self, mock_settings, model_endpoint):
        mock_settings.minimax_api_key = "test-api-key"
        mock_settings.anthropic_max_retries = 7

        with patch("letta.llm_api.minimax_client.anthropic") as mock_anthropic:
            mock_anthropic.AsyncAnthropic.return_value = MagicMock()

            self.client.get_byok_overrides = MagicMock(return_value=(None, None, None))
            await self.client._get_anthropic_client_async(_make_llm_config(model_endpoint=model_endpoint), async_client=True)

            mock_anthropic.AsyncAnthropic.assert_called_once_with(
                api_key="test-api-key",
                base_url=model_endpoint,
                max_retries=7,
            )

    @patch.object(MiniMaxClient.__bases__[0], "build_request_data")
    def test_m3_uses_adaptive_thinking_when_enabled(self, mock_parent):
        mock_parent.return_value = {"temperature": 0.7, "model": "MiniMax-M3"}

        result = self.client.build_request_data(
            agent_type=AgentType.letta_v1_agent,
            messages=[],
            llm_config=_make_llm_config("MiniMax-M3", enable_reasoner=True),
        )

        assert result["thinking"] == {"type": "adaptive"}
        assert result["temperature"] == 1.0

    @patch.object(MiniMaxClient.__bases__[0], "build_request_data")
    def test_m3_can_disable_thinking(self, mock_parent):
        mock_parent.return_value = {"temperature": 0.7, "model": "MiniMax-M3"}

        result = self.client.build_request_data(
            agent_type=AgentType.letta_v1_agent,
            messages=[],
            llm_config=_make_llm_config("MiniMax-M3", enable_reasoner=False),
        )

        assert "thinking" not in result
        assert result["temperature"] == 0.7

    @patch.object(MiniMaxClient.__bases__[0], "build_request_data")
    def test_m27_forces_thinking_when_reasoner_disabled(self, mock_parent):
        mock_parent.return_value = {"temperature": 0.7, "model": "MiniMax-M2.7"}

        result = self.client.build_request_data(
            agent_type=AgentType.letta_v1_agent,
            messages=[],
            llm_config=_make_llm_config("MiniMax-M2.7", enable_reasoner=False),
        )

        assert result["thinking"] == {"type": "enabled", "budget_tokens": 1024}
        assert result["temperature"] == 1.0

    @patch.object(MiniMaxClient.__bases__[0], "build_request_data")
    def test_temperature_zero_clamped(self, mock_parent):
        mock_parent.return_value = {"temperature": 0, "model": "MiniMax-M3"}

        result = self.client.build_request_data(
            agent_type=AgentType.letta_v1_agent,
            messages=[],
            llm_config=_make_llm_config("MiniMax-M3", enable_reasoner=False, temperature=0),
        )

        assert result["temperature"] == 0.01

    @patch.object(MiniMaxClient.__bases__[0], "build_request_data")
    def test_temperature_negative_clamped(self, mock_parent):
        mock_parent.return_value = {"temperature": -0.5, "model": "MiniMax-M3"}

        result = self.client.build_request_data(
            agent_type=AgentType.letta_v1_agent,
            messages=[],
            llm_config=_make_llm_config("MiniMax-M3", enable_reasoner=False, temperature=-0.5),
        )

        assert result["temperature"] == 0.01

    @patch.object(MiniMaxClient.__bases__[0], "build_request_data")
    def test_temperature_above_one_clamped(self, mock_parent):
        mock_parent.return_value = {"temperature": 1.5, "model": "MiniMax-M3"}

        result = self.client.build_request_data(
            agent_type=AgentType.letta_v1_agent,
            messages=[],
            llm_config=_make_llm_config("MiniMax-M3", enable_reasoner=False, temperature=1.5),
        )

        assert result["temperature"] == 1.0

    @patch.object(MiniMaxClient.__bases__[0], "build_request_data")
    def test_temperature_valid_not_modified(self, mock_parent):
        mock_parent.return_value = {"temperature": 0.7, "model": "MiniMax-M3"}

        result = self.client.build_request_data(
            agent_type=AgentType.letta_v1_agent,
            messages=[],
            llm_config=self.llm_config,
        )

        assert result["temperature"] == 0.7


class TestMiniMaxClientUsesStandardMessagesAPI:
    """Tests to verify MiniMax client uses the standard messages API."""

    def test_request_uses_messages_not_beta(self):
        client = MiniMaxClient(put_inner_thoughts_first=True)
        llm_config = _make_llm_config("MiniMax-M3")

        with patch.object(client, "_get_anthropic_client") as mock_get_client:
            mock_anthropic_client = MagicMock()
            mock_response = MagicMock()
            mock_response.model_dump.return_value = {"content": [{"type": "text", "text": "Hello"}]}
            mock_anthropic_client.messages.create.return_value = mock_response
            mock_get_client.return_value = mock_anthropic_client

            client.request({"model": "MiniMax-M3"}, llm_config)

            mock_anthropic_client.messages.create.assert_called_once()
            assert not hasattr(mock_anthropic_client, "beta") or not mock_anthropic_client.beta.messages.create.called

    @pytest.mark.asyncio
    async def test_request_async_uses_messages_not_beta(self):
        client = MiniMaxClient(put_inner_thoughts_first=True)
        llm_config = _make_llm_config("MiniMax-M3")

        with patch.object(client, "_get_anthropic_client_async") as mock_get_client:
            mock_anthropic_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.model_dump.return_value = {"content": [{"type": "text", "text": "Hello"}]}
            mock_anthropic_client.messages.create.return_value = mock_response
            mock_get_client.return_value = mock_anthropic_client

            await client.request_async({"model": "MiniMax-M3"}, llm_config)

            mock_anthropic_client.messages.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_stream_async_uses_messages_not_beta(self):
        client = MiniMaxClient(put_inner_thoughts_first=True)
        llm_config = _make_llm_config("MiniMax-M3")

        with patch.object(client, "_get_anthropic_client_async") as mock_get_client:
            mock_anthropic_client = AsyncMock()
            mock_stream = AsyncMock()
            mock_anthropic_client.messages.create.return_value = mock_stream
            mock_get_client.return_value = mock_anthropic_client

            await client.stream_async({"model": "MiniMax-M3"}, llm_config)

            mock_anthropic_client.messages.create.assert_called_once()
            call_kwargs = mock_anthropic_client.messages.create.call_args[1]
            assert call_kwargs.get("stream") is True
