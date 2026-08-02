from typing import Literal

import anthropic
from pydantic import Field

from letta.errors import ErrorCode, LLMAuthenticationError, LLMError
from letta.log import get_logger
from letta.schemas.enums import ProviderCategory, ProviderType
from letta.schemas.llm_config import LLMConfig
from letta.schemas.providers.base import Provider

logger = get_logger(__name__)

# MiniMax model specifications from official documentation
# https://platform.minimax.io/docs/guides/models-intro
MODEL_LIST = [
    {
        "name": "MiniMax-M3",
        "context_window": 1000000,
        "max_output": 128000,
        "description": "Adaptive reasoning with multimodal input support.",
    },
    {
        "name": "MiniMax-M2.7",
        "context_window": 204800,
        "max_output": 128000,
        "description": "Always-on reasoning for text workloads.",
    },
]

DEFAULT_MODEL_NAME = MODEL_LIST[0]["name"]
DEFAULT_CONTEXT_WINDOW = MODEL_LIST[0]["context_window"]


class MiniMaxProvider(Provider):
    """
    MiniMax provider using Anthropic-compatible API.

    MiniMax models support native interleaved thinking without requiring beta headers.
    The API uses the standard messages endpoint (not beta).

    Documentation: https://platform.minimax.io/docs/api-reference/text-anthropic-api
    """

    provider_type: Literal[ProviderType.minimax] = Field(ProviderType.minimax, description="The type of the provider.")
    provider_category: ProviderCategory = Field(ProviderCategory.base, description="The category of the provider (base or byok)")
    api_key: str | None = Field(None, description="API key for the MiniMax API.", deprecated=True)
    base_url: str = Field("https://api.minimax.io/anthropic", description="Base URL for the MiniMax Anthropic-compatible API.")

    async def check_api_key(self):
        """Check if the API key is valid by making a test request to the MiniMax API."""
        api_key = await self.api_key_enc.get_plaintext_async() if self.api_key_enc else None
        if not api_key:
            raise ValueError("No API key provided")

        try:
            # Use async Anthropic client pointed at MiniMax's Anthropic-compatible endpoint
            client = anthropic.AsyncAnthropic(api_key=api_key, base_url=self.base_url)
            # Use count_tokens as a lightweight check - similar to Anthropic provider
            await client.messages.count_tokens(model=DEFAULT_MODEL_NAME, messages=[{"role": "user", "content": "a"}])
        except anthropic.AuthenticationError as e:
            raise LLMAuthenticationError(message=f"Failed to authenticate with MiniMax: {e}", code=ErrorCode.UNAUTHENTICATED)
        except Exception as e:
            raise LLMError(message=f"{e}", code=ErrorCode.INTERNAL_SERVER_ERROR)

    def get_default_max_output_tokens(self, model_name: str) -> int:
        """Get the default max output tokens for MiniMax models."""
        # All MiniMax models support 128K output tokens
        return 128000

    def get_model_context_window_size(self, model_name: str) -> int | None:
        """Get the context window size for a MiniMax model."""
        # Use the current primary model as the fallback context window.
        for model in MODEL_LIST:
            if model["name"] == model_name:
                return model["context_window"]
        return DEFAULT_CONTEXT_WINDOW

    async def list_llm_models_async(self) -> list[LLMConfig]:
        """
        Return available MiniMax models.

        MiniMax doesn't have a models listing endpoint, so we use a hardcoded list.
        """
        configs = []
        for model in MODEL_LIST:
            configs.append(
                LLMConfig(
                    model=model["name"],
                    model_endpoint_type="minimax",
                    model_endpoint=self.base_url,
                    context_window=model["context_window"],
                    handle=self.get_handle(model["name"]),
                    max_tokens=model["max_output"],
                    # MiniMax models support native thinking, similar to Claude's extended thinking
                    put_inner_thoughts_in_kwargs=True,
                    # MiniMax models support parallel tool calling via Anthropic-compatible API
                    parallel_tool_calls=True,
                    provider_name=self.name,
                    provider_category=self.provider_category,
                )
            )
        return configs
