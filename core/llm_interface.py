"""
LLM Interface - Unified interface for language models

Supports multiple providers:
- Local models (llama-cpp-python, transformers)
- API models (OpenAI, Anthropic)
- Mock LLM (for testing)

Integrates with the config system and provides consistent interface.
"""

import os
import sys
import time
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.append(str(Path(__file__).parent.parent))
from models import ModelConfig, ModelProvider


def _set_below_normal_priority() -> None:
    """
    Lower the current process's CPU scheduling priority so Cortex never
    starves other applications during inference.

    Windows: sets BELOW_NORMAL_PRIORITY_CLASS via the Win32 API.
    Linux/macOS: increments the nice value by 5 (higher = less aggressive).
    Silent no-op if the call fails for any reason.
    """
    try:
        if sys.platform == "win32":
            import ctypes
            # BELOW_NORMAL_PRIORITY_CLASS = 0x00004000
            handle = ctypes.windll.kernel32.GetCurrentProcess()
            ctypes.windll.kernel32.SetPriorityClass(handle, 0x00004000)
        else:
            current = os.nice(0)
            os.nice(max(0, 5 - current))  # bump up by 5, don't exceed 19
    except Exception:  # noqa: BLE001
        pass


class MessageRole(Enum):
    """Message roles in conversation"""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class Message:
    """Represents a message in the conversation"""

    def __init__(self, role: MessageRole, content: str):
        self.role = role
        self.content = content

    def to_dict(self) -> Dict[str, str]:
        return {"role": self.role.value, "content": self.content}


class LLMResponse:
    """Response from LLM"""

    def __init__(
        self,
        content: str,
        model: str,
        tokens_used: int = 0,
        finish_reason: str = "stop",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.content = content
        self.model = model
        self.tokens_used = tokens_used
        self.finish_reason = finish_reason
        self.metadata = metadata or {}


class BaseLLM(ABC):
    """Base class for all LLM implementations"""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.model_name = config.model_name

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """
        Generate text from a prompt.

        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature (overrides config)
            max_tokens: Max tokens to generate (overrides config)

        Returns:
            LLMResponse object
        """
        pass

    @abstractmethod
    def chat(
        self,
        messages: List[Message],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """
        Chat-style generation with message history.

        Args:
            messages: List of Message objects
            temperature: Sampling temperature
            max_tokens: Max tokens to generate

        Returns:
            LLMResponse object
        """
        pass

    def _get_temperature(self, override: Optional[float] = None) -> float:
        """Get temperature, preferring override"""
        return override if override is not None else self.config.temperature

    def _get_max_tokens(self, override: Optional[int] = None) -> int:
        """Get max tokens, preferring override"""
        return override if override is not None else self.config.max_tokens


# Simulated inference delay for MockLLM (seconds)
_MOCK_RESPONSE_DELAY_S = 0.1


class MockLLM(BaseLLM):
    """Mock LLM for testing when real models aren't available."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        print(f"[MockLLM] Initialized with model: {self.model_name}")

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """Generate mock response"""

        # Simulate processing time
        time.sleep(_MOCK_RESPONSE_DELAY_S)

        # Generate simple mock response
        response = self._generate_mock_response(prompt, system_prompt)

        return LLMResponse(
            content=response,
            model=self.model_name,
            tokens_used=len(response.split()),
            finish_reason="stop",
            metadata={"mock": True},
        )

    def chat(
        self,
        messages: List[Message],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """Chat with mock responses"""

        # Get last user message
        user_messages = [m for m in messages if m.role == MessageRole.USER]
        last_prompt = user_messages[-1].content if user_messages else ""

        # Get system message if exists
        system_messages = [m for m in messages if m.role == MessageRole.SYSTEM]
        system_prompt = system_messages[0].content if system_messages else None

        return self.generate(last_prompt, system_prompt, temperature, max_tokens)

    def _generate_mock_response(
        self, prompt: str, system_prompt: Optional[str] = None
    ) -> str:
        """Generate a simple mock response based on the prompt"""

        prompt_lower = prompt.lower()

        # Task planning
        if (
            "plan" in prompt_lower
            or "steps" in prompt_lower
            or "break down" in prompt_lower
        ):
            return """I'll break this down into steps:

1. Analyze the requirements
2. Create a detailed plan
3. Execute each step in sequence
4. Verify the results
5. Report completion

This approach ensures systematic completion of the task."""

        # File operations
        elif (
            "file" in prompt_lower or "read" in prompt_lower or "write" in prompt_lower
        ):
            return "I'll handle the file operation. The file has been processed successfully."

        # Code generation
        elif (
            "code" in prompt_lower
            or "function" in prompt_lower
            or "script" in prompt_lower
        ):
            return """Here's the code:

```python
def example_function():
    # Generated code
    result = process_data()
    return result
```

This code implements the requested functionality."""

        # Web search
        elif (
            "search" in prompt_lower
            or "find" in prompt_lower
            or "look up" in prompt_lower
        ):
            return "Based on my search, here are the key findings: The information you requested has been located and compiled."

        # Data analysis
        elif "analyze" in prompt_lower or "data" in prompt_lower:
            return "Analysis complete. The data shows consistent patterns with the expected distribution."

        # General response
        else:
            return f"I understand your request: '{prompt}'. I'll process this task and provide the appropriate response."


class LlamaCppLLM(BaseLLM):
    """LLM using llama-cpp-python for GGUF models"""

    def __init__(self, config: ModelConfig):
        super().__init__(config)

        # Lower this process's CPU priority so Cortex never starves other apps.
        # On Windows: BELOW_NORMAL_PRIORITY_CLASS (0x4000).
        # On Linux/macOS: renice to +5 (lower = less aggressive).
        _set_below_normal_priority()

        try:
            from llama_cpp import Llama

            self.llm = Llama(
                model_path=config.model_path,
                n_ctx=config.context_window,
                n_threads=config.threads,
                n_batch=config.batch_size,   # smaller = less per-burst CPU spike
                n_gpu_layers=config.gpu_layers if config.use_gpu else 0,
                use_mmap=True,               # memory-mapped weights — less RAM pressure
                verbose=False,
            )
            print(f"[LlamaCpp] Loaded model: {config.model_path}")

        except ImportError:
            raise ImportError(
                "llama-cpp-python not installed. Install with: pip install llama-cpp-python"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")

    # Stop tokens for Mistral Instruct — end-of-sentence and role markers
    _STOP_TOKENS = ["</s>", "[INST]", "[/INST]"]

    def _build_instruct_prompt(
        self, user_message: str, system_prompt: Optional[str] = None
    ) -> str:
        """
        Format a prompt in Mistral Instruct style.

        Mistral Instruct format (single turn):
            <s>[INST] {optional_system}\\n\\n{user} [/INST]

        Without the [INST] wrapper the model treats the input as raw text
        to complete, causing it to hallucinate both sides of the conversation.
        """
        inner = f"{system_prompt}\n\n{user_message}" if system_prompt else user_message
        return f"[INST] {inner} [/INST]"

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """Generate text using llama.cpp with Mistral Instruct formatting."""
        full_prompt = self._build_instruct_prompt(prompt, system_prompt)

        response = self.llm(
            full_prompt,
            max_tokens=self._get_max_tokens(max_tokens),
            temperature=self._get_temperature(temperature),
            top_p=self.config.top_p,
            stop=self._STOP_TOKENS,
        )

        try:
            content = response["choices"][0]["text"].strip()
            tokens_used = response["usage"]["total_tokens"]
            finish_reason = response["choices"][0]["finish_reason"]
        except (KeyError, IndexError) as e:
            raise RuntimeError(
                f"Unexpected llama.cpp response structure: {e}. Got: {response}"
            )

        return LLMResponse(
            content=content,
            model=self.model_name,
            tokens_used=tokens_used,
            finish_reason=finish_reason,
        )

    def chat(
        self,
        messages: List[Message],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """
        Chat using llama.cpp with Mistral multi-turn format.

        Mistral multi-turn format:
            <s>[INST] turn1 [/INST] reply1</s>[INST] turn2 [/INST]
        """
        parts: List[str] = []
        system_prefix = ""

        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                # Prepend system content to the first user turn
                system_prefix = msg.content + "\n\n"
            elif msg.role == MessageRole.USER:
                user_text = system_prefix + msg.content
                system_prefix = ""
                parts.append(f"[INST] {user_text} [/INST]")
            elif msg.role == MessageRole.ASSISTANT:
                parts.append(f"{msg.content}</s>")

        full_prompt = "".join(parts)

        response = self.llm(
            full_prompt,
            max_tokens=self._get_max_tokens(max_tokens),
            temperature=self._get_temperature(temperature),
            top_p=self.config.top_p,
            stop=self._STOP_TOKENS,
        )

        try:
            content = response["choices"][0]["text"].strip()
            tokens_used = response["usage"]["total_tokens"]
            finish_reason = response["choices"][0]["finish_reason"]
        except (KeyError, IndexError) as e:
            raise RuntimeError(
                f"Unexpected llama.cpp response structure: {e}. Got: {response}"
            )

        return LLMResponse(
            content=content,
            model=self.model_name,
            tokens_used=tokens_used,
            finish_reason=finish_reason,
        )


class OpenAILLM(BaseLLM):
    """LLM using OpenAI API"""

    def __init__(self, config: ModelConfig):
        super().__init__(config)

        try:
            from openai import OpenAI

            self.client = OpenAI(api_key=config.api_key)
            print(f"[OpenAI] Initialized with model: {config.model_name}")

        except ImportError:
            raise ImportError(
                "openai package not installed. Install with: pip install openai"
            )

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """Generate using OpenAI API"""

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.config.model_name,
            messages=messages,
            temperature=self._get_temperature(temperature),
            max_tokens=self._get_max_tokens(max_tokens),
        )

        return LLMResponse(
            content=response.choices[0].message.content,
            model=response.model,
            tokens_used=response.usage.total_tokens,
            finish_reason=response.choices[0].finish_reason,
        )

    def chat(
        self,
        messages: List[Message],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """Chat using OpenAI API"""

        openai_messages = [msg.to_dict() for msg in messages]

        response = self.client.chat.completions.create(
            model=self.config.model_name,
            messages=openai_messages,
            temperature=self._get_temperature(temperature),
            max_tokens=self._get_max_tokens(max_tokens),
        )

        return LLMResponse(
            content=response.choices[0].message.content,
            model=response.model,
            tokens_used=response.usage.total_tokens,
            finish_reason=response.choices[0].finish_reason,
        )


class LLMInterface:
    """
    Main LLM interface that automatically selects the right provider.

    This is what the rest of the system should use.
    """

    def __init__(self, config: ModelConfig, use_mock: bool = False):
        """
        Initialize LLM interface.

        Args:
            config: Model configuration
            use_mock: Force use of mock LLM (for testing)
        """
        self.config = config

        if use_mock:
            self.llm = MockLLM(config)
            return

        # Select provider based on config
        if config.provider == ModelProvider.LOCAL:
            try:
                self.llm = LlamaCppLLM(config)
            except (ImportError, RuntimeError) as e:
                print(f"Warning: Could not load local model: {e}")
                print("Falling back to MockLLM")
                self.llm = MockLLM(config)

        elif config.provider == ModelProvider.OPENAI:
            try:
                self.llm = OpenAILLM(config)
            except ImportError as e:
                print(f"Warning: Could not load OpenAI: {e}")
                print("Falling back to MockLLM")
                self.llm = MockLLM(config)

        else:
            print(
                f"Provider {config.provider.value} not yet implemented, using MockLLM"
            )
            self.llm = MockLLM(config)

    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate text from prompt"""
        return self.llm.generate(prompt, **kwargs)

    def chat(self, messages: List[Message], **kwargs) -> LLMResponse:
        """Chat with message history"""
        return self.llm.chat(messages, **kwargs)

    def ask(self, question: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        """
        Convenience method to ask a question and get just the text response.

        Args:
            question: The question to ask
            system_prompt: Optional system instructions
            **kwargs: Additional generation parameters

        Returns:
            String response
        """
        response = self.generate(question, system_prompt=system_prompt, **kwargs)
        return response.content


if __name__ == "__main__":
    # Test the LLM interface
    print("Testing LLM Interface...")

    # Create config
    config = ModelConfig(
        provider=ModelProvider.LOCAL,
        model_name="mistral-7b-mock",
        temperature=0.7,
        max_tokens=512,
    )

    # Initialize with mock (since we don't have real models)
    llm = LLMInterface(config, use_mock=True)

    print("\n--- Testing Generate ---")
    response = llm.generate(
        "Create a plan to organize project files",
        system_prompt="You are a helpful AI assistant.",
    )
    print(f"Response: {response.content}")
    print(f"Tokens: {response.tokens_used}")
    print(f"Model: {response.model}")

    print("\n--- Testing Chat ---")
    messages = [
        Message(MessageRole.SYSTEM, "You are a helpful coding assistant."),
        Message(MessageRole.USER, "Write a Python function to calculate factorial"),
        Message(MessageRole.ASSISTANT, "I'll help you with that."),
        Message(MessageRole.USER, "Make it recursive"),
    ]

    response = llm.chat(messages)
    print(f"Response: {response.content}")

    print("\n--- Testing Ask (Convenience) ---")
    answer = llm.ask("What are the steps to build a web scraper?")
    print(f"Answer: {answer}")

    print("\nLLM Interface test completed!")
