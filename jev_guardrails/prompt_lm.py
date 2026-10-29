"""Text-only DSPy proposer engine backed by the official OpenAI/Azure SDK.

Uses DSPy's engine protocol rather than treating Open-Jev as a generative LM.
Credentials are loaded from .env/environment and never included in public config.
"""

import os
from pathlib import Path


class OpenAISDKEngine:
    def __init__(self, client, model):
        """Configure this object without running inference.
        
        Args:
            client: Configured Open-Jev client exposing ask(); reused for classification requests.
            model: Classifier instance for local inference, or model/deployment identifier when configuring a backend or SDK engine.
        """
        self.client, self.model = client, model

    def complete(self, request):
        """Generate one text completion through the configured SDK and convert it to DSPy format.
        
        Args:
            request: DSPy engine Request containing text messages and generation configuration. Tools, images, and structured response formats are unsupported.
        """
        from dspy.lm15 import Message, Response, TextPart, Usage

        if request.tools or request.config.response_format is not None:
            raise ValueError("Prompt proposer engine supports plain text chat only; use dspy.ChatAdapter")
        messages = []
        if request.system is not None:
            system = request.system
            if not isinstance(system, str):
                if any(not isinstance(part, TextPart) for part in system):
                    raise ValueError("Only text system messages are supported")
                system = "".join(part.text for part in system)
            messages.append({"role": "system", "content": system})
        for message in request.messages:
            if any(not isinstance(part, TextPart) for part in message.parts):
                raise ValueError("Only text messages are supported")
            messages.append({"role": message.role, "content": message.text or ""})
        kwargs = {"model": self.model, "messages": messages}
        for field in ("temperature", "max_tokens", "top_p", "seed"):
            if (value := getattr(request.config, field, None)) is not None:
                kwargs[field] = value
        if request.config.stop:
            kwargs["stop"] = list(request.config.stop)
        # One SDK completion per engine call. DSPy owns multi-candidate fan-out,
        # so Azure deployments do not need to support n > 1 in a single request.
        completion = self.client.chat.completions.create(**kwargs)
        choice = completion.choices[0]
        text = choice.message.content
        if not isinstance(text, str) or not text.strip():
            raise ValueError("The prompt proposal model returned empty text")
        if choice.finish_reason not in ("stop", "length"):
            raise ValueError(f"Unsupported prompt completion finish reason: {choice.finish_reason}")
        usage = completion.usage
        return Response(
            id=completion.id, model=completion.model or self.model,
            message=Message.assistant([TextPart(text)]), finish_reason=choice.finish_reason,
            usage=Usage(input_tokens=getattr(usage, "prompt_tokens", None),
                        output_tokens=getattr(usage, "completion_tokens", None),
                        total_tokens=getattr(usage, "total_tokens", None)),
        )

    def stream(self, request):
        """Yield DSPy events for a completed SDK response; this buffers the response first.
        
        Args:
            request: DSPy engine Request containing text messages and generation configuration. Tools, images, and structured response formats are unsupported.
        """
        from dspy.lm15 import response_to_events
        yield from response_to_events(self.complete(request))

    def close(self):
        """Close the underlying SDK client.
        """
        self.client.close()


def prompt_lm_from_env(env_file, *, provider="azure", temperature=.7, max_tokens=1500):
    """Return (dspy.LM, sanitized_config), without making any network request.
    
    Existing environment variables take precedence over .env values. Azure uses
    AZURE_OPENAI_API_KEY, ENDPOINT, DEPLOYMENT and API_VERSION. OpenAI-compatible
    mode uses OPENAI_API_KEY, OPENAI_MODEL and optional OPENAI_BASE_URL.
    
    Args:
        env_file: Path to the .env file containing provider credentials and settings. Existing environment variables take precedence; secrets are not returned.
        provider: "azure" (default) uses AZURE_OPENAI_API_KEY/ENDPOINT/DEPLOYMENT/API_VERSION; "openai" uses OPENAI_API_KEY/MODEL and optional OPENAI_BASE_URL.
        temperature: Proposal sampling temperature (default 0.7), subject to provider support. Higher values generally vary rewrites more; even low values do not guarantee reproducibility.
        max_tokens: Maximum generated tokens per proposal completion (default 1500); too small a limit can truncate the instruction.
    """
    import dspy
    from dotenv import dotenv_values
    from openai import AzureOpenAI, OpenAI

    settings = {**dotenv_values(Path(env_file)), **os.environ}
    def required(name):
        """Required.
        
        Args:
            name: Dataset key from the downloader registry, or configuration variable name when resolving a required setting.
        """
        value = settings.get(name)
        if not value:
            raise ValueError(f"Set {name} in {env_file} or the environment")
        return value
    if provider == "azure":
        deployment = required("AZURE_OPENAI_DEPLOYMENT")
        endpoint = required("AZURE_OPENAI_ENDPOINT")
        version = required("AZURE_OPENAI_API_VERSION")
        client = AzureOpenAI(api_key=required("AZURE_OPENAI_API_KEY"), azure_endpoint=endpoint,
                             api_version=version, timeout=120, max_retries=0)
        public = {"provider": "azure", "deployment": deployment, "endpoint": endpoint, "api_version": version}
    elif provider == "openai":
        deployment = required("OPENAI_MODEL")
        base_url = settings.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"
        client = OpenAI(api_key=required("OPENAI_API_KEY"), base_url=base_url, timeout=120, max_retries=0)
        public = {"provider": "openai", "model": deployment, "base_url": base_url}
    else:
        raise ValueError("provider must be azure or openai")
    # Avoid cross-endpoint cache collisions when deployments share a model name.
    public["cache"] = False
    lm = dspy.LM(f"{provider}/{deployment}", engine=OpenAISDKEngine(client, deployment),
                 temperature=temperature, max_tokens=max_tokens, cache=False)
    return lm, public
