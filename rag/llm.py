"""Unified, multi-provider LLM layer with API-key rotation.

The pipeline talks to LLMs only through this module. It holds an ordered list
of providers (built from config.PROVIDER_CHAIN — Groq and/or Gemini keys read
from the environment) and exposes two backend-agnostic calls:

    generate(system, user)        -> str           (full response)
    generate_stream(system, user) -> Iterator[str] (token pieces)

When a provider returns a quota / rate-limit error (HTTP 429 / RESOURCE_
EXHAUSTED), the layer advances to the next provider in the chain and retries,
so the app survives free-tier daily caps as long as any key has budget left.
A module-level cursor remembers the last working provider so exhausted keys
are not retried first on every call.

Both SDKs are handled internally: Groq via chat.completions, Gemini via
google.genai. Callers never see which provider answered.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Iterator

from config import LLM_TEMPERATURE, PROVIDER_CHAIN, VISION_PROVIDER_CHAIN

#: Substrings identifying a quota / rate-limit error worth rotating away from.
_QUOTA_SIGNATURES = ("429", "RESOURCE_EXHAUSTED", "rate_limit", "quota",
                     "insufficient_quota", "exceeded")
#: Transient server errors: retry the SAME provider briefly before rotating.
_TRANSIENT_SIGNATURES = ("503", "UNAVAILABLE", "overloaded", "500", "internal")


@dataclass
class Provider:
    """One configured LLM endpoint.

    Attributes:
        kind (str): "groq" or "gemini".
        api_key (str): The API key for this endpoint.
        model (str): Model id to call on this provider.
    """

    kind: str
    api_key: str
    model: str


# Built once on first use: the providers whose env keys are actually set.
_providers: list[Provider] | None = None
_vision_providers: list[Provider] | None = None
#: Index of the provider to try first (advances as keys get exhausted).
_cursor = 0
_vision_cursor = 0
#: Lazily-created SDK clients, keyed by api_key so each key reuses one client.
_clients: dict[str, object] = {}


def _build_chain(chain: list, label: str) -> list[Provider]:
    """Build a provider list from a config chain and the environment.

    Args:
        chain (list): Sequence of (kind, env_var, model) tuples.
        label (str): Human label for the startup log / error message.

    Returns:
        list[Provider]: Providers whose key env var is set, in order.

    Raises:
        RuntimeError: If no provider in the chain has a key configured.
    """
    seen_keys: set[str] = set()
    providers: list[Provider] = []
    for kind, env_var, model in chain:
        key = os.environ.get(env_var)
        if key and key not in seen_keys:  # skip unset & duplicate keys
            seen_keys.add(key)
            providers.append(Provider(kind=kind, api_key=key, model=model))
    if not providers:
        names = ", ".join(env for _, env, _ in chain)
        raise RuntimeError(f"No {label} API key found. Set one of: {names}.")
    print(f"  {label} providers configured: "
          f"{', '.join(p.kind for p in providers)}")
    return providers


def _load_providers() -> list[Provider]:
    """Build (once) the text-generation provider list.

    Returns:
        list[Provider]: Configured text providers in failover order.
    """
    global _providers
    if _providers is None:
        _providers = _build_chain(PROVIDER_CHAIN, "LLM")
    return _providers


def _load_vision_providers() -> list[Provider]:
    """Build (once) the vision provider list.

    Returns:
        list[Provider]: Configured vision providers in failover order.
    """
    global _vision_providers
    if _vision_providers is None:
        _vision_providers = _build_chain(VISION_PROVIDER_CHAIN, "Vision")
    return _vision_providers


def _client_for(provider: Provider):
    """Return a cached SDK client for a provider, creating it on first use.

    Args:
        provider (Provider): The provider to get a client for.

    Returns:
        object: A groq.Groq or google.genai.Client instance.
    """
    if provider.api_key not in _clients:
        if provider.kind == "groq":
            from groq import Groq
            _clients[provider.api_key] = Groq(api_key=provider.api_key)
        else:
            from google import genai
            _clients[provider.api_key] = genai.Client(api_key=provider.api_key)
    return _clients[provider.api_key]


def _is_quota(msg: str) -> bool:
    """Whether an error message indicates an exhausted quota / rate limit."""
    return any(sig in msg for sig in _QUOTA_SIGNATURES)


def _is_transient(msg: str) -> bool:
    """Whether an error message indicates a transient server error."""
    return any(sig in msg for sig in _TRANSIENT_SIGNATURES)


# ---------------------------------------------------------------------------
# Per-provider raw calls
# ---------------------------------------------------------------------------

def _groq_generate(client, model, system, user, stream):
    """Call Groq chat.completions; return str or yield pieces if stream."""
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    if stream:
        def _gen():
            for chunk in client.chat.completions.create(
                model=model, messages=messages,
                temperature=LLM_TEMPERATURE, stream=True,
            ):
                piece = chunk.choices[0].delta.content or ""
                if piece:
                    yield piece
        return _gen()
    resp = client.chat.completions.create(
        model=model, messages=messages,
        temperature=LLM_TEMPERATURE, stream=False,
    )
    return resp.choices[0].message.content or ""


def _gemini_generate(client, model, system, user, stream):
    """Call Gemini generate_content; return str or yield pieces if stream."""
    from google.genai import types

    cfg = types.GenerateContentConfig(
        system_instruction=system, temperature=LLM_TEMPERATURE,
    )
    if stream:
        def _gen():
            for chunk in client.models.generate_content_stream(
                model=model, contents=user, config=cfg,
            ):
                piece = chunk.text or ""
                if piece:
                    yield piece
        return _gen()
    resp = client.models.generate_content(model=model, contents=user, config=cfg)
    return resp.text or ""


def _call_provider(provider: Provider, system: str, user: str, stream: bool):
    """Dispatch one call to the right SDK for a provider."""
    client = _client_for(provider)
    if provider.kind == "groq":
        return _groq_generate(client, provider.model, system, user, stream)
    return _gemini_generate(client, provider.model, system, user, stream)


# ---------------------------------------------------------------------------
# Public, rotation-aware API
# ---------------------------------------------------------------------------

def _attempt(system: str, user: str, stream: bool, max_transient_retries: int = 2):
    """Try providers in chain order, rotating past exhausted ones.

    Starts at the current cursor (last known-good provider) and walks forward.
    On a quota error, advances the cursor and tries the next provider. On a
    transient server error, retries the same provider a couple of times before
    moving on. Raises only when every provider has been exhausted.

    Args:
        system (str): System prompt.
        user (str): User message.
        stream (bool): Whether to return a token iterator.
        max_transient_retries (int): Per-provider retries on 5xx/overload.

    Returns:
        str | Iterator[str]: The response (or token iterator) from the first
            provider that succeeds.

    Raises:
        RuntimeError: If all providers fail with quota errors.
        Exception: The last non-quota error if it is not retryable.
    """
    global _cursor
    providers = _load_providers()
    n = len(providers)
    last_err: Exception | None = None

    for offset in range(n):
        idx = (_cursor + offset) % n
        provider = providers[idx]
        transient_left = max_transient_retries
        while True:
            try:
                if stream:
                    # A streaming call only fires the request when the
                    # generator is consumed, so pull the first chunk HERE,
                    # inside the try, to surface any 429 where we can rotate.
                    gen = _call_provider(provider, system, user, stream=True)
                    first = next(gen, None)
                    _cursor = idx  # request accepted — this provider works
                    return _prepend(first, gen)
                result = _call_provider(provider, system, user, stream=False)
                _cursor = idx  # remember this working provider
                return result
            except Exception as e:  # noqa: BLE001 — classify then route
                msg = str(e)
                last_err = e
                if _is_transient(msg) and transient_left > 0:
                    transient_left -= 1
                    print(f"  [{provider.kind}] transient error; retrying "
                          f"same key in 3s...")
                    time.sleep(3)
                    continue
                if _is_quota(msg) or _is_transient(msg):
                    nxt = providers[(idx + 1) % n] if n > 1 else None
                    where = f" -> trying {nxt.kind}" if nxt and offset < n - 1 else ""
                    print(f"  [{provider.kind}] key exhausted/unavailable{where}.")
                    break  # rotate to next provider
                raise  # non-quota, non-transient error: surface immediately

    raise RuntimeError(
        f"All {n} LLM provider(s) exhausted or unavailable. "
        f"Last error: {last_err}")


def _prepend(first, rest):
    """Yield an already-pulled first chunk, then the rest of a generator.

    Args:
        first: The first token piece (or None if the stream was empty).
        rest (Iterator[str]): The remaining token pieces.

    Yields:
        str: first (if not None), then each remaining piece.
    """
    if first is not None:
        yield first
    yield from rest


def generate(system_prompt: str, user_message: str) -> str:
    """Generate a full response, rotating across providers on quota errors.

    Args:
        system_prompt (str): System instructions.
        user_message (str): User message (context + question).

    Returns:
        str: The model's complete response text.
    """
    return _attempt(system_prompt, user_message, stream=False)


def generate_stream(system_prompt: str, user_message: str) -> Iterator[str]:
    """Stream a response as token pieces, rotating providers on quota errors.

    Note: rotation happens at stream *initiation*. If a provider accepts the
    request then fails mid-stream, that error propagates (the request was
    already counted), matching the previous single-provider behaviour.

    Args:
        system_prompt (str): System instructions.
        user_message (str): User message (context + question).

    Yields:
        str: Successive non-empty text pieces of the response.
    """
    yield from _attempt(system_prompt, user_message, stream=True)


def active_provider_name() -> str:
    """Return the kind of the currently-selected provider (for display).

    Returns:
        str: "groq" or "gemini", or "none" if not yet initialised.
    """
    if _providers and 0 <= _cursor < len(_providers):
        return _providers[_cursor].kind
    return "none"


def get_gemini_client():
    """Return a Gemini client for the first configured Gemini key.

    Returns:
        google.genai.Client: A Gemini client.

    Raises:
        RuntimeError: If no Gemini key is configured.
    """
    for provider in _load_vision_providers():
        if provider.kind == "gemini":
            return _client_for(provider)
    raise RuntimeError("No Gemini API key configured.")


# ---------------------------------------------------------------------------
# Vision (image description), rotation-aware across vision-capable providers
# ---------------------------------------------------------------------------

def _groq_describe(client, model, image_bytes, prompt, max_tokens):
    """Describe an image with a Groq multimodal model (base64 data URL)."""
    import base64
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    resp = client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            ],
        }],
        max_tokens=max_tokens,
    )
    return (resp.choices[0].message.content or "").strip()


def _gemini_describe(client, model, image_bytes, prompt, max_tokens):
    """Describe an image with a Gemini multimodal model."""
    from google.genai import types
    resp = client.models.generate_content(
        model=model,
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
            prompt,
        ],
        config=types.GenerateContentConfig(max_output_tokens=max_tokens),
    )
    return (resp.text or "").strip()


def describe_image(image_bytes: bytes, prompt: str,
                   max_tokens: int = 1024) -> str:
    """Describe an image, rotating across vision providers on quota errors.

    Tries each configured vision provider (Groq Llama-4, then Gemini) in
    order, advancing past any that return a quota/rate-limit error — exactly
    like text generation. Raises only when every vision provider is exhausted,
    so the caller can store a clean placeholder rather than a raw 429 string.

    Args:
        image_bytes (bytes): Encoded image data.
        prompt (str): Description instructions (plus any page context).
        max_tokens (int): Maximum tokens for the description.

    Returns:
        str: The model-generated description.

    Raises:
        RuntimeError: If all vision providers are exhausted/unavailable.
    """
    global _vision_cursor
    providers = _load_vision_providers()
    n = len(providers)
    last_err: Exception | None = None

    for offset in range(n):
        idx = (_vision_cursor + offset) % n
        provider = providers[idx]
        try:
            client = _client_for(provider)
            if provider.kind == "groq":
                out = _groq_describe(client, provider.model, image_bytes,
                                     prompt, max_tokens)
            else:
                out = _gemini_describe(client, provider.model, image_bytes,
                                       prompt, max_tokens)
            _vision_cursor = idx
            return out
        except Exception as e:  # noqa: BLE001 — classify then route
            msg = str(e)
            last_err = e
            if _is_quota(msg) or _is_transient(msg):
                print(f"  [vision/{provider.kind}] exhausted/unavailable; "
                      f"rotating...")
                continue
            raise  # other errors surface immediately

    raise RuntimeError(
        f"All {n} vision provider(s) exhausted. Last error: {last_err}")
