import time
from fluiq.tracer import log_trace
from fluiq.integrations.shared.models import LogTrace, TraceType
from fluiq.integrations.shared.context import is_in_langchain_llm, current_parent_id
from fluiq.integrations.OpenAI.helper.utils import _to_jsonable


def _summarize_audio_input(kwargs):
    file = kwargs.get("file")
    if file is None:
        return None
    name = getattr(file, "name", None)
    return {"filename": name} if name else {"type": type(file).__name__}


def _safe_jsonable(obj):
    import json
    if obj is None:
        return None
    try:
        result = _to_jsonable(obj)
        return json.loads(json.dumps(result, default=str))
    except Exception:
        return {"type": type(obj).__name__}


def _emit_endpoint_trace(api, kwargs, response, start, end, *, redact_input=None):
    try:
        usage = getattr(response, "usage", None)
        payload = LogTrace(
            type="llm",
            integration=TraceType.OpenAI,
            api=api,
            model=kwargs.get("model") or getattr(response, "model", None),
            input=redact_input if redact_input is not None else _safe_jsonable(
                kwargs.get("input") or kwargs.get("prompt")
            ),
            response=_safe_jsonable(response),
            latency=end - start,
            parent_id=current_parent_id(),
            tokens=_safe_jsonable(usage),
        )
        log_trace(payload.model_dump(mode="json"))
    except Exception:
        pass


def _make_sync_wrapper(original, api, redact_input=None):
    def wrapped(self, *args, **kwargs):
        if is_in_langchain_llm():
            return original(self, *args, **kwargs)
        start = time.time()
        response = original(self, *args, **kwargs)
        end = time.time()
        _emit_endpoint_trace(
            api, kwargs, response, start, end,
            redact_input=redact_input(kwargs) if callable(redact_input) else redact_input,
        )
        return response
    return wrapped


def _make_async_wrapper(original, api, redact_input=None):
    async def wrapped(self, *args, **kwargs):
        if is_in_langchain_llm():
            return await original(self, *args, **kwargs)
        start = time.time()
        response = await original(self, *args, **kwargs)
        end = time.time()
        _emit_endpoint_trace(
            api, kwargs, response, start, end,
            redact_input=redact_input(kwargs) if callable(redact_input) else redact_input,
        )
        return response
    return wrapped


def patch_openai_embeddings():
    from openai.resources.embeddings import Embeddings
    Embeddings.create = _make_sync_wrapper(Embeddings.create, "embeddings")


def patch_openai_embeddings_async():
    from openai.resources.embeddings import AsyncEmbeddings
    AsyncEmbeddings.create = _make_async_wrapper(AsyncEmbeddings.create, "embeddings")


def patch_openai_images():
    from openai.resources.images import Images
    Images.generate = _make_sync_wrapper(Images.generate, "images.generate")
    Images.edit = _make_sync_wrapper(Images.edit, "images.edit")
    Images.create_variation = _make_sync_wrapper(Images.create_variation, "images.variation")


def patch_openai_images_async():
    from openai.resources.images import AsyncImages
    AsyncImages.generate = _make_async_wrapper(AsyncImages.generate, "images.generate")
    AsyncImages.edit = _make_async_wrapper(AsyncImages.edit, "images.edit")
    AsyncImages.create_variation = _make_async_wrapper(AsyncImages.create_variation, "images.variation")


def patch_openai_audio():
    from openai.resources.audio.transcriptions import Transcriptions
    from openai.resources.audio.translations import Translations
    from openai.resources.audio.speech import Speech
    Transcriptions.create = _make_sync_wrapper(
        Transcriptions.create, "audio.transcriptions",
        redact_input=_summarize_audio_input,
    )
    Translations.create = _make_sync_wrapper(
        Translations.create, "audio.translations",
        redact_input=_summarize_audio_input,
    )
    Speech.create = _make_sync_wrapper(Speech.create, "audio.speech")


def patch_openai_audio_async():
    from openai.resources.audio.transcriptions import AsyncTranscriptions
    from openai.resources.audio.translations import AsyncTranslations
    from openai.resources.audio.speech import AsyncSpeech
    AsyncTranscriptions.create = _make_async_wrapper(
        AsyncTranscriptions.create, "audio.transcriptions",
        redact_input=_summarize_audio_input,
    )
    AsyncTranslations.create = _make_async_wrapper(
        AsyncTranslations.create, "audio.translations",
        redact_input=_summarize_audio_input,
    )
    AsyncSpeech.create = _make_async_wrapper(AsyncSpeech.create, "audio.speech")
