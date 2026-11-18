"""Install per-attempt LiteLLM accounting in GraphRAG child processes.

This module is loaded by Python only when graphrag_bridge adds this directory
to the child process's PYTHONPATH and supplies RAGSTUDY_USAGE_EVENTS.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any


events_path = os.getenv("RAGSTUDY_USAGE_EVENTS")
if events_path:
    import litellm
    from litellm.integrations.custom_logger import CustomLogger
    from ragstudy.usage_tracking import UsageRecorder, estimate_tokens

    class RagstudyUsageLogger(CustomLogger):
        def __init__(self) -> None:
            super().__init__(turn_off_message_logging=True)
            self.recorder = UsageRecorder(
                Path(events_path),
                system="graphrag",
                operation=os.getenv("RAGSTUDY_USAGE_OPERATION", "unknown"),
            )

        @staticmethod
        def _request_value(kwargs: dict[str, Any]) -> Any:
            return kwargs.get("messages", kwargs.get("input", kwargs.get("prompt", "")))

        @staticmethod
        def _model_kind(kwargs: dict[str, Any]) -> str:
            call_type = str(kwargs.get("call_type", "")).lower()
            return "embedding" if "embedding" in call_type or ("input" in kwargs and "messages" not in kwargs) else "generator"

        @staticmethod
        def _usage(response_obj: Any) -> tuple[int, int] | None:
            usage = getattr(response_obj, "usage", None)
            if usage is None and isinstance(response_obj, dict):
                usage = response_obj.get("usage")
            if usage is None:
                return None
            getter = usage.get if isinstance(usage, dict) else lambda name, default=0: getattr(usage, name, default)
            input_tokens = int(getter("prompt_tokens", getter("input_tokens", 0)) or 0)
            output_tokens = int(getter("completion_tokens", getter("output_tokens", 0)) or 0)
            return input_tokens, output_tokens

        def log_success_event(self, kwargs, response_obj, start_time, end_time):
            model = str(kwargs.get("model", "unknown"))
            model_kind = self._model_kind(kwargs)
            usage = self._usage(response_obj)
            if usage is None or usage == (0, 0):
                input_tokens = estimate_tokens(self._request_value(kwargs), model)
                response_value = getattr(response_obj, "content", response_obj)
                output_tokens = 0 if model_kind == "embedding" else estimate_tokens(response_value, model)
                source = "estimated"
            else:
                input_tokens, output_tokens = usage
                source = "provider"
            self.recorder.record(
                model=model, model_kind=model_kind, input_tokens=input_tokens,
                output_tokens=output_tokens, usage_source=source, status="ok",
                request_id=str(kwargs.get("litellm_call_id", "")) or None,
            )

        def log_failure_event(self, kwargs, response_obj, start_time, end_time):
            model = str(kwargs.get("model", "unknown"))
            self.recorder.record(
                model=model,
                model_kind=self._model_kind(kwargs),
                input_tokens=estimate_tokens(self._request_value(kwargs), model),
                output_tokens=None,
                usage_source="partial_estimate",
                status="failed",
                error=str(response_obj),
                request_id=str(kwargs.get("litellm_call_id", "")) or None,
            )

        async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
            self.log_success_event(kwargs, response_obj, start_time, end_time)

        async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
            self.log_failure_event(kwargs, response_obj, start_time, end_time)

    # `callbacks` is LiteLLM's supported all-events hook and avoids registering
    # separate sync/async lists that can double-count asynchronous requests.
    litellm.callbacks = [RagstudyUsageLogger()]
