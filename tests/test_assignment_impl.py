from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from google.genai import types


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def test_rate_limit_audit_and_metrics(tmp_path):
    from assignment.audit_log import AuditLogPlugin
    from assignment.monitoring import MonitoringAlert
    from assignment.rate_limiter import RateLimitPlugin

    class Context:
        user_id = "student-1"

    limiter = RateLimitPlugin(max_requests=1, window_seconds=60)
    message = types.Content(role="user", parts=[types.Part.from_text(text="balance")])
    assert asyncio.run(limiter.on_user_message_callback(invocation_context=Context(), user_message=message)) is None
    assert asyncio.run(limiter.on_user_message_callback(invocation_context=Context(), user_message=message)) is not None

    audit = AuditLogPlugin()
    audit.record_input(user_id="student-1", text="balance", request_id="r1")
    audit.record_output(user_id="student-1", text="blocked", blocked=True, layer="rate_limiter", request_id="r1")
    audit.export_json(str(tmp_path / "audit.json"))
    assert (tmp_path / "audit.json").exists()

    monitor = MonitoringAlert(block_rate_threshold=0.5)
    monitor.total_requests, monitor.blocked_requests = 2, 1
    assert monitor.check_metrics() == []
    monitor.blocked_requests = 2
    assert monitor.check_metrics()[0].metric == "block_rate"
