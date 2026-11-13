"""LangChain's standard sandbox suite (`langchain_tests` SandboxIntegrationTests).

`TestMicroVMSandboxStandard` runs it against a real MicroVM and needs
credentials, so it is marked `live` and deselected by default:

    uv run pytest -m live -q

It reads DO_MICROVM_API_TOKEN and DO_MICROVM_IMAGE (plus the optional
DO_MICROVM_* settings) from the environment or the repository's .env, creates
one MicroVM for the class, and deletes it afterwards. With
DO_MICROVM_TRANSPORT=endpoint (and DO_MICROVM_ACCESS_TOKEN) it runs through
the endpoint transport; the test fails if that fell back to exec.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
from deepagents.backends.protocol import SandboxBackendProtocol
from langchain_tests.integration_tests import SandboxIntegrationTests
from standard_suite_gaps import upstream_gaps

from langchain_digitalocean_microvm import MicroVMConfig, MicroVMLifecycle

pytestmark = pytest.mark.live


def _load_dotenv() -> None:
    env_file = Path(__file__).resolve().parents[2] / ".env"
    if env_file.exists():
        from dotenv import load_dotenv

        load_dotenv(env_file, override=False)


class TestMicroVMSandboxStandard(SandboxIntegrationTests):
    locals().update(upstream_gaps())  # known deepagents/langchain-tests disagreements

    @pytest.fixture(scope="class")
    def sandbox(self) -> Iterator[SandboxBackendProtocol]:
        _load_dotenv()
        config = MicroVMConfig.from_env()
        if config.transport not in ("exec", "endpoint") or not config.api_token or not config.image:
            pytest.skip("needs DO_MICROVM_TRANSPORT=exec or endpoint, DO_MICROVM_API_TOKEN and DO_MICROVM_IMAGE")
        lifecycle = MicroVMLifecycle(config)
        sandbox_id = f"std-tests-{uuid.uuid4().hex[:8]}"
        backend = lifecycle.get_or_create(sandbox_id=sandbox_id)
        if config.transport == "endpoint":
            assert not backend.transport.uses_fallback, "the endpoint transport fell back to exec"  # type: ignore[attr-defined]
        try:
            yield backend
        finally:
            lifecycle.delete(sandbox_id=sandbox_id)
