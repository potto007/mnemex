"""Tests to verify all imports are correct and non-conflicting."""

import subprocess
import sys
from collections import defaultdict

import pytest


class TestTopLevelImports:
    """Test top-level package imports."""

    def test_rlm_import(self):
        """Test that main mnemex package can be imported."""
        import mnemex

        assert hasattr(mnemex, "RLM")
        assert "RLM" in mnemex.__all__

    def test_rlm_rlm_import(self):
        """Test that RLM class can be imported from mnemex."""
        from mnemex import RLM

        assert RLM is not None

    def test_rlm_core_rlm_import(self):
        """Test that RLM can be imported from mnemex.core.rlm."""
        from mnemex.core.rlm import RLM

        assert RLM is not None


class TestClientImports:
    """Test client module imports."""

    def test_clients_module_import(self):
        """Test that clients module can be imported."""
        import mnemex.clients

        assert hasattr(mnemex.clients, "get_client")
        assert hasattr(mnemex.clients, "BaseLM")

    def test_base_lm_import(self):
        """Test BaseLM import."""
        from mnemex.clients.base_lm import BaseLM

        assert BaseLM is not None

    def test_openai_client_import(self):
        """Test OpenAIClient import."""
        pytest.importorskip("openai")
        from mnemex.clients.openai import OpenAIClient

        assert OpenAIClient is not None

    def test_anthropic_client_import(self):
        """Test AnthropicClient import."""
        pytest.importorskip("anthropic")
        from mnemex.clients.anthropic import AnthropicClient

        assert AnthropicClient is not None

    def test_portkey_client_import(self):
        """Test PortkeyClient import."""
        pytest.importorskip("portkey_ai")
        from mnemex.clients.portkey import PortkeyClient

        assert PortkeyClient is not None

    def test_get_client_function(self):
        """Test get_client function import."""
        from mnemex.clients import get_client

        assert callable(get_client)


class TestCoreImports:
    """Test core module imports."""

    def test_core_types_import(self):
        """Test core types imports."""
        from mnemex.core.types import (
            ClientBackend,
            CodeBlock,
            ModelUsageSummary,
            QueryMetadata,
            REPLResult,
            RLMIteration,
            RLMMetadata,
            UsageSummary,
        )

        assert ClientBackend is not None
        assert CodeBlock is not None
        assert ModelUsageSummary is not None
        assert QueryMetadata is not None
        assert REPLResult is not None
        assert RLMIteration is not None
        assert RLMMetadata is not None
        assert UsageSummary is not None

    def test_core_rlm_import(self):
        """Test core RLM import."""
        from mnemex.core.rlm import RLM

        assert RLM is not None

    def test_core_lm_handler_import(self):
        """Test LMHandler import."""
        from mnemex.core.lm_handler import LMHandler

        assert LMHandler is not None

    def test_core_comms_utils_import(self):
        """Test comms_utils imports."""
        from mnemex.core.comms_utils import (
            LMRequest,
            LMResponse,
            send_lm_request,
            send_lm_request_batched,
            socket_recv,
            socket_send,
        )

        assert LMRequest is not None
        assert LMResponse is not None
        assert callable(send_lm_request)
        assert callable(send_lm_request_batched)
        assert callable(socket_recv)
        assert callable(socket_send)


class TestEnvironmentImports:
    """Test environment module imports."""

    def test_environments_module_import(self):
        """Test that environments module can be imported."""
        import mnemex.environments

        assert hasattr(mnemex.environments, "get_environment")
        assert hasattr(mnemex.environments, "BaseEnv")
        assert hasattr(mnemex.environments, "LocalREPL")

    def test_base_env_import(self):
        """Test BaseEnv import."""
        from mnemex.environments.base_env import BaseEnv, IsolatedEnv, NonIsolatedEnv

        assert BaseEnv is not None
        assert IsolatedEnv is not None
        assert NonIsolatedEnv is not None

    def test_local_repl_import(self):
        """Test LocalREPL import."""
        from mnemex.environments.local_repl import LocalREPL

        assert LocalREPL is not None

    def test_modal_repl_import(self):
        """Test ModalREPL import."""
        pytest.importorskip("modal")
        from mnemex.environments.modal_repl import ModalREPL

        assert ModalREPL is not None

    def test_docker_repl_import(self):
        """Test DockerREPL import."""
        from mnemex.environments.docker_repl import DockerREPL

        assert DockerREPL is not None

    def test_prime_repl_import(self):
        """Test PrimeREPL import."""
        pytest.importorskip("prime_sandboxes")
        from mnemex.environments.prime_repl import PrimeREPL

        assert PrimeREPL is not None

    def test_get_environment_function(self):
        """Test get_environment function import."""
        from mnemex.environments import get_environment

        assert callable(get_environment)


class TestLoggerImports:
    """Test logger module imports."""

    def test_logger_module_import(self):
        """Test that logger module can be imported."""
        import mnemex.logger

        assert hasattr(mnemex.logger, "RLMLogger")
        assert hasattr(mnemex.logger, "VerbosePrinter")
        assert "RLMLogger" in mnemex.logger.__all__
        assert "VerbosePrinter" in mnemex.logger.__all__

    def test_rlm_logger_import(self):
        """Test RLMLogger import."""
        from mnemex.logger.rlm_logger import RLMLogger

        assert RLMLogger is not None

    def test_verbose_import(self):
        """Test VerbosePrinter import."""
        from mnemex.logger.verbose import VerbosePrinter

        assert VerbosePrinter is not None


class TestUtilsImports:
    """Test utils module imports."""

    def test_parsing_import(self):
        """Test parsing module import."""
        from mnemex.utils.parsing import (
            find_code_blocks,
            format_execution_result,
            format_iteration,
        )

        assert callable(find_code_blocks)
        assert callable(format_iteration)
        assert callable(format_execution_result)

    def test_prompts_import(self):
        """Test prompts module import."""
        from mnemex.utils.prompts import (
            RLM_SYSTEM_PROMPT,
            USER_PROMPT,
            build_rlm_system_prompt,
            build_user_prompt,
        )

        assert RLM_SYSTEM_PROMPT is not None
        assert USER_PROMPT is not None
        assert callable(build_rlm_system_prompt)
        assert callable(build_user_prompt)

    def test_rlm_utils_import(self):
        """Test rlm_utils module import."""
        from mnemex.utils.rlm_utils import filter_sensitive_keys

        assert callable(filter_sensitive_keys)


class TestImportConflicts:
    """Test for import conflicts and naming issues."""

    def test_no_duplicate_names_in_rlm_all(self):
        """Test that __all__ in mnemex.__init__ has no duplicates."""
        import mnemex

        if hasattr(mnemex, "__all__"):
            all_items = mnemex.__all__
            assert len(all_items) == len(set(all_items)), (
                f"Duplicate items in mnemex.__all__: {all_items}"
            )

    def test_no_duplicate_names_in_logger_all(self):
        """Test that __all__ in mnemex.logger.__init__ has no duplicates."""
        import mnemex.logger

        if hasattr(mnemex.logger, "__all__"):
            all_items = mnemex.logger.__all__
            assert len(all_items) == len(set(all_items)), (
                f"Duplicate items in mnemex.logger.__all__: {all_items}"
            )

    def test_all_declarations_match_exports(self):
        """Test that __all__ declarations match actual exports."""
        import mnemex
        import mnemex.logger

        # Test mnemex.__all__
        if hasattr(mnemex, "__all__"):
            for name in mnemex.__all__:
                assert hasattr(mnemex, name), f"mnemex.__all__ declares '{name}' but it's not exported"

        # Test mnemex.logger.__all__
        if hasattr(mnemex.logger, "__all__"):
            for name in mnemex.logger.__all__:
                assert hasattr(mnemex.logger, name), (
                    f"mnemex.logger.__all__ declares '{name}' but it's not exported"
                )

    def test_no_circular_imports(self):
        """Test that modules can be imported without circular import errors.

        Runs in a SUBPROCESS so imports are genuinely fresh. Deleting entries
        from sys.modules in-process and re-importing creates new class objects
        while already-imported modules keep the old ones, breaking isinstance
        checks for every test that runs afterwards."""
        # Core modules that should always be importable
        core_modules = [
            "mnemex",
            "mnemex.clients",
            "mnemex.clients.base_lm",
            "mnemex.core",
            "mnemex.core.types",
            "mnemex.core.rlm",
            "mnemex.core.lm_handler",
            "mnemex.core.comms_utils",
            "mnemex.environments",
            "mnemex.environments.base_env",
            "mnemex.environments.local_repl",
            "mnemex.environments.docker_repl",
            "mnemex.logger",
            "mnemex.logger.rlm_logger",
            "mnemex.logger.verbose",
            "mnemex.utils",
            "mnemex.utils.parsing",
            "mnemex.utils.prompts",
            "mnemex.utils.rlm_utils",
        ]

        # Optional modules imported only when their dependency is available
        optional_modules = [
            ("mnemex.clients.openai", "openai"),
            ("mnemex.clients.anthropic", "anthropic"),
            ("mnemex.clients.portkey", "portkey_ai"),
            ("mnemex.environments.modal_repl", "modal"),
            ("mnemex.environments.prime_repl", "prime_sandboxes"),
        ]

        script = (
            "import importlib\n"
            f"for name in {core_modules!r}:\n"
            "    importlib.import_module(name)\n"
            f"for name, dep in {optional_modules!r}:\n"
            "    try:\n"
            "        importlib.import_module(dep)\n"
            "    except ImportError:\n"
            "        continue\n"
            "    importlib.import_module(name)\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True
        )
        assert proc.returncode == 0, f"Circular/broken import:\n{proc.stderr}"

    def test_no_naming_conflicts_across_modules(self):
        """Test that there are no naming conflicts across different modules."""
        # Collect all public names from each module
        module_exports: dict[str, set[str]] = {}

        # Check main modules
        import mnemex
        import mnemex.clients
        import mnemex.environments
        import mnemex.logger

        if hasattr(mnemex, "__all__"):
            module_exports["mnemex"] = set(mnemex.__all__)
        else:
            module_exports["mnemex"] = {name for name in dir(mnemex) if not name.startswith("_")}

        if hasattr(mnemex.clients, "__all__"):
            module_exports["mnemex.clients"] = set(mnemex.clients.__all__)
        else:
            module_exports["mnemex.clients"] = {
                name for name in dir(mnemex.clients) if not name.startswith("_")
            }

        if hasattr(mnemex.environments, "__all__"):
            module_exports["mnemex.environments"] = set(mnemex.environments.__all__)
        else:
            module_exports["mnemex.environments"] = {
                name for name in dir(mnemex.environments) if not name.startswith("_")
            }

        if hasattr(mnemex.logger, "__all__"):
            module_exports["mnemex.logger"] = set(mnemex.logger.__all__)
        else:
            module_exports["mnemex.logger"] = {
                name for name in dir(mnemex.logger) if not name.startswith("_")
            }

        # Check for conflicts (same name in multiple modules)
        name_to_modules: dict[str, list[str]] = defaultdict(list)
        for module_name, exports in module_exports.items():
            for export_name in exports:
                name_to_modules[export_name].append(module_name)

        conflicts = {name: modules for name, modules in name_to_modules.items() if len(modules) > 1}
        # Filter out common Python builtins/dunders and typing imports that are expected
        expected_duplicates = {
            "__file__",
            "__name__",
            "__package__",
            "__path__",
            "__doc__",
            "__loader__",
            "__spec__",
            "__cached__",
            "Any",  # Common typing import
            "Literal",  # Common typing import
            "Optional",  # Common typing import
            "Union",  # Common typing import
            "Dict",  # Common typing import
            "List",  # Common typing import
            "Tuple",  # Common typing import
            "Callable",  # Common typing import
        }
        conflicts = {
            name: modules for name, modules in conflicts.items() if name not in expected_duplicates
        }

        if conflicts:
            conflict_msg = "\n".join(
                f"  '{name}' exported from: {', '.join(modules)}"
                for name, modules in conflicts.items()
            )
            pytest.fail(f"Found naming conflicts across modules:\n{conflict_msg}")


class TestImportCompleteness:
    """Test that all expected imports are available."""

    def test_all_client_classes_importable(self):
        """Test that all client classes can be imported."""
        from mnemex.clients.base_lm import BaseLM

        # Verify BaseLM is a class
        assert isinstance(BaseLM, type)

        # Test optional client classes
        try:
            pytest.importorskip("openai")
            from mnemex.clients.openai import OpenAIClient

            assert isinstance(OpenAIClient, type)
        except Exception:
            pass

        try:
            pytest.importorskip("anthropic")
            from mnemex.clients.anthropic import AnthropicClient

            assert isinstance(AnthropicClient, type)
        except Exception:
            pass

        try:
            pytest.importorskip("portkey_ai")
            from mnemex.clients.portkey import PortkeyClient

            assert isinstance(PortkeyClient, type)
        except Exception:
            pass

    def test_all_environment_classes_importable(self):
        """Test that all environment classes can be imported."""
        from mnemex.environments.base_env import BaseEnv, IsolatedEnv, NonIsolatedEnv
        from mnemex.environments.docker_repl import DockerREPL
        from mnemex.environments.local_repl import LocalREPL

        # Verify they're all classes
        assert isinstance(BaseEnv, type)
        assert isinstance(IsolatedEnv, type)
        assert isinstance(NonIsolatedEnv, type)
        assert isinstance(LocalREPL, type)
        assert isinstance(DockerREPL, type)

        # Test optional ModalREPL
        try:
            pytest.importorskip("modal")
            from mnemex.environments.modal_repl import ModalREPL

            assert isinstance(ModalREPL, type)
        except Exception:
            pass

        # Test optional PrimeREPL
        try:
            pytest.importorskip("prime_sandboxes")
            from mnemex.environments.prime_repl import PrimeREPL

            assert isinstance(PrimeREPL, type)
        except Exception:
            pass
