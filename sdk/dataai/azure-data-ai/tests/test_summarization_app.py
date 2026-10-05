# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Tests for the optional Gradio demo; install samples/requirements-gradio.txt to run."""

import importlib.util
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from azure.core.exceptions import (
    ClientAuthenticationError,
    DecodeError,
    HttpResponseError,
    ResourceNotFoundError,
    ServiceRequestError,
    ServiceResponseError,
)
from azure.data.ai import InferenceClient

gr = pytest.importorskip("gradio", minversion="6.28.0", reason="Optional Gradio demo dependencies are not installed.")


@pytest.fixture(scope="module")
def app():
    path = Path(__file__).resolve().parents[1] / "samples" / "summarization_app.py"
    spec = importlib.util.spec_from_file_location("summarization_app", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def sdk(app, monkeypatch):
    aml_credential, onelake_credential = MagicMock(), MagicMock()
    aml_credential.__enter__.return_value = aml_credential
    onelake_credential.__enter__.return_value = onelake_credential
    credentials = MagicMock(side_effect=[aml_credential, onelake_credential])
    client = MagicMock()
    client.__enter__.return_value = client
    client.summarize.return_value = {"summary": "A concise summary.", "extra": {"preserved": True}}
    constructor = MagicMock(return_value=client)
    monkeypatch.setattr(app, "DefaultAzureCredential", credentials)
    monkeypatch.setattr(app, "InferenceClient", constructor)
    return constructor, client, credentials, aml_credential, onelake_credential


def test_text_summary_uses_sdk_and_does_not_access_onelake(app, sdk):
    constructor, client, credentials, aml, storage = sdk
    result = app.summarize("Paste text", "  Whole document.\n", "https://unused.example/file", 40)
    assert result == ("A concise summary.", {"summary": "A concise summary."})
    constructor.assert_called_once_with(
        app.AML_ENDPOINT,
        aml,
        credential_scopes=["https://ml.azure.com/.default"],
        connection_timeout=10,
        read_timeout=600,
        retry_total=0,
    )
    client.summarize.assert_called_once_with({"text": "  Whole document.\n", "max_words": 40})
    credentials.assert_called_once()
    storage.__enter__.assert_not_called()
    aml.__exit__.assert_called_once()
    client.__exit__.assert_called_once()


def test_onelake_summary_uses_fixed_server_settings_and_separate_credential(app, sdk):
    _, client, credentials, aml, storage = sdk
    result = app.summarize("OneLake document", "Ignore the hidden text field.", app.DOCUMENT_URL, 100)
    assert result[0] == "A concise summary."
    client.summarize.assert_called_once_with(
        {
            "documentUrl": app.DOCUMENT_URL,
            "onelakeEndpoint": app.ONELAKE_ENDPOINT,
            "onelakeWorkspaceId": app.WORKSPACE_ID,
            "onelakeLakehouseId": app.LAKEHOUSE_ID,
            "max_words": 100,
        },
        onelake_credential=storage,
    )
    assert credentials.call_count == 2
    assert aml is not storage
    aml.__exit__.assert_called_once()
    storage.__exit__.assert_called_once()
    client.__exit__.assert_called_once()


def test_demo_does_not_display_service_diagnostics(app, sdk):
    _, client, _, _, _ = sdk
    client.summarize.return_value = {
        "summary": "Only the summary is displayed.",
        "latency_ms": 1234,
        "metrics": {"elapsed_seconds": 1.234},
    }
    assert app.summarize("Paste text", "Document contents.", "", 100) == (
        "Only the summary is displayed.",
        {"summary": "Only the summary is displayed."},
    )


def test_busy_button_uses_text_without_performance_metrics(app):
    summary, response, busy = app.start_summarization()
    assert (summary, response) == ("", {})
    assert busy["value"] == "Generating summary..."
    assert busy["interactive"] is False
    ready = app.finish_summarization()
    assert ready["value"] == "Generate summary"
    assert ready["interactive"] is True


@pytest.mark.parametrize("value", [0, -1, 1.5, True, None, "100"])
def test_invalid_word_limit_never_constructs_clients(app, sdk, value):
    constructor, _, credentials, _, _ = sdk
    with pytest.raises(gr.Error, match="positive integer"):
        app.summarize("Paste text", "document", "", value)
    constructor.assert_not_called()
    credentials.assert_not_called()


def test_unknown_source_is_explicit_error(app, sdk):
    constructor, _, credentials, _, _ = sdk
    with pytest.raises(gr.Error, match="Choose"):
        app.summarize("Untrusted source", "document", "", 100)
    constructor.assert_not_called()
    credentials.assert_not_called()


@pytest.mark.parametrize(
    "error,message",
    [
        (ClientAuthenticationError("secret backend details"), "Sign-in or access was denied"),
        (ResourceNotFoundError("secret backend details"), "not found"),
        (DecodeError("secret backend details"), "invalid response"),
        (HttpResponseError("secret backend details"), "service rejected"),
        (ServiceRequestError("secret backend details"), "failed or timed out"),
        (ServiceResponseError("secret backend details"), "failed or timed out"),
        (UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid"), "valid UTF-8"),
        (ValueError("Invalid OneLake file URL."), "Invalid OneLake file URL"),
        (ImportError("missing storage"), "OneLake support is missing"),
    ],
)
def test_errors_are_explicit_and_clients_are_closed(app, sdk, error, message):
    _, client, _, aml, storage = sdk
    client.summarize.side_effect = error
    with pytest.raises(gr.Error, match=message) as caught:
        app.summarize("OneLake document", "", app.DOCUMENT_URL, 100)
    assert "secret backend details" not in str(caught.value)
    assert caught.value.print_exception is False
    client.__exit__.assert_called_once()
    aml.__exit__.assert_called_once()
    storage.__exit__.assert_called_once()


def test_service_status_is_visible_without_response_body(app, sdk):
    _, client, _, _, _ = sdk
    error = HttpResponseError("sensitive response body")
    error.status_code = 429
    client.summarize.side_effect = error
    with pytest.raises(gr.Error, match="HTTP 429") as caught:
        app.summarize("Paste text", "document", "", 100)
    assert "sensitive response body" not in str(caught.value)


def test_real_sdk_constructs_the_expected_qwen_request(app, monkeypatch, token_credential, transport, respond):
    credential = MagicMock()
    credential.__enter__.return_value = token_credential
    monkeypatch.setattr(app, "DefaultAzureCredential", MagicMock(return_value=credential))

    def client_factory(endpoint, supplied_credential, **kwargs):
        return InferenceClient(endpoint, supplied_credential, transport=transport, **kwargs)

    monkeypatch.setattr(app, "InferenceClient", client_factory)
    respond((200, {"summary": "A real SDK response."}, {}))
    summary, response = app.summarize("Paste text", "Document contents.", "", 75)
    assert summary == "A real SDK response."
    assert response == {"summary": "A real SDK response."}
    sent = transport.send.call_args.args[0]
    assert sent.url == app.AML_ENDPOINT
    assert json.loads(sent.content) == {"text": "Document contents.", "max_words": 75}
    assert token_credential.get_token.call_args.args == ("https://ml.azure.com/.default",)


def test_real_sdk_rejects_outside_onelake_url_before_any_network_call(app, monkeypatch, token_credential, transport):
    credential = MagicMock()
    credential.__enter__.return_value = token_credential
    monkeypatch.setattr(app, "DefaultAzureCredential", MagicMock(return_value=credential))

    def client_factory(endpoint, supplied_credential, **kwargs):
        return InferenceClient(endpoint, supplied_credential, transport=transport, **kwargs)

    monkeypatch.setattr(app, "InferenceClient", client_factory)
    with pytest.raises(gr.Error, match="request's OneLake endpoint"):
        app.summarize("OneLake document", "", "https://untrusted.example/document.txt", 100)
    transport.send.assert_not_called()
    token_credential.get_token.assert_not_called()


@pytest.mark.parametrize(
    "source,text_visible,url_visible", [("Paste text", True, False), ("OneLake document", False, True)]
)
def test_input_selection(app, source, text_visible, url_visible):
    text, url = app.change_source(source)
    assert text["visible"] is text_visible
    assert url["visible"] is url_visible
    assert app.clear_outputs() == ("", {})


def test_ui_build_does_not_authenticate_or_infer(app, sdk):
    constructor, _, credentials, _, _ = sdk
    demo = app.create_demo()
    try:
        config = demo.get_config_file()
        assert config["title"] == app.DEMO_TITLE == "Inference Service - Summarization API"
        assert demo.analytics_enabled is False
        assert all(dependency["api_visibility"] == "private" for dependency in config["dependencies"])
        assert all(dependency["show_progress"] == "hidden" for dependency in config["dependencies"])
        restore_button = [
            dependency
            for dependency in config["dependencies"]
            if dependency["api_name"].startswith("finish_summarization")
        ]
        assert {
            (dependency["trigger_only_on_success"], dependency["trigger_only_on_failure"])
            for dependency in restore_button
        } == {(True, False), (False, True)}
        props = {
            component["props"].get("label"): component["props"]
            for component in config["components"]
            if component["props"].get("label")
        }
        assert props["Input source"]["value"] == "Paste text"
        assert props["Document text"]["value"] == app.EXAMPLE_TEXT
        assert props["OneLake file URL"]["visible"] is False
        assert props["OneLake file URL"]["value"] == app.DOCUMENT_URL
        assert props["Maximum summary words"]["value"] == 100
        assert props["Summary"]["interactive"] is False
        assert props["Summary"]["placeholder"] == "Your summary will appear here."
        assert not any("time" in label.lower() or "latency" in label.lower() for label in props)
        assert props["Qwen3 endpoint"]["interactive"] is False
        static_copy = " ".join(
            component["props"].get("value", "")
            for component in config["components"]
            if component["type"] in ("markdown", "html")
        )
        assert "DefaultAzureCredential" not in static_copy
        assert "anonymous remote access" not in static_copy
        assert "latency" not in static_copy.lower()
        header = next(
            component for component in config["components"] if component["props"].get("elem_id") == "demo-header"
        )
        assert f"<h1>{app.DEMO_TITLE}</h1>" in header["props"]["value"]
        constructor.assert_not_called()
        credentials.assert_not_called()
    finally:
        demo.close()


def test_launch_is_local_only_and_does_not_expose_files(app, monkeypatch):
    demo = MagicMock()
    monkeypatch.setattr(app, "create_demo", MagicMock(return_value=demo))
    monkeypatch.setenv("GRADIO_SERVER_NAME", "0.0.0.0")
    monkeypatch.setenv("GRADIO_SHARE", "true")
    app.main(["--port", "7861"])
    demo.queue.assert_called_once_with(default_concurrency_limit=1, max_size=8, api_open=False)
    options = demo.launch.call_args.kwargs
    assert options["server_name"] == "127.0.0.1"
    assert options["server_port"] == 7861
    assert options["share"] is False
    assert options["show_error"] is False
    assert options["run_history"] is False
    assert options["enable_monitoring"] is False
    assert options["mcp_server"] is False
    assert options["strict_cors"] is True
    assert options["allowed_paths"] == []
    assert options["blocked_paths"] == [str(Path(__file__).resolve().parents[4])]


@pytest.mark.parametrize("port", ["0", "-1", "65536"])
def test_invalid_port_does_not_start_server(app, monkeypatch, port):
    create = MagicMock()
    monkeypatch.setattr(app, "create_demo", create)
    with pytest.raises(SystemExit):
        app.main(["--port", port])
    create.assert_not_called()
