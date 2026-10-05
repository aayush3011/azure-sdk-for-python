# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Offline coverage of the handwritten summarization POC."""

import builtins
from copy import deepcopy
from io import BytesIO
import inspect
import json
from pathlib import Path
import runpy
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from azure.core import MatchConditions
from azure.core.credentials import AccessToken, AzureKeyCredential, TokenCredential
from azure.core.exceptions import (
    ClientAuthenticationError,
    DecodeError,
    HttpResponseError,
    ResourceNotFoundError,
    ServiceRequestError,
    ServiceResponseError,
)
from azure.core.pipeline.transport import RequestsTransport
from azure.core.utils import case_insensitive_dict

from azure.data.ai import InferenceClient
from azure.data.ai._client import InferenceClient as GeneratedInferenceClient
from azure.data.ai.aio import InferenceClient as AsyncInferenceClient


AML_ENDPOINT = "https://example.eastus2.inference.ml.azure.com"
AML_SCOPE = "https://ml.azure.com/.default"
ONELAKE_ENDPOINT = "https://daily-onelake.dfs.fabric.microsoft.com"
WORKSPACE_ID = "11111111-1111-1111-1111-111111111111"
LAKEHOUSE_ID = "22222222-2222-2222-2222-222222222222"
DOCUMENT_ROOT = f"{ONELAKE_ENDPOINT}/{WORKSPACE_ID}/{LAKEHOUSE_ID}/Files"


@pytest.fixture
def aml_credential(token_credential):
    credential = MagicMock(spec=TokenCredential)
    credential.get_token.return_value = AccessToken(
        "aml-test-token", token_credential.get_token.return_value.expires_on
    )
    return credential


@pytest.fixture
def aml_client(transport, aml_credential):
    def create(endpoint=AML_ENDPOINT, **kwargs):
        kwargs.setdefault("credential_scopes", [AML_SCOPE])
        kwargs.setdefault("retry_total", 0)
        return InferenceClient(endpoint, aml_credential, transport=transport, **kwargs)

    return create


@pytest.fixture
def document_request():
    return {
        "documentUrl": f"{DOCUMENT_ROOT}/design_patterns.md",
        "onelakeEndpoint": ONELAKE_ENDPOINT,
        "onelakeWorkspaceId": WORKSPACE_ID,
        "onelakeLakehouseId": LAKEHOUSE_ID,
    }


@pytest.fixture
def onelake_file(monkeypatch):
    from azure.storage.filedatalake import DataLakeFileClient

    file_client = MagicMock(spec=DataLakeFileClient)
    file_client.__enter__.return_value = file_client
    content = b"\xef\xbb\xbf# Design patterns\n\nA caf\xc3\xa9 example.\n"
    file_client.get_file_properties.return_value = SimpleNamespace(
        size=len(content), etag='"test-etag"', content_settings=SimpleNamespace(content_encoding=None)
    )
    file_client.download_file.return_value.chunks.return_value = [content[:1], content[1:8], content[8:]]
    factory = MagicMock(return_value=file_client)
    monkeypatch.setattr("azure.storage.filedatalake.DataLakeFileClient", factory)
    return factory, file_client, content


def test_constructor_and_reranking_are_inherited_unchanged():
    assert InferenceClient is GeneratedInferenceClient
    assert InferenceClient.__init__ is GeneratedInferenceClient.__init__
    assert inspect.signature(InferenceClient) == inspect.signature(GeneratedInferenceClient)
    assert InferenceClient.semantic_rerank is GeneratedInferenceClient.semantic_rerank
    assert InferenceClient.close is GeneratedInferenceClient.close
    assert not hasattr(AsyncInferenceClient, "summarize")


def test_summarize_is_defined_on_the_generated_operations_mixin():
    from azure.data.ai._operations._operations import _InferenceClientOperationsMixin

    assert InferenceClient.summarize is _InferenceClientOperationsMixin.summarize
    assert GeneratedInferenceClient.summarize is _InferenceClientOperationsMixin.summarize
    assert InferenceClient.summarize.__module__ == "azure.data.ai._operations._operations"


def test_text_round_trip_preserves_input_and_response(transport, respond, aml_client):
    request = {"text": "  The complete caf\u00e9 document.\n"}
    original = deepcopy(request)
    payload = {"summary": "A summary.", "futureMetadata": {"tokens": 7}}
    respond((200, payload, {}))
    with aml_client() as client:
        result = client.summarize(request)
    sent = transport.send.call_args.args[0]
    assert sent.method == "POST"
    assert sent.url == f"{AML_ENDPOINT}/summarize"
    assert sent.headers["Authorization"] == "Bearer aml-test-token"
    assert "Ocp-Apim-Subscription-Key" not in sent.headers
    assert sent.headers["Content-Type"] == "application/json"
    assert sent.headers["Accept"] == "application/json"
    assert json.loads(sent.content) == {"text": original["text"], "max_words": 100}
    assert request == original
    assert result == payload
    assert isinstance(result, dict)


def test_aml_ignores_inference_api_version_and_preserves_options(transport, respond, aml_client, aml_credential):
    respond((200, {"summary": "ok"}, {"X-Correlation-ID": "summarization-correlation"}))
    responses = []
    with aml_client(f"{AML_ENDPOINT}/summarize", api_version="poc-version") as client:
        client.summarize(
            {"text": "document"},
            connection_timeout=2,
            read_timeout=3,
            headers={"x-test-header": "inference-only"},
            raw_response_hook=responses.append,
        )
    sent = transport.send.call_args.args[0]
    assert sent.url == f"{AML_ENDPOINT}/summarize"
    assert sent.headers["Authorization"] == "Bearer aml-test-token"
    assert sent.headers["x-test-header"] == "inference-only"
    assert transport.send.call_args.kwargs["connection_timeout"] == 2
    assert transport.send.call_args.kwargs["read_timeout"] == 3
    assert aml_credential.get_token.call_args.args == (AML_SCOPE,)
    assert responses[0].http_response.headers["X-Correlation-ID"] == "summarization-correlation"


@pytest.mark.parametrize("suffix", ["", "/", "/summarize", "/summarize/"])
def test_aml_base_and_summarization_urls_do_not_duplicate_path(transport, respond, aml_client, suffix):
    respond((200, {"summary": "ok"}, {}))
    with aml_client(AML_ENDPOINT + suffix) as client:
        client.summarize({"text": "document"})
    assert transport.send.call_args.args[0].url == f"{AML_ENDPOINT}/summarize"


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://example.inference.azure.com",
        "https://example.eastus2.inference.ml.azure.com.evil.example/summarize",
        "https://user@example.eastus2.inference.ml.azure.com/summarize",
        "https://example.eastus2.inference.ml.azure.com:8443/summarize",
        "http://example.eastus2.inference.ml.azure.com/summarize",
        f"{AML_ENDPOINT}/base",
        f"{AML_ENDPOINT}/score",
        f"{AML_ENDPOINT}/score/",
        f"{AML_ENDPOINT}/summarize/summarize",
        f"{AML_ENDPOINT}/summarize?key=not-a-real-key",
        f"{AML_ENDPOINT}/summarize#fragment",
    ],
)
def test_wrong_aml_endpoint_fails_before_download(
    transport, aml_client, token_credential, document_request, onelake_file, endpoint
):
    factory, _, _ = onelake_file
    with aml_client(endpoint) as client:
        with pytest.raises(ValueError):
            client.summarize(document_request, onelake_credential=token_credential)
    factory.assert_not_called()
    transport.send.assert_not_called()


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"credential_scopes": ["https://dbinference.azure.com/.default"]},
        {"credential_scopes": ["https://storage.azure.com/.default"]},
        {"credential_scopes": [AML_SCOPE, "https://storage.azure.com/.default"]},
    ],
)
def test_wrong_aml_scope_fails_before_token_or_document_access(
    transport, aml_credential, token_credential, document_request, onelake_file, options
):
    factory, _, _ = onelake_file
    with InferenceClient(AML_ENDPOINT, aml_credential, transport=transport, **options) as client:
        with pytest.raises(ValueError, match="credential_scopes"):
            client.summarize(document_request, onelake_credential=token_credential)
    aml_credential.get_token.assert_not_called()
    factory.assert_not_called()
    transport.send.assert_not_called()


def test_aml_rejects_key_and_async_credentials_before_document_access(
    transport, token_credential, async_token_credential, document_request, onelake_file
):
    factory, _, _ = onelake_file
    for credential in (AzureKeyCredential("test-key"), async_token_credential):
        with InferenceClient(AML_ENDPOINT, credential, credential_scopes=[AML_SCOPE], transport=transport) as client:
            with pytest.raises(TypeError, match="synchronous TokenCredential"):
                client.summarize(document_request, onelake_credential=token_credential)
    factory.assert_not_called()
    transport.send.assert_not_called()


@pytest.mark.parametrize("max_words", [0, -1, True, False, 1.5, "100", None, float("nan"), float("inf")])
def test_invalid_max_words_fails_before_download(
    transport, aml_client, token_credential, document_request, onelake_file, max_words
):
    factory, _, _ = onelake_file
    with aml_client() as client:
        with pytest.raises(ValueError, match="max_words"):
            client.summarize({**document_request, "max_words": max_words}, onelake_credential=token_credential)
    factory.assert_not_called()
    transport.send.assert_not_called()


@pytest.mark.parametrize("options", [{"task": "summarization"}, {"max_new_tokens": 128}, {"temperature": 0.0}])
def test_legacy_scoring_options_are_rejected(
    transport, aml_client, token_credential, document_request, onelake_file, options
):
    factory, _, _ = onelake_file
    with aml_client() as client:
        with pytest.raises(ValueError, match="Supported request fields"):
            client.summarize({**document_request, **options}, onelake_credential=token_credential)
    factory.assert_not_called()
    transport.send.assert_not_called()


@pytest.mark.parametrize("use_document", [False, True])
@pytest.mark.parametrize("max_words", [1, 40])
def test_max_words_is_request_scoped(
    transport, respond, aml_client, token_credential, document_request, onelake_file, use_document, max_words
):
    _, _, content = onelake_file
    request = {
        **(document_request if use_document else {"text": "document"}),
        "max_words": max_words,
    }
    original = deepcopy(request)
    respond((200, {"summary": "first"}, {}), (200, {"summary": "second"}, {}))
    with aml_client() as client:
        client.summarize(request, onelake_credential=token_credential)
        first = json.loads(transport.send.call_args.args[0].content)
        client.summarize({"text": "Next document"})
        second = json.loads(transport.send.call_args.args[0].content)
    assert first == {
        "text": content.decode("utf-8-sig") if use_document else "document",
        "max_words": max_words,
    }
    assert second == {"text": "Next document", "max_words": 100}
    assert request == original


@pytest.mark.parametrize("payload", [{}, {"generated_text": "wrong contract"}, {"summary": None}, {"summary": 42}])
def test_response_requires_summary(respond, aml_client, payload):
    respond((200, payload, {}))
    with aml_client() as client:
        with pytest.raises(DecodeError, match="summary"):
            client.summarize({"text": "document"})


@pytest.mark.parametrize(
    "input_request,error",
    [
        (None, TypeError),
        ([], TypeError),
        ({}, ValueError),
        ({"text": "one", "documentUrl": "two"}, ValueError),
        ({"text": "", "documentUrl": None}, ValueError),
        ({"text": ""}, ValueError),
        ({"text": None}, ValueError),
        ({"text": 42}, ValueError),
        ({"text": " \n\t"}, ValueError),
        ({"text": "document", "unexpected": True}, ValueError),
        ({"text": "document", "onelakeEndpoint": ONELAKE_ENDPOINT}, ValueError),
    ],
)
def test_invalid_requests_do_not_send(transport, aml_client, input_request, error):
    with aml_client() as client:
        with pytest.raises(error):
            client.summarize(input_request)
    transport.send.assert_not_called()


@pytest.mark.parametrize("limit", [0, -1, True, 1.5, "1024", None])
def test_invalid_limits(transport, aml_client, limit):
    with aml_client() as client:
        with pytest.raises(ValueError, match="positive integer"):
            client.summarize({"text": "document"}, max_input_bytes=limit)
    transport.send.assert_not_called()


def test_utf8_limit_is_in_bytes_not_characters(transport, respond, aml_client):
    respond((200, {"summary": "ok"}, {}))
    with aml_client() as client:
        with pytest.raises(ValueError, match="max_input_bytes"):
            client.summarize({"text": "\u00e9"}, max_input_bytes=1)
        transport.send.assert_not_called()
        assert client.summarize({"text": "\u00e9"}, max_input_bytes=2) == {"summary": "ok"}


def test_default_limit_is_one_mebibyte(transport, respond, aml_client):
    respond((200, {"summary": "ok"}, {}))
    with aml_client() as client:
        with pytest.raises(ValueError, match="max_input_bytes"):
            client.summarize({"text": "x" * (1024 * 1024 + 1)})
        transport.send.assert_not_called()
        client.summarize({"text": "x" * (1024 * 1024)})


def test_invalid_unicode_and_streaming_do_not_send(transport, aml_client):
    with aml_client() as client:
        with pytest.raises(UnicodeEncodeError):
            client.summarize({"text": "\ud800"})
        with pytest.raises(ValueError, match="Streaming"):
            client.summarize({"text": "document"}, stream=True)
    transport.send.assert_not_called()


@pytest.mark.parametrize(
    "status,error",
    [
        (400, HttpResponseError),
        (401, ClientAuthenticationError),
        (403, ClientAuthenticationError),
        (404, ResourceNotFoundError),
        (429, HttpResponseError),
        (500, HttpResponseError),
    ],
)
def test_aml_errors_are_preserved(transport, respond, aml_client, status, error):
    payload = {"error": {"code": "TestError", "message": "The request failed."}}
    respond((status, payload, {"Retry-After": "3"}))
    with aml_client() as client:
        with pytest.raises(error) as caught:
            client.summarize({"text": "document"})
    assert caught.value.status_code == status
    assert caught.value.response.json() == payload
    assert caught.value.response.headers["Retry-After"] == "3"


@pytest.mark.parametrize("status", [201, 204, 302, 307, 308])
def test_unexpected_status_and_redirects_do_not_succeed(transport, respond, aml_client, status):
    respond((status, {}, {"Location": "https://untrusted.example/collect"}))
    with aml_client() as client:
        with pytest.raises(HttpResponseError):
            client.summarize({"text": "document"}, permit_redirects=True)
    transport.send.assert_called_once()
    assert transport.send.call_args.args[0].url.startswith(AML_ENDPOINT)


@pytest.mark.parametrize("payload", [None, "summary", [], 42])
def test_non_object_response_is_an_error(transport, respond, aml_client, payload):
    respond((200, payload, {}))
    with aml_client() as client:
        with pytest.raises(DecodeError, match="JSON object"):
            client.summarize({"text": "document"})


def test_malformed_json_response_is_an_error(transport, respond, aml_client):
    respond((200, {}, {}))

    def malformed(response):
        response.http_response.json.side_effect = ValueError("Not JSON")

    with aml_client() as client:
        with pytest.raises(DecodeError, match="not valid JSON"):
            client.summarize({"text": "document"}, raw_response_hook=malformed)


def test_aml_network_errors_are_not_hidden(transport, aml_client):
    error = ServiceRequestError("Timed out")
    transport.send.side_effect = error
    with aml_client() as client:
        with pytest.raises(ServiceRequestError) as caught:
            client.summarize({"text": "document"})
    assert caught.value is error


def test_onelake_download_is_bounded_and_request_scoped(
    transport, respond, aml_client, token_credential, document_request, onelake_file
):
    factory, file_client, content = onelake_file
    original = deepcopy(document_request)
    respond((200, {"summary": "ok"}, {}), (200, {"summary": "direct"}, {}))
    with aml_client() as client:
        assert client.summarize(
            document_request,
            onelake_credential=token_credential,
            max_input_bytes=len(content),
            onelake_connection_timeout=2,
            onelake_read_timeout=3,
            headers={"x-test-header": "inference-only"},
        ) == {"summary": "ok"}
        sent = transport.send.call_args.args[0]
        assert json.loads(sent.content) == {"text": content.decode("utf-8-sig"), "max_words": 100}
        assert sent.headers["Authorization"] == "Bearer aml-test-token"
        assert client.summarize({"text": "Next document"}) == {"summary": "direct"}
    assert document_request == original
    factory.assert_called_once_with(
        account_url=ONELAKE_ENDPOINT,
        file_system_name=WORKSPACE_ID,
        file_path=f"{LAKEHOUSE_ID}/Files/design_patterns.md",
        credential=token_credential,
        audience="https://storage.azure.com/",
        connection_timeout=2,
        read_timeout=3,
        retry_total=3,
        retry_to_secondary=False,
        permit_redirects=False,
        max_single_get_size=len(content),
        max_chunk_get_size=len(content),
    )
    file_client.download_file.assert_called_once_with(
        offset=0,
        length=len(content),
        etag='"test-etag"',
        match_condition=MatchConditions.IfNotModified,
        max_concurrency=1,
        decompress=False,
    )
    file_client.__exit__.assert_called_once()


@pytest.mark.parametrize(
    "document_url",
    [
        "https://daily.fabric.microsoft.com/groups/workspace/lakehouses/item",
        f"http://daily-onelake.dfs.fabric.microsoft.com/{WORKSPACE_ID}/{LAKEHOUSE_ID}/Files/file.txt",
        f"https://user@daily-onelake.dfs.fabric.microsoft.com/{WORKSPACE_ID}/{LAKEHOUSE_ID}/Files/file.txt",
        f"https://daily-onelake.dfs.fabric.microsoft.com:8443/{WORKSPACE_ID}/{LAKEHOUSE_ID}/Files/file.txt",
        f"https://daily-onelake.dfs.fabric.microsoft.com.evil.example/{WORKSPACE_ID}/{LAKEHOUSE_ID}/Files/file.txt",
        f"{DOCUMENT_ROOT}/file.txt?sig=not-a-real-token",
        f"{DOCUMENT_ROOT}/file.txt#fragment",
        f"{DOCUMENT_ROOT}/file.pdf",
        f"{DOCUMENT_ROOT}/../file.txt",
        f"{DOCUMENT_ROOT}/%2e%2e/file.txt",
        f"{DOCUMENT_ROOT}/%252e%252e/file.txt",
        f"{DOCUMENT_ROOT}/%2f..%2ffile.txt",
        f"{DOCUMENT_ROOT}//file.txt",
        f"{DOCUMENT_ROOT}/./file.txt",
        f"{DOCUMENT_ROOT}/file.txt/",
        f"{DOCUMENT_ROOT}/file\\name.txt",
        f"{DOCUMENT_ROOT}/file%5cname.txt",
        f"{DOCUMENT_ROOT}/file%00name.txt",
        f"{DOCUMENT_ROOT}/file%2name.txt",
        f"{DOCUMENT_ROOT}/file%name.txt",
        f"{DOCUMENT_ROOT}/file name.txt",
        f"{DOCUMENT_ROOT}/file\nname.txt",
        f"{ONELAKE_ENDPOINT}/{WORKSPACE_ID}/{LAKEHOUSE_ID}/Tables/file.txt",
        f"{ONELAKE_ENDPOINT}/{WORKSPACE_ID}/{WORKSPACE_ID}/Files/file.txt",
        f"{ONELAKE_ENDPOINT}/{LAKEHOUSE_ID}/{LAKEHOUSE_ID}/Files/file.txt",
        f"abfss://{WORKSPACE_ID}@daily-onelake.dfs.fabric.microsoft.com/{LAKEHOUSE_ID}/Files/file.txt",
    ],
)
def test_unsafe_document_urls_are_rejected(
    transport, aml_client, token_credential, document_request, onelake_file, document_url
):
    factory, _, _ = onelake_file
    document_request["documentUrl"] = document_url
    with aml_client() as client:
        with pytest.raises(ValueError):
            client.summarize(document_request, onelake_credential=token_credential)
    factory.assert_not_called()
    transport.send.assert_not_called()


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://untrusted.example",
        "http://onelake.dfs.fabric.microsoft.com",
        "https://onelake.dfs.fabric.microsoft.com.evil.example",
        "https://evil.onelake.dfs.fabric.microsoft.com",
        f"{ONELAKE_ENDPOINT}/extra",
        f"{ONELAKE_ENDPOINT}?query=value",
        f"{ONELAKE_ENDPOINT}#fragment",
    ],
)
def test_unsafe_configured_endpoints_are_rejected(
    transport, aml_client, token_credential, document_request, onelake_file, endpoint
):
    factory, _, _ = onelake_file
    document_request["onelakeEndpoint"] = endpoint
    document_request["documentUrl"] = f"{endpoint}/{WORKSPACE_ID}/{LAKEHOUSE_ID}/Files/file.txt"
    with aml_client() as client:
        with pytest.raises(ValueError):
            client.summarize(document_request, onelake_credential=token_credential)
    factory.assert_not_called()
    transport.send.assert_not_called()


@pytest.mark.parametrize("field", ["onelakeEndpoint", "onelakeWorkspaceId", "onelakeLakehouseId"])
@pytest.mark.parametrize("value", [None, "", "not-valid"])
def test_required_onelake_configuration(transport, aml_client, token_credential, document_request, field, value):
    document_request[field] = value
    with aml_client() as client:
        with pytest.raises(ValueError):
            client.summarize(document_request, onelake_credential=token_credential)
    transport.send.assert_not_called()


@pytest.mark.parametrize("host", ["onelake", "daily-onelake", "westus-onelake"])
@pytest.mark.parametrize("file_name", ["file.txt", "design_patterns.md", "folder/caf%C3%A9%20notes.MD"])
def test_supported_onelake_hosts_and_paths(
    transport, respond, aml_client, token_credential, document_request, onelake_file, host, file_name
):
    factory, _, _ = onelake_file
    endpoint = f"https://{host}.dfs.fabric.microsoft.com"
    document_request["onelakeEndpoint"] = endpoint
    document_request["documentUrl"] = f"{endpoint}/{WORKSPACE_ID}/{LAKEHOUSE_ID}/Files/{file_name}"
    respond((200, {"summary": "ok"}, {}))
    with aml_client() as client:
        client.summarize(document_request, onelake_credential=token_credential)
    assert factory.call_args.kwargs["account_url"] == endpoint
    expected_name = file_name.replace("caf%C3%A9%20notes", "caf\u00e9 notes")
    assert factory.call_args.kwargs["file_path"] == f"{LAKEHOUSE_ID}/Files/{expected_name}"


def test_document_requires_a_separate_sync_token_credential(
    transport, aml_client, document_request, async_token_credential
):
    with aml_client() as client:
        with pytest.raises(ValueError, match="onelake_credential"):
            client.summarize(document_request)
        for credential in (AzureKeyCredential("storage-key"), "raw-token", async_token_credential):
            with pytest.raises(TypeError, match="synchronous TokenCredential"):
                client.summarize(document_request, onelake_credential=credential)
    transport.send.assert_not_called()


@pytest.mark.parametrize("option", ["onelake_connection_timeout", "onelake_read_timeout"])
@pytest.mark.parametrize("value", [0, -1, float("inf"), float("nan"), True, "60"])
def test_invalid_storage_timeouts(transport, aml_client, token_credential, document_request, option, value):
    with aml_client() as client:
        with pytest.raises(ValueError, match=option):
            client.summarize(document_request, onelake_credential=token_credential, **{option: value})
    transport.send.assert_not_called()


@pytest.mark.parametrize("size", [0, 5])
def test_empty_or_oversized_files_are_rejected_before_download(
    transport, aml_client, token_credential, document_request, onelake_file, size
):
    _, file_client, _ = onelake_file
    file_client.get_file_properties.return_value.size = size
    with aml_client() as client:
        with pytest.raises(ValueError):
            client.summarize(document_request, onelake_credential=token_credential, max_input_bytes=4)
    file_client.download_file.assert_not_called()
    file_client.__exit__.assert_called_once()
    transport.send.assert_not_called()


@pytest.mark.parametrize("data", [b"\xff\xfe", b" \r\n", b"\xef\xbb\xbf"])
def test_invalid_or_blank_document_content(
    transport, aml_client, token_credential, document_request, onelake_file, data
):
    _, file_client, _ = onelake_file
    file_client.get_file_properties.return_value.size = len(data)
    file_client.download_file.return_value.chunks.return_value = [data]
    with aml_client() as client:
        with pytest.raises(ValueError):
            client.summarize(document_request, onelake_credential=token_credential)
    file_client.__exit__.assert_called_once()
    transport.send.assert_not_called()


@pytest.mark.parametrize(
    "data,error", [(b"12", ServiceResponseError), (b"1234", ServiceResponseError), (b"12345", ValueError)]
)
def test_download_size_is_checked_without_truncation(
    transport, aml_client, token_credential, document_request, onelake_file, data, error
):
    _, file_client, _ = onelake_file
    file_client.get_file_properties.return_value.size = 3
    file_client.download_file.return_value.chunks.return_value = [data]
    with aml_client() as client:
        with pytest.raises(error):
            client.summarize(document_request, onelake_credential=token_credential, max_input_bytes=4)
    file_client.__exit__.assert_called_once()
    transport.send.assert_not_called()


@pytest.mark.parametrize("missing_etag", [False, True])
def test_compression_and_missing_etag_fail_before_download(
    transport, aml_client, token_credential, document_request, onelake_file, missing_etag
):
    _, file_client, _ = onelake_file
    properties = file_client.get_file_properties.return_value
    if missing_etag:
        properties.etag = None
        error = ServiceResponseError
    else:
        properties.content_settings.content_encoding = "gzip"
        error = ValueError
    with aml_client() as client:
        with pytest.raises(error):
            client.summarize(document_request, onelake_credential=token_credential)
    file_client.download_file.assert_not_called()
    transport.send.assert_not_called()


@pytest.mark.parametrize("stage", ["get_file_properties", "download_file", "chunks"])
@pytest.mark.parametrize(
    "error",
    [
        ResourceNotFoundError("File not found"),
        HttpResponseError("Forbidden", status_code=403),
        HttpResponseError("File changed", status_code=412),
        ServiceRequestError("Connection timed out"),
        ServiceResponseError("Read timed out"),
    ],
)
def test_storage_failures_do_not_call_aml(
    transport, aml_client, token_credential, document_request, onelake_file, stage, error
):
    _, file_client, _ = onelake_file
    if stage == "chunks":
        file_client.download_file.return_value.chunks.side_effect = error
    else:
        getattr(file_client, stage).side_effect = error
    with aml_client() as client:
        with pytest.raises(type(error)) as caught:
            client.summarize(document_request, onelake_credential=token_credential)
    assert caught.value is error
    file_client.__exit__.assert_called_once()
    transport.send.assert_not_called()


def test_optional_dependency_is_needed_only_for_document_requests(
    monkeypatch, transport, respond, aml_client, token_credential, document_request
):
    original_import = builtins.__import__

    def without_storage(name, *args, **kwargs):
        if name.startswith("azure.storage"):
            raise ModuleNotFoundError("No module named 'azure.storage'", name="azure.storage")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_storage)
    respond((200, {"summary": "ok"}, {}), (200, {"scores": []}, {}))
    with aml_client() as client:
        assert client.summarize({"text": "document"}) == {"summary": "ok"}
        with pytest.raises(ImportError, match=r"azure-data-ai\[onelake\]"):
            client.summarize(document_request, onelake_credential=token_credential)
    with InferenceClient(
        "https://example.inference.azure.com", AzureKeyCredential("test-key"), transport=transport
    ) as reranker:
        assert reranker.semantic_rerank({"query": "test", "documents": ["document"]}) == {"scores": []}
    assert transport.send.call_count == 2


def test_broken_transitive_storage_dependency_is_not_masked(
    monkeypatch, transport, aml_client, token_credential, document_request
):
    original_import = builtins.__import__

    def broken_storage(name, *args, **kwargs):
        if name == "azure.storage.filedatalake":
            raise ModuleNotFoundError("Missing transitive dependency", name="cryptography")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", broken_storage)
    with aml_client() as client:
        with pytest.raises(ModuleNotFoundError) as caught:
            client.summarize(document_request, onelake_credential=token_credential)
    assert caught.value.name == "cryptography"
    transport.send.assert_not_called()


def test_real_storage_client_uses_storage_scope_and_conditional_range(
    monkeypatch, transport, respond, aml_client, aml_credential, token_credential, document_request
):
    import requests
    from azure.storage.filedatalake import DataLakeFileClient

    content = b"\xef\xbb\xbf# A whole Markdown document.\n"
    session = requests.Session()
    close = MagicMock(wraps=session.close)
    monkeypatch.setattr(session, "close", close)

    def storage_response(method, url, **kwargs):
        response = requests.Response()
        response.status_code = 200 if method == "HEAD" else 206
        response.reason = "OK" if method == "HEAD" else "Partial Content"
        response.headers = case_insensitive_dict(
            {
                "Content-Type": "text/markdown",
                "Content-Length": str(len(content)),
                "ETag": '"actual-storage-etag"',
                "Last-Modified": "Thu, 01 Oct 2026 00:00:00 GMT",
                "x-ms-resource-type": "file",
            }
        )
        if method == "GET":
            response.headers["Content-Range"] = f"bytes 0-{len(content) - 1}/{len(content)}"
        response.url = url
        response.request = requests.Request(method, url, headers=kwargs["headers"]).prepare()
        response.raw = BytesIO(b"" if method == "HEAD" else content)
        return response

    send = MagicMock(side_effect=storage_response)
    monkeypatch.setattr(session, "request", send)

    def storage_client(**kwargs):
        return DataLakeFileClient(
            transport=RequestsTransport(
                session=session,
                connection_timeout=kwargs["connection_timeout"],
                read_timeout=kwargs["read_timeout"],
            ),
            **kwargs,
        )

    monkeypatch.setattr("azure.storage.filedatalake.DataLakeFileClient", storage_client)
    respond((200, {"summary": "ok"}, {}))
    with aml_client() as client:
        assert client.summarize(document_request, onelake_credential=token_credential) == {"summary": "ok"}
    assert send.call_count == 2
    head, get = send.call_args_list
    assert head.args[0] == "HEAD"
    storage_url = (
        f"https://daily-onelake.blob.fabric.microsoft.com/{WORKSPACE_ID}/{LAKEHOUSE_ID}/Files/design_patterns.md"
    )
    assert head.args[1].startswith(storage_url)
    assert get.args[0] == "GET"
    assert get.args[1].startswith(storage_url)
    headers = case_insensitive_dict(get.kwargs["headers"])
    assert headers["If-Match"] == '"actual-storage-etag"'
    assert headers.get("x-ms-range", headers.get("Range")) == f"bytes=0-{len(content) - 1}"
    assert headers["Authorization"] == "Bearer test-token"
    assert "Ocp-Apim-Subscription-Key" not in headers
    assert get.kwargs["timeout"] == (10, 60)
    assert token_credential.get_token.called
    assert all(
        call.args == ("https://storage.azure.com/.default",) for call in token_credential.get_token.call_args_list
    )
    assert aml_credential.get_token.call_args.args == (AML_SCOPE,)
    assert transport.send.call_args.args[0].headers["Authorization"] == "Bearer aml-test-token"
    assert json.loads(transport.send.call_args.args[0].content) == {
        "text": content.decode("utf-8-sig"),
        "max_words": 100,
    }
    assert close.called


def test_summarization_sample(monkeypatch, capsys):
    import azure.identity

    client = MagicMock()
    client.__enter__.return_value = client
    client.summarize.return_value = {"summary": "Sample summary."}
    constructor = MagicMock(return_value=client)
    monkeypatch.setattr("azure.data.ai.InferenceClient", constructor)
    inference_credential, storage_credential = MagicMock(), MagicMock()
    inference_credential.__enter__.return_value = inference_credential
    storage_credential.__enter__.return_value = storage_credential
    credentials = MagicMock(side_effect=[inference_credential, storage_credential])
    monkeypatch.setattr(azure.identity, "DefaultAzureCredential", credentials)
    runpy.run_path(str(Path(__file__).resolve().parents[1] / "samples" / "summarization.py"), run_name="__main__")
    constructor.assert_called_once()
    assert constructor.call_args.args[0] == "https://qwen3-summarization.eastus2.inference.ml.azure.com/summarize"
    assert constructor.call_args.args[1] is inference_credential
    assert constructor.call_args.kwargs == {
        "credential_scopes": [AML_SCOPE],
        "connection_timeout": 10,
        "read_timeout": 600,
    }
    client.summarize.assert_called_once()
    request = client.summarize.call_args.args[0]
    assert set(request) == {"documentUrl", "onelakeEndpoint", "onelakeWorkspaceId", "onelakeLakehouseId", "max_words"}
    assert request["documentUrl"].startswith(
        f"{request['onelakeEndpoint']}/{request['onelakeWorkspaceId']}/{request['onelakeLakehouseId']}/Files/"
    )
    assert request["max_words"] == 100
    assert client.summarize.call_args.kwargs == {"onelake_credential": storage_credential}
    assert credentials.call_count == 2
    storage_credential.__exit__.assert_called_once()
    assert capsys.readouterr().out == "Sample summary.\n"
    inference_credential.__exit__.assert_called_once()
    client.__exit__.assert_called_once()
