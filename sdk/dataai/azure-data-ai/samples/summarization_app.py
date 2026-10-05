# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Local Gradio demo of the Inference Service summarization API."""

import argparse
from pathlib import Path
from typing import Any, Optional

import gradio as gr
from azure.core.exceptions import (
    ClientAuthenticationError,
    DecodeError,
    HttpResponseError,
    ResourceNotFoundError,
    ServiceRequestError,
    ServiceResponseError,
)
from azure.data.ai import InferenceClient
from azure.identity import DefaultAzureCredential

DEMO_TITLE = "Inference Service - Summarization API"
AML_ENDPOINT = "https://qwen3-summarization.eastus2.inference.ml.azure.com/summarize"
ONELAKE_ENDPOINT = "https://daily-onelake.dfs.fabric.microsoft.com"
WORKSPACE_ID = "f2f58846-6636-4c9a-b901-c196f9f3d904"
LAKEHOUSE_ID = "38bde7a1-b24e-4d51-a8ad-269406b3c99c"
DOCUMENT_URL = f"{ONELAKE_ENDPOINT}/{WORKSPACE_ID}/{LAKEHOUSE_ID}/Files/design_patterns.md"
SOURCES = ["Paste text", "OneLake document"]
EXAMPLE_TEXT = (
    "Azure Machine Learning managed online endpoints provide scalable HTTPS endpoints for real-time model inference. "
    "Deployments define the model, environment, compute, and request configuration. "
    "Applications can authenticate with Microsoft Entra ID instead of storing API keys. "
    "Azure Data AI can download a document from OneLake and send its contents to a summarization model, "
    "without saving the document to local disk."
)
CSS = """
.gradio-container {
    width: 100% !important;
    max-width: 1240px !important;
    box-sizing: border-box !important;
    margin: 0 auto !important;
    padding: clamp(16px, 3vw, 28px) clamp(12px, 2vw, 24px) 40px !important;
}
html { scrollbar-gutter: stable; }
.gradio-container > .main { padding: 0 !important; }
#demo-header {
    padding: 28px 32px;
    margin-bottom: 8px;
    border: 1px solid #dbe7f7;
    border-radius: 20px;
    background: linear-gradient(115deg, #edf5ff, #f1f9ff 55%, #f6f3ff);
    box-shadow: 0 8px 28px rgba(30, 64, 175, .05);
}
#demo-header .eyebrow {
    margin: 0 0 10px;
    color: #2563eb;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: .14em;
    text-transform: uppercase;
}
#demo-header h1 {
    margin: 0;
    color: #0f172a;
    font-size: clamp(24px, 3vw, 34px);
    font-weight: 750;
    letter-spacing: -.035em;
    line-height: 1.25;
}
#demo-header .description { margin: 12px 0 18px; color: #52647c; font-size: 15px; line-height: 1.6; }
#demo-header .badges { display: flex; flex-wrap: wrap; gap: 8px; }
#demo-header .badge {
    padding: 5px 11px;
    border: 1px solid #d5e2f3;
    border-radius: 999px;
    background: rgba(255, 255, 255, .75);
    color: #334a68;
    font-size: 12px;
    font-weight: 600;
}
#workspace { gap: 20px; align-items: stretch; }
#input-panel, #output-panel {
    padding: 22px;
    border: 1px solid var(--border-color-primary);
    border-radius: 18px;
    background: var(--block-background-fill);
    box-shadow: 0 5px 20px rgba(15, 23, 42, .035);
}
.panel-heading h3 { margin: 0 0 4px !important; font-size: 19px !important; letter-spacing: -.02em; }
.panel-heading p { color: var(--body-text-color-subdued); font-size: 13px; }
#input-text textarea, #summary-output textarea { font-size: 15px; line-height: 1.7; }
#summary-output textarea { min-height: 290px; }
#summarize-button {
    min-height: 46px;
    border: 0;
    border-radius: 12px;
    background: linear-gradient(110deg, #2563eb, #087dcc);
    color: white;
    font-weight: 650;
    box-shadow: 0 4px 12px rgba(37, 99, 235, .18);
}
#summarize-button:hover { background: linear-gradient(110deg, #1d4ed8, #0369a1); }
#clear-button { min-height: 46px; border-radius: 12px; }
#demo-note p { color: var(--body-text-color-subdued); font-size: 12px; line-height: 1.6; }
#service-details { border-radius: 14px; }
.dark #demo-header {
    border-color: #263b58;
    background: linear-gradient(115deg, #132440, #122a38 55%, #242039);
}
.dark #demo-header .eyebrow { color: #93c5fd; }
.dark #demo-header h1 { color: #f1f5f9; }
.dark #demo-header .description { color: #b3c2d5; }
.dark #demo-header .badge { background: rgba(15, 23, 42, .4); border-color: #364760; color: #d8e6f6; }
@media (max-width: 700px) {
    #demo-header { padding: 22px 20px; }
    #workspace { flex-direction: column !important; }
    #input-panel, #output-panel { width: 100%; min-width: 0 !important; padding: 18px; }
}
"""


def clear_outputs() -> tuple[str, dict[str, Any]]:
    return "", {}


def start_summarization() -> tuple[str, dict[str, Any], dict[str, Any]]:
    return "", {}, gr.update(value="Generating summary...", interactive=False)


def finish_summarization() -> dict[str, Any]:
    return gr.update(value="Generate summary", interactive=True)


def change_source(source: str) -> tuple[dict[str, Any], dict[str, Any]]:
    if source not in SOURCES:
        raise gr.Error("Choose Paste text or OneLake document.", print_exception=False)
    return gr.update(visible=source == SOURCES[0]), gr.update(visible=source == SOURCES[1])


def summarize(source: str, text: str, document_url: str, max_words: int) -> tuple[str, dict[str, Any]]:
    """Call the SDK with server-configured endpoints and separately owned credentials."""
    try:
        if source not in SOURCES:
            raise ValueError("Choose Paste text or OneLake document.")
        if isinstance(max_words, bool) or not isinstance(max_words, int) or max_words <= 0:
            raise ValueError("Maximum summary words must be a positive integer.")

        with DefaultAzureCredential() as aml_credential:
            with InferenceClient(
                AML_ENDPOINT,
                aml_credential,
                credential_scopes=["https://ml.azure.com/.default"],
                connection_timeout=10,
                read_timeout=600,
                retry_total=0,
            ) as client:
                if source == SOURCES[0]:
                    result = client.summarize({"text": text, "max_words": max_words})
                else:
                    request = {
                        "documentUrl": document_url,
                        "onelakeEndpoint": ONELAKE_ENDPOINT,
                        "onelakeWorkspaceId": WORKSPACE_ID,
                        "onelakeLakehouseId": LAKEHOUSE_ID,
                        "max_words": max_words,
                    }
                    with DefaultAzureCredential() as onelake_credential:
                        result = client.summarize(request, onelake_credential=onelake_credential)
    except ClientAuthenticationError as exc:
        raise gr.Error(
            "Sign-in or access was denied. Run az login locally and check AML scoring and OneLake file-read permissions.",
            print_exception=False,
        ) from exc
    except ResourceNotFoundError as exc:
        raise gr.Error(
            "The document or model endpoint was not found. Check the OneLake file URL and configured AML endpoint.",
            print_exception=False,
        ) from exc
    except DecodeError as exc:
        raise gr.Error(
            "The model returned an invalid response instead of a summary. Check the deployed API contract.",
            print_exception=False,
        ) from exc
    except HttpResponseError as exc:
        status = f" (HTTP {exc.status_code})" if exc.status_code is not None else ""
        raise gr.Error(
            f"The service rejected the request{status}. Check access, input limits, and endpoint availability.",
            print_exception=False,
        ) from exc
    except (ServiceRequestError, ServiceResponseError) as exc:
        raise gr.Error(
            "The download or model request failed or timed out. Check network access to OneLake and AML, then retry.",
            print_exception=False,
        ) from exc
    except UnicodeError as exc:
        raise gr.Error("The document must contain valid UTF-8 text.", print_exception=False) from exc
    except (ValueError, TypeError) as exc:
        raise gr.Error(str(exc), print_exception=False) from exc
    except ImportError as exc:
        raise gr.Error(
            "OneLake support is missing. Install this checkout with the [onelake] extra in the demo environment.",
            print_exception=False,
        ) from exc

    summary = result["summary"]
    return summary, {"summary": summary}


def create_demo() -> gr.Blocks:
    """Build the UI without authenticating, downloading, or running inference."""
    with gr.Blocks(title=DEMO_TITLE, analytics_enabled=False, fill_width=True) as demo:
        gr.HTML(
            f'<div class="eyebrow">Document intelligence</div>'
            f"<h1>{DEMO_TITLE}</h1>"
            '<p class="description">Turn your documents into clear, concise summaries. '
            "Start with text or connect a OneLake file.</p>"
            '<div class="badges"><span class="badge">Qwen3</span>'
            '<span class="badge">Text + OneLake</span>'
            '<span class="badge">Azure Data AI SDK</span></div>',
            elem_id="demo-header",
        )
        with gr.Row(equal_height=True, elem_id="workspace"):
            with gr.Column(min_width=360, elem_id="input-panel"):
                gr.Markdown(
                    "### Source document\nChoose the content and how concise the summary should be.",
                    elem_classes="panel-heading",
                )
                source = gr.Dropdown(choices=SOURCES, value=SOURCES[0], label="Input source", interactive=True)
                text = gr.Textbox(
                    value=EXAMPLE_TEXT,
                    label="Document text",
                    lines=10,
                    max_lines=20,
                    max_length=1024 * 1024,
                    placeholder="Paste the document you want to summarize.",
                    elem_id="input-text",
                )
                document_url = gr.Textbox(
                    value=DOCUMENT_URL,
                    label="OneLake file URL",
                    info="Only the configured workspace and lakehouse are allowed. UTF-8 .txt and .md files only.",
                    lines=3,
                    visible=False,
                )
                max_words = gr.Number(
                    value=100,
                    minimum=1,
                    precision=0,
                    label="Maximum summary words",
                    info="Request a shorter or more detailed summary. Default: 100 words.",
                )
                with gr.Row():
                    submit = gr.Button("Generate summary", variant="primary", scale=2, elem_id="summarize-button")
                    clear = gr.Button("Clear result", scale=1, elem_id="clear-button")
            with gr.Column(min_width=360, elem_id="output-panel"):
                gr.Markdown(
                    "### Your summary\nReview the result, then copy it with one click.",
                    elem_classes="panel-heading",
                )
                summary = gr.Textbox(
                    label="Summary",
                    lines=10,
                    max_lines=24,
                    interactive=False,
                    buttons=["copy"],
                    placeholder="Your summary will appear here.",
                    elem_id="summary-output",
                    scale=1,
                )
                with gr.Accordion("Summary JSON", open=False):
                    response = gr.JSON(label="Response", value={})

        gr.Markdown(
            "The document stays in memory. Only its text and the word limit are sent to Qwen3.",
            elem_id="demo-note",
        )
        with gr.Accordion("Configured services", open=False, elem_id="service-details"):
            gr.Textbox(value=AML_ENDPOINT, label="Qwen3 endpoint", interactive=False)
            gr.Textbox(value=ONELAKE_ENDPOINT, label="OneLake endpoint", interactive=False)
            gr.Textbox(value=WORKSPACE_ID, label="Workspace ID", interactive=False)
            gr.Textbox(value=LAKEHOUSE_ID, label="Lakehouse ID", interactive=False)
        outputs = [summary, response]
        source.change(
            change_source,
            inputs=source,
            outputs=[text, document_url],
            queue=False,
            show_progress="hidden",
            api_visibility="private",
        ).then(clear_outputs, outputs=outputs, queue=False, show_progress="hidden", api_visibility="private")
        clear.click(clear_outputs, outputs=outputs, queue=False, show_progress="hidden", api_visibility="private")
        prediction = submit.click(
            start_summarization,
            outputs=[*outputs, submit],
            queue=False,
            show_progress="hidden",
            api_visibility="private",
        ).then(
            summarize,
            inputs=[source, text, document_url, max_words],
            outputs=outputs,
            concurrency_limit=1,
            show_progress="hidden",
            api_visibility="private",
        )
        prediction.success(
            finish_summarization, outputs=submit, queue=False, show_progress="hidden", api_visibility="private"
        )
        prediction.failure(
            finish_summarization, outputs=submit, queue=False, show_progress="hidden", api_visibility="private"
        )
    return demo


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=7860, help="Local HTTP port (default: 7860).")
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535.")

    if not hasattr(InferenceClient, "summarize"):
        raise RuntimeError("Install this checkout's azure-data-ai package into the Python environment running the app.")
    demo = create_demo()
    demo.queue(default_concurrency_limit=1, max_size=8, api_open=False)
    demo.launch(
        server_name="127.0.0.1",
        server_port=args.port,
        share=False,
        show_error=False,
        enable_monitoring=False,
        run_history=False,
        mcp_server=False,
        footer_links=[],
        allowed_paths=[],
        blocked_paths=[str(Path(__file__).resolve().parents[4])],
        max_file_size="1mb",
        strict_cors=True,
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="indigo",
            neutral_hue="slate",
            radius_size="lg",
            font=["system-ui", "-apple-system", "Segoe UI", "sans-serif"],
            font_mono=["ui-monospace", "Consolas", "monospace"],
        ),
        css=CSS,
    )


if __name__ == "__main__":
    main()
