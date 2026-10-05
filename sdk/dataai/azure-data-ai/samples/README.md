# Azure Data AI samples

The reranking samples use Azure Data AI, hosted by Azure Inference Service. The generated
SDK accepts dictionary requests or `SemanticRerankingInferenceContent` models and
returns `SemanticRerankingInferenceResult`. Dictionary-style response access remains supported.
The samples do not require a Cosmos DB account.

The SDK takes its endpoint and credential directly from the application. Environment
variables are optional configuration conventions, not SDK requirements.

The synchronous `semantic_reranking.py` sample uses the endpoint, key, and request configured
in `get_sample_inputs()`. Preserve local configuration privately; do not commit real credentials.
The reranking samples read these environment variables:

| Variable | Value |
| --- | --- |
| `AZURE_DATA_AI_ENDPOINT` | Azure Data AI endpoint. |
| `AZURE_DATA_AI_KEY` | API subscription key, required for the key-auth samples. |

Never put real credentials in source code intended to be shared.

- `python samples/semantic_reranking.py`: synchronous `AzureKeyCredential` example with sentence scores.
- `python samples/semantic_reranking_async.py`: asynchronous `AzureKeyCredential` example.
- `python samples/semantic_reranking_entra.py`: synchronous Microsoft Entra example using
  `DefaultAzureCredential`. Install `azure-identity` and configure a supported credential with permission
  to invoke the service endpoint.
- `python samples/semantic_reranking_entra_json.py`: Microsoft Entra example for
  JSON documents, selected fields, and document/sentence scores, using `DefaultAzureCredential`.

For the async sample, install `aiohttp`. The reranking Entra samples only need
`AZURE_DATA_AI_ENDPOINT`, not `AZURE_DATA_AI_KEY`.

Both clients accept `AzureKeyCredential` or the appropriate sync/async token
credential. Wrap raw keys with `AzureKeyCredential`. Key authentication uses `Ocp-Apim-Subscription-Key` and requires a
key issued for an endpoint with key-based authentication enabled.

## Summarization POC

`summarization.py` calls the Qwen3 AML managed online endpoint directly at
`https://qwen3-summarization.eastus2.inference.ml.azure.com/summarize`. It uses
the deployment's contract, **not** a standard contract for every AML model.
This replaces the previous Phi `/score` backend; generated reranking and the
`InferenceClient` constructor are unchanged.

Install from the package directory with
`python -m pip install -e ".[onelake]" azure-identity`, using your configured package
feed. Use the same Python executable for installation and for running the sample;
another interpreter may still import an older installed package without `summarize`.
The local sample sets its AML summarization URL and OneLake values directly in `main()`.
Keep private configuration local when sharing the sample.

```bash
python samples/summarization.py
```

This reads the configured UTF-8 `.txt` or `.md` document from OneLake.
To use a different document, update these values in `main()`:

| Variable | Where to find the value |
| --- | --- |
| `endpoint` | HTTPS form of the host in **Copy ABFS path**, e.g. `https://daily-onelake.dfs.fabric.microsoft.com`. |
| `workspace_id` | Workspace GUID after `/groups/` in the Fabric browser URL or before `@` in an ID-based ABFS path. |
| `lakehouse_id` | Lakehouse GUID after `/lakehouses/` in the browser URL or the first path segment in an ID-based ABFS path. |
| `file_path` | Unescaped path beginning with `Files/`, e.g. `Files/design_patterns.md`. |

### Local browser demo

`summarization_app.py` is a Gradio UI titled **Inference Service - Summarization API**
for demonstrating the same SDK operation.
It includes a text/OneLake input selector, a summary word limit, a copyable summary,
and an expandable summary JSON view. The Qwen3 endpoint
and OneLake location are already configured to match the command-line sample.
Opening the page does not download a file or call the model; click **Generate summary** to
submit. The default input is harmless example text.

From the **repository root**, create a separate environment so UI dependencies do
not change the SDK development environment:

```bash
python3.11 -m venv .venv-gradio
export UV_DEFAULT_INDEX="https://packagefeedproxy.microsoft.io/pypi/simple/"
export PIP_INDEX_URL="$UV_DEFAULT_INDEX"
.venv-gradio/bin/python -m pip install -e "sdk/dataai/azure-data-ai[onelake]" \
  -r sdk/dataai/azure-data-ai/samples/requirements-gradio.txt
.venv-gradio/bin/python sdk/dataai/azure-data-ai/samples/summarization_app.py
```

Open `http://127.0.0.1:7860` and screen-share the browser. Use `--port 7861` if the
default port is occupied. Authenticate locally with a supported
`DefaultAzureCredential` identity, such as `az login`. The app obtains separate
AML and Storage credentials in Python; no token or key is sent to the browser.
The browser may show the summary and API response, so only share documents
appropriate for your audience.

The app binds only to loopback, disables Gradio public sharing, telemetry and run
history, and does not expose the repository through Gradio's file-serving route.
It is not a public hosted app; the localhost address is not a remote invitation
link. Do not tunnel or expose it without adding authentication and document
authorization. There are no Azure resources to provision.

The demo queues one model call at a time, disables automatic inference retries,
and uses the sample's 600-second AML read timeout. Errors are displayed rather
than returned as successful summaries. The demo shows only the summary and its JSON
representation, without diagnostic metadata or performance metrics. The button
indicates when generation is in progress without showing a runtime counter.

All OneLake settings are per request. The SDK accepts a storage HTTPS
`documentUrl`, not a browser or ABFS link; the sample constructs it from these
settings. Supply the endpoint and GUIDs from **trusted application configuration**.
Matching URL fields is not caller authorization: an App Service must authenticate
callers, authorize each document, and scope its backend identity appropriately.
Fabric must grant file-read permissions and enable external-app access.

### AML authentication and request contract

This deployment uses `AADToken` authentication. Use a refreshable, synchronous
Entra credential, not a copied token or `AzureKeyCredential`:

```python
from azure.data.ai import InferenceClient
from azure.identity import DefaultAzureCredential

with DefaultAzureCredential() as credential:
    with InferenceClient(
        "https://qwen3-summarization.eastus2.inference.ml.azure.com/summarize",
        credential,
        credential_scopes=["https://ml.azure.com/.default"],
        connection_timeout=10,
        read_timeout=600,
    ) as client:
        result = client.summarize(
            {"text": "The complete document...", "max_words": 100}
        )
        print(result["summary"])
```

For local use, `az login` can supply the identity; App Service can use its managed
identity. The identity needs the endpoint's
`Microsoft.MachineLearningServices/workspaces/onlineEndpoints/score/action`
permission. The SDK uses its existing bearer policy with the explicitly configured
AML scope. The generated default `https://dbinference.azure.com/.default` remains
unchanged for reranking and is rejected by `summarize` to avoid a wrong-audience
request. OneLake uses its own credential and `https://storage.azure.com/.default`.
Credentials and OneLake locations are never included in the AML JSON body.

The wire request is `POST /summarize`, without an automatically added `api-version`,
with `{"text": "...", "max_words": 100}`. `max_words` must be a positive integer
and defaults to 100 for each call, including OneLake requests. The SDK returns
the JSON object unchanged, requiring a `summary` string and preserving any
additional metadata the service returns. The sample prints `result["summary"]`.
The old `task`, `max_new_tokens`, and `temperature` fields are not accepted.
There is no fallback to `/score` or normalization of a `generated_text` response.

### POC boundaries

OneLake support is optional; text-only summarization and reranking do not import
Storage. Downloads stay in memory, decode strict UTF-8 with an optional BOM, and
use an ETag-conditioned bounded read. `max_input_bytes` defaults to 1 MiB,
including the file BOM. Invalid encoding, empty/whitespace-only input, compression,
oversized/incomplete downloads, and service errors raise exceptions. There is no
input truncation, chunked summarization, or disk staging. `max_words` requests an
output word limit, not an input token limit; the deployed model's context limit
still applies. The SDK does not locally truncate the summary to enforce the word limit.

The sample uses 10-second connection and 600-second read timeouts for AML. OneLake
defaults to 10-second connection and 60-second read timeouts with up to three
retries; customize them using `onelake_connection_timeout` and
`onelake_read_timeout`. These are network timeouts, not a combined end-to-end
deadline. The SDK closes its Storage client, not caller-owned credentials.
Redirects are disabled for both services. Public AML managed endpoints and
global/regional/daily OneLake DFS endpoints are supported, not custom/private-link
hosts. This handwritten, sync-only POC is not a production TypeSpec API.
The operation is implemented directly in `_operations/_operations.py`, alongside
`semantic_rerank`, with no `_patch.py` client replacement. Regenerating from
TypeSpec can overwrite this manual addition.

To discover another deployment's contract, check its OpenAPI document first.
If it only declares `{}`, inspect that deployment's `score.py` input parser;
the model name and generic Consume snippet do not define the payload.
See [AML OpenAPI support](https://github.com/Azure/azureml-examples/tree/main/cli/endpoints/online/managed/openapi),
[AML authentication](https://learn.microsoft.com/azure/machine-learning/how-to-authenticate-online-endpoint),
[OneLake access](https://learn.microsoft.com/fabric/onelake/onelake-access-api), and
the [Python SDK design guidelines](https://azure.github.io/azure-sdk/python_design.html).

## Model selection and request options

Each sample sets `model` directly in its request dictionary. Change that field to a
model supported by your endpoint, or remove it to leave model selection to the service.
The synchronous example configures its request in `get_sample_inputs()`.
The TypeSpec example uses `semantic-reranker-v1`; choose a model supported by your
endpoint rather than assuming that every endpoint exposes that model.

The request dictionaries demonstrate `topK`, `batchSize`, `sort`, `returnDocuments`,
`returnSentenceScore`, and `documentType`. They use the generated
`azure.data.ai.types.SemanticRerankingInferenceContent` TypedDict for type checking.
These are service request-body fields, not method keyword arguments.

For JSON documents, JSON-encode each document string and use `documentType: "json"`.
Supply `targetPaths`, using dot notation for nested properties and commas for
multiple paths, such as `"meta.content,id"`.
See the [JSON document and model-selection example](https://github.com/Azure/azure-sdk-for-python/blob/main/sdk/dataai/azure-data-ai/README.md#model-selection-and-json-documents).

Python method names use snake_case, but dictionary keys keep the service's JSON
names: `topK`, `returnDocuments`, `returnSentenceScore`, and `sentenceScores`.
Read response entries from `scores`, not the legacy `Scores` spelling.
Sentence-score indices are zero-based and can exceed 2; sentence scores range
from 0 to 1 inclusive. Iterate all returned sentence entries without truncation.

## JSON documents with Microsoft Entra

The JSON sample uses the same Entra prerequisites and `AZURE_DATA_AI_ENDPOINT`
configuration, without an API key:

```bash
python samples/semantic_reranking_entra_json.py
```

Use a model supported by your endpoint, or remove the request's `model` field for its default.
The sample serializes each document with `json.dumps`, sets `documentType` to
`"json"`, and passes `"title,description"` as the comma-separated `targetPaths`
string. Do not pass dictionaries directly in the `documents` array.

Returned document text remains a JSON-encoded string; the sample uses `json.loads`
to display its fields. It also prints any returned sentence scores. Only one
JSON sample is provided; automated coverage exercises both sync/async clients and
generated-model and dictionary requests with `AzureKeyCredential` and Entra authentication.
