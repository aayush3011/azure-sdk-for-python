# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Summarize a OneLake .txt/.md file using the Qwen3 AML deployment."""

from urllib.parse import quote

from azure.data.ai import InferenceClient
from azure.identity import DefaultAzureCredential


def main() -> None:
    with DefaultAzureCredential() as aml_credential:
        with InferenceClient(
            "https://qwen3-summarization.eastus2.inference.ml.azure.com/summarize",
            aml_credential,
            credential_scopes=["https://ml.azure.com/.default"],
            connection_timeout=10,
            read_timeout=600,
        ) as client:
            endpoint = "https://daily-onelake.dfs.fabric.microsoft.com"
            workspace_id = "f2f58846-6636-4c9a-b901-c196f9f3d904"
            lakehouse_id = "38bde7a1-b24e-4d51-a8ad-269406b3c99c"
            file_path = "Files/design_patterns.md"
            request = {
                "documentUrl": f"{endpoint}/{workspace_id}/{lakehouse_id}/{quote(file_path, safe='/')}",
                "onelakeEndpoint": endpoint,
                "onelakeWorkspaceId": workspace_id,
                "onelakeLakehouseId": lakehouse_id,
                "max_words": 100,
            }
            with DefaultAzureCredential() as onelake_credential:
                result = client.summarize(request, onelake_credential=onelake_credential)

    print(result["summary"])


if __name__ == "__main__":
    main()
