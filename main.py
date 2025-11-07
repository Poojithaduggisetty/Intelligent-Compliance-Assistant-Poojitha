import os
from dotenv import load_dotenv
import requests
from azure.search.documents import SearchClient
from azure.search.documents.models import QueryType
from azure.core.credentials import AzureKeyCredential
 
# Load environment variables
load_dotenv()
 
# Azure OpenAI (REST) settings
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")  # e.g. gpt-4o
 
# Azure Cognitive Search settings
SEARCH_ENDPOINT = os.getenv("TEXTEMBED_SEARCH_ENDPOINT")
SEARCH_API_KEY = os.getenv("TEXTEMBED_SEARCH_API_KEY")
SEARCH_INDEX = os.getenv("TEXTEMBED_SEARCH_INDEX")
 
# Basic env validation
if not all([AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_KEY, AZURE_OPENAI_DEPLOYMENT]):
    raise SystemExit("Missing AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_KEY or AZURE_OPENAI_DEPLOYMENT in .env")
 
# Search client may be optional; create only if vars present
search_client = None
if SEARCH_ENDPOINT and SEARCH_API_KEY and SEARCH_INDEX:
    try:
        search_client = SearchClient(
            endpoint=SEARCH_ENDPOINT,
            index_name=SEARCH_INDEX,
            credential=AzureKeyCredential(SEARCH_API_KEY),
        )
    except Exception as e:
        print(f"Warning: failed to initialize SearchClient: {e}")
        search_client = None
 
# Simple in-memory store
_memory_store: dict = {}
 
def add_interaction(user_id: str, interaction: str):
    _memory_store.setdefault(user_id, []).append({
        "id": f"interaction-{user_id}-{abs(hash(interaction))}",
        "text": interaction,
    })
 
def get_user_history(user_id: str):
    return _memory_store.get(user_id, [])[:50]
 
def send_chat_request(prompt: str, max_tokens: int = 800, temperature: float = 0.2) -> str:
    """
    Call Azure OpenAI chat completions REST API for the configured deployment.
    Returns the assistant text or an error string.
    """
    try:
        base = AZURE_OPENAI_ENDPOINT.rstrip("/")
        url = f"{base}/openai/deployments/{AZURE_OPENAI_DEPLOYMENT}/chat/completions?api-version=2023-10-01-preview"
        headers = {"api-key": AZURE_OPENAI_KEY, "Content-Type": "application/json"}
        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        # Azure OpenAI chat response shape: choices[0].message.content
        return data.get("choices", [{}])[0].get("message", {}).get("content", "") or str(data)
    except Exception as e:
        return f"ERROR: chat request failed: {e}"
 
def text_search(query: str):
    """
    Query Azure Cognitive Search if configured. Returns list of text snippets.
    """
    if not search_client:
        return []
 
    try:
        results = search_client.search(query, query_type=QueryType.SIMPLE)
        out = []
        for r in results:
            # r may be a SearchResult; convert to dict-like if possible
            try:
                doc = dict(r)
            except Exception:
                doc = r
            # common field names: 'content', 'text', 'description', or entire doc
            text = None
            for key in ("content", "text", "description", "body"):
                if isinstance(doc, dict) and key in doc and doc[key]:
                    text = doc[key]
                    break
            if text is None:
                text = str(doc)
            out.append(text)
        return out
    except Exception as e:
        return [f"Search error: {e}"]
 
def orchestrate_compliance_process(user_id: str, user_query: str):
    # Step 1: validation
    validation_prompt = f"Validate compliance for this action and answer with a short verdict (Compliant / Violation) and a one-line rationale:\n\n{user_query}"
    validation_result = send_chat_request(validation_prompt)
    add_interaction(user_id, f"Validation result: {validation_result}")
 
    # If the model indicates a violation, gather documents and alert
    if isinstance(validation_result, str) and "violation" in validation_result.lower():
        search_results = text_search(user_query)
        context_text = "\n\n".join(search_results) if search_results else "(no documents found)"
        add_interaction(user_id, f"Retrieved docs: {context_text}")
 
        alert_prompt = (
            f"You are a compliance alerting agent. Alert user {user_id} about a policy violation.\n\n"
            f"Validation:\n{validation_result}\n\nContext documents:\n{context_text}\n\n"
            "Provide a concise alert message and recommended next steps."
        )
        alert_result = send_chat_request(alert_prompt)
        add_interaction(user_id, f"Alert result: {alert_result}")
 
        return {"status": "violation_detected", "alert": alert_result, "validation": validation_result}
    else:
        return {"status": "compliant", "message": validation_result}
 
if __name__ == "__main__":
    # quick manual test
    user_id = "user123"
    test_query = "Transfer customer data outside the EU."
    result = orchestrate_compliance_process(user_id, test_query)
    print("Orchestration result:")
    print(result)
 
    history = get_user_history(user_id)
    print("\nUser compliance interaction history:")
    for record in history:
        print(f"- {record['text']}")