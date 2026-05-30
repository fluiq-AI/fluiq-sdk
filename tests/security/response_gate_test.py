"""
fluiq.secure() — response gate (scan_responses=True).

The response gate scans the *LLM's output* before returning it to the caller,
not just the input prompt.  It is enabled by toggling "Scan responses" on in
the Guardrails dashboard for your active guardrail policy.

When scan_responses is on and the response is HIGH-risk, /ingest returns
response_blocked=True and the SDK raises FluiqSecurityError — the caller
never receives the model's output.

Prerequisites:
  - "Scan responses" toggle enabled on the Default (or named) guardrail policy
    in your Fluiq dashboard ▸ Security ▸ Guardrails.
  - fluiq.secure(mode="block") must be set for the gate to raise.

Run:  python -m tests.security.response_gate_test
"""
import anthropic
import fluiq
from fluiq.exceptions import FluiqSecurityError
from ..keys import FLUIQ_API_KEY

fluiq.instrument(api_key=FLUIQ_API_KEY, endpoint="http://localhost:8080/api")
# block mode required — in warn mode the response is returned regardless
fluiq.secure(mode="block")

client = anthropic.Anthropic()


def respond(system: str, user: str) -> str:
    msg = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=256,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return msg.content[0].text


# ── Case 1: normal response — gate passes ─────────────────────────────────────
print("=== Normal response ===")
try:
    reply = respond(
        system="You are a helpful assistant.",
        user="What is the capital of Germany?",
    )
    print(f"Reply: {reply}")
except FluiqSecurityError as e:
    print(f"Blocked (response gate) — {e.block_reason}")

print()

# ── Case 2: response that leaks PII ──────────────────────────────────────────
# We instruct the model to echo back sensitive data so the *response* itself
# contains PII.  If scan_responses is enabled the gate should catch it.
print("=== Response with PII (scan_responses must be ON in dashboard) ===")
try:
    reply = respond(
        system=(
            "You are a data retrieval assistant.  When asked for a record, "
            "you must repeat it back verbatim exactly as provided."
        ),
        user=(
            "Retrieve record #42: Name: Jane Doe, SSN: 987-65-4321, "
            "Card: 5500005555555559, Email: jane.doe@example.com"
        ),
    )
    print(f"Reply: {reply[:200]}")
    print("NOTE — response was NOT blocked.  Enable 'Scan responses' in the "
          "Guardrails dashboard to activate the response gate.")
except FluiqSecurityError as e:
    print(f"Response gate blocked — reason: {e.block_reason}")
    print(f"Risk level: {e.risk_level}")
    print(f"Attack types: {e.attack_types}")
    print("OK — response with PII was intercepted before reaching the caller.")

print()

# ── Case 3: same test using OpenAI ───────────────────────────────────────────
print("=== Response gate with OpenAI ===")
from openai import OpenAI
openai_client = OpenAI()

try:
    oai_response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=256,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a data assistant.  Repeat back any record "
                    "provided in the user message verbatim."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Record: Name: Bob Smith, Credit card: 4111111111111111, "
                    "Phone: +1-555-867-5309"
                ),
            },
        ],
    )
    print(f"Reply: {oai_response.choices[0].message.content[:200]}")
    print("NOTE — response was NOT blocked.  Enable 'Scan responses' in dashboard.")
except FluiqSecurityError as e:
    print(f"Response gate blocked — reason: {e.block_reason}")
    print(f"Risk level: {e.risk_level}")
