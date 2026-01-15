"""
ai_inbox_triage - triage support emails from CSV and generate draft replies.

Run:
    python triage.py --input sample_emails.csv --dry-run
"""

import argparse
import csv
import json
import logging
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List
from dotenv import load_dotenv
load_dotenv()

# Constants
DEFAULT_MODEL = "gpt-4o-mini"  # change as needed
ALLOWED_CATEGORIES = {"Billing", "Bug", "FeatureRequest", "Account", "SalesLead", "Spam", "Other"}
ALLOWED_URGENCY = {"Low", "Medium", "High"}

# Rough cost estimate constants
COST_PER_1K_TOKENS_USD = 0.002  # placeholder, very rough

# Short company profile embedded for prompts
COMPANY_PROFILE = (
    "Acme Support\n"
    "Business hours: Mon-Fri 9:00-17:00 PT\n"
    "Refund policy: Full refund within 14 days for annual plans; pro-rated refunds for monthly subscriptions.\n"
    "Tone: professional, empathetic, concise."
)

# Required CSV columns
REQUIRED_COLUMNS = {"id", "from", "subject", "body", "date"}

# Import OpenAI client only when needed
try:
    from openai import OpenAI
    from openai import OpenAI, OpenAIError, AuthenticationError, RateLimitError, APITimeoutError
except Exception:
    OpenAI = None  # will check later; dry-run should still work
    OpenAIError = Exception
    AuthenticationError = Exception
    RateLimitError = Exception
    Timeout = Exception


def setup_logging(log_path: Path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # Avoid adding duplicate handlers
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s", "%H:%M:%S"))
        logger.addHandler(ch)
    # File handler
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
    logger.addHandler(fh)


def validate_input_file(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        missing = [c for c in REQUIRED_COLUMNS if c not in headers]
        return missing


def validate_model_json(data: Any) -> None:
    """
    Validate JSON returned by model matches required schema.
    Raises ValueError with helpful messages on failure.
    """
    if not isinstance(data, dict):
        raise ValueError("Model output is not a JSON object")
    # required keys
    keys = {"category", "urgency", "summary", "action_items", "entities", "draft_reply"}
    missing = keys - data.keys()
    if missing:
        raise ValueError(f"Missing keys in model JSON: {sorted(list(missing))}")
    # category
    cat = data["category"]
    if not isinstance(cat, str) or cat not in ALLOWED_CATEGORIES:
        raise ValueError(f"Invalid category: {repr(cat)}. Allowed: {sorted(ALLOWED_CATEGORIES)}")
    # urgency
    urg = data["urgency"]
    if not isinstance(urg, str) or urg not in ALLOWED_URGENCY:
        raise ValueError(f"Invalid urgency: {repr(urg)}. Allowed: {sorted(ALLOWED_URGENCY)}")
    # summary
    summary = data["summary"]
    if not isinstance(summary, str) or not summary.strip():
        raise ValueError("Summary must be a non-empty string.")
    # action_items
    ai = data["action_items"]
    if not isinstance(ai, list) or not all(isinstance(x, str) for x in ai):
        raise ValueError("action_items must be a list of strings.")
    # entities
    ent = data["entities"]
    if not isinstance(ent, dict):
        raise ValueError("entities must be a JSON object/dictionary.")
    # draft_reply
    dr = data["draft_reply"]
    if not isinstance(dr, str) or not dr.strip():
        raise ValueError("draft_reply must be a non-empty string.")


def call_openai(messages: List[Dict[str, str]], model: str, timeout: int = 15) -> Dict[str, Any]:
    """
    Call OpenAI with messages, expect assistant to return JSON-only content.
    Retries once with a 'repair' instruction if parsing fails.
    Raises exceptions for unrecoverable failures.
    """
    if OpenAI is None:
        raise RuntimeError("OpenAI client is not available. Install the openai package.")

    client = OpenAI()
    # First attempt
    try:
        resp = client.chat.completions.create(model=model, messages=messages, timeout=timeout)
        content = _extract_content_from_response(resp)
        try:
            parsed = json.loads(content)
            validate_model_json(parsed)
            return parsed
        except Exception as e:
            # Attempt one repair
            logging.info("Initial JSON parsing/validation failed; attempting one repair request.")
            repair_msg = [
                {"role": "system", "content": "You must respond with ONLY valid JSON matching the required schema. No explanation."},
                {"role": "user", "content": (
                    "Previous response was not valid JSON or failed validation. "
                    "Please reply with ONLY valid JSON exactly matching the schema:\n"
                    '{"category":"...","urgency":"...","summary":"...","action_items":["..."],"entities": {...},"draft_reply":"..."}'
                )}
            ]
            # include original user message to provide context
            repair_msgs = repair_msg + messages[-1:]
            resp2 = client.chat.completions.create(model=model, messages=repair_msgs, timeout=timeout)
            content2 = _extract_content_from_response(resp2)
            parsed2 = json.loads(content2)
            validate_model_json(parsed2)
            return parsed2
    except AuthenticationError as e:
        raise RuntimeError("Authentication error when calling OpenAI API. Check OPENAI_API_KEY.") from e
    except RateLimitError as e:
        raise RuntimeError("Rate limit reached when calling OpenAI API.") from e
    except Timeout as e:
        raise RuntimeError("Request to OpenAI timed out.") from e
    except OpenAIError as e:
        raise RuntimeError(f"OpenAI API error: {e}") from e
    except json.JSONDecodeError as e:
        raise RuntimeError("Failed to parse JSON from OpenAI response.") from e


def _extract_content_from_response(resp) -> str:
    """
    Extract the assistant message content from the response object.
    This function attempts to be robust across client return shapes.
    """
    # Try known shape for client.chat.completions.create
    try:
        choices = getattr(resp, "choices", None) or resp.get("choices")
        if choices and len(choices) > 0:
            message = choices[0].get("message") if isinstance(choices[0], dict) else choices[0].message
            content = message.get("content") if isinstance(message, dict) else message.content
            return content
    except Exception:
        pass
    # Fallback: try dict access
    try:
        data = dict(resp)
        return data.get("choices", [])[0].get("message", {}).get("content", "")
    except Exception:
        pass
    raise RuntimeError("Unexpected OpenAI response format")


def simple_dryrun_response(row: Dict[str, str]) -> Dict[str, Any]:
    """
    Generate a fake but valid model response using simple keyword rules.
    """
    body = (row.get("subject", "") + " " + row.get("body", "")).lower()
    category = "Other"
    urgency = "Low"
    action_items = []
    entities = {}

    if any(k in body for k in ("refund", "charged", "billing", "payment", "invoice")):
        category = "Billing"
        action_items = ["Verify charge in billing system", "Issue refund if duplicate"]
        urgency = "Medium"
    if any(k in body for k in ("crash", "error", "stack trace", "bug", "freeze")):
        category = "Bug"
        action_items = ["Reproduce the bug using provided steps", "Request logs/screenshot"]
        urgency = "High"
    if any(k in body for k in ("feature", "export", "csv", "integration")):
        category = "FeatureRequest"
        action_items = ["Log feature request to roadmap", "Follow up with product team"]
        urgency = "Low"
    if any(k in body for k in ("password", "reset", "login", "access")):
        category = "Account"
        action_items = ["Verify account email", "Send password reset link"]
        urgency = "High"
    if any(k in body for k in ("pricing", "enterprise", "volume", "sla")):
        category = "SalesLead"
        action_items = ["Provide pricing sheet", "Schedule sales call"]
        urgency = "Medium"
    if any(k in body for k in ("won a prize", "click http", "free iphone", "claim")):
        category = "Spam"
        action_items = []
        urgency = "Low"
    summary = (row.get("subject", "")[:200]).strip()
    draft_reply = f"Hi,\n\nThanks for reaching out regarding \"{row.get('subject', '')}\". We will review and follow up shortly.\n\nBest,\nSupport Team"
    # minimal entities detection
    email_match = re.search(r"[\w\.-]+@[\w\.-]+\.\w+", row.get("body", "")) or re.search(r"[\w\.-]+@[\w\.-]+\.\w+", row.get("from", ""))
    if email_match:
        entities["email"] = email_match.group(0)
    return {
        "category": category,
        "urgency": urgency,
        "summary": summary,
        "action_items": action_items,
        "entities": entities,
        "draft_reply": draft_reply,
    }


def build_messages_for_row(row: Dict[str, str], company_profile: str) -> List[Dict[str, str]]:
    """
    Construct system/user messages instructing the model to return ONLY valid JSON matching schema.
    """
    system_content = (
        f"You are a helpful customer support assistant for the following company:\n{company_profile}\n"
        "Respond ONLY with valid JSON matching the schema exactly. Do NOT include any prose outside the JSON."
    )
    schema = {
        "category": "one of " + ", ".join(sorted(ALLOWED_CATEGORIES)),
        "urgency": "one of " + ", ".join(sorted(ALLOWED_URGENCY)),
        "summary": "short (max ~2 sentences) summary string",
        "action_items": "array of short strings describing next actionable steps (may be empty)",
        "entities": "JSON object with detected entities (may be empty)",
        "draft_reply": "a concise draft reply (keep under ~120 words) as a string"
    }
    user_content = (
        "Incoming email:\n"
        f"From: {row.get('from','')}\n"
        f"Date: {row.get('date','')}\n"
        f"Subject: {row.get('subject','')}\n\n"
        f"Body:\n{row.get('body','')}\n\n"
        "Return JSON with this schema (and only JSON):\n"
        + json.dumps(schema, ensure_ascii=False, indent=2)
    )
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]


def write_draft_file(drafts_dir: Path, row_id: str, row: Dict[str, str], model_json: Dict[str, Any]) -> Path:
    """
    Write a draft file at drafts_dir/<id>.txt with headers and DRAFT REPLY block.
    Returns relative path to the draft file.
    """
    filename = f"{row_id}.txt"
    path = drafts_dir / filename
    with path.open("w", encoding="utf-8") as f:
        f.write(f"Subject: {row.get('subject','')}\n")
        f.write(f"From: {row.get('from','')}\n")
        f.write(f"Category: {model_json.get('category','')}\n")
        f.write(f"Urgency: {model_json.get('urgency','')}\n")
        f.write(f"Summary: {model_json.get('summary','')}\n\n")
        f.write("DRAFT REPLY:\n")
        f.write(model_json.get("draft_reply", "").strip() + "\n")
    return path


def estimate_cost(prompt_chars: int, output_chars: int) -> float:
    """
    Very rough estimate: tokens ~= chars/4, cost per 1k tokens constant.
    """
    tokens = (prompt_chars + output_chars) / 4.0
    cost = tokens / 1000.0 * COST_PER_1K_TOKENS_USD
    return round(cost, 6)


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    parser = argparse.ArgumentParser(description="Triage support emails from CSV and generate draft replies.")
    parser.add_argument("--input", "-i", required=True, help="Path to input CSV")
    parser.add_argument("--output", "-o", default="output", help="Output directory (default: output)")
    parser.add_argument("--max-emails", type=int, default=50, help="Maximum number of emails to process")
    parser.add_argument("--dry-run", action="store_true", help="Do not call OpenAI; generate deterministic fake responses")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Model name (default: {DEFAULT_MODEL})")
    parser.add_argument("--skip-spam", action="store_true", help="If model classifies as Spam, do not write a draft file")
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    try:
        missing = validate_input_file(input_path)
    except FileNotFoundError as e:
        print(str(e), file=sys.stderr)
        sys.exit(2)
    except Exception as e:
        print(f"Failed to validate input file: {e}", file=sys.stderr)
        sys.exit(2)
    if missing:
        print(f"Input file is missing required columns: {missing}", file=sys.stderr)
        sys.exit(2)

    # Prepare output directories
    output_dir = Path(args.output)
    drafts_dir = output_dir / "drafts"
    output_dir.mkdir(parents=True, exist_ok=True)
    drafts_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    log_path = output_dir / "triage.log"
    setup_logging(log_path)
    logging.info("Starting ai_inbox_triage")
    logging.info("Input: %s", str(input_path))
    logging.info("Output: %s", str(output_dir))
    logging.info("Dry run: %s", args.dry_run)
    logging.info("Model: %s", args.model)

    # Simplified environment check: read OPENAI_API_KEY directly (no .env fallback).
    openai_key = os.environ.get("OPENAI_API_KEY")
    if not args.dry_run:
        if not openai_key:
            logging.error("OPENAI_API_KEY is not set; set it in your environment or IDE run configuration.")
            print("OPENAI_API_KEY is not set in environment; cannot run in real mode.", file=sys.stderr)
            sys.exit(2)
        if OpenAI is None:
            logging.error("openai package not installed. Install dependencies: pip install -r requirements.txt")
            print("openai package not installed. Install dependencies: pip install -r requirements.txt", file=sys.stderr)
            sys.exit(2)

    processed_count = 0
    success_count = 0
    failed_count = 0
    skipped_spam_count = 0
    triaged_rows = []

    # For cost estimation: accumulate prompt and output character counts
    total_prompt_chars = 0
    total_output_chars = 0

    triaged_csv_path = output_dir / "triaged_results.csv"
    fieldnames = ["id", "from", "subject", "category", "urgency", "summary", "action_items", "entities_json", "draft_path", "error"]

    # Read and process rows
    with input_path.open("r", encoding="utf-8", newline="") as f_in:
        reader = csv.DictReader(f_in)
        rows = []
        for row in reader:
            rows.append(row)
            if len(rows) >= args.max_emails:
                break

        total_to_process = len(rows)
        for idx, row in enumerate(rows, start=1):
            processed_count += 1
            row_id = (row.get("id") or "").strip()
            display_id = row_id or f"row{idx}"
            logging.info("Processing %d/%d (id=%s)", idx, total_to_process, display_id)
            # Prepare default result template
            result = {
                "id": row_id,
                "from": row.get("from", ""),
                "subject": row.get("subject", ""),
                "category": "",
                "urgency": "",
                "summary": "",
                "action_items": "",
                "entities_json": "",
                "draft_path": "",
                "error": ""
            }
            try:
                if args.dry_run:
                    model_json = simple_dryrun_response(row)
                    # approximate prompt/output chars
                    prompt_chars = len((row.get("subject","") or "") + (row.get("body","") or ""))
                    output_chars = len(json.dumps(model_json, ensure_ascii=False))
                    total_prompt_chars += prompt_chars
                    total_output_chars += output_chars
                else:
                    messages = build_messages_for_row(row, COMPANY_PROFILE)
                    prompt_chars = sum(len(m.get("content", "")) for m in messages)
                    total_prompt_chars += prompt_chars
                    try:
                        model_json = call_openai(messages, args.model, timeout=15)
                        output_chars = len(json.dumps(model_json, ensure_ascii=False))
                        total_output_chars += output_chars
                    except Exception as e:
                        raise RuntimeError(f"OpenAI call failed: {e}")
                # Validate (again)
                validate_model_json(model_json)
                # Fill result fields
                result["category"] = model_json.get("category", "")
                result["urgency"] = model_json.get("urgency", "")
                result["summary"] = model_json.get("summary", "")
                action_items = model_json.get("action_items", [])
                result["action_items"] = " | ".join(action_items) if action_items else ""
                result["entities_json"] = json.dumps(model_json.get("entities", {}), ensure_ascii=False)
                # Draft file write (skip if spam & skip_spam)
                if args.skip_spam and result["category"] == "Spam":
                    skipped_spam_count += 1
                    result["draft_path"] = ""
                else:
                    try:
                        draft_path = write_draft_file(drafts_dir, row_id or f"row{idx}", row, model_json)
                        # store relative path
                        result["draft_path"] = str(Path("drafts") / draft_path.name)
                    except Exception as e:
                        # don't fail entire run for a single file write error
                        result["error"] = f"Failed to write draft file: {e}"
                success_count += 1
            except Exception as e:
                # Record error and continue
                err_msg = str(e)
                logging.error("Error processing id=%s: %s", display_id, err_msg)
                result["error"] = err_msg
                failed_count += 1
            triaged_rows.append(result)
            print(f"Processed {idx}/{total_to_process} (id={display_id})")

    # Write triaged_results.csv
    try:
        with triaged_csv_path.open("w", encoding="utf-8", newline="") as fo:
            writer = csv.DictWriter(fo, fieldnames=fieldnames)
            writer.writeheader()
            for r in triaged_rows:
                writer.writerow(r)
        logging.info("Wrote triaged results to %s", str(triaged_csv_path))
    except Exception as e:
        logging.error("Failed to write triaged_results.csv: %s", e)
        print(f"Failed to write triaged_results.csv: {e}", file=sys.stderr)

    # Append run_log.json
    run_log_path = output_dir / "run_log.json"
    notes = "Dry run" if args.dry_run else "Real run"

    # Use accumulated totals for cost estimate
    estimated_cost = estimate_cost(total_prompt_chars, total_output_chars)
    run_entry = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "input_path": str(input_path),
        "output_path": str(output_dir),
        "model": args.model,
        "processed_count": processed_count,
        "success_count": success_count,
        "failed_count": failed_count,
        "skipped_spam_count": skipped_spam_count,
        "dry_run": bool(args.dry_run),
        "estimated_cost_usd": estimated_cost,
        "notes": f"{notes}. Cost estimate is very rough (tokens ~= chars/4; cost per 1k tokens={COST_PER_1K_TOKENS_USD})."
    }

    # Append to run_log.json
    try:
        if run_log_path.exists():
            with run_log_path.open("r", encoding="utf-8") as rf:
                data = json.load(rf)
                if not isinstance(data, list):
                    data = [data]
        else:
            data = []
        data.append(run_entry)
        with run_log_path.open("w", encoding="utf-8") as wf:
            json.dump(data, wf, ensure_ascii=False, indent=2)
        logging.info("Appended run log to %s", str(run_log_path))
    except Exception as e:
        logging.error("Failed to write run_log.json: %s", e)

    logging.info("Done. Processed=%d success=%d failed=%d skipped_spam=%d", processed_count, success_count, failed_count, skipped_spam_count)
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
    except SystemExit as e:
        raise
    except Exception as e:
        logging.exception("Fatal error: %s", e)
        print(f"Fatal error: {e}", file=sys.stderr)
        sys.exit(1)
    else:
        sys.exit(0)

