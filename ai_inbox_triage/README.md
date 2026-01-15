# ai_inbox_triage

Small CLI tool to triage support emails from a CSV and generate draft replies using the OpenAI API.

Features:
- Reads CSV emails and classifies them into categories and urgencies.
- Produces a short summary, action items, entities, and a draft reply per email.
- Writes results to `output/triaged_results.csv`, individual drafts to `output/drafts/`, and run history to `output/run_log.json`.
- Supports a `--dry-run` mode that generates deterministic fake responses without calling the API.

Setup

1. Create and activate a Python 3.11 virtual environment:
   - Windows (cmd):
     python -m venv .venv
     .venv\Scripts\activate
   - macOS / Linux:
     python -m venv .venv
     source .venv/bin/activate

2. Install dependencies:
   pip install -r requirements.txt

3. Provide your OpenAI API key:
   Set the OPENAI_API_KEY environment variable in your shell or in your IDE's run configuration.
   - Windows (cmd):
     set OPENAI_API_KEY=sk-...
   - macOS / Linux:
     export OPENAI_API_KEY=sk-...

   Note: a .env file is optional; the script reads the environment variable directly.

Quick examples

- Dry-run (no API calls):
  python triage.py --input sample_emails.csv --dry-run

- Real run (make sure OPENAI_API_KEY is set):
  python triage.py --input sample_emails.csv --output my_out --model gpt-4o-mini

Input CSV format

Required header columns (case-sensitive): `id`, `from`, `subject`, `body`, `date`.
Use `csv.DictReader`; rows missing required columns cause a fatal error.

Outputs

- output/triaged_results.csv
  Columns:
    id, from, subject, category, urgency, summary, action_items, entities_json, draft_path, error

- output/drafts/<id>.txt
  Each draft contains header fields and a "DRAFT REPLY:" section. Draft files are not created for Spam when `--skip-spam` is used.

- output/run_log.json
  Appends a run entry with metadata and a rough cost estimate.

Notes

- The OpenAI API key must be set in OPENAI_API_KEY; never commit keys to source control.
- The tool expects the model to return valid JSON only. If parsing fails once, the tool retries a single "repair" request.
- Drafts are generated automatically but must be reviewed by a human before sending.

Troubleshooting

- Invalid key / authentication error:
  - Ensure OPENAI_API_KEY is correct and has permissions.
  - Check network connectivity.

- Rate limit / API errors:
  - Wait and retry later; the program records per-email errors and continues.

- JSON parse errors:
  - The tool retries once with a repair instruction. If parsing still fails, the error is recorded for that email.

Security & Privacy

- Do not include sensitive personal data in the sample CSV.
- This tool does not send emails; it only writes draft files for human review.

License

Freelance-style sample project. Use at your own risk.
