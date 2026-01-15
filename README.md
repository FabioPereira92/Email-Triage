# 📧 AI Email Triage & Drafting Tool (CLI)

A production-ready **AI-powered email triage tool** that processes support emails in bulk, classifies urgency and intent, extracts actionable insights, and generates **human-reviewable draft replies**.

Designed for teams that need to **handle high volumes of inbound email faster** — without sacrificing control or quality.

---

## ✨ What This Tool Does

Given a CSV of incoming support emails, the tool automatically:

- 📂 **Classifies** each email (e.g. Support, Billing, Sales, Spam)
- 🚨 Assigns **urgency levels** (Low / Medium / High / Critical)
- 🧠 Produces:
  - A short **summary**
  - **Action items**
  - Extracted **entities** (products, dates, accounts, etc.)
- ✉️ Generates a **draft reply** for human review (never auto-sent)
- 🗂 Writes structured outputs for auditing, tracking, and iteration

This is a **human-in-the-loop system**: AI accelerates the work, humans stay in control.

---

## 🧠 Example Workflow

```
Incoming Emails (CSV)
        ↓
AI Classification & Analysis
        ↓
Structured Results + Draft Replies
        ↓
Human Review & Send
```

📸 **Input CSV:**  
<img width="1363" height="322" alt="image" src="https://github.com/user-attachments/assets/2b06a548-f06f-4542-8e01-2e51040e6446" />

---

## 📊 Outputs

### 1️⃣ `triaged_results.csv`

A structured overview suitable for dashboards or further automation.

**Columns include:**
- `category`
- `urgency`
- `summary`
- `action_items`
- `entities_json`
- `draft_path`
- `error` (if any)

📸 **Triaged_results.csv:**  
<img width="1873" height="322" alt="image" src="https://github.com/user-attachments/assets/93bd6ffd-d444-42ca-b219-d9f1e2613623" />


---

### 2️⃣ Draft Reply Files (`output/drafts/`)

Each non-spam email gets its own draft file containing:

- Email metadata
- AI analysis summary
- **DRAFT REPLY** section (clearly marked)

📸 **Draft.txt:**  
<img width="715" height="372" alt="image" src="https://github.com/user-attachments/assets/79391646-2526-46f0-8063-efc56421f3ec" />


---

### 3️⃣ Run Log (`output/run_log.json`)

A persistent audit trail containing:
- Timestamp
- Model used
- Number of emails processed
- Errors encountered
- Rough cost estimate

This is especially useful for **cost monitoring and compliance**.

---

## 🧪 Dry-Run Mode (No API Calls)

For testing, demos, and CI environments, the tool supports a deterministic `--dry-run` mode:

```bash
python triage.py --input sample_emails.csv --dry-run
```

- No OpenAI API calls
- Predictable fake outputs
- Same file structure as real runs

---

## 🚀 Quick Start

### 1️⃣ Environment Setup

Create and activate a Python 3.11 virtual environment:

```bash
# Windows (cmd)
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

### 2️⃣ API Key Configuration

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=sk-...
```

> 🔒 **Never commit API keys to source control.**

---

### 3️⃣ Run the Tool

**Dry run (recommended first):**
```bash
python triage.py --input sample_emails.csv --dry-run
```

**Real run:**
```bash
python triage.py --input sample_emails.csv --output my_out --model gpt-4o-mini
```

---

## 📥 Input Format

The input must be a CSV with the following **case-sensitive headers**:

```
id, from, subject, body, date
```

- Parsed using `csv.DictReader`
- Missing required columns cause a **fatal error**
- Errors are recorded per email; the run continues

---

## 🛠 Design & Reliability Considerations

- Strict JSON-only model outputs  
- Automatic single retry with a repair instruction  
- Deterministic dry-run mode  
- Per-run cost visibility  
- Human-in-the-loop by design  

Drafts are **never sent automatically**.

---

## 🔐 Security & Privacy Notes

- Do not include sensitive personal data in sample CSV files
- This tool **does not send emails**
- All outputs remain local to the machine

---

## 🤝 Disclosure: Use of AI Tools

This project was developed using **ChatGPT and GitHub Copilot as productivity tools**.

All system design decisions — including prompt structure, output schemas, error handling, retries, cost tracking, and human-in-the-loop safeguards — were **designed, validated, and implemented by the author**.

AI tools were used to accelerate development, not replace engineering judgment.

---

## 💼 Freelance Use Cases

- Customer support teams  
- Shared inbox triage  
- Pre-processing for ticketing systems  
- Internal operations automation  
- AI-assisted helpdesk workflows  

Custom integrations (CRM, ticketing systems, internal tools) can be built on top.

---

## ⚠️ Disclaimer

AI-generated drafts **must always be reviewed by a human** before sending.  
This tool is designed to assist — not replace — human judgment.

---

## 📄 License

MIT License
