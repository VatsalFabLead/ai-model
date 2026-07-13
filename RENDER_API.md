# Custom Model API — Render Deployment Guide

Production API for **custom-nexus-v1**, deployed on [Render](https://render.com) via `render.yaml`.

| Item | Value |
|------|-------|
| **Service name** | `ai-model-api` |
| **Base URL** | `https://ai-model-api-2906.onrender.com` |
| **API prefix** | `/v1` |
| **Interactive docs** | `https://ai-model-api-2906.onrender.com/docs` |
| **HTML API reference** | `https://ai-model-api-2906.onrender.com/api-docs` |
| **OpenAPI JSON** | `https://ai-model-api-2906.onrender.com/openapi.json` |
| **Health check** | `GET /health` (no auth) |
| **Model ID** | `custom-nexus-v1` |

> **Cold starts:** On Render's free tier the service sleeps when idle. The first request after inactivity can take 30–60 seconds.

---

## Authentication

All `/v1/*` endpoints require an API key. Set `API_KEY` in the Render dashboard (auto-generated on first deploy if using `render.yaml`).

Send the key using either header:

```http
Authorization: Bearer YOUR_API_KEY
```

or

```http
X-API-Key: YOUR_API_KEY
```

**401** is returned when the key is missing or invalid.

---

## Rate limits

- **300 requests / minute** per IP (configurable via `RATE_LIMIT` env var)
- Long-running AI endpoints may take up to **180 seconds** (`GUNICORN_TIMEOUT`, `REQUEST_TIMEOUT_SECONDS`)

---

## Quick start

### 1. Check health (no auth)

```bash
curl https://ai-model-api-2906.onrender.com/health
```

```json
{
  "status": "ok",
  "model_ready": true,
  "model_id": "custom-nexus-v1"
}
```

### 2. Chat completion

```bash
curl -X POST "https://ai-model-api-2906.onrender.com/v1/chat/completions" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "custom-nexus-v1",
    "messages": [{"role": "user", "content": "Say hello in one sentence."}],
    "max_tokens": 60
  }'
```

### 3. Unified Nexus invoke (all tools)

Single entry point for every AI tool:

```bash
curl -X POST "https://ai-model-api-2906.onrender.com/v1/nexus/invoke" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "custom-nexus-v1",
    "tool": "seo_content",
    "input": {
      "topic": "Email marketing basics",
      "keywords": "email, marketing",
      "tone": "professional",
      "use_ai": true
    }
  }'
```

Response shape:

```json
{
  "model": "custom-nexus-v1",
  "tool": "seo_content",
  "result": { },
  "elapsed_ms": 1234.5
}
```

---

## Unified Nexus API

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/v1/nexus/status` | Model readiness + tool catalog |
| `GET` | `/v1/nexus/tools` | List all available tools |
| `POST` | `/v1/nexus/invoke` | Run any tool by `tool` id |

### Tool catalog (`tool` values for `/v1/nexus/invoke`)

| Tool ID | Description | Direct endpoint |
|---------|-------------|-----------------|
| `chat` | Chat completions | `POST /v1/chat/completions` |
| `seo_content` | SEO Content Generator | `POST /v1/seo-content/generate` |
| `seo_optimizer` | SEO Content Optimizer | `POST /v1/seo-optimizer/optimize` |
| `title_meta` | SEO Title & Meta | `POST /v1/title-meta/generate` |
| `seo_keywords` | SEO Keyword Generator | `POST /v1/seo-keywords/generate` |
| `schema_markup` | Schema Markup Generator | `POST /v1/schema-markup/generate` |
| `email_new` | Email Assistant — New | `POST /v1/email-assistant/new-email` |
| `email_reply` | Email Assistant — Reply | `POST /v1/email-assistant/reply` |
| `email_cold` | Email Assistant — Cold | `POST /v1/email-assistant/cold-email` |
| `plagiarism_check` | Plagiarism Check | `POST /v1/plagiarism-check/check` |
| `plagiarism_remove` | Plagiarism Remove & Rewrite | `POST /v1/plagiarism-check/remove` |
| `cover_letter` | Cover Letter Generator | `POST /v1/cover-letter/generate` |
| `resume_builder` | Resume Builder | `POST /v1/resume-builder/generate` |

---

## Core endpoints

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `GET` | `/` | No | Service info + doc links |
| `GET` | `/health` | No | Health / model readiness |
| `GET` | `/docs` | No | Swagger UI |
| `GET` | `/v1/models` | Yes | List available models |
| `POST` | `/v1/chat/completions` | Yes | OpenAI-style chat |

### Chat request fields

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| `messages` | array | Yes | `{role, content}` — roles: `user`, `assistant`, `system` |
| `model` | string | No | Default: `custom-nexus-v1` |
| `max_tokens` | int | No | 1–2048 |
| `temperature` | float | No | 0.0–2.0 |
| `top_p` | float | No | 0.0–1.0 |
| `stream` | bool | No | Must be `false` (streaming not supported) |
| `backend` | string | No | `custom`, `ollama`, `llm`, `auto` |

---

## SEO Content Generator

**Prefix:** `/v1/seo-content`

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/categories` | Supported content categories |
| `GET` | `/tones` | Allowed tones |
| `GET` | `/languages` | Supported languages |
| `GET` | `/version` | Generator version |
| `GET` | `/pipeline` | RAG pipeline architecture |
| `GET` | `/schema` | Output structure reference |
| `POST` | `/generate` | Generate SEO article |

**POST `/generate` — key fields**

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `topic` | string | required | 1–300 chars |
| `keywords` | string \| array | — | Target keywords |
| `tone` | string | `professional` | `professional`, `casual`, `friendly`, `formal` |
| `word_count` | int | `1000` | 100–2500 |
| `category` | string | `blog_article` | e.g. `how_to_guide`, `listicle` |
| `language` | string | — | e.g. `English`, `Hindi` |
| `use_ai` | bool | `true` | Custom model polish |
| `use_rag` | bool | `true` | Open-dataset retrieval |
| `discover_keywords` | bool | `false` | Auto-discover keywords from web |

---

## SEO Content Optimizer

**Prefix:** `/v1/seo-optimizer`

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/categories` | Supported categories |
| `GET` | `/tones` | Allowed tones |
| `GET` | `/languages` | Supported languages |
| `GET` | `/version` | Generator version |
| `GET` | `/pipeline` | RAG pipeline architecture |
| `POST` | `/optimize` | Analyze and optimize existing content |

**POST `/optimize` — key fields**

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `content` | string | required | 1–12000 chars |
| `keywords` | string \| array | — | Target keywords |
| `tone` | string | `professional` | |
| `language` | string | — | |
| `category` | string | `blog_article` | |
| `use_ai` | bool | `true` | |
| `use_rag` | bool | `true` | |

Returns `optimized_content`, `seo_score_before` / `seo_score_after`, `suggestions`, `issues_before` / `issues_after`, metadata, FAQs, internal links, and pipeline analysis.

---

## SEO Title & Meta

**Prefix:** `/v1/title-meta`

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/categories` | Supported categories |
| `GET` | `/tones` | Allowed tones |
| `GET` | `/languages` | Supported languages |
| `GET` | `/version` | Generator version |
| `GET` | `/pipeline` | Pipeline architecture |
| `POST` | `/generate` | Generate title + meta variations |

**POST `/generate` — key fields:** `topic` (required), `variations` (10–50), `tone`, `category`, `language`, `use_ai`, `use_rag`

---

## SEO Keywords

**Prefix:** `/v1/seo-keywords`

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/version` | Generator version |
| `GET` | `/pipeline` | Pipeline architecture |
| `POST` | `/generate` | Generate keyword list |

**POST `/generate` — key fields:** `seed_keyword` (required), `variations` / `max_items` (10–50), `discover_web`, `use_ai`, `use_rag`

---

## Schema Markup

**Prefix:** `/v1/schema-markup`

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/types` | Schema.org types (optional `?category=`) |
| `GET` | `/categories` | Type categories |
| `GET` | `/languages` | Supported languages |
| `GET` | `/properties` | Properties per schema type |
| `GET` | `/version` | Generator version |
| `GET` | `/pipeline` | Pipeline architecture |
| `POST` | `/generate` | Generate JSON-LD |

**POST `/generate` — key fields:** `schema_type`, `name` (required), `data` (object), `language`, `ai_enhance`, `use_rag`

---

## Email Assistant

**Prefix:** `/v1/email-assistant`

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/version` | Generator version |
| `GET` | `/pipeline` | Pipeline architecture |
| `POST` | `/new-email` | Compose new email from context |
| `POST` | `/reply` | Reply to an email |
| `POST` | `/cold-email` | Cold outreach email |

**Tones:** `professional`, `casual`, `friendly`, `formal`

| Endpoint | Required fields |
|----------|-----------------|
| `/new-email` | `context` |
| `/reply` | `original_email` |
| `/cold-email` | `company_name`, `purpose_offer`, `value_proposition` |

---

## Plagiarism Checker

**Prefix:** `/v1/plagiarism-check`

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/check` | Scan content for similarity |
| `POST` | `/remove` | Rewrite plagiarized content |

**Body:** `{ "content": "..." }` — minimum 40 characters, max 50,000.

---

## Resume Builder

**Prefix:** `/v1/resume-builder`

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/templates` | Available resume templates |
| `GET` | `/version` | Generator version |
| `GET` | `/pipeline` | Pipeline architecture |
| `POST` | `/generate` | Generate resume |

**POST `/generate` — required:** `full_name`, `job_title`, `email`, `phone`

**Optional:** `linkedin`, `portfolio`, `education`, `experience`, `skills`, `summary`, `projects`, `certifications`, `achievements`, `languages`, `template` (`modern`, `classic`, `executive`, `minimal`, `creative`), `use_ai`, `use_rag`

---

## Cover Letter

**Prefix:** `/v1/cover-letter`

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/version` | Generator version |
| `GET` | `/pipeline` | Pipeline architecture |
| `POST` | `/generate` | Generate cover letter |

**POST `/generate` — required:** `job_role`, `company_name`, `skills_experience`

---

## Post Scheduler

**Prefix:** `/v1/post-scheduler`

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/platforms` | Supported social platforms |
| `POST` | `/suggest-content` | AI post copy for a platform |
| `POST` | `/suggest-hashtags` | Hashtag suggestions |
| `POST` | `/generate` | Post + hashtags combined |

**Platforms:** `instagram`, `linkedin`, `twitter`, `tiktok`, etc.

---

## Web UI pages (no API prefix)

| Path | Description |
|------|-------------|
| `/chat_page` | Chat hub HTML UI |
| `/model_test` | Model test page |
| `/model_test/status` | Model test status JSON |

---

## HTTP status codes

| Code | Meaning |
|------|---------|
| `200` | Success |
| `400` | Invalid request / validation error |
| `401` | Missing or invalid API key |
| `429` | Rate limit exceeded |
| `500` | Server / inference error |
| `501` | Not implemented (e.g. `stream: true` on chat) |
| `503` | Model loading or unavailable |

---

## Test all endpoints locally or on Render

```powershell
# Local
$env:API_KEY = "your-key"
.\scripts\test_all_apis.ps1

# Render production
.\scripts\test_all_apis.ps1 -BaseUrl "https://ai-model-api-2906.onrender.com" -ApiKey "YOUR_KEY"
```

---

## Deploy to Render

1. Push this repo to GitHub (`main` branch).
2. In Render: **New → Blueprint** → select repo → apply `render.yaml`.
3. Copy `API_KEY` from the Render dashboard **Environment** tab.
4. Auto-deploy is enabled on every push to `main`.

Key env vars (set in `render.yaml`):

| Variable | Production value |
|----------|------------------|
| `API_PREFIX` | `/v1` |
| `MODEL_ID` | `custom-nexus-v1` |
| `MODEL_BACKEND` | `custom` |
| `RATE_LIMIT` | `300/minute` |
| `GUNICORN_TIMEOUT` | `180` |
| `API_KEY` | Auto-generated (set manually if needed) |

---

## Example: Nexus invoke for each tool

```json
// seo_optimizer
{ "tool": "seo_optimizer", "input": { "content": "Your article text...", "keywords": ["ERP", "manufacturing"], "use_ai": true } }

// title_meta
{ "tool": "title_meta", "input": { "topic": "Electric Vehicles", "variations": 10, "use_ai": false } }

// seo_keywords
{ "tool": "seo_keywords", "input": { "seed_keyword": "digital marketing", "max_items": 10, "discover_web": true } }

// email_new
{ "tool": "email_new", "input": { "subject": "Update", "context": "Inform client about delay.", "tone": "professional" } }

// resume_builder
{ "tool": "resume_builder", "input": { "full_name": "Jane Doe", "job_title": "Developer", "email": "j@example.com", "phone": "+1-555-0100", "use_ai": true } }

// plagiarism_check
{ "tool": "plagiarism_check", "input": { "content": "At least forty characters of text to scan for similarity..." } }
```
