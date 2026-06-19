# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Open-Vocabulary Food Quality Detection POC — evaluate whether YOLO-World can detect food freshness/quality states (e.g. "moldy bread", "fresh banana") via natural language prompts, without custom training. Research/experimentation tool, not production.

## Stack

- **Frontend**: Next.js + Tailwind CSS
- **Backend**: FastAPI (Python)
- **Model**: YOLO-World via Ultralytics
- **Image processing**: OpenCV, Pillow
- **Containerization**: Docker

## Architecture

```
Frontend (Next.js)  →  POST /detect (multipart: image + prompts + threshold)
                    →  FastAPI backend
                    →  YOLO-World inference (Ultralytics)
                    →  Returns: detections JSON + annotated image URL
```

Frontend layout: left panel (upload + prompt input + threshold slider + detect button), right panel (annotated image + detection results table).

## API Contract

### POST /detect

Request: multipart form-data with `image` (JPG/PNG ≤10MB), `prompts` (comma-separated string), `threshold` (float, default 0.25).

Response:
```json
{
  "detections": [
    { "label": "moldy bread", "confidence": 0.84, "bbox": [120, 45, 340, 280] }
  ],
  "annotated_image_url": "/outputs/result_001.jpg"
}
```

## Dev Commands

Once scaffold exists, expected commands:

**Backend:**
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

**Frontend:**
```bash
cd frontend
npm install
npm run dev      # starts on port 3000
npm run build
npm run lint
```

**Docker (full stack):**
```bash
docker compose up --build
docker compose down
```

## Key Constraints

- Single image upload only (no video, no streaming)
- No auth, no multi-user
- Confidence threshold slider default: 0.25, adjustable dynamically
- Inference target: <5 seconds
- Graceful error handling for: invalid images, empty prompts, model failures

## Engineering Principles (from PRD)

Prioritize fast iteration and simplicity. Avoid microservices, premature optimization, unnecessary abstractions. This is a hackable research tool.

## Development Phases

1. Backend YOLO-World inference working
2. Frontend upload + visualization
3. Prompt experimentation support
4. Logging + evaluation tools (store image, prompts, detections, timestamps)
5. Optional VLM fallback (YOLO low-confidence → Gemini/GPT-4V)

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.
