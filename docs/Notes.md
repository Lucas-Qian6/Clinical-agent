<!--
Tip: keep it scannable. One strong header, one punchy subtitle, then quick links.
-->
<h1 align="center">Emotion-based Clinical Agent</h1>
<p align="center">
  Dectect emotion -> Reinforce Learning -> LLM judge
</p>


 🎯 Background
- We want to create a clinical agent that understands human emotion, mood and sentiment in a deeply parameterized manner.
- **Potential Strategy:** Mapping human "states" into embeddings. It provides a representation layer that allows for vectorization and directionality in terms of modifying human behavior and modulating the human condition.

 🔬 Market Research
 - Oracle Health’s Clinical AI Agent is focused on streamlining / augmenting existing clinical workflows in a broad sense. The focus is on integration into workflows leveraging EHRs and focuses on things like charting, documentation, and medication and order management. 

 🧰 Stack
- API: `ChatGPT`, `Gemini`, `Huggingface`
- Torch
- Colab


 🧭 What We did
**10/2025 - 01/2026**
- **Aim:** Research on how to map discrete emotion classification `GoEmotion` into continuous values.
- 1) For a single sentence, used generative model `SamLowe/roberta-base-go_emotions` (Teacher Model) to get 28 emotion probabilities.
  2) Clinical interpretable look-up table: a fixed valence weight for each emotion `v`.
  3) Computed score = p * v for each emotion.
  4) Used MPNet `all-mpnet-base-v2` to score into 784-d vector.
  5) Trained models to predict the valence scores.
- Found `Hourglass model`, using 4-dimenstion to descirbe each emotion.

**01/2026 - 02/2026**
- **Aim:** Synthesize the dataset and make small benchmark using DPO.
- Start with DPO trainer since it's the simplest.
- 1) Found `MentalChat16K` dataset, which contains single turn: Patient - Therapist.
  2) Used `Gemini-2-flash` to generate clinically bad responses.
  3) Implemented `Weaver` (ensemble of weak verifiers) to filter our low-quanlity bad responses: 
  - Checks Semantic Consistency with Gold Standard: `mental/mental-bert-base-uncased`
  - Checks for warmth, empathy, and lack of judgment: `hadresh-savani/bert-base-uncased-emotion`
  - Checks if it's safety, leading users to suicide: `unitary/unbiased-toxic-roberta`
  - Checks adherence to CBT protocols (Heuristic): `Clinical_Protocol`
  - Checks Logical Coherence: `cross-encoder/ms-marco-MiniLM-L-6-v2`
  4) Got 500 data and trained the `unsloth/llama-3-8b-instruct-bnb-4bit`, using peft_model to decrease the amount of learnable parameters.
  5) Evaluate using the `Weaver`.

**02/2026 - Present**
- **Aim:** Set up the benchmark using DPO
- Found look-up table in research of using `SenticNet` to map Goemotion to Hourglass model.
- Existing emotion detection models classify emotions into 28 dimension `GoEmotion`.
- Use `bhadresh-savani/bert-base-go-emotion` to detect emotions from patient's sentence and used look-up table to transfer them into Hourglass dimensions.
- Found `Psychotherapy-LLM/PsychoCounsel-Preference`, an existing clinical dataset for training DPO models. So I detected and added emotions vectors to each row of this dataset.
- Trained on the same dataset above, `Psychotherapy-LLM/PsychoCounsel-Llama3-8B` becomes my baseline model. Their paper also reveal that we can use LLM like `ChatGPT`  as judge when evaluating the performance
- `TraitBasis` uses activations in the model to recognize human traits (emotions). 
So I added an emotional layer to the pre-trained model `Llama3-8B` and trained it using the same dataset.
**Problems:** Trained model's winning rate is only 5%. And it seems since I wrapped the model with peft mode, it only updated a few learnable parameters, and omit other parameters including my customized layers.

**Found:** Small language models (SLMs) + test-time scaling (TTS) + verification > large language models (LLMs).

**What I tried**
- Steps:
  1. …
  2. …
  3. …
**What failed**
- Error: `...`
- Symptom: “...”
- Root cause (in one sentence): …
**Fix**
- Change: …
- Why it works: …
**Takeaway**
- ✅ …
- ❌ …
 Attempt 2 — “Dockerize early”
(Repeat the same pattern)
 🚀 Setup
 Prereqs
- `docker` + `docker compose`
- `node` / `python` / `java`
- (Optional) `make`
 Quickstart
cp .env.example .env
docker compose up --build
Verify
- Open http://localhost:3000/health
- Expect: {"status":"ok"}
🧯 Debug Notes
| Problem | Symptom | Fix |
|---|---|---|
| DB connection refused | API keeps restarting | Check compose network + correct DATABASE_URL |
| JWT invalid | 401 on refresh | Verify signing key + clock drift |
🗺️ Next Steps
- [ ] Add rate limiting
- [ ] Add structured logging + trace IDs
- [ ] Add CI (tests + lint)
- [ ] Deploy + run a load test
📄 License
MIT
Where to get “nice-looking icons”
- Use Shields badges (fastest): `https://shields.io/`
- Use simple section icons (emoji) in headers like `✨ 🎯 🧰 🧭 🚀 🧯 🗺️` (works everywhere)
Where to get pretty images
- Diagrams: Excalidraw, draw.io, Figma
- Screenshots: clean terminal theme + crop + consistent width
- GIFs: `ffmpeg` / `peek` / `kap` (Mac) / ScreenToGif (Windows)
---
**2) “Story-style” template (good for Zhihu / Medium)**
```md
# I Built a Backend From Scratch — Here’s the Ugly Truth (and the Fixes)
> I documented every wrong turn on purpose, because that’s where the learning was.
## 0. What I was trying to build
- Goal:
- Constraints:
- Success criteria:
## 1. My first idea (and why it felt right)
- Reasoning:
- What I expected to happen:
## 2. First failure
**Symptom**
- …
**What I tried**
1) …
2) …
3) …
**What I searched**
- Keywords I used:
  - "..."
  - "..."
- The one concept I didn’t understand yet:
  - …
**Root cause**
- One sentence:
**Fix**
- …
**Lesson**
- …
## 3. Second failure (repeat)
…
## 4. The final working approach
- What changed:
- Why it worked:
- Trade-offs:
## 5. What I’d do next
1) …
2) …
3) …
## Appendix: Commands / Config snippets
```bash
# …
---
**3) “Nice words” mini-phrases (use these in headings)**
- “What I believed vs what was true”
- “The moment it broke”
- “The real root cause”
- “The fix that finally stuck”
- “Trade-offs I accepted”
- “If I restarted today”
---
**4) Enumerations that look good**
- Use “short lead + detail” pattern:
  1) Bold lead: explanation.
  2) Bold lead: explanation.
- Use checklists for roadmap:
  - [ ] add CI
  - [ ] add metrics
- Use tables for common issues (very readable).
---
**5) Special symbols / separators (pick 1 style and stick to it)**
- Section dividers: `---` or `***`
- Callouts (simple, clean):
  - `Note:` …
  - `Pitfall:` …
  - `Rule:` …
- Unicode symbols (use sparingly):
  - Arrows: `→` `↳`
  - Checks: `✓` `✗`
  - Bullets: `•` `·`
  - Emphasis: `—` (em dash), `…`
  - Brackets: `【】` (CN style)
---