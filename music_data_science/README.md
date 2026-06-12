# Music Data Science (MDS)

Stem-oriented orchestration, data, and memory layer on top of the
[ACE-Step 1.5](https://github.com/cazador0/ACE-Step-1.5) music generation engine.

MDS turns ACE-Step's task primitives (`text2music`, `repaint`, `cover`, `extract`,
`lego`, `complete`) into a stem-manipulation workflow: describe a song as a structured
**SongBlueprint**, render it, pull it apart into stems, and make prompt-driven changes
to individual stems ("make the drums punchier from 0:45 to 1:10") — with every
generation scored, recorded, and reproducible.

This is part 2 of the three-repo MDS system:

| Repo | Role |
| --- | --- |
| [cazador0/mcp-servers](https://github.com/cazador0/mcp-servers) (`src/scrum-master`, branch `claude/peaceful-dijkstra-m0sycf`) | Engagement layer: agent communication / Scrum-Master MCP server |
| [cazador0/awesome-python-audio](https://github.com/cazador0/awesome-python-audio) (branch `claude/peaceful-dijkstra-m0sycf`) | UI layer: Obsidian vault + planning docs (the markdown corpus this package's RAPTOR memory indexes) |
| **this package** (`music_data_science/` in cazador0/ACE-Step-1.5) | Business/science logic: stem workflows, data, memory, evaluation |

## Architecture

```text
+---------------------------------------------------------------+
|  ENGAGEMENT LAYER                  UI LAYER                    |
|  Agents via MCP                    Obsidian vault              |
|  (cazador0/mcp-servers,            (cazador0/                  |
|   src/scrum-master)                 awesome-python-audio)      |
+-------------------+----------------------+--------------------+
                    |                      |
                    v                      v
+---------------------------------------------------------------+
|  DATA LAYER                 music_data_science.data            |
|  Redis hot cache (optional, in-memory fallback)  -- cache.py   |
|  pandas telemetry frames, csv/parquet round-trip -- frames.py  |
+-------------------------------+-------------------------------+
                                |
                                v
+---------------------------------------------------------------+
|  MEMORY & CONTEXT LAYER     music_data_science.memory          |
|  SQLite system-of-record (sessions, stems,       -- store.py   |
|    generations, scores)                                        |
|  RAPTOR-lite summary tree over the markdown      -- raptor.py  |
|    vault + Best-of-N context retrieval                         |
+-------------------------------+-------------------------------+
                                |
                                v
+---------------------------------------------------------------+
|  ORCHESTRATION              music_data_science.{stems,         |
|  prompt -> StemEdit/EditPlan          pipeline, evaluation}    |
|  StemSession workflows: generate / separate /                  |
|    regenerate / add_layer / best_of_n / recombine              |
|  Best-of-N scoring: metadata adherence + rubric                |
+-------------------------------+-------------------------------+
                                |  HTTP: /release_task, /query_result
                                v
+---------------------------------------------------------------+
|  GENERATION ENGINE          ACE-Step 1.5 API server            |
|  task types: text2music, repaint, cover, cover-nofsq,          |
|              extract, lego, complete                           |
+---------------------------------------------------------------+
```

## Prompt -> Stem translation and the certainty problem

Being honest up front: **100% certainty that a generative model's output matches a
user's intent is an open research problem.** Diffusion/LM music generation is
probabilistic; no prompt translation layer can guarantee that "punchier drums" sounds
punchier to *you*. What MDS provides instead is an engineering path that converges on
intent iteratively and measurably:

1. **Structured intermediate representation.** Prompts are never sent raw. They are
   translated into a `SongBlueprint` / `StemEdit` — typed objects with explicit stem
   classes, operations, and time windows — so what *will* be asked of the engine is
   inspectable before any GPU time is spent.
2. **Constrained metadata control.** BPM, key/scale, time signature, duration, and
   `track_classes` are passed explicitly, turning "vibes" into checkable constraints.
3. **Reproducibility: seed pinning + `audio_code_string` caching.** Pinned seeds make
   a generation repeatable; cached audio semantic codes (`mds:codes:{seed}:{hash}`)
   let the same musical material be re-rendered or re-edited without re-rolling the dice.
4. **Minimal-diff edits via windowed repaint.** Edits map to `repaint` with an explicit
   `repainting_start`/`repainting_end` window and a preservation mode
   (`conservative`/`balanced`/`aggressive`), so "tweak the bass in the bridge" cannot
   rewrite the chorus.
5. **Best-of-N with automatic scoring and acceptance thresholds.** Each edit can be run
   N times with distinct pinned seeds; every candidate is scored on metadata adherence,
   a professional-standards rubric, and (optionally) embedding similarity. Candidates
   below the acceptance threshold are rejected, and every score breakdown is persisted —
   so the loop "generate, measure, keep the best, refine the blueprint" replaces the
   fantasy of one-shot certainty.

## Install

```sh
cd music_data_science
uv venv && uv pip install -e .          # core: pydantic, httpx, pandas, numpy
uv pip install -e ".[data]"             # optional: redis, sqlite-vec
```

## Quickstart

Start the ACE-Step API server first (from the repository root):

```sh
./start_api_server.sh      # serves /release_task + /query_result, default port 8001
```

Generate a song, then change one stem:

```python
from music_data_science.pipeline.client import AceStepClient
from music_data_science.pipeline.orchestrator import StemSession
from music_data_science.stems.models import SongBlueprint, StemClass, StemSpec
from music_data_science.stems.translator import PromptTranslator

blueprint = SongBlueprint(
    global_caption="warm lo-fi hip hop with dusty drums and a mellow rhodes",
    bpm=84,
    key_scale="F major",
    time_signature="4",
    duration=120.0,
    stems=[StemSpec(stem_class=StemClass.DRUMS, caption="dusty boom-bap kit")],
)

client = AceStepClient("http://127.0.0.1:8001")
session = StemSession(client, blueprint, name="lofi-demo")

session.generate()                          # text2music
session.separate()                          # extract -> vocals/drums/bass/guitar stems

plan = PromptTranslator().translate("make the drums punchier from 0:45 to 1:10")
session.regenerate(plan.edits[0])           # windowed repaint on the drum stem
```

Best-of-N with automatic scoring:

```python
from music_data_science.evaluation.scorer import BestOfNScorer

scorer = BestOfNScorer(expected=blueprint, threshold=0.7)
candidates = session.best_of_n(plan.edits[0], n=4, scorer=scorer)
winner = candidates[0]
print(winner.breakdown.total, winner.breakdown.components, winner.breakdown.passed)
```

Retrieve planning context from the Obsidian vault with RAPTOR-lite:

```python
from music_data_science.memory.raptor import RaptorTree

tree = RaptorTree("path/to/obsidian-vault").build()
decision = tree.best_of_n("which drum sound did we settle on for the chorus?", n=5)
print(decision.winner.text if decision.winner else "no context found")
for candidate in decision.trace:            # full score trace: relevance/recency/authority
    print(candidate.total, candidate.node.source)
```

## Package map

| Module | Responsibility |
| --- | --- |
| `stems/models.py` | `StemClass` (aligned with ACE-Step `TRACK_NAMES`), `StemSpec`, `SongBlueprint`, `StemEdit`, `EditPlan` |
| `stems/translator.py` | Deterministic prompt grammar -> `StemEdit`/`EditPlan`; `to_request()` emits `GenerateMusicRequest` payloads; optional `llm_refine` hook |
| `pipeline/client.py` | `AceStepClient`: `/release_task` submit + `/query_result` polling |
| `pipeline/orchestrator.py` | `StemSession`: generate / separate / regenerate / add_layer / best_of_n / recombine |
| `pipeline/mixdown.py` | Local WAV stem recombination (sum + peak-normalize) |
| `memory/store.py` | SQLite system-of-record: sessions, stems, generations, scores |
| `memory/raptor.py` | RAPTOR-lite tree + Best-of-N context retrieval over the vault |
| `memory/vector_store.py` | Embedded SQLite vector store (Strategy A durable vectors); RAPTOR persist/restore |
| `data/cache.py` | Redis adapter with in-memory fallback; documented keyspace |
| `data/queue.py` | Redis Streams task queue with consumer groups; in-memory fallback |
| `data/frames.py` | pandas telemetry frames; csv/parquet round-trip |
| `evaluation/scorer.py` | Best-of-N scoring: metadata adherence, rubric, embedding hook |
| `integrations/refine.py` | Optional Claude-backed blueprint/plan refinement via structured outputs (`[llm]` extra) |
| `examples/end_to_end.py` | Full-loop walkthrough; `--dry-run` exercises every layer without a server |

## Tests

Stdlib `unittest`, no network/GPU/model downloads (HTTP is mocked with
`httpx.MockTransport`):

```sh
cd music_data_science
uv run --with pydantic --with httpx --with pandas --with numpy \
  python -m unittest discover -s music_data_science -p "*_test.py" -t .
```
