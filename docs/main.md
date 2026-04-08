# Main Pipeline Conductor Documentation

Purpose
- The algorithm's conductor.
- Imports all 6 main component modules (`tokenizer`, `embeddings`, `attention`, `node`, `run_nodes`, `synthesizer`).
- Wires the modules sequentially running the full pipeline from raw text injection to structural synthesis. 

Implementation Details
- Initializes a sample corpus to populate the `tokenizer` vocabulary.
- Configures shared base intelligence configurations (`embedding_dim`, `EmbeddingMap`, `SelfAttention`).
- Establishes the 6 requested perspective instances (Optimist, Pessimist, Analyst, Philosopher, Innovator, Pragmatist), bridging them with varying system prompts.
- Performs the actual asynchronous concurrent node processing (the fan-out) via `run_nodes.py`.
- Finalizes convergence running the collected arrays into `synthesizer.py`.

Modification Log (2026-04-08)
1. Read `main.md` outlining the conductor responsibility.
2. Drafted standard `main.py` workflow script. 
3. Linked previously coded dependencies smoothly into a central graph.
4. Created 6 explicit perspectives as intended by the architectural design model.
5. Output console trackers dynamically reading shapes and sequence lengths representing functional end-to-end data passage natively.
6. Expanded the training corpus significantly to inject around ~150 distinct words establishing a far more robust vocabulary baseline for embeddings to learn distinct meanings from.
7. Noted exact integration mapping inside `main.md` mapping file modifications.  