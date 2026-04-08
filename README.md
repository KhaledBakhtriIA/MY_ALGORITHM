# Multiperspective Synthesis Algorithm

Welcome to the repository for a novel algorithm architecture designed to process data from multiple unique perspectives simultaneously. This project breaks down a single input into six distinct evaluations (nodes), processes them using custom attention mechanisms, and synthesizes them into a converged output.

## System Architecture & Data Flow

This algorithm operates on a "fan-out, fan-in" paradigm, where one input query becomes six simultaneous perspectives that are finally woven together. 

### 1. Preprocessing (`tokenizer.py`)
**Role:** Converts raw text data into integer token IDs.
Raw text and questions are processed here so the models only read numbers. It uses a custom vocabulary builder with special handled tokens (`<PAD>`, `<UNK>`).

### 2. Semantic Mapping (`embeddings.py`)
**Role:** Converts token IDs into continuous vectors (floating-point numbers).
This is where the meaning starts to take shape. Similar concepts end up with similar vectors in the mapped space. It creates the core matrix that maps every token to a vector.

### 3. Core Intelligence (`attention.py`)
**Role:** Determines relevance and "bonds". 
Given a sequence of vectors, the attention module calculates which parts of the data are mathematically relevant to other parts, measuring resemblance between tokens across the provided data.

### 4. Perspective Instances (`node.py`)
**Role:** The individual perspective engines.
A Node takes a system prompt plus the input data, runs them through the embedding and attention modules, and returns an output representing that specific perspective's reading of the data. 

### 5. The Fan-Out (`run_nodes.py`)
**Role:** Concurrent execution supervisor.
This module instantiates and runs all 6 `Node` perspectives on the same input simultaneously, collecting their individual outputs.

### 6. The Convergence (`synthesizer.py`)
**Role:** Merges the distinct perspectives into a final result.
It takes the 6 outputs and combines them. Initially, this performs an averaging operation, but it is architected to later find tension, contradictions, and agreements between the different perspective nodes.

### 7. The Conductor (`main.py`)
**Role:** The overarching pipeline.
Imports all components and runs the full pipeline—from raw text ingestion to the final synthesized output.

---

## Modification Log
* **2026-04-08:** Created detailed architectural README mapping out the responsibilities of each component (Tokenizer, Embeddings, Attention, Node, Run Nodes, Synthesizer, and Main)