# Perspective Node Module

Purpose
- The individual perspective engine instance.
- Takes a system prompt and input data, pushing them through embedding and attention layers.
- Outputs a contextualized vector representation denoting its specific reading of the data.
- Forms the core component for the fan-out architecture.

Available classes/objects
- `PerspectiveNode(name, system_prompt, vocab, embedding_map, attention_module)`: 
  - Initializes a perspective with a designated name and prompting constraint, sharing the underlying vocab and embedding maps of the main system.
- `process(data)`:
  - Takes a raw input string from the user, prefixes it with the node's system prompt, tokenizes, embeds, and runs mathematical attention on the prompt-data bond.
  - Returns a dictionary with the node's name, prompt, and output vectors (the perspective representation).

Modification Log (2026-04-08)
1. Read the `node.md` documentation outlining the system prompt instance and fan-out goals.
2. Created standard Python class implementation in `node.py`. 
3. Connected previous modules: `tokenizer.encode`, `EmbeddingMap`, and `SelfAttention`.
4. Orchestrated pipeline sequentially per node: combine text -> encode IDs -> embed sequences -> compute attention.
5. Documented modification updates systematically tracking today's progress per user preference.
6. (2026-04-13) Implemented `RelationalPerspectiveNode` within `node.py`. It is an upgraded node which acts to model relationships between concepts rather than specific concepts themselves. It has an internal `relation_memory` and encodes pairs using a custom linear translation mapper (`self.W_rel` and `self.b_rel`). Uses cosine similarity to retrieve closest matching stored relationships when queried with single concept queries. 
