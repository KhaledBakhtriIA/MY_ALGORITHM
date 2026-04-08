Tokenizer module documentation

Purpose
- Converts raw text into tokens and token ids so other algorithm modules can process numeric input.

Available functions
- tokenize_words(text): splits input text into lowercase word tokens.
- build_vocab(lines): builds a token to id mapping from multiple text lines.
- encode(text, vocab): converts a text string into a list of ids using the vocabulary.
- build_id_to_token(vocab): reverses a vocabulary mapping from token-to-id into id-to-token.
- decode(token_ids, id_to_token): converts ids back into a readable token string.

Special tokens
- <PAD> has id 0.
- <UNK> has id 1.

Modification log (2026-04-08)
1. Inspected tokenizer source and identified syntax and indentation errors.
2. Replaced incomplete loop-only implementation with complete callable functions.
3. Added tokenization, vocabulary creation, encoding, and decoding flow.
4. Added <PAD> and <UNK> handling for stable unknown-token behavior.
5. Added `build_id_to_token` utility to decouple decoding dependencies.
6. Updated this documentation to reflect implementation and record each step.