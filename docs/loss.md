# Loss Function Module Documentation

Purpose
- Measures how "wrong" the algorithm's output is compared to an expected target.
- Represents the foundational mathematical step of evaluating network performance before diving into backpropagation tracking and weight updates.
- Uses Cross-Entropy Loss, which is the standard methodology for vocabulary-oriented token prediction networks.

Available functions
- `cross_entropy_loss(logits, target_id)`:
  - Takes raw prediction logits from the `output_layer`.
  - Normalizes them using Softmax to get probabilities.
  - Calculates the negative log likelihood of the *correct* target token ID.
- `sequence_loss(sequence_logits, target_ids)`:
  - Takes a 2D array of sequence logits and evaluates them incrementally against an expected sequence of correct Target IDs.
  - Returns the average loss across the evaluated valid steps to gauge overall sequence accuracy.

Modification Log (2026-04-08)
1. Read the user request highlighting the need for a mechanism to measure error / "wrongness."
2. Created `loss.py` to establish the mathematical foundation for measuring prediction error.
3. Implemented `cross_entropy_loss` using numerically stable softmax calculations and negative log likelihood handling (with epsilons to avoid catastrophic log(0) domain math errors).
4. Created a `sequence_loss` wrapper to allow error averaging across entire phrase generations.
5. Created `loss.md` documentation satisfying the requirement to formally track modification steps per standard repository preferences.