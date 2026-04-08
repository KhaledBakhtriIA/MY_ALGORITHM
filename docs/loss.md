# Loss Function Module Documentation

Purpose
- Measures how "wrong" the algorithm's output is compared to an expected target.
- Represents the foundational mathematical step of evaluating network performance before diving into backpropagation tracking and weight updates.
- Uses Cross-Entropy Loss, which is the standard methodology for vocabulary-oriented token prediction networks.

Available functions
- `cross_entropy_loss(predictions, targets)`:
  - Takes prediction probabilities and correct target IDs.
  - Returns a single loss number.
  - For each position: sets `target_token` = correct next word, `predicted_probability` = what model gave that word, `loss` = -log(predicted_probability). Returns the average loss metric across the sequence.

Modification Log (2026-04-08)
1. Initially created `loss.py` to establish the mathematical foundation measuring prediction error using log likelihoods.
2. Refactored `loss.py` to match the exact mathematical standard specified: a single `cross_entropy_loss(predictions, targets)` taking strictly probability distributions instead of raw logits, looping over length, extracting `target_token` probability, and evaluating `-log(predicted_probability)` per positional guess.
3. Updated `train.py` to calculate Softmax probability matrices inside the loop before passing arrays to the new unified loss function wrapper.
4. Reflected strict API requirement constraint compliance inside this document.