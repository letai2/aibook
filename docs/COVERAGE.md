# Lesson-by-lesson educational audit — 2026-09-21

Audience: a capable Python programmer with no assumed DL, PyTorch, probability
or calculus background. Every lesson's source, rendered lesson/answer, exercise,
project connection and associated visual brief was reviewed. Three independent
read-only reviews covered parts 1–3, 4–7 and 8–10; implementation and integration
were done in the main workspace. This is an editorial judgment, not a learner study.

“Retained” means the existing explanation and exercise met the stated entry
contract; it does not mean the lesson was skipped. Prerequisites below name the
knowledge actually needed, rather than merely the preceding lesson number.

| Lesson | Entry knowledge supplied earlier | Audit disposition |
| --- | --- | --- |
| `01-model` | Python functions, arithmetic, loops | Retained explicit prediction/MSE and changing one weight. |
| `01-learning` | Model, prediction, target, Loss | Retained ML/DL map; new Neuron lesson supplies later numerical depth. |
| `02-token` | Python lists/dictionaries; text | Retained IDs versus numerical magnitude and Unicode distinctions. |
| `03-counts` | IDs and counting | Retained add-one Sampling; milestone's unsmoothed reporting now labeled. |
| `04-splits` | Training versus Inference | Corrected duplicate Validation gloss; clarified independent Test workflow. |
| `05-shape` | Lists and indexing | Retained Scalar/Vector/Matrix/Tensor, B/T/C and semantic axes. |
| `05a-vector-operations` | Shape and indexing | **Added** vector arithmetic, Scalar multiplication, Σ, Weighted sum and list pitfalls. |
| `06-dot` | Vector operations | Retained numerical Dot product and Norm; now has arithmetic prerequisite. |
| `07-matmul` | Dot product and rectangular arrays | Retained non-square Matrix multiplication, Transpose and checks. |
| `08-probability` | Ratios and counting | Expanded uncertainty, conditional sequences and logarithm as inverse power. |
| `09-softmax` | Probability, exp, sums | Retained numerical normalization, stability and invariances. |
| `10-entropy` | Logits, Softmax, logarithms | Added NLL→LogSumExp derivation before stable implementation. |
| `11-derivative` | Functions and finite differences | Added Objective/Loss surface and concrete partial changes; fixed prime-symbol terminology. |
| `12-chain` | Derivative, Gradient | Added shared-weight two-branch calculation and finite-difference check. |
| `12-sgd` | Chain Rule and Parameter update | Retained full/single/mini-batch calculations; restored distinct Mini-batch term. |
| `12b-neuron` | Weighted sum, Gradient, SGD | **Added** manual Neuron→ReLU→Loss→Gradient→update and Layer interpretation. |
| `13-torch` | Hand-written math and one trained Neuron | Linked exact Windows environment workflow; retained zero-background Tensor entry. |
| `14-index-device` | Tensor creation | Retained dtype, indexing, slicing and CPU/GPU contracts. |
| `15-broadcast` | Shape, reduction, indexing | Added silent wrong-axis 3×3 example using only APIs already introduced. |
| `16-reshape` | Tensor axes and broadcasting | Scoped matmul claim; added executable element-wise versus matrix product. |
| `17-autograd` | Manual branching Backpropagation | Retained fresh graphs, accumulation, no_grad, detach and item distinctions. |
| `18-module` | Neuron, Matrix product, Autograd | Added numerical Linear example and explicit link from manual Neuron to Module. |
| `19-network` | Linear/Module and activations | Added affine-composition proof and XOR shapes/42-Parameter count before APIs. |
| `20-loader` | Tensor/Module and Python protocols | Retained Dataset, Batch, Epoch/Step, final Batch and Windows worker caveat. |
| `21-tokenizer` | IDs, dictionaries, split discipline | Retained unknown handling and Unicode; glossary now uses CharacterTokenizer. |
| `22-bpe` | Character Vocabulary | Retained deterministic merges; preserved BPE abbreviation. |
| `23-shift` | Sequence, Dataset, slicing | Retained N−T windows and aligned next-token targets. |
| `24-data-contract` | Train-only Vocabulary and split | Retained fingerprint versus data-quality distinction. |
| `25-embedding` | Parameter, indexing, Matrix product | Added numerical one-hot lookup; protected embedding.weight[ids] spelling. |
| `26-positions` | Embedding and vector addition | Retained additive Shape contract and qualified order claims. |
| `26a-sequence-models` | MLP/Linear, Embedding, position, fixed windows | Added context-MLP versus token-only/RNN bridge; numeric order example and executable real v1 comparison. |
| `26b-sequence-memory` | Vector/Affine/Tanh and sequence | Added recurrence shapes; repaired contextual-representation mistranslation; optional LSTM/GRU. |
| `27-attention-why` | Sequence-memory limitation and Weighted sum | Retained motivated numerical information mixing. |
| `28-qkv` | Learned Linear projections | Added completely worked Query Projection and Parameter/result distinction. |
| `29-scores` | Dot product and Transpose | Retained QKᵀ meanings; moved full Attention lab to after masking lesson. |
| `30-scaling` | Scores and Softmax | Added tiny mean/Variance calculation before statistical scaling argument. |
| `31-values` | Attention weights and Weighted sum | Retained Python/PyTorch numeric comparison and score/weight/output distinction. |
| `32-self` | Complete unmasked Attention | Clarified stored last_weights and provisional language-output contract. |
| `33-mask` | Scores, Boolean Tensor, Softmax | Replaced opaque helper with actual tril/~mask/masked_fill and assertions; full lab now here. |
| `34-causal-test` | SingleHead and causal mask | Replaced premature final MiniGPT with already-taught SingleHead test; clarified Dropout targets. |
| `35-split-heads` | reshape versus transpose | Added negative-axis explanation; retained semantic index assertion. |
| `36-merge-heads` | Head partition and matmul | Retained correct merge, contiguous/view and memory calculation. |
| `37-ffn` | Linear, nonlinear activations, Sequential | Reference explicitly constructs four operations before using project wrapper. |
| `38-residual` | Vector addition and Chain Rule | Added Scalar derivative before optional Jacobian notation. |
| `39-layernorm` | Mean, Variance, feature axis | Added constant-vector executable check; retained population-variance formula. |
| `40-block` | Attention, FFN, Residual, LayerNorm | Retained assembly; repaired diagram's missing intermediate assignment. |
| `41-stack` | One block and registered submodules | Reference now compares independent one-/two-block models and repeats full-model causal test. |
| `42-families` | Causal Transformer blocks | Retained Encoder/Decoder/Cross-Attention distinction. |
| `43-lm-head` | Linear, Vocabulary, NLL | Added explicit (b,t)→b*T+t alignment. |
| `44-parameters` | Module parameter registration and complete block | Derived 12C²+13C component-by-component. |
| `45-trace` | All forward components | Disambiguated output head versus Attention Head and LayerNorm versus Norm. |
| `46-gradient-path` | Autograd and full model | Added runnable detach failure; retained None versus zero distinction. |
| `47-loop` | Model Loss/backward/Optimizer | Added early real CPU recipe, flags and outputs; numeric decoupled Weight decay. |
| `48-evaluate` | eval/no_grad and weighted means | Made B×T target count explicit; retained mode restoration and overlapping-window caveat. |
| `49-rate` | LR, Norm, real training command | Added clipping calculation and reproducible three-rate recipe linked to lesson 47. |
| `49b-schedule` | Steps and LR | Clarified W=0 and optional Warmup with constant schedule. |
| `50-checkpoint` | Module state, Tokenizer, training | Added self-contained temporary-file round trip and prediction/Vocabulary checks. |
| `51-resume` | Checkpoint, Optimizer history, RNG | Added exact 6 versus 4+2 commands and all-state tensor comparison. |
| `52-first-run` | Training, evaluation, saved runs | Retained integration; explained narrower inspection context. |
| `53-curves` | Splits and comparable evaluation | Added independent-Test CLI and descriptive three-seed range; fixed ordinary-sense Token error. |
| `54-generate` | Model logits and context slicing | Added real-checkpoint greedy CLI before Temperature lesson. |
| `55-temperature` | Softmax and autoregressive generation | Defined repeated-trigram denominator and continuation-only scope. |
| `56-topkp` | Probability mass and Sampling | Added explicit list-level renormalization before tensor abstraction. |
| `57-prompts` | Unknown IDs and context cropping | Added runnable encode/crop/print example. |
| `58-ablation` | Causal tests and fair comparisons | Retained controlled experiments; repaired positional-signal terminology. |
| `59-capacity` | Parameter formula and train/validation evidence | Added concrete C/L counts; retained fixed-C Head count distinction. |
| `60-bug-clinic` | dtype, shapes, Cross-Entropy | Added actual failing/fixed class-axis example; repaired symptom terminology. |
| `61-one-batch` | Training loop and debugging | Clarified zero-Dropout condition behind fixed-Loss experiment. |
| `62-journal` | Reproducible experiments | Retained consolidation, prediction/observation and restrained claims. |
| `62b-lifecycle` | Complete Mini-GPT, training, generation, evidence limits | Added required architecture/base/foundation/product map; SFT/preference alternatives, LoRA/RAG distinction. |
| `63-scale` | Model parameter/activation costs | Explicitly signposted optional scope; no cluster prerequisite. |
| `64-cache` | Autoregression, Q/K/V and positions | Clarified old/new T and fixed Quantization gloss. |
| `65-sft` | Shifted targets and Cross-Entropy | Added exact separator/first-answer Loss-mask alignment. |
| `65b-lora` | Linear shapes, Gradient, freezing | Retained low-rank arithmetic and initialization caveats. |
| `66-preference` | SFT and data-dependent objective | Retained introductory, optional preference-learning scope. |
| `67-rag` | Prompt context and model limitations | Retained toy retrieval; added external-content trust boundary. |

## Chapters, checkpoints and references

All 36 chapter indexes inherit the reviewed lesson order. The only new chapter
is appended to part 2, preserving prior chapter and lesson URLs. All 10 part
introductions and checkpoint tasks/answers were checked against their entry
lessons. Their calculations remain consistent: shifted windows, Attention
weights, 7104 Parameters, weighted Loss 1.4, Top-p normalization, KV bytes and
LoRA counts. Checkpoint 8 now follows actual training/resume command scaffolding.

All 84 learning units and all chapter/part/answer/concept/reference continuations
are checked automatically. Diagrams distinguish explanatory fixtures from real
checkpoint data. Glossary inspection uncovered API-name errors and orphaned
examples; key entries now have authored numerical examples, and source-derived
entries include question plus answer and expandable complete lesson context.
The canonicalization layer is tested not to alter formulas, code or API spelling.

The Mini-GPT implementation was reviewed across tokenizer/data contracts,
attention, block assembly, model output, training, checkpoint/resume, evaluation,
Sampling, inspection and stages. No architecture change was needed: the fixes
are explicit independent-Test evaluation, UTF-8 CLI boundaries, milestone
probability labeling and the educational examples. Full test and packaging
evidence is in [REVIEW.md](REVIEW.md).


Second review: all 76 lessons were read across three specialist passes. Current changes add Regression/Classification, fixed-context MLP, Cross-attention dimensions, the training-to-assistant map, precise mean-Loss/SFT-mask interpretation, and selective glossary links. All 12 laboratories were reviewed cell by cell against their connected lessons. See [Jupyter mapping](NOTEBOOKS.md) and [current review evidence](REVIEW.md).
