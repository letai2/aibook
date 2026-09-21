"use strict";
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const { softmax, tokenization, attention, tensorShapes, validateInspection, conceptSegments } = require("../../book_src/assets/experience.js");

let count = 0;
function test(name, action) { action(); count++; console.log(`ok ${count} - ${name}`); }
function near(a, b, tolerance = 1e-10) { assert.ok(Math.abs(a - b) < tolerance, `${a} versus ${b}`); }
function clone(value) { return JSON.parse(JSON.stringify(value)); }

test("authored concept labels preserve literal text and prefer full terms", () => {
  const concepts = {Attention:"attention", "Self-Attention":"self-attention", Token:"token"};
  const text = "Self-Attention و Tokenها؛ Attention، attention.py، someToken <script>";
  const segments = conceptSegments(text,concepts);
  assert.equal(segments.map(segment => segment.text).join(""),text);
  assert.deepEqual(segments.filter(segment => segment.slug).map(segment => segment.slug),["self-attention","token","attention"]);
  assert.deepEqual(conceptSegments("plain",{}),[{text:"plain"}]);
});

test("stable softmax, shift invariance and temperature", () => {
  const ordinary = softmax([2, 1, 0]);
  near(ordinary.reduce((a, b) => a + b), 1);
  ordinary.forEach((value, i) => near(value, softmax([1002, 1001, 1000])[i]));
  assert.ok(softmax([2, 1, 0], .5)[0] > ordinary[0]);
  assert.ok(softmax([2, 1, 0], 2)[0] < ordinary[0]);
  assert.deepEqual(softmax([0, -Infinity]), [1, 0]);
  assert.throws(() => softmax([-Infinity, -Infinity]));
  assert.throws(() => softmax([1, NaN]));
  assert.throws(() => softmax([1, 2], 0));
});

test("Unicode codepoints, fixed train vocabulary and unknown symbols", () => {
  const a = tokenization("cats 🐈"), b = tokenization("می‌رود"), c = tokenization("ي ی ك ک");
  assert.equal(a.tokens.length, 6); assert.equal(a.ids.at(-1), 0);
  assert.equal(b.tokens[2], "\u200c"); assert.notEqual(b.ids[2], 0);
  assert.equal(c.ids[0], 0); assert.notEqual(c.ids[2], 0);
  assert.deepEqual(a.vocabulary, b.vocabulary);
  assert.deepEqual(a.vocabulary.slice(1), a.vocabulary.slice(1).sort((x, y) => x.codePointAt(0) - y.codePointAt(0)));
});

test("manual QKV transpose, mask, row sums and weighted values", () => {
  const masked = attention(true), free = attention(false);
  assert.deepEqual(masked.raw, [[1, 2, 0], [1, 0, 1], [2, 2, 1]]);
  assert.deepEqual(masked.weights[0], [1, 0, 0]);
  assert.deepEqual(masked.output[0], [1, 0]);
  assert.equal(masked.weights[1][2], 0); assert.ok(free.weights[1][2] > 0);
  masked.weights.forEach(row => near(row.reduce((a, b) => a + b), 1));
  assert.deepEqual(masked.weights[2], free.weights[2]);
  near(masked.output[1][0], masked.weights[1][0]); near(masked.output[1][1], 2 * masked.weights[1][1]);
});

test("tensor shapes and quadratic score-matrix bytes", () => {
  const first = tensorShapes(2, 8, 32, 4), twice = tensorShapes(2, 16, 32, 4);
  assert.deepEqual(first.heads, [2, 4, 8, 8]);
  assert.equal(first.scoreBytes, 2 * 4 * 8 * 8 * 4); assert.equal(twice.scoreBytes, 4 * first.scoreBytes);
  assert.throws(() => tensorShapes(2, 8, 31, 4));
  assert.throws(() => tensorShapes(2, 8, 32, 0));
  assert.throws(() => tensorShapes(2.5, 8, 32, 4));
});

// A deliberately small complete fixture tests the contract without Python or optional run artifacts.
function fixture() {
  const probability = softmax([0, 1]);
  return {
    schema: "mini-gpt-inspection-v1",
    source: {kind: "checkpoint", checkpoint_name: "test.pt", checkpoint_sha256: "a".repeat(64), step: 1,
      format_version: 2, torch_version: "2.14.0+cpu", config: {vocab_size: 2, context_length: 1,
        embedding_dim: 2, num_heads: 1, num_layers: 1, dropout: 0}, training_metadata: {purpose: "test fixture"}},
    input: {prompt: "a", full_token_ids: [1], context_token_ids: [1], context_tokens: ["a"], unknown_positions: [], truncated: false, context_limit: 1},
    shapes: {token_ids: [1, 1], embedding: [1, 1, 2], qkv: [1, 1, 1, 2], scores: [1, 1, 1, 1], logits: [1, 1, 2]},
    embeddings: {token: [[1, 0]], position: [[0, 1]], combined: [[1, 1]]},
    layers: [{index: 0, input_shape: [1, 1, 2], output_shape: [1, 1, 2], attention_shape: [1, 1, 1, 1], feed_forward_shape: [1, 1, 2]}],
    attention: {q: [[1, 0]], k: [[1, 1]], v: [[2, 3]], raw_scores: [[1]], scaled_scores: [[1 / Math.sqrt(2)]],
      weights: [[1]], weighted_values: [[2, 3]], layer: 0, head: 0, head_dim: 2, mask: [[true]], masked_scores: [[1 / Math.sqrt(2)]]},
    output: {vocabulary: ["<|unk|>", "a"], logits: [[0, 1]], probabilities: [probability], greedy_id: 1,
      top_candidates: [{id: 1, token: "a", probability: probability[1]}, {id: 0, token: "<|unk|>", probability: probability[0]}]},
    generation: {settings: {temperature: 1, top_k: null, top_p: null, greedy: true, seed: 42},
      steps: [{index: 1, context_token_ids: [1], token_id: 1, token: "a", logits: [0, 1], probabilities: probability, sampling_probabilities: [0, 1]}], text: "aa", token_ids: [1, 1]},
    metrics: {columns: ["step", "train_batch_loss", "validation_loss", "gradient_norm", "learning_rate"],
      rows: [{step: 1, train_batch_loss: 1, validation_loss: 1.1, gradient_norm: .5, learning_rate: .01}]},
    notes: ["Fixture values only; not the shipped real sample."],
  };
}

test("complete strict inspection fixture accepted and not mutated", () => {
  const data = fixture(), snapshot = JSON.stringify(data);
  assert.equal(validateInspection(data), data); assert.equal(JSON.stringify(data), snapshot);
});

test("malformed schema, dimensions, IDs and nonfinite data rejected", () => {
  const changes = [
    data => { data.schema = "other"; }, data => { data.extra = "not allowed"; },
    data => { data.shapes.qkv = [1, 1, 1, 3]; }, data => { data.embeddings.token[0].pop(); },
    data => { data.attention.q[0][0] = Infinity; }, data => { data.input.full_token_ids[0] = 99; },
    data => { data.output.vocabulary[1] = "aa"; }, data => { data.attention.mask[0][0] = false; },
    data => { data.attention.weights[0][0] = .5; }, data => { data.output.probabilities[0][0] = -.2; },
    data => { data.generation.steps[0].sampling_probabilities = [1, 0]; },
    data => { data.generation.text = "wrong"; }, data => { data.metrics.rows[0].step = -1; },
    data => { data.metrics.rows.push(clone(data.metrics.rows[0])); },
  ];
  changes.forEach(change => { const data = fixture(); change(data); assert.throws(() => validateInspection(data)); });
});

test("oversized nesting and prototype keys rejected; HTML strings remain inert data", () => {
  const data = fixture(); let cursor = data.source.training_metadata;
  for (let i = 0; i < 30; i++) { cursor.next = {}; cursor = cursor.next; }
  assert.throws(() => validateInspection(data));
  const polluted = fixture(); polluted.source.training_metadata = JSON.parse('{"__proto__":{"polluted":true}}');
  assert.throws(() => validateInspection(polluted));
  const html = fixture(); html.source.checkpoint_name = '<img src=x onerror="alert(1)">';
  assert.equal(validateInspection(html), html); // Renderer uses textContent, never parses this string as HTML.
});

const realPath = path.join(__dirname, "..", "..", "data", "inspection-sample.json");
if (fs.existsSync(realPath)) {
  test("real exported Python checkpoint satisfies the exact same schema", () => {
    const real = JSON.parse(fs.readFileSync(realPath, "utf8"));
    validateInspection(real);
    assert.equal(real.source.step, 100); assert.equal(real.input.context_tokens.length, 8);
    assert.equal(real.generation.steps.length, 4);
  });
}
console.log(`${count} browser math/schema tests passed`);
