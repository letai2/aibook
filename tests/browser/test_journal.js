"use strict";

// Pure contract tests only: no DOM, browser, storage, network, or user notebook.
// Run: node --test tests/browser/test_journal.js
const assert = require("node:assert/strict");
const test = require("node:test");
const { validateEntry, validateDraft, validateNotebook, pendingDraft, exportNotebook, mergeImport } =
  require("../../book_src/assets/journal.js");
const DATE_LIMIT = 8640000000000000;
const FORMAT = "mini-gpt-journal-v1";
const limits = { lesson: 100, title: 180, config: 4000, prediction: 6000,
  observation: 8000, interpretation: 6000, question: 4000 };

function entry(overrides = {}) {
  return { id: "test-entry", created: 1000, updated: 1001,
    lesson: "22-bpe", title: "آزمایش نشانه‌بندی",
    config: "python -m mini_gpt.milestones --help",
    prediction: "می‌روم؛ فکر می‌کنم طول کوتاه‌تر شود.",
    observation: "«ي» با «ی» و «ك» با «ک» یکسان نیست.",
    interpretation: "نتیجهٔ یک بذر، حکم کلی نیست.",
    question: "آزمایش بعدی چیست؟", ...overrides };
}
function notebook(entries = [entry()], overrides = {}) {
  return { format: FORMAT, entries, deleted: [], draft: null, ...overrides };
}
let nextID = 0;
const makeId = () => "generated-" + (++nextID);

test("Persian, Latin, emoji, combining marks and ZWNJ survive unchanged", () => {
  const unicode = "می‌روم | ي/ی ك/ک | hello | 🐈 | ا\u0650 | \u2028\u2029 | ↦";
  const source = entry({ title: unicode, observation: unicode, question: unicode });
  const result = validateEntry(source);
  assert.equal(result.title, unicode);
  assert.equal(result.observation, unicode);
  assert.equal(result.question, unicode);
  assert.notEqual(result, source);
  assert.deepEqual(validateNotebook(notebook([source])).entries[0], source);
});

test("HTML-looking text stays data and unknown/prototype keys are stripped", () => {
  const source = JSON.parse(JSON.stringify(entry()).slice(0, -1) +
    ',"__proto__":{"polluted":true},"constructor":"ignored","html":"ignored"}');
  source.title = '<img src=x onerror="alert(1)">';
  source.observation = '</script><script>alert("x")</script>';
  const result = validateEntry(source);
  assert.equal(result.title, source.title);
  assert.equal(result.observation, source.observation);
  assert.equal(Object.getPrototypeOf(result), Object.prototype);
  for (const key of ["__proto__", "constructor", "html"]) assert.equal(Object.hasOwn(result, key), false);
  assert.equal({}.polluted, undefined);
  // The DOM renderer must continue using textContent; validation does not parse HTML.
});

test("validation does not mutate caller entries, arrays or metadata", () => {
  const source = notebook([entry()], { unrelated: "ignored" });
  source.entries[0].unexpected = { retainedOnlyInOriginal: true };
  const before = JSON.stringify(source), result = validateNotebook(source);
  assert.equal(JSON.stringify(source), before);
  assert.notEqual(result.entries, source.entries);
  assert.equal(Object.hasOwn(result, "unrelated"), false);
  assert.equal(Object.hasOwn(result.entries[0], "unexpected"), false);
});

test("empty notebook and absent optional tombstones/draft are accepted", () => {
  assert.deepEqual(validateNotebook({ format: FORMAT, entries: [] }), notebook([]));
});

test("wrong formats, missing entries, nonarrays and duplicate IDs are rejected", () => {
  for (const value of [null, false, 0, "", [], {}, { format: "other", entries: [] },
    { format: FORMAT }, { format: FORMAT, entries: {} }, notebook([entry(), entry()])]) {
    assert.throws(() => validateNotebook(value));
  }
});

test("text fields enforce type and their exact UTF-16 length bounds", () => {
  for (const [field, maximum] of Object.entries(limits)) {
    assert.doesNotThrow(() => validateEntry(entry({ [field]: "a".repeat(maximum) })));
    assert.throws(() => validateEntry(entry({ [field]: "a".repeat(maximum + 1) })));
    for (const bad of [null, undefined, 5, false, [], {}]) {
      assert.throws(() => validateEntry(entry({ [field]: bad })));
    }
  }
  assert.doesNotThrow(() => validateEntry(entry({ title: "🐈".repeat(90) })));
  assert.throws(() => validateEntry(entry({ title: "🐈".repeat(91) })));
});

test("saved notes need a nonblank title while drafts may be untitled", () => {
  for (const title of ["", " ", "\t\n", "\u2003"]) {
    assert.throws(() => validateEntry(entry({ title })));
    assert.doesNotThrow(() => validateEntry(entry({ title }), true));
  }
  assert.equal(validateNotebook(notebook([], { draft: entry({ id: "draft", title: "" }) })).draft.title, "");
});

test("IDs are required, nonblank and bounded strings", () => {
  for (const id of ["", " ", null, undefined, 7, false, "a".repeat(101)]) {
    assert.throws(() => validateEntry(entry({ id })));
  }
  assert.doesNotThrow(() => validateEntry(entry({ id: "a".repeat(100) })));
});

test("timestamps must be safe integer milliseconds in order", () => {
  for (const bad of [-1, 0.5, NaN, Infinity, -Infinity, "1000", null, Number.MAX_SAFE_INTEGER + 1]) {
    assert.throws(() => validateEntry(entry({ created: bad })));
    assert.throws(() => validateEntry(entry({ updated: bad })));
  }
  assert.throws(() => validateEntry(entry({ created: 1002, updated: 1001 })));
  assert.doesNotThrow(() => validateEntry(entry({ created: 0, updated: 0 })));
});

test("dates outside the JS Date range are rejected before render/persistence", () => {
  for (const bad of [DATE_LIMIT + 1, Number.MAX_SAFE_INTEGER]) {
    assert.throws(() => validateEntry(entry({ updated: bad })));
    assert.throws(() => validateEntry(entry({ created: bad, updated: bad })));
  }
  const result = validateEntry(entry({ updated: DATE_LIMIT }));
  assert.doesNotThrow(() => new Date(result.updated).toISOString());
});

test("capacity is 500 entries plus a separately recoverable draft", () => {
  const entries = Array.from({ length: 500 }, (_, i) => entry({ id: "entry-" + i }));
  assert.equal(validateNotebook(notebook(entries)).entries.length, 500);
  assert.throws(() => validateNotebook(notebook([...entries, entry({ id: "entry-500" })])));
  assert.equal(validateNotebook(notebook(entries, { draft: entry({ id: "draft", title: "" }) })).entries.length, 500);
});

test("draft must be null, absent or a valid record, not a falsy wrong type", () => {
  for (const draft of [false, 0, "", [], {}, { title: "incomplete" }]) {
    assert.throws(() => validateNotebook(notebook([], { draft })));
  }
  assert.equal(validateNotebook(notebook([], { draft: null })).draft, null);
});

test("tombstones have valid IDs, Date-safe timestamps and bounded count", () => {
  for (const deleted of [null, "x", {}, [null], [{ id: "", time: 1 }],
    [{ id: 3, time: 1 }], [{ id: "x", time: -1 }], [{ id: "x", time: 1.5 }],
    [{ id: "x", time: Infinity }], [{ id: "x", time: DATE_LIMIT + 1 }]]) {
    assert.throws(() => validateNotebook(notebook([], { deleted })));
  }
  const deleted = [{ id: "برگه-۱", time: 1000 }];
  assert.deepEqual(validateNotebook(notebook([], { deleted })).deleted, deleted);
  assert.throws(() => validateNotebook(notebook([], {
    deleted: Array.from({ length: 5001 }, (_, i) => ({ id: "x-" + i, time: 1000 })),
  })));
});

test("malformed nested entries/drafts are rejected without partial validation", () => {
  const invalid = entry({ observation: { html: "<img>" } });
  assert.throws(() => validateNotebook(notebook([entry(), { ...invalid, id: "second" }])));
  assert.throws(() => validateNotebook(notebook([], { draft: invalid })));
});

test("draft preserves its original base revision across serialization/reload", () => {
  const oldDraft = entry({ updated: 2000, baseUpdated: 1001, observation: "old unsaved edit" });
  const current = entry({ updated: 3000, observation: "newer edit in another tab" });
  const restored = validateDraft(JSON.parse(JSON.stringify(oldDraft)));
  assert.equal(restored.baseUpdated, 1001);
  assert.notEqual(restored.baseUpdated, current.updated);
  assert.notEqual(pendingDraft(restored, [current]), null);
  assert.equal(validateDraft(entry()).baseUpdated, null);
  assert.throws(() => validateDraft(entry({ baseUpdated: DATE_LIMIT + 1 })));
});

test("unchanged saved form is not a pending draft or export duplicate", () => {
  const saved = entry(), formRecord = { ...saved, created: 2000, updated: 2001, baseUpdated: saved.updated };
  assert.equal(pendingDraft(formRecord, [saved]), null);
  assert.equal(exportNotebook(notebook([saved]), formRecord).draft, null);
  const changed = { ...formRecord, observation: "genuine new observation" };
  assert.equal(pendingDraft(changed, [saved]).observation, changed.observation);
});

test("full 500-entry export restores and repeated imports do not duplicate saved draft", () => {
  const entries = Array.from({ length: 500 }, (_, i) => entry({ id: "entry-" + i, title: "note " + i }));
  const source = notebook(entries);
  const exported = exportNotebook(source, { ...entries[499], baseUpdated: entries[499].updated });
  assert.equal(exported.draft, null);
  const first = mergeImport(notebook([]), exported, makeId);
  assert.equal(first.notebook.entries.length, 500);
  assert.equal(first.recoveryDraft, null);
  const repeated = mergeImport(first.notebook, exported, makeId);
  assert.equal(repeated.added, 0);
  assert.equal(repeated.notebook.entries.length, 500);
  // A backup produced by the old implementation may include that same saved form.
  const legacy = { ...exported, draft: { ...entries[499], created: 2000, updated: 2001 } };
  assert.equal(mergeImport(first.notebook, legacy, makeId).recoveryDraft, null);
});

test("500 entries and a genuine draft restore without promoting it to entry501", () => {
  const entries = Array.from({ length: 500 }, (_, i) => entry({ id: "entry-" + i }));
  const draft = entry({ id: "draft", title: "", observation: "unsaved distinct result", baseUpdated: null });
  const result = mergeImport(notebook([]), notebook(entries, { draft }), makeId);
  assert.equal(result.notebook.entries.length, 500);
  assert.equal(result.recoveryDraft.observation, draft.observation);
});

test("conflicting import preserves both versions; overcapacity does not mutate input", () => {
  const existing = notebook([entry()]);
  const incoming = notebook([entry({ observation: "different imported observation" })]);
  const result = mergeImport(existing, incoming, makeId);
  assert.equal(result.copies, 1);
  assert.equal(result.notebook.entries.length, 2);
  assert.equal(result.notebook.entries[0].observation, existing.entries[0].observation);
  assert.equal(result.notebook.entries[1].observation, incoming.entries[0].observation);
  assert.notEqual(result.notebook.entries[0].id, result.notebook.entries[1].id);
  const full = notebook(Array.from({ length: 500 }, (_, i) => entry({ id: "entry-" + i })));
  const before = JSON.stringify(full);
  assert.throws(() => mergeImport(full, notebook([entry({ id: "extra" })]), makeId));
  assert.equal(JSON.stringify(full), before);
});

test("imported deletion records never erase notes from the current notebook", () => {
  const existing = notebook([entry()]);
  const incoming = notebook([], { deleted: [{ id: "test-entry", time: 2000 }] });
  const result = mergeImport(existing, incoming, makeId);
  assert.equal(result.notebook.entries.length, 1);
  assert.equal(result.notebook.entries[0].id, "test-entry");
});
