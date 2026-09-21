"use strict";
(() => {
  const FORMAT = "mini-gpt-journal-v1";
  const DATE_LIMIT = 8640000000000000;
  const fields = { lesson: 100, title: 180, config: 4000, prediction: 6000, observation: 8000, interpretation: 6000, question: 4000 };
  const labels = { config: "پیکربندی و فرمان", prediction: "پیش‌بینی", observation: "مشاهده و شاهد", interpretation: "تفسیر و کشف", question: "پرسش و گام بعد" };
  const string = (value, max) => typeof value === "string" && value.length <= max;
  const timestamp = value => Number.isSafeInteger(value) && value >= 0 && value <= DATE_LIMIT;
  function validateEntry(entry, allowUntitled = false) {
    if (!entry || typeof entry !== "object" || Array.isArray(entry) ||
        !string(entry.id, 100) || !entry.id.trim() || !timestamp(entry.created) ||
        !timestamp(entry.updated) || entry.updated < entry.created) throw Error("invalid entry");
    for (const [field, max] of Object.entries(fields)) if (!string(entry[field], max)) throw Error("invalid field");
    if (!allowUntitled && !entry.title.trim()) throw Error("missing title");
    return Object.fromEntries(["id", "created", "updated", ...Object.keys(fields)].map(key => [key, entry[key]]));
  }
  function validateDraft(value) {
    const entry = validateEntry(value, true);
    const base = value.baseUpdated === undefined ? null : value.baseUpdated;
    if (base !== null && !timestamp(base)) throw Error("invalid draft base");
    return { ...entry, baseUpdated: base };
  }
  function validateNotebook(data) {
    if (!data || data.format !== FORMAT || !Array.isArray(data.entries) || data.entries.length > 500) throw Error("invalid notebook");
    const entries = data.entries.map(value => validateEntry(value));
    if (new Set(entries.map(value => value.id)).size !== entries.length) throw Error("duplicate IDs");
    const deleted = data.deleted === undefined ? [] : data.deleted;
    if (!Array.isArray(deleted) || deleted.length > 5000 || deleted.some(value =>
      !value || !string(value.id, 100) || !value.id.trim() || !timestamp(value.time))) throw Error("invalid deleted");
    const draft = data.draft === undefined || data.draft === null ? null : validateDraft(data.draft);
    return { format: FORMAT, entries, deleted: deleted.map(value => ({ id: value.id, time: value.time })), draft };
  }
  function hasText(entry) {
    return ["title", ...Object.keys(labels)].some(name => entry[name]?.trim());
  }
  function sameFields(a, b) {
    return Object.keys(fields).every(name => a[name] === b[name]);
  }
  function pendingDraft(record, entries) {
    const draft = validateDraft(record);
    if (!hasText(draft)) return null;
    const saved = entries.find(entry => entry.id === draft.id);
    return saved && sameFields(saved, draft) ? null : draft;
  }
  function exportNotebook(notebook, record) {
    const result = validateNotebook(notebook);
    result.draft = pendingDraft(record, result.entries);
    return result;
  }
  function mergeImport(current, incoming, makeId) {
    const result = validateNotebook(current);
    const imported = validateNotebook(incoming);
    let added = 0, copies = 0;
    for (const entry of imported.entries) {
      const same = result.entries.find(value => value.id === entry.id);
      if (same && sameFields(same, entry)) continue;
      const record = { ...entry };
      if (same) {
        record.id = makeId();
        record.title = record.title.slice(0, 150) + " · نسخهٔ واردشده";
        copies++;
      }
      result.entries.push(validateEntry(record));
      added++;
    }
    // A genuine unsaved draft is recovered separately, not forced into a full notebook.
    const recoveryDraft = imported.draft && hasText(imported.draft) &&
      !result.entries.some(entry => sameFields(entry, imported.draft)) ? imported.draft : null;
    result.draft = null;
    // Imported tombstones never delete a current note.
    return { notebook: validateNotebook(result), recoveryDraft, added, copies };
  }
  if (typeof module !== "undefined" && module.exports) {
    module.exports = { validateEntry, validateDraft, validateNotebook, sameFields, pendingDraft, exportNotebook, mergeImport };
  }
  if (typeof document === "undefined") return;
  const surface = document.querySelector("[data-journal]");
  if (!surface) return;
  const form = surface.querySelector("form");
  const key = FORMAT, legacyDraftKey = "mini-gpt-journal-draft-v1";
  const draftPrefix = "mini-gpt-journal-draft-v2:", sessionKey = "mini-gpt-journal-tab-v2";
  const empty = () => ({ format: FORMAT, entries: [], deleted: [], draft: null });
  const uuid = () => globalThis.crypto?.randomUUID?.() || String(Date.now()) + "-" + Math.random().toString(16).slice(2);
  const state = surface.querySelector("[data-journal-status]");
  const draftStatus = surface.querySelector("[data-draft-status]");
  let notebook = empty(), editing = null, loadedUpdate = null, storageOK = true, blockedRead = false;
  let draftCreated = Date.now(), previousDraftKey = null;
  // A fresh slot on every page lifetime also protects duplicated tabs whose sessionStorage was cloned.
  // The prior slot remains recoverable; it is never silently deleted by a new tab/page.
  const draftKey = draftPrefix + uuid();
  const memoryDrafts = new Map();
  try { previousDraftKey = sessionStorage.getItem(sessionKey); sessionStorage.setItem(sessionKey, draftKey); } catch (_) { /* recovery list still works without sessionStorage */ }
  function read() {
    const raw = localStorage.getItem(key);
    return raw ? validateNotebook(JSON.parse(raw)) : empty();
  }
  try { notebook = read(); } catch (_) {
    storageOK = false; blockedRead = true;
    state.textContent = "خواندن دفتر ذخیره‌شده ممکن نشد. دادهٔ قبلی بازنویسی نمی‌شود؛ نوشتهٔ تازه را با خروجی نگه دارید.";
  }
  function refreshStored() {
    if (storageOK) {
      try { notebook = read(); } catch (_) { storageOK = false; blockedRead = true; }
    }
  }
  function persist() {
    // After a failed write, keep this page memory-only. Retrying a stale snapshot later
    // could overwrite another tab's successful changes; recovery is explicit via export/import.
    if (blockedRead || !storageOK) return false;
    try {
      const validated = validateNotebook(notebook);
      localStorage.setItem(key, JSON.stringify(validated));
      notebook = validated; storageOK = true; return true;
    } catch (_) { storageOK = false; return false; }
  }
  function mutate(action) {
    // Serialize read/modify/write across tabs where Web Locks is available.
    if (navigator.locks?.request) return navigator.locks.request(key, action);
    return Promise.resolve().then(action);
  }
  const select = surface.querySelector("[data-journal-lesson]");
  function element(tag, text, className) {
    const node = document.createElement(tag);
    if (text !== undefined) node.textContent = text;
    if (className) node.className = className;
    return node;
  }
  const option = element("option", "آزمایش آزاد"); option.value = ""; select.append(option);
  for (const item of window.BOOK.units) {
    const op = element("option", item.title); op.value = item.id; select.append(op);
  }
  const requested = new URL(location.href).searchParams.get("lesson");
  if (window.BOOK.units.some(item => item.id === requested)) select.value = requested;
  function formValues() {
    return Object.fromEntries(Object.keys(fields).map(name => [name, form.elements.namedItem(name).value]));
  }
  function setValues(entry) {
    if (entry?.lesson && ![...select.options].some(option => option.value === entry.lesson)) {
      const unknown = element("option", "درس نسخهٔ دیگر: " + entry.lesson);
      unknown.value = entry.lesson; select.append(unknown);
    }
    for (const name of Object.keys(fields)) form.elements.namedItem(name).value = entry?.[name] || "";
  }
  function draftRecord() {
    return { id: editing || "draft", created: draftCreated, updated: Math.max(Date.now(), draftCreated),
      baseUpdated: loadedUpdate, ...formValues() };
  }
  function currentPending() {
    return pendingDraft(draftRecord(), storageOK ? notebook.entries : []);
  }
  function writeDraft(slot, record) {
    memoryDrafts.set(slot, record);
    try {
      if (blockedRead) throw Error("blocked");
      localStorage.setItem(slot, JSON.stringify(record));
      memoryDrafts.delete(slot); return true;
    } catch (_) { return false; }
  }
  function removeOwnDraft() {
    memoryDrafts.delete(draftKey);
    try { localStorage.removeItem(draftKey); } catch (_) { /* no other slot is touched */ }
  }
  const recovery = element("section", undefined, "draft-recovery");
  recovery.dataset.draftRecovery = "";
  const recoveryHeading = element("h2", "پیش‌نویس‌های قابل بازیابی");
  recoveryHeading.tabIndex = -1;
  recovery.append(recoveryHeading);
  const recoveryList = element("div"); recoveryList.dataset.draftRecoveryList = ""; recovery.append(recoveryList);
  form.after(recovery);
  function allDrafts() {
    const drafts = new Map(memoryDrafts);
    let unreadable = 0;
    try {
      for (let index = 0; index < localStorage.length; index++) {
        const slot = localStorage.key(index);
        if (slot !== legacyDraftKey && !slot?.startsWith(draftPrefix)) continue;
        if (drafts.has(slot)) continue;
        try { drafts.set(slot, validateDraft(JSON.parse(localStorage.getItem(slot)))); }
        catch (_) { unreadable++; }
      }
    } catch (_) { /* keep in-memory recoveries usable */ }
    return { drafts, unreadable };
  }
  function restoreDraft(record) {
    const draft = validateDraft(record);
    setValues(draft); draftCreated = draft.created;
    editing = draft.id === "draft" ? null : draft.id;
    loadedUpdate = draft.baseUpdated; // Never infer a new base from the current saved entry.
  }
  function renderRecoveries() {
    recoveryList.replaceChildren();
    const { drafts, unreadable } = allDrafts();
    const signatures = new Set();
    const current = draftRecord();
    let count = 0;
    for (const [slot, draft] of [...drafts].sort((a, b) => b[1].updated - a[1].updated)) {
      if (slot === draftKey || !hasText(draft) || sameFields(draft, current) ||
          notebook.entries.some(entry => sameFields(entry, draft))) continue;
      const signature = JSON.stringify(Object.keys(fields).map(name => draft[name]));
      if (signatures.has(signature)) continue;
      signatures.add(signature); count++;
      const article = element("article", undefined, "draft-recovery-item");
      const title = draft.title.trim() || "پیش‌نویس بدون عنوان";
      article.append(element("h3", title), element("p", new Date(draft.updated).toLocaleString("fa-IR")));
      const recover = element("button", "بازیابی این پیش‌نویس"); recover.type = "button";
      recover.setAttribute("aria-label", "بازیابی پیش‌نویس: " + title);
      recover.addEventListener("click", () => {
        if (currentPending() && !window.confirm("فرم فعلی با این پیش‌نویس جایگزین شود؟ پیش‌نویس فعلی ابتدا برای بازیابی نگه داشته می‌شود.")) return;
        const pending = currentPending();
        if (pending) writeDraft(draftPrefix + uuid(), pending);
        restoreDraft(draft); saveDraft(); form.elements.title.focus();
        draftStatus.textContent = "پیش‌نویس بازیابی شد؛ نسخهٔ منبع حذف نشده است.";
      });
      const remove = element("button", "حذف این پیش‌نویس", "danger"); remove.type = "button";
      remove.setAttribute("aria-label", "حذف پیش‌نویس: " + title);
      remove.addEventListener("click", () => {
        if (!window.confirm("فقط این نسخهٔ پیش‌نویس حذف شود؟ برگه‌های ثبت‌شده و دیگر زبانه‌ها دست‌نخورده‌اند.")) return;
        memoryDrafts.delete(slot);
        try { localStorage.removeItem(slot); } catch (_) { draftStatus.textContent = "حذف نسخهٔ ذخیره‌شده ممکن نشد."; }
        renderRecoveries(); recoveryHeading.focus();
      });
      article.append(recover, remove); recoveryList.append(article);
    }
    if (!count) recoveryList.append(element("p", "پیش‌نویس دیگری برای بازیابی نیست."));
    if (unreadable) recoveryList.append(element("p", "بعضی پیش‌نویس‌های قبلی معتبر نبودند؛ بدون بازنویسی در مرورگر باقی مانده‌اند."));
  }
  function saveDraft() {
    const record = currentPending();
    if (!record) {
      removeOwnDraft();
      draftStatus.textContent = hasText(formValues()) ? "تغییر ثبت‌نشده‌ای در این برگه نیست." : "برگهٔ تازه آماده است.";
    } else {
      const ok = writeDraft(draftKey, record);
      draftStatus.textContent = ok ? "پیش‌نویس این زبانه ذخیره شد؛ برای ورود به فهرست، ثبت یادداشت را بزنید."
        : "پیش‌نویس فقط در این صفحه است؛ پیش از بستن، خروجی بگیرید.";
    }
    renderRecoveries();
  }
  try {
    const raw = previousDraftKey ? localStorage.getItem(previousDraftKey) : null;
    if (raw) {
      const draft = validateDraft(JSON.parse(raw));
      if (pendingDraft(draft, notebook.entries)) {
        restoreDraft(draft); saveDraft(); draftStatus.textContent = "پیش‌نویس قبلی این زبانه بازیابی شد.";
      }
    }
  } catch (_) { draftStatus.textContent = "پیش‌نویس قبلی قابل خواندن نبود؛ دفتر ثبت‌شده مستقل است."; }
  form.addEventListener("input", saveDraft); form.addEventListener("change", saveDraft);
  function startNew(ask = true) {
    if (ask && currentPending() && !window.confirm("پیش‌نویس فعلی کنار گذاشته شود؟ یادداشت‌های ثبت‌شده حذف نمی‌شوند.")) return;
    form.reset(); editing = null; loadedUpdate = null; draftCreated = Date.now();
    if (window.BOOK.units.some(item => item.id === requested)) select.value = requested;
    removeOwnDraft(); draftStatus.textContent = "برگهٔ تازه آماده است."; renderRecoveries();
  }
  surface.querySelector("[data-journal-new]").addEventListener("click", () => startNew());
  function render() {
    const target = surface.querySelector("[data-journal-entries]"); target.replaceChildren();
    const query = surface.querySelector("[data-journal-search]").value.trim().toLocaleLowerCase();
    const entries = [...notebook.entries].sort((a, b) => b.updated - a.updated)
      .filter(entry => Object.values(entry).join(" ").toLocaleLowerCase().includes(query));
    if (!entries.length) target.append(element("p", query ? "یادداشتی با این عبارت پیدا نشد." :
      "هنوز برگه‌ای ثبت نشده. یک پیش‌بینی کوچک برای شروع کافی است.", "empty-journal"));
    for (const entry of entries) {
      const section = element("article", undefined, "journal-entry"); section.append(element("h3", entry.title));
      const time = element("time", new Date(entry.updated).toLocaleString("fa-IR"));
      time.dateTime = new Date(entry.updated).toISOString(); section.append(time);
      const lesson = window.BOOK.units.find(item => item.id === entry.lesson);
      if (lesson) {
        const paragraph = element("p"), a = element("a", lesson.title);
        const targetURL = new URL(lesson.path, window.BookUI.root);
        const currentURL = new URL(location.href);
        for (const name of ["progress", "done"]) {
          const value = currentURL.searchParams.get(name);
          if (value !== null) targetURL.searchParams.set(name, value);
        }
        a.href = targetURL; a.dataset.bookLink = ""; paragraph.append(a); section.append(paragraph);
      }
      const dl = element("dl");
      for (const [field, label] of Object.entries(labels)) {
        if (entry[field]) dl.append(element("dt", label), element("dd", entry[field]));
      }
      section.append(dl);
      const controls = element("div", undefined, "control-row");
      const edit = element("button", "بازکردن برای ویرایش"); edit.type = "button";
      edit.setAttribute("aria-label", "ویرایش یادداشت: " + entry.title);
      edit.addEventListener("click", () => {
        if (currentPending() && !window.confirm("پیش‌نویس فعلی با این یادداشت جایگزین شود؟ ابتدا یک نسخه برای بازیابی نگه داشته می‌شود.")) return;
        const pending = currentPending(); if (pending) writeDraft(draftPrefix + uuid(), pending);
        setValues(entry); editing = entry.id; loadedUpdate = entry.updated; draftCreated = entry.created;
        saveDraft(); form.elements.title.focus();
      });
      const remove = element("button", "حذف این برگه", "danger"); remove.type = "button";
      remove.setAttribute("aria-label", "حذف یادداشت: " + entry.title);
      remove.addEventListener("click", async () => {
        if (!window.confirm("«" + entry.title + "» از دفتر حذف شود؟ بازیابی فقط از خروجی قبلی ممکن است.")) return;
        try {
          await mutate(() => {
            refreshStored(); notebook.entries = notebook.entries.filter(value => value.id !== entry.id);
            notebook.deleted = [...notebook.deleted, { id: entry.id, time: Date.now() }].slice(-5000);
            if (editing === entry.id) startNew(false);
            const ok = persist(); render(); form.elements.title.focus();
            state.textContent = ok ? "برگه حذف شد؛ خروجی قبلی در صورت وجود قابل بازیابی است." :
              "حذف فقط در این صفحه اعمال شد؛ ذخیرهٔ مرورگر در دسترس نیست.";
          });
        } catch (_) { state.textContent = "حذف انجام نشد؛ دفتر را دوباره بررسی کنید."; }
      });
      controls.append(edit, remove); section.append(controls); target.append(section);
    }
    renderRecoveries();
  }
  form.addEventListener("submit", async event => {
    event.preventDefault();
    try {
      await mutate(() => {
        refreshStored();
        const time = Date.now(), old = notebook.entries.find(entry => entry.id === editing);
        const conflict = editing && (!old || old.updated !== loadedUpdate);
        const record = { id: old && !conflict ? old.id : uuid(), created: old && !conflict ? old.created : time,
          updated: Math.max(time, (old?.updated || 0) + 1), ...formValues() };
        if (conflict) record.title = record.title.slice(0, 150) + " · نسخهٔ هم‌زمان";
        const candidate = notebook.entries.filter(entry => entry.id !== record.id); candidate.push(record);
        try { validateEntry(record); if (candidate.length > 500) throw Error("full"); }
        catch (_) { state.textContent = "عنوان و زمان معتبر لازم است و دفتر حداکثر ۵۰۰ برگه می‌پذیرد. چیزی حذف نشد."; return; }
        notebook.entries = candidate; const ok = persist();
        editing = record.id; loadedUpdate = record.updated; draftCreated = record.created;
        // A conflict creates a named copy; reflect that canonical title in the form before draft comparison.
        setValues(record); render(); saveDraft();
        state.textContent = (conflict ? "تغییر هم‌زمان دیده شد؛ هر دو نسخه نگه داشته شدند. " : "") +
          (ok ? "یادداشت ثبت شد. برای پشتیبان، خروجی دفتر بگیرید." : "یادداشت فقط در حافظهٔ این صفحه است؛ همین حالا خروجی بگیرید.");
      });
    } catch (_) { state.textContent = "ثبت انجام نشد؛ متن فرم دست‌نخورده است. خروجی بگیرید."; }
  });
  surface.querySelector("[data-journal-search]").addEventListener("input", render);
  surface.querySelector("[data-journal-export]").addEventListener("click", () => {
    refreshStored();
    const data = { ...exportNotebook(notebook, draftRecord()), exported_at: new Date().toISOString() };
    const text = window.BookUI.downloadJSON(data, "mini-gpt-journal.json");
    const preview = surface.querySelector("[data-journal-preview]"); preview.hidden = false; preview.open = true;
    surface.querySelector("[data-journal-json]").value = text;
    state.textContent = "خروجی دفتر و تغییر ثبت‌نشدهٔ همین فرم آماده شد. برای پیش‌نویس زبانهٔ دیگر، ابتدا آن را بازیابی و سپس خروجی بگیرید.";
  });
  surface.querySelector("[data-journal-import]").addEventListener("change", async event => {
    const file = event.target.files[0]; if (!file) return;
    try {
      if (file.size > 50_000_000) throw Error("large");
      const data = validateNotebook(JSON.parse(await file.text()));
      await mutate(() => {
        refreshStored(); const merged = mergeImport(notebook, data, uuid);
        notebook = merged.notebook; const ok = persist();
        if (merged.recoveryDraft) writeDraft(draftPrefix + uuid(), merged.recoveryDraft);
        render();
        state.textContent = merged.added + " برگه افزوده شد؛ " + merged.copies + " تعارض با نگهداری هر دو نسخه حل شد. " +
          (merged.recoveryDraft ? "پیش‌نویس واردشده در بخش بازیابی نگه داشته شد. " : "") +
          (ok ? "دفتر قبلی حفظ شد." : "ذخیرهٔ مرورگر ممکن نشد؛ خروجی بگیرید.");
      });
    } catch (_) {
      state.textContent = "فایل دفتر معتبر نیست، بیش از ۵۰ مگابایت است یا ظرفیت ۵۰۰ برگه را رد می‌کند؛ دفتر فعلی حفظ شد.";
    } finally { event.target.value = ""; }
  });
  window.addEventListener("storage", event => {
    if (event.key === key) {
      refreshStored(); render();
      state.textContent = "دفتر در زبانهٔ دیگر تغییر کرد. متن فرم دست‌نخورده است؛ ویرایش ناسازگار به‌صورت نسخهٔ جدا ثبت می‌شود.";
    }
    if (event.key === legacyDraftKey || event.key?.startsWith(draftPrefix)) renderRecoveries();
  });
  window.addEventListener("beforeunload", event => {
    if (memoryDrafts.size || !storageOK && notebook.entries.length) { event.preventDefault(); event.returnValue = ""; }
  });
  if (!navigator.locks?.request) {
    const notice = element("p", "این مرورگر قفل مشترک زبانه‌ها را ندارد؛ ثبت هم‌زمان دفتر را از دو زبانه انجام ندهید. پیش‌نویس‌ها جدا نگه داشته می‌شوند.", "privacy-note");
    form.before(notice);
  }
  render();
})();
