"use strict";
(function () {
  const TOKEN_REFERENCE = "مدل می‌رود. cat cats سلام!";
  const Q = [[1, 0], [0, 1], [1, 1]];
  const K = [[1, 1], [2, 0], [0, 1]];
  const V = [[1, 0], [0, 2], [3, 1]];

  function softmax(logits, temperature = 1) {
    if (!Array.isArray(logits) || !logits.length || !Number.isFinite(temperature) || temperature <= 0 ||
        logits.some(value => typeof value !== "number" || Number.isNaN(value) || value === Infinity)) {
      throw new Error("امتیازها و دما معتبر نیستند.");
    }
    const maximum = logits.reduce((maximum, value) => Math.max(maximum, value), -Infinity);
    if (!Number.isFinite(maximum)) throw new Error("حداقل یک امتیاز مجاز لازم است.");
    const exponents = logits.map(value => Math.exp((value - maximum) / temperature));
    const sum = exponents.reduce((a, b) => a + b, 0);
    return exponents.map(value => value / sum);
  }

  function tokenization(text) {
    const vocabulary = ["<|unk|>", ...new Set([...TOKEN_REFERENCE])];
    vocabulary.splice(1, vocabulary.length - 1, ...vocabulary.slice(1).sort((a, b) => a.codePointAt(0) - b.codePointAt(0)));
    const mapping = new Map(vocabulary.map((token, id) => [token, id]));
    const tokens = [...text];
    return { vocabulary, tokens, ids: tokens.map(token => mapping.get(token) ?? 0) };
  }

  function attention(causal = true) {
    const raw = Q.map(query => K.map(key => query.reduce((sum, value, index) => sum + value * key[index], 0)));
    const scaled = raw.map(row => row.map(value => value / Math.sqrt(2)));
    const mask = Q.map((_, row) => K.map((__, column) => !causal || column <= row));
    const masked = scaled.map((row, i) => row.map((value, j) => mask[i][j] ? value : -Infinity));
    const weights = masked.map(row => softmax(row));
    const output = weights.map(row => V[0].map((_, d) => row.reduce((sum, weight, index) => sum + weight * V[index][d], 0)));
    return { raw, scaled, mask, masked, weights, output };
  }

  function tensorShapes(B, T, C, H) {
    const limits = [16, 1024, 2048, 32];
    if ([B, T, C, H].some((value, index) => !Number.isInteger(value) || value < 1 || value > limits[index])) {
      throw new Error("B از ۱ تا ۱۶، T تا ۱۰۲۴، C تا ۲۰۴۸ و H تا ۳۲؛ همه عدد صحیح مثبت‌اند.");
    }
    if (C % H !== 0) throw new Error("C باید بر H بخش‌پذیر باشد؛ هر سر سهم یکسانی از ویژگی‌ها دارد.");
    return { B, T, C, H, D: C / H, ids: [B, T], embedding: [B, T, C],
      beforeTranspose: [B, T, H, C / H], heads: [B, H, T, C / H], scores: [B, H, T, T],
      scoreElements: B * H * T * T, scoreBytes: B * H * T * T * 4 };
  }

  // The importer treats every file field as data. Limits are also enforced without recursive calls.
  function boundedTree(value) {
    const stack = [[value, 0]];
    let visited = 0;
    while (stack.length) {
      const [item, depth] = stack.pop();
      if (++visited > 1500000 || depth > 24) throw new Error("ساختار فایل بیش از حد بزرگ یا تو‌در‌تو است.");
      if (typeof item === "number" && !Number.isFinite(item)) throw new Error("عدد نامتناهی یا نامعتبر در فایل است.");
      if (typeof item === "string" && item.length > 1000000) throw new Error("رشتهٔ متنی بیش از حد بلند است.");
      if (item !== null && typeof item === "object") {
        const prototype = Object.getPrototypeOf(item);
        if (!Array.isArray(item) && prototype !== Object.prototype && prototype !== null) throw new Error("شیء نامعتبر.");
        const keys = Object.keys(item);
        if (keys.length + stack.length + visited > 1500000) throw new Error("تعداد خانه‌های فایل بیش از حد مجاز است.");
        for (const key of keys) {
          if (["__proto__", "prototype", "constructor"].includes(key)) throw new Error("کلید نامعتبر در فایل.");
          stack.push([item[key], depth + 1]);
        }
      } else if (!["string", "number", "boolean"].includes(typeof item) && item !== null) {
        throw new Error("فقط دادهٔ JSON مجاز است.");
      }
    }
  }

  function validateInspection(data) {
    boundedTree(data);
    const fail = label => { throw new Error(`فایل با قرارداد بازرسی سازگار نیست: ${label}`); };
    const object = (value, keys, label) => {
      if (value === null || typeof value !== "object" || Array.isArray(value) ||
          Object.keys(value).length !== keys.length || keys.some(key => !Object.hasOwn(value, key))) fail(label);
    };
    const integer = (value, min, max, label) => { if (!Number.isInteger(value) || value < min || value > max) fail(label); };
    const string = (value, maximum, label) => { if (typeof value !== "string" || value.length > maximum) fail(label); };
    const finite = (value, label) => { if (typeof value !== "number" || !Number.isFinite(value)) fail(label); };
    const shape = (value, expected, label) => { if (!Array.isArray(value) || value.length !== expected.length || value.some((n, i) => n !== expected[i])) fail(label); };
    const vector = (value, length, label, check = finite) => {
      if (!Array.isArray(value) || value.length !== length) fail(label);
      value.forEach((number, index) => check(number, `${label}[${index}]`));
    };
    const matrix = (value, rows, columns, label, check = finite) => {
      if (!Array.isArray(value) || value.length !== rows) fail(label);
      value.forEach((row, index) => vector(row, columns, `${label}[${index}]`, check));
    };
    const probability = (value, label) => { finite(value, label); if (value < 0 || value > 1.000001) fail(label); };
    const distribution = (row, length, label) => {
      vector(row, length, label, probability);
      if (Math.abs(row.reduce((sum, value) => sum + value, 0) - 1) > 0.0001) fail(`${label}: مجموع احتمال`);
    };
    object(data, ["schema", "source", "input", "shapes", "embeddings", "layers", "attention", "output", "generation", "metrics", "notes"], "ریشه");
    if (data.schema !== "mini-gpt-inspection-v1") fail("نسخهٔ قرارداد");
    object(data.source, ["kind", "checkpoint_name", "checkpoint_sha256", "step", "format_version", "torch_version", "config", "training_metadata"], "منبع");
    if (data.source.kind !== "checkpoint" || data.source.format_version !== 2) fail("نوع منبع");
    string(data.source.checkpoint_name, 255, "نام فایل");
    if (typeof data.source.checkpoint_sha256 !== "string" || !/^[a-f0-9]{64}$/.test(data.source.checkpoint_sha256)) fail("اثر انگشت");
    integer(data.source.step, 0, Number.MAX_SAFE_INTEGER, "گام آموزش");
    string(data.source.torch_version, 100, "نسخهٔ PyTorch");
    if (data.source.training_metadata === null || typeof data.source.training_metadata !== "object" || Array.isArray(data.source.training_metadata)) fail("اطلاعات آموزش");
    const config = data.source.config;
    object(config, ["vocab_size", "context_length", "embedding_dim", "num_heads", "num_layers", "dropout"], "تنظیمات");
    const N = config.vocab_size, C = config.embedding_dim, H = config.num_heads, L = config.num_layers;
    integer(N, 2, 100000, "اندازهٔ واژگان"); integer(C, 1, 8192, "ویژگی‌ها");
    integer(H, 1, 256, "سرها"); integer(L, 1, 256, "لایه‌ها");
    integer(config.context_length, 1, 1000000, "زمینهٔ مدل");
    if (C % H !== 0 || typeof config.dropout !== "number" || config.dropout < 0 || config.dropout >= 1) fail("تنظیمات مدل");
    const D = C / H;
    object(data.output, ["vocabulary", "logits", "probabilities", "greedy_id", "top_candidates"], "خروجی");
    const vocabulary = data.output.vocabulary;
    vector(vocabulary, N, "واژگان", (token, label) => string(token, 16, label));
    if (vocabulary[0] !== "<|unk|>" || vocabulary.slice(1).some(token => [...token].length !== 1) || new Set(vocabulary).size !== N) fail("نشانه‌های واژگان");
    const id = (number, label) => integer(number, 0, N - 1, label);
    const input = data.input;
    object(input, ["prompt", "full_token_ids", "context_token_ids", "context_tokens", "unknown_positions", "truncated", "context_limit"], "ورودی");
    string(input.prompt, 1000000, "متن آغازین");
    const chars = [...input.prompt];
    if (!chars.length) fail("متن خالی");
    vector(input.full_token_ids, chars.length, "شناسه‌های متن", id);
    const mapping = new Map(vocabulary.map((token, index) => [token, index]));
    if (input.full_token_ids.some((number, index) => number !== (mapping.get(chars[index]) ?? 0))) fail("نگاشت متن و شناسه");
    integer(input.context_limit, 1, Math.min(64, config.context_length), "زمینهٔ بازرسی");
    const T = Math.min(input.context_limit, chars.length);
    shape(input.context_token_ids, input.full_token_ids.slice(-T), "شناسه‌های زمینه");
    shape(input.context_tokens, input.context_token_ids.map(number => vocabulary[number]), "نشانه‌های زمینه");
    shape(input.unknown_positions, input.full_token_ids.flatMap((number, index) => number === 0 ? [index] : []), "جایگاه ناشناخته");
    if (input.truncated !== (chars.length > input.context_limit)) fail("پرچم بریدن متن");
    const s = data.shapes;
    object(s, ["token_ids", "embedding", "qkv", "scores", "logits"], "شکل‌ها");
    shape(s.token_ids, [1, T], "شکل شناسه"); shape(s.embedding, [1, T, C], "شکل نمایش");
    shape(s.qkv, [1, H, T, D], "شکل QKV"); shape(s.scores, [1, H, T, T], "شکل امتیاز توجه"); shape(s.logits, [1, T, N], "شکل خروجی");
    object(data.embeddings, ["token", "position", "combined"], "نمایش‌ها");
    for (const key of ["token", "position", "combined"]) matrix(data.embeddings[key], T, C, `نمایش ${key}`);
    if (!Array.isArray(data.layers) || data.layers.length !== L) fail("بلوک‌ها");
    data.layers.forEach((layer, index) => {
      object(layer, ["index", "input_shape", "output_shape", "attention_shape", "feed_forward_shape"], "خلاصهٔ بلوک");
      if (layer.index !== index) fail("شمارهٔ بلوک");
      for (const key of ["input_shape", "output_shape", "feed_forward_shape"]) shape(layer[key], [1, T, C], key);
      shape(layer.attention_shape, [1, H, T, T], "شکل توجه بلوک");
    });
    const a = data.attention;
    object(a, ["q", "k", "v", "raw_scores", "scaled_scores", "weights", "weighted_values", "layer", "head", "head_dim", "mask", "masked_scores"], "توجه");
    integer(a.layer, 0, L - 1, "لایهٔ منتخب"); integer(a.head, 0, H - 1, "سر منتخب");
    if (a.head_dim !== D) fail("ویژگی هر سر");
    for (const key of ["q", "k", "v", "weighted_values"]) matrix(a[key], T, D, key);
    for (const key of ["raw_scores", "scaled_scores"]) matrix(a[key], T, T, key);
    matrix(a.mask, T, T, "پوشش", (value, label) => { if (typeof value !== "boolean") fail(label); });
    matrix(a.masked_scores, T, T, "امتیاز پوشیده", (value, label) => { if (value !== null) finite(value, label); });
    if (!Array.isArray(a.weights) || a.weights.length !== T) fail("سطرهای وزن");
    a.weights.forEach((row, index) => {
      distribution(row, T, "وزن توجه");
      row.forEach((weight, column) => {
        if (a.mask[index][column] !== (column <= index) ||
            (!a.mask[index][column] && (a.masked_scores[index][column] !== null || weight !== 0)) ||
            (a.mask[index][column] && a.masked_scores[index][column] !== a.scaled_scores[index][column])) fail("قرارداد علّیت");
      });
    });
    matrix(data.output.logits, T, N, "امتیازهای خروجی");
    if (!Array.isArray(data.output.probabilities) || data.output.probabilities.length !== T) fail("احتمال‌های خروجی");
    data.output.probabilities.forEach(row => distribution(row, N, "توزیع خروجی"));
    id(data.output.greedy_id, "انتخاب بیشینه");
    const lastLogits = data.output.logits[T - 1];
    const largestLogit = lastLogits.reduce((maximum, value) => Math.max(maximum, value), -Infinity);
    if (data.output.greedy_id !== lastLogits.indexOf(largestLogit)) fail("انتخاب بیشینهٔ آخرین موقعیت");
    if (!Array.isArray(data.output.top_candidates) || data.output.top_candidates.length !== Math.min(10, N)) fail("نامزدها");
    const candidates = new Set();
    data.output.top_candidates.forEach(candidate => {
      object(candidate, ["id", "token", "probability"], "نامزد"); id(candidate.id, "شناسهٔ نامزد");
      if (candidates.has(candidate.id) || candidate.token !== vocabulary[candidate.id] || candidate.probability !== data.output.probabilities[T - 1][candidate.id]) fail("مقدار نامزد");
      candidates.add(candidate.id);
    });
    const generation = data.generation;
    object(generation, ["settings", "steps", "text", "token_ids"], "تولید");
    const settings = generation.settings;
    object(settings, ["temperature", "top_k", "top_p", "greedy", "seed"], "روش تولید");
    finite(settings.temperature, "دما");
    if (settings.temperature <= 0 || typeof settings.greedy !== "boolean") fail("روش تولید");
    if (settings.top_k !== null) integer(settings.top_k, 1, Number.MAX_SAFE_INTEGER, "k");
    if (settings.top_p !== null && (typeof settings.top_p !== "number" || settings.top_p <= 0 || settings.top_p > 1)) fail("p");
    integer(settings.seed, -Number.MAX_SAFE_INTEGER, Number.MAX_SAFE_INTEGER, "بذر");
    if (!Array.isArray(generation.steps) || generation.steps.length > 4) fail("تعداد گام تولید");
    const produced = input.full_token_ids.slice();
    generation.steps.forEach((step, index) => {
      object(step, ["index", "context_token_ids", "token_id", "token", "logits", "probabilities", "sampling_probabilities"], "گام تولید");
      if (step.index !== index + 1) fail("شمارهٔ گام تولید");
      shape(step.context_token_ids, produced.slice(-input.context_limit), "زمینهٔ گام تولید");
      id(step.token_id, "شناسهٔ تولیدشده");
      if (step.token !== vocabulary[step.token_id]) fail("نشانهٔ تولیدشده");
      vector(step.logits, N, "امتیاز گام تولید"); distribution(step.probabilities, N, "توزیع خام تولید");
      distribution(step.sampling_probabilities, N, "توزیع انتخاب");
      if (step.sampling_probabilities[step.token_id] <= 0) fail("انتخاب نامزد حذف‌شده");
      produced.push(step.token_id);
    });
    shape(generation.token_ids, produced, "شناسه‌های متن تولیدی");
    if (generation.text !== produced.map(number => number === 0 ? "�" : vocabulary[number]).join("")) fail("متن تولیدی");
    if (data.metrics !== null) {
      object(data.metrics, ["columns", "rows"], "گزارش آموزش");
      const columns = ["step", "train_batch_loss", "validation_loss", "gradient_norm"];
      if (data.metrics.columns?.length === 5) columns.push("learning_rate");
      shape(data.metrics.columns, columns, "ستون‌های گزارش");
      if (!Array.isArray(data.metrics.rows) || data.metrics.rows.length > 10000) fail("سطرهای گزارش");
      let previous = 0;
      data.metrics.rows.forEach(row => {
        object(row, columns, "سطر گزارش"); columns.forEach(key => finite(row[key], key));
        integer(row.step, previous + 1, Number.MAX_SAFE_INTEGER, "گام گزارش"); previous = row.step;
        if (columns.some(key => row[key] < 0)) fail("مقدار منفی گزارش");
      });
    }
    if (!Array.isArray(data.notes) || data.notes.length > 30) fail("یادداشت‌ها");
    data.notes.forEach(note => string(note, 4000, "یادداشت"));
    return data;
  }

  function conceptSegments(text, concepts) {
    const names = Object.keys(concepts).sort((a,b) => b.length-a.length);
    if (!names.length) return [{text}];
    const escaped = names.map(name => name.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
    const pattern = new RegExp("(?<![A-Za-z0-9_.])(?:"+escaped.join("|")+")(?![A-Za-z0-9_.])", "g");
    const segments = []; let start = 0;
    for (const match of text.matchAll(pattern)) {
      if (match.index > start) segments.push({text:text.slice(start,match.index)});
      segments.push({text:match[0], slug:concepts[match[0]]}); start = match.index+match[0].length;
    }
    if (start < text.length) segments.push({text:text.slice(start)});
    return segments;
  }
  const api = { softmax, tokenization, attention, tensorShapes, validateInspection, conceptSegments };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  if (typeof document === "undefined") return;

  const element = (tag, text, className) => {
    const node = document.createElement(tag);
    if (text !== undefined) node.textContent = String(text);
    if (className) node.className = className;
    return node;
  };
  // Call only for authored labels/explanations, never imported text, table data,
  // model output or editable records. User content remains literal text.
  function explain(node) {
    const text = node.textContent;
    node.replaceChildren();
    const root = new URL(".", new URL(document.body.dataset.root, location.href));
    for (const segment of conceptSegments(text, window.BOOK?.concepts || {})) {
      if (!segment.slug) { node.append(document.createTextNode(segment.text)); continue; }
      const anchor = element("a", undefined, "term-link"), label = element("bdi", segment.text);
      label.dir = "ltr"; anchor.append(label); anchor.dataset.term = segment.slug;
      anchor.title = `توضیح ${segment.text}`;
      const initialOrigin = new URL(location.pathname, location.href); initialOrigin.hash = "main";
      const initialTarget = new URL(`glossary/${segment.slug}.html`, root);
      initialTarget.searchParams.set("return", initialOrigin.href);
      anchor.href = initialTarget.href; // Native open-in-new-tab also keeps this page as its origin.
      anchor.addEventListener("click", () => {
        const station = anchor.closest(".lab-station, [data-token-lab], [data-probability-lab], [data-attention-lab], [data-shape-lab]");
        const origin = new URL(location.pathname, location.href);
        origin.hash = station?.closest(".lab-station")?.id || station?.id || "main";
        const target = new URL(anchor.href); target.searchParams.set("return", origin.href); anchor.href = target.href;
      });
      node.append(anchor);
    }
    return node;
  }
  const authored = (tag,text,className) => explain(element(tag,text,className));
  const number = value => value === null ? "−∞ · ممنوع" : typeof value === "boolean" ? (value ? "۱ · مجاز" : "۰ · ممنوع") :
    (Math.abs(value) < 0.0001 && value !== 0 ? value.toExponential(5) : Number(value.toPrecision(7)).toString());
  const exact = value => value === null || typeof value === "boolean" ? number(value) : String(value);
  const labelToken = token => token === " " ? "␠ فاصله" : token === "\n" ? "↵ خط تازه" : token === "\t" ? "⇥ زبانه" : token === "\u200c" ? "نیم‌فاصله" : token;
  function table(headers, rows, caption, selectedRow = -1) {
    const wrap = element("div", undefined, "table-wrap");
    const node = element("table", undefined, "matrix");
    if (caption) node.append(authored("caption", caption));
    const head = element("thead"), headRow = element("tr");
    headers.forEach(text => { const cell = authored("th", text); cell.scope = "col"; headRow.append(cell); });
    head.append(headRow); node.append(head);
    const body = element("tbody");
    let page = 0;
    const pageSize = 200;
    const pager = element("div", undefined, "control-row"), range = element("span");
    range.setAttribute("role", "status");
    const previous = element("button", "۲۰۰ سطر قبلی"), next = element("button", "۲۰۰ سطر بعدی");
    previous.type = next.type = "button";
    function draw() {
      body.replaceChildren();
      rows.slice(page * pageSize, (page + 1) * pageSize).forEach((row, localIndex) => {
        const tr = element("tr"); if (page * pageSize + localIndex === selectedRow) tr.className = "selected-row";
        row.forEach((text, column) => { const cell = element(column === 0 ? "th" : "td", text); if (column === 0) cell.scope = "row"; tr.append(cell); });
        body.append(tr);
      });
      range.textContent = `سطرهای ${page * pageSize + 1} تا ${Math.min(rows.length, (page + 1) * pageSize)} از ${rows.length}؛ هیچ سطری حذف نشده است.`;
      previous.disabled = page === 0; next.disabled = (page + 1) * pageSize >= rows.length;
    }
    previous.addEventListener("click", () => { page--; draw(); }); next.addEventListener("click", () => { page++; draw(); });
    draw(); node.append(body); wrap.append(node);
    if (rows.length > pageSize) { pager.append(previous, range, next); wrap.append(pager); }
    return wrap;
  }
  function matrix(values, caption, selectedRow = -1, heat = false) {
    const headers = ["سطر / ستون", ...values[0].map((_, index) => index)];
    const compact = value => typeof value === "number" ? (value !== 0 && Math.abs(value)<0.0001 ? value.toExponential(2) : String(Number(value.toFixed(4)))) : exact(value);
    const panel = element("div");
    const grid = table(headers,values.map((row,index)=>[index,...row.map(compact)]),caption+" · نمایش گرد‌شده",selectedRow);
    [...grid.querySelectorAll("tbody tr")].forEach((row,i)=>[...row.children].slice(1).forEach((cell,j)=>{
      const value=values[i][j];cell.title=exact(value);
      if(value===null||value===false)cell.classList.add("masked");
      else if(heat&&typeof value==="number")cell.style.backgroundColor=`rgba(48,108,85,${Math.max(0,Math.min(1,value))*.45})`;
    }));
    panel.append(grid);
    const details=element("details");details.append(element("summary","عددهای کامل این ماتریس؛ بدون گردکردن"));
    details.append(table(headers,values.map((row,index)=>[index,...row.map(exact)]),caption+" · دقت کامل فایل",selectedRow));panel.append(details);return panel;
  }
  function vector(values, caption) {
    const box = element("div", undefined, "vector-grid"); box.append(authored("h3", caption));
    const code = element("code", "[" + values.map(exact).join(", ") + "]", "vector-values"); code.dir = "ltr";
    box.append(code); return box;
  }
  function bars(labels, probabilities) {
    const fragment = document.createDocumentFragment();
    probabilities.forEach((probability, index) => {
      const row = element("div", undefined, "probability-row");
      row.append(element("span", labels[index]));
      const track = element("div", undefined, "probability-track"), fill = element("div", undefined, "probability-fill");
      fill.style.width = `${Math.max(0, Math.min(1, probability)) * 100}%`;
      track.setAttribute("aria-hidden", "true"); track.append(fill); row.append(track, element("span", `${number(probability * 100)}٪`));
      fragment.append(row);
    });
    return fragment;
  }

  function setupTokens(root) {
    root.id ||= "term-ref-live-tokens-0";
    const input = root.querySelector("[data-token-text]"), strip = root.querySelector("[data-token-strip]"), detail = root.querySelector("[data-token-detail]");
    let selected = 0;
    function render() {
      const data = tokenization(input.value); selected = Math.min(selected, Math.max(0, data.tokens.length - 1));
      strip.replaceChildren();
      data.tokens.forEach((token, index) => {
        const id = data.ids[index], button = element("button", `${labelToken(token)} · ${id}`);
        button.type = "button"; button.setAttribute("aria-pressed", String(selected === index));
        button.addEventListener("click", () => { selected = index; render(); strip.children[index].focus(); });
        strip.append(button);
      });
      if (!data.tokens.length) { detail.textContent = "متنی بنویسید تا Tokenها دیده شوند."; explain(detail); return; }
      const token = data.tokens[selected], id = data.ids[selected];
      const manual = [id / 10, (id % 3) - 1, id === 0 ? 0 : 1];
      detail.textContent = `موقعیت ${selected}؛ نقطه‌کد U+${token.codePointAt(0).toString(16).toUpperCase()}؛ شناسه ${id}${id === 0 ? " (ناشناخته)" : ""}؛ سطر مثال دستی: [${manual.map(number).join(", ")}]. ${data.tokens.length} نقطه‌کد و ${data.vocabulary.length} سطر در Vocabulary داریم. این Vector وزن آموزش‌دیده نیست.`;
      explain(detail);
    }
    input.addEventListener("input", () => { selected = 0; render(); });
    root.querySelectorAll("[data-token-example]").forEach(button => button.addEventListener("click", () => { input.value = button.dataset.tokenExample; selected = 0; render(); }));
    render();
  }

  function setupProbabilities(root) {
    root.id ||= "term-ref-live-probability-0";
    function render() {
      const result = root.querySelector("[data-probability-result]");
      try {
        const fields = [...root.querySelectorAll("[data-logit]")];
        if (fields.some(field => field.value.trim() === "")) throw new Error("هر سه امتیاز را وارد کنید.");
        const logits = fields.map(field => Number(field.value));
        if (logits.some(value => !Number.isFinite(value) || value < -20 || value > 20)) throw new Error("امتیازها باید از منفی ۲۰ تا ۲۰ باشند.");
        const temperature = Number(root.querySelector("[data-temperature]").value);
        const probabilities = softmax(logits, temperature);
        root.querySelector("[data-temperature-value]").textContent = number(temperature);
        root.querySelector("[data-probability-bars]").replaceChildren(bars(["الف · هدف", "ب", "ج"], probabilities));
        result.textContent = `جمع احتمال‌ها: ${number(probabilities.reduce((a, b) => a + b, 0))}؛ Loss هدف الف: −ln(p) = ${number(-Math.log(probabilities[0]))}. با همین Logits و همین هدف محاسبه شده است.`;
        explain(result);
      } catch (error) { result.textContent = error.message; }
    }
    root.querySelectorAll("input").forEach(input => input.addEventListener("input", render)); render();
  }

  function setupAttention(root) {
    root.id ||= "term-ref-live-attention-0";
    function render() {
      const causal = root.querySelector("[data-causal]").checked, row = Number(root.querySelector("[data-query]").value), data = attention(causal);
      root.querySelector("[data-attention-scores]").replaceChildren(matrix(data.scaled, "امتیاز پیش از Causal Mask", row));
      root.querySelector("[data-attention-weights]").replaceChildren(matrix(data.weights, "وزن هر Key برای هر Query", row, true));
      const terms = data.weights[row].map((weight, index) => `${number(weight)}×[${V[index].join(",")}]`).join(" + ");
      root.querySelector("[data-attention-result]").textContent = `سطر ${row}: ${terms} = [${data.output[row].map(number).join(", ")}]. جمع وزن‌ها ${number(data.weights[row].reduce((a, b) => a + b, 0))} است. ${causal ? "تنها خود موقعیت و گذشته مجازند." : "آینده دیده می‌شود؛ این حالت برای آموزش زبان علّی معتبر نیست."}`;
    }
    root.querySelectorAll("input,select").forEach(input => input.addEventListener("change", render)); render();
  }

  function setupShapes(root) {
    root.id ||= "term-ref-live-shapes-0";
    function render() {
      const output = root.querySelector("[data-shape-result]");
      try {
        const data = tensorShapes(...["B", "T", "C", "H"].map(key => Number(root.querySelector(`[data-dimension="${key}"]`).value)));
        const lines = [["شناسه", data.ids], ["Embedding", data.embedding], ["تقسیم ویژگی‌ها", data.beforeTranspose], ["جابه‌جایی زمان و Head", data.heads], ["امتیازهای Attention", data.scores], ["ادغام دوباره", data.embedding]];
        const shapes = table(["گام", "Shape"], lines.map(([name, shape]) => [name, `(${shape.join(", ")})`]), `D = C/H = ${data.D}`);
        shapes.querySelectorAll("tbody th").forEach(explain);
        output.replaceChildren(shapes,
          element("p", `${data.scoreElements.toLocaleString("fa-IR")} عدد × ۴ بایت = ${data.scoreBytes.toLocaleString("fa-IR")} بایت (${number(data.scoreBytes / 1048576)} MiB). MiB یعنی ۱۰۴۸۵۷۶ بایت؛ فقط یک ماتریس امتیاز، نه کل مدل.`, "readout"));
      } catch (error) { output.replaceChildren(element("p", error.message, "readout")); }
    }
    root.querySelectorAll("input").forEach(input => input.addEventListener("input", render)); render();
  }

  function lossChart(rows) {
    if (!rows.length) return element("p", "گزارش، سطر عددی ندارد.");
    const ns = "http://www.w3.org/2000/svg", svg = document.createElementNS(ns, "svg");
    svg.setAttribute("viewBox", "0 0 720 310"); svg.setAttribute("role", "img"); svg.setAttribute("class", "loss-chart");
    const title = document.createElementNS(ns, "title"); title.textContent = "منحنی Loss مربوط به Batch آموزش و ارزیابی؛ جدول دقیق در ادامه است."; svg.append(title);
    const minStep = rows[0].step, maxStep = rows[rows.length - 1].step;
    const maxLoss = Math.max(1e-9, ...rows.flatMap(row => [row.train_batch_loss, row.validation_loss]));
    const x = step => 55 + ((step - minStep) / Math.max(1, maxStep - minStep)) * 640;
    const y = loss => 265 - loss / maxLoss * 220;
    const add = (tag, attributes, text) => { const node = document.createElementNS(ns, tag); for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, String(value)); if (text) node.textContent = text; svg.append(node); return node; };
    add("path", { d: "M55 35 V265 H700", fill: "none", stroke: "currentColor", "stroke-width": 1.5 });
    add("text", { x: 55, y: 292, "font-size": 13 }, String(minStep)); add("text", { x: 665, y: 292, "font-size": 13 }, String(maxStep));
    add("text", { x: 5, y: 45, "font-size": 12 }, number(maxLoss)); add("text", { x: 22, y: 269, "font-size": 12 }, "0");
    const series = [{ key: "train_batch_loss", color: "#91621d", dash: "7 5", label: "Batch آموزش · خط‌چین" }, { key: "validation_loss", color: "#16706c", dash: "none", label: "ارزیابی · پیوسته" }];
    series.forEach((line, index) => {
      add("polyline", { points: rows.map(row => `${x(row.step)},${y(row[line.key])}`).join(" "), fill: "none", stroke: line.color, "stroke-width": 2.5, "stroke-dasharray": line.dash });
      rows.forEach(row => add("circle", { cx: x(row.step), cy: y(row[line.key]), r: 2.7, fill: line.color }));
      add("text", { x: 100 + 300 * index, y: 20, fill: line.color, "font-size": 14 }, line.label);
    });
    return svg;
  }

  function setupInspector(root) {
    let current = null, importSequence = 0;
    const status = root.querySelector("[data-inspect-status]"), output = root.querySelector("[data-inspect-output]"),
      position = root.querySelector("[data-inspect-position]"), view = root.querySelector("[data-inspect-view]"),
      content = root.querySelector("[data-inspect-content]");
    function install(data) {
      const validated = validateInspection(data); // Do not mutate the current view until validation succeeds.
      const options = validated.input.context_tokens.map((token, index) => {
        const option = element("option", `${index} · ${labelToken(token)} · ID ${validated.input.context_token_ids[index]}`); option.value = index; return option;
      });
      current = validated; position.replaceChildren(...options); position.value = String(options.length - 1);
      const source = root.querySelector("[data-inspect-source]");
      source.replaceChildren(element("p", `${current.source.checkpoint_name} · گام آموزش ${current.source.step} · PyTorch ${current.source.torch_version} · لایه ${current.attention.layer} / سر ${current.attention.head}`),
        element("p", `متن: ${current.input.prompt}`), authored("p", `Shapeها: IDs (${current.shapes.token_ids.join(",")}) → Embedding (${current.shapes.embedding.join(",")}) → QKV (${current.shapes.qkv.join(",")}) → امتیاز Attention (${current.shapes.scores.join(",")}) → Logits (${current.shapes.logits.join(",")})`),
        element("p", `SHA-256: ${current.source.checkpoint_sha256}`, "meta"),
        authored("p", `زمینهٔ بازرسی ${current.input.context_limit}؛ ${current.input.truncated ? "بخش آغازین متن در محاسبه بریده شده است" : "متن اولیه در زمینه جا می‌شود"}؛ ${current.input.unknown_positions.length} Token ناشناخته.`, "meta"));
      const notes = element("details"); notes.append(element("summary", "قرارداد و اطلاعات اجرای منبع"));
      current.notes.forEach(note => notes.append(element("p", note)));
      const metadata = element("pre"); metadata.append(element("code", JSON.stringify(current.source.training_metadata, null, 2))); notes.append(metadata); source.append(notes);
      output.hidden = false; render(); status.textContent = "دادهٔ بازرسی معتبر بارگذاری شد. عددها محلی‌اند؛ مدلی در مرورگر آموزش نمی‌بیند.";
    }
    function render() {
      if (!current) return;
      const at = Number(position.value), a = current.attention, part = document.createDocumentFragment();
      part.append(element("p", `موقعیت ${at} · ${labelToken(current.input.context_tokens[at])} · شناسه ${current.input.context_token_ids[at]}`, "readout"));
      switch (view.value) {
        case "embeddings":
          part.append(vector(current.embeddings.token[at], "سطر جدول Token Embedding"), vector(current.embeddings.position[at], "سطر جدول Positional Embedding"), vector(current.embeddings.combined[at], "جمع دو Embedding؛ ورودی بلوک‌ها"));
          part.append(table(["بلوک", "ورودی", "وزن Attention", "Feed-Forward Network", "خروجی"], current.layers.map(layer => [layer.index, layer.input_shape.join("×"), layer.attention_shape.join("×"), layer.feed_forward_shape.join("×"), layer.output_shape.join("×")]), "خلاصهٔ Shape همهٔ بلوک‌ها"));
          break;
        case "qkv":
          part.append(vector(a.q[at], "Query موقعیت منتخب"), vector(a.k[at], "Key موقعیت منتخب"), vector(a.v[at], "Value موقعیت منتخب"), vector(a.weighted_values[at], "خروجی ترکیب این Head: weights @ V"));
          part.append(authored("p", "این Query، Key و Value از ورودی نرمال‌شدهٔ همین بلوک آمده‌اند؛ مستقیماً از متن یا جدول خام Token Embedding ساخته نشده‌اند.")); break;
        case "scores":
          part.append(matrix(a.raw_scores, "Dot product خام QKᵀ", at), matrix(a.scaled_scores, `تقسیم بر √D؛ D=${a.head_dim}`, at), matrix(a.masked_scores, "امتیاز پس از Causal Mask؛ ممنوع یعنی −∞ نه صفر", at)); break;
        case "weights":
          part.append(matrix(a.mask, "Causal Mask بولی: مجاز / ممنوع", at), matrix(a.weights, "وزن پیش از Dropout؛ در این اجرای eval، Dropout خاموش است", at, true), vector(a.weighted_values[at], "جمع وزن‌دار Value برای موقعیت منتخب"));
          part.append(authored("p", `جمع سطر منتخب = ${number(a.weights[at].reduce((sum, value) => sum + value, 0))}. سطر Query است و ستون Key؛ این ضرایب توضیح کامل علت پاسخ نیستند.`)); break;
        case "logits": {
          const order = current.output.vocabulary.map((_, index) => index).sort((x, y) => current.output.probabilities[at][y] - current.output.probabilities[at][x] || x - y);
          part.append(authored("p", "همهٔ نامزدهای Vocabulary، مرتب از احتمال بیشتر؛ پیش‌بینی این سطر برای Token بعد از موقعیت منتخب است."));
          part.append(table(["شناسه", "Token", "Logits", "احتمال"], order.map(id => [id, labelToken(current.output.vocabulary[id]), exact(current.output.logits[at][id]), exact(current.output.probabilities[at][id])]), "امتیاز و احتمال کامل؛ هیچ نامزد حذف نشده است")); break;
        }
        case "generation":
          part.append(authored("p", `تنظیمات: Temperature ${current.generation.settings.temperature}؛ k=${current.generation.settings.top_k ?? "خاموش"}؛ p=${current.generation.settings.top_p ?? "خاموش"}؛ انتخاب حریصانه ${current.generation.settings.greedy ? "بله" : "خیر"}؛ بذر ${current.generation.settings.seed}.`), element("p", `متن خروجی: ${current.generation.text}`, "readout"));
          if (!current.generation.steps.length) part.append(element("p", "این فایل با صفر گام تولید صادر شده است."));
          current.generation.steps.forEach(step => {
            const block = element("section", undefined, "generation-step");
            block.append(element("h3", `گام ${step.index}: ${labelToken(step.token)} · ID ${step.token_id}`), element("p", `زمینهٔ شناسه‌ها: [${step.context_token_ids.join(", ")}]`),
              authored("p", "احتمال خام مدل پیش از Temperature و فیلتر است. احتمال انتخاب پس از آن‌هاست؛ در انتخاب حریصانه یک خانه احتمال یک دارد."));
            const details = element("details"); details.append(element("summary", "همهٔ امتیازها و دو توزیع این گام"));
            const order = current.output.vocabulary.map((_, index) => index).sort((x, y) => step.probabilities[y] - step.probabilities[x] || x - y);
            details.append(table(["شناسه", "Token", "Logits", "احتمال خام", "احتمال انتخاب"], order.map(id => [id, labelToken(current.output.vocabulary[id]), exact(step.logits[id]), exact(step.probabilities[id]), exact(step.sampling_probabilities[id])]), "توزیع کامل گام تولید")); block.append(details); part.append(block);
          }); break;
        case "metrics":
          if (current.metrics === null) { part.append(element("p", "این فایل گزارش آموزش ندارد. هنگام صدور، --metrics را با مسیر metrics.csv همان اجرا بدهید.")); break; }
          part.append(authored("p", "خط Training مربوط به یک Batch همان گام است؛ خط ارزیابی میانگین پنجره‌های کنارگذاشته‌شده است. نویز Training و اختلاف داده و حالت را با تعمیم اشتباه نگیرید. گزارش ممکن است تا گامی پس از بهترین Checkpoint ادامه داشته باشد."), lossChart(current.metrics.rows));
          part.append(table(["گام", "Loss مربوط به Batch آموزش", "Loss ارزیابی", "اندازهٔ Gradient", "Learning Rate"], current.metrics.rows.map(row => [row.step, exact(row.train_batch_loss), exact(row.validation_loss), exact(row.gradient_norm), row.learning_rate === undefined ? "در گزارش قدیمی ثبت نشده" : exact(row.learning_rate)]), "عددهای گزارش واقعی؛ Learning Rate و Loss هم‌مقیاس نیستند و روی یک نمودار رسم نشده‌اند")); break;
      }
      content.replaceChildren(part);
    }
    root.querySelector("[data-inspect-demo]").addEventListener("click", () => {
      ++importSequence;
      try { if (!window.INSPECTION_DEMO) throw new Error("نمونهٔ همراه پیدا نشد؛ فایل JSON خودتان را وارد کنید."); install(window.INSPECTION_DEMO); }
      catch (error) { status.textContent = `${error.message} نمای قبلی تغییر نکرد.`; }
    });
    root.querySelector("[data-inspect-file]").addEventListener("change", async event => {
      const file = event.target.files[0], sequence = ++importSequence; if (!file) return;
      try {
        if (file.size > 20 * 1024 * 1024) throw new Error("حداکثر اندازهٔ فایل ۲۰ MiB است.");
        const text = await file.text(); if (sequence !== importSequence) return;
        const data = JSON.parse(text); install(data);
      } catch (error) { if (sequence === importSequence) status.textContent = `ورود ناموفق: ${error.message} نمای قبلی حفظ شد.`; }
      finally { event.target.value = ""; }
    });
    position.addEventListener("change", render); view.addEventListener("change", render);
  }

  function initialize() {
    document.querySelectorAll("[data-token-lab]").forEach(setupTokens);
    document.querySelectorAll("[data-probability-lab]").forEach(setupProbabilities);
    document.querySelectorAll("[data-attention-lab]").forEach(setupAttention);
    document.querySelectorAll("[data-shape-lab]").forEach(setupShapes);
    document.querySelectorAll("[data-inspector]").forEach(setupInspector);
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", initialize);
  else initialize();
})();
