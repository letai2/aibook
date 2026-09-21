"use strict";
(() => {
  const manifest = window.BOOK;
  const allowed = new Set(manifest.units.map(x => x.id));
  const key = "mini-gpt-progress-v3";
  const root = new URL(".", new URL(document.body.dataset.root, location.href));
  // Carry a reading location through the concept graph, without storing it or
  // accepting arbitrary/external redirect targets. Native Back also keeps scroll.
  const readingReturn = new URL(location.href).searchParams.get('return');
  const glossarySearch = document.querySelector('[data-glossary-search]');
  glossarySearch?.addEventListener('input', () => {
    const query = glossarySearch.value.trim().toLocaleLowerCase();
    let count = 0;
    document.querySelectorAll('[data-concept-entry]').forEach(entry => {
      const match = entry.dataset.conceptSearch.toLocaleLowerCase().includes(query);
      entry.hidden = !match;
      if (match) count += 1;
    });
    document.querySelector('[data-glossary-count]').textContent = query ? `${count} مفهوم پیدا شد` : '';
    document.querySelector('.glossary-more').open = Boolean(query);
  });
  if (readingReturn && document.querySelector('[data-glossary-return]')) {
    try {
      const destination = new URL(readingReturn, location.href);
      const route = decodeURIComponent(destination.pathname.slice(root.pathname.length));
      if (destination.protocol === root.protocol && destination.host === root.host &&
          destination.pathname.startsWith(root.pathname) && manifest.pages?.includes(route) &&
          !destination.search && /^#(?:term-ref-(?:[a-z0-9]+-)*\d+|main|tokens|probability|attention|shapes|model)$/.test(destination.hash)) {
        const back = document.querySelector('[data-glossary-return]');
        back.href = destination.href;
        back.textContent = 'بازگشت به همان جای مطالعه';
        document.querySelectorAll('a[data-term], .glossary-related a').forEach(anchor => {
          const target = new URL(anchor.href, location.href);
          target.searchParams.set('return', readingReturn);
          anchor.href = target.href;
        });
      }
    } catch (_) { /* Keep the useful, static introductory-lesson fallback. */ }
  }
  let states = Object.create(null), storageOK = true;
  const status = message => { const node = document.querySelector("[data-status]"); if (node) node.textContent = message; };
  const validIDs = ids => Array.isArray(ids) && ids.every(x => typeof x === "string" && allowed.has(x));
  function parseStates(raw) {
    if (!raw) return Object.create(null);
    const data = JSON.parse(raw);
    if (!data || data.version !== 3 || typeof data.states !== "object" || !data.states) throw Error("invalid progress");
    const out = Object.create(null);
    for (const [id, value] of Object.entries(data.states)) {
      if (!allowed.has(id) || !value || typeof value.done !== "boolean" || !Number.isSafeInteger(value.time) || value.time < 0) throw Error("invalid progress");
      out[id] = {done:value.done, time:value.time};
    }
    return out;
  }
  function merge(other) {
    for (const [id, value] of Object.entries(other)) if (!states[id] || value.time > states[id].time) states[id] = value;
  }
  let stored = null;
  try {
    stored = localStorage.getItem(key); states = parseStates(stored);
    if (!stored) {
      const old = JSON.parse(localStorage.getItem("mini-gpt-book-edition-2") || "[]");
      if (validIDs(old)) old.forEach(id => states[id] = {done:true,time:1});
    }
  } catch (_) { storageOK = false; }
  // URL state is a fallback only, never a stale authority over an existing browser record.
  if (!stored) {
    const incoming = new URL(location.href).searchParams.get("done");
    const ids = incoming ? incoming.split(",") : [];
    if (validIDs(ids)) ids.forEach(id => states[id] = {done:true,time:1});
  }
  // File origins may isolate storage per page: transport revisions INCLUDING undone items.
  // Existing HTTP storage remains authoritative over bookmark/query state.
  if(location.protocol === "file:" || !storageOK){
    const carried=new URL(location.href).searchParams.get("progress");
    if(carried&&carried.length<=30000){try{merge(parseStates(carried));}catch(_){status("پیشرفت موجود در پیوند معتبر نبود و نادیده گرفته شد.");}}
  }
  function persist() {
    try { merge(parseStates(localStorage.getItem(key))); localStorage.setItem(key, JSON.stringify({version:3,states})); }
    catch (_) { storageOK = false; }
  }
  function render() {
    const ids = Object.keys(states).filter(id => states[id].done).sort();
    const done = new Set(ids);
    document.querySelectorAll("[data-progress-count]").forEach(x => x.textContent = `${ids.length} / ${allowed.size}`);
    document.querySelectorAll("progress").forEach(x => {x.max=allowed.size;x.value=ids.length;});
    document.querySelectorAll("[data-unit]").forEach(x => x.classList.toggle("completed",done.has(x.dataset.unit)));
    const button = document.querySelector("[data-complete]");
    if (button) { const yes=done.has(button.dataset.complete); button.setAttribute("aria-pressed",String(yes));button.textContent=yes?"✓ انجام شد؛ برای لغو کلیک کنید":"تمرین و خودسنجی را انجام دادم"; }
    const fallback = !storageOK || location.protocol === "file:";
    document.querySelectorAll("a[data-book-link]").forEach(a => {
      const url=new URL(a.getAttribute("href"),location.href);
      url.searchParams.delete("done");
      if(fallback) url.searchParams.set("progress",JSON.stringify({version:3,states})); else url.searchParams.delete("progress");
      a.href=url.href;
    });
    const current=new URL(location.href);
    current.searchParams.delete("done");
    if(fallback) current.searchParams.set("progress",JSON.stringify({version:3,states})); else current.searchParams.delete("progress");
    try { history.replaceState(null,"",current); } catch (_) { /* some file origins reject history changes */ }
    if(!storageOK) status("ذخیرهٔ مرورگر در دسترس نیست. پیشرفت در پیوندها منتقل می‌شود؛ خروجی JSON بگیرید.");
  }
  document.querySelector("[data-complete]")?.addEventListener("click",event=>{
    persist(); const id=event.currentTarget.dataset.complete;
    states[id]={done:!states[id]?.done,time:Math.max(Date.now(),(states[id]?.time||0)+1)};
    persist();render();
  });
  window.addEventListener("storage",event=>{if(event.key===key){try{merge(parseStates(event.newValue));render();}catch(_){status("وضعیت زبانهٔ دیگر معتبر نبود؛ تغییری اعمال نشد.");}}});
  document.querySelectorAll("pre").forEach(pre=>{
    if(pre.closest("[data-export-preview]"))return;
    const wrapper=document.createElement("div");wrapper.className="codebox";pre.parentNode.insertBefore(wrapper,pre);wrapper.append(pre);
    const label=document.createElement("span");label.className="code-label";label.textContent=(pre.dataset.codeLanguage||"code").toUpperCase();label.setAttribute("aria-hidden","true");wrapper.append(label);
    const button=document.createElement("button");button.type="button";button.textContent="کپی کد";wrapper.insertBefore(button,pre);
    button.addEventListener("click",async()=>{
      try{if(!navigator.clipboard||!window.isSecureContext)throw Error("clipboard unavailable");await navigator.clipboard.writeText(pre.textContent);button.textContent="کپی شد";}
      catch(_){const range=document.createRange();range.selectNodeContents(pre);const selection=window.getSelection();selection.removeAllRanges();selection.addRange(range);button.textContent="انتخاب شد؛ Ctrl+C بزنید";}
    });
  });
  function downloadJSON(value,name){
    const serialized=JSON.stringify(value,null,2);
    try{const url=URL.createObjectURL(new Blob([serialized],{type:"application/json"}));const a=document.createElement("a");a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),2000);}catch(_){/* visible text is always available */}
    return serialized;
  }
  window.BookUI={downloadJSON,root};
  document.querySelector("[data-export]")?.addEventListener("click",()=>{
    persist();const data={edition:3,completed:Object.keys(states).filter(id=>states[id].done).sort()};
    const preview=document.querySelector("[data-export-preview]");preview.hidden=false;preview.open=true;
    document.querySelector("[data-export-code]").textContent=downloadJSON(data,"learning-progress.json");
    status("خروجی آماده است؛ اگر دانلود انجام نشد، نسخهٔ متنی را کپی کنید.");
  });
  document.querySelector("[data-import]")?.addEventListener("change",async event=>{
    const file=event.target.files[0];if(!file)return;
    try{if(file.size>50000)throw Error("too large");const data=JSON.parse(await file.text());if(![2,3].includes(data.edition)||!validIDs(data.completed))throw Error("invalid");persist();const time=Date.now();data.completed.forEach(id=>states[id]={done:true,time:Math.max(time,(states[id]?.time||0)+1)});persist();render();status("پیشرفت ادغام شد؛ موارد انجام‌شدهٔ فعلی حذف نشدند.");}
    catch(_){status("فایل معتبر نیست؛ خروجی پیشرفت ویرایش دوم یا سوم را وارد کنید. وضعیت فعلی حفظ شد.");}
    event.target.value="";
  });
  const search=document.querySelector("[data-lesson-search]");
  const menus=[...document.querySelectorAll('.header-actions > details, .atlas > details')];
  menus.forEach(menu=>menu.addEventListener('toggle',()=>{if(menu.open)menus.forEach(other=>{if(other!==menu)other.open=false;});}));
  document.addEventListener('keydown',event=>{if(event.key==='Escape'){const active=menus.find(menu=>menu.open);if(active){active.open=false;active.querySelector('summary').focus();}}});
  document.addEventListener('click',event=>{if(!event.target.closest('.header-actions'))menus.forEach(menu=>menu.open=false);});
  function revealStation(){
    const target=document.getElementById(location.hash.slice(1));
    if(!target)return;
    if(target.classList.contains('lab-station'))target.open=true;
    // A glossary return can point inside a collapsed experiment or project note.
    for(let parent=target.parentElement;parent;parent=parent.parentElement){
      if(parent.tagName==='DETAILS')parent.open=true;
    }
    target.scrollIntoView({block:'start'});
  }
  window.addEventListener('hashchange',revealStation);
  window.addEventListener('load',revealStation);
  revealStation();
  search?.addEventListener("input",()=>{
    const target=document.querySelector("[data-search-results]");target.replaceChildren();const q=search.value.trim().toLocaleLowerCase();if(!q)return;
    const matches=manifest.units.filter(x=>(x.title+" "+x.id).toLocaleLowerCase().includes(q)).slice(0,15);
    for(const item of matches){const a=document.createElement("a");a.textContent=item.title;a.href=new URL(item.path,root);a.dataset.bookLink="";target.append(a);}
    if(!matches.length)target.textContent="در عنوان درس‌ها پیدا نشد؛ واژهٔ کوتاه‌تری امتحان کنید.";
    render();
  });
  try{
    const current=document.body.dataset.unitId;
    if(allowed.has(current))localStorage.setItem("mini-gpt-last-lesson",current);
    const saved=localStorage.getItem("mini-gpt-last-lesson");const item=manifest.units.find(x=>x.id===saved);const resume=document.querySelector("[data-resume]");
    if(item&&resume){resume.href=new URL(item.path,root);resume.textContent="ادامه از آخرین صفحه: "+item.title;resume.dataset.bookLink="";resume.hidden=false;}
  }catch(_){/* resume is optional */}
  persist();render();
})();
