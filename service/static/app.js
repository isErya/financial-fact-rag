/* filing desk: the page script. No framework, no build step.

   Flow per question: POST /ask with dry_run true and paint the scope, the
   sources and the context budget (no model request, well under a second);
   then POST /ask with dry_run false and paint the answer and its evidence
   checks. A refusal stops after the first step. Every string that came
   from a filing or from the model lands in the DOM through textContent.
*/
(function () {
  "use strict";

  var QUESTIONS = [
    {label: "What are the primary risk factors facing Apple, Tesla, and JPMorgan, and how do they compare?",
     text: "What are the primary risk factors facing Apple, Tesla, and JPMorgan, and how do they compare?"},
    {label: "How has NVIDIA's revenue and growth outlook changed over the last two years?",
     text: "How has NVIDIA's revenue and growth outlook changed over the last two years?"},
    {label: "What regulatory risks do the major pharmaceutical companies face, and how are they addressing them?",
     text: "What regulatory risks do the major pharmaceutical companies face, and how are they addressing them?"},
    {label: "Bank disclosure brief",
     text: "Prepare a Q3 2025 bank-disclosure brief for a PE portfolio CFO. Compare JPMorgan and Bank of America on CET1 ratio, estimated uninsured deposits, and third-quarter net interest income. State the reporting period and units, cite each figure, and flag disclosures that are not comparable."}
  ];

  var REASON_TEXT = {
    newest_10q: "newest 10-Q",
    newest_10k: "newest 10-K",
    annual_baseline: "annual baseline",
    comparative: "comparative columns",
    comparative_columns: "comparative columns"
  };
  var SECTION_TEXT = {
    "1": "Item 1 Business",
    "1A": "Item 1A Risk Factors",
    "7": "Item 7 MD&A",
    "8": "Item 8 Financial Statements",
    "I.1": "10-Q Part I Item 1 Financial Statements",
    "I.2": "10-Q Part I Item 2 MD&A",
    "II.1A": "10-Q Part II Item 1A Risk Factors"
  };
  var REFUSAL_TEXT = {
    needs_company: "No company named. Name at least one of the covered companies; the list follows.",
    not_covered: "The companies named are not in the corpus. The covered companies follow.",
    period_not_covered: "The period asked for is outside the filings held for these companies."
  };
  // A figure in claim text: optional $ and parenthesis, digits with commas
  // and decimals, optional % and scale word. Mirrors the server's check
  // closely enough to find the same tokens for the page's own lookup.
  var FIGURE_RE = /(^|[^A-Za-z\d.])(\$?)(\(?)(\d(?:[\d,]*\d)?(?:\.\d+)?)(\)?)(%?)(?:\s*(thousand|million|billion)s?\b)?/gi;
  var NUMBER_RE = /\d(?:[\d,]*\d)?(?:\.\d+)?/g;
  var DATE_RE = /\b\d{4}-\d{2}-\d{2}\b|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?\s+\d{1,2}(?:\s*,\s*\d{4})?\b/gi;
  var REDUCED = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  var state = {
    health: null,
    question: "",
    dry: null,
    full: null,
    sources: {},
    controller: null,
    timer: null,
    run: 0
  };

  // -- DOM helpers -----------------------------------------------------------

  function $(id) { return document.getElementById(id); }

  function el(tag, cls, text) {
    var node = document.createElement(tag);
    if (cls) { node.className = cls; }
    if (text !== undefined && text !== null) { node.textContent = String(text); }
    return node;
  }

  function clear(node) { while (node.firstChild) { node.removeChild(node.firstChild); } }

  function badge(kind, text) {
    return el("span", "bh-badge" + (kind ? " bh-badge--" + kind : ""), text);
  }

  function chip(text, onClick, title) {
    var button = el("button", "fd-chip", text);
    button.type = "button";
    if (title) { button.title = title; }
    button.addEventListener("click", onClick);
    return button;
  }

  function callout(kind, eyebrow, text) {
    var box = el("div", "bh-callout" + (kind ? " bh-callout--" + kind : ""));
    if (eyebrow) { box.appendChild(el("span", "bh-eyebrow", eyebrow)); }
    if (text) { box.appendChild(el("div", null, text)); }
    return box;
  }

  function preBlock(text) {
    var pre = el("pre", "fd-pre", text);
    return pre;
  }

  function fmt(n) {
    if (n === null || n === undefined) { return "n/a"; }
    return Number(n).toLocaleString("en-US");
  }

  function setStatus(text) { $("status").textContent = text || ""; }

  function show(id, visible) { $(id).hidden = !visible; }

  // -- startup --------------------------------------------------------------

  function init() {
    var chips = $("chips");
    QUESTIONS.forEach(function (q) {
      var button = el("button", "bh-btn bh-btn--ghost", q.label);
      button.type = "button";
      button.addEventListener("click", function () {
        $("question").value = q.text;
        $("question").focus();
      });
      chips.appendChild(button);
    });
    $("ask").addEventListener("click", runAsk);
    $("question").addEventListener("keydown", function (event) {
      if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
        event.preventDefault();
        runAsk();
      }
    });
    $("details-toggle").addEventListener("click", function () {
      var open = $("details").hidden;
      $("details").hidden = !open;
      $("details-toggle").setAttribute("aria-expanded", open ? "true" : "false");
      $("details-toggle").textContent = open ? "hide details" : "details";
    });
    fetch("/health").then(function (r) { return r.json(); }).then(paintHealth).catch(function () {
      $("foot-index").textContent = "index: /health unreachable";
    });
  }

  function paintHealth(health) {
    state.health = health;
    show("key-banner", !health.llm_ready);
    if (health.index_error) {
      $("index-banner-text").textContent = health.index_error;
      show("index-banner", true);
      $("foot-index").textContent = "index: missing";
    } else {
      $("foot-index").textContent = "index: " + fmt(health.chunks) + " chunks over " + fmt(health.files) +
        " filings from " + fmt(health.tickers) + " companies" +
        (health.index_built_at ? ", built " + health.index_built_at : "") +
        (health.dense ? ", dense vectors on" : ", lexical only");
    }
    $("foot-model").textContent = "backend: " + health.backend + " / " + health.model +
      (health.llm_ready ? "" : " (no key)");
  }

  // -- the ask flow -----------------------------------------------------------

  function post(question, dryRun, signal) {
    return fetch("/ask", {
      method: "POST",
      headers: {"content-type": "application/json"},
      body: JSON.stringify({question: question, dry_run: dryRun}),
      signal: signal
    }).then(function (response) {
      return response.text().then(function (text) {
        var body = null;
        try { body = JSON.parse(text); } catch (err) { body = null; }
        var detail = body && body.detail ? (typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail)) : text;
        return {ok: response.ok, status: response.status, body: body, detail: detail};
      });
    });
  }

  function runAsk() {
    var question = $("question").value.trim();
    if (!question) { $("question").focus(); return; }
    askQuestion(question);
  }

  function askQuestion(question) {
    state.run += 1;
    var run = state.run;
    if (state.controller) { state.controller.abort(); }
    state.controller = new AbortController();
    var signal = state.controller.signal;
    state.question = question;
    state.dry = null;
    state.full = null;
    state.sources = {};
    stopTimer();
    resetBands();
    setStatus("reading the scope");
    $("ask").disabled = true;

    post(question, true, signal).then(function (res) {
      if (run !== state.run) { return; }
      $("ask").disabled = false;
      if (!res.ok) {
        setStatus("");
        show("scope-band", true);
        $("scope").appendChild(callout("alert", "request failed (HTTP " + res.status + ")", res.detail));
        return;
      }
      state.dry = res.body;
      indexSources(res.body);
      paintScope(res.body);
      paintSources(res.body);
      paintDetails(res.body);
      if (res.body.status !== "ok") {
        setStatus("stopped at the scope; no model request");
        return;
      }
      show("answer-band", true);
      if (state.health && !state.health.llm_ready) {
        setStatus("scope and sources ready; answer step disabled");
        $("answer").appendChild(callout("warn", "answer step disabled",
          "No API key configured; the interpreted scope and sources are shown, no model request was made."));
        return;
      }
      setStatus("scope and sources ready; asking the model");
      requestAnswer(question, run, signal);
    }).catch(function (err) {
      if (err && err.name === "AbortError") { return; }
      $("ask").disabled = false;
      setStatus("");
      show("scope-band", true);
      $("scope").appendChild(callout("alert", "request failed", String(err)));
    });
  }

  function requestAnswer(question, run, signal) {
    startTimer();
    clear($("answer"));
    post(question, false, signal).then(function (res) {
      if (run !== state.run) { return; }
      stopTimer();
      if (res.status === 501) {
        setStatus("scope and sources ready; answer step not wired yet");
        $("answer").appendChild(callout("warn", "not wired yet",
          "The answer step returned HTTP 501: " + res.detail + ". The scope and sources above are what the model will read."));
        return;
      }
      if (!res.ok) {
        setStatus("model request failed");
        paintAnswerError(res.status, res.detail, question, run, signal);
        return;
      }
      state.full = res.body;
      indexSources(res.body);
      paintReplay(res.body);
      paintAnswer(res.body);
      paintSources(res.body);
      paintDetails(res.body);
      setStatus(res.body.answer ? "answer ready" : "the model replied but no answer parsed");
    }).catch(function (err) {
      if (err && err.name === "AbortError") { return; }
      stopTimer();
      setStatus("model request failed");
      paintAnswerError(0, String(err), question, run, signal);
    });
  }

  function paintAnswerError(status, detail, question, run, signal) {
    var box = callout("alert", "model request failed" + (status ? " (HTTP " + status + ")" : ""), detail);
    var retry = el("button", "bh-btn bh-btn--ghost", "retry");
    retry.type = "button";
    retry.style.marginTop = "12px";
    retry.addEventListener("click", function () {
      // A new request against the same scope; the dry run is not repeated.
      clear($("answer"));
      requestAnswer(question, run, signal);
    });
    box.appendChild(retry);
    $("answer").appendChild(box);
  }

  function resetBands() {
    ["scope", "answer", "sources", "details"].forEach(function (id) { clear($(id)); });
    ["scope-band", "answer-band", "sources-band", "replay-banner"].forEach(function (id) { show(id, false); });
    $("progress").textContent = "";
    $("details").hidden = true;
    $("details-toggle").setAttribute("aria-expanded", "false");
    $("details-toggle").textContent = "details";
  }

  function startTimer() {
    var started = Date.now();
    $("progress").textContent = "model request in progress: 0 s";
    state.timer = setInterval(function () {
      $("progress").textContent = "model request in progress: " + Math.round((Date.now() - started) / 1000) + " s";
    }, 1000);
  }

  function stopTimer() {
    if (state.timer) { clearInterval(state.timer); state.timer = null; }
    $("progress").textContent = "";
  }

  function paintReplay(payload) {
    if (!payload.replayed) { show("replay-banner", false); return; }
    $("replay-banner-text").textContent = "Replayed from a saved response dated " +
      (payload.replayed_date || "unknown") + "; no live request was made.";
    show("replay-banner", true);
  }

  // -- scope ------------------------------------------------------------------

  function withoutCompany(question, company, plan) {
    // Removing a company means asking the question without its name. A
    // group phrase ("the major pharmaceutical companies") names several
    // companies at once, so it is replaced by the names that remain.
    var alias = company.matched_alias || company.name;
    var sharing = plan.companies.filter(function (c) {
      return c.matched_alias === company.matched_alias && c.ticker !== company.ticker;
    });
    var replacement = "";
    if (sharing.length) {
      var names = sharing.map(function (c) { return c.name; });
      replacement = names.length === 1 ? names[0] : names.slice(0, -1).join(", ") + " and " + names[names.length - 1];
    }
    var at = question.toLowerCase().indexOf(alias.toLowerCase());
    if (at < 0) { return question; }
    var before = question.slice(0, at);
    var after = question.slice(at + alias.length);
    var remaining = plan.companies.filter(function (c) {
      return c.ticker !== company.ticker && c.matched_alias !== company.matched_alias;
    }).map(function (c) { return c.matched_alias || c.name; });
    if (!replacement) {
      // Take the list separator out along with the name, so "facing Apple,
      // Tesla, and JPMorgan" loses "Apple, " and never keeps a comma right
      // after "facing". The separator after the name belongs to the list
      // only when another in-scope name follows it; otherwise the name was
      // last in the list and the separator before it goes instead.
      var sep = after.match(/^\s*,?\s*(?:and\s+)?/);
      var next = after.slice(sep ? sep[0].length : 0).toLowerCase();
      var startsWithName = remaining.some(function (name) {
        return next.indexOf(name.toLowerCase()) === 0;
      });
      if (sep && sep[0] && startsWithName) {
        after = after.slice(sep[0].length);
      } else {
        before = before.replace(/\s*,?\s*and\s*$|\s*,\s*$/i, "");
        if (!/^[,.?!;:]/.test(after)) { before += " "; }
      }
    }
    var text = before + replacement + after;
    if (remaining.length === 2) {
      // Two names left read as a pair, so the Oxford comma from the longer
      // list goes: "Tesla, and JPMorgan" becomes "Tesla and JPMorgan".
      var a = remaining[0].replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
      var b = remaining[1].replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
      text = text.replace(new RegExp("(" + a + "),\\s+(?:and\\s+)?(" + b + ")", "i"), "$1 and $2")
        .replace(new RegExp("(" + b + "),\\s+(?:and\\s+)?(" + a + ")", "i"), "$1 and $2");
    }
    text = text.replace(/\s+,/g, ",")
      .replace(/,(\s*,)+/g, ",")
      .replace(/,\s*and\s*,/g, ",")
      .replace(/\band\s*,/g, ",")
      .replace(/,\s*(and\s+)?(?=[,.?!]|$)/g, "")
      .replace(/\s+([.?!,])/g, "$1")
      .replace(/\s{2,}/g, " ")
      .trim();
    return text;
  }

  function paintScope(payload) {
    var plan = payload.plan;
    var root = $("scope");
    clear(root);
    show("scope-band", true);

    var line = el("p", "fd-scope-line");
    line.appendChild(badge("solid", "status: " + payload.status));
    line.appendChild(document.createTextNode(" "));
    line.appendChild(badge(null, "period: " + plan.period_mode));
    line.appendChild(document.createTextNode(" "));
    line.appendChild(badge(null, plan.comparison ? "comparison" : plan.timeline ? "timeline" : "single scope"));
    line.appendChild(document.createTextNode(" "));
    line.appendChild(badge(null, "budget " + fmt(plan.budget_tokens) + " tokens"));
    root.appendChild(line);

    if (payload.status !== "ok") {
      var refusal = callout("warn", "cannot answer: " + payload.status.replace(/_/g, " "),
        REFUSAL_TEXT[payload.status] || "The corpus cannot answer this question as asked.");
      if (plan.unresolved && plan.unresolved.length) {
        refusal.appendChild(el("p", null, "Not in the corpus: " + plan.unresolved.join(", ")));
      }
      refusal.appendChild(preBlock(payload.coverage || ""));
      root.appendChild(refusal);
      if (!plan.companies.length) { return; }
    }

    var grid = el("div", "fd-companies");
    plan.companies.forEach(function (company) {
      grid.appendChild(companyCard(company, plan, payload));
    });
    root.appendChild(grid);

    var sections = el("div", "fd-sections");
    sections.appendChild(el("span", "bh-eyebrow", "favoured sections"));
    var weights = Object.keys(plan.sections || {}).filter(function (k) { return k !== "*"; })
      .sort(function (a, b) { return plan.sections[b] - plan.sections[a]; });
    var other = plan.sections["*"];
    weights.forEach(function (item) {
      sections.appendChild(badge(plan.sections[item] > (other || 0) ? "solid" : null,
        (SECTION_TEXT[item] || "Item " + item) + " x" + plan.sections[item]));
      sections.appendChild(document.createTextNode(" "));
    });
    if (other !== undefined) {
      sections.appendChild(badge(null, "everything else x" + other));
    }
    if (plan.note_intent) {
      sections.appendChild(el("p", "bh-muted bh-small", "Note intent: " + plan.note_intent));
    }
    root.appendChild(sections);

    if (plan.notes && plan.notes.length) {
      var notes = el("ul", "fd-notes");
      plan.notes.forEach(function (note) { notes.appendChild(el("li", null, note)); });
      root.appendChild(notes);
    }
    if (plan.stale && plan.stale.length) {
      root.appendChild(callout("warn", "stale filer", plan.stale.join(", ") +
        ": the newest filing in the corpus is years old; figures are read as historical."));
    }
  }

  function companyCard(company, plan, payload) {
    var card = el("div", "fd-company");
    var head = el("div", "fd-company-head");
    head.appendChild(badge("solid", company.ticker));
    head.appendChild(el("span", "fd-company-name", company.name));
    if (plan.stale && plan.stale.indexOf(company.ticker) >= 0) {
      head.appendChild(badge("warn", "stale filer"));
    }
    if (company.covered === false) {
      head.appendChild(badge("warn", "period not covered"));
    }
    if (payload.status === "ok" && plan.companies.length > 1) {
      var remove = el("button", "bh-btn bh-btn--ghost fd-remove", "remove");
      remove.type = "button";
      remove.title = "Ask again without " + company.name;
      remove.addEventListener("click", function () {
        var next = withoutCompany(state.question, company, plan);
        $("question").value = next;
        askQuestion(next);
      });
      head.appendChild(remove);
    }
    card.appendChild(head);
    card.appendChild(el("div", "bh-muted bh-small", "matched \"" + company.matched_alias + "\"" +
      (company.available && company.available.length ? " | filings held: " + company.available.join(", ") : "")));

    var list = el("ul", "fd-filings");
    (plan.buckets || []).filter(function (b) { return b.ticker === company.ticker; }).forEach(function (bucket) {
      bucket.files.forEach(function (f) {
        var item = el("li");
        item.appendChild(el("strong", null, f.form + " " + f.fiscal_label));
        item.appendChild(el("span", null, (f.form === "10-K" ? "year ended " : "quarter ended ") + f.period_end +
          ", filed " + f.filing_date));
        item.appendChild(badge(null, REASON_TEXT[f.reason] || String(f.reason).replace(/_/g, " ")));
        list.appendChild(item);
      });
    });
    if (!list.firstChild) {
      list.appendChild(el("li", null, "no filing in scope"));
    }
    card.appendChild(list);
    return card;
  }

  // -- answer ---------------------------------------------------------------

  function flagsByClaim(payload) {
    var out = {};
    var checks = payload.checks;
    if (!checks || !checks.flags) { return out; }
    checks.flags.forEach(function (flag) {
      var key = flag.claim_id || "_";
      (out[key] = out[key] || []).push(flag);
    });
    return out;
  }

  function paintAnswer(payload) {
    var root = $("answer");
    clear(root);
    show("answer-band", true);
    var answer = payload.answer;
    if (payload.backend === "fake" && !payload.replayed) {
      root.appendChild(callout(null, "stand-in backend",
        "The fake backend answered; no model request was made. The plumbing below (claims, checks, sources) is real."));
    }
    if (!answer) {
      root.appendChild(callout("alert", "no answer parsed", payload.llm_error || "the reply did not parse"));
      if (payload.raw_text) { root.appendChild(preBlock(payload.raw_text.slice(0, 2000))); }
      return;
    }
    var flags = flagsByClaim(payload);
    var claimsById = {};
    answer.claims.forEach(function (c) { claimsById[c.id] = c; });

    var summary = el("div");
    answer.summary.forEach(function (sentence) {
      var p = el("p", "fd-sentence" + (sentence.claim_ids.length ? "" : " fd-dim"));
      p.appendChild(el("span", null, sentence.text));
      if (!sentence.claim_ids.length) {
        p.appendChild(badge("warn", "unlinked"));
      }
      sentence.claim_ids.forEach(function (id) {
        p.appendChild(claimChip(id, claimsById));
      });
      summary.appendChild(p);
    });
    root.appendChild(summary);

    if (answer.table && answer.table.length) {
      root.appendChild(answerTable(answer.table, claimsById));
    }
    if (answer.not_comparable && answer.not_comparable.length) {
      var nc = callout("warn", "not comparable");
      var list = el("ul");
      answer.not_comparable.forEach(function (entry) {
        list.appendChild(el("li", null, entry.dimension + " (" + entry.tickers.join(", ") + "): " + entry.reason));
      });
      nc.appendChild(list);
      root.appendChild(nc);
    }
    if (answer.gaps && answer.gaps.length) {
      var gaps = callout(null, "gaps");
      var glist = el("ul");
      answer.gaps.forEach(function (gap) { glist.appendChild(el("li", null, gap)); });
      gaps.appendChild(glist);
      root.appendChild(gaps);
    }

    var claimsHead = el("div");
    claimsHead.appendChild(el("span", "bh-eyebrow", "claims"));
    root.appendChild(claimsHead);
    var claims = el("ul", "fd-claims");
    answer.claims.forEach(function (claim) {
      claims.appendChild(claimItem(claim, flags[claim.id] || [], payload));
    });
    root.appendChild(claims);
  }

  function claimChip(id, claimsById) {
    var claim = claimsById[id];
    var target = claim && claim.citations.length ? claim.citations[0] : null;
    return chip(id, function () {
      var node = $("claim-" + id);
      if (target && $("src-" + target)) { goToSource(target); }
      else if (node) { node.scrollIntoView({behavior: REDUCED ? "auto" : "smooth", block: "center"}); }
    }, claim ? claim.text : "unknown claim " + id);
  }

  function answerTable(rows, claimsById) {
    var columns = [];
    rows.forEach(function (row) {
      row.cells.forEach(function (cell) {
        if (columns.indexOf(cell.column) < 0) { columns.push(cell.column); }
      });
    });
    var wrap = el("div", "fd-table-wrap");
    var table = el("table", "bh-table");
    var thead = el("thead");
    var tr = el("tr");
    var th0 = el("th", null, "dimension");
    th0.scope = "col";
    tr.appendChild(th0);
    columns.forEach(function (c) {
      var th = el("th", null, c);
      th.scope = "col";
      tr.appendChild(th);
    });
    thead.appendChild(tr);
    table.appendChild(thead);
    var tbody = el("tbody");
    rows.forEach(function (row) {
      var r = el("tr");
      var label = el("th", null, row.dimension);
      label.scope = "row";
      r.appendChild(label);
      columns.forEach(function (c) {
        var cell = null;
        row.cells.forEach(function (x) { if (x.column === c && !cell) { cell = x; } });
        var td = el("td", "bh-num");
        if (cell) {
          var figure = el("span", "fd-cell", cell.text);
          var first = cell.claim_ids.length ? claimsById[cell.claim_ids[0]] : null;
          if (first && first.citations.length) {
            figure.title = "go to source " + first.citations[0];
            figure.addEventListener("click", function () { goToSource(first.citations[0]); });
          }
          td.appendChild(figure);
          cell.claim_ids.forEach(function (id) { td.appendChild(claimChip(id, claimsById)); });
          if (!cell.claim_ids.length) { td.appendChild(badge("warn", "unlinked")); }
        }
        r.appendChild(td);
      });
      tbody.appendChild(r);
    });
    table.appendChild(tbody);
    wrap.appendChild(table);
    return wrap;
  }

  function claimItem(claim, flags, payload) {
    var item = el("li", "fd-claim");
    item.id = "claim-" + claim.id;
    var head = el("div", "fd-claim-head");
    head.appendChild(badge("solid", claim.id));
    if (claim.tickers.length) { head.appendChild(badge(null, claim.tickers.join(", "))); }
    head.appendChild(badge(null, "period " + (claim.period_end || "none") + " " + (claim.period_kind || "")));
    claim.citations.forEach(function (cid) {
      head.appendChild(chip(cid, function () { goToSource(cid); }, "go to source " + cid));
    });
    item.appendChild(head);
    item.appendChild(el("p", "fd-claim-text", claim.text));
    item.appendChild(el("p", "fd-claim-quote", "quote: " + claim.quote));
    var badges = el("div", "fd-badges");
    claimBadges(claim, flags, payload).forEach(function (b) { badges.appendChild(b); });
    item.appendChild(badges);
    return item;
  }

  function kinds(flags, kind) {
    return flags.filter(function (f) { return f.kind === kind; });
  }

  function claimBadges(claim, flags, payload) {
    var out = [];
    if (!payload.checks) { return out; }
    // Quote.
    if (!claim.citations.length) {
      out.push(badge("alert", "no citation given"));
    } else if (kinds(flags, "quote_not_found").length) {
      out.push(badge("alert", "quote not found in cited excerpt"));
    } else if (kinds(flags, "approximate_quote").length) {
      out.push(badge("warn", "approximate quote"));
    } else if (!kinds(flags, "citation_unknown").length) {
      out.push(badge(null, "quote found in source"));
    }
    kinds(flags, "citation_unknown").forEach(function (f) { out.push(badge("alert", f.detail)); });
    kinds(flags, "quote_too_long").forEach(function (f) { out.push(badge("warn", "quote too long: " + f.detail)); });
    // Figures.
    var figures = figuresIn(claim.text);
    var missing = kinds(flags, "figure_not_in_chunk");
    var outside = kinds(flags, "figure_not_in_quote");
    missing.forEach(function (f) { out.push(badge("alert", "figure not in cited excerpt: " + f.detail.split(" is in")[0])); });
    outside.forEach(function (f) { out.push(badge("warn", "figure in excerpt, outside the quote: " + f.source_string)); });
    if (figures.length && !missing.length && !outside.length && claim.citations.length) {
      out.push(badge(null, figures.length === 1 ? "figure in quote" : figures.length + " figures in quote"));
    }
    kinds(flags, "unit_converted").forEach(function (f) {
      out.push(badge("warn", "matched after unit conversion: " + f.source_string));
    });
    kinds(flags, "units_mismatch").forEach(function (f) { out.push(badge("warn", "units: " + f.detail)); });
    // Columns.
    var mismatches = kinds(flags, "column_mismatch").concat(kinds(flags, "duration_mismatch"));
    var unverified = kinds(flags, "column_unverified");
    mismatches.forEach(function (f) {
      out.push(badge("alert", "column mismatch: claim says " + claim.period_end + " " + (claim.period_kind || "") +
        ", figure sits in " + (f.source_string || "an undated column")));
    });
    unverified.forEach(function () { out.push(badge("warn", "column unverified")); });
    if (!mismatches.length && !unverified.length) {
      matchedColumns(claim, figures).forEach(function (m) {
        out.push(badge(null, "column: " + m.column.label));
      });
    }
    kinds(flags, "ticker_out_of_scope").forEach(function (f) { out.push(badge("warn", f.detail)); });
    return out;
  }

  // -- figures and columns (the page's own lookup, for the success badge
  //    and the source highlight; the server's flags win on any conflict) --

  function figuresIn(text) {
    var masked = text.replace(DATE_RE, " ");
    var out = [];
    var m;
    FIGURE_RE.lastIndex = 0;
    while ((m = FIGURE_RE.exec(masked)) !== null) {
      var digits = m[4];
      var bare = !m[2] && !m[6] && !m[7] && digits.indexOf(".") < 0 && digits.indexOf(",") < 0;
      if (bare && /^(19|20)\d{2}$/.test(digits)) { continue; }
      if (bare && parseInt(digits, 10) <= 12) { continue; }
      out.push({text: digits, value: parseFloat(digits.replace(/,/g, ""))});
    }
    return out;
  }

  function numberIn(text, value) {
    var m;
    NUMBER_RE.lastIndex = 0;
    while ((m = NUMBER_RE.exec(text)) !== null) {
      if (parseFloat(m[0].replace(/,/g, "")) === value) { return m[0]; }
    }
    return null;
  }

  function isNumberCell(cell) {
    return /^\$?\(?-?\d[\d,]*(\.\d+)?\)?%?$/.test(cell.trim());
  }

  function cellValue(cell) {
    var m = /^\$?\(?-?(\d[\d,]*(?:\.\d+)?)\)?%?$/.exec(cell.trim());
    return m ? parseFloat(m[1].replace(/,/g, "")) : null;
  }

  function tableRows(source) {
    // Each data row of a table chunk with the character span of every
    // cell, so a column can be underlined in place.
    var rows = [];
    var offset = 0;
    source.text.split("\n").forEach(function (line) {
      if (line.indexOf("|") >= 0) {
        var cells = [];
        var start = 0;
        line.split("|").forEach(function (raw) {
          cells.push({text: raw, start: offset + start, end: offset + start + raw.length});
          start += raw.length + 1;
        });
        var values = isNumberCell(cells[0].text) ? cells : cells.slice(1);
        rows.push({cells: cells, values: values});
      }
      offset += line.length + 1;
    });
    return rows;
  }

  function columnOfFigure(source, figure) {
    if (source.column_source !== "parsed" || !source.columns.length) { return null; }
    var rows = tableRows(source);
    for (var i = 0; i < rows.length; i++) {
      var values = rows[i].values;
      if (values.length !== source.columns.length) { continue; }
      var positions = [];
      values.forEach(function (cell, k) { if (cellValue(cell.text) === figure.value) { positions.push(k); } });
      if (positions.length === 1) {
        var column = null;
        source.columns.forEach(function (c) { if (c.index === positions[0]) { column = c; } });
        if (column) { return {column: column, position: positions[0]}; }
      }
    }
    return null;
  }

  function matchedColumns(claim, figures) {
    var out = [];
    var seen = {};
    figures.forEach(function (figure) {
      claim.citations.forEach(function (cid) {
        var source = state.sources[cid];
        if (!source || source.kind !== "table") { return; }
        if (numberIn(claim.quote, figure.value) === null && numberIn(source.text, figure.value) === null) { return; }
        var hit = columnOfFigure(source, figure);
        if (hit && !seen[cid + ":" + hit.position]) {
          seen[cid + ":" + hit.position] = true;
          out.push({cid: cid, column: hit.column, position: hit.position});
        }
      });
    });
    return out;
  }

  // -- sources --------------------------------------------------------------

  function indexSources(payload) {
    state.sources = {};
    (payload.sources || []).forEach(function (s) { state.sources[s.cid] = s; });
  }

  function highlightPlan(payload) {
    // Per cid: the quote ranges to fill and the column positions to underline.
    var plan = {};
    var answer = payload.answer;
    if (!answer) { return plan; }
    var flags = flagsByClaim(payload);
    answer.claims.forEach(function (claim) {
      var own = flags[claim.id] || [];
      var quoteCids = claim.citations.slice();
      own.forEach(function (f) { if (f.kind === "approximate_quote" && f.cid && quoteCids.indexOf(f.cid) < 0) { quoteCids.push(f.cid); } });
      quoteCids.forEach(function (cid) {
        var entry = plan[cid] = plan[cid] || {quotes: [], columns: []};
        if (claim.quote) { entry.quotes.push(claim.quote); }
      });
      matchedColumns(claim, figuresIn(claim.text)).forEach(function (m) {
        var entry = plan[m.cid] = plan[m.cid] || {quotes: [], columns: []};
        if (entry.columns.indexOf(m.position) < 0) { entry.columns.push(m.position); }
      });
      own.forEach(function (f) {
        if ((f.kind === "column_mismatch" || f.kind === "duration_mismatch") && f.cid && f.source_string) {
          var source = state.sources[f.cid];
          if (!source) { return; }
          source.columns.forEach(function (c) {
            if (c.label === f.source_string) {
              var entry = plan[f.cid] = plan[f.cid] || {quotes: [], columns: []};
              if (entry.columns.indexOf(c.index) < 0) { entry.columns.push(c.index); }
            }
          });
        }
      });
    });
    return plan;
  }

  function quoteRange(text, quote) {
    // Tokens of the quote separated by any run of whitespace or pipes, so a
    // quote taken across table cells or a line break still lands.
    var tokens = quote.split(/[\s|]+/).filter(Boolean).map(function (t) {
      return t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    });
    if (!tokens.length) { return null; }
    var re = new RegExp(tokens.join("[\\s|]+"), "i");
    var m = re.exec(text);
    return m ? {start: m.index, end: m.index + m[0].length} : null;
  }

  function renderHighlighted(source, entry) {
    var pre = el("pre", "bh-pre fd-source-text");
    var text = source.text;
    var ranges = [];
    if (entry) {
      entry.quotes.forEach(function (q) {
        var r = quoteRange(text, q);
        if (r) { ranges.push({start: r.start, end: r.end, cls: "fd-quote"}); }
      });
      if (entry.columns.length) {
        tableRows(source).forEach(function (row) {
          if (row.values.length !== source.columns.length) { return; }
          entry.columns.forEach(function (position) {
            var cell = row.values[position];
            if (cell && cell.end > cell.start) { ranges.push({start: cell.start, end: cell.end, cls: "fd-col"}); }
          });
        });
      }
    }
    if (!ranges.length) {
      pre.textContent = text;
      return pre;
    }
    var points = [0, text.length];
    ranges.forEach(function (r) { points.push(r.start); points.push(r.end); });
    points = points.filter(function (p, i, arr) { return arr.indexOf(p) === i; }).sort(function (a, b) { return a - b; });
    for (var i = 0; i < points.length - 1; i++) {
      var s = points[i], e = points[i + 1];
      var classes = [];
      ranges.forEach(function (r) {
        if (r.start <= s && r.end >= e && classes.indexOf(r.cls) < 0) { classes.push(r.cls); }
      });
      var piece = text.slice(s, e);
      if (classes.length) {
        var mark = el("mark", classes.join(" "), piece);
        pre.appendChild(mark);
      } else {
        pre.appendChild(document.createTextNode(piece));
      }
    }
    return pre;
  }

  function sectionName(source) {
    if (source.item === "COVER") { return "Cover"; }
    var name = source.form === "10-Q" ? "Part " + source.part + " Item " + source.item.split(".").pop() : "Item " + source.item;
    if (source.item_title) { name += " " + source.item_title; }
    if (source.note_title) { name += " > " + source.note_title; }
    return name;
  }

  function paintSources(payload) {
    var root = $("sources");
    clear(root);
    var sources = payload.sources || [];
    if (!sources.length) { show("sources-band", false); return; }
    show("sources-band", true);
    var plan = highlightPlan(payload);
    var order = (payload.plan.companies || []).map(function (c) { return c.ticker; });
    var groups = {};
    sources.forEach(function (s) { (groups[s.ticker] = groups[s.ticker] || []).push(s); });
    Object.keys(groups).sort(function (a, b) {
      var ia = order.indexOf(a), ib = order.indexOf(b);
      return (ia < 0 ? 99 : ia) - (ib < 0 ? 99 : ib);
    }).forEach(function (ticker) {
      var group = el("div", "fd-group");
      var title = el("h3", "bh-h3", groups[ticker][0].company + " ");
      title.appendChild(el("span", "bh-badge bh-badge--solid", ticker));
      group.appendChild(title);
      var cards = el("div", "fd-cards");
      groups[ticker].forEach(function (s) { cards.appendChild(sourceCard(s, plan[s.cid])); });
      group.appendChild(cards);
      root.appendChild(group);
    });
  }

  function sourceCard(source, entry) {
    var card = el("article", "bh-card");
    card.id = "src-" + source.cid;
    var head = el("div", "fd-card-head");
    head.appendChild(badge("solid", source.cid));
    head.appendChild(el("strong", null, source.form + " " + source.fiscal_label));
    head.appendChild(el("span", null, (source.form === "10-K" ? "year ended " : "quarter ended ") + source.period_end +
      ", filed " + source.filing_date));
    head.appendChild(badge(null, source.kind));
    var row = contextRow(source.cid);
    if (row && row.pinned) { head.appendChild(badge(null, "seated: " + row.pinned)); }
    if (source.url && source.url.indexOf("https://www.sec.gov/") === 0) {
      var link = el("a", "bh-link", "EDGAR filing");
      link.href = source.url;
      link.target = "_blank";
      link.rel = "noopener";
      head.appendChild(link);
    }
    card.appendChild(head);
    var meta = sectionName(source);
    if (source.units) { meta += " | units: " + source.units + (source.units_source === "declared" ? "" : " (" + source.units_source + ")"); }
    if (source.columns && source.columns.length) {
      meta += " | columns: " + source.columns.map(function (c) { return c.label; }).join(", ");
      if (source.column_source !== "parsed") { meta += " (column shape " + (source.column_source || "unknown") + ")"; }
    }
    meta += " | " + fmt(source.n_tokens) + " tokens";
    card.appendChild(el("div", "fd-card-meta", meta));
    card.appendChild(renderHighlighted(source, entry));
    if (entry && (entry.quotes.length || entry.columns.length)) {
      card.appendChild(el("div", "fd-legend",
        (entry.quotes.length ? "quoted passage highlighted" : "") +
        (entry.quotes.length && entry.columns.length ? "; " : "") +
        (entry.columns.length ? "matched column underlined" : "")));
    }
    return card;
  }

  function contextRow(cid) {
    var payload = state.full || state.dry;
    var rows = payload && payload.context ? payload.context : [];
    for (var i = 0; i < rows.length; i++) { if (rows[i].cid === cid) { return rows[i]; } }
    return null;
  }

  function goToSource(cid) {
    var card = $("src-" + cid);
    if (!card) { return; }
    card.scrollIntoView({behavior: REDUCED ? "auto" : "smooth", block: "center"});
    card.classList.remove("fd-flash");
    void card.offsetWidth;
    card.classList.add("fd-flash");
  }

  // -- details --------------------------------------------------------------

  function paintDetails(payload) {
    var root = $("details");
    clear(root);
    var plan = payload.plan;
    if (payload.status !== "ok") {
      root.appendChild(el("p", "bh-muted", "No context was assembled; the question was refused at the scope."));
      return;
    }

    root.appendChild(el("h3", "bh-h3", "context budget"));
    var budget = payload.budget || {};
    var meterLine = el("div", "fd-meter-line");
    var meter = el("div", "bh-meter");
    var fill = el("span");
    var share = budget.budget_tokens ? Math.min(100, 100 * budget.tokens_est / budget.budget_tokens) : 0;
    fill.style.width = share.toFixed(1) + "%";
    meter.appendChild(fill);
    meterLine.appendChild(meter);
    meterLine.appendChild(el("span", null, fmt(budget.tokens_est) + " of " + fmt(budget.budget_tokens) +
      " tokens estimated; " + fmt(budget.chunks_dropped) + " candidates dropped; provider input tokens: " +
      (budget.input_tokens_actual === undefined || budget.input_tokens_actual === null ? "none yet" : fmt(budget.input_tokens_actual))));
    root.appendChild(meterLine);
    root.appendChild(contextTable(payload.context || []));

    root.appendChild(el("h3", "bh-h3", "quotas and seats"));
    var quotas = el("ul");
    (plan.quotas || []).forEach(function (q) {
      quotas.appendChild(el("li", null, q.ticker + " / " + q.bucket + ": " + q.chunks + " chunks; section leads at positions " +
        JSON.stringify(q.pinned) + "; row-label seats " + JSON.stringify(q.row_label || [])));
    });
    root.appendChild(quotas);
    var pinned = (payload.context || []).filter(function (r) { return r.pinned; });
    root.appendChild(el("p", "bh-small", "Seated excerpts: " + (pinned.length ?
      pinned.map(function (r) { return r.cid + " (" + r.pinned + ")"; }).join(", ") : "none")));

    root.appendChild(el("h3", "bh-h3", "sub-queries"));
    var subs = el("ol");
    (plan.sub_queries || []).forEach(function (q) { subs.appendChild(el("li", null, q)); });
    root.appendChild(subs);

    root.appendChild(el("h3", "bh-h3", "evidence checks"));
    if (payload.checks) {
      var c = payload.checks;
      root.appendChild(el("p", "fd-stat", "quotes found " + c.quotes_found[0] + "/" + c.quotes_found[1] +
        " | figures in quote " + c.figures_in_quote[0] + "/" + c.figures_in_quote[1] +
        " | columns matched " + c.columns_matched[0] + "/" + c.columns_matched[1] + ", unverified " + c.columns_unverified +
        " | units declared " + c.units_declared[0] + "/" + c.units_declared[1] +
        " | unlinked sentences " + c.unlinked_sentences.length));
      if (c.flags.length) {
        var flags = el("ul");
        c.flags.forEach(function (f) {
          var li = el("li");
          li.appendChild(badge(f.kind.indexOf("not_found") >= 0 || f.kind.indexOf("not_in_chunk") >= 0 || f.kind.indexOf("mismatch") >= 0 ? "alert" : "warn",
            f.kind.replace(/_/g, " ")));
          li.appendChild(document.createTextNode(" " + (f.claim_id ? f.claim_id + ": " : "") + f.detail +
            (f.source_string ? " [source: " + f.source_string + "]" : "") + " "));
          if (f.cid) { li.appendChild(chip(f.cid, function () { goToSource(f.cid); })); }
          flags.appendChild(li);
        });
        root.appendChild(flags);
      } else {
        root.appendChild(el("p", "bh-small", "flags: none"));
      }
    } else {
      root.appendChild(el("p", "bh-muted bh-small", payload.dry_run ? "dry run: no answer to check yet" : "no answer to check"));
    }

    root.appendChild(el("h3", "bh-h3", "stat line"));
    var usage = payload.usage || {};
    var timing = payload.timing_ms || {};
    var latency = Object.keys(timing).map(function (k) { return k + " " + (timing[k] === null ? "n/a" : timing[k] + " ms"); }).join(", ");
    root.appendChild(el("p", "fd-stat",
      "requests attempted " + fmt(payload.llm_attempts) + " / completed " + fmt(payload.llm_completed) +
      " | input " + fmt(usage.input_tokens) + ", output " + fmt(usage.output_tokens) + " tokens" +
      (usage.cache_read_input_tokens ? ", cache read " + fmt(usage.cache_read_input_tokens) : "") +
      " | cost " + (payload.cost_usd === null || payload.cost_usd === undefined ? "n/a" : "$" + Number(payload.cost_usd).toFixed(4)) +
      " | latency: " + (latency || "n/a") +
      " | prompt " + (payload.prompt_version || "n/a") +
      " | backend " + (payload.backend || "none") + ", model " + (payload.model || "none") +
      (payload.stop_reason ? " | stop " + payload.stop_reason : "")));

    root.appendChild(el("h3", "bh-h3", "show the request"));
    var req = el("div", "fd-request");
    if (payload.prompt) {
      req.appendChild(el("p", "fd-stat", "request id " + (payload.request_id || "none") + " | attempts " + fmt(payload.llm_attempts) +
        " | completed " + fmt(payload.llm_completed) + " | max_retries " + fmt(payload.max_retries)));
      req.appendChild(el("span", "bh-eyebrow", "system"));
      req.appendChild(el("pre", "bh-pre", payload.prompt.system));
      req.appendChild(el("span", "bh-eyebrow", "user"));
      req.appendChild(el("pre", "bh-pre", payload.prompt.user));
    } else {
      req.appendChild(el("p", "bh-muted bh-small", "No request made yet. The excerpts block below is what the user turn will carry."));
      req.appendChild(el("span", "bh-eyebrow", "excerpts as rendered"));
      req.appendChild(el("pre", "bh-pre", payload.rendered || ""));
    }
    root.appendChild(req);
  }

  function contextTable(rows) {
    var wrap = el("div", "fd-table-wrap");
    var table = el("table", "bh-table");
    var head = el("tr");
    ["cid", "company", "filing", "section", "kind", "tokens", "seat"].forEach(function (h) {
      var th = el("th", null, h);
      th.scope = "col";
      head.appendChild(th);
    });
    var thead = el("thead");
    thead.appendChild(head);
    table.appendChild(thead);
    var body = el("tbody");
    rows.forEach(function (r) {
      var tr = el("tr");
      var cidCell = el("td");
      cidCell.appendChild(chip(r.cid, function () { goToSource(r.cid); }));
      tr.appendChild(cidCell);
      tr.appendChild(el("td", null, r.ticker));
      tr.appendChild(el("td", null, r.form + " " + r.fiscal_label));
      tr.appendChild(el("td", null, "Item " + r.item + (r.note_title ? " > " + r.note_title : "")));
      tr.appendChild(el("td", null, r.kind));
      tr.appendChild(el("td", "bh-num", fmt(r.n_tokens)));
      tr.appendChild(el("td", null, r.pinned || ""));
      body.appendChild(tr);
    });
    table.appendChild(body);
    wrap.appendChild(table);
    return wrap;
  }

  init();
})();
