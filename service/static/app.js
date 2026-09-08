/* financial facts: the page script. No framework, no build step.

   Flow per question: POST /ask with dry_run true and paint the filings read,
   the excerpts and the working papers (no model request, well under a
   second); then POST /ask with dry_run false and paint the brief and its
   evidence checks. A refusal stops after the first step. Every string that
   came from a filing or from the model lands in the DOM through textContent.

   Layout contract: nothing is appended bare. Every block goes through row(),
   which puts labels, numerals, codes and raw telemetry in a 176px margin and
   leaves the 640px measure carrying only prose.
*/
(function () {
  "use strict";

  // The four questions the build was tuned against. Clicking one fills the
  // question box; the panel edits it or types its own.
  var EXAMPLES = [
    {label: "What are the primary risk factors facing Apple, Tesla, and JPMorgan, and how do they compare?",
     text: "What are the primary risk factors facing Apple, Tesla, and JPMorgan, and how do they compare?"},
    {label: "How has NVIDIA's revenue and growth outlook changed over the last two years?",
     text: "How has NVIDIA's revenue and growth outlook changed over the last two years?"},
    {label: "What regulatory risks do the major pharmaceutical companies face, and how are they addressing them?",
     text: "What regulatory risks do the major pharmaceutical companies face, and how are they addressing them?"},
    {label: "Prepare a Q3 2025 bank disclosure brief comparing JPMorgan and Bank of America.",
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
  // The section name without its item number, for the sentence that says
  // what was read first; the numbers stay in the margin.
  var SECTION_PLAIN = {
    "1": "the business overview",
    "1A": "risk factors",
    "1C": "cybersecurity",
    "3": "legal proceedings",
    "7": "management's discussion",
    "8": "the financial statements",
    "I.1": "the quarterly financial statements",
    "I.2": "the quarterly management's discussion",
    "II.1": "quarterly legal proceedings",
    "II.1A": "quarterly risk factors"
  };
  // What a chosen SEC item actually covers, in a partner's words rather than
  // the item number; 10-Q items map to their 10-K counterpart. Read in a
  // fixed order so a two-topic headline always reads the same way (the
  // statements before the discussion, thereafter no bearing).
  var SECTION_TOPIC = {
    "1A": "risk factors", "II.1A": "risk factors",
    "7": "management's discussion", "I.2": "management's discussion",
    "8": "the financial statements", "I.1": "the financial statements",
    "1": "the business overview",
    "3": "legal proceedings", "II.1": "legal proceedings",
    "1C": "cybersecurity"
  };
  var TOPIC_ORDER = ["risk factors", "the financial statements", "management's discussion",
    "the business overview", "legal proceedings", "cybersecurity"];
  // What a claim's period_kind means in a sentence. The enum itself never
  // reaches the page.
  var PERIOD_KIND = {
    quarter: "quarter ended",
    nine_months: "nine months ended",
    fiscal_year: "year ended",
    point_in_time: "as of"
  };
  // The largest type on a refused screen states which of the three things
  // went wrong, rather than a single "not answered".
  var REFUSAL_TITLE = {
    not_covered: "not covered",
    period_not_covered: "period not held",
    needs_company: "no company named"
  };
  // A check kind, as a phrase. The raw enum is a server field name; printed
  // as-is it puts "unverified" and "chunk" on a client-facing badge.
  var KIND_TEXT = {
    quote_not_found: "quote not in the excerpt",
    approximate_quote: "quote approximate",
    citation_unknown: "citation not recognised",
    figure_not_in_chunk: "figure not in the excerpt",
    figure_elsewhere_in_chunk: "figure outside the quote",
    figures_unchecked: "figures not testable",
    sign_differs: "sign differs",
    bare_figure_from_percent_cell: "bare figure from a percent cell",
    unit_converted: "units converted",
    units_mismatch: "units do not match",
    units_unchecked: "units not testable",
    column_mismatch: "wrong column",
    duration_mismatch: "wrong period column",
    column_unverified: "column not confirmed",
    column_other_period: "column is another period",
    ticker_out_of_scope: "company outside the scope"
  };
  var REFUSAL_PLAIN = {
    not_covered: "the companies named are not among the 54 on file",
    period_not_covered: "the period asked for is not on file for these companies",
    needs_company: "the question named no company"
  };
  // Trimmed once, from the end: a company's own registered name minus its
  // corporate suffix, used only when several companies share one matched
  // alias (a group phrase such as "major pharmaceutical companies" cannot
  // itself head a company line).
  var LEGAL_SUFFIX_RE = /,?\s+(?:and\s+Company|Incorporated|Corporation|Corp\.?|Company|Inc\.?|&\s*Co\.?|Group|plc|Ltd\.?|LLC)\s*$/i;
  // A figure in claim text: optional $ and parenthesis, digits with commas
  // and decimals, optional % and scale word. Mirrors the server's check
  // closely enough to find the same tokens for the page's own lookup.
  var FIGURE_RE = /(^|[^A-Za-z\d.])(\$?)(\(?)(\d(?:[\d,]*\d)?(?:\.\d+)?)(\)?)(%?)(?:\s*(thousand|million|billion)s?\b)?/gi;
  var NUMBER_RE = /\d(?:[\d,]*\d)?(?:\.\d+)?/g;
  var DATE_RE = /\b\d{4}-\d{2}-\d{2}\b|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?\s+\d{1,2}(?:\s*,\s*\d{4})?\b/gi;
  var REDUCED = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  var state = {
    health: null,
    coverage: null,
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

  // The document grid: rail nodes to the margin, column nodes to the
  // measure. wide widens the second track over the empty third one, which
  // is what tables, the ledger and the request block use.
  function row(railNodes, colNodes, wide) {
    var g = el("div", "fd-grid");
    var rail = el("div", "fd-rail");
    (railNodes || []).forEach(function (n) { if (n) { rail.appendChild(n); } });
    var col = el("div", wide ? "fd-wide" : "fd-col");
    (colNodes || []).forEach(function (n) { if (n) { col.appendChild(n); } });
    g.appendChild(rail);
    g.appendChild(col);
    return g;
  }

  function label(text) { return el("span", "fd-label", text); }
  function record(text) { return el("div", "fd-record", text); }
  function h3(text) { return el("h3", "fd-h3", text); }
  function tickerBadge(text) { return el("span", "bh-badge bh-badge--solid", text); }
  function check(level, text) { return el("span", "fd-check fd-check--" + level, text); }

  function chip(text, onClick, title, ariaText) {
    var button = el("button", "fd-chip", text);
    button.type = "button";
    if (title) { button.title = title; }
    button.setAttribute("aria-label", ariaText || ("Go to excerpt " + text));
    button.addEventListener("click", onClick);
    return button;
  }

  function fmt(n) {
    if (n === null || n === undefined) { return "n/a"; }
    return Number(n).toLocaleString("en-US");
  }

  var MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

  // "2025-09-27" -> "27 Sep 2025". Parsed from the digits directly, never
  // through Date(), so no timezone can move the day. This is the only date
  // format on the page; a reader tracing a chip from a finding to its
  // excerpt must not see the same day written two ways.
  function fmtDateShort(iso) {
    var m = /^(\d{4})-(\d{2})-(\d{2})$/.exec(String(iso || ""));
    if (!m) { return iso; }
    return String(parseInt(m[3], 10)) + " " + MONTHS[parseInt(m[2], 10) - 1] + " " + m[1];
  }

  // The index build timestamp is a full ISO datetime; every other date on
  // the page is a day, so only the date part is kept and run through the
  // same short format.
  function fmtBuiltAt(iso) {
    var m = /^(\d{4}-\d{2}-\d{2})/.exec(String(iso || ""));
    return m ? fmtDateShort(m[1]) : iso;
  }

  function periodText(kind, end) {
    var word = PERIOD_KIND[kind] || "period ended";
    return end ? word + " " + fmtDateShort(end) : "";
  }

  function joinNames(names) {
    if (!names.length) { return ""; }
    if (names.length === 1) { return names[0]; }
    if (names.length === 2) { return names[0] + " and " + names[1]; }
    return names.slice(0, -1).join(", ") + " and " + names[names.length - 1];
  }

  function setStatus(text) { $("status").textContent = text || ""; }

  function show(id, visible) { $(id).hidden = !visible; }

  // -- startup --------------------------------------------------------------

  function init() {
    var examples = $("examples");
    EXAMPLES.forEach(function (q) {
      var item = el("li");
      var link = el("button", "fd-example");
      link.type = "button";
      // The label is wrapped so the hover underline lands on the text and
      // not on the row numeral drawn by the list counter.
      link.appendChild(el("span", null, q.label));
      link.addEventListener("click", function () {
        $("question").value = q.text;
        runAsk();
      });
      item.appendChild(link);
      examples.appendChild(item);
    });
    $("ask").addEventListener("click", runAsk);
    $("question").addEventListener("keydown", function (event) {
      if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
        event.preventDefault();
        runAsk();
      }
    });
    fetch("/health").then(function (r) { return r.json(); }).then(paintHealth).catch(function () {
      $("foot-index").textContent = "the index could not be reached";
    });
    // A refusal names the companies the index does hold. That list is read
    // from /coverage, which is structured, rather than from the coverage
    // block in the answer payload, which is prose written for the model.
    fetch("/coverage").then(function (r) { return r.json(); }).then(function (body) {
      state.coverage = body;
    }).catch(function () { state.coverage = null; });
  }

  function statTile(value, text) {
    var tile = el("div", "bh-stat");
    tile.appendChild(el("div", "bh-stat-value", value));
    tile.appendChild(el("div", "bh-stat-label", text));
    return tile;
  }

  function paintHealth(health) {
    state.health = health;
    show("key-banner", !health.llm_ready);
    if (health.index_error) {
      $("index-banner-text").textContent = health.index_error;
      show("index-banner", true);
      $("foot-index").textContent = "index missing; see the banner above";
    } else {
      $("foot-index").textContent = "Filed with the SEC, indexed " + fmtBuiltAt(health.index_built_at);
    }
    $("foot-model").textContent = health.backend === "fake" ?
      "Demonstration mode: no live model call" :
      "Model: " + health.model + (health.llm_ready ? "" : " (no key configured)");
    var stats = $("cover-stats");
    clear(stats);
    if (health.index_error) { return; }
    stats.appendChild(statTile(fmt(health.tickers), "US public companies"));
    stats.appendChild(statTile(fmt(health.files), "annual and quarterly filings"));
    stats.appendChild(statTile(fmt(health.chunks), "passages indexed"));
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
    state.dry = null;
    state.full = null;
    state.sources = {};
    stopTimer();
    resetBands();
    setStatus("Selecting filings");
    $("ask").disabled = true;

    post(question, true, signal).then(function (res) {
      if (run !== state.run) { return; }
      $("ask").disabled = false;
      if (!res.ok) {
        setStatus("");
        show("scope-band", true);
        $("scope").appendChild(errorRow(res.status, res.detail));
        return;
      }
      state.dry = res.body;
      indexSources(res.body);
      paintScope(res.body);
      paintSources(res.body);
      paintDetails(res.body);
      if (res.body.status !== "ok") {
        setStatus("Not answered; no model request was made");
        return;
      }
      show("answer-band", true);
      if (state.health && !state.health.llm_ready) {
        setStatus("Answer step disabled; filings and excerpts are shown");
        $("answer").appendChild(row([label("answer step disabled")],
          [el("p", "fd-note-read", "No API key is configured, so no model request was made. The filings read and the excerpts are shown.")]));
        return;
      }
      setStatus("Reading the excerpts");
      requestAnswer(question, run, signal);
    }).catch(function (err) {
      if (err && err.name === "AbortError") { return; }
      $("ask").disabled = false;
      setStatus("");
      show("scope-band", true);
      $("scope").appendChild(errorRow(0, String(err)));
    });
  }

  function requestAnswer(question, run, signal) {
    startTimer();
    clear($("answer"));
    post(question, false, signal).then(function (res) {
      if (run !== state.run) { return; }
      stopTimer();
      if (!res.ok) {
        setStatus(res.status === 501 ? "Answer step disabled; filings and excerpts are shown" :
          "Not answered; the model request did not complete");
        $("answer").appendChild(errorRow(res.status, res.detail));
        return;
      }
      state.full = res.body;
      indexSources(res.body);
      paintReplay(res.body);
      paintAnswer(res.body);
      paintSources(res.body);
      paintDetails(res.body);
      setStatus(res.body.answer ? "Brief ready" : "The model replied but the brief did not parse");
    }).catch(function (err) {
      if (err && err.name === "AbortError") { return; }
      stopTimer();
      setStatus("Not answered; the model request did not complete");
      $("answer").appendChild(errorRow(0, String(err)));
    });
  }

  // A failure in front of the room gets a sentence a reader can act on, and
  // the raw status and detail go to the margin so nothing is concealed.
  var ERROR_COPY = {
    "501": ["answer step not enabled",
      "This build does not have the answer step turned on. The filings read and the excerpts above are exactly what the model would receive."],
    "429": ["rate limited",
      "The model service is rate limiting this key. The filings and excerpts below stand; ask again in a moment."],
    "500": ["model unavailable",
      "The model service did not respond. The filings and excerpts below stand; ask again to make a new request."],
    "502": ["model unavailable",
      "The model service did not respond. The filings and excerpts below stand; ask again to make a new request."],
    "503": ["model unavailable",
      "The model service did not respond. The filings and excerpts below stand; ask again to make a new request."]
  };

  function errorRow(status, detail) {
    var copy = ERROR_COPY[String(status)] || ["request failed",
      "The request did not complete. The filings and excerpts below stand; ask again to make a new request."];
    var raw = (status ? "HTTP " + status + ": " : "") + (detail || "no detail returned");
    return row([label(copy[0]), record(raw)], [el("p", "fd-note-read", copy[1])]);
  }

  function resetBands() {
    ["scope", "answer", "sources", "details"].forEach(function (id) { clear($(id)); });
    ["scope-band", "answer-band", "sources-band", "method-band", "replay-banner"].forEach(function (id) { show(id, false); });
    $("progress").textContent = "";
    $("scope-title-text").textContent = "filings read";
  }

  function startTimer() {
    var started = Date.now();
    $("progress").textContent = "Reading. 0 s";
    state.timer = setInterval(function () {
      $("progress").textContent = "Reading. " + Math.round((Date.now() - started) / 1000) + " s";
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

  // -- filings read -----------------------------------------------------------

  // The latest period end held for each form, so a filing can be called the
  // newest only when it is.
  function newestPeriodByForm(files) {
    var out = {};
    (files || []).forEach(function (f) {
      if (!out[f.form] || f.period_end > out[f.form]) { out[f.form] = f.period_end; }
    });
    return out;
  }

  function bucketFilesFor(buckets, ticker) {
    var out = [];
    (buckets || []).forEach(function (b) {
      if (b.ticker === ticker) { out = out.concat(b.files); }
    });
    return out;
  }

  // The fiscal years named by "FYxxxx" bucket labels (the window/year period
  // modes); "latest" and "quarter" labels do not match and are handled by
  // their own branch of periodPhrase instead.
  function fiscalYearsOf(buckets) {
    var years = {};
    (buckets || []).forEach(function (b) {
      var m = /^FY(\d{4})/.exec(b.label);
      if (m) { years[m[1]] = true; }
    });
    return Object.keys(years).map(Number).sort(function (a, b) { return a - b; });
  }

  // "FY2024 Q2" sorts inside its own fiscal year, and the annual report sorts
  // last within that year because it covers all of it.
  function fiscalKey(fiscalLabel) {
    var m = /^FY(\d{4})(?:\s+Q(\d))?/.exec(String(fiscalLabel || ""));
    if (!m) { return 999999; }
    return parseInt(m[1], 10) * 10 + (m[2] ? parseInt(m[2], 10) : 4);
  }

  function sortedFiscal(labels) {
    return (labels || []).slice().sort(function (a, b) { return fiscalKey(a) - fiscalKey(b); });
  }

  // plan.notes are free text, each prefixed "TICKER: ...", written for a
  // reader already looking at the machinery. Grouped by ticker so a filing's
  // reason clause can quote the one that explains it.
  function groupNotesByTicker(notes) {
    var out = {};
    (notes || []).forEach(function (n) {
      var m = /^([A-Za-z.]+): /.exec(n);
      if (m) { (out[m[1]] = out[m[1]] || []).push(n); }
    });
    return out;
  }

  function findNote(notes, re) {
    for (var i = 0; i < notes.length; i++) { if (re.test(notes[i])) { return notes[i]; } }
    return null;
  }

  // The short name a headline uses for one company. A unique matched_alias
  // ("Apple", "Bank of America") is already the right length; a shared one
  // ("major pharmaceutical companies") names a whole group, not this one
  // company, so its own registered name is used instead, corporate suffix
  // trimmed off.
  function displayName(company, companies) {
    var shared = companies.filter(function (c) { return c.matched_alias === company.matched_alias; }).length > 1;
    if (!shared) { return company.matched_alias || company.name; }
    return company.name.replace(LEGAL_SUFFIX_RE, "");
  }

  function companyByTicker(companies, ticker) {
    var hit = (companies || []).filter(function (c) { return c.ticker === ticker; })[0];
    return hit || null;
  }

  // What will be read, for one company's chosen filings: which SEC items
  // carry the highest weight, translated out of item numbers. Ties keep
  // every topic that reached the top weight; TOPIC_ORDER fixes the order so
  // two topics always read the same way regardless of which fired first.
  function topicPhrase(sections) {
    if (!sections) { return ""; }
    var items = Object.keys(sections).filter(function (k) { return k !== "*"; });
    if (!items.length) { return ""; }
    var max = Math.max.apply(null, items.map(function (k) { return sections[k]; }));
    var present = {};
    items.forEach(function (k) {
      if (sections[k] === max && SECTION_TOPIC[k]) { present[SECTION_TOPIC[k]] = true; }
    });
    var ordered = TOPIC_ORDER.filter(function (t) { return present[t]; });
    Object.keys(present).forEach(function (t) { if (ordered.indexOf(t) < 0) { ordered.push(t); } });
    return joinNames(ordered);
  }

  // The period clause of the headline: what stands in for "now" for this
  // question, in the reader's words rather than the plan's period_mode code.
  function periodPhrase(plan) {
    var singular = plan.companies.length === 1;
    var buckets = plan.buckets || [];
    if (plan.period_mode === "latest") {
      // Grouped by which forms actually sit in each company's bucket, not
      // just whether a 10-K is present: a company whose newest 10-Q points
      // back to its 10-K (Apple's risk factors, say) carries both, and
      // saying only "annual report" for it would be wrong, not just vague.
      var onlyK = [], onlyQ = [], both = [];
      plan.companies.forEach(function (c) {
        var files = bucketFilesFor(buckets, c.ticker);
        if (!files.length) { return; }
        var hasK = files.some(function (f) { return f.form === "10-K"; });
        var hasQ = files.some(function (f) { return f.form === "10-Q"; });
        var name = displayName(c, plan.companies);
        if (hasK && hasQ) { both.push(name); }
        else if (hasK) { onlyK.push(name); }
        else if (hasQ) { onlyQ.push(name); }
      });
      var total = onlyK.length + onlyQ.length + both.length;
      if (!total) { return singular ? "its most recent filing" : "their most recent filings"; }
      if (onlyK.length === total) { return singular ? "its most recent annual report" : "their most recent annual reports"; }
      if (onlyQ.length === total) { return singular ? "its most recent quarterly report" : "their most recent quarterly reports"; }
      if (both.length === total) {
        return singular ? "its most recent annual report and most recent quarterly report" :
          "their most recent annual and quarterly reports";
      }
      // A mixed set: most companies read one form, a named few read both.
      // Naming them is the fix, not a vaguer clause that covers for them.
      if (both.length && !onlyQ.length) {
        return "their most recent annual reports, plus " + joinNames(both) +
          "'s most recent quarterly report" + (both.length > 1 ? "s" : "");
      }
      if (both.length && !onlyK.length) {
        return "their most recent quarterly reports, plus " + joinNames(both) +
          "'s most recent annual report" + (both.length > 1 ? "s" : "");
      }
      return singular ? "its most recent filings" : "their most recent filings";
    }
    if (plan.period_mode === "quarter") {
      var dates = {};
      buckets.forEach(function (b) { dates[b.period_end] = true; });
      var dateKeys = Object.keys(dates);
      var withBaseline = plan.companies.length > 0 && plan.companies.every(function (c) {
        return bucketFilesFor(buckets, c.ticker).some(function (f) { return f.reason === "annual_baseline"; });
      });
      var base = dateKeys.length === 1 ?
        (singular ? "its quarterly report for the quarter ended " + fmtDateShort(dateKeys[0])
                  : "their quarterly reports for the quarter ended " + fmtDateShort(dateKeys[0])) :
        (singular ? "its quarterly report" : "their quarterly reports");
      if (withBaseline) { base += ", plus " + (singular ? "its" : "each company's") + " most recent annual report"; }
      return base;
    }
    if (plan.period_mode === "window" || plan.period_mode === "year") {
      var years = fiscalYearsOf(buckets);
      if (!years.length) { return singular ? "its filings for the years asked" : "their filings for the years asked"; }
      if (years.length === 1) {
        return singular ? "its FY" + years[0] + " annual report" : "their FY" + years[0] + " annual reports";
      }
      var span = "FY" + years[0] + " through FY" + years[years.length - 1];
      return (singular ? "its filings for " : "their filings for ") + span;
    }
    return singular ? "its most recent filings" : "their most recent filings";
  }

  // A question that names a whole group ("the major pharmaceutical
  // companies") rather than individual companies resolves every hit to the
  // same matched_alias. Naming that phrase, before the names it resolved
  // to, is the clearest evidence the scope was decided on purpose rather
  // than dumped; null when the companies were named individually, or when
  // the shared alias is just one of their own names.
  function groupPhrase(companies) {
    if (companies.length < 2) { return null; }
    var alias = companies[0].matched_alias;
    if (!alias) { return null; }
    var shared = companies.every(function (c) { return c.matched_alias === alias; });
    if (!shared) { return null; }
    var isOwnName = companies.some(function (c) { return c.name.toLowerCase().indexOf(alias.toLowerCase()) === 0; });
    return isOwnName ? null : alias;
  }

  // The sentence a partner reads without opening the working papers: which
  // companies, what stands in for "now", why those sections.
  function headlineSentence(plan) {
    if (!plan.companies.length) { return ""; }
    var names = plan.companies.map(function (c) { return displayName(c, plan.companies); });
    var topic = topicPhrase(plan.sections);
    var group = groupPhrase(plan.companies);
    var subject = group ?
      (/^the\s/i.test(group) ? group : "the " + group) + " (" + joinNames(names) + ")" :
      joinNames(names);
    return "Reading " + subject + ": " + periodPhrase(plan) +
      (topic ? ", focused mainly on " + topic : "") + ".";
  }

  // The reason a chosen filing is in scope, in a clause rather than a code.
  // annual_baseline and comparative_columns can mean one of several things
  // depending on which rule added the filing; plan.notes says which, and the
  // clause quotes that reason back in plain words. Falls back to the short
  // REASON_TEXT label, then to the raw code, if no note matches.
  function reasonClause(file, ticker, notesByTicker, newest) {
    var notes = notesByTicker[ticker] || [];
    // A multi-year question puts every filing of the window in the bucket,
    // and each one carries the reason its bucket was selected. Only one of
    // them is actually the newest, so the rest say nothing here: the form,
    // the fiscal label and the period end above already state what they are,
    // and the headline sentence names the window they sit in.
    if (file.reason === "newest_10k") {
      return newest["10-K"] === file.period_end ? "newest annual report" : "";
    }
    if (file.reason === "newest_10q") {
      return newest["10-Q"] === file.period_end ? "newest quarterly report" : "";
    }
    if (file.reason === "annual_baseline") {
      if (findNote(notes, /points to the annual report/)) {
        return "included because its risk-factor section points back to the annual report";
      }
      if (findNote(notes, /Item [\dA-Z.]+ is absent/)) {
        return "included because the quarterly filing carries no such section of its own";
      }
      if (findNote(notes, /is the annual baseline/)) {
        return "no annual report on file predates this quarter; the nearest one available is included";
      }
      return "most recent annual report, included for annual-period figures";
    }
    if (file.reason === "comparative_columns") {
      if (findNote(notes, /as a prior-year column/)) {
        return "not filed on its own; the year asked shows only as a prior-year column here";
      }
      return "included for comparison";
    }
    return REASON_TEXT[file.reason] || String(file.reason || "").replace(/_/g, " ");
  }

  // When two or more companies' latest annual report sits in a different
  // fiscal year, a side-by-side reading assumes they line up; this says when
  // they do not. Checked only for a "now" snapshot (latest or quarter mode):
  // a window/year question already states its span in the headline, and a
  // single company has nothing to be paired against.
  function annualAnchorNote(plan) {
    if (plan.companies.length < 2) { return null; }
    if (plan.period_mode !== "latest" && plan.period_mode !== "quarter") { return null; }
    var buckets = plan.buckets || [];
    var byYear = {};
    plan.companies.forEach(function (c) {
      var k = bucketFilesFor(buckets, c.ticker).filter(function (f) { return f.form === "10-K"; })[0];
      if (!k) { return; }
      (byYear[k.fiscal_label] = byYear[k.fiscal_label] || []).push(displayName(c, plan.companies));
    });
    var years = Object.keys(byYear).sort();
    if (years.length < 2) { return null; }
    var parts = years.map(function (fy) {
      var names = byYear[fy];
      return joinNames(names) + (names.length === 1 ? " reports " : " report ") + fy;
    });
    return "The companies' latest annual reports cover different fiscal years: " + parts.join("; ") +
      ". Annual figures are not directly comparable.";
  }

  // plan.not_covered lines are written for the model ("AAPL: no filings for
  // fiscal 2019"); this reads the ticker back out and says it in the same
  // voice as the rest of the section.
  function friendlyNotCovered(entry, companies) {
    var m = /^([A-Za-z.]+): no filings for (.+)$/.exec(entry);
    if (!m) { return entry; }
    var company = companyByTicker(companies, m[1]);
    var name = company ? displayName(company, companies) : m[1];
    return name + " has no filings on file for " + m[2] + ".";
  }

  // Each break in comparability as its own line above the ledger. A method
  // note about excerpt selection is not a peer of these and is printed
  // separately, below.
  function cautions(payload) {
    var plan = payload.plan;
    var out = [];
    var anchor = annualAnchorNote(plan);
    if (anchor) { out.push(anchor); }
    if (plan.unresolved && plan.unresolved.length) {
      out.push((plan.unresolved.length === 1 ? plan.unresolved[0] + " is" : joinNames(plan.unresolved) + " are") +
        " not among the 54 companies on file.");
    }
    if (plan.not_covered && plan.not_covered.length) {
      plan.not_covered.forEach(function (e) { out.push(friendlyNotCovered(e, plan.companies)); });
    }
    var notesByTicker = groupNotesByTicker(plan.notes);
    (plan.stale || []).forEach(function (ticker) {
      var company = companyByTicker(plan.companies, ticker);
      var name = company ? displayName(company, plan.companies) : ticker;
      var note = findNote(notesByTicker[ticker] || [], /newest filing is/);
      var m = note && /newest filing is ([^;]+); treat it as stale/.exec(note);
      out.push(name + "'s newest filing on file is from " + (m ? fmtDateShort(m[1]) : "an earlier period") +
        "; read the figures as historical.");
    });
    return out.map(function (text) { return el("p", "fd-caution", text); });
  }

  function methodNote(payload) {
    var count = (payload.sources || []).length;
    if (!count) { return ""; }
    return fmt(count) + " excerpts were selected from these filings, not the full documents.";
  }

  function paintScope(payload) {
    var plan = payload.plan;
    var root = $("scope");
    clear(root);
    show("scope-band", true);

    if (payload.status !== "ok") {
      $("scope-title-text").textContent = REFUSAL_TITLE[payload.status] || "not answered";
      root.appendChild(row([label("cannot answer")], [
        el("p", "fd-refusal", refusalStatement(payload)),
        el("p", "fd-refusal-detail", refusalDetail(payload))
      ]));
      coverageRows(plan).forEach(function (node) { root.appendChild(node); });
      return;
    }

    $("scope-title-text").textContent = "filings read";
    root.appendChild(row([], [el("p", "fd-standfirst", headlineSentence(plan))]));
    cautions(payload).forEach(function (node) {
      root.appendChild(row([label("not directly comparable")], [node]));
    });

    var notesByTicker = groupNotesByTicker(plan.notes);
    var ledger = el("div", "fd-ledger");
    plan.companies.forEach(function (company) {
      ledger.appendChild(companyRow(company, plan, notesByTicker));
    });
    root.appendChild(ledger);

    var note = methodNote(payload);
    if (note) { root.appendChild(row([], [el("p", "fd-note", note)])); }
  }

  function refusalStatement(payload) {
    var plan = payload.plan;
    if (payload.status === "not_covered") {
      var names = plan.unresolved || [];
      if (!names.length) { return "None of the companies named are among the 54 on file."; }
      return names.length === 1 ?
        names[0] + " is not one of the 54 companies on file." :
        joinNames(names) + " are not among the 54 companies on file.";
    }
    if (payload.status === "period_not_covered") {
      var named = (plan.companies || []).map(function (c) { return displayName(c, plan.companies); });
      return (named.length ? joinNames(named) : "The company named") +
        (named.length === 1 ? " has" : " have") + " no filings on file for the period asked.";
    }
    return "No company was named. Name at least one of the 54 companies on file.";
  }

  function refusalDetail(payload) {
    if (payload.status === "period_not_covered") {
      return "Nothing was retrieved and no model request was made. The periods on file for the companies you named are listed below.";
    }
    return "Nothing was retrieved and no model request was made. The companies on file are listed below.";
  }

  // What the index does hold, for a question it cannot answer. A company
  // that was recognised but asked about outside its filing range gets its
  // own periods; anything else gets the 54 companies, always open.
  function coverageRows(plan) {
    var named = (plan.companies || []).filter(function (c) { return c.available && c.available.length; });
    if (named.length) {
      return named.map(function (c) {
        var col = [];
        col.push(el("p", "fd-entity", c.name));
        var labels = sortedFiscal(c.available);
        col.push(el("p", "fd-note-read", "Filings on file: " + labels[0] + " through " + labels[labels.length - 1] + "."));
        var strip = el("div", "fd-periods");
        labels.forEach(function (text) { strip.appendChild(check("ok", text)); });
        col.push(strip);
        return row([tickerBadge(c.ticker)], col);
      });
    }
    var companies = (state.coverage && state.coverage.companies) || [];
    if (!companies.length) {
      return [row([label("companies on file")],
        [el("p", "fd-note-read", "The company list did not load. It is served at /coverage.")])];
    }
    var grid = el("div", "fd-coverage");
    companies.forEach(function (c) {
      var cell = el("div", "fd-coverage-cell");
      var line = el("div", "fd-coverage-name");
      line.appendChild(el("strong", null, c.ticker));
      line.appendChild(document.createTextNode(" " + c.name));
      cell.appendChild(line);
      cell.appendChild(el("div", "fd-coverage-span", filingSpan(c.filings)));
      grid.appendChild(cell);
    });
    return [row([label(companies.length + " companies on file")], [grid], true)];
  }

  // "FY2022 Q2 to FY2026 Q1", or the single label when only one filing is
  // held. Ordered by period end, which is what the fiscal label is derived
  // from, so the ends of the range are the oldest and newest filings.
  function filingSpan(filings) {
    var sorted = (filings || []).slice().sort(function (a, b) {
      return a.period_end < b.period_end ? -1 : a.period_end > b.period_end ? 1 : 0;
    });
    if (!sorted.length) { return ""; }
    var first = sorted[0].fiscal_label;
    var last = sorted[sorted.length - 1].fiscal_label;
    return first === last ? first : first + " to " + last;
  }

  function companyRow(company, plan, notesByTicker) {
    var wrap = el("div", "fd-company");
    var rail = el("div", "fd-company-rail");
    rail.appendChild(tickerBadge(company.ticker));
    if (plan.stale && plan.stale.indexOf(company.ticker) >= 0) {
      rail.appendChild(check("note", "stale filer"));
    }
    if (company.covered === false) {
      rail.appendChild(check("note", "period not covered"));
    }
    wrap.appendChild(rail);

    var body = el("div");
    body.appendChild(el("h3", "fd-entity", company.name));

    var companyBuckets = (plan.buckets || []).filter(function (b) { return b.ticker === company.ticker; });
    var allFiles = bucketFilesFor(plan.buckets, company.ticker);
    var newest = newestPeriodByForm(allFiles);

    if (companyBuckets.length > 1) {
      // A multi-bucket trend (a quarterly or multi-year question): one
      // fiscal-year group per bucket, so filings read in order within each
      // year instead of a flattened list that zigzags across years.
      companyBuckets.forEach(function (b) {
        var group = el("div", "fd-bucket-group");
        group.appendChild(el("span", "fd-label fd-label--muted", b.label));
        group.appendChild(filingList(b.files, company.ticker, notesByTicker, newest));
        body.appendChild(group);
      });
    } else {
      body.appendChild(filingList(allFiles, company.ticker, notesByTicker, newest));
    }
    wrap.appendChild(body);
    return wrap;
  }

  function filingList(files, ticker, notesByTicker, newest) {
    var list = el("ul", "fd-filings");
    files.forEach(function (f) {
      var item = el("li", "fd-filing-row");
      var text = el("div", "fd-filing-label");
      text.appendChild(el("strong", null, f.form + " " + f.fiscal_label));
      text.appendChild(document.createTextNode(" " +
        (f.form === "10-K" ? "year ended " : "quarter ended ") + fmtDateShort(f.period_end)));
      item.appendChild(text);
      item.appendChild(el("div", "fd-filing-reason", reasonClause(f, ticker, notesByTicker, newest)));
      list.appendChild(item);
    });
    if (!list.firstChild) {
      var empty = el("li", "fd-filing-row");
      empty.appendChild(el("div", "fd-filing-label", "no filing in scope"));
      list.appendChild(empty);
    }
    return list;
  }

  // -- the brief ---------------------------------------------------------------

  // Flags are disagreements and notes are the facts beside them (which
  // column a comparative figure sits in, why a scale was not read); a claim
  // badge has to show both, so they are grouped together by "where".
  function rowsByClaim(payload) {
    var out = {};
    var checks = payload.checks;
    if (!checks) { return out; }
    (checks.flags || []).concat(checks.notes || []).forEach(function (r) {
      var key = r.where || "_";
      (out[key] = out[key] || []).push(r);
    });
    return out;
  }

  // Every number in the brief renders 600-weight with tabular figures inside
  // 400-weight prose, so a CFO's eye lands on findings and not on words.
  // Dates are masked to spaces of the same length first, so a match index
  // taken from the masked copy still points at the right character of the
  // original.
  function emphasizeFigures(parent, text) {
    var src = String(text === null || text === undefined ? "" : text);
    var masked = src.replace(DATE_RE, function (d) { return new Array(d.length + 1).join(" "); });
    var last = 0;
    var m;
    FIGURE_RE.lastIndex = 0;
    while ((m = FIGURE_RE.exec(masked)) !== null) {
      var digits = m[4];
      var bare = !m[2] && !m[6] && !m[7] && digits.indexOf(".") < 0 && digits.indexOf(",") < 0;
      // A bare year or a small bare count is not a reported figure.
      if (bare && (/^(19|20)\d{2}$/.test(digits) || parseInt(digits, 10) <= 12)) { continue; }
      var start = m.index + m[1].length;
      var end = m.index + m[0].length;
      if (start > last) { parent.appendChild(document.createTextNode(src.slice(last, start))); }
      parent.appendChild(el("span", "fd-fig", src.slice(start, end)));
      last = end;
    }
    if (last < src.length) { parent.appendChild(document.createTextNode(src.slice(last))); }
  }

  function paintAnswer(payload) {
    var root = $("answer");
    clear(root);
    show("answer-band", true);
    var answer = payload.answer;

    if (payload.backend === "fake" && !payload.replayed) {
      root.appendChild(row([label("demonstration mode")], [el("p", "fd-note-read",
        "No model was called for this answer. The filings read, the excerpts, the citations and the checks are produced by the same code path as a live run.")]));
    }
    if (!answer) {
      root.appendChild(row([label("the brief did not parse"), record(payload.llm_error || "no parser detail")],
        [el("p", "fd-note-read", "The model replied, but the reply did not arrive in the shape this page reads. The reply as received is in the margin below; the filings and excerpts stand.")]));
      if (payload.raw_text) {
        root.appendChild(row([label("the reply as received")], [el("pre", "bh-pre", payload.raw_text.slice(0, 2000))], true));
      }
      return;
    }

    var flags = rowsByClaim(payload);
    var claimsById = {};
    answer.claims.forEach(function (c) { claimsById[c.id] = c; });

    var summaryNodes = [];
    answer.summary.forEach(function (sentence) {
      var p = el("p", "fd-sentence" + (sentence.claim_ids.length ? "" : " fd-dim"));
      emphasizeFigures(p, sentence.text);
      if (!sentence.claim_ids.length) {
        p.appendChild(document.createTextNode(" "));
        p.appendChild(check("flag", "no source cited for this sentence"));
      }
      sentence.claim_ids.forEach(function (id) {
        p.appendChild(document.createTextNode(" "));
        p.appendChild(claimChip(id, claimsById));
      });
      summaryNodes.push(p);
    });
    root.appendChild(row([label("summary")], summaryNodes));
    // The only thing on the page that says the chips navigate. It is text,
    // not a control, so it costs nothing against the two-control rule.
    root.appendChild(row([], [el("p", "fd-note",
      "Navy tags are links. Clicking one moves the page to the exact excerpt the figure came from.")]));

    if (answer.table && answer.table.length) {
      root.appendChild(row([label("comparison")], answerTable(answer.table, claimsById, payload.plan), true));
    }
    if (answer.not_comparable && answer.not_comparable.length) {
      var list = el("ul", "fd-list");
      answer.not_comparable.forEach(function (entry) {
        list.appendChild(el("li", null, entry.dimension + " (" + entry.tickers.join(", ") + "): " + entry.reason));
      });
      root.appendChild(row([label("not comparable")], [list]));
    }
    if (answer.gaps && answer.gaps.length) {
      var glist = el("ul", "fd-list");
      answer.gaps.forEach(function (gap) { glist.appendChild(el("li", null, gap)); });
      root.appendChild(row([label("gaps")], [glist]));
    }

    // The heading stays an h3 for the document outline and is set at rail
    // scale, so no heading on the page outranks its own section title.
    root.appendChild(row([h3("findings")], []));
    var claims = el("ul", "fd-claims");
    answer.claims.forEach(function (claim) {
      claims.appendChild(claimItem(claim, flags[claim.id] || [], payload));
    });
    root.appendChild(claims);
  }

  function claimChip(id, claimsById) {
    var claim = claimsById[id];
    var target = claim && claim.citations.length ? claim.citations[0] : null;
    var aria = target ? "Go to excerpt " + target + ", the source for finding " + id :
      "Go to finding " + id;
    return chip(id, function () {
      var node = $("claim-" + id);
      if (target && $("src-" + target)) { goToSource(target); }
      else if (node) { node.scrollIntoView({behavior: REDUCED ? "auto" : "smooth", block: "start"}); }
    }, claim ? claim.text : "unknown finding " + id, aria);
  }

  // Every claim behind one table row, resolved through the claim index.
  function claimsOfRow(tableRow, claimsById) {
    var out = [], seen = {};
    (tableRow.cells || []).forEach(function (cell) {
      (cell.claim_ids || []).forEach(function (id) {
        if (seen[id]) { return; }
        seen[id] = true;
        if (claimsById[id]) { out.push(claimsById[id]); }
      });
    });
    return out;
  }

  function periodsOf(claims) {
    var seen = {}, out = [];
    claims.forEach(function (c) {
      var text = periodText(c.period_kind, c.period_end);
      if (!text) { text = "period not stated"; }
      if (!seen[text]) { seen[text] = true; out.push(text); }
    });
    return out;
  }

  // The units every cited excerpt behind a set of claims declares. A null
  // anywhere is a disagreement: a figure without units is not a figure.
  function unitsOf(claims) {
    var seen = {}, out = [], inferred = false, missing = false;
    claims.forEach(function (c) {
      (c.citations || []).forEach(function (cid) {
        var source = state.sources[cid];
        if (!source || !source.units) { missing = true; return; }
        if (source.units_source !== "declared") { inferred = true; }
        if (!seen[source.units]) { seen[source.units] = true; out.push(source.units); }
      });
    });
    return {units: out, inferred: inferred, missing: missing};
  }

  function answerTable(rows, claimsById, plan) {
    var columns = [];
    rows.forEach(function (r) {
      r.cells.forEach(function (cell) {
        if (columns.indexOf(cell.column) < 0) { columns.push(cell.column); }
      });
    });
    var wrap = el("div", "fd-table-wrap");
    var table = el("table", "bh-table fd-table");
    var thead = el("thead");
    var tr = el("tr");
    var th0 = el("th", null, "line item");
    th0.scope = "col";
    tr.appendChild(th0);
    columns.forEach(function (c) {
      var th = el("th", null, c);
      th.scope = "col";
      var company = companyByTicker(plan && plan.companies, c);
      if (company) {
        th.appendChild(el("span", "fd-th-sub", displayName(company, plan.companies)));
      }
      tr.appendChild(th);
    });
    thead.appendChild(tr);
    table.appendChild(thead);

    var tbody = el("tbody");
    rows.forEach(function (r) {
      var claims = claimsOfRow(r, claimsById);
      var periods = periodsOf(claims);
      var units = unitsOf(claims);
      var unitsOk = !units.missing && units.units.length === 1;
      var broken = periods.length > 1 || (claims.length > 0 && !unitsOk);

      var line = el("tr");
      var head = el("th", null, r.dimension);
      head.scope = "row";
      if (broken) { head.className = "fd-row-break"; }
      if (claims.length) {
        if (periods.length === 1) {
          head.appendChild(el("span", "fd-row-meta",
            periods[0] + (unitsOk ? " . " + units.units[0] + (units.inferred ? " (inferred)" : "") : "")));
        } else {
          head.appendChild(el("span", "fd-row-meta fd-row-meta--break",
            "periods differ: " + joinNames(periods)));
        }
        if (!unitsOk) {
          head.appendChild(el("span", "fd-row-meta fd-row-meta--break",
            "units not stated in the cited excerpts"));
        }
      }
      line.appendChild(head);

      columns.forEach(function (c) {
        var cell = null;
        r.cells.forEach(function (x) { if (x.column === c && !cell) { cell = x; } });
        var td = el("td", "bh-num");
        if (cell) {
          var figure = el("span");
          emphasizeFigures(figure, cell.text);
          td.appendChild(figure);
          (cell.claim_ids || []).forEach(function (id) { td.appendChild(claimChip(id, claimsById)); });
          if (!cell.claim_ids || !cell.claim_ids.length) {
            td.appendChild(check("flag", "no source cited"));
          } else if (broken) {
            // On a broken row each cell states its own period and units, so
            // the reader can see which company is the odd one out.
            var own = (cell.claim_ids || []).map(function (id) { return claimsById[id]; })
              .filter(function (x) { return !!x; });
            var ownPeriods = periodsOf(own);
            var ownUnits = unitsOf(own);
            td.appendChild(el("span", "fd-cell-meta", ownPeriods.join("; ") +
              (ownUnits.units.length === 1 && !ownUnits.missing ?
                " . " + ownUnits.units[0] + (ownUnits.inferred ? " (inferred)" : "") :
                " . units not stated")));
          }
        } else {
          td.appendChild(check("flag", "no source cited"));
        }
        line.appendChild(td);
      });
      tbody.appendChild(line);
    });
    table.appendChild(tbody);
    wrap.appendChild(table);
    return [wrap, el("p", "fd-scroll-note", "This table is wider than the screen. Scroll it sideways.")];
  }

  // Reader order: what is claimed, for what period, in whose words, then
  // what the checks found. The internal row id sits in the margin.
  function claimItem(claim, flags, payload) {
    var item = el("li", "fd-claim");
    item.id = "claim-" + claim.id;

    var rail = el("div", "fd-rail");
    rail.appendChild(record(claim.id));
    var col = el("div");

    var text = el("p", "fd-claim-text");
    emphasizeFigures(text, claim.text);
    col.appendChild(text);

    var period = periodText(claim.period_kind, claim.period_end);
    if (!period) {
      period = "period not stated by the model";
      item.className = "fd-claim fd-claim--break";
    }
    col.appendChild(el("p", "fd-claim-meta",
      (claim.tickers.length ? claim.tickers.join(", ") + " . " : "") + period));

    if (claim.quote) {
      var quote = el("blockquote", "fd-claim-quote", "\"" + claim.quote + "\"");
      claim.citations.forEach(function (cid) {
        quote.appendChild(document.createTextNode(" "));
        quote.appendChild(chip(cid, function () { goToSource(cid); },
          "excerpt " + cid, "Go to excerpt " + cid + ", the source for finding " + claim.id));
      });
      col.appendChild(quote);
    }

    var badges = el("div", "fd-badges");
    claimBadges(claim, flags, payload).forEach(function (b) { badges.appendChild(b); });
    col.appendChild(badges);

    item.appendChild(rail);
    item.appendChild(col);
    return item;
  }

  function kinds(flags, kind) {
    return flags.filter(function (f) { return f.kind === kind; });
  }

  // Every badge says which of three outcomes a check reached: matched, a
  // note beside it, or a flag. The flag is the loudest thing in the block,
  // which is the opposite of the previous ordering. A server message is
  // printed whole after a fixed prefix: splitting it on a literal substring
  // dumps raw text into a client-facing badge the moment it is reworded.
  function claimBadges(claim, flags, payload) {
    var out = [];
    if (!payload.checks) { return out; }
    // Quote.
    if (!claim.citations.length) {
      out.push(check("flag", "no source cited for this figure"));
    } else if (kinds(flags, "quote_not_found").length) {
      out.push(check("flag", "the quoted words are not in the cited excerpt"));
    } else if (kinds(flags, "approximate_quote").length) {
      out.push(check("note", "the quote matches the excerpt only approximately"));
    } else if (!kinds(flags, "citation_unknown").length) {
      out.push(check("ok", "quote found in the cited excerpt"));
    }
    kinds(flags, "citation_unknown").forEach(function (f) { out.push(check("flag", f.detail)); });
    // Figures.
    var figures = figuresIn(claim.text);
    var missing = kinds(flags, "figure_not_in_chunk");
    var outside = kinds(flags, "figure_elsewhere_in_chunk");
    var unchecked = kinds(flags, "figures_unchecked");
    missing.forEach(function (f) {
      out.push(check("flag", "this figure is not printed in the cited excerpt: " + f.detail));
    });
    outside.forEach(function (f) {
      out.push(check("note", "the figure is in the excerpt but outside the quoted words: " + f.detail));
    });
    unchecked.forEach(function (f) { out.push(check("note", "figures could not be checked: " + f.detail)); });
    if (figures.length && !missing.length && !outside.length && !unchecked.length && claim.citations.length) {
      out.push(check("ok", figures.length === 1 ? "the figure is printed in the quote" :
        figures.length + " figures are printed in the quote"));
    }
    kinds(flags, "sign_differs").forEach(function (f) {
      out.push(check("flag", "the sign differs from the filing: " + f.detail));
    });
    kinds(flags, "bare_figure_from_percent_cell").forEach(function (f) { out.push(check("note", f.detail)); });
    // Units.
    kinds(flags, "unit_converted").forEach(function (f) {
      out.push(check("ok", "units match after conversion: " + f.detail));
    });
    kinds(flags, "units_mismatch").forEach(function (f) {
      out.push(check("flag", "units do not match the filing: " + f.detail));
    });
    kinds(flags, "units_unchecked").forEach(function (f) {
      out.push(check("note", "units could not be checked: " + f.detail));
    });
    // Columns.
    var mismatches = kinds(flags, "column_mismatch").concat(kinds(flags, "duration_mismatch"));
    var unverified = kinds(flags, "column_unverified");
    mismatches.forEach(function (f) {
      out.push(check("flag", "the claim says " + (periodText(claim.period_kind, claim.period_end) || "no period") +
        ", the figure sits in the " + (f.source_string || "undated") + " column"));
    });
    unverified.forEach(function () {
      out.push(check("note", "the column this figure sits in could not be confirmed"));
    });
    kinds(flags, "column_other_period").forEach(function (f) { out.push(check("ok", f.detail)); });
    // The page's own column lookup fills in only where the server ran the
    // column check and said nothing against it; a figure the server could not
    // place must never show a column badge.
    if (!mismatches.length && !unverified.length && !missing.length && !outside.length && !unchecked.length) {
      matchedColumns(claim, figures).forEach(function (m) {
        out.push(check("ok", "read from the " + m.column.label + " column"));
      });
    }
    kinds(flags, "ticker_out_of_scope").forEach(function (f) { out.push(check("note", f.detail)); });
    return out;
  }

  // -- figures and columns (the page's own lookup, for the success badge
  //    and the source highlight; the server's flags win on any conflict) --

  function figuresIn(text) {
    // Masked to spaces of the same length so any index taken here still
    // points at the same character of the original string.
    var masked = String(text || "").replace(DATE_RE, function (d) { return new Array(d.length + 1).join(" "); });
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

  // -- the excerpts -----------------------------------------------------------

  function indexSources(payload) {
    state.sources = {};
    (payload.sources || []).forEach(function (s) { state.sources[s.cid] = s; });
  }

  function highlightPlan(payload) {
    // Per cid: the quote ranges to fill and the column positions to underline.
    var plan = {};
    var answer = payload.answer;
    if (!answer) { return plan; }
    var flags = rowsByClaim(payload);
    answer.claims.forEach(function (claim) {
      var own = flags[claim.id] || [];
      // A quote is searched only in the excerpts the claim cites, so those are
      // the excerpts to highlight it in.
      claim.citations.forEach(function (cid) {
        var entry = plan[cid] = plan[cid] || {quotes: [], columns: []};
        if (claim.quote) { entry.quotes.push(claim.quote); }
      });
      matchedColumns(claim, figuresIn(claim.text)).forEach(function (m) {
        var entry = plan[m.cid] = plan[m.cid] || {quotes: [], columns: []};
        if (entry.columns.indexOf(m.position) < 0) { entry.columns.push(m.position); }
      });
      // A flagged column is underlined too, so the reader can see the cell the
      // figure really sits in; the flag carries the label, and the claim's own
      // citations say which excerpts to look in for it.
      own.forEach(function (f) {
        if (f.kind !== "column_mismatch" && f.kind !== "duration_mismatch") { return; }
        claim.citations.forEach(function (cid) {
          var source = state.sources[cid];
          if (!source) { return; }
          source.columns.forEach(function (c) {
            if (c.label !== f.source_string) { return; }
            var entry = plan[cid] = plan[cid] || {quotes: [], columns: []};
            if (entry.columns.indexOf(c.index) < 0) { entry.columns.push(c.index); }
          });
        });
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
    var pre = el("pre", "fd-source-text" + (source.kind === "table" ? " fd-source-text--table" : ""));
    var text = source.text;
    var ranges = [];
    if (entry) {
      entry.quotes.forEach(function (q) {
        var r = quoteRange(text, q);
        if (r) { ranges.push({start: r.start, end: r.end, cls: "fd-quote"}); }
      });
      if (entry.columns.length) {
        tableRows(source).forEach(function (row2) {
          if (row2.values.length !== source.columns.length) { return; }
          entry.columns.forEach(function (position) {
            var cell = row2.values[position];
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
        pre.appendChild(el("mark", classes.join(" "), piece));
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
      var group = el("div", "fd-source-group");
      group.appendChild(el("hr", "fd-rule"));
      var title = el("h3", "fd-source-entity", groups[ticker][0].company);
      title.appendChild(tickerBadge(ticker));
      group.appendChild(title);

      // Cited excerpts first: after the answer lands, the passages a claim
      // quoted are what the reader came for, and everything else is what the
      // model also had. Nothing is hidden, the order is fixed.
      var cited = [], rest = [];
      groups[ticker].forEach(function (s) {
        var entry = plan[s.cid];
        if (entry && (entry.quotes.length || entry.columns.length)) { cited.push(s); } else { rest.push(s); }
      });
      cited.forEach(function (s) { group.appendChild(sourceRow(s, plan[s.cid], true)); });
      if (cited.length && rest.length) {
        group.appendChild(row([label("also supplied to the model, not cited")], []));
      }
      rest.forEach(function (s) { group.appendChild(sourceRow(s, plan[s.cid], false)); });
      root.appendChild(group);
    });
  }

  function sourceRow(source, entry, cited) {
    var wrap = el("div", "fd-source");
    wrap.id = "src-" + source.cid;

    var rail = el("div", "fd-source-rail");
    if (cited) { rail.appendChild(el("span", "fd-cited-mark", "quoted in the brief")); }
    rail.appendChild(el("span", "fd-source-cid", source.cid));
    rail.appendChild(el("span", "fd-source-tag", source.kind));
    var form = el("span", "fd-source-line");
    form.appendChild(el("strong", null, source.form + " " + source.fiscal_label));
    rail.appendChild(form);
    rail.appendChild(el("span", "fd-source-line",
      (source.form === "10-K" ? "year ended " : "quarter ended ") + fmtDateShort(source.period_end) +
      ", filed " + fmtDateShort(source.filing_date)));
    rail.appendChild(el("span", "fd-source-line", sectionName(source)));
    // The EDGAR address stays on screen and stays selectable. It is not a
    // link: the page runs offline in a container, so a click would open a
    // dead tab, and a link here would be a third class of control.
    if (source.url && source.url.indexOf("https://www.sec.gov/") === 0) {
      rail.appendChild(record(source.url));
    }
    wrap.appendChild(rail);

    var body = el("div");
    body.appendChild(renderHighlighted(source, entry));
    var facts = el("div", "fd-source-facts");
    if (source.units) {
      facts.appendChild(source.units_source === "declared" ?
        check("ok", "units: " + source.units + " (declared in the filing)") :
        check("note", "units read as " + source.units + ", inferred from the header, not declared"));
    }
    if (source.kind === "table" && source.column_source !== "parsed") {
      facts.appendChild(check("note", "the column layout could not be read from the header"));
    }
    if (facts.firstChild) { body.appendChild(facts); }
    wrap.appendChild(body);
    return wrap;
  }

  // A marker, not a flash: it survives the second or two a panel needs to
  // refocus on a projector, and it behaves identically under reduced motion,
  // where an animated ring produces nothing at all.
  function goToSource(cid) {
    var card = $("src-" + cid);
    if (!card) { return; }
    var prev = document.querySelector(".fd-source.fd-current");
    if (prev) { prev.classList.remove("fd-current"); }
    card.classList.add("fd-current");
    card.scrollIntoView({behavior: REDUCED ? "auto" : "smooth", block: "start"});
  }

  // -- working papers ----------------------------------------------------------

  function sectionBuckets(sections) {
    var items = Object.keys(sections || {}).filter(function (k) { return k !== "*"; });
    if (!items.length) { return null; }
    var weights = items.map(function (k) { return sections[k]; });
    var top = Math.max.apply(null, weights);
    var first = [], next = [];
    items.forEach(function (k) {
      var name = SECTION_PLAIN[k] || SECTION_TEXT[k] || ("Item " + k);
      if (sections[k] === top) { first.push(name); } else { next.push(name); }
    });
    var uniq = function (list) {
      var seen = {}, out = [];
      list.forEach(function (n) { if (!seen[n]) { seen[n] = true; out.push(n); } });
      return out;
    };
    return {first: uniq(first), next: uniq(next)};
  }

  function checkLine(name, pair, unchecked, unit) {
    if (!pair || !pair[1]) { return "Nothing to check: this answer has no " + unit + "."; }
    return name + ": " + pair[0] + " of " + pair[1] + " matched" +
      (unchecked ? "; " + unchecked + " could not be tested" : "") + ".";
  }

  function paintDetails(payload) {
    var root = $("details");
    clear(root);
    show("method-band", true);
    var plan = payload.plan;

    // a. What the scope came to, before anything was retrieved.
    var scopeLine;
    if (payload.status !== "ok") {
      scopeLine = "Refused at the scope: " + (REFUSAL_PLAIN[payload.status] || "the question could not be scoped") + ".";
    } else {
      scopeLine = "Scope: " + plan.companies.length + (plan.companies.length === 1 ? " company, " : " companies, ") +
        periodPhrase(plan) + ". " +
        (plan.comparison ? "Read as a side-by-side comparison." :
          plan.timeline ? "Read as a trend over time." : "Read as a single scope.");
    }
    root.appendChild(row([h3("scope decided"),
      record("status " + payload.status + " . period_mode " + plan.period_mode +
        " . budget " + fmt(plan.budget_tokens) + " tokens")],
      [el("p", "fd-note-read", scopeLine)]));
    if (payload.status !== "ok") { return; }

    // b. Which sections were ranked above the rest, as a sentence; the
    // multipliers that produced the ranking stay in the margin.
    var buckets = sectionBuckets(plan.sections);
    if (buckets) {
      var weightBits = Object.keys(plan.sections).map(function (k) { return k + " x" + plan.sections[k]; });
      var sentence = "Read first: " + joinNames(buckets.first) + ".";
      if (buckets.next.length) { sentence += " Read next: " + joinNames(buckets.next) + "."; }
      sentence += " The rest of each filing was searched but ranked below these.";
      var railNodes = [h3("sections prioritised"), record(weightBits.join(" . "))];
      var colNodes = [el("p", "fd-note-read", sentence)];
      if (plan.note_intent) { colNodes.push(el("p", "fd-note", "Note intent: " + plan.note_intent)); }
      root.appendChild(row(railNodes, colNodes));
    }

    // c. What each company holds, as a range rather than a label dump.
    if (plan.companies && plan.companies.length) {
      plan.companies.forEach(function (c) {
        var labels = sortedFiscal(c.available);
        var line = displayName(c, plan.companies) + " (" + c.ticker + "): " +
          (labels.length ? labels.length + " filings on file, " + labels[0] + " through " + labels[labels.length - 1] + "." :
            "no filings on file.");
        var raw = "matched \"" + c.matched_alias + "\"" +
          (c.missing_years && c.missing_years.length ? " . missing fiscal " + c.missing_years.join(", ") : "");
        root.appendChild(row([h3("periods held"), record(raw)], [el("p", "fd-note-read", line)]));
      });
    }

    // d.
    if (plan.notes && plan.notes.length) {
      var notes = el("ul", "fd-list");
      plan.notes.forEach(function (note) { notes.appendChild(el("li", null, note)); });
      root.appendChild(row([h3("scope notes")], [notes]));
    }

    // e. What was actually sent, with a meter whose 100% mark is visible.
    var budget = payload.budget || {};
    var assembled = "Excerpts totalling " + fmt(budget.tokens_est) + " of the " + fmt(budget.budget_tokens) +
      " token ceiling were sent. " + fmt(budget.chunks_dropped) + " lower-ranked candidates did not fit.";
    var meter = el("div", "bh-meter fd-meter");
    var fill = el("span");
    var share = budget.budget_tokens ? Math.min(100, 100 * budget.tokens_est / budget.budget_tokens) : 0;
    fill.style.width = share.toFixed(1) + "%";
    meter.appendChild(fill);
    root.appendChild(row([h3("context assembled"),
      record("provider input tokens " + fmt(budget.input_tokens_actual))],
      [el("p", "fd-note-read", assembled), meter]));
    root.appendChild(row([], contextTable(payload.context || []), true));

    // f.
    var quotaNodes = [];
    (plan.quotas || []).forEach(function (q) {
      quotaNodes.push(el("p", "fd-note-read", q.ticker + ": " + q.chunks + " excerpts kept" +
        (q.pinned && q.pinned.length ? "; the opening of each prioritised section was guaranteed a place" : "") + "."));
    });
    var seated = (payload.context || []).filter(function (r) { return r.pinned; });
    quotaNodes.push(el("p", "fd-note", seated.length ?
      "Guaranteed places went to " + seated.map(function (r) { return r.cid; }).join(", ") + "." :
      "No excerpt needed a guaranteed place."));
    var quotaRaw = (plan.quotas || []).map(function (q) {
      return q.ticker + " positions " + (q.pinned && q.pinned.length ? q.pinned.join(", ") : "none") +
        " . row-label seats " + (q.row_label && q.row_label.length ? q.row_label.join(", ") : "none");
    }).join(" | ");
    root.appendChild(row([h3("retrieval quotas"), record(quotaRaw || "none")], quotaNodes));

    // g.
    if (plan.sub_queries && plan.sub_queries.length) {
      var subs = el("ol", "fd-list");
      plan.sub_queries.forEach(function (q) { subs.appendChild(el("li", null, q)); });
      root.appendChild(row([h3("sub-queries")], [subs]));
    }

    // h. Five plain statements. A zero denominator says there was nothing to
    // check rather than printing 0/0, which a numerate reader reads as
    // undefined; telling those two apart is the value of the check layer.
    if (payload.checks) {
      var c = payload.checks;
      var lines = [
        checkLine("Quotes traced to the cited excerpt", c.quotes_located, 0, "quotes"),
        checkLine("Figures printed inside the quoted words", c.figures_in_quote, c.figures_unchecked, "figures"),
        checkLine("Figures traced to the right table column", c.columns_matched, c.columns_unverified, "table figures"),
        checkLine("Units matched against the filing", c.units_matched, c.units_unchecked, "figures with stated units"),
        c.unlinked.length ?
          c.unlinked.length + " summary sentence" + (c.unlinked.length === 1 ? "" : "s") + " carry no supporting finding." :
          "Every summary sentence carries at least one supporting finding."
      ];
      var checkNodes = lines.map(function (t) { return el("p", "fd-note-read", t); });
      [["flagged", c.flags, "flag"], ["could not be tested", c.notes, "note"]].forEach(function (group) {
        checkNodes.push(el("p", "fd-note", group[0] + (group[1].length ? ":" : ": none")));
        if (!group[1].length) { return; }
        var list = el("ul", "fd-list");
        group[1].forEach(function (f) {
          var li = el("li");
          li.appendChild(check(group[2], KIND_TEXT[f.kind] || f.kind.replace(/_/g, " ")));
          li.appendChild(document.createTextNode(" " + (f.where ? f.where + ": " : "") + f.detail +
            (f.source_string ? " [in the " + f.source_string + " column]" : "")));
          list.appendChild(li);
        });
        checkNodes.push(list);
      });
      var statLine = "quotes located " + c.quotes_located[0] + "/" + c.quotes_located[1] +
        " | figures in quote " + c.figures_in_quote[0] + "/" + c.figures_in_quote[1] + ", unchecked " + c.figures_unchecked +
        " | columns matched " + c.columns_matched[0] + "/" + c.columns_matched[1] + ", unverified " + c.columns_unverified +
        " | units matched " + c.units_matched[0] + "/" + c.units_matched[1] + ", unchecked " + c.units_unchecked +
        " | unlinked " + c.unlinked.length;
      root.appendChild(row([h3("checks run"), record(statLine)], checkNodes));
    } else {
      root.appendChild(row([h3("checks run")], [el("p", "fd-note-read",
        payload.dry_run ? "No answer has been produced yet, so nothing has been checked." :
          "No answer was produced, so nothing was checked.")]));
    }

    // i. The run, as one sentence plus three numbers a reader can feel.
    var usage = payload.usage || {};
    var timing = payload.timing_ms || {};
    var totalMs = 0;
    Object.keys(timing).forEach(function (k) { if (typeof timing[k] === "number") { totalMs += timing[k]; } });
    var runLine = "Answered from one model request in " + (totalMs / 1000).toFixed(1) + " s. " +
      fmt(usage.input_tokens) + " input and " + fmt(usage.output_tokens) + " output tokens." +
      (payload.cost_usd !== null && payload.cost_usd !== undefined ?
        " Cost $" + Number(payload.cost_usd).toFixed(4) + " for this answer." : "");
    var searchLine = "Search: " + (state.health && state.health.dense ? "keyword and meaning-based." : "keyword only.");
    var tiles = el("div", "fd-stats--record");
    tiles.appendChild(statTile((totalMs / 1000).toFixed(1) + " s", "time to answer"));
    tiles.appendChild(statTile(fmt((payload.sources || []).length), "excerpts read"));
    // A stat tile never prints n/a: with no cost to show, the tile changes
    // subject rather than rendering a null.
    if (payload.cost_usd !== null && payload.cost_usd !== undefined) {
      tiles.appendChild(statTile("$" + Number(payload.cost_usd).toFixed(4), "cost of this answer"));
    } else {
      tiles.appendChild(statTile(fmt(usage.input_tokens), "tokens sent to the model"));
    }
    var latency = Object.keys(timing).map(function (k) {
      return k + " " + (timing[k] === null ? "n/a" : timing[k] + " ms");
    }).join(", ");
    var runRaw = "attempts " + fmt(payload.llm_attempts) + "/" + fmt(payload.llm_completed) +
      " . prompt " + (payload.prompt_version || "n/a") +
      " . stop " + (payload.stop_reason || "n/a") +
      " . request " + (payload.request_id || "none") +
      " . " + (latency || "no timings") +
      " . backend " + (payload.backend || "none") +
      " . model " + (payload.model || "none") +
      (usage.cache_read_input_tokens ? " . cache read " + fmt(usage.cache_read_input_tokens) : "");
    root.appendChild(row([h3("run record"), record(runRaw)],
      [el("p", "fd-note-read", runLine), tiles, el("p", "fd-note-read", searchLine)]));

    // j. The exact request, last, where its length costs nothing.
    var reqNodes = [];
    if (payload.prompt) {
      reqNodes.push(el("p", "fd-note-read", "The exact request. Nothing else was sent to the model."));
      reqNodes.push(el("p", "fd-note", "This block scrolls inside itself."));
      reqNodes.push(el("span", "fd-label", "system"));
      reqNodes.push(scrollablePre(payload.prompt.system));
      reqNodes.push(el("span", "fd-label", "user"));
      reqNodes.push(scrollablePre(payload.prompt.user));
    } else {
      reqNodes.push(el("p", "fd-note-read",
        "No request has been made yet. The block below is the excerpts exactly as the user turn will carry them."));
      reqNodes.push(el("p", "fd-note", "This block scrolls inside itself."));
      reqNodes.push(scrollablePre(payload.rendered || ""));
    }
    var reqWrap = el("div", "fd-request");
    reqNodes.forEach(function (n) { reqWrap.appendChild(n); });
    root.appendChild(row([h3("the request sent")], [reqWrap], true));
  }

  // tabindex makes the overflow reachable from the keyboard; a scrollable
  // block with no focusable child cannot be scrolled without a mouse.
  function scrollablePre(text) {
    var pre = el("pre", "bh-pre", text);
    pre.tabIndex = 0;
    return pre;
  }

  function contextTable(rows) {
    var wrap = el("div", "fd-table-wrap");
    var table = el("table", "bh-table fd-table");
    var head = el("tr");
    ["excerpt", "company", "filing", "section", "kind", "tokens", "guaranteed place"].forEach(function (h) {
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
      cidCell.appendChild(chip(r.cid, function () { goToSource(r.cid); }, "excerpt " + r.cid,
        "Go to excerpt " + r.cid));
      tr.appendChild(cidCell);
      tr.appendChild(el("td", null, r.ticker));
      tr.appendChild(el("td", null, r.form + " " + r.fiscal_label));
      tr.appendChild(el("td", null, (SECTION_TEXT[r.item] || ("Item " + r.item)) +
        (r.note_title ? " > " + r.note_title : "")));
      tr.appendChild(el("td", null, r.kind));
      tr.appendChild(el("td", "bh-num", fmt(r.n_tokens)));
      tr.appendChild(el("td", null, r.pinned ? String(r.pinned).replace(/_/g, " ") : ""));
      body.appendChild(tr);
    });
    table.appendChild(body);
    wrap.appendChild(table);
    return [wrap, el("p", "fd-scroll-note", "This table is wider than the screen. Scroll it sideways.")];
  }

  init();
})();
