# Where this goes if the client is sold

Each item below is tied to something the deal team already does by hand. None of them changes
the one rule the demo runs on: code decides what the model reads, the model writes, and code
checks what it wrote against the source.

1. **Nightly ingestion from EDGAR for the portfolio and the comparables watchlist.** New 10-K,
   10-Q, and 8-K filings and earnings transcripts land in the same parser and index by morning,
   with a per-filing fingerprint so nothing is indexed twice.
2. **Deterministic figures from XBRL company facts.** Reported values come from the structured
   filing data with their period and units attached, and the model narrates around a table it
   did not have to read; the evidence checks then compare text to facts, not text to text.
3. **A new-filing brief.** When a portfolio company files, the question the team asked last
   quarter runs again against the new filing, and the associate receives a brief of what moved,
   with sources.
4. **Deal-room documents on the same pipeline.** Confidential information memoranda, quality of
   earnings reports, and lender presentations get the same section detection, table
   preservation, and source links, inside the client's own storage.
5. **Evaluation in the delivery pipeline.** The tuning and held-out sets run on every prompt or
   retrieval change; a change ships only when the evidence-check rates and the graded rubric hold.
   A model-graded faithfulness score joins the deterministic checks as a second opinion, never as
   the gate.
6. **Deployment in the client's tenancy.** The service runs in the client's cloud account with
   their identity provider, retention policy, and audit log; the model provider is a
   configuration switch, so the same code runs on OpenAI, Azure OpenAI, or Anthropic.
7. **More than one request when the constraint lifts.** A second pass that re-retrieves for
   any claim the checks flagged, and batch screens across the whole portfolio, are the first two
   uses of a second model call.
