# What this is worth to a deal team, and how I would measure it

## Who uses it

Three people at a mid-market private equity firm read the same filings today:

- the deal associate screening a handful of comparables before a memo (risk factors, MD&A,
  segment revenue);
- the portfolio operations lead re-reading each quarter's 10-Q for the companies the fund
  already owns ("what changed since last filing");
- the CFO office and the credit team asking how safe the operating banks and counterparties
  are (capital, uninsured deposits, net interest income, what the bond book is worth).

All three do it by hand, from 250 to 400 page documents, and the output they need is a short
brief where every figure carries its period, its units, and a page they can open.

## What changes

One question in, one model request out, and every figure in the brief links to the row and
column it came from. The reviewer's job moves from finding the numbers to checking them, and
the checking is faster because the source opens beside the claim with its row label and column
header visible.

What the system does on its own: it decides which companies, filings, and sections to read
(shown before the request), keeps tables intact with their units and column headers, makes
exactly one model request, and then checks each quote and figure against the excerpt it cites.
What it does not do: it does not judge whether the brief is right. A person still reads it.

## What I would measure in a pilot (the numbers that matter are the client's, not mine)

1. Time to an analyst-accepted brief, including source checking, for a fixed set of questions
   the team already answers by hand. Measured before the pilot (current process) and during it.
2. Question volume per week and per associate, from the request log.
3. The share of figures the reviewer changed after checking the source, by flag type. This is
   the number that tells you whether the evidence checks are catching the right things.
4. Cost per question from the usage tile, and the monthly total at the observed volume.

Hours saved per month = questions per month x (baseline minutes - pilot minutes) / 60. Every
term in that formula comes from the client's own log and baseline, so the claim is theirs to
audit.

## The one number I measured myself

<fill in during milestone 8: minutes to source-check the six figures of the bank brief with
the source panel, against minutes to find and check the same six figures in the two raw 10-Qs,
timed on the same day. Report both numbers and the date.>

## Assumptions, stated as assumptions

- A three-company risk comparison takes an associate most of a working day today. Unmeasured;
  a pilot baseline replaces it.
- A reviewer needs 10 to 20 minutes to check a brief with linked sources. Unmeasured beyond the
  single timing above.
- A firm with 20 associates asking 15 questions a week each would run about 1,300 questions a
  month; at the per-question cost on the tile that is a small line item next to the review
  time. The volume is a placeholder until a request log exists.
- A comparable deployment for filing review at a federal financial regulator measured about a
  30 percent reduction in review time. That figure explains why I chose this workflow; it is
  not a prediction for this client.
