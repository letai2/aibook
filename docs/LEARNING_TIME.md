# Learning-time estimates

These are editorial planning ranges for a learner who already knows basic Python,
not measured completion statistics, confidence intervals, or deadlines. The
generated book shows them for every lesson, chapter, part and the complete path.
The live per-part totals are on `dist/learning-time.html` and in the current review.

`book_src/learning_time.py` is the single source of estimates. Each of the 92
lessons has an explicit reviewed activity profile; assignment is not guessed from
the title or computed solely from text length.

| Profile | Concept minutes before multiplier | Difficulty multiplier | Code tracing | Lab writing/variation/repair | Explanation/self-check | Prerequisite recall |
| --- | --- | --- | --- | --- | --- | --- |
| Introduce | 5–10 | 1.00 | 3–6 | 10–18 | 5–10 | 0–5 |
| Calculate | 8–14 | 1.15 | 5–10 | 15–25 | 8–15 | 3–7 |
| Construct | 8–15 | 1.25 | 8–15 | 20–35 | 10–18 | 5–10 |
| Investigate | 10–18 | 1.35 | 8–15 | 20–35 | 10–20 | 5–10 |
| Integrate | 12–20 | 1.50 | 12–20 | 30–50 | 15–25 | 8–15 |

The first reading allowance is separate: visible prose words divided by an
assumed 180–100 words/minute, with 3–5 minute minima. These are deliberately
identified planning assumptions, **not validated Persian reading-speed data**.
Code is excluded from that word count and has its own activity allowance. Only
conceptual understanding receives the multiplier; it is not compounded over all
other activities. The lesson total rounds each endpoint upward to five minutes.

Book exercise work already performed in the lab is counted once. The explanation
allowance is for self-check/reflection, not a duplicate charge for the same TODO.
Chapter totals sum lesson ranges. Part totals add a separate synthesis checkpoint;
book totals sum parts. Checkpoint allowances are 45–75 minutes in parts1–4,
60–100 in parts5–9, 60–105 in parts10–14 and 90–150 in part15. Those assignments
reflect increasingly integrated deliverables rather than longer reading alone.
Displayed aggregate hours round outward to half an hour; exact minute endpoints
are retained in HTML data attributes and tested for additive consistency.

Excluded: optional review notebooks (12 × approximately45–90 minutes), initial
installation (plan a separate1–3-hour session; network/device problems may take
longer), long model runs, optional papers, breaks and open-ended personal projects.
CPU lab runtime is usually much shorter than the learner's thinking/writing time;
machine runtime is not used as a learning-time proxy.

The [Rice University workload-estimation guide](https://cte.rice.edu/resources/workload-estimator)
informed the separation of reading purpose, difficulty and task type. Its English
reading-rate tables were not imported as evidence for Persian learners. The
specific profiles, multipliers and minute budgets above are our editorial choices.

Record actual time and a difficulty note for three early lessons, then adjust
your study schedule. Longer sessions can be split without changing the expected
deliverable. The next evidence-based refinement would be a voluntary learner
pilot; none has yet been conducted. Tests verify coverage, sensible bounds,
component sums and aggregate arithmetic, not the empirical accuracy of estimates.
