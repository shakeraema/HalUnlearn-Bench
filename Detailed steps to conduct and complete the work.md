Detailed steps to conduct and complete the work

Phase 1 — Positioning (2–3 days, before touching Colab)

Read the Attention Shifting paper and Ghostbusters paper in full.
Write a 200-word "differentiation paragraph": what UHIF/Props 1–2/Trilemma/HalUnlearn-Bench give that they don't (formal proofs + a reusable composite-score benchmark protocol, as opposed to a single new method).
Insert this paragraph into your Related Work section now, so every later decision is anchored to a clear claim.

Phase 2 — Math check (parallel, 1–2 days)
4. Send Propositions 1, 2, and the Trilemma proof to one professor or PhD student for a sanity check. Ask specifically: "does the Lipschitz continuity assumption in Prop 1 actually hold for transformer attention layers, or does it need a caveat?"
5. Revise the proofs based on feedback before any more work — this is cheap now, expensive after submission.

Phase 3 — Local smoke test on the M1 Air (half a day)
6. Download the notebook, install deps, get an OpenRouter API key, set OPENROUTER_API_KEY.
7. Edit CFG in Cell 2: set n_authors=3, phase0_epochs=1, unlearn_epochs=1, seeds=[42].
8. Run all 9 cells top to bottom. Fix any errors (TOFU schema mismatches are the most likely one — check the actual column names once loaded).
9. Confirm RF and HR are no longer NaN/0.0 — if they still are, that's a bug to fix before scaling up, not something to run past.

Phase 4 — Calibrate tau (half a day)
10. From step 9's outputs, print the norm_entropy values for a handful of clearly-abstaining vs. clearly-confident answers.
11. Pick a entropy_threshold_tau that cleanly separates them; update Cell 2.

Phase 5 — Move to Colab, full-scale dry run (1 day)
12. Upload the notebook to Colab, select the T4 GPU runtime.
13. Set CFG back to full scale (n_authors=200, phase0_epochs=6, unlearn_epochs=3) but keep seeds=[42] only for now.
14. Run it once fully. This is where you'll discover real timing and OpenRouter rate-limit issues — budget a day for debugging here.

Phase 6 — Full multi-seed run (several days, mostly unattended)
15. Set seeds=[42, 100, 2026] (or 5 seeds if time allows) and start the run.
16. Because of the checkpointing, you can let it run in chunks across multiple Colab sessions — it auto-resumes.
17. Monitor OpenRouter usage/cost as it runs (probe generation + LLM-judge calls add up at 200 authors × multiple seeds).

Phase 7 — Human validation of the LLM-judge (half a day)
18. Pull ~100 of the saved HR judgments from the checkpoint JSON.
19. Have yourself and one other person label them independently as hallucination/abstain, blind to the model's verdict.
20. Compute Cohen's kappa between judge and humans; report it in the paper.

Phase 8 — Analysis and writing (1–2 weeks)
21. Use Cell 9's summary table as your Results section base.
22. Check whether ME/NPO cluster near the Pareto frontier as Proposition 2 predicts — this is your key empirical validation of the theory.
23. Update the "Pilot Results" section of your existing doc with the real numbers, replacing the NaN placeholder table.
24. Write the Discussion section connecting the Trilemma proof to what you actually observed as η (learning rate) varied.

Phase 9 — Submission
25. Format for TMLR (or your chosen journal) — check their template and page-limit rules.
26. Release the notebook and results as code/data alongside submission — reviewers increasingly expect this, and it directly supports your novelty claim.
27. Submit.