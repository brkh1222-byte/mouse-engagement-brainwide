# Refined research plan (working copy, 2026-11-24)

Full version with rationale and citations: Claude Doc https://claude.ai/artifact/PYBb94penovduagu6xcd1o (Part B). This file is the compact repo copy that feeds `preregistration/prereg.md` (T1.4).

## B.1 Research question
After removing block prior, movement, arousal and time-in-session, which brain regions' pre-stimulus population activity separates GLM-HMM engaged from disengaged trials in the IBL Brain-Wide Map, and does that pattern agree with the engagement network in Steinmetz et al. 2019?

| ID | Hypothesis | Test | Status |
|---|---|---|---|
| H1 | Engagement decodable pre-stimulus in ≥1 region after controls | Fisher-combined p vs pseudosession null; BH q<0.05 | Confirmatory |
| H2 | Survives movement/arousal controls | Model B (covariates regressed out) above null, q<0.05; share of effect retained | Confirmatory |
| H3 | Midbrain-arousal + frontal-motor > primary sensory | Pre-listed groups (e.g. MRN, SCm, PAG, MOs, ACAd vs VISp, SSp-bfd, AUDp); one-sided permutation | Confirmatory, directional |
| H4 | IBL map agrees with Steinmetz | Spearman ρ over shared Beryl regions; one-sided region-permutation p<0.05 | Confirmatory, low power |
| E1 | Engagement map ≠ prior map | Compare with Findling 2025 map | Exploratory |
| E2 | Abrupt vs ramping change at state switches | Decoder output aligned to switches | Exploratory |

## B.2 Design
1. GLM-HMM on each BWM mouse's full trained-stage history; inputs: signed contrast, previous choice, previous rewarded side, bias, **estimated block prior** (Mohammadi et al. 2025). K=2–5 by CV log-likelihood; global fit then per-mouse. Engaged = largest stimulus weight; label trials with **filtered** posterior > 0.8.
2. Inclusion: ≥400 trials; ≥30 labelled trials per class; ≥10 good units per region. Keep all trials (no 80 ms–2 s RT filter).
3. Features: spike counts −400 to −100 ms (inside enforced 400–700 ms wheel quiescence); secondary −600 to −100 ms (Findling window). Beryl regions.
4. Covariates: block prior estimate, wheel speed, licks, face motion energy / pose, pupil (QC-passing), trial number, cumulative reward.
5. Decoder: L2 logistic, nested CV with contiguous folds, balanced accuracy, equal neuron subsampling. Model A raw; **Model B covariates regressed out (primary)**.
6. Nulls: primary = synthetic sessions (pseudo-stimuli → simulate GLM-HMM choices → re-infer states), 200/session; secondary = state sequences from other sessions.
7. Steinmetz: 3-choice GLM-HMM pooled over 10 mice; same decoding; also recompute Steinmetz engagement index; map to Beryl.

## B.3 Analysis
Session: corrected balanced accuracy = real − null median; p = rank in null. Region: Fisher combination, median effect, ≥2 sessions. BH q<0.05 across regions (A and B separately). Mixed model with mouse and lab random intercepts as a check.
Robustness: posterior 0.7/0.9; −600 to −100 window; K±1; state-swap null; Stouffer. "Robust" = H1 region set changes by < 1/3.
H4: Spearman ρ, 10,000 region-label permutations, bootstrap 95% CI; for GLM-HMM label and for Steinmetz's engagement index.

## B.4 Outcomes
Positive (specific + replicates) / positive but embodied (fails Model B) / global not local (H3 fails) / no replication (H4 fails) / null (H1 fails). All publishable because pre-registered; report per-region effects with intervals.

## B.5 Paper and venue
Target eNeuro (has "Research Article: Confirmation" and "Negative Results" types); bioRxiv preprint on submission; fallback Royal Society Open Science; JEI needs an adult mentor.
Figures: (1) task + GLM-HMM states vs blocks; (2) pipeline + example null; (3) Swanson flatmap, Model A vs B; (4) A vs B scatter + region groups (H2, H3); (5) Steinmetz replication ρ (H4); (6) engagement vs prior map, switch-aligned dynamics (E1, E2); S1–S4 counts, robustness, neuron curves, mouse/lab.

## B.6 Timeline (4 weeks, starting Mon 2026-11-30)
- Wk 1 (Nov 30–Dec 4): email Melin/Churchland; env + Open Alyx account; GLM-HMM fits; count labelled trials; **commit pre-registration Fri Dec 4 (gate)**.
- Wk 2 (Dec 7–11): spikes + covariates; decoder + synthetic-session null on 10 pilot sessions.
- Wk 3 (Dec 14–18): full BWM run + aggregation; Steinmetz GLM-HMM + decoding.
- Wk 4 (Dec 21–24 + after break): figures; `paper/notes.md`; student writes paper; mentor review; bioRxiv.
Minimum 2-week path: drop E1/E2 and Steinmetz video covariates; IBL H1–H3 only.

## B.7 Top risks
Melin overlap (email day 1); too few disengaged trials (count before prereg); biased states = blocks (prior in model); H4 power (ρ≈0.45 needed with ~30 regions for 80% power); unstable Steinmetz fit (pool mice; fall back to Steinmetz engagement index); video QC gaps; analysis drift (prereg + dated amendments).

## Suggested edits
See `notes/citation_audit.md` and the doc's "Suggested changes" section (PROPOSAL.md §1–§10, TASKS.md T1.1–T4.3, CLAUDE.md null wording and ONE login).
