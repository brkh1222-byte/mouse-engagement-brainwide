# TASKS: mouse-engagement-brainwide

Work top to bottom. Tick boxes as tasks complete. Read PROPOSAL.md and CLAUDE.md first.

## Phase 0: Setup
- [x] **T0.1** Environment via `uv` (no conda): `uv venv --python 3.11 .venv`; pins in `requirements.txt` (top level) and `requirements.lock` (full freeze). CPU only (JAX CPU, numba). `src/config.py` holds paths and all pre-registered constants. (pynwb deferred to T4.1.)

## Phase 1: Checks, behaviour and pre-registration
- [x] **T1.1 Competition check.** Search bioRxiv, arXiv and IBL publications for Melin et al. 2026 and any brain-wide engagement or GLM-HMM Neuropixels analysis. Summarise in `notes/competition.md`. **Flag if overlapping.**
- [ ] **T1.2** Write `src/ibl_sessions.py`: connect with ONE (public credentials), list the Brainwidemap sessions, and download trials tables into `data/ibl/trials/`.
  *Done when:* about 459 sessions are listed.
- [ ] **T1.3** Write `src/glmhmm.py` to fit a 3-state GLM-HMM (Ashwood 2022 design) across mice and label trials.
  *Done when:* states reproduce Ashwood's engaged/biased pattern; `results/glmhmm_states.parquet` and a figure exist.
- [ ] **T1.4 PRE-REGISTRATION.** Write `preregistration/prereg.md` covering:
  - the hypotheses H1–H3;
  - the pre-stimulus window;
  - the session inclusion rule (minimum disengaged trials);
  - covariates, decoder, null and multiple-comparison method.

  **The student reviews it; commit and push before any neural decoding.**

## Phase 2: Neural features
- [ ] **T2.1** Write `src/spikes.py` to get pre-stimulus spike counts per good unit and per region (Beryl atlas), using ONE/brainbox. Cache per session.
- [ ] **T2.2** Write `src/covariates.py` for block side, wheel speed, whisker motion energy and trial number.

## Phase 3: Decoding
- [ ] **T3.1** Write `src/decode.py`: per region × session, nested-CV L2 logistic decoding of engagement with covariates, plus the session-permutation null. Test it on 10 sessions first.
- [ ] **T3.2** Full run on all eligible sessions, CPU only on this Mac: resumable session-by-session streaming (download → features → delete raw), `caffeinate -i nice`, at most 2 heavy processes; then region aggregation (Stouffer + FDR) into `results/region_effects.csv`.
- [ ] **T3.3** Brain map figure (iblatlas swanson/flatmap).

## Phase 4: Replication
- [ ] **T4.1** Write `src/steinmetz.py` to load Steinmetz 2019 (DANDI 000017 or Figshare), fit the GLM-HMM and decode the same way.
- [ ] **T4.2** Compute the cross-dataset Spearman correlation over shared regions, with a figure.

## Phase 5: Write-up
- [ ] **T5.1** Write `paper/notes.md` with bullet results and deviations from the pre-registration. Claude drafts `paper/main.tex`; the researcher reviews and edits it (AI-use disclosure; authorship TBD).
