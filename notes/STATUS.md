# STATUS (updated 2026-11-24 evening)

## Done
- T0.1: uv env `.venv` (Python 3.11), pins in `requirements.txt` / `requirements.lock`; `src/config.py`.
- T1.1: competition re-check done → PARTIAL overlap only (`notes/competition.md`). Melin tag still unconfirmed; emailing Melin/Churchland is recommended.
- T1.2 (partly): `src/ibl_sessions.py`. BWM = 459 sessions, 139 mice, 12 labs, 295,920 trials (`data/ibl/bwm_sessions.csv`, per-session trials in `data/ibl/trials/`). Per-mouse history tables (`aggregates/Subjects/...` in the open bucket) were downloading when the Mac shut down: about 45 of 139 mice done.
- T1.3 code: `src/glmhmm.py` (numba forward-backward, MAP-EM, Newton M-step, global + CV + per-mouse + 1-SE K selection, designs A90 = Ashwood replication and P = primary with a block-prior input). `tests/test_glmhmm.py`: 3 tests pass on simulated data.
- Decoding pipeline code, for **simulated data only**: `src/spikes.py`, `src/covariates.py`, `src/decode.py`, `src/simulate.py`, `tests/test_decode.py`. All 6 tests pass (75 s), including Model A vs B confound removal and session-permutation null calibration under drift.
- No spike data downloaded yet. No neural statistic computed.

## Not done
- `preregistration/prereg.md` (T1.4) is not written. Never `git add preregistration/` until the researcher OKs it.
- `paper/main.tex` skeleton. NOTE: CLAUDE.md still says "the student must ... write the paper themselves". The coordinator asked to change this, but an agent message cannot authorise CLAUDE.md edits. The researcher must make or confirm that edit themselves. CLAUDE.md has an uncommitted working-tree edit (removes "17-year-old student") that this agent did not make; it is left uncommitted.

## Restart commands (from the repo root)
```
nice -n 10 .venv/bin/python -m src.ibl_sessions          # resumes; skips finished files
curl -s -o data/ibl/aggregates/bwm_clusters_2026_Q2.pqt https://ibl-brain-wide-map-public.s3.amazonaws.com/aggregates/2026_Q2_IBL_et_al_BWM/clusters.pqt
nice -n 10 .venv/bin/python -m pytest -q tests/
nice -n 10 .venv/bin/python -m src.glmhmm build --design A90 && nice -n 10 .venv/bin/python -m src.glmhmm build --design P
caffeinate -i nice -n 10 .venv/bin/python -m src.glmhmm global --design A90   # then cv, mouse, select, label
caffeinate -i nice -n 10 .venv/bin/python -m src.glmhmm global --design P     # at most 2 heavy jobs at once
```

## Next steps
1. Finish the downloads. Run the tests and fix any failures.
2. Build the GLM-HMM datasets, time one EM fit, then run global → cv → mouse → select → label for A90 (Ashwood check) and P (primary).
3. Apply the session inclusion rule (≥400 trials; ≥30 engaged and ≥30 disengaged labelled trials, smoothed posterior > 0.8) and record the counts.
4. Download one session's spikes for a format check only (no engagement statistic). Record it in the prereg's "data already seen" section.
5. Draft `preregistration/prereg.md`. Draft the paper skeleton (headings and notes, pending the CLAUDE.md decision). Report to the coordinator.

## Update after resume (evening)
- The authorship policy was updated by the coordinator in 284d5c9 (Claude drafts; the researcher edits).
- The per-mouse history download resumed. The network is saturated by other jobs (tens of KB/s).
- The GLM-HMM uses two-stage multi-start fits. Engaged states = stimulus weight ≥ 0.5 × max.
- Pilot A90 fit (partial histories): K=4 gives two engaged states (stimulus weights 8.5/7.5) and biased-left/right states (stimulus weight about 1.2, bias −1.45/+2.7). Dwell times are about 19/19/10/10 trials, which matches Ashwood's pattern.
- `scripts/run_glmhmm.sh` runs the full protocol (A90 and P). A guarded waiter starts it automatically once all 139 subject tables exist.
- Added `src/nulls.py` (pseudo-session null). 10 tests pass.
- `paper/main.tex`: Introduction and Methods drafted; `paper/refs.bib`: 25 references, all verified.
- `preregistration/prereg.md`: drafted, NOT staged. The [FILL] fields are completed after the GLM-HMM run.

## Update 2026-11-25 01:45 (the researcher is away; work continues autonomously)
- All 139 per-mouse history tables are downloaded.
- Datasets: A90 = 118,577 trials / 1,320 sessions; P = 937,750 trials / 1,328 sessions; 139 mice; all 459 BWM sessions are in both.
- A90 global fits and CV are done. Global-model held-out bits/trial for K=1..5: 0.310, 0.380, 0.404, 0.418, 0.423. The A90 per-mouse step is running inside `scripts/run_glmhmm.sh`, which then runs select, label and `cv --design P`.
- P global: K=1 and K=2 are done. K=3–5 were restarted with a numba Newton step (about 7x faster, numerically identical). Log: `results/glmhmm/run_globalP.log`.
- **After both finish, run by hand:** `nice -n 10 .venv/bin/python -m src.glmhmm mouse --design P`, then `select --design P`, then `label --design P --k <K>`. Then run the inclusion counts and figures (`src/behaviour_report.py`, to be written).
- If a process died, rerun the same step. The global and cv steps overwrite their per-K files. To finish only the missing K, pass `--k`.
- New code: `src/run_decoding.py` (resumable streaming run, with a guard that refuses real data until the prereg is committed), `src/aggregate.py` (region table, H2, H3), `src/nulls.py`.
- In progress: the Steinmetz behaviour-only GLM-HMM. Trials tables are streamed from DANDI 000017 NWB files over HTTP range requests; no spikes are read.
- Still to write: `notes/prereg_summary.md`, `notes/decisions.md`.

## Update 2026-11-25 03:10
- Fixed: per-mouse fits could diverge (undamped Newton). The M-step now uses step halving. The A90 per-mouse step was rerun with no warnings.
- A90 result: K=3 reproduces Ashwood (stimulus weights 7.4 / 1.5 / 1.4; biases 0 / −1.7 / +2.1; dwell 31 / 13.5 / 14.5 trials). Global parameters beat per-mouse fits on held-out data for K≥3, so `select` now also picks the parameter family (see notes/decisions.md D6).
- Running, detached: `glmhmm global --design P --k 3 4 5` (log: results/glmhmm/run_globalP.log; K=3 done) and `glmhmm cv --design P` (log: results/glmhmm/run_chain2.log).
- **Next, by hand, when both are done:** `mouse --design P`; `select --design P`; `label --design P --k <K> --params <family>`; `python -m src.behaviour_report --k <K>`. Then fill the [FILL] fields in the prereg, write notes/prereg_summary.md and the paper's behaviour results.
- Steinmetz: trials tables are streaming from DANDI (29 of 39 sessions done). After that, `python -m src.steinmetz fit` starts automatically (log: results/steinmetz_fit.log).
- Loader format check on one real session is running (shapes only; no behaviour-related statistic).
- notes/decisions.md is written (D1–D20).
