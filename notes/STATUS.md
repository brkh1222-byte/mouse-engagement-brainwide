# STATUS (checkpoint 2026-11-24, before the Mac shut down)

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
