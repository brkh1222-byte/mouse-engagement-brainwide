# CLAUDE.md

This is a research project by a 17-year-old student. It maps which brain regions' pre-stimulus activity predicts latent engagement state (GLM-HMM) in the IBL Brain-Wide Map, and tests replication in Steinmetz et al. 2019. Read PROPOSAL.md first.

- **Pre-registration:** `preregistration/prereg.md` must be committed before any neural decoding. Never edit it after commit; add dated amendments.
- **Data:** IBL via ONE-api with the public credentials (or the AWS open bucket); Steinmetz via DANDI 000017 or Figshare. Cached data under `data/` are git-ignored.
- **Statistics:** always control for block prior and movement. Use session-permutation nulls (IBL method), not naive trial shuffles. Apply FDR across regions.
- **Integrity:** the student must understand and write the paper themselves. Explain methods when implementing them. Verify every citation.
