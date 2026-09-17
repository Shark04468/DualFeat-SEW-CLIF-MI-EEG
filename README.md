# v1.1.0 revision evidence

See [the new revision archive](revisions/v1.1.0/README.md) and [release v1.1.0](https://github.com/Shark04468/DualFeat-SEW-CLIF-MI-EEG/releases/tag/v1.1.0). Historical root code and v1.0.1 are preserved. New formal trial-level classification predictions/logits are not included. Zenodo v1.1.0 is published at https://doi.org/10.5281/zenodo.22809543. Its ZIP was downloaded and verified byte-identical to the GitHub release asset. The frozen ZIP retains its original pre-deposition status note; this updated metadata provides the current citation. Source: MIT; new results/descriptions: CC BY 4.0.

---

## Historical release documentation

The following describes the historical release, not coverage of the v1.1.0 additions.

# DualFeat-SEW-CLIF for Motor-Imagery EEG

This repository contains the versioned source code, experiment configurations,
tests, and analysis scripts associated with the manuscript:

> Evidence for sample-regime-dependent benefits of complementary leaky integrate-and-fire temporal-state decoding in motor-imagery EEG

The final paper evaluates a structured SEW-CLIF spiking decoder against a
capacity-matched continuous ANN-SEW control using the same frozen ATCNet and
FBCNet representations, fusion path, causal residual topology, label subsets,
and statistical readout. Earlier delay-phase DPC-SNN experiments remain in the
repository as development history; they are not presented as the final paper's
primary contribution.

## Frozen release

- Source-code release: `v1.0.1`
- Reproducibility archive (concept DOI; all versions): `https://doi.org/10.5281/zenodo.21867113`
- Frozen v1.0.0 evidence: `https://doi.org/10.5281/zenodo.21867114`
- Submission-matched v1.0.1 evidence archive: `https://doi.org/10.5281/zenodo.21905851`
- Archive SHA-256: see the SHA-256 manifest in the exact Zenodo version archive

Version `v1.0.1` is a source-maintenance patch. It adds manuscript/figure audit
utilities, restores two historical launch helpers, and includes `manifest.json`
in the V25 training-file integrity set. It does not change the frozen v1.0.0
trial predictions, metrics, checkpoints, or scientific conclusions.

The separate evidence archive contains resolved configurations, selected epoch
records, trial-level predictions and logits, participant-level metrics,
aggregation and statistics scripts, protocol/provenance manifests, environment
information, figures, and a complete SHA-256 manifest.

## Data policy

This repository does not redistribute raw or processed EEG signals, feature
caches, fitted feature standardizers, gain caches, or model checkpoints.
Download the source datasets from their official repositories and run the
provided preprocessing scripts.

- BCI Competition IV 2a/2b: https://bnci-horizon-2020.eu/database/data-sets
- OpenBMI: https://doi.org/10.5524/100542

See `THIRD_PARTY_DATA_AND_LICENSES.md` in the evidence archive for the frozen
data-redistribution decision and dataset-specific notes.

## Installation

Python 3.10 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[all,dev]"
```

The exact audited environment is included in the evidence archive under
`environment/`.

## Final experiment entry points

The paper-facing campaigns are implemented by the following scripts and locked
configuration files:

- BCI2a development and matched controls: `scripts/evaluate_v28_controls_gate.py`
- OpenBMI replication: `scripts/run_v29_openbmi_subject.py`
- Genuinely unseen BCI2b boundary evaluation: `scripts/run_v30_bnci2014_004_subject.py`
- Decoder-label learning curve: `scripts/run_v31_decoder_learning_curve_subject.py`
- BCI2a objective-purity and fusion attribution: `scripts/run_v32_purity_fold.py`
- Cross-dataset zero-penalty extensions: `scripts/run_v33_e29zp_subject.py`,
  `scripts/run_v33_e30zp_subject.py`, and `scripts/run_v33_e31zp_subject.py`

Associated resolved YAML configurations are under `configs/experiments/`.
Commands that require frozen feature representations or trial-level evidence
must be run against the corresponding artifacts from the Zenodo archive.

## Replaying reported statistics

The paper's reported metrics can be recomputed from the released trial-level
predictions and logits without redistributing EEG waveforms. Start with:

```bash
python scripts/aggregate_v33_primary_metrics.py --help
python scripts/aggregate_v32_purity.py --help
python scripts/aggregate_v32_fusion.py --help
python scripts/aggregate_v33_external_zp.py --help
python scripts/aggregate_v33_e31zp.py --help
```

The evidence archive README records its directory layout and integrity checks.

## Scope of the claims

The final evidence supports a bounded, dataset-dependent, decoder-level effect.
It does not establish universal SNN superiority, end-to-end few-shot learning,
measured neuromorphic energy efficiency, or anatomical interpretation of the
earlier learned delay parameters.

## Citation

Citation metadata are provided in `CITATION.cff`. Cite both the associated
article and the exact Zenodo version DOI for the frozen evidence archive.
