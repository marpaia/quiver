# ECF sigma-factor activator corpus

Characterized **activator** components, the counterpart to Cello's repressor-only gates. Extracytoplasmic
function (ECF) sigma factors activate their cognate promoters, so each row here becomes an `activator`
Component (`role="activator"`), grounding a `hill1` stage with `alpha = [ymin, ymax]` (output high when the
sigma factor is high). Merged with the Cello corpus (`quiver.merge(load_cello(), load_ecf(), name="cello+ecf")`
or `library_from_name("cello+ecf")`), these let a design alternate activation and repression instead of
forming repressor chains only.

The loader is `quiver.load_ecf()` (a thin wrapper over the generic `quiver.load_flat_csv`). It reads
`ecf_activators.csv` in this directory and returns an **empty library if that file is absent**, so an
unpopulated corpus is a no-op rather than an error.

## Schema (`ecf_activators.csv`)

One row per characterized sigma:promoter switch. Columns:

| column | required | meaning |
| --- | --- | --- |
| `name` | yes | display/part name carried into LOICA/SBOL export |
| `regulator` | no (defaults to `name`) | the sigma factor — the **orthogonality key** (no regulator twice in one circuit) |
| `role` | no (defaults to `activator`) | `activator` / `repressor` / `sensor` |
| `ymax`, `ymin` | either this pair… | ON and OFF expression levels (RPU) |
| `dynamic_range` | …or this | fold activation; when given without `ymax`/`ymin`, `ymin` is set to a nominal OFF level and `ymax = ymin * dynamic_range` |
| `K`, `n` | no | Hill half-max and cooperativity; default when absent |
| `family`, `organism` | no | metadata |
| `quality` | no (defaults to `literature`) | `measured` / `fitted` / `literature` |
| `source` | no | citation/accession, recorded in provenance |

Only the **dynamic range** is treated as a reliable cross-corpus quantity (matching quiver's realizable clamp,
which trusts dimensionless quantities only); absolute levels derived from a fold-change are nominal, which is
why fold-change-only rows are tagged `quality="literature"`.

## Populating it (provenance)

Real ECF characterization lives in paper supplements, not a clean machine-readable feed. To populate this
corpus, drop a `ecf_activators.csv` here built from a characterized source, e.g.:

- **Rhodius et al. 2013**, *Design of orthogonal genetic switches based on a crosstalk map of σs, anti-σs, and
  promoters* (Mol Syst Biol) — ~20 orthogonal ECF σ:promoter activating pairs with activities.
- **Bervoets et al. 2018**, *A sigma factor toolbox for orthogonal gene expression in E. coli* (Nucleic Acids
  Research) — characterized dynamic ranges.

Transcribe the reported ON/OFF levels or fold-changes into the schema above (do not estimate values that were
not measured). A deterministic `ingest_ecf.py` that parses a specific supplement file can live alongside this
README, mirroring `data/cello/ingest_cello.py`.
