# Cello kinetic corpus

Characterized Hill kinetics for genetic parts, extracted from Cello's user constraint files. Each gate and
input sensor is characterized by a Hill response function `y = ymin + (ymax - ymin) / (1 + (x/K)ⁿ)`. Quiver
uses these `(ymax, ymin, K, n)` values to set the realizable parameter ranges its tuning clamp enforces
(`quiver/generate/realizable.py`).

## Provenance

- Source: [CIDARLAB/Cello-UCF](https://github.com/CIDARLAB/Cello-UCF) @ `222827d30a28730404f0601d64f63436354fee9d`, v2 UCF files.
- `ingest_cello.py` walks every organism UCF (`files/v2/ucf`) and input file (`files/v2/input`), joins each `gates`/`input_sensors` entry to its `models` Hill parameters, and writes `cello_kinetics.csv`.
- Regenerate: `git clone https://github.com/CIDARLAB/Cello-UCF && uv run --no-project python ingest_cello.py Cello-UCF`

## Contents

- 84 rows: 63 repressor gates and 21 input sensors, across 3 organisms (`Eco`, `SC`, `Bth`).
- 28 distinct regulators, in 2 families: TetR-family protein repressors and CRISPRi (dCas9/gRNA).
- 14 distinct small-molecule sensors (aTc, IPTG, arabinose, cumate, vanillic acid, naringenin, DAPG, AHLs).
- Hill-parameter ranges over gates: `K` from 0.01 to 1.25 (median 0.13), `n` from 1.15 to 4.48 (median 2.72), dynamic range from 8.9 to 1.6×10³ (median 155). Sensor dynamic range from 12 to 3.4×10³ (median 211).
- Values are in RPU (relative promoter units); `K` is in input-RPU.

## Files

- `ingest_cello.py`: the ingestion.
- `cello_kinetics.csv`: the corpus, one row per gate or sensor.
