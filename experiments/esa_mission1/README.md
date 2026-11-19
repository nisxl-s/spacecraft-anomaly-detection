# ESA Mission1 independent pilot intake

The preregistered scientific design is in [ESA_Pilot_Protocol.md](../../docs/esa-mission1-protocol.md). The pilot uses ESA Anomaly Dataset v2, Mission1 channels 41–46. The official release lists `ESA-Mission1.zip` as 3.8 GB with MD5 `9770ad12ed730238f37c42d5c27ab436`. The code verifies the local archive against that publisher digest before it opens any nested pickle and records its SHA-256. As with any pickle, use only the official publisher archive. The official benchmark's Mission1 preparation script uses training observations through the split instant and test observations after it. [ESA release](https://zenodo.org/records/15237121) · [Official Mission1 preparation](https://github.com/kplabs-pl/ESA-ADB/blob/main/notebooks/data-prep/Mission1_semisupervised_prep_from_raw.py)

## Run the synthetic validation

Requirements are Python 3.11 or newer, NumPy, and pandas 2.0 or newer. The dry run creates its own temporary CSV fixtures and never reads an ESA label or telemetry value.

```bash
python experiments/esa_mission1/dry_run.py --output experiments/esa_mission1/dry_run.json
```

## Ingest the official release once available

Use the local official archive after obtaining it from the publisher. Keep the fit, calibration and test commands at their corresponding stages of the protocol. The test ingestion command should follow the model and threshold freeze.

```bash
python experiments/esa_mission1/ingest.py --official-archive ESA-Mission1.zip --phase fit --output esa_fit_input
python experiments/esa_mission1/ingest.py --official-archive ESA-Mission1.zip --phase calibration --output esa_calibration_input
python experiments/esa_mission1/ingest.py --official-archive ESA-Mission1.zip --phase test --output esa_test_input
```

Each command writes six per-channel NPZ files and a JSON manifest. The arrays contain an hourly timestamp in Unix nanoseconds, the causally aligned channel value, the eligible-target mask, and named phase masks. Later blocks can use preceding observations as model input context. The saved test mask excludes that preceding context from the evaluation exposure.

`annotations.py` validates label/type joins, retains distinct event IDs, and returns the valid support of each anomaly, including disjoint fragments. It applies annotation-based quality censoring after emission, out of the detector state. An independent scoring runner must use the frozen matching rule for disjoint truth support. The supplied fixtures exercise intake and annotation handling. Real model fitting, threshold selection and evaluation remain pending the official archive.

## Fit detectors and write label-free scores

The forecasting stage requires PyTorch in addition to NumPy and pandas. Its synthetic smoke test runs one epoch on each of six generated fixture channels, verifies fit-only model artifacts, and scores synthetic calibration and test blocks without annotations.

```bash
python experiments/esa_mission1/forecast_smoke.py --output experiments/esa_mission1/forecast_smoke.json
```

When the verified archive is present, these commands fit the preregistered models and emit calibration scores without consulting labels.

```bash
python experiments/esa_mission1/forecast.py fit --input esa_fit_input --output esa_fitted
python experiments/esa_mission1/forecast.py calibration --input esa_calibration_input --models esa_fitted --output esa_calibration_scores
```

A complete official calibration runner must still evaluate disjoint event support, apply the duration policy, and freeze multiplier selection. The official test scoring stage is gated until that runner is available. The smoke-test epoch override is accepted only for generated fixture manifests.
