# Tuned business entity resolution

This local pipeline uses only the supplied challenge data. Run commands from this directory. Python 3.12 was used.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe src/pipeline.py --data-root "PATH/TO/student_resource" --cache cache --artifacts artifacts --output ../../output --stage all --workers 8
```

To use the supplied fitted model, change `--stage all` to `--stage predict`. The first run builds disk-backed training/test indexes as needed. Index files are large; reuse the same cache directory on subsequent runs. Full-test inference is substantially longer than fitting the classifier.

In the original local workspace, the existing environment and caches can be reused in VS Code's PowerShell terminal:

```powershell
$workspace = "C:\Users\ragha\Documents\Codex\2026-09-25\files-mentioned-by-the-user-6ab10eb3b23ba"
$pythonExe = "$workspace\work\venv\Scripts\python.exe"
$dataRoot = "C:\Users\ragha\Downloads\6ab10eb3b23ba_student_resource\student_resource"
& $pythonExe src/pipeline.py --data-root $dataRoot --cache "$workspace\work\cache" --artifacts artifacts --output "$workspace\outputs\vscode_run_v2" --stage predict --workers 8
```

## Format

Both output files have exactly two tab-separated columns. ID lists in the second column use commas. Empty second cells are valid singleton predictions. Every test reference must appear, including France, as required by pages 1–2 of the provided PDF. Do not replace commas with tabs or add spaces to align columns.

## Validation

```powershell
python src/validate_streaming.py --test-dir "PATH/TO/student_resource/dataset/test" --output ../../output --report artifacts/validation_streaming.json
python src/validate_official_batched.py --test-dir "PATH/TO/student_resource/dataset/test" --output ../../output --report artifacts/validation_official.json
python src/test_pipeline.py
```

The streaming validator checks every row, complete source-1 coverage, unique IDs, target-ID existence, and match containment. The unchanged official validator is additionally applied in bounded-memory batches. The official whole-file validator can consume substantial memory on the complete candidate file.

## Training and evaluation

The protocol reconstructs and excludes the previous 30,000-entity sample and the 2,000 development entities. A new fixed-seed sample of 40,000 entities is split into 32,000 fit, 4,000 tuning, and 4,000 holdout entities. All supplied training target records are searchable. Ground-truth pairs are never injected into candidate sets.

Training uses broader retrieval buckets of at most 5,000 records to expose the classifier to varied negatives. Three LightGBM configurations are compared on the tuning set. The chosen classifier is refitted on the 36,000 fit+tuning entities. The 4,000 holdout entities remain excluded from fitting.

Inference uses buckets of at most 2,000 records, selected on development data for a speed/recall tradeoff. Its probability threshold is calibrated with the original 32,000-fit model on the 4,000 tuning entities. The final reported holdout metric evaluates the delivered 36,000-fit model with this inference policy. Metrics include the earlier broad-policy check separately; that auxiliary result does not select the inference policy or threshold.

Candidate retrieval combines seven exact/prefix blocking indexes and the four least-frequent eligible name/address token postings per field. The final shortlist is a deduplicated union of the best 80 candidates by combined similarity, best 40 by name, and best 40 by address (at most 160). This protects records with missing business names. The exact shortlist scored by the classifier is written to `candidate_pairs.tsv`.

The classifier has 45 features: 29 original text/numeric/source features plus 16 features for normalized numeric overlap, house-number agreement, missing fields, compact text, and number-normalized similarity. IDs are only used for linkage and deterministic ties; their numeric values are not predictive features. Country labels remain an open set.

Reported performance is macro F0.5, the challenge metric, rather than ordinary classification accuracy. Test/leaderboard performance is unknown. France has no labeled training examples and therefore no local labeled country benchmark.

## Resume

Add `--resume` only after an interrupted prediction with the exact same model, threshold, data, and candidate policy. The pipeline verifies aligned output prefixes, trims incomplete trailing rows, and continues in source order. Never resume an original-model output with the tuned model.

The MIT license covers the original source and fitted tree model. LightGBM's MIT license is included. No external business identity services, lookup, geocoding, or data augmentation are used.
