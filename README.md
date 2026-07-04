# MCPC-RAG: Multi-Source Patient Context RAG for Medication Recommendation

This repository implements a multi-source patient-context retrieval-augmented generation pipeline for medication recommendation. The system builds a unified clinical knowledge base from MIMIC-IV, MIMIC-III, and eICU patient records, retrieves clinically similar historical cases, aggregates their prescription evidence into medication candidates, and constrains an LLM to select medications and generate evidence-grounded recommendation reports.

The current main entry point is:

```text
main.py
```

It supports FAISS retrieval, field-aware reranking, medication candidate construction, candidate-constrained LLM selection, and optional report generation.

## 1. Pipeline Overview

The full workflow contains four major steps.

### 1.1 Multi-Source Patient Context Knowledge Base Construction

The project converts patient records from MIMIC-IV, MIMIC-III, and eICU into a shared admission/stay-level schema. Each row represents one patient visit and contains:

| Field | Meaning |
|---|---|
| `source` | Dataset source, such as `mimic_iv`, `mimic_iii`, or `eicu` |
| `subject_id` | Patient identifier |
| `hadm_id` | Admission/stay identifier |
| `diagnoses` | Diagnosis names or diagnosis-derived text |
| `medications` | Ground-truth medication labels, represented as ATC codes |
| `procedures` | Procedures, operations, ventilation, consultations, or treatment procedures |
| `lab_summary` | Summarized laboratory profiles, such as creatinine, glucose, potassium, WBC, etc. |
| `vital_summary` | Summarized vital signs, mainly for eICU |
| `nurse_charting_summary` | Optional nursing charting summaries, mainly for eICU variants |
| `clinical_state` | Unified patient context text used for semantic retrieval |
| `hospitalid` | Hospital identifier when available |

The `clinical_state` text is the main retrieval representation. It may include demographics, diagnoses, procedures, laboratory profiles, vital signs, medication history, and treatment-related context.

### 1.2 Similar Case Retrieval and Field-Aware Reranking

For a target patient, the system first retrieves a high-recall set of semantically similar historical cases from the FAISS vector database. When field-aware retrieval is enabled, the retrieved cases are reranked using:

- semantic similarity from FAISS,
- diagnosis similarity,
- procedure similarity,
- laboratory-profile similarity,
- vital-sign similarity,
- nursing-record similarity,
- demographic similarity,
- treatment-context similarity.

The final top-k cases are used as evidence for medication candidate construction.

### 1.3 Case Evidence Aggregation and Medication Candidate Construction

The medications prescribed in retrieved similar cases are aggregated into a candidate medication set. Candidate scores are computed from:

- active or recent medication history,
- case similarity score,
- retrieved-case rank,
- medication frequency across retrieved cases,
- optional RAG tendency evidence,
- optional initial draft medications.

The ranked candidate list becomes the evidence-based search space for the LLM.

### 1.4 Medication Recommendation with Candidate-Constrained LLM

The LLM is constrained to select medications only from the ranked candidate set. This reduces unsupported generation and makes recommendations traceable to retrieved cases and prescription evidence. The system can also generate a clinical recommendation report explaining the selected medications and their evidence.

## 2. Repository Structure

```text
.
+-- all-MiniLM-L6-v2/                         # Local sentence-transformer embedding model
+-- data/                                     # Processed datasets and optional raw data
|   +-- raw/                                  # Raw MIMIC/eICU tables should be placed here
|   +-- mimic_train.csv
|   +-- mimic_test.csv
|   +-- mimic_train_clinical_state.csv
|   +-- eicu_train_clinical_state.csv
|   +-- mimiciii_train_clinical_state.csv
|   +-- mimiciv_eicu_mimiciii_train_clinical_state.csv
+-- script/                                   # Data processing and vector DB scripts
+-- tools/                                    # Analysis, plotting, and utility scripts
+-- vector_db/                                # FAISS vector indexes
+-- output/                                   # Experiment results and analysis outputs
+-- prompt.py                                 # LLM prompts and output parsing functions
+-- main_candidate_multi_multimodel_field_aware_context.py
+-- main_candidate_multi_multimodel_rich_context.py
+-- requirements.txt
```

## 3. Hardware and Software Requirements

### 3.1 Recommended Hardware

For full-scale experiments with MIMIC-IV, MIMIC-III, and eICU:

- OS: Windows 10/11 or Linux.
- CPU: 8 cores or more recommended.
- RAM: 32 GB minimum; 64 GB or more recommended for large CSV preprocessing.
- Disk: at least 100 GB free space for raw tables, processed CSVs, FAISS indexes, and outputs.
- GPU: optional but recommended.
  - Embedding can run on CPU, but GPU speeds up sentence-transformer encoding.
  - Local LLM inference through Ollama benefits from an NVIDIA GPU with sufficient VRAM.

For small smoke tests:

- 16 GB RAM is usually sufficient.
- CPU-only execution is possible if using an external OpenAI-compatible API.

### 3.2 Software

- Python 3.9 or 3.10 is recommended.
- Conda is recommended for environment management.
- FAISS CPU version is used by default.
- A local copy of `sentence-transformers/all-MiniLM-L6-v2` should be placed at:

```text
all-MiniLM-L6-v2/
```

- LLM backend:
  - OpenAI-compatible API, such as GPT-5.5 through `BITIDEA_API_KEY`, or
  - Ollama local models, such as `qwen3:8b`.

## 4. Environment Setup

Create and activate a conda environment:

```powershell
conda create -n pace python=3.9 -y
conda activate pace
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install sentence-transformers
```

If you want GPU acceleration for embeddings, install a CUDA-compatible PyTorch build before installing `sentence-transformers`. For example, for CUDA 11.8:

```powershell
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 torchaudio==2.1.2+cu118 `
  --index-url https://download.pytorch.org/whl/cu118
```

Verify the environment:

```powershell
python -c "import pandas, faiss, sentence_transformers, langchain_core, langchain_community; print('environment ok')"
python -c "from sentence_transformers import SentenceTransformer; m=SentenceTransformer('all-MiniLM-L6-v2'); print(m.get_sentence_embedding_dimension())"
```

Expected embedding dimension:

```text
384
```

## 5. LLM Configuration

### 5.1 OpenAI-Compatible API

The default OpenAI-compatible model name in the main scripts is `gpt-5.5`. Set your API key:

```powershell
$env:BITIDEA_API_KEY="your_api_key_here"
```

You can override the API endpoint and model in the command line:

```powershell
--openai_base_url "https://api.bitidea.cn/v1/chat/completions" `
--openai_api_key_env BITIDEA_API_KEY `
--generator_model gpt-5.5 `
--verifier_model gpt-5.5
```

### 5.2 Ollama Local Model

If using Ollama, make sure the Ollama service is running and the model is available:

```powershell
ollama pull qwen3:8b
ollama serve
```

Then use `--llm_provider ollama` or the corresponding provider argument if configured in your run command.

## 6. Data Preparation

### 6.1 MIMIC-IV

Place MIMIC-IV raw tables under:

```text
data/raw/
```

Commonly required files include:

```text
admissions.csv
diagnoses_icd.csv
d_icd_diagnoses.csv
procedures_icd.csv
d_icd_procedures.csv
prescriptions.csv
labevents.csv
d_labitems.csv
demo_atc_mapping.csv
```

Build train/test CSVs from raw MIMIC-IV tables:

```powershell
python script/build_mimic_train_test_from_tables.py `
  --admissions_csv data/raw/admissions.csv `
  --diagnosis_icd_csv data/raw/diagnoses_icd.csv `
  --d_icd_diagnoses_csv data/raw/d_icd_diagnoses.csv `
  --prescriptions_csv data/raw/prescriptions.csv `
  --demo_atc_mapping_csv data/raw/demo_atc_mapping.csv `
  --output_train_csv data/mimic_train.csv `
  --output_test_csv data/mimic_test.csv `
  --output_full_csv data/mimic_full_admissions.csv
```

Build clinical-state enhanced MIMIC-IV CSVs:

```powershell
python script/build_mimic_clinical_state_dataset.py `
  --input_csvs data/mimic_train.csv data/mimic_test.csv `
  --output_csvs data/mimic_train_clinical_state.csv data/mimic_test_clinical_state.csv `
  --procedures_csv data/raw/procedures_icd.csv `
  --d_icd_procedures_csv data/raw/d_icd_procedures.csv `
  --labevents_csv data/raw/labevents.csv `
  --d_labitems_csv data/raw/d_labitems.csv
```

### 6.2 eICU

Place eICU raw tables under a local directory, for example:

```text
data/raw/eicu/
```

Build eICU clinical-state CSV:

```powershell
python script/build_eicu_clinical_state_dataset.py `
  --eicu_dir data/raw/eicu `
  --output_csv data/eicu_clinical_state.csv
```

Split into train/test:

```powershell
python script/split_eicu_clinical_state.py `
  --input_csv data/eicu_clinical_state.csv `
  --output_train_csv data/eicu_train_clinical_state.csv `
  --output_test_csv data/eicu_test_clinical_state.csv
```

If nursing charting is used, build or augment the nurse-charting version with:

```powershell
python script/augment_eicu_with_nurse_charting.py
```

### 6.3 MIMIC-III

Place MIMIC-III raw tables under a local directory, for example:

```text
data/raw/mimiciii/
```

Build MIMIC-III clinical-state train/test CSVs:

```powershell
python script/build_mimiciii_clinical_state_dataset.py `
  --mimiciii_dir data/raw/mimiciii `
  --ndc_atc_mapping_csv data/raw/demo_atc_mapping.csv `
  --output_full_csv data/mimiciii_clinical_state.csv `
  --output_train_csv data/mimiciii_train_clinical_state.csv `
  --output_test_csv data/mimiciii_test_clinical_state.csv
```

## 7. Build the Multi-Source Knowledge Base

Combine training data from MIMIC-IV, eICU, and MIMIC-III:

```powershell
python script/combine_clinical_state_sources.py `
  --input_csvs data/mimic_train_clinical_state.csv data/eicu_train_clinical_state.csv data/mimiciii_train_clinical_state.csv `
  --source_names mimic_iv eicu mimic_iii `
  --output_csv data/mimiciv_eicu_mimiciii_train_clinical_state.csv
```

If using the eICU nurse-charting version:

```powershell
python script/combine_clinical_state_sources.py `
  --input_csvs data/mimic_train_clinical_state.csv data/eicu_train_clinical_state_nurse.csv data/mimiciii_train_clinical_state.csv `
  --source_names mimic_iv eicu mimic_iii `
  --output_csv data/mimiciv_eicu_nurse_mimiciii_train_clinical_state.csv
```

## 8. Build the FAISS Vector Database

Build a FAISS index from the multi-source clinical-state knowledge base:

```powershell
python script/build_mixed_clinical_state_vector_db.py `
  --input_csv data/mimiciv_eicu_mimiciii_train_clinical_state.csv `
  --output_dir vector_db/mimiciv_eicu_mimiciii_train_clinical_state_faiss_index `
  --text_col clinical_state `
  --model_path all-MiniLM-L6-v2 `
  --index_name faiss_mimiciv_eicu_mimiciii_train_clinical_state_index
```

For the nurse-charting version:

```powershell
python script/build_mixed_clinical_state_vector_db.py `
  --input_csv data/mimiciv_eicu_nurse_mimiciii_train_clinical_state.csv `
  --output_dir vector_db/mimiciv_eicu_nurse_mimiciii_train_clinical_state_faiss_index `
  --text_col clinical_state `
  --model_path all-MiniLM-L6-v2 `
  --index_name faiss_mimiciv_eicu_nurse_mimiciii_train_clinical_state_index
```

## 9. Run Medication Recommendation

### 9.1 Recommended Field-Aware Pipeline

Run on MIMIC-IV test data:

```powershell
python main_candidate_multi_multimodel_field_aware_context.py `
  --data_path data/mimic_test_clinical_state.csv `
  --vector_db vector_db/mimiciv_eicu_mimiciii_train_clinical_state_faiss_index `
  --rag_index_name faiss_mimiciv_eicu_mimiciii_train_clinical_state_index `
  --output_file output/one_step_rag/field_aware_mimiciv_test.json `
  --generator_model gpt-5.5 `
  --verifier_model gpt-5.5 `
  --retrieve_patients 7 `
  --field_aware_stage1_k 50 `
  --selection_strategy vote_llm_select `
  --vote_top_n 20 `
  --min_final_meds 2 `
  --max_final_meds 12 `
  --disable_rag_tendency
```

Run on eICU test data:

```powershell
python main_candidate_multi_multimodel_field_aware_context.py `
  --data_path data/eicu_test_clinical_state.csv `
  --vector_db vector_db/mimiciv_eicu_mimiciii_train_clinical_state_faiss_index `
  --rag_index_name faiss_mimiciv_eicu_mimiciii_train_clinical_state_index `
  --output_file output/one_step_rag/field_aware_eicu_test.json `
  --generator_model gpt-5.5 `
  --verifier_model gpt-5.5 `
  --retrieve_patients 7 `
  --field_aware_stage1_k 50 `
  --selection_strategy vote_llm_select `
  --vote_top_n 20 `
  --min_final_meds 2 `
  --max_final_meds 12 `
  --disable_rag_tendency
```

For a quick smoke test, add:

```powershell
--max_samples 5
```

### 9.2 Important Runtime Arguments

| Argument | Meaning |
|---|---|
| `--data_path` | Test CSV to evaluate |
| `--vector_db` | FAISS vector database directory |
| `--rag_index_name` | FAISS index name used when building the vector DB |
| `--output_file` | JSON result path |
| `--generator_model` | Model for focus extraction, tendency analysis, and report generation |
| `--verifier_model` | Model for candidate-constrained medication selection |
| `--retrieve_patients` | Final top-k retrieved cases |
| `--field_aware_stage1_k` | Number of FAISS candidates retrieved before field-aware reranking |
| `--enable_field_aware_retrieval` | Enable field-aware reranking; enabled by default in the field-aware script |
| `--no-enable_field_aware_retrieval` | Disable field-aware reranking and use FAISS-only ranking |
| `--threshold` | Minimum retrieval score threshold |
| `--selection_strategy` | Final recommendation strategy: `rag_vote`, `vote_llm`, `vote_llm_prune`, or `vote_llm_select` |
| `--vote_top_n` | Number of top ranked candidates shown to the LLM or used by RAG vote |
| `--min_final_meds` | Minimum final recommendation size |
| `--max_final_meds` | Maximum final recommendation size |
| `--disable_doctor_summary` | Skip clinical report generation |
| `--disable_rag_tendency` | Skip optional RAG tendency analysis to reduce LLM calls |
| `--resume_existing` | Resume from an existing output JSON |

### 9.3 FAISS-Only Rich Context Pipeline

The older rich-context entry point does not use field-aware reranking. It performs FAISS retrieval, score filtering, duplicate removal, and top-k selection:

```powershell
python main_candidate_multi_multimodel_rich_context.py `
  --data_path data/mimic_test_clinical_state.csv `
  --vector_db vector_db/mimiciv_eicu_mimiciii_train_clinical_state_faiss_index `
  --rag_index_name faiss_mimiciv_eicu_mimiciii_train_clinical_state_index `
  --output_file output/one_step_rag/rich_context_mimiciv_test.json `
  --generator_model gpt-5.5 `
  --verifier_model gpt-5.5 `
  --retrieve_patients 7 `
  --selection_strategy vote_llm_select `
  --vote_top_n 20 `
  --disable_rag_tendency
```

## 10. Output Format

Each experiment writes a JSON file with two top-level fields:

```text
summary
results
```

`summary` contains aggregate metrics:

- `macro_f1`
- `macro_precision`
- `macro_recall`
- `macro_jaccard`
- `avg_candidate_recall_upper_bound`
- `avg_candidate_size`
- `avg_final_size`
- `avg_ground_truth_size`
- `avg_runtime_seconds_per_patient`
- `avg_llm_calls_per_patient`

Each item in `results` contains patient-level information:

- `patient_id`
- `diagnoses`
- `ground_truth` and `ground_truth_list`
- `active_history` and `recent_visit_history`
- `rag_patients`: retrieved similar cases
- `candidate_medications`
- `rag_logging`: search queries, selected source counts, ranked candidate medications, candidate vote scores
- `final_answer`: selected medications and model reasoning
- `doctor_summary`: optional evidence-grounded clinical recommendation report
- `metrics`: patient-level F1, Precision, Recall, Jaccard, candidate recall upper bound, and set sizes

## 11. Evaluation and Analysis Utilities

Compute PRAUC from RAG candidate vote scores:

```powershell
python tools/compute_rag_prauc.py output/one_step_rag/field_aware_eicu_test.json
```

Generate eICU top-k sensitivity plots from saved results:

```powershell
python tools/plot_eicu_topk_sensitivity.py
```

Generate assumed top-k sensitivity figures used for drafting:

```powershell
python tools/plot_assumed_eicu_topk_sensitivity.py
```

Analyze result JSON files:

```powershell
python script/analyze_rag_result_json.py output/one_step_rag/field_aware_eicu_test.json
```

## 12. Notes on Reproducibility

- Use patient-level splits to avoid train/test leakage.
- Do not include target test admissions in the FAISS knowledge base.
- `clinical_state` should not include ground-truth medications when used as the query if label leakage is a concern.
- The main scripts support excluding medication history from query text through the default clinical-state query handling.
- Set `--seed 42` for stable LLM/API behavior when supported by the backend.
- Results from local LLMs and external LLM APIs may vary across model versions and providers.

## 13. Common Issues

### Missing local embedding model

If `SentenceTransformer` cannot load `all-MiniLM-L6-v2`, download `sentence-transformers/all-MiniLM-L6-v2` and place it in:

```text
all-MiniLM-L6-v2/
```

### FAISS index name mismatch

The `--rag_index_name` used during inference must match the `--index_name` used when building the vector DB.

### Out-of-memory during preprocessing

Large MIMIC/eICU tables can be memory intensive. Use chunked scripts where available and ensure enough disk/RAM is available.

### Slow LLM calls

Use the following options for faster debugging:

```powershell
--max_samples 5 --disable_rag_tendency --disable_doctor_summary
```

### Need pure RAG vote baseline

Use:

```powershell
--selection_strategy rag_vote
```

This skips the candidate-constrained LLM selection stage and recommends the top ranked RAG-vote candidates.
