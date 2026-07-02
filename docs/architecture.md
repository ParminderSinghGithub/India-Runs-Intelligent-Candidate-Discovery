# System Architecture

> RedrobAI · India Runs Data & AI Challenge · Track 1

## High-Level Overview

```mermaid
graph TB
    subgraph Offline["Offline Preprocessing (one-time, ~15–40 min)"]
        direction TB
        A[candidates.jsonl<br/>100k profiles] --> B[CandidateParser]
        B --> C[RetrievalDocumentBuilder]
        C --> D[EmbeddingEngine<br/>BAAI/bge-base-en-v1.5<br/>768 dim]
        D --> E[(FAISS IndexFlatIP<br/>100k vectors)]
        E --> F1[faiss.index]
        E --> F2[candidate_lookup.pkl]
        E --> F3[embedding_metadata.pkl]
    end

    subgraph Online["Online Ranking (~20 sec per job)"]
        direction TB
        G[job_description.json] --> H[JobDescriptionParser]
        H --> I[ParsedJob<br/>SearchQuery · CandidateFilters]
        I --> J[Retriever<br/>FAISS top-K search]
        F1 -.->|load once| J
        J --> K[HybridRanker]
        K --> L1[CareerScorer 38%]
        K --> L2[SkillScorer 34%]
        K --> L3[SemanticScore 18%]
        K --> L4[BehaviorScorer 5%]
        K --> L5[ConsistencyScorer 5%]
        L1 & L2 & L3 & L4 & L5 --> M[Calibrated Weighted Sum]
        M --> N[SubmissionGenerator]
        N --> O1[submission.csv]
        N --> O2[ranking.json]
        N --> O3[pipeline_report.json]
        N --> O4[submission.xlsx]
    end

    style Offline fill:#1e293b,stroke:#334155,color:#f1f5f9
    style Online fill:#0f172a,stroke:#334155,color:#f1f5f9
```

---

## Component Architecture

```mermaid
classDiagram
    class CandidateParser {
        +parse_many(path) List~Candidate~
        +from_dict(data) Candidate
        +from_jsonl_line(line) Candidate
    }

    class JobDescriptionParser {
        +parse_from_file(path) ParsedJob
    }

    class EmbeddingEngine {
        +model_name: str
        +batch_size: int
        +load_model()
        +embed_documents(docs) ndarray
        +embed_query(text) ndarray
    }

    class Retriever {
        +force_rebuild: bool
        +load_index()
        +build_index(docs)
        +search(query, k) List~Dict~
        +get_index_size() int
    }

    class HybridRanker {
        +career_scorer: CareerScorer
        +skill_scorer: SkillScorer
        +behavior_scorer: BehaviorScorer
        +consistency_scorer: ConsistencyScorer
        +rank_candidates(job, top_k) List~HybridScoreResult~
        +rank_retrieval_results(job, results) List~HybridScoreResult~
        +calculate_hybrid_score(context) HybridScoreResult
    }

    class CareerScorer {
        +score(context) ScoreResult
    }

    class SkillScorer {
        +score(context) ScoreResult
    }

    class BehaviorScorer {
        +score(context) ScoreResult
    }

    class ConsistencyScorer {
        +score(context) ScoreResult
    }

    class SubmissionGenerator {
        +generate_submission(...) Dict
        +build_rows(...) List~SubmissionRow~
    }

    class SubmissionValidator {
        +validate_rows(rows) List~str~
        +validate_csv_file(path) List~str~
    }

    class ReasonGenerator {
        +generate(candidate, result, job) str
    }

    class XlsxExporter {
        +export(rows, path) Path
    }

    class PipelineValidator {
        +validate() PipelineValidationReport
    }

    HybridRanker --> CareerScorer
    HybridRanker --> SkillScorer
    HybridRanker --> BehaviorScorer
    HybridRanker --> ConsistencyScorer
    HybridRanker --> Retriever
    SubmissionGenerator --> ReasonGenerator
    SubmissionGenerator --> SubmissionValidator
    SubmissionGenerator --> XlsxExporter
```

---

## Data Model

```mermaid
erDiagram
    Candidate {
        string candidate_id
        Profile profile
        List career_history
        List education
        List skills
        RedrobSignals redrob_signals
    }

    Profile {
        string anonymized_name
        string current_title
        string current_company
        string current_industry
        float years_of_experience
        string location
    }

    Career {
        string title
        string company
        string description
        int duration_months
    }

    Skill {
        string name
        string proficiency
        int duration_months
        int endorsements
    }

    RedrobSignals {
        bool open_to_work_flag
        bool verified_email
        bool verified_phone
        float recruiter_response_rate
        int notice_period_days
        float profile_completeness_score
    }

    ParsedJob {
        JobDescription job_description
        SearchQuery search_query
        CandidateFilters candidate_filters
    }

    HybridScoreResult {
        string candidate_id
        float semantic_score
        float career_score
        float skill_score
        float behavior_score
        float consistency_score
        float weighted_final_score
        float confidence
    }

    SubmissionRow {
        string candidate_id
        int rank
        float score
        string reasoning
    }

    Candidate ||--|| Profile : has
    Candidate ||--o{ Career : has
    Candidate ||--o{ Skill : has
    Candidate ||--|| RedrobSignals : has
    HybridScoreResult ||--|| Candidate : scores
    SubmissionRow ||--|| HybridScoreResult : derives_from
```

---

## Scorer Weight Distribution

| Scorer | Weight | Primary Signal |
|---|---|---|
| CareerScorer | **38%** | Role title, progression, industry, responsibilities |
| SkillScorer | **34%** | Required/preferred skill match, proficiency, duration |
| SemanticScore | **18%** | FAISS cosine similarity (BGE-Base-EN-v1.5) |
| BehaviorScorer | **5%** | Open-to-work, verified contacts, response rate, notice |
| ConsistencyScorer | **5%** | Profile data integrity and internal consistency |
| EducationScorer | 0% | Wired up; disabled for this challenge configuration |

### Calibration Adjustment

After the weighted sum, a small calibration term (±0.08) is applied based on
the **weaker** of career and skill evidence:

- joint ≥ 0.60 → bonus up to +0.10 (corroborated technical match)  
- joint < 0.40 → penalty up to −0.08 (weak technical fit)  
- otherwise → 0

This prevents a high semantic score from masking a poor technical fit.

---

## Artifact Inventory

| Artifact | Size | Description |
|---|---|---|
| `artifacts/faiss/faiss.index` | ~307 MB | IndexFlatIP — 768-dim L2-normalized vectors |
| `artifacts/faiss/candidate_lookup.pkl` | ~1.8 MB | `Dict[candidate_id, metadata]` |
| `artifacts/faiss/embedding_metadata.pkl` | ~17 MB | Per-vector metadata (title, experience) |
| `outputs/submission.csv` | ~54 KB | Official submission |
| `outputs/submission.xlsx` | varies | Formatted Excel export |
| `outputs/ranking.json` | ~560 KB | Full score breakdown |
| `outputs/pipeline_report.json` | ~57 KB | Timing, weights, top-10 |
