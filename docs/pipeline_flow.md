# Pipeline Flow

> RedrobAI · India Runs Data & AI Challenge · Track 1

## End-to-End Pipeline

```mermaid
flowchart TD
    START(["Start"]) --> CHECK{"FAISS artifacts\nexist?"}

    CHECK -->|No| BUILD["python run_pipeline.py --rebuild-index\n(one-time, 15–40 min on CPU)"]
    BUILD --> ARTIFACTS[("artifacts/faiss/\nfaiss.index\ncandidate_lookup.pkl\nembedding_metadata.pkl")]

    CHECK -->|Yes| LOAD["Load FAISS index\n& candidate lookup"]
    ARTIFACTS --> LOAD

    LOAD --> JD["Parse job_description.json\nJobDescriptionParser"]
    JD --> SQ["Build SearchQuery\n(identity · career · skills · combined)"]
    SQ --> FAISS_SEARCH["FAISS top-K retrieval\n~1.8s · 100 candidates"]

    FAISS_SEARCH --> RESOLVE["Resolve each candidate\nfrom candidates.jsonl\n(lazy, cached)"]

    RESOLVE --> CAREER["CareerScorer\n38% · role title · progression\n· industry · responsibilities"]
    RESOLVE --> SKILL["SkillScorer\n34% · required/preferred match\n· proficiency · duration"]
    RESOLVE --> SEM["Semantic score\n18% · cosine similarity"]
    RESOLVE --> BEH["BehaviorScorer\n5% · open-to-work · contacts\n· response rate · notice"]
    RESOLVE --> CON["ConsistencyScorer\n5% · data integrity"]

    CAREER & SKILL & SEM & BEH & CON --> WEIGHTED["Weighted sum\n+ calibration adjustment (±0.08)"]

    WEIGHTED --> SORT["Sort descending by\nweighted_final_score"]

    SORT --> REASON["ReasonGenerator\nDeterministic template selection\nvia candidate_id hash"]

    REASON --> VALIDATE["SubmissionValidator\n100 rows · unique IDs\nunique reasoning · descending scores"]

    VALIDATE -->|Pass| WRITE["Write outputs"]
    VALIDATE -->|Fail| ERROR(["Raise ValidationError"])

    WRITE --> CSV["submission.csv"]
    WRITE --> XLSX["submission.xlsx"]
    WRITE --> JSON["ranking.json"]
    WRITE --> REPORT["pipeline_report.json"]
    WRITE --> META["submission_metadata.yaml"]

    CSV & XLSX & JSON & REPORT & META --> END(["Done · ~20s"])

    style START fill:#6366f1,color:#fff,stroke:none
    style END fill:#22d3ee,color:#0f172a,stroke:none
    style ERROR fill:#f87171,color:#fff,stroke:none
    style CHECK fill:#1e293b,color:#f1f5f9,stroke:#334155
    style ARTIFACTS fill:#0f172a,color:#94a3b8,stroke:#334155
```

---

## Offline Indexing Flow

```mermaid
flowchart LR
    JSONL["candidates.jsonl\n100k lines"] --> PARSE["CandidateParser\nfrom_jsonl_line"]
    PARSE --> DOC["RetrievalDocumentBuilder\nBuild text corpus per candidate"]
    DOC --> EMBED["EmbeddingEngine\nBAAI/bge-base-en-v1.5\nBatch size 32 · CPU/GPU"]
    EMBED --> NORM["L2 Normalize\nembeddings"]
    NORM --> FAISS_BUILD["FAISS IndexFlatIP\nbuild_index(docs)"]
    FAISS_BUILD --> SAVE1["faiss.index\n~307 MB"]
    FAISS_BUILD --> SAVE2["candidate_lookup.pkl\n~1.8 MB · 100k entries"]
    FAISS_BUILD --> SAVE3["embedding_metadata.pkl\n~17 MB · 100k records"]
```

---

## Scoring Stage Detail

```mermaid
flowchart TD
    CTX["ScoringContext\ncandidate · job_description · config"]

    CTX --> C1["CareerScorer"]
    CTX --> C2["SkillScorer"]
    CTX --> C3["BehaviorScorer"]
    CTX --> C4["ConsistencyScorer"]

    C1 -->|ScoreResult| W1["× 0.38"]
    C2 -->|ScoreResult| W2["× 0.34"]
    C3 -->|ScoreResult| W3["× 0.05"]
    C4 -->|ScoreResult| W4["× 0.05"]
    SEM["Semantic similarity"] --> W5["× 0.18"]

    W1 & W2 & W3 & W4 & W5 --> SUM["Raw weighted sum\nclamped to 0–1"]
    SUM --> CAL{"joint tech evidence\nmin(career, skill)"}
    CAL -->|">= 0.60"| BONUS["+bonus up to 0.10"]
    CAL -->|"< 0.40"| PENALTY["-penalty up to 0.08"]
    CAL -->|"0.40–0.60"| ZERO["no adjustment"]
    BONUS & PENALTY & ZERO --> FINAL["weighted_final_score"]
```

---

## ReasonGenerator Logic

```mermaid
flowchart LR
    CID["candidate_id"] --> HASH["SHA-256 hash\ndeterministic seed"]

    SCORE["weighted_final_score"] --> BAND{"Score band"}
    BAND -->|">= 0.82"| EX["excellent opening\n6 templates"]
    BAND -->|">= 0.72"| ST["strong opening\n6 templates"]
    BAND -->|">= 0.60"| TE["technical opening\n6 templates"]
    BAND -->|"< 0.60"| PA["partial opening\n6 templates"]

    EX & ST & TE & PA --> PICK1["pick via hash"]

    EVIDENCE["matched_items + reasons\n+ career descriptions"] --> FAMILY{"Specialism family"}
    FAMILY -->|"recommend..."| RC["recommendation\n5 templates"]
    FAMILY -->|"retrieval..."| RT["retrieval\n5 templates"]
    FAMILY -->|"ranking..."| RK["ranking\n5 templates"]
    FAMILY -->|"production..."| PR["production\n5 templates"]
    FAMILY -->|"platform..."| PL["platform\n5 templates"]

    RC & RT & RK & PR & PL --> PICK2["pick via hash"]

    REQUIRED["required_skills"] --> MATCH["matched skills clause\n6 templates"]
    REQUIRED --> MISSING["uncertainty clause\n8 templates"]

    SIGNALS["RedrobSignals"] --> BEH["behavior clause\n8 templates"]

    FINAL_SCORE["weighted_final_score"] --> TONE{"Tone band"}
    TONE -->|">= 0.75"| HC["high closing\n6 templates"]
    TONE -->|">= 0.60"| MC["medium closing\n6 templates"]
    TONE -->|"< 0.60"| CC["cautious closing\n6 templates"]

    PICK1 & PICK2 & MATCH & MISSING & BEH & HC & MC & CC --> JOIN["Join clauses"]
    JOIN --> REASONING["Final reasoning string\n120–600 chars"]
```

---

## Validation Flow

```mermaid
flowchart TD
    ROWS["List~SubmissionRow~"] --> V1{"Exactly 100 rows?"}
    V1 -->|No| E1["Error: row count"]
    V1 -->|Yes| V2{"Unique candidate IDs?\nformat CAND_XXXXXXX"}
    V2 -->|No| E2["Error: duplicate / bad ID"]
    V2 -->|Yes| V3{"Ranks 1-100\nstrictly sequential?"}
    V3 -->|No| E3["Error: rank gap or duplicate"]
    V3 -->|Yes| V4{"Scores in 0–1\nnon-increasing order?"}
    V4 -->|No| E4["Error: score range / order"]
    V4 -->|Yes| V5{"Unique reasonings?\nmax 1000 chars each?"}
    V5 -->|No| E5["Error: duplicate / too long"]
    V5 -->|Yes| V6{"All IDs exist in\ncandidates.jsonl?"}
    V6 -->|No| E6["Error: unknown candidate"]
    V6 -->|Yes| PASS(["PASS"])

    style PASS fill:#22d3ee,color:#0f172a,stroke:none
    style E1 fill:#f87171,color:#fff,stroke:none
    style E2 fill:#f87171,color:#fff,stroke:none
    style E3 fill:#f87171,color:#fff,stroke:none
    style E4 fill:#f87171,color:#fff,stroke:none
    style E5 fill:#f87171,color:#fff,stroke:none
    style E6 fill:#f87171,color:#fff,stroke:none
```
