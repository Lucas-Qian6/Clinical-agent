:::mermaid
graph TD
    subgraph Input
        P["Patient Prompt"] --> M["Clinical Agent (LLM)"]
        P --> Context["Context Analyzer"]
    end

    M -->|Generates| R["Response"]

    subgraph "The Judges"
        direction TB
        
        %% Weaver Branch
        R --> W["Weaver Ensemble"]
        W -->|"5 Weak Learners"| W_Score["Quality Score<br/>(Safety, Logic, Tone)"]
        
        %% Hourglass Branch
        R --> GE["GoEmotions BERT"]
        GE -->|"27 Labels"| UAM["Unified Affective Matrix"]
        
        subgraph "Unified Affective Matrix"
            UAM --> D1["Introspection"]
            UAM --> D2["Temper"]
            UAM --> D3["Attitude"]
            UAM --> D4["Sensitivity"]
        end
        
        D1 & D2 & D3 & D4 --> HG_Score["Clinical Polarity Score"]
    end

    subgraph "Scoring Logic"
        Context -->|"Is Patient Distressed?"| TP_Check{"Toxic Positivity<br/>Check"}
        HG_Score --> TP_Check
        R --> TP_Check
        
        TP_Check --"High Joy + Sad Patient"--> Penalty["Score = -1.0"]
        TP_Check --"Aligned"--> Norm["Normalize Score 0-1"]
        
        W_Score --> Final{"Weighted Sum"}
        Norm --> Final
        Penalty --> Final
    end

    Final -->|"40% Weaver / 60% Hourglass"| Result["Final Therapeutic Score"]

    style W fill:#e1f5fe,stroke:#01579b
    style UAM fill:#fff3e0,stroke:#e65100
    style TP_Check fill:#ffebee,stroke:#b71c1c
:::
:::mermaid
graph LR
    subgraph "Phase 1: Vector Extraction (Calibration)"
        DS["Holdout Dataset"] -->|"Chosen/Rejected"| W_Filter["Weaver Filter"]
        W_Filter -->|"Valid Contrastive Pairs"| Pairs["Good vs Bad Pairs"]
        
        Pairs -->|"Get Hidden States"| H_States["Hidden Layers"]
        H_States -->|"Mean(Good) - Mean(Bad)"| Vector(("Steering Vector"))
        
        style Vector fill:#ccff90,stroke:#33691e,stroke-width:2px
    end

    subgraph "Phase 2: Inference (The Probe)"
        NewP["New Prompt"] --> Agent["Model"]
        Agent -->|"Generate"| NewR["Response"]
        
        NewR -->|"Extract Hidden State"| Internal["Internal Representation"]
        Vector -->|"Cosine Similarity"| Probe["Probe Score"]
        
        Internal --> Probe
    end
    
    Probe -->|"Replaces Weaver"| FastScore["Fast Quality Signal"]
    FastScore -->|"Combine"| HG["Hourglass Check"]
:::