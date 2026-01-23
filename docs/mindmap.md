:::mermaid
graph LR;
    %% Root
    Root((Clinical Data Pipeline<br/>Synthetic Generation))-->Source[Data Source];

    %% Data Source Branch
    Source-->Shen(ShenLab Dataset);
    Shen-->Input(Patient Prompts);
    Shen-->Gold(Gold Standard<br/>'Chosen' Response);

    %% Generation Branch
    Root-->Gen[The Generator<br/>gemini-2-flash];
    Gen-->Persona(Persona:<br/>Unskilled Therapist);
    
    Persona-->Err1(Toxic Positivity);
    Persona-->Err2(Premature Advice);
    Persona-->Err3(Dismissal);

    Gen-->Tech(Robust Engineering);
    Tech-->Retry(Auto-Retry Logic);
    Tech-->Safe(Safety Block Handling);
    Tech-->Len(Strict Length Check);

    %% Weaver Branch (The Core Innovation)
    Root-->Weaver[The Weaver<br/>'Clinical Jury'];
    Weaver-->Ens(Ensemble of 5 Verifiers);
    
    Ens-->V1(📋 Clinical Protocol<br/>Heuristics);
    Ens-->V2(🔗 Dialogue Logic<br/>Cross-Encoder);
    Ens-->V3(🛡️ Safety Guard<br/>Toxic-RoBERTa);
    Ens-->V4(🧠 Correctness<br/>MentalBERT);
    Ens-->V5(❤️ Therapeutic Tone<br/>Emotion-BERT);

    %% Calibration Sub-branch
    Weaver-->Calib[Calibration];
    Calib-->LogReg(Logistic Regression);
    LogReg-->Hybrid(Hybrid Weights<br/>Data Signal + Safety Floor);

    %% Output Branch
    Root-->Output[Final Output];
    Output-->Filter(Scoring & Filtering);
    Filter-->DPO(High-Quality Pairs<br/>For DPO/GRPO);
    Filter-->Audit(Human Validation);
:::