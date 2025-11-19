# Dev Log

## 2025-11-17
- Initial setup of project
    - Teacher model: SamLowe/roberta-base-go_emotions
    - Dataset: GoEmotions("simplified")
        - Input: sentences
        - Output: 28 emotion probabilities (each probability is independent)
    - Valence score:
        - clinical interpretable look-up table: a fixed valence weight for each emotion v
        - score = p * v for each emotion
    - Student model:
        - Embedding: MPNet (all-mpnet-base-v2)
            - Input: sentences 
            - Output: 784-d vector
        - Train models to predict the valence scores
            - Ridge Regression
            - PLS + Isotonic calibration
            - PLS
            - KNeighborsRegressor
    - If score > 0.5: dangerous; Else: nothing

## 2025-11-18
- Ideas:
    - Need some expert to put label(dangerous/nothing) based on their input sentence. the expert don’t need to assign a correct score.
    - Computing overall distribution of these 28-dim vectors across the clinical dataset
        - Understand emotional landscape of your clinical population.
        - See whether the teacher behaves weirdly
        - Find typical “danger patterns”
        