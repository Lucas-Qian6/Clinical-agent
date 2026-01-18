import numpy as np
import torch
from transformers import pipeline, AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from typing import List, Dict
import torch.nn.functional as F
import re

class WeakVerifier:
    """Base class for a weak verifier in the Weaver ensemble."""
    def __init__(self, name: str, weight: float = 1.0):
        self.name = name
        self.weight = weight

    def score(self, text: str, reference: str = "", context: str = "") -> float:
        raise NotImplementedError

class ClinicalCorrectnessVerifier(WeakVerifier):
    """
    FORMERLY: PsychBertVerifier
    ROLE: Checks Semantic Consistency with Gold Standard.
    MODEL: mental/mental-bert-base-uncased
    """
    def __init__(self, device=0):
        super().__init__("Clinical_Correctness", weight=1.0)
        model_name = "mental/mental-bert-base-uncased"
        self.device = torch.device(f"cuda:{device}" if torch.cuda.is_available() and device >= 0 else "cpu")

        try:
            print(f"Loading {model_name} for Correctness checks...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name).to(self.device)
            self.model.eval()
            print("   MentalBERT loaded successfully.")

        except Exception as e:
            msg = str(e)
            if "401" in msg or "gated" in msg.lower():
                print(f"❌ AUTH ERROR: {model_name} is a gated model.")
                print("   Please run `huggingface-cli login` or use the updated train script.")
            else:
                print(f"Error loading Clinical Model: {e}")
            self.model = None

    def _get_embedding(self, text: str):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1)

    def score(self, text: str, reference: str = "", context: str = "") -> float:
        if not self.model or not reference:
            return 0.5

        try:
            emb_response = self._get_embedding(text)
            emb_reference = self._get_embedding(reference)
            cosine_sim = F.cosine_similarity(emb_response, emb_reference).item()
            return max(0.0, (cosine_sim + 1) / 2)
        except Exception as e:
            print(f"   [Correctness Error]: {e}")
            return 0.5

class TherapeuticToneVerifier(WeakVerifier):
    """
    FORMERLY: EmpathyVerifier
    ROLE: Checks for warmth, empathy, and lack of judgment.
    """
    def __init__(self, device=0):
        super().__init__("Therapeutic_Tone", weight=1.2)
        self.model_name = "bhadresh-savani/bert-base-uncased-emotion"
        self.device = device if torch.cuda.is_available() and device >= 0 else -1
        self.pipe = pipeline("text-classification", model=self.model_name, device=self.device, top_k=None)

    def score(self, text: str, reference: str = "", context: str = "") -> float:
        try:
            results = self.pipe(text, truncation=True, max_length=512)[0]
            scores = {item['label']: item['score'] for item in results}
            negativity = scores.get('anger', 0) + scores.get('fear', 0) + scores.get('sadness', 0)
            empathy_bonus = scores.get('love', 0) * 0.5
            return max(0.0, min(1.0, 1.0 - negativity + empathy_bonus))
        except:
            return 0.5

class SafetyVerifier(WeakVerifier):
    """
    FORMERLY: SafetyVerifier
    ROLE: The 'Red Line' check.
    """
    def __init__(self, device=0):
        super().__init__("Safety_Guard", weight=2.0)
        self.model_name = "unitary/unbiased-toxic-roberta"
        self.device = device if torch.cuda.is_available() and device >= 0 else -1
        self.pipe = pipeline("text-classification", model=self.model_name, device=self.device, top_k=None)

    def score(self, text: str, reference: str = "", context: str = "") -> float:
        try:
            result = self.pipe(text, top_k=None, truncation=True, max_length=512)
            toxic_sum = sum(l['score'] for l in result if l['score'] > 0.5)
            return max(0.0, 1.0 - toxic_sum)
        except:
            return 1.0

class ClinicalProtocolVerifier(WeakVerifier):
    """
    FORMERLY: HeuristicVerifier
    ROLE: Checks adherence to CBT protocols (Style).
    """
    def __init__(self):
        super().__init__("Clinical_Protocol", weight=1.5)

    def score(self, text: str, reference: str = "", context: str = "") -> float:
        score = 0.8
        text_lower = text.lower()

        if "you should" in text_lower or "why don't you" in text_lower:
            score -= 0.4
        if len(text.split()) < 5:
            score -= 0.5
        if "?" in text:
            score += 0.2

        return max(0.0, min(1.0, score))

class DialogueLogicVerifier(WeakVerifier):
    """
    FORMERLY: RelevanceVerifier
    ROLE: Checks Logical Coherence.
    """
    def __init__(self, device=0):
        super().__init__("Dialogue_Logic", weight=1.3)
        self.model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        self.device = device if torch.cuda.is_available() and device >= 0 else -1
        try:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(self.model_name, device=self.device)
            self.use_pipeline = False
        except:
            self.pipe = pipeline("text-classification", model=self.model_name, device=self.device)
            self.use_pipeline = True

    def score(self, text: str, reference: str = "", context: str = "") -> float:
        if not context:
            return 0.5

        if self.use_pipeline:
            result = self.pipe({"text": context, "text_pair": text}, truncation=True, max_length=512)
            return result['score'] if result['label'] == 'LABEL_1' else 1 - result['score']
        else:
            scores = self.model.predict([(context, text)])
            return float(1 / (1 + np.exp(-scores[0])))

class WeaverEnsemble:
    """
    The Clinical Jury: Aggregates all 5 signals.
    """
    def __init__(self, verifiers: List[WeakVerifier]):
        self.verifiers = verifiers

    def evaluate_pair(self, chosen: str, rejected: str, prompt: str) -> Dict:
        scores_chosen = {}
        scores_rejected = {}
        weighted_sum_c = 0
        weighted_sum_r = 0
        total_weight = 0

        for v in self.verifiers:
            if v.name == "Clinical_Correctness":
                s_c = v.score(chosen, reference=chosen)
                s_r = v.score(rejected, reference=chosen)
            elif v.name == "Dialogue_Logic":
                s_c = v.score(chosen, context=prompt)
                s_r = v.score(rejected, context=prompt)
            else:
                s_c = v.score(chosen)
                s_r = v.score(rejected)

            scores_chosen[v.name] = s_c
            scores_rejected[v.name] = s_r

            weighted_sum_c += s_c * v.weight
            weighted_sum_r += s_r * v.weight
            total_weight += v.weight

        final_c = weighted_sum_c / total_weight
        final_r = weighted_sum_r / total_weight
        margin = final_c - final_r

        return {
            "margin": margin,
            "rejected_score": final_r,
            "chosen_score": final_c,
            "is_valid_pair": margin > 0.15,
            "is_rejected_plausible": 0.2 < final_r < 0.65
        }