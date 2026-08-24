import torch
import torch.nn.functional as F


class WeightedEnsemble:
    """
    Weighted probability ensemble for AST, WavLM, and HuBERT.

    Default weights:
        AST   = 0.50
        WavLM = 0.20
        HuBERT = 0.30
    """

    def __init__(
        self,
        ast_model,
        wavlm_model,
        hubert_model,
        ast_weight=0.50,
        wavlm_weight=0.20,
        hubert_weight=0.30,
    ):
        self.ast_model = ast_model
        self.wavlm_model = wavlm_model
        self.hubert_model = hubert_model

        self.ast_weight = ast_weight
        self.wavlm_weight = wavlm_weight
        self.hubert_weight = hubert_weight

    def predict(
        self,
        ast_input,
        wavlm_input,
        hubert_input,
    ):
        """Return the weighted ensemble prediction."""

        with torch.no_grad():
            ast_logits = self.ast_model(ast_input)
            wavlm_logits = self.wavlm_model(wavlm_input)
            hubert_logits = self.hubert_model(hubert_input)

            ast_probs = F.softmax(ast_logits, dim=-1)
            wavlm_probs = F.softmax(wavlm_logits, dim=-1)
            hubert_probs = F.softmax(hubert_logits, dim=-1)

            ensemble_probs = (
                self.ast_weight * ast_probs
                + self.wavlm_weight * wavlm_probs
                + self.hubert_weight * hubert_probs
            )

            predictions = torch.argmax(
                ensemble_probs,
                dim=-1,
            )

        return predictions, ensemble_probs