# src/evaluation/evaluator.py
#basic evaluator (SAM & RMSE)
import torch
import numpy as np
from torchmetrics.functional import spectral_angle_mapper

class HSIGeneEvaluator:
    def __init__(self):
        self.metrics_history = []

    def evaluate_spectral_fidelity(self, generated_hsi, reference_hsi):
        # generated_hsi, reference_hsi: (B,C,H,W) or (C,H,W)
        if generated_hsi.dim() == 3:
            generated_hsi = generated_hsi.unsqueeze(0)
        if reference_hsi.dim() == 3:
            reference_hsi = reference_hsi.unsqueeze(0)
        sam = spectral_angle_mapper(generated_hsi, reference_hsi, reduction='mean').item()
        rmse = torch.sqrt(torch.mean((generated_hsi - reference_hsi)**2)).item()
        results = {'sam': sam, 'rmse': rmse}
        self.metrics_history.append(results)
        return results

    def save_evaluation_report(self, results: dict, save_path: str):
        import json
        # Ensure serializable
        for k,v in results.items():
            try:
                json.dumps(v)
            except:
                results[k] = float(v)
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)
        print("Evaluation report saved to:", save_path)
