# src/utils/anomaly_insertion.py
#insert synthetic anomalies
import torch
import numpy as np
import random
from typing import Dict, Tuple

class DefenseAnomalyInserter:
    def __init__(self, spectral_library_path: str = None):
        self.anomaly_signatures = self._load_anomaly_signatures()

        self.anomaly_types = {
            'camouflaged_vehicle': {'size_range':(3,8),'abundance_range':(0.3,0.7),'signature_key':'vehicle_metal_paint'},
            'troop_movement': {'size_range':(2,5),'abundance_range':(0.2,0.5),'signature_key':'human_fabric_mix'},
            'temporary_structure': {'size_range':(5,12),'abundance_range':(0.3,0.6),'signature_key':'tarp_metal_mix'}
        }

    def _load_anomaly_signatures(self):
        sigs = {
            'vehicle_metal_paint': self._create_vehicle_signature(),
            'human_fabric_mix': self._create_fabric_signature(),
            'tarp_metal_mix': self._create_tarp_signature(),
            'vegetation': self._create_vegetation_signature(),
            'soil': self._create_soil_signature()
        }
        return sigs

    def _create_vehicle_signature(self):
        wavelengths = np.linspace(400, 2500, 200)
        reflectance = np.ones_like(wavelengths) * 0.1
        reflectance[20:40] = 0.05
        reflectance[60:80] = 0.3
        reflectance[100:120] = 0.15
        noise = np.random.normal(0, 0.02, len(reflectance))
        return torch.tensor(np.clip(reflectance + noise, 0, 1), dtype=torch.float32)

    def _create_fabric_signature(self):
        wavelengths = np.linspace(400,2500,200)
        reflectance = np.ones_like(wavelengths) * 0.12
        reflectance[30:60] = 0.28
        noise = np.random.normal(0, 0.02, len(reflectance))
        return torch.tensor(np.clip(reflectance + noise, 0, 1), dtype=torch.float32)

    def _create_tarp_signature(self):
        wavelengths = np.linspace(400,2500,200)
        reflectance = np.ones_like(wavelengths) * 0.15
        reflectance[30:50] = 0.3
        noise = np.random.normal(0, 0.025, len(reflectance))
        return torch.tensor(np.clip(reflectance + noise, 0, 1), dtype=torch.float32)

    def _create_vegetation_signature(self):
        wavelengths = np.linspace(400,2500,200)
        reflectance = np.ones_like(wavelengths) * 0.1
        reflectance[80:120] = 0.4
        return torch.tensor(reflectance, dtype=torch.float32)

    def _create_soil_signature(self):
        wavelengths = np.linspace(400,2500,200)
        reflectance = np.linspace(0.1,0.3,len(wavelengths))
        return torch.tensor(reflectance, dtype=torch.float32)

    def insert_anomaly(self, hsi_cube: torch.Tensor, anomaly_type: str = None, position: Tuple[int,int]=None, create_groundtruth:bool=True) -> Dict:
        """
        hsi_cube: (C, H, W)
        Returns dict with keys: modified_hsi (C,H,W), mask (H,W)
        """
        C,H,W = hsi_cube.shape
        if anomaly_type is None:
            anomaly_type = random.choice(list(self.anomaly_types.keys()))
        cfg = self.anomaly_types[anomaly_type]
        signature = self.anomaly_signatures[cfg['signature_key']]
        if len(signature) != C:
            signature = torch.nn.functional.interpolate(signature.unsqueeze(0).unsqueeze(0), size=C, mode='linear', align_corners=False).squeeze()
        min_s,max_s = cfg['size_range']
        a_size = random.randint(min_s, max_s)
        if position is None:
            pos_y = random.randint(0, H - a_size)
            pos_x = random.randint(0, W - a_size)
        else:
            pos_y, pos_x = position
        abundance = random.uniform(*cfg['abundance_range'])
        modified = hsi_cube.clone()
        mask = torch.zeros((H,W), dtype=torch.float32)
        for yy in range(a_size):
            for xx in range(a_size):
                y = pos_y + yy; x = pos_x + xx
                # mix signature into pixel
                modified[:, y, x] = (1 - abundance) * modified[:, y, x] + abundance * signature
                mask[y,x] = 1.0
        return {'modified_hsi': modified, 'mask': mask, 'metadata': {'type': anomaly_type, 'pos':(pos_y,pos_x), 'size': a_size}}
