"""
Automated Dataset Download Script
Run this first to download all required datasets
"""

import os
import requests
from pathlib import Path

class DatasetDownloader:
    def __init__(self, data_root="data/raw_datasets"):
        self.data_root = Path(data_root)
        self.data_root.mkdir(parents=True, exist_ok=True)
        
    def setup_eurosat(self):
        """Setup EuroSAT dataset download info"""
        print("Setting up EuroSAT dataset info...")
        eurosat_dir = self.data_root / "eurosat"
        eurosat_dir.mkdir(exist_ok=True)
        
        readme_content = """
# EuroSAT Dataset

To download the EuroSAT dataset:

1. Visit: https://zenodo.org/record/7711810
2. Download EuroSAT_MS.zip (Multi-spectral version)  
3. Extract to this directory

Alternative sources:
- TensorFlow Datasets: tensorflow_datasets.load('eurosat')
- Hugging Face: https://huggingface.co/datasets/blanchon/EuroSAT_RGB
- Kaggle: https://www.kaggle.com/datasets/apollo2506/eurosat-dataset

Dataset Info:
- 27,000 labeled samples
- 10 land cover classes
- 13 Sentinel-2 spectral bands
- 64x64 pixel patches
        """
        
        with open(eurosat_dir / "README.md", 'w', encoding='utf-8') as f:
            f.write(readme_content)
            
        print("EuroSAT setup completed")
    
    def setup_chikusei(self):
        """Setup Chikusei dataset download info"""
        print("Setting up Chikusei dataset info...")
        chikusei_dir = self.data_root / "chikusei"
        chikusei_dir.mkdir(exist_ok=True)
        
        readme_content = """
# Chikusei Dataset

To download the Chikusei dataset:

1. Visit: https://naotoyokoya.com/Download.html
2. Register and download Chikusei_20170519.zip
3. Extract to this directory

Dataset Info:
- Spectral Range: 343-1018nm
- Spectral Bands: 128 bands
- Spatial Resolution: 2.5m GSD
- Image Size: 2517×2335 pixels
- Sensor: Headwall Photonics airborne sensor
        """
        
        with open(chikusei_dir / "README.md", 'w', encoding='utf-8') as f:
            f.write(readme_content)
        
        print("Chikusei setup completed")
    
    def setup_other_datasets(self):
        """Setup info for other datasets"""
        
        datasets_info = {
            "dfc2013": {
                "url": "https://hyperspectral.ee.uh.edu/?page_id=459",
                "description": "IEEE GRSS Data Fusion Contest 2013",
                "specs": "Spectral Range: 380-1050nm, 144 bands, 2.5m GSD"
            },
            "dfc2018": {
                "url": "https://ieee-dataport.org/competitions/2018-ieee-grss-data-fusion-challenge", 
                "description": "IEEE GRSS Data Fusion Contest 2018",
                "specs": "Spectral Range: 380-1050nm, 48 bands, 1m GSD"
            },
            "xiong_an": {
                "contact": "caoxiangyong@xjtu.edu.cn",
                "description": "Xiong'an New Area Dataset (HSIGene paper)",
                "specs": "Spectral Range: 400-1000nm, 250 bands, 0.5m GSD"
            },
            "heihe": {
                "url": "https://data.tpdc.ac.cn/en/topic/heihe",
                "description": "Heihe River Basin Dataset",
                "specs": "Third Pole Environment Data Center"
            }
        }
        
        for dataset_name, info in datasets_info.items():
            dataset_dir = self.data_root / dataset_name
            dataset_dir.mkdir(exist_ok=True)
            
            readme_content = f"""
# {info['description']}

Dataset: {dataset_name}
Specifications: {info['specs']}

Download Instructions:
"""
            
            if 'url' in info:
                readme_content += f"1. Visit: {info['url']}\n2. Register and download\n3. Extract to this directory\n"
            
            if 'contact' in info:
                readme_content += f"1. Contact: {info['contact']}\n2. Request dataset access\n3. Place files in this directory\n"
            
            with open(dataset_dir / "README.md", 'w', encoding='utf-8') as f:
                f.write(readme_content)
            
            print(f"{dataset_name} setup completed")
    
    def download_all(self):
        """Setup all datasets"""
        print("Setting up dataset download information...")
        
        self.setup_eurosat()
        self.setup_chikusei()
        self.setup_other_datasets()
        
        print("Dataset setup completed!")
        print("Check data/raw_datasets/ for download instructions")

if __name__ == "__main__":
    downloader = DatasetDownloader()
    downloader.download_all()