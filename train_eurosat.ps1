# train_eurosat.ps1
# Train HSIGene on the EuroSAT dataset

Set-Location "D:\Desktop\CU\Research\5th sem\IndianDefenseHSIGene"
.\venv\Scripts\Activate

Write-Host "🚀 Training on EuroSAT..."
python main_train.py --config config\hsigene_eurosat.yaml

Write-Host "🖼 Generating sample outputs..."
python generate_hsi.py --model models\checkpoints\hsigene_eurosat_epoch_9.pth --config config\hsigene_eurosat.yaml --num 5

Write-Host "🎉 EuroSAT training + generation completed!"
