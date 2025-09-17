# train_bhuvan.ps1
# Train HSIGene on the ISRO Bhuvan dataset

Set-Location "D:\Desktop\CU\Research\5th sem\IndianDefenseHSIGene"
.\venv\Scripts\Activate

Write-Host "🚀 Training on Bhuvan..."
python main_train.py --config config\hsigene_bhuvan.yaml

Write-Host "🖼 Generating sample outputs..."
python generate_hsi.py --model models\checkpoints\hsigene_bhuvan_epoch_9.pth --config config\hsigene_bhuvan.yaml --num 5

Write-Host "🎉 Bhuvan training + generation completed!"
