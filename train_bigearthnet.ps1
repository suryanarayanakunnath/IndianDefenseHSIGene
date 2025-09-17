# train_bigearthnet.ps1
# Train HSIGene on the BigEarthNet-S2 dataset

Set-Location "D:\Desktop\CU\Research\5th sem\IndianDefenseHSIGene"
.\venv\Scripts\Activate

Write-Host "🚀 Training on BigEarthNet-S2..."
python main_train.py --config config\hsigene_bigearthnet.yaml

Write-Host "🖼 Generating sample outputs..."
python generate_hsi.py --model models\checkpoints\hsigene_bigearthnet_epoch_9.pth --config config\hsigene_bigearthnet.yaml --num 5

Write-Host "🎉 BigEarthNet training + generation completed!"
