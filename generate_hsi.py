# generate_hsi.py - inference / generation utility
import torch, yaml, argparse, os
from src.models.generator import HSIGeneGenerator
from src.utils.anomaly_insertion import DefenseAnomalyInserter

class HSIGenerator:
    def __init__(self, model_path, config_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(model_path, map_location=self.device)
        if config_path:
            with open(config_path,'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = checkpoint.get('config', {})
        self.generator = HSIGeneGenerator(in_channels=self.config.get('in_channels',14),
                                          out_channels=self.config.get('out_channels',200)).to(self.device)
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.generator.eval()
        self.anom = DefenseAnomalyInserter()
        print("Model loaded")

    def generate(self, num_samples=4, output_dir='./generated', add_anomalies=False):
        os.makedirs(output_dir, exist_ok=True)
        for i in range(num_samples):
            inp = torch.randn(1, self.config.get('in_channels',14), 64,64).to(self.device)
            with torch.no_grad():
                out = self.generator(inp).cpu()
            if add_anomalies:
                res = self.anom.insert_anomaly(out.squeeze(0))
                out = res['modified_hsi'].unsqueeze(0)
            torch.save({'input': inp.cpu(), 'generated': out}, os.path.join(output_dir, f'sample_{i}.pt'))
        print("Generated", num_samples, "samples")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='models/hsigene_best.pth')
    parser.add_argument('--config', default='config/hsigene_config.yaml')
    parser.add_argument('--num', type=int, default=4)
    parser.add_argument('--output', default='./generated')
    parser.add_argument('--add_anomalies', action='store_true')
    args = parser.parse_args()
    gen = HSIGenerator(args.model, args.config)
    gen.generate(num_samples=args.num, output_dir=args.output, add_anomalies=args.add_anomalies)

if __name__ == '__main__':
    main()
