import argparse
from utils.model_config import Config
from utils.train import Trainer


def main(args):
  config = Config(epochs=args.epochs, batch_size=args.batch_size, num_attention_layers=args.num_attention_layers)
  trainer = Trainer(config=config)
  trainer.train()


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--epochs', type=int, default=5, help='num_epochs')
  parser.add_argument('--batch_size', type=int, default=8, help='batch size')
  parser.add_argument('--num_attention_layers', type=int, default=5,
                      help='number of hidden layers for cross-modality encoder')

  args = parser.parse_args()
  main(args)
