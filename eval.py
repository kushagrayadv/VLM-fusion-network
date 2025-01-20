import argparse
from utils.model_config import Config
from utils.train import Trainer


def main(args):
  config = Config(batch_size=args.batch_size)
  trainer = Trainer(config=config)
  trainer.test()


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--batch_size', type=int, default=8, help='batch size')

  args = parser.parse_args()
  main(args)
