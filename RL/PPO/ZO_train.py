"""Run from any directory: python RL/PPO/ZO_train.py ZO reentrant_2."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import yaml
from RL.PPO.ZO_trainer import ZerothOrderTrainer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('policy_config', nargs='?', default='ZO')
    parser.add_argument('env_config', nargs='?', default='reentrant_2')
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()

    def read_config(name, directory):
        path = Path(name)
        if path.suffix != '.yaml':
            path = path.with_suffix('.yaml')
        if not path.is_absolute():
            path = directory / path
        with path.open() as stream:
            return yaml.safe_load(stream)

    config = read_config(args.policy_config, ROOT / 'RL' / 'policy_configs')
    env_config = read_config(args.env_config, ROOT / 'configs' / 'env')
    output = args.output_dir or ROOT / 'RL' / 'PPO' / env_config['name']
    trainer = ZerothOrderTrainer(env_config, config, output)
    print(f'Using {trainer.workers} CPUs; saving to {output}', flush=True)
    trainer.train()


if __name__ == '__main__':
    main()
