import pprint
import sys

from functions.trainer import Trainer
from options_pcr import options as options_pcr
from options_psgn import options as options_psgn

if __name__ == "__main__":
    ckpt_file = "checkpoints/pcr-v7.5/005.ckpt"
    options_text = pprint.pformat(vars(options_pcr))
    print(options_text)
    trainer = Trainer(options_pcr, state_file=ckpt_file)
    trainer.train()
