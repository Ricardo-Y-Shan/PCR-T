import sys

from functions.tester import Tester
from options_pcr import options as options_pcr
from options_psgn import options as options_psgn

if __name__ == "__main__":
    ckpt_file = "checkpoints/pcr-v7.5/012.ckpt"
    tester = Tester(options_pcr, state_file=ckpt_file, data_type='test', detail_report=True)
    tester.test()
