import sys

from functions.predictor import Predictor
from options_pcr import options as options_pcr
from options_psgn import options as options_psgn

if __name__ == "__main__":
    predictor = Predictor(options_pcr, data_type="failure", name="failure",
                          ckpt_file="checkpoints/pcr-v7.5/025.ckpt",
                          detail_record=True)
    predictor.predict_all()
