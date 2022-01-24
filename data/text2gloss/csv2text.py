import os
from cv2 import seamlessClone
from numpy import save
import pandas as pd
import warnings
from collections import defaultdict
from tqdm import tqdm
import os.path as osp


def save_text(csv_path, text_path, train=True):
    tag = 'train' if train else 'val'
    csv_path = osp.join(csv_path, 'how2sign_realigned_{}.csv'.format(tag))
    text_path = osp.join(text_path, 'how2sign_realigned_{}.txt'.format(tag))
    data = pd.read_csv(csv_path, on_bad_lines='skip', delimiter="\t")
    debug = 0
    warnings.filterwarnings('ignore')

    key_json_files = []
    sentences = []
    vocabulary = defaultdict(int)

    with open(text_path, "w") as f:
        for i in tqdm(range(len(data))):
            if debug and i >= debug: break
            sent = data["SENTENCE"][i].strip()
            f.write(sent + "\n")

if __name__ == "__main__":
    csv_path = "../../Data/"
    text_path = "./"
    save_text(csv_path, text_path, train=True)
    save_text(csv_path, text_path, train=False)