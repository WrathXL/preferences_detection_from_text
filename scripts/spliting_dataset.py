import os
import pandas as pd
from corpus_process import Corpus

data_dir = os.path.join(os.pardir, 'data')
tsv_files = ['/sensitive1.tsv', '/sensitive2.tsv', '/sensitive3.tsv']


corpus = Corpus([data_dir + f for f in tsv_files])
