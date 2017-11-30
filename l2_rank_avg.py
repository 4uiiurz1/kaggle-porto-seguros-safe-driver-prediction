import sys
import os
from datetime import datetime

import pandas as pd
import numpy as np

from glob import glob

from tqdm import tqdm

from package.loss_and_metric import gini_norm
from package.util import init_logging, log_info


#model_name = 'l2_rank_avg_%s'%datetime.now().strftime('%m%d%H%M')
model_name = 'l2_rank_avg'
init_logging(os.path.join('log', '%s.log'%model_name))


submission_paths = [
    # Kernel
    'Froza_and_Pascal.csv.gz',
    'rgf_submit.csv.gz',
    # My model
    'l1_lgb_11182109.csv.gz',
    'l1_xgb_11230441.csv.gz'
]

log_info('l1_models:')
for submission_path in submission_paths:
    log_info('- %s'%submission_path)

submissions = [pd.read_csv(os.path.join('submission', f), index_col=0) for f in submission_paths]
submissions = pd.concat(submissions, axis=1)
submissions.columns = submission_paths

submission = pd.read_csv('input/sample_submission.csv')
submission['target'] = np.array(np.mean(submissions.rank() / submissions.shape[0], axis=1))

submission.to_csv(os.path.join('submission', '%s.csv.gz'%model_name),
    index=False, compression='gzip')
