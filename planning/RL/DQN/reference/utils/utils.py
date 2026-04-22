from time import time
import os


class Log:
    def __init__(self, log_dir):
        os.makedirs(log_dir) if not os.path.isdir(log_dir) else None
        self.log = open(log_dir+'/log.txt', 'wt')

    def write(self, s):
        print(s)
        self.log.write(s+'\n')

    def close(self):
        self.log.close()



def ela_t(start_t):
    ela_sec = time() - start_t
    if ela_sec < 60:
        ela = f'{ela_sec:.0f}s'
    elif 60 <= ela_sec < 60 * 60:
        ela = f'{ela_sec // 60:.0f}m {ela_sec % 60:.0f}s'
    else:
        ela_min = ela_sec // 60
        ela = f'{ela_min // 60:.0f}h {ela_sec % 60:.0f}m'
    return ela