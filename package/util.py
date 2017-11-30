import logging
import time
import datetime

def log_info(message):
    ts = time.time()
    logging.info(message)
    print(message)

def init_logging(filename):
    logging.basicConfig(format='%(message)s',
        level=logging.INFO, filename=filename)

class Logger:
    def __init__(self, filename):
        self.filename = filename
        f = open(self.filename, 'w')
        f.close()

    def info(self, message):
        f = open(self.filename, 'a')
        if message == '\n':
            f.write('\n')
        else:
            f.write(message+'\n')
        f.close()
        print(message)
