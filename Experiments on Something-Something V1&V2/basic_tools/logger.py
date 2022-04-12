import os
import sys
import logging

class Logger:
    def __init__(self, path, mode='w'):
        assert mode in {'w', 'a'}, 'unknown mode for logger %s' % mode

        fh = logging.FileHandler(path, mode=mode)
        formatter = logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s] - %(message)s')
        fh.setFormatter(formatter)
        # ch = logging.StreamHandler(sys.__stdout__)

        self.logger = logging.getLogger()
        self.logger.addHandler(fh)
        # self.logger.addHandler(ch)

    def write(self, message):
        if message == "\n": return
        # Remove \n at the end.
        self.logger.info(message.strip())

    def flush(self):
        # for python 3 compatibility.
        pass

