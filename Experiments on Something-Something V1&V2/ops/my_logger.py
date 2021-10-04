import sys
import time
from datetime import datetime


# TODO(Yue) Overrided the logger
class Logger(object):
    def __init__(self, log_prefix=""):
        self._terminal = sys.stdout
        self._timestr = datetime.fromtimestamp(time.time()).strftime("%m%d-%H%M%S")
        self._log_path = None
        self._log_dir_name = None
        self._log_file_name = None
        self._history_records = [" ".join(["python"] + sys.argv + ["\n"])]  # TODO(yue) remember the CLI input
        self._write_mode = "bear_in_mind"
        self._prefix = log_prefix

    # TODO(yue) pre_write and create_log help when we don't want to save logs so early because of some early-check process
    # we just bear in mind, and when we realy need to write them down, we do that
    # without Logger: terminal
    # bear_in_mind:   terminal->RAM
    # take_notes:     RAM->FILE
    # normal:         terminal->FILE
    def create_log(self, log_path, test_mode, t, bs, k):
        self._log_dir_name = log_path
        if test_mode:
            self._log_file_name = "test-%s-t%02d-bz%02d-k%02d.txt" % (self._timestr, t, bs, k)
        else:
            self._log_file_name = "log-%s.txt" % self._timestr
        self._log_path = log_path + "/" + self._log_file_name
        self.log = open(self._log_path, "a", 1)
        self._write_mode = "take_notes"
        for record in self._history_records:
            self.write(record)
        self._history_records = []
        self._write_mode = "normal"

    def write(self, message):
        if self._write_mode in ["bear_in_mind", "normal"]:
            self._terminal.write(message)
        if self._write_mode in ["take_notes", "normal"]:
            self.log.write(message.replace("\033[0m", ""). \
                replace("\033[95m", "").replace("\033[94m", "").replace("\033[93m", "").replace("\033[92m",
                                                                                                "").replace(
                "\033[91m", ""))
        else:
            self._history_records.append(message)

    def flush(self):
        pass

    def close_log(self):
        a = 1
        self.log.close()
        return sys.stdout
