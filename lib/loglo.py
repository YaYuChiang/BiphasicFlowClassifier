import logging
import time
import os
import types
import sys
# import traceback

from logging.handlers import TimedRotatingFileHandler

# PURPOSE:
# logs message to file
# logging files each day that will be different file names
# Note that path of logs refers the path of this file

# 2021.4.17 by panda


class Simple:
    splits = " "

    @staticmethod
    def d(self, *msgs):
        assert isinstance(self, logging.Logger)
        buff = ""
        for msg in msgs:
            buff += "{}{}".format(msg, Simple.splits)
        self.debug(buff)

    @staticmethod
    def i(self, *msgs):
        buff = ""
        for msg in msgs:
            buff += "{}{}".format(msg, Simple.splits)
        self.info(buff)

    @staticmethod
    def w(self, *msgs):
        assert isinstance(self, logging.Logger)
        buff = ""
        for msg in msgs:
            buff += "{}{}".format(msg, Simple.splits)
        self.warning(buff)

    @staticmethod
    def e(self, *msgs):
        assert isinstance(self, logging.Logger)
        buff = ""
        for msg in msgs:
            buff += "{}{}".format(msg, Simple.splits)
        self.error(buff)

    @staticmethod
    def c(self, *msgs):
        assert isinstance(self, logging.Logger)
        buff = ""
        for msg in msgs:
            buff += "{}{}".format(msg, Simple.splits)
        self.critical(buff)


def init_log(filepath=__file__,
             show_console=True,
             save_log=False,
             master="",
             level=1,
             log_to_var_log=False,
             fmt_console_date='%H:%M:%S'):
    basename = os.path.splitext(os.path.basename(filepath))[0]
    # logging.basicConfig(format='%(asctime)s [%(levelname)s] (%(threadName)-10s) %(message)s')

    _logger = logging.getLogger(basename)

    if show_console:
        consoleh = logging.StreamHandler(sys.stdout)
        if os.name == "nt":
            format_log = '%(asctime)s %(filename).3s:%(lineno)3d %(levelname)-.4s {} %(message)s'.format(master)
        else:
            format_log = '%(asctime)s.%(msecs)03d %(filename).5s:%(lineno)d %(levelname)-.1s {} %(message)s'.format(master)
        # consoleh.setFormatter(logging.Formatter('%(asctime)s (%(levelname)-.1s {} %(message)s'.format(master)))
        consoleh.setFormatter(logging.Formatter(format_log,
                                                datefmt=fmt_console_date))
        _logger.addHandler(consoleh)

    if save_log:
        if os.name == "nt":
            log_path = os.path.join(os.path.dirname(__file__), "putil/logs", "log_" + basename)
        else:
            if log_to_var_log is False:
                # log_path = os.path.join(os.path.dirname(__file__), "log_" + basename)
                log_path = os.path.join(os.path.dirname(__file__), "log")
            else:
                # remember to create dir first
                log_path = os.path.join('/var/log/python/', basename)
        if not os.path.exists(log_path):
            os.makedirs(log_path)

        # start_time = datetime.datetime.now().strftime('%y%m%d')
        logfile_path = os.path.join(log_path, basename + ".log")
        fileh = TimedRotatingFileHandler(logfile_path,
                                         when="midnight",
                                         backupCount=365)
        fileh.setFormatter(logging.Formatter('%(asctime)s %(filename).3s:%(lineno)d (%(levelname)-.1s %(message)s'))
        _logger.addHandler(fileh)

    try:
        level = int(level)
    except:
        level = 0
        
    if level == 0:
        _logger.setLevel(level=logging.DEBUG)
    elif level == 1:
        _logger.setLevel(level=logging.INFO)
    elif level == 2:
        _logger.setLevel(level=logging.WARNING)
    elif level == 3:
        _logger.setLevel(level=logging.ERROR)
    elif level == 4:
        _logger.setLevel(level=logging.CRITICAL)

    ## cant show the line nb, should not them
    # _logger.d = types.MethodType(Simple.d, _logger)
    # _logger.i = types.MethodType(Simple.i, _logger)
    # _logger.w = types.MethodType(Simple.w, _logger)
    # _logger.e = types.MethodType(Simple.e, _logger)
    # _logger.c = types.MethodType(Simple.c, _logger)

    _logger.warn = _logger.warning

    return _logger


if __name__ == "__main__":
    lo = init_log(__file__, save_log=True, level=0)
    while True:
        lo.debug("@Q@")
        lo.info("@Q@!")
        lo.warning("@Q@!!")
        lo.error("@Q@!!?")
        lo.fatal("@Q@!!!??")

        gee = "gee"
        lo.d("@Q@", gee)
        lo.i("@Q@!", gee)
        lo.w("@Q@!!", gee)
        lo.e("@Q@!!?", gee)
        lo.c("@Q@!!!??", gee)
        time.sleep(75)

