import logging
import os

class Logger():
    def __init__(self, dir, name='SleepVST', run_name=None, level=logging.DEBUG):
        """
        :param dir: 로그 파일을 저장할 디렉토리
        :param name: 로그 파일 이름
        :param level: 로그 레벨
        """
        # GREEN = "\033[92m"
        # RED = "\033[91m"
        # YELLOW = "\033[93m"
        # RESET = "\033[0m"
        # BOLD = "\033[1m"
        # UNDERLINE = "\033[4m"
        # # 색상 설정
        # self.colors = {
        #     'DEBUG': GREEN,
        #     'INFO': GREEN,
        #     'WARNING': YELLOW,
        #     'ERROR': RED,
        #     'CRITICAL': RED
        # }
        # # 로그 레벨 설정
        # self.levels = {
        #     'DEBUG': logging.DEBUG,
        #     'INFO': logging.INFO,
        #     'WARNING': logging.WARNING,
        #     'ERROR': logging.ERROR,
        #     'CRITICAL': logging.CRITICAL
        # }
        # # 로그 레벨에 따라 색상 설정
        # logging.addLevelName(logging.DEBUG, f"{GREEN}{logging.getLevelName(logging.DEBUG)}{RESET}")
        # logging.addLevelName(logging.INFO, f"{GREEN}{logging.getLevelName(logging.INFO)}{RESET}")
        # logging.addLevelName(logging.WARNING, f"{YELLOW}{logging.getLevelName(logging.WARNING)}{RESET}")
        # logging.addLevelName(logging.ERROR, f"{RED}{logging.getLevelName(logging.ERROR)}{RESET}")
        # logging.addLevelName(logging.CRITICAL, f"{RED}{logging.getLevelName(logging.CRITICAL)}{RESET}")
        # 로그 파일 설정
        self.name = name
        self.level = level
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)

        # 로그 파일 핸들러 설정
        filename = name
        if run_name:
            filename = f"{name}_{run_name}"
        log_file = os.path.join(dir, filename + ".log")
        if not os.path.exists(dir):
            os.makedirs(dir)
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        # 포맷터 설정
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
            )
        ch.setFormatter(formatter)
        fh.setFormatter(formatter)
        # 핸들러 추가
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
        # 로그 레벨 설정
        self.logger.setLevel(level)

    def debug(self, msg):
        self.logger.debug(msg)

    def info(self, msg):
        self.logger.info(msg)

    def warning(self, msg):
        self.logger.warning(msg)

    def error(self, msg):
        self.logger.error(msg)

    def critical(self, msg):
        self.logger.critical(msg)