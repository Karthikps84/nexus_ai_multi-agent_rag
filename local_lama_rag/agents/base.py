from utils.logger import get_logger


class BaseAgent:
    """Every agent gets a logger tagged with its own class name -- makes
    LangSmith/log output easy to filter by agent when debugging."""

    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
