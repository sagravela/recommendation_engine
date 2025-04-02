from rich.logging import RichHandler
import logging

FORMAT = "%(message)s"
logging.basicConfig(
    level="INFO", format=FORMAT, datefmt="[%X]", handlers=[RichHandler(show_path=False)]
)

log = logging.getLogger("rich")
