from helpers.settings import Settings
from helpers.logging import Logging
from helpers.es import ES

# Settings
settings = Settings()

# Logging
logging = Logging("outliers")
logging.add_stdout_handler()

# Initialize ES
es = ES(settings, logging)
