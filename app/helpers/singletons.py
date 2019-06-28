from helpers.settings import Settings
from helpers.logging import Logging
from helpers.es import ES

# Settings
settings: Settings = Settings()

# Logging
logging: Logging = Logging("outliers")
logging.add_stdout_handler()

# Initialize ES
es: ES = ES(settings, logging)
