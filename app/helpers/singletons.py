from helpers.settings import Settings
from helpers.logging import Logging
from helpers.es import ES

# Settings
settings = Settings()

# Logging
logging = Logging("outliers")
logging.add_stdout_handler()

logging.logger.info("List whitelist literals: " + str(settings.whitelist_literals_config))
logging.logger.info("tfl_20190809 in: " +
                    str("rare process not on disk: WerFault.exe" in settings.whitelist_literals_config))

# Initialize ES
es = ES(settings, logging)
