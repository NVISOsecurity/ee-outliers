
from helpers.singletons import settings
import copy


class UpdateSettings:

    def __init__(self):
        args_config = settings.args.config
        self.backup_default_args_config = copy.deepcopy(args_config)

    def change_configuration_path(self, new_path):
        settings.args.config = [new_path]
        settings.process_configuration_files()

    def restore_default_configuration_path(self):
        settings.args.config = self.backup_default_args_config
        settings.error_parsing_config = None
        settings.process_configuration_files()
