import os


class FileModificationWatcher:
    """
    Check if files that are added to this watcher have been changed.
    Use to check if files such as configuration files need to be reloaded once they become dirty.
    """
    def __init__(self, files=[]):
        self._previous_mtimes = {}
        self.add_files(files)

    def add_files(self, files):
        """
        Add multiple file to the watcher

        :param files: list of files
        """
        for f in files:
            self._previous_mtimes[f] = os.path.getmtime(f)

    def files_changed(self):
        """
        Check and update if file linked to this watcher have been modified (since the last call of this function)

        :return: True if modified, False otherwise
        """
        changed = []
        for f in self._previous_mtimes:
            if self._previous_mtimes[f] != os.path.getmtime(f):
                self._previous_mtimes[f] = os.path.getmtime(f)
                changed.append(self._previous_mtimes[f])
        return changed
