import os


class FileModificationWatcher:
    _previous_mtimes = {}

    def __init__(self, files=[]):
        self.add_files(files)

    def add_files(self, files):
        for f in files:
            self._previous_mtimes[f] = os.path.getmtime(f)

    def files_changed(self):
        changed = []
        for f in self._previous_mtimes:
            if self._previous_mtimes[f] != os.path.getmtime(f):
                self._previous_mtimes[f] = os.path.getmtime(f)
                changed.append(self._previous_mtimes[f])
        return changed
