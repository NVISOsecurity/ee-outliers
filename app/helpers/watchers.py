import os

from typing import Dict, List


class FileModificationWatcher:
    _previous_mtimes: Dict[str, float] = {}

    def __init__(self, files: List[str] = []) -> None:
        self.add_files(files)

    def add_files(self, files: List[str]) -> None:
        """
        Aff multiple file to the watcher

        :param files: list of files
        """
        for f in files:
            self._previous_mtimes[f] = os.path.getmtime(f)

    def files_changed(self) -> List[float]:
        """
        Check and update if file linked to this watcher have been modified (since de last call of this function)

        :return: True if modified, False otherwise
        """
        changed: List[float] = []
        for f in self._previous_mtimes:
            if self._previous_mtimes[f] != os.path.getmtime(f):
                self._previous_mtimes[f] = os.path.getmtime(f)
                changed.append(self._previous_mtimes[f])
        return changed
