import os

from typing import Dict, List

class FileModificationWatcher:
    _previous_mtimes: Dict[str, float] = {}

    def __init__(self, files: List[str]=[]) -> None:
        self.add_files(files)

    def add_files(self, files: List[str]) -> None:
        for f in files:
            self._previous_mtimes[f] = os.path.getmtime(f)

    def files_changed(self) -> List[float]:
        changed: List[float] = []
        for f in self._previous_mtimes:
            if self._previous_mtimes[f] != os.path.getmtime(f):
                self._previous_mtimes[f] = os.path.getmtime(f)
                changed.append(self._previous_mtimes[f])
        return changed