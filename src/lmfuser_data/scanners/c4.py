from collections.abc import Iterator
from io import BytesIO
import json
import requests
import gzip

from .interface import Scanner
from ..interfaces import Row, RowBase


class C4Row(RowBase):
    text: str
    timestamp: str
    url: str


class C4Scanner(Scanner[C4Row]):
    """
    C4Scanner is a scanner for C4 files.
    It provides the basic interface for scanning a single file.
    """

    def __init__(self, path: str, **kwargs) -> None:
        super().__init__(path, **kwargs)
        self._lines: list[str] | None = None

    def _load_data(self) -> None:
        if self._lines is not None:
            return
        if self.path.startswith('https'):
            resp = requests.get(self.path)
            resp.raise_for_status()
            content = resp.content
        else:
            with open(self.path, 'rb') as f:
                content = f.read()

        with gzip.open(BytesIO(content), mode='rt', encoding='utf-8') as f:
            self._lines = f.readlines()

    def __getitem__(self, index: int) -> C4Row:
        self._load_data()
        assert self._lines is not None, 'Data not properly loaded'

        line = self._lines[index]
        row = json.loads(line)
        return C4Row(
            text=str(row['text']),
            timestamp=str(row['timestamp']),
            url=str(row['url'])
        )

    def __len__(self) -> int:
        self._load_data()
        assert self._lines is not None, 'Data not properly loaded'
        return len(self._lines)

    def __iter__(self) -> Iterator[C4Row]:
        for idx in range(len(self)):
            yield self[idx]

    @classmethod
    def check_file(cls, path: str) -> bool:
        if path.lower().endswith('.json.gz'):
            return True
        return False
