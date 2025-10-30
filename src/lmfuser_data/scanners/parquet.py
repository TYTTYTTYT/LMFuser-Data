from collections.abc import Iterator, Sequence
from io import BytesIO
import requests
import logging

import pandas as pd

from .interface import Scanner
from ..interfaces import Row
from ..utils import retry

logger = logging.getLogger(__name__)


class ParquetScanner(Scanner):
    """
        Scanner for Parquet files.
    """

    def __init__(self, path: str, **kwargs) -> None:
        super().__init__(path, **kwargs)
        self._rows: Sequence[Row] | None = None

    @retry(tries=10, delay=0.1, backoff=2, logger=logger)
    def _load_data(self) -> None:
        if self._rows is not None:
            return
        if self.path.startswith('https'):
            resp = requests.get(self.path)
            resp.raise_for_status()
            content = resp.content
        else:
            with open(self.path, 'rb') as f:
                content = f.read()

        table = pd.read_parquet(BytesIO(content))

        self._rows = table.to_dict(orient='records')

    def __getitem__(self, index: int) -> Row:
        self._load_data()
        assert self._rows is not None, 'Data not properly loaded'

        return self._rows[index]

    def __len__(self) -> int:
        self._load_data()
        assert self._rows is not None, 'Data not properly loaded'
        return len(self._rows)

    def __iter__(self) -> Iterator[Row]:
        for idx in range(len(self)):
            yield self[idx]

    @classmethod
    def check_file(cls, path: str) -> bool:
        if path.lower().endswith('.parquet'):
            return True
        return False
