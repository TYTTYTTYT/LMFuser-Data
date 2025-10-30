from typing import Callable, Literal
from collections.abc import Iterable, Sequence
import tempfile
import os

import torch.distributed as dist
from cloudpickle import pickle

from ..data_loader import DataLoader, R, T
from ..interfaces import Index, Batch, Row
from ..scanners import C4Scanner

DATA_URL = "https://huggingface.co/datasets/allenai/c4/resolve/1ddc917116b730e1859edef32896ec5c16be51d0/multilingual/c4-{language}{split_suffix}.tfrecord-{index:05d}-of-{n_shards:05d}.json.gz"

LANGUAGES = [
    "af",
    "am",
    "ar",
    "az",
    "be",
    "bg",
    "bg-Latn",
    "bn",
    "ca",
    "ceb",
    "co",
    "cs",
    "cy",
    "da",
    "de",
    "el",
    "el-Latn",
    "en",
    "eo",
    "es",
    "et",
    "eu",
    "fa",
    "fi",
    "fil",
    "fr",
    "fy",
    "ga",
    "gd",
    "gl",
    "gu",
    "ha",
    "haw",
    "hi",
    "hi-Latn",
    "hmn",
    "ht",
    "hu",
    "hy",
    "id",
    "ig",
    "is",
    "it",
    "iw",
    "ja",
    "ja-Latn",
    "jv",
    "ka",
    "kk",
    "km",
    "kn",
    "ko",
    "ku",
    "ky",
    "la",
    "lb",
    "lo",
    "lt",
    "lv",
    "mg",
    "mi",
    "mk",
    "ml",
    "mn",
    "mr",
    "ms",
    "mt",
    "my",
    "ne",
    "nl",
    "no",
    "ny",
    "pa",
    "pl",
    "ps",
    "pt",
    "ro",
    "ru",
    "ru-Latn",
    "sd",
    "si",
    "sk",
    "sl",
    "sm",
    "sn",
    "so",
    "sq",
    "sr",
    "st",
    "su",
    "sv",
    "sw",
    "ta",
    "te",
    "tg",
    "th",
    "tr",
    "uk",
    "und",
    "ur",
    "uz",
    "vi",
    "xh",
    "yi",
    "yo",
    "zh",
    "zh-Latn",
    "zu",
]

N_SHARDS_PER_SPLIT = {
    "af": {"train": 64, "validation": 1},
    "am": {"train": 16, "validation": 1},
    "ar": {"train": 1024, "validation": 4},
    "az": {"train": 256, "validation": 1},
    "be": {"train": 128, "validation": 1},
    "bg": {"train": 1024, "validation": 1},
    "bg-Latn": {"train": 4, "validation": 1},
    "bn": {"train": 512, "validation": 1},
    "ca": {"train": 512, "validation": 1},
    "ceb": {"train": 8, "validation": 1},
    "co": {"train": 8, "validation": 1},
    "cs": {"train": 1024, "validation": 2},
    "cy": {"train": 256, "validation": 1},
    "da": {"train": 1024, "validation": 1},
    "de": {"train": 2048, "validation": 16},
    "el": {"train": 1024, "validation": 2},
    "el-Latn": {"train": 16, "validation": 1},
    "en": {"train": 11264, "validation": 128},
    "eo": {"train": 32, "validation": 1},
    "es": {"train": 2048, "validation": 16},
    "et": {"train": 256, "validation": 1},
    "eu": {"train": 64, "validation": 1},
    "fa": {"train": 1024, "validation": 2},
    "fi": {"train": 1024, "validation": 1},
    "fil": {"train": 64, "validation": 1},
    "fr": {"train": 2048, "validation": 16},
    "fy": {"train": 16, "validation": 1},
    "ga": {"train": 16, "validation": 1},
    "gd": {"train": 16, "validation": 1},
    "gl": {"train": 128, "validation": 1},
    "gu": {"train": 64, "validation": 1},
    "ha": {"train": 8, "validation": 1},
    "haw": {"train": 2, "validation": 1},
    "hi": {"train": 1024, "validation": 2},
    "hi-Latn": {"train": 16, "validation": 1},
    "hmn": {"train": 8, "validation": 1},
    "ht": {"train": 8, "validation": 1},
    "hu": {"train": 1024, "validation": 2},
    "hy": {"train": 128, "validation": 1},
    "id": {"train": 1024, "validation": 4},
    "ig": {"train": 4, "validation": 1},
    "is": {"train": 128, "validation": 1},
    "it": {"train": 1024, "validation": 8},
    "iw": {"train": 1024, "validation": 1},
    "ja": {"train": 1024, "validation": 8},
    "ja-Latn": {"train": 8, "validation": 1},
    "jv": {"train": 8, "validation": 1},
    "ka": {"train": 256, "validation": 1},
    "kk": {"train": 256, "validation": 1},
    "km": {"train": 64, "validation": 1},
    "kn": {"train": 64, "validation": 1},
    "ko": {"train": 1024, "validation": 1},
    "ku": {"train": 16, "validation": 1},
    "ky": {"train": 64, "validation": 1},
    "la": {"train": 64, "validation": 1},
    "lb": {"train": 32, "validation": 1},
    "lo": {"train": 8, "validation": 1},
    "lt": {"train": 512, "validation": 1},
    "lv": {"train": 256, "validation": 1},
    "mg": {"train": 8, "validation": 1},
    "mi": {"train": 4, "validation": 1},
    "mk": {"train": 128, "validation": 1},
    "ml": {"train": 128, "validation": 1},
    "mn": {"train": 128, "validation": 1},
    "mr": {"train": 1024, "validation": 1},
    "ms": {"train": 512, "validation": 1},
    "mt": {"train": 128, "validation": 1},
    "my": {"train": 64, "validation": 1},
    "ne": {"train": 256, "validation": 1},
    "nl": {"train": 1024, "validation": 4},
    "no": {"train": 1024, "validation": 1},
    "ny": {"train": 4, "validation": 1},
    "pa": {"train": 32, "validation": 1},
    "pl": {"train": 1024, "validation": 4},
    "ps": {"train": 16, "validation": 1},
    "pt": {"train": 1024, "validation": 4},
    "ro": {"train": 1024, "validation": 2},
    "ru": {"train": 4096, "validation": 32},
    "ru-Latn": {"train": 32, "validation": 1},
    "sd": {"train": 64, "validation": 1},
    "si": {"train": 64, "validation": 1},
    "sk": {"train": 512, "validation": 1},
    "sl": {"train": 256, "validation": 1},
    "sm": {"train": 4, "validation": 1},
    "sn": {"train": 8, "validation": 1},
    "so": {"train": 64, "validation": 1},
    "sq": {"train": 128, "validation": 1},
    "sr": {"train": 256, "validation": 1},
    "st": {"train": 2, "validation": 1},
    "su": {"train": 4, "validation": 1},
    "sv": {"train": 1024, "validation": 2},
    "sw": {"train": 32, "validation": 1},
    "ta": {"train": 256, "validation": 1},
    "te": {"train": 128, "validation": 1},
    "tg": {"train": 64, "validation": 1},
    "th": {"train": 1024, "validation": 1},
    "tr": {"train": 1024, "validation": 4},
    "uk": {"train": 1024, "validation": 2},
    "und": {"train": 3072, "validation": 32},
    "ur": {"train": 128, "validation": 1},
    "uz": {"train": 32, "validation": 1},
    "vi": {"train": 1024, "validation": 4},
    "xh": {"train": 2, "validation": 1},
    "yi": {"train": 16, "validation": 1},
    "yo": {"train": 2, "validation": 1},
    "zh": {"train": 1024, "validation": 2},
    "zh-Latn": {"train": 8, "validation": 1},
    "zu": {"train": 8, "validation": 1},
}

def _get_lang_paths(lang: str, split: Literal['train', 'validation']) -> list[str]:
    num = N_SHARDS_PER_SPLIT[lang][split]
    suffix = '' if split == 'train' else 'validation'
    return [
        DATA_URL.format(language=lang, split_suffix=suffix, index=i, n_shards=num)
        for i in range(num)
    ]

def init_c4_dataloader(
    languages: list[str],
    language_weights: list[float],
    split: Literal['train', 'validation'],
    batch_size: int,
    seed: int = 42,
    shuffle: bool = True,
    pre_fetch_factor: int = 32,
    num_workers: int = 1,
    ignore_error: bool = False,
    qps: float | None = None,
    map_fn: Callable[[Row], Row] | None = None,
    flow_fn: Callable[[Iterable[Row]], Iterable[Row]] | None = None,
    batch_map_fn: Callable[[Batch], Batch] | None = None,
    indexes: list[list[Index]] | None = None,
    instruct_timeout: float | None = 600.0,
    worker_timeout: float | None = 600.0,
    restart_cnt: int | None = None,
    num_ranks: int | None = None,
    rank_idx: int | None = None,
    recover_states: bytes | None = None,
    data_dir: str | None = None,
) -> DataLoader:
    if num_ranks is None:
        if dist.is_initialized():
            num_ranks = dist.get_world_size()
        else:
            num_ranks = 1
    if rank_idx is None:
        if dist.is_initialized():
            rank_idx = dist.get_rank()
        else:
            rank_idx = 0

    for lang in languages:
        assert lang in LANGUAGES, f"Language {lang} is not supported."

    with tempfile.TemporaryDirectory() as tmp_dir:
        print(tmp_dir)
        if data_dir is not None:
            tmp_dir = data_dir
        path_list: Sequence[str] = []
        for lang in languages:
            paths = _get_lang_paths(lang, split)
            path = os.path.join(tmp_dir, f'{lang}.txt')
            with open(path, 'w') as f:
                f.write('\n'.join(paths))
            path_list.append(path)
        if recover_states is not None:
            states = pickle.loads(recover_states)
            assert isinstance(states, dict)
            states['path_list'] = path_list
            return DataLoader.from_states(pickle.dumps(states))

        return DataLoader(
            batch_size=batch_size,
            path_list=path_list,
            scanner_type=C4Scanner,
            seed=seed,
            shuffle=shuffle,
            pre_fetch_factor=pre_fetch_factor,
            indexes=indexes,
            infinite=False,
            map_fn=map_fn,
            flow_fn=flow_fn,
            ignore_error=ignore_error,
            qps=qps,
            instruct_timeout=instruct_timeout,
            worker_timeout=worker_timeout,
            restart_cnt=restart_cnt,
            num_workers=num_workers,
            num_ranks=num_ranks,
            rank_idx=rank_idx,
            batch_map_fn=batch_map_fn,
            distributor_weights=language_weights
        )
