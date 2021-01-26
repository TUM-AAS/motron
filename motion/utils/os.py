import errno
import os
from tqdm import tqdm
from urllib.request import urlretrieve


def maybe_makedir(path: str) -> None:
    try:
        # Create output directory if it does not exist
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def download_file(url: str, path: str, verbose: bool = False) -> None:
    if verbose:
        def reporthook(t):
            """Wraps tqdm instance.
            Don't forget to close() or __exit__()
            the tqdm instance once you're done with it (easiest using `with` syntax).
            """
            last_b = [0]

            def update_to(b=1, bsize=1, tsize=None):
                """
                b  : int, optional
                    Number of blocks transferred so far [default: 1].
                bsize  : int, optional
                    Size of each block (in tqdm units) [default: 1].
                tsize  : int, optional
                    Total size (in tqdm units). If [default: None] remains unchanged.
                """
                if tsize is not None:
                    t.total = tsize
                t.update((b - last_b[0]) * bsize)
                last_b[0] = b

            return update_to
        with tqdm(unit='B', unit_scale=True, unit_divisor=1024, miniters=1, desc=url) as t:
            urlretrieve(url, path, reporthook=reporthook(t))
    else:
        urlretrieve(url, path)
