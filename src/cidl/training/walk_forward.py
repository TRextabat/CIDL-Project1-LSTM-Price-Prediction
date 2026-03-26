"""Walk-forward cross-validation splitter for time series."""

from typing import Generator


def walk_forward_split(
    n_samples: int,
    n_folds: int = 3,
    purge: int = 30,
) -> Generator[tuple[list[int], list[int]], None, None]:
    """Generate expanding walk-forward train/test splits.

    The training set grows with each fold while test sets are non-overlapping.
    A purge gap between train and test prevents information leakage.

    Fold layout (3 folds):
        Fold 0: [====TRAIN====]---purge---[==TEST==]
        Fold 1: [========TRAIN========]---purge---[==TEST==]
        Fold 2: [============TRAIN============]---purge---[==TEST==]

    Args:
        n_samples: Total number of samples.
        n_folds: Number of walk-forward folds (default: 3).
        purge: Number of bars to skip between train and test (default: 30).

    Yields:
        (train_indices, test_indices) tuples.
    """
    test_size = n_samples // (n_folds + 1)

    for fold in range(n_folds):
        train_end = test_size * (fold + 1)
        test_start = train_end + purge
        test_end = test_start + test_size

        if test_end > n_samples:
            test_end = n_samples

        if test_start >= n_samples:
            break

        train_indices = list(range(0, train_end))
        test_indices = list(range(test_start, test_end))

        if len(test_indices) > 0:
            yield train_indices, test_indices
