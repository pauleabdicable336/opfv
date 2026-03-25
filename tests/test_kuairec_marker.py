import os

import pytest


@pytest.mark.kuairec
def test_kuairec_placeholder() -> None:
    """Gate real-data tests: set KUAIREC_ROOT to your CSV tree to enable."""
    root = os.environ.get("KUAIREC_ROOT")
    if not root or not os.path.isdir(os.path.join(root, "data")):
        pytest.skip("KUAIREC_ROOT not set or missing data/ subdirectory")
