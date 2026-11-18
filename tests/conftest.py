import socket
from urllib.parse import urlparse

import pytest
from map_lambda import LOCAL_ENDPOINT_URL


# after `-m` has deselected tests
@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config, items):
    if not any(item.get_closest_marker("lambda_integration") for item in items):
        return
    endpoint = urlparse(LOCAL_ENDPOINT_URL)
    try:
        socket.create_connection((endpoint.hostname, endpoint.port), timeout=1).close()
    except OSError:
        raise pytest.UsageError(
            f"no local Lambda emulator at {LOCAL_ENDPOINT_URL}: start one with "
            "`cargo lambda watch` in rust/, or deselect these tests with "
            '-m "not lambda_integration"'
        ) from None
