from types import SimpleNamespace

import pytest

from zarr_datafusion_lambda import sql

KEYS = ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN")


@pytest.fixture
def chain(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """boto3's chain finding `chain.credentials`, with no `AWS_*` keys set and none exported."""
    for key in KEYS:
        # Set first, so the variables export_credentials sets are undone too.
        monkeypatch.setenv(key, "")
        monkeypatch.delenv(key)
    monkeypatch.setattr(sql, "_exported", False)
    found = SimpleNamespace(credentials=None, asked=0)

    class Session:
        def get_credentials(self):
            found.asked += 1
            if found.credentials is None:
                return None
            return SimpleNamespace(get_frozen_credentials=lambda: found.credentials)

    monkeypatch.setattr(sql.boto3, "Session", Session)
    return found


def _frozen(token: str | None = "token") -> SimpleNamespace:
    return SimpleNamespace(access_key="key", secret_key="secret", token=token)


def test_credentials_from_the_chain_are_put_in_the_environment(chain):
    chain.credentials = _frozen()

    sql.export_credentials()

    assert [sql.os.environ[key] for key in KEYS] == ["key", "secret", "token"]


def test_long_term_keys_have_no_session_token(chain):
    chain.credentials = _frozen(token=None)

    sql.export_credentials()

    assert "AWS_SESSION_TOKEN" not in sql.os.environ


def test_keys_already_in_the_environment_are_kept(chain, monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "mine")
    chain.credentials = _frozen()

    sql.export_credentials()

    assert sql.os.environ["AWS_ACCESS_KEY_ID"] == "mine"
    assert chain.asked == 0


def test_without_credentials_the_environment_is_left_alone(chain):
    sql.export_credentials()

    assert not any(key in sql.os.environ for key in KEYS)


def test_the_chain_is_asked_once_a_process(chain):
    sql.export_credentials()
    chain.credentials = _frozen()
    sql.export_credentials()

    assert chain.asked == 1
    assert "AWS_ACCESS_KEY_ID" not in sql.os.environ
