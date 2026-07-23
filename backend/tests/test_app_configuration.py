from unittest.mock import patch

from fastapi.middleware.cors import CORSMiddleware

from app.main import _build_app


def test_build_app_allows_all_cors_origins() -> None:
    application = _build_app()
    cors = next(item for item in application.user_middleware if item.cls is CORSMiddleware)
    assert cors.kwargs["allow_origins"] == ["*"]


def test_build_app_configures_asynctor_access_log() -> None:
    with patch("app.main.config_access_log") as configure:
        application = _build_app()
    configure.assert_called_once_with(application)
