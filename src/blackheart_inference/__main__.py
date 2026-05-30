"""``python -m blackheart_inference`` entry — uvicorn launcher.

Reads ``Settings`` once at startup so a bad config fails before the
event loop opens. Single worker on purpose: model bytes live in the
process and a multi-worker setup would multiply RAM by N.
"""

from __future__ import annotations

import uvicorn

from .main import create_app
from .settings import Settings


def main() -> None:
    settings = Settings()  # type: ignore[call-arg]
    settings.assert_prod_safe()
    app = create_app(settings)
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        log_config=None,  # structlog owns logging
        workers=1,
    )


if __name__ == "__main__":
    main()
