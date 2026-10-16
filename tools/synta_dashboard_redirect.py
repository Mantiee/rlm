"""Development entry point for the packaged compatibility redirect."""

from rlm.v100.dashboard_redirect import (
    RedirectHandler,
    RedirectServer,
    main,
)

__all__ = ["RedirectHandler", "RedirectServer", "main"]

if __name__ == "__main__":
    main()
