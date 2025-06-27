"""
Health check
"""

import json
from typing import Any


def healthcheck() -> dict[str, Any]:
    """health check function."""
    return {"status": "UP"}


if __name__ == "__main__":
    result = healthcheck()
    result_json = json.dumps(result)
    print(result)
    exit(0)
