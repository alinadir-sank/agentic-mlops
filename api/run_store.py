import redis
import json
from typing import Any

redis_client = redis.Redis(
    host="localhost",
    port=6379,
    decode_responses=True,
)

RUN_PREFIX = "mlops:run:"
RUN_INDEX = "mlops:runs"

def save_run(thread_id: str, run_data: dict[str, Any]) -> None:
    key = f"{RUN_PREFIX}{thread_id}"

    redis_client.set(
        key,
        json.dumps(run_data),
    )

    # maintain sorted index by creation/update time
    created_at = run_data.get("created_at_ts", 0)

    redis_client.zadd(
        RUN_INDEX,
        {thread_id: created_at},
    )

def get_run(thread_id: str) -> dict[str, Any] | None:
    key = f"{RUN_PREFIX}{thread_id}"

    data = redis_client.get(key)

    if not data:
        return None

    return json.loads(data)

def list_all_runs(limit: int = 100) -> list[dict]:
    ids = redis_client.zrevrange(
        RUN_INDEX,
        0,
        limit - 1,
    )

    runs = []

    for thread_id in ids:
        run = get_run(thread_id)

        if run:
            runs.append(run)

    return runs