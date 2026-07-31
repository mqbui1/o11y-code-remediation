"""
Entrypoint for the o11y-code-remediation service.

Starts the OTLP receiver + profiling/remediation API in the foreground
(blocks forever). See receiver/app.py for route details and
deploy/otelcol-config-snippet.yml for how to feed it telemetry.

Env vars:
  PORT                 (default 4318)
  HOST                 (default 0.0.0.0)
  ENVIRONMENT          deployment.environment filter for AlwaysOn profiling (default "")
  LLM_PROVIDER         bedrock (default) | ollama | openai — see config.py
"""

import logging
import os
import time

from receiver.app import start_receiver

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def main() -> None:
    port = int(os.environ.get("PORT", "4318"))
    host = os.environ.get("HOST", "0.0.0.0")
    environment = os.environ.get("ENVIRONMENT", "")

    thread = start_receiver(port=port, host=host, environment=environment)
    try:
        while thread.is_alive():
            time.sleep(1)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
