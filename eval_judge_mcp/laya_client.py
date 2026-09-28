"""
laya_client.py

Loads the Laya typed-decisions checkpoint exactly once and hands back the
same agent instance on every subsequent call, instead of reloading it (a
multi-second cold load, see Phase 1 timing) on every judge call. This is the
lazy-loaded-singleton pattern: the expensive resource is built on first use
and cached in a module-level variable.

We load `typed-decisions` directly rather than going through laya.Router,
since this server only ever needs the typed-decision checkpoint -- no
language routing or general-purpose text generation is involved here.
"""

import logging
import threading

import laya

from . import config

logger = logging.getLogger(__name__)

_agent = None
_agent_lock = threading.Lock()


def get_agent():
    """Return the shared Laya agent, loading it on first call."""
    global _agent
    if _agent is None:
        with _agent_lock:
            if _agent is None:  # re-check inside the lock (another thread may have loaded it)
                logger.info(
                    "Loading Laya checkpoint %s (subfolder=%s)...",
                    config.LAYA_MODEL_ID, config.LAYA_SUBFOLDER,
                )
                _agent = laya.load(
                    config.LAYA_MODEL_ID,
                    subfolder=config.LAYA_SUBFOLDER,
                    device=config.LAYA_DEVICE,
                )
                logger.info("Laya checkpoint loaded.")
    return _agent