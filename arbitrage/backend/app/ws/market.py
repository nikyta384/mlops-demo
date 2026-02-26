"""
WebSocket – real-time market data streaming.

Protocol:
  Client → Server:  { "action": "subscribe", "connector_long": "...",
                       "connector_short": "...", "token": "..." }
  Server → Client:  { "type": "market_update", "data": <MarketSnapshot> }
               or   { "type": "error", "message": "..." }

The server fetches a fresh snapshot every WS_BROADCAST_INTERVAL seconds and
broadcasts it to the connected client.
"""

from __future__ import annotations

import asyncio
import json
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.core.config import settings
from app.services.market_data import market_data_service

log = logging.getLogger(__name__)
router = APIRouter()


@router.websocket("/ws/market")
async def market_ws(websocket: WebSocket) -> None:
    await websocket.accept()
    sub: dict | None = None

    try:
        # First message must be a subscribe command
        raw = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
        msg = json.loads(raw)

        if msg.get("action") != "subscribe":
            await websocket.send_json(
                {"type": "error", "message": "First message must be a subscribe action"}
            )
            await websocket.close()
            return

        connector_long = msg.get("connector_long", "")
        connector_short = msg.get("connector_short", "")
        token = msg.get("token", "")

        if (
            connector_long not in settings.SUPPORTED_CONNECTORS
            or connector_short not in settings.SUPPORTED_CONNECTORS
            or token not in settings.SUPPORTED_TOKENS
        ):
            await websocket.send_json(
                {"type": "error", "message": "Invalid connector or token"}
            )
            await websocket.close()
            return

        sub = {
            "connector_long": connector_long,
            "connector_short": connector_short,
            "token": token,
        }
        await websocket.send_json({"type": "subscribed", "params": sub})

        # Broadcast loop
        while True:
            try:
                snapshot = await market_data_service.get_snapshot(
                    connector_long, connector_short, token
                )
                await websocket.send_text(
                    json.dumps(
                        {
                            "type": "market_update",
                            "data": snapshot.model_dump(),
                        },
                        default=str,
                    )
                )
            except WebSocketDisconnect:
                raise
            except Exception as exc:  # noqa: BLE001
                log.warning("WS snapshot error: %s", exc)
                await websocket.send_json({"type": "error", "message": str(exc)})

            await asyncio.sleep(settings.WS_BROADCAST_INTERVAL)

    except WebSocketDisconnect:
        log.info("WebSocket client disconnected (sub=%s)", sub)
    except asyncio.TimeoutError:
        await websocket.close(code=4008)
    except Exception as exc:  # noqa: BLE001
        log.exception("Unexpected WS error: %s", exc)
        try:
            await websocket.close(code=1011)
        except Exception:  # noqa: BLE001
            pass
