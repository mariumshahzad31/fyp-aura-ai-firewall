# AURA Mobile Client (API-First)

The production backend is **OpenAPI 3** at `/docs` on the FastAPI service (default `http://localhost:8000`).

## Authentication

1. `POST /api/v1/auth/token` with form fields `username` and `password` (OAuth2 password flow).
2. Use `Authorization: Bearer <access_token>` on subsequent calls.

## Core Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Liveness |
| GET | `/api/v1/status` | Models, firewall, packet monitor, behavioral profiles |
| POST | `/api/v1/predict` | Batch scoring (`records`, `include_explanation`, `use_llm`) |
| GET | `/api/v1/alerts` | Recent alerts from shared JSONL ring |
| GET | `/api/v1/logs/tail` | Admin: tail log files |
| GET | `/api/v1/intel/cve/{cve_id}` | Dataset + NVD correlation |
| POST | `/api/v1/firewall/block` | Admin: block IPv4 |
| POST | `/api/v1/firewall/unblock` | Admin: remove block |

## Mobile Implementation Notes

- Use HTTPS in production; pin certificates or use mutual TLS for high-security deployments.
- Responses are JSON; CORS is configurable via `AURA_CORS_ORIGINS`.
- For push notifications, subscribe to your own queue fed by `logs/aura_alerts.jsonl` or a webhook you add.
