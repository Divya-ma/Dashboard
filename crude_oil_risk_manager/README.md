# Crude Oil Risk Manager

## Overview

Crude Oil Risk Manager is a standalone, single-user portfolio risk
management application purpose-built for a professional crude oil futures
trader. It tracks live and historical positions across crude oil
contracts and structures, and surfaces P&L, exposure, correlation, and
Value-at-Risk metrics through a Dash-based web dashboard, with configurable
in-app and Microsoft Teams alerts for risk limit breaches and contract
rolls. The application is designed to run either locally, as a Docker
container, or as a Windows Service for continuous unattended operation.

## Architecture

The project is organized into modular packages, each with a single
responsibility:

```
crude_oil_risk_manager/
├── core/            # Business logic: P&L, exposure, correlation, VaR, regression, alerts
│   └── models/      # Domain models (Contract, Structure, Position, ...)
├── adapters/         # Pluggable market data sources
│   ├── base.py       # Adapter interface
│   ├── live/          # Live vendor adapter
│   ├── historical/    # Historical vendor adapter
│   └── mock/           # Mock adapters for local dev/testing
├── db/               # SQLite schema and repository layer
├── data/historical/  # Cached historical data files
├── ui/               # Dash application, layouts, and callbacks
├── config/           # Environment-driven application settings
├── service/          # Windows Service installation scripts (NSSM)
└── tests/            # Test suite
```

Business logic in `core/` is kept independent of the UI and data adapters,
so the same risk calculations can be tested and reused regardless of
where market data originates or how results are displayed. `adapters/`
abstracts live and historical data sources behind a common interface,
allowing the vendor implementation to be swapped for the `mock/` adapters
during local development.

## Setup

### Local development

1. Clone the repository and move into the project directory.
2. Copy the environment template and fill in real values:
   ```bash
   cp .env.example .env
   ```
3. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```
4. Run the app locally:
   ```bash
   python ui/app.py
   ```
   The dashboard will be available at `http://localhost:8050` by default.

### Running as a Windows Service (NSSM)

1. Install [NSSM](https://nssm.cc/) and ensure `nssm.exe` is on your `PATH`.
2. From an elevated command prompt, run:
   ```bash
   service\nssm_install.bat
   ```
   This registers a service named `CrudeOilRiskManager` that starts
   `ui/app.py`, logs stdout/stderr to `logs/`, and restarts automatically
   5 seconds after any failure.
3. Start the service:
   ```bash
   nssm start CrudeOilRiskManager
   ```
4. To remove the service:
   ```bash
   service\nssm_uninstall.bat
   ```

## Opening the dashboard from another PC

By default `python ui/app.py` binds to `0.0.0.0:8050`, so it already listens on
every network interface — the only things standing between that and another
PC are (1) a firewall rule and (2) a login, since **the dashboard has no
authentication unless you set one up** (see below).

### 1. Add a login (do this first if the other PC isn't on a fully trusted network)

Set `AUTH_USERNAME` and `AUTH_PASSWORD` in `.env` (see `.env.example`). This
turns on an HTTP Basic Auth prompt for the whole app via
[`dash-auth`](https://pypi.org/project/dash-auth/). Leaving them empty keeps
the app login-free, which is fine only if you are certain every device that
can reach the machine's address is one you trust (e.g. your own two PCs on a
home network with no one else on it).

### 2. Reach the machine from the other PC

Pick based on where the other PC actually is:

- **Same office/home network (LAN)**: find this machine's local IP
  (`ipconfig` → IPv4 Address, something like `192.168.1.23`), allow inbound
  TCP `8050` in Windows Firewall, then browse to `http://192.168.1.23:8050`
  from the other PC. Simplest option, no extra software, but only works
  while both machines share that network.
- **Different network / remote (e.g. home ↔ office, or travelling)**: install
  [Tailscale](https://tailscale.com/) (free for personal/small-team use) on
  both PCs. It creates a private, encrypted mesh network between only the
  devices you sign in — no router port-forwarding, no exposing anything to
  the public internet. Once both are on it, browse to
  `http://<this-pc's-tailscale-ip>:8050` from the other PC. This is the
  recommended default when you're not sure which case you're in, since it
  works for both LAN and remote without changing anything else.
- **Formal server deployment** (a machine that's always on, for more than
  two people): run it via Docker (below) or as the Windows Service, on a
  machine reachable by whichever of the two methods above fits, and keep
  `AUTH_USERNAME`/`AUTH_PASSWORD` set.

Never forward port 8050 directly on your home/office router to the public
internet — that exposes the app (and, without a login, all positions and
P&L) to the entire internet. Tailscale (or a VPN) avoids that entirely.

## Docker

Build and run the application in a container:

```bash
docker compose up --build
```

This uses `docker-compose.yml`, which:
- Builds the image from the multi-stage `Dockerfile` (based on
  `python:3.11-slim`, served with `gunicorn`).
- Maps container port `8050` to host port `8050`.
- Mounts `./data` and `./db` as volumes so historical data and the
  SQLite database persist across container restarts.
- Loads configuration from `.env`.
- Restarts automatically (`restart: always`).

## Development Notes

### Running tests

```bash
pytest
```

Shared fixtures live in `tests/conftest.py`.

### Adding a new data adapter

1. Implement the adapter interface defined in `adapters/base.py`.
2. Place live vendor implementations under `adapters/live/` and
   historical vendor implementations under `adapters/historical/`.
3. For local development or testing without a real vendor connection,
   add or extend the corresponding adapter in `adapters/mock/`.
4. Wire the new adapter into configuration via `config/settings.py` so
   it can be selected without code changes elsewhere in the app.
