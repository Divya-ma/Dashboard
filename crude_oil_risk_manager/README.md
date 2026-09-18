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
