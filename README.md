# LangChain Chat Application

A full-stack chat application with machine learning capabilities, built with FastAPI, React, and Docker.

## Prerequisites

- Docker (v20.10+)
- Docker Compose (v2.0+)
- Git

## Getting Started

### Clone the Repository

```bash
git clone <repository-url>
cd ml_langchain_chat
```

## Running with Docker Compose

### Start the Application

To start all services in detached mode (runs in the background):

```bash
docker compose up -d
```

### Stop the Application

To stop all running services:

```bash
docker compose down
```

### View Logs

View logs for all services:

```bash
docker compose logs -f
```

View logs for a specific service (backend, frontend, or llm):

```bash
docker compose logs -f backend
docker compose logs -f frontend
docker compose logs -f llm
```

## Services

### 1. Frontend

- **URL**: http://localhost:3000
- **Description**: React-based user interface

### 2. Backend API

- **URL**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Endpoints**:
  - `GET /health` - Check service status
  - `POST /train` - Train a new model
  - `POST /predict` - Make predictions

### 3. Jupyter Notebooks

- **URL**: http://localhost:8888
- **Token**: `dev`
- **Description**: Interactive development environment for data science

## Development

### Rebuilding Services

After making changes to the code, rebuild and restart the services:

```bash
docker compose up -d --build
```

### Training the Model

The application comes with a pre-trained model, but you can train a new one:

```bash
curl -X POST http://localhost:8000/train
```

### Making Predictions

Example API call to make a prediction:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [1, 2]}'
```

## Troubleshooting

### Check Service Status

```bash
docker compose ps
```

### View Service Logs

```bash
docker compose logs <service_name>
```

### Clean Up

To stop all services and remove containers, networks, and volumes:

```bash
docker compose down -v
```
