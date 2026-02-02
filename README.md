# Text-to-SQL Agentic RAG Chatbot Workshop

Transforming Natural Language Questions into Actionable Database Insights Using Multi-Agent Architecture.

## Introduction

In this workshop, you will learn how to build an intelligent **Multi-Agent Text-to-SQL Chatbot**. This system utilizes **LangGraph** for orchestration and **Chainlit** for the UI, capable of:

- Accepting natural language queries.
- Generating and executing SQL against an e-commerce database.
- Auto-correcting errors.
- visualizing results.

## Architecture

The system uses a state machine architecture:

1.  **Guardrails Agent**: Validates scope.
2.  **SQL Agent**: Generates queries.
3.  **Executor**: Runs queries.
4.  **Error Recovery**: Fixes failed queries.
5.  **Visualization Agent**: Generates Plotly charts.

## Getting Started

### Prerequisites

- Docker
- Make
- API Keys (Google Gemini, OpenAI, or others supported by LiteLLM)

### Setup

1.  **Configure Environment**:
    ```bash
    cp .env.example .env
    # Edit .env and add your API keys
    ```

2.  **Run with Docker**:
    ```bash
    make dev
    ```
    Access the app at `http://localhost:8000`.

3.  **Run Locally**:
    ```bash
    make install
    make setup-data # Downloads dataset from Kaggle (~40MB) and initializes DB
    make run
    ```

## Workshop Tasks

- Explore `text2sql_agent.py` to understand the agent flow.
- Modify the system prompts to change agent behavior.
- Try different models by changing `DEFAULT_MODEL` in `.env`.
