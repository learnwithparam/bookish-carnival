.PHONY: help install setup-data dev run clean

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies using uv
	@echo "📦 Installing dependencies..."
	@uv sync

setup-data: ## Initialize the database (downloads data if needed)
	@echo "🗄️ Initializing data and database..."
	@uv run python download_data.py
	@uv run python db_init.py

dev: ## Run with Docker Compose
	@echo "🐳 Starting Docker..."
	@docker compose up --build

run: ## Run locally
	@echo "🚀 Running Chainlit app locally..."
	@uv run chainlit run app.py -w

clean: ## Clean up artifacts
	@echo "🧹 Cleaning up..."
	@rm -rf .venv
	@find . -type d -name "__pycache__" -exec rm -rf {} +

test-notebook: ## Run the notebook to verify it executes correctly
	@echo "🧪 Testing notebook..."
	@uv run jupyter nbconvert --to notebook --execute --stdout multi-agent-chatbot.ipynb > /dev/null

run-notebook: ## Start JupyterLab
	@echo "🚀 Starting JupyterLab..."
	@uv run jupyter lab

validate-notebook: ## Validate notebook format
	@echo "🔍 Validating notebook..."
	@uv run jupyter nbconvert --to notebook --stdout multi-agent-chatbot.ipynb > /dev/null

