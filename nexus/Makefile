install-dev:
	@echo "Installing development packages..."
	uv pip install -e ".[dev]"

install-hooks: install-dev
	@echo "Installing pre-commit git hooks..."
	uv run pre-commit install

format:
	uv run ruff format . --preview

lint:
	uv run ruff check . --unsafe-fixes --preview

typecheck:
	uv run ty check nexus services/apps services/packages

check: format lint typecheck

test:
	uv run pytest

test-cov:
	uv run pytest --cov=nexus --cov-report=term-missing

test-unit:
	uv run pytest tests/unit

test-integration:
	uv run pytest tests/integration

SERVICE ?= label
service-sync:
	uv sync --all-packages --all-extras

service-run:
	uv run --package nexus-$(SERVICE) nexus-$(SERVICE)

service-test:
	uv run --package nexus-$(SERVICE) --extra dev python -m pytest services/apps/$(SERVICE)/tests

service-build:
	docker build -f deployment/services/$(SERVICE)/Dockerfile -t nexus-$(SERVICE):local .

service-helm-lint:
	helm lint deployment/services/$(SERVICE)/helm/nexus-$(SERVICE)

service-helm-template:
	helm template nexus-$(SERVICE) deployment/services/$(SERVICE)/helm/nexus-$(SERVICE)

service-deploy:
	helm upgrade --install nexus-$(SERVICE) deployment/services/$(SERVICE)/helm/nexus-$(SERVICE) --set image.repository=nexus-$(SERVICE) --set image.tag=local

ci-format:
	uv run ruff format . --check --preview

ci-lint:
	uv run ruff check . --preview --no-fix

ci-check: ci-format ci-lint typecheck

DOCS_PORT ?= 30000
serve-docs:
	uv run mkdocs serve --dev-addr 0.0.0.0:$(DOCS_PORT) --livereload

docker-runtime:
	docker build -f deployment/docker/Dockerfile.runtime -t nexus-runtime .

BUNDLE_OS ?= windows
BUNDLE_ARCH ?= x86_64
BUNDLE_OUTPUT ?= build/bundle
BUNDLE_REPO ?= nexus-gui-bootstrap-bundle

.PHONY: bundle
bundle:
	$(if $(filter windows linux,$(BUNDLE_OS)),,$(error Unsupported BUNDLE_OS: $(BUNDLE_OS); supported: windows linux))
	$(if $(filter x86_64,$(BUNDLE_ARCH)),,$(error Unsupported BUNDLE_ARCH: $(BUNDLE_ARCH); supported: x86_64))
	uv run --no-sync python deployment/bootstrap/$(BUNDLE_OS)/main.py --arch "$(BUNDLE_ARCH)" --output "$(BUNDLE_OUTPUT)"

.PHONY: bundle-publish
bundle-publish:
	uv run --no-sync python deployment/bootstrap/publish.py --bundle "$(BUNDLE_OUTPUT)/$(BUNDLE_OS)-$(BUNDLE_ARCH)" --repo "$(BUNDLE_REPO)"

test-container:
	@RUNTIME=$${RUNTIME:-nexus-runtime:latest}; \
	IMAGE=$${IMAGE:?IMAGE environment variable is required}; \
	CONTAINER_NAME="nexus-runtime"; \
	if docker ps -a --format '{{.Names}}' | grep -q "^$${CONTAINER_NAME}$$"; then \
		echo "Container $${CONTAINER_NAME} already exists, skipping creation"; \
	else \
		echo "Creating runtime container $${CONTAINER_NAME} from $${RUNTIME}"; \
		docker run -d --name "$${CONTAINER_NAME}" "$${RUNTIME}" true; \
	fi; \
	echo "Starting $${IMAGE} with volumes from $${CONTAINER_NAME}"; \
	docker run --volumes-from "$${CONTAINER_NAME}" -it --rm -p 32000:32000 "$${IMAGE}" /nix/runtime/nix-env/bin/bash /nix/runtime/nexus/deployment/runtime/start.sh -v 3.13 -p 32000 -H 0.0.0.0
