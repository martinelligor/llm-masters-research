SHELL = /bin/bash

define PRINT_HELP_PYSCRIPT
import re, sys

print("Please use 'make <target>' where <target> is one of\n")
for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
print("\nCheck the Makefile for more information")
endef
export PRINT_HELP_PYSCRIPT

.PHONY: help
.DEFAULT_GOAL := help
help:
	python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

.PHONY: install
install:
	poetry install

.PHONY: upgrade
upgrade:
	# poetry update
	poetry up
	poetry sort

.PHONY: package
package:
	rm -rf dist/
	poetry build

.PHONY: deploy-docker
deploy-docker:
	docker-compose up -d --build

.PHONY: delete-docker
delete-docker:
	docker-compose down

.PHONY: deploy-redis
deploy-redis:
	docker run -d --name redis -p 6379:6379  redis/redis-stack-server:latest

.PHONY: insert-data
insert-data:
	 /bin/bash ${PWD}/scripts/insert_data.sh

.PHONY: run-streamlit
run-streamlit:
	streamlit run ${PWD}/streamlit/chat.py

.PHONY: sleep
sleep:
	sleep 5

.PHONY: run
run: deploy-docker sleep insert-data run-streamlit
