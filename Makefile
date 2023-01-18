black:
	poetry run black $(ARGS) ./

isort:
	poetry run isort $(ARGS) ./

flake:
	poetry run autoflake --in-place --recursive --ignore-init-module-imports --remove-unused-variables --remove-all-unused-imports ./

mypy:
	poetry run mypy --no-site-packages --ignore-missing-imports --no-strict-optional ./

format:	flake black isort mypy # run all formatters at once
