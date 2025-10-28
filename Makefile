.PHONY: lint test check

lint:
	flake8 wurun.py test_wurun.py --max-line-length=120 --ignore=E203,W503

test:
	pytest test_wurun.py -v

check: lint test
	@echo "All checks passed!"