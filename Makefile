.PHONY: dev init-db seed test compile

dev:
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8010

init-db:
	python scripts/init_db.py

seed:
	python scripts/seed_demo_data.py

test:
	pytest -q

compile:
	python -m compileall app scripts tests

