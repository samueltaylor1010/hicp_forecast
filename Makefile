.PHONY: app-data run export serve test

app-data:
	python scripts/build_app_data.py

run:
	shiny run --reload app/app.py

export: app-data
	shinylive export app docs --template-params scripts/shinylive_template_params.json
	touch docs/.nojekyll

serve:
	python -m http.server 8000 --directory docs

test:
	pytest -q
