PKG=fran
PY_SRC=$(PKG) tests setup.py

test:
	pytest

install:
	pip install -U .

install-dev:
	pip install -r requirements.txt && pip install -e .

clean:
	rm -f *.csv **/*.pyc
	rm -rf $(PKG).egg-info/ build/ dist/ **/__pycache__/ .pytest_cache/

dist: clean lint
	python setup.py sdist bdist_wheel

release: dist
	twine upload dist/*

fmt:
	black $(PY_SRC)

lint:
	black --check $(PY_SRC)
	flake8 $(PY_SRC)

readme:
	bash scripts/update_readme.sh

patch:
	bash scripts/bump_version.sh patch

minor:
	bash scripts/bump_version.sh minor

major:
	bash scripts/bump_version.sh major
