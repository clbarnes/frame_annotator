PKG="fran"

test:
	pytest

install:
	pip install -U .

install-dev:
	pip install -r requirements.txt && pip install -e .

clean:
	rm -f *.csv
	rm -rf $(PKG).egg-info/ build/ dist/

dist: clean lint
	python setup.py sdist bdist_wheel

release: dist
	twine upload dist/*

fmt:
	black fran test setup.py

lint:
	black --check fran test setup.py
	flake8

readme:
	bash scripts/update_readme.sh

patch:
	bash scripts/bump_version.sh patch

minor:
	bash scripts/bump_version.sh minor

major:
	bash scripts/bump_version.sh major
