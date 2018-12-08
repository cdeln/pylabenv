
.PHONY: build clean install deploy

build:
	python3 setup.py sdist bdist_wheel

clean:
	rm -rf build
	rm -rf dist
	rm -rf *.egg-info

install:
	python3 setup.py install

deploy:
	twine upload --repository-url https://test.pypi.org/legacy/ dist/* --verbose
