rm -r build/ dist/ *.egg-info/

pip install wheel twine
python setup.py bdist_wheel
twine upload --repository pypi dist/*
