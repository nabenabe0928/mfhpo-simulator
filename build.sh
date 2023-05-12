rm -r build/ dist/ repo_name.egg-info/

pip install wheel twine
python setup.py bdist_wheel
twine upload --repository pypi dist/*
