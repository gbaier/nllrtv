from setuptools import setup

name = 'nllrtv'
version = '1.0'

setup(
    name="nllrtv",
    version="1.0",
    description="code for nonlocal low-rank SAR stack despeckling",
    author="Gerald Baier",
    author_email="gerald.baier@riken.jp",
    packages=["nllrtv", "nllrtv.nltensor", "nllrtv.data"],
    package_data={"nllrtv": ['data/*.npy']},
    command_options={
        'build_sphinx': {
            'project': ('setup.py', name),
            'version': ('setup.py', version),
            'source_dir': ('setup.py', 'doc')}},
)
