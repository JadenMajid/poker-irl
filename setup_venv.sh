#! /bin/bash

uv venv venv 
source ./venv/bin/activate
uv pip install $(cat requirements.txt)