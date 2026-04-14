#! /bin/bash

uv venv venv 
uv pip install $(cat requirements.txt)