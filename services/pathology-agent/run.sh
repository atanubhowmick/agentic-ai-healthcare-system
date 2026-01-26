#!/bin/bash

uvicorn main:app --app-dir ./src --host 127.0.0.1 --port 8003
