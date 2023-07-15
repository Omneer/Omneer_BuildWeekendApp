# Omneer Build Weekend App

Backend service for Omneer diagnostic service for neurodegenerative diseases.

## How it works 

XGBoostClassifier model with weights saved in data/weights returns predictions through FastAPI for whether given patient has neurodegenrative disease or not.

## Build and run Docker

```shell
sh ./scripts/build_docker.sh
```