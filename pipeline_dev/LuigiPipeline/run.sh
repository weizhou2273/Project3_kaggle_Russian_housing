#!/usr/bin/env bash

rm /tmp/russian_housing_model.pkl
rm /tmp/test_clean_out.csv
rm /tmp/train_clean_out.csv

timestamp(){
	date +%s
}

python luigi_ml_pipeline.py --workers 4 Predict > ./logs/"$(timestamp)"-luigi-log.txt