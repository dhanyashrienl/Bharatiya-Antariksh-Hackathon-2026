#!/bin/bash
# Simple wrapper for the download script
echo "Starting data download..."
# Example product ID and bands. 
# Users should edit these or create a config file.
PRODUCT_ID="LC9123456789"
BANDS="SR_B2,SR_B3,SR_B4,ST_B10"
START_DATE="2023-01-01"
END_DATE="2023-12-31"
OUTPUT_PATH="./input/${PRODUCT_ID}_product"
EE_PROJECT_ID="your-google-earth-engine-project-id"

python scripts/download.py $PRODUCT_ID $BANDS $START_DATE $END_DATE $OUTPUT_PATH --ee_project_id $EE_PROJECT_ID
