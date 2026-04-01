#!/bin/bash

mkdir -p data/

echo "Downloading openfoodfacts data..."
wget https://static.openfoodfacts.org/data/openfoodfacts-products.jsonl.gz -O "data/openfoodfacts-products.jsonl.gz"

echo "Done!"

