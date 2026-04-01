import os
from pathlib import Path
import urllib.request
import pandas as pd

basepath = Path(__file__).parent.parent / "data/nhanes/"


def main():
	year_code = "2017"
	download_convert_nhanes(year_code)

def download_convert_nhanes(year_code):
	os.makedirs(basepath / year_code / "rawdata", exist_ok=True)
	os.makedirs(basepath / year_code / "cleaned", exist_ok=True)
	os.makedirs(basepath / year_code / "processed", exist_ok=True)

	files = [
		("DR1IFF_J", "day1_interview"),
		("DR2IFF_J", "day2_interview"),
		("DEMO_J", "demographics"),
		("DR1TOT_J", "day1_total"),
		("DR2TOT_J", "day2_total"),
		("DRXFCD_J", "food_codes"),
		("WHQ_J", "weight_history"),
	]

	for file_code, file_name in files:
		print("Downloading", file_name)
		path = download_nhanes(year_code, file_code)

		print("Converting", file_name)
		convert_nhanes(path, file_name, year_code)

def download_nhanes(year_code, file_code):
	url = f"https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{year_code}/Datafiles/{file_code}.xpt"
	path = basepath / year_code / "rawdata" / (file_code + ".xpt")
	print(f"Downloading {url} to {path}")
	urllib.request.urlretrieve(url, path)
	return path

def convert_nhanes(path, file_name, year_code):
	col_name_lookup = get_col_lookup(year_code)
	data = pd.read_sas(path, encoding="ascii")
	data = data.rename(columns=col_name_lookup)
	num_cols = data.select_dtypes(include=[int, float]).columns
	data[num_cols] = data[num_cols].map(lambda x: 0 if 0 < x and x < 1e-10 else x)
	data.to_csv(basepath / year_code / "cleaned" / (file_name + ".csv"), index=False)

def get_col_lookup(year_code):
	df = pd.read_csv(basepath / year_code / "nhanes_cols.csv")
	lookup = df.set_index("abbrev")["col"].to_dict()
	return lookup

if __name__ == "__main__":
	main()
