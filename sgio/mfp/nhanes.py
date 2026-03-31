import os
import json
import urllib.request
from io import StringIO
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

basepath = Path(__file__).parent.parent / "data/nhanes"
script_dir = Path(__file__).parent


def main():
	year_code = "2017"

	# Step 1: fetch metadata (produces nhanes_cols.csv needed by download step)
	file_codes = [
		"DR1IFF_J",
		"DR2IFF_J",
		"DEMO_J",
		"DR1TOT_J",
		"DR2TOT_J",
		"WHQ_J",
	]
	(basepath / year_code).mkdir(parents=True, exist_ok=True)
	meta = get_nhanes_meta(year_code, file_codes)
	get_nhanes_cols(meta, year_code)

	# Step 2: download raw files and convert to CSV
	download_convert_nhanes(year_code)

	# Step 3: process into final outputs
	process_foods(year_code)
	process_subjects(year_code)


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def get_nhanes_meta(year_code, file_codes):
	meta = []
	for file_code in tqdm(file_codes):
		url = f"https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{year_code}/DataFiles/{file_code}.htm"
		print(f"Getting metadata from {url}")
		response = requests.get(url)
		response.raise_for_status()

		soup = BeautifulSoup(response.text, "html.parser")
		col_elems = soup.find_all("h3", class_="vartitle")

		meta_ = [_process_col_meta(elem.parent, file_code) for elem in col_elems]
		meta.extend(meta_)

	with open(basepath / year_code / "nhanes_meta.json", "w") as f:
		json.dump(meta, f, indent="\t")

	return meta


def get_nhanes_cols(meta, year_code):
	cols = [(m["col"], m["col_id"]) for m in meta]

	manual_list = [
		("DRXFDCD", "food_code"),
		("DRXFCSD", "food_desc_short"),
		("DRXFCLD", "food_desc_long"),
	]
	cols.extend(manual_list)

	col_df = pd.DataFrame(cols, columns=["abbrev", "col"])
	col_df = col_df.drop_duplicates()
	col_df.to_csv(basepath / year_code / "nhanes_cols.csv", index=False)

	return col_df


def _process_col_meta(elem, file_code):
	def get_dt_value(label):
		if dt_elem := elem.find("dt", string=label):
			value = dt_elem.find_next_sibling("dd").text
			value = _replace_multiple(value.strip(), _whitespace_replacements)
			return value
		return ""

	col = get_dt_value("Variable Name: ")
	col_name = get_dt_value("SAS Label: ")
	question = get_dt_value("English Text: ")
	instructions = get_dt_value("English Instructions: ")
	target_pop = get_dt_value("Target: ")

	title_elem = elem.find("h3", class_="vartitle")
	col_id = _title_to_col(title_elem.text)

	table_elem = elem.find("table", class_="values")
	if table_elem:
		table_df = pd.read_html(StringIO(str(table_elem)))[0]
		table_df.rename(
			columns={
				"Code or Value": "value",
				"Value Description": "description",
				"Count": "count",
				"Cumulative": "cumulative",
				"Skip to Item": "skip_to",
			},
			inplace=True,
		)
		table_df = table_df[["value", "description", "count"]]
		table = table_df.to_dict(orient="records")
		categorical = "Range of Values" not in table_df.description.values
	else:
		table = []
		categorical = None

	return {
		"col": col,
		"col_id": col_id,
		"col_name": col_name,
		"file_code": file_code,
		"question": question,
		"instructions": instructions,
		"target_pop": target_pop,
		"categorical": categorical,
		"values": table,
	}


_whitespace_replacements = {
	"\n": " ",
	"\r": "",
	"\t": " ",
	"   ": " ",
	"  ": " ",
}

_col_replacements = {
	" ": "_",
	"(": "",
	")": "",
	"-": "_",
	",": "",
	".": "",
	":": "",
	"+": "plus",
	"#": "num",
	"?": "",
	"'": "",
	"__": "_",
	"w/": "with",
	"b/w": "btwn",
	"/": "_or_",
}


def _replace_multiple(text, replacements):
	for i, j in replacements.items():
		text = text.replace(i, j)
	return text


def _title_to_col(title):
	col = title.split(" - ")[1:]
	col = " ".join(col)
	col = col.strip().lower()
	col = _replace_multiple(col, _col_replacements)
	return col


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

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
		path = _download_nhanes(year_code, file_code)

		print("Converting", file_name)
		_convert_nhanes(path, file_name, year_code)


def _download_nhanes(year_code, file_code):
	url = f"https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{year_code}/Datafiles/{file_code}.xpt"
	path = basepath / year_code / "rawdata" / (file_code + ".xpt")
	print(f"Downloading {url} to {path}")
	urllib.request.urlretrieve(url, path)
	return path


def _convert_nhanes(path, file_name, year_code):
	col_name_lookup = _get_col_lookup(year_code)
	data = pd.read_sas(path, encoding="ascii")
	data = data.rename(columns=col_name_lookup)
	num_cols = data.select_dtypes(include=[int, float]).columns
	data[num_cols] = data[num_cols].map(lambda x: 0 if 0 < x < 1e-10 else x)
	data.to_csv(basepath / year_code / "cleaned" / (file_name + ".csv"), index=False)


def _get_col_lookup(year_code):
	df = pd.read_csv(basepath / year_code / "nhanes_cols.csv")
	return df.set_index("abbrev")["col"].to_dict()


# ---------------------------------------------------------------------------
# Process
# ---------------------------------------------------------------------------

def process_foods(year_code):
	day1 = pd.read_csv(basepath / year_code / "cleaned" / "day1_interview.csv")
	day1.insert(1, "day", 1)
	day2 = pd.read_csv(basepath / year_code / "cleaned" / "day2_interview.csv")
	day2.insert(1, "day", 2)
	foods = pd.concat([day1, day2], ignore_index=True)

	selected_cols = _get_selected_cols("interview")
	foods = foods[selected_cols]
	foods = _decode_binary(foods)

	foods.to_csv(basepath / year_code / "processed" / "foods.csv", index=False)


def process_subjects(year_code):
	totals = pd.read_csv(basepath / year_code / "cleaned" / "day1_total.csv")
	totals = totals[_get_selected_cols("total")]

	demographics = pd.read_csv(basepath / year_code / "cleaned" / "demographics.csv")
	demographics = demographics[_get_selected_cols("demographics")]

	weight_history = pd.read_csv(basepath / year_code / "cleaned" / "weight_history.csv")
	weight_history = weight_history[_get_selected_cols("weight_history")]

	subjects = pd.merge(totals, demographics, on="respondent_sequence_number")
	subjects = pd.merge(subjects, weight_history, on="respondent_sequence_number")
	subjects = _decode_binary(subjects)

	subjects.to_csv(basepath / year_code / "processed" / "subjects.csv", index=False)


def _get_selected_cols(file_id):
	with open(script_dir / "nhanes_selected_cols.json") as f:
		data = json.load(f)
	return data[file_id]


def _decode_binary(df):
	with open(script_dir / "nhanes_categoricals_manual.json") as f:
		meta_dict = json.load(f)
	for col, vals in meta_dict.items():
		if col not in df.columns:
			continue
		if set(vals.values()) <= {True, False, None}:
			vals = {int(k): v for k, v in vals.items()}
			df[col] = df[col].apply(lambda x: vals.get(x, False))
	return df


if __name__ == "__main__":
	main()
