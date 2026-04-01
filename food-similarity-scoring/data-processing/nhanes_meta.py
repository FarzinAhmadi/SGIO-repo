import requests
from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm
import json
from io import StringIO

from pathlib import Path
basepath = Path(__file__).parent.parent / "data/nhanes/"


def main():
	year_code = "2017"
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
	process_categoricals(meta, year_code)

def get_nhanes_meta(year_code, file_codes):
	meta = []
	for file_code in tqdm(file_codes):
		url = f"https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{year_code}/DataFiles/{file_code}.htm"
		print(f"Getting metadata from {url}")
		response = requests.get(url)
		response.raise_for_status()

		soup = BeautifulSoup(response.text, "html.parser")
		col_elems = soup.find_all("h3", class_="vartitle")

		meta_ = [process_col_meta(elem.parent, file_code) for elem in col_elems]
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

def process_col_meta(elem, file_code):
	def get_dt_value(label):
		if dt_elem := elem.find("dt", string=label):
			value = dt_elem.find_next_sibling("dd").text
			value = replace_multiple(value.strip(), whitespace_replacements)
			return value
		else:
			return ""

	col = get_dt_value("Variable Name: ")
	col_name = get_dt_value("SAS Label: ")
	question = get_dt_value("English Text: ")
	instructions = get_dt_value("English Instructions: ")
	target_pop = get_dt_value("Target: ")

	title_elem = elem.find("h3", class_="vartitle")
	col_id = title_to_col(title_elem.text)

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

def process_categoricals(meta, year_code):
	def parse_val(v):
		val = v["value"]
		desc = v["description"]
		if desc == "Yes":
			desc = True
		elif desc == "No":
			desc = False
		elif desc == "Missing" or desc == "Don't Know" or desc == "Don't know" or desc == "Refused":
			desc = None
		elif desc[-1] == "," or desc[-1] == "?":
			desc = desc[:-1]
		elif desc[-4:] == ", or":
			desc = desc[:-4]
		return (val, desc)
	def parse_values(vals):
		vals = [parse_val(v) for v in vals]
		vals = {v[0]:v[1] for v in vals}
		del vals["."]
		if len(vals) == 1:
			key = list(vals.keys())[0]
			vals = {key: True}
		return vals

	meta = {m["col_id"]: parse_values(m["values"]) for m in meta if m["categorical"]}
	with open(basepath / year_code / "categoricals.json", "w") as f:
		json.dump(meta, f, indent="\t")

def replace_multiple(text, replacements):
	for i, j in replacements.items():
		text = text.replace(i, j)
	return text

whitespace_replacements = {
	"\n": " ",
	"\r": "",
	"\t": " ",
	"   ": " ",
	"  ": " ",
}

col_replacements = {
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

def title_to_col(title):
	col = title.split(" - ")[1:]
	col = " ".join(col)
	col = col.strip()
	col = col.lower()
	col = replace_multiple(col, col_replacements)
	return col

if __name__ == "__main__":
	main()
