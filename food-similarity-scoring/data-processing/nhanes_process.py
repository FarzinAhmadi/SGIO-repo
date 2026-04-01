import pandas as pd
import json

from pathlib import Path
basepath = Path(__file__).parent.parent / "data/nhanes"


def main():
	year_code = "2017"
	process_foods(year_code)
	process_subjects(year_code)

def process_foods(year_code):
	day1 = pd.read_csv(basepath / year_code / "cleaned" / "day1_interview.csv")
	day1.insert(1, "day", 1)
	day2 = pd.read_csv(basepath / year_code / "cleaned" / "day2_interview.csv")
	day2.insert(1, "day", 2)
	foods = pd.concat([day1, day2], ignore_index=True)

	selected_cols = get_selected_cols(year_code, "interview")
	foods = foods[selected_cols]

	# foods = decode_categorical(foods, year_code)
	foods = decode_binary(foods, year_code)

	foods.to_csv(basepath / year_code / "processed" / "foods.csv", index=False)

def process_subjects(year_code):
	totals = pd.read_csv(basepath / year_code / "cleaned" / "day1_total.csv")
	totals_cols = get_selected_cols(year_code, "total")
	totals = totals[totals_cols]

	demographics = pd.read_csv(basepath / year_code / "cleaned" / "demographics.csv")
	demographics_cols = get_selected_cols(year_code, "demographics")
	demographics = demographics[demographics_cols]

	weight_history = pd.read_csv(basepath / year_code / "cleaned" / "weight_history.csv")
	weight_history_cols = get_selected_cols(year_code, "weight_history")
	weight_history = weight_history[weight_history_cols]

	subjects = pd.merge(totals, demographics, on="respondent_sequence_number")
	subjects = pd.merge(subjects, weight_history, on="respondent_sequence_number")

	# subjects = decode_categorical(subjects, year_code)
	subjects = decode_binary(subjects, year_code)

	subjects.to_csv(basepath / year_code / "processed" / "subjects.csv", index=False)

def decode_categorical(df, year_code):
	with open(basepath / year_code / "categoricals_manual.json", "r") as f:
		meta_dict = json.load(f)
	for col, vals in meta_dict.items():
		if col not in df.columns:
			continue
		vals = {int(k): v for (k,v) in vals.items()}
		df[col] = df[col].apply(lambda x: vals.get(x, ""))
	return df

def decode_binary(df, year_code):
	with open(basepath / year_code / "categoricals_manual.json", "r") as f:
		meta_dict = json.load(f)
	for col, vals in meta_dict.items():
		if col not in df.columns:
			continue
		if set.issubset(set(vals.values()), {True, False, None}):
			vals = {int(k): v for (k,v) in vals.items()}
			df[col] = df[col].apply(lambda x: vals.get(x, False))
	return df

def get_selected_cols(year_code, file_id):
	with open(basepath / year_code / "selected_cols.json") as f:
		data = json.load(f)
	return data[file_id]

if __name__ == "__main__":
	main()
