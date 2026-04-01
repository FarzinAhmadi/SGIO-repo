import pandas as pd
import curl_cffi.requests as requests

from pathlib import Path
basepath = Path(__file__).parent.parent


def main():
	year_code = "2017-2018"
	data_path = basepath / f"data/usda/{year_code}/"

	data_path.mkdir(exist_ok=True, parents=True)
	(data_path/"rawdata").mkdir(exist_ok=True)
	(data_path/"processed").mkdir(exist_ok=True)

	download_usda(year_code, data_path)
	process_descriptions(data_path)
	servings_data = process_servings(data_path)
	process_nutrition(servings_data, data_path)

def download_usda(year_code, data_path):
	file_meta = [
		(f"https://www.ars.usda.gov/ARSUserFiles/80400530/apps/{year_code}%20FNDDS%20At%20A%20Glance%20-%20Foods%20and%20Beverages.xlsx", "descriptions"),
		(f"https://www.ars.usda.gov/ARSUserFiles/80400530/apps/{year_code}%20FNDDS%20At%20A%20Glance%20-%20Portions%20and%20Weights.xlsx", "servings"),
		(f"https://www.ars.usda.gov/ARSUserFiles/80400530/apps/{year_code}%20FNDDS%20At%20A%20Glance%20-%20FNDDS%20Nutrient%20Values.xlsx", "nutrient_values"),
	]

	for url, fn in file_meta:
		print(f"Downloading {url} to {data_path/f'rawdata/{fn}.xlsx'}")
		response = requests.get(url, impersonate="chrome", timeout=60)
		response.raise_for_status()  # Raise an error for bad status codes
		with open(data_path/f"rawdata/{fn}.xlsx", "wb") as f:
			f.write(response.content)

def process_descriptions(data_path):
	desc_data = pd.read_excel(data_path/"rawdata/descriptions.xlsx", skiprows=1, engine="openpyxl")
	desc_data.drop(columns=["Additional food description"], inplace=True)
	desc_data.rename(
		columns={
			"Food code": "food_code",
			"Main food description": "food_desc",
			"WWEIA Category number": "category_code",
			"WWEIA Category description": "category_desc",
		},
		inplace=True,
	)
	desc_data.to_csv(data_path/"processed/descriptions.csv", index=False)
	return desc_data

def process_servings(data_path):
	servings_data = pd.read_excel(data_path/"rawdata/servings.xlsx", skiprows=1)
	servings_data = servings_data[servings_data["Seq num"] == 1]
	servings_data.rename(
		columns={
			"Food code": "food_code",
			"Portion weight (g)": "serving_size_g",
		},
		inplace=True,
	)
	servings_data = servings_data[["food_code", "serving_size_g"]]
	servings_data.to_csv(data_path/"processed/servings.csv", index=False)
	return servings_data

def process_nutrition(servings_data, data_path):
	nutrition_data = pd.read_excel(data_path/"rawdata/nutrient_values.xlsx", skiprows=1)
	nutrition_data.rename(columns={"Food code": "food_code"}, inplace=True)

	nutrition_cols = {
		"Energy (kcal)": "energy_kcal",
		"Protein (g)": "protein_gm",
		"Carbohydrate (g)": "carbohydrate_gm",
		"Sugars, total\n(g)": "total_sugars_gm",
		"Fiber, total dietary (g)": "dietary_fiber_gm",
		"Total Fat (g)": "total_fat_gm",
		"Fatty acids, total saturated (g)": "total_saturated_fatty_acids_gm",
		"Cholesterol (mg)": "cholesterol_mg",
		"Iron\n(mg)": "iron_mg",
		"Sodium (mg)": "sodium_mg",
	}
	nutrition_data.rename(columns=nutrition_cols, inplace=True)

	nutrition_data = nutrition_data.merge(servings_data, on="food_code", how="left")
	nutrition_data = nutrition_data[["food_code", "serving_size_g"] + list(nutrition_cols.values())]

	nutrition_data.to_csv(data_path/"processed/nutrient_values.csv", index=False)
	return nutrition_data

if __name__ == "__main__":
	main()
