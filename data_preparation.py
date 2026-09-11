import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def prepare_dataset():
    city = pd.read_excel("City.xlsx")
    country = pd.read_excel("Country.xlsx")
    region = pd.read_excel("Region.xlsx")
    continent = pd.read_excel("Continent.xlsx")
    item = pd.read_excel("Item.xlsx")
    type_df = pd.read_excel("Type.xlsx")
    mode = pd.read_excel("Mode.xlsx")
    transaction = pd.read_excel("Transaction.xlsx")
    user = pd.read_excel("User.xlsx")

    items_full = item.merge(type_df, on="AttractionTypeId", how="left")
    items_full = items_full.merge(city, left_on="AttractionCityId", right_on="CityId", how="left")
    items_full = items_full.merge(country, on="CountryId", how="left")
    items_full = items_full.merge(region, on="RegionId", how="left")
    items_full = items_full.merge(continent, on="ContinentId", how="left")

    trans_full = transaction.merge(user, on="UserId", how="left")
    trans_full = trans_full.merge(items_full, on="AttractionId", how="left")
    trans_full = trans_full.merge(mode, left_on="VisitMode", right_on="VisitModeId", how="left")

    trans_full = trans_full.dropna(subset=["UserId", "AttractionId", "Rating"])
    for col in ["CityName", "Country", "Region", "Continent", "AttractionType", "VisitMode_y"]:
        if col in trans_full.columns:
            trans_full[col] = trans_full[col].fillna("Unknown")

    label_encoders = {}
    for col in ["VisitMode_y", "CityName", "Country", "Region", "Continent", "AttractionType"]:
        if col in trans_full.columns:
            le = LabelEncoder()
            le.fit(trans_full[col].astype(str))
            label_encoders[col] = le

    scaler = StandardScaler()
    trans_full["Rating_scaled"] = scaler.fit_transform(trans_full[["Rating"]])

    final_dataset = trans_full[[
        "UserId", "AttractionId", "Attraction", "Rating", "Rating_scaled",
        "VisitMode_y", "CityName", "Country", "Region", "Continent", "AttractionType"
    ]].rename(columns={"VisitMode_y": "VisitMode"})

    return final_dataset, label_encoders, scaler

if __name__ == "__main__":
    dataset, encoders, scaler = prepare_dataset()
    print(dataset.head())
    print(dataset.dtypes)
