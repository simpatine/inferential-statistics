import pandas as pd
import argparse


def clean_energy_data(data, year_period = [2020, 2025]):

    # Remove every row that is not a known state, I supposed that these are the ones with no ISO code
    data_new = data[data["iso_code"].notna()]

    # Remove every row that falls outside the specified period
    data_new = data_new[(year_period[0] <= data["year"]) & (data["year"] <= year_period[1])]

    # Remove every not useful column

    useful_columns = {
        "country",
        "year",
        "iso_code",
        "biofuel_consumption",
        "biofuel_electricity",
        "biofuel_share_elec",
        "biofuel_share_energy"
    }

    data_new = data_new.filter(items = useful_columns)

    return data_new


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Energy transition and public health comparator')
    parser.add_argument("--energy-data-path", type=str, default="data/owid-energy-data.csv", help="Path to the energy usage of each country")
    
    parser.add_argument("--raw-data", default=False, action="store_true")
    
    parser.add_argument("--verbose", default=False, action="store_true")


    args = parser.parse_args()

    energy_data = pd.read_csv(args.energy_data_path)

    if not args.raw_data:
        if args.verbose:
            print("Cleaning up data...")

        energy_data = clean_energy_data(energy_data)
        print(energy_data[energy_data["iso_code"] == "USA"])

        if args.verbose:
            print("Data cleaned.")