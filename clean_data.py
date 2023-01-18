import csv
from pathlib import Path

MAIN_PATH = Path("data/neo4j_db/raw_data")
files = ["Books.csv", "Users.csv", "Ratings.csv"]


if __name__ == "__main__":

    for file in files:
        # with open(MAIN_PATH / f"Short{file}", 'r') as csv_file:
        with open(MAIN_PATH / f"{file}", 'r') as csv_file:
            data = list(csv.reader(csv_file, delimiter=','))

        print(f"{len(data)=}")
        print(data[0])
        print(data[1])

        if file == "Books.csv":
            data = [
                (
                    row[0].replace('"', '').replace("\\", ""),
                    row[1].replace('"', '').replace("\\", ""),
                    row[2].replace('"', '').replace("\\", ""),
                    row[3],
                    row[4].replace('"', '').replace("\\", ""),
                    row[5],
                    row[6],
                    row[7]
                )
                for row in data
            ]
        elif file == "Users.csv":
            data = [
                (
                    row[0],
                    row[1].replace('"', '').replace("\\", ""),
                    row[2]
                )
                for row in data
            ]
        elif file == "Ratings.csv":
            data = [
                (
                    row[0],
                    row[1].replace('"', '').replace("\\", ""),
                    row[2]
                )
                for row in data
            ]

        # with open(MAIN_PATH / f"clean_{file.lower()}", 'w') as csv_file:
        with open(MAIN_PATH / f"clean_full_{file.lower()}", 'w') as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            writer.writerows(data)
