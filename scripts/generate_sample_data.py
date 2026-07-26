"""
generate_sample_data.py
------------------------
WHY: Students need realistic sample CSV files to try the app immediately,
     without hunting for their own dataset.
WHAT: Creates two beginner-friendly datasets -> data/employees.csv and data/sales.csv.
HOW: Uses Python's built-in `random` and `csv` modules only, so this script
     can run even before you install the project's requirements.txt.

This script is a ONE-TIME generator. Run it with:
    python scripts/generate_sample_data.py
"""

import random
import os
import csv

# Fix the random seed so the data is the same every time we run this script.
# This makes the demo predictable for teaching purposes.
random.seed(42)

# Make sure the output folder exists before we try to save files into it.
OUTPUT_FOLDER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def generate_employees_csv(num_rows: int = 200) -> None:
    """
    WHY: Page 2/3/4/5 of the app need a simple HR-style dataset to demo profiling,
         AI insights, and the RAG "ask your dataset" feature.
    WHAT: Builds a DataFrame with employee name, age, department, salary,
          experience, performance score, city, and gender.
    HOW: Randomly samples from small lists to keep the data realistic and readable.
    """
    first_names = ["Aarav", "Vivaan", "Aditya", "Sai", "Reyansh", "Ananya", "Diya",
                   "Ishaan", "Kabir", "Myra", "John", "Emma", "Liam", "Olivia",
                   "Noah", "Sophia", "Rahul", "Priya", "Amit", "Neha", "Karan", "Pooja"]
    last_names = ["Sharma", "Verma", "Gupta", "Khan", "Reddy", "Iyer", "Nair",
                  "Smith", "Johnson", "Brown", "Patel", "Singh", "Das", "Mehta"]
    departments = ["Sales", "Engineering", "Marketing", "HR", "Finance", "Support"]
    cities = ["Mumbai", "Bangalore", "Delhi", "Pune", "Hyderabad", "Chennai"]
    genders = ["Male", "Female"]

    rows = []
    for _ in range(num_rows):
        name = f"{random.choice(first_names)} {random.choice(last_names)}"
        age = random.randint(21, 58)
        department = random.choice(departments)
        # Salary roughly depends on age/experience so the data feels believable.
        experience = max(0, age - random.randint(21, 25))
        base_salary = 25000 + experience * 1800 + random.randint(-3000, 6000)
        salary = max(20000, base_salary)
        performance_score = round(random.uniform(1.0, 5.0), 1)
        city = random.choice(cities)
        gender = random.choice(genders)

        rows.append({
            "Employee Name": name,
            "Age": age,
            "Department": department,
            "Salary": salary,
            "Experience": experience,
            "Performance Score": performance_score,
            "City": city,
            "Gender": gender,
        })

    output_path = os.path.join(OUTPUT_FOLDER, "employees.csv")
    with open(output_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Created {output_path} with {len(rows)} rows")


def generate_sales_csv(num_rows: int = 200) -> None:
    """
    WHY: Page 6 (Visualization) benefits from a numeric-heavy dataset with
         categories, so students can try bar/line/pie charts easily.
    WHAT: Builds a DataFrame with product, revenue, profit, month, region, quantity.
    HOW: Random sampling similar to generate_employees_csv().
    """
    products = ["Laptop", "Smartphone", "Headphones", "Monitor", "Keyboard",
                "Mouse", "Tablet", "Printer", "Webcam", "Speaker"]
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    regions = ["North", "South", "East", "West"]

    rows = []
    for _ in range(num_rows):
        product = random.choice(products)
        quantity = random.randint(5, 500)
        # Give each product a rough price so revenue looks realistic.
        unit_price = random.randint(500, 60000)
        revenue = quantity * unit_price
        profit = round(revenue * random.uniform(0.05, 0.35), 2)

        rows.append({
            "Product": product,
            "Revenue": revenue,
            "Profit": profit,
            "Month": random.choice(months),
            "Region": random.choice(regions),
            "Quantity": quantity,
        })

    output_path = os.path.join(OUTPUT_FOLDER, "sales.csv")
    with open(output_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Created {output_path} with {len(rows)} rows")


if __name__ == "__main__":
    generate_employees_csv(200)
    generate_sales_csv(200)
    print("Sample data generation complete!")
