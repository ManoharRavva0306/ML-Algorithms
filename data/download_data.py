import os
import pandas as pd
import numpy as np

def generate_bigmart_data(filepath="data/Train.csv", num_samples=8523):
    """
    Generates a realistic BigMart Sales Prediction dataset adhering to Kaggle specifications
    if the CSV dataset file does not already exist.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    if os.path.exists(filepath):
        print(f"Dataset already exists at '{filepath}'.")
        return

    print(f"Generating BigMart Sales dataset at '{filepath}'...")
    np.random.seed(42)

    item_types = [
        'Dairy', 'Soft Drinks', 'Meat', 'Fruits and Vegetables', 'Household',
        'Baking Goods', 'Snack Foods', 'Frozen Foods', 'Breakfast',
        'Health and Hygiene', 'Hard Drinks', 'Canned', 'Breads',
        'Starchy Foods', 'Others', 'Seafood'
    ]

    fat_contents = ['Low Fat', 'Regular', 'LF', 'reg', 'low fat']
    outlets = [
        ('OUT049', 1999, 'Medium', 'Tier 1', 'Supermarket Type1'),
        ('OUT018', 2009, 'Medium', 'Tier 3', 'Supermarket Type2'),
        ('OUT010', 1998, np.nan, 'Tier 3', 'Grocery Store'),
        ('OUT013', 1987, 'High', 'Tier 3', 'Supermarket Type1'),
        ('OUT027', 1985, 'Medium', 'Tier 3', 'Supermarket Type3'),
        ('OUT045', 2002, np.nan, 'Tier 2', 'Supermarket Type1'),
        ('OUT017', 2007, np.nan, 'Tier 2', 'Supermarket Type1'),
        ('OUT046', 1997, 'Small', 'Tier 1', 'Supermarket Type1'),
        ('OUT035', 2004, 'Small', 'Tier 2', 'Supermarket Type1'),
        ('OUT019', 1985, 'Small', 'Tier 1', 'Grocery Store')
    ]

    item_ids = [f"{np.random.choice(['FDA', 'DRC', 'FDN', 'NCB', 'FDB'])}{np.random.randint(10, 99)}" for _ in range(num_samples)]
    weights = np.random.uniform(4.5, 21.5, num_samples)
    # Introduce missing values in Item_Weight (~17%)
    missing_weight_mask = np.random.rand(num_samples) < 0.17
    weights[missing_weight_mask] = np.nan

    fat_list = np.random.choice(fat_contents, num_samples, p=[0.5, 0.3, 0.08, 0.07, 0.05])
    visibility = np.random.exponential(scale=0.06, size=num_samples)
    visibility = np.clip(visibility, 0, 0.32)
    # Introduce ~6% exact zero visibilities (classic BigMart anomaly)
    zero_vis_mask = np.random.rand(num_samples) < 0.06
    visibility[zero_vis_mask] = 0.0

    selected_types = np.random.choice(item_types, num_samples)
    mrps = np.random.uniform(31.0, 267.0, num_samples)

    outlet_indices = np.random.choice(len(outlets), num_samples)
    selected_outlets = [outlets[i] for i in outlet_indices]

    outlet_id = [o[0] for o in selected_outlets]
    outlet_year = [o[1] for o in selected_outlets]
    outlet_size = [o[2] for o in selected_outlets]
    outlet_loc = [o[3] for o in selected_outlets]
    outlet_type = [o[4] for o in selected_outlets]

    # Calculate realistic sales target (Item_Outlet_Sales)
    base_sales = mrps * 15.0
    type_multiplier = np.array([2.5 if ot == 'Supermarket Type3' else (0.4 if ot == 'Grocery Store' else 1.2) for ot in outlet_type])
    noise = np.random.normal(1.0, 0.2, num_samples)
    sales = np.maximum(33.0, base_sales * type_multiplier * noise)

    df = pd.DataFrame({
        'Item_Identifier': item_ids,
        'Item_Weight': weights,
        'Item_Fat_Content': fat_list,
        'Item_Visibility': visibility,
        'Item_Type': selected_types,
        'Item_MRP': mrps,
        'Outlet_Identifier': outlet_id,
        'Outlet_Establishment_Year': outlet_year,
        'Outlet_Size': outlet_size,
        'Outlet_Location_Type': outlet_loc,
        'Outlet_Type': outlet_type,
        'Item_Outlet_Sales': sales
    })

    df.to_csv(filepath, index=False)
    print(f"Dataset successfully saved with {num_samples} rows and {len(df.columns)} columns.")

if __name__ == "__main__":
    generate_bigmart_data()
