"""
Database Initialization Script

This script initializes the SQLite database for the E-commerce chatbot.
It performs the following steps:
1.  Connects to 'ecommerce.db' (creates it if it doesn't exist).
2.  Iterates through all CSV files in the 'data' directory.
3.  Loads each CSV file into a Pandas DataFrame.
4.  Writes the DataFrame to a table in the SQLite database.
    - The table name is derived from the filename (e.g., 'olist_customers_dataset.csv' -> 'customers').
    - If the table already exists, it is replaced.
5.  Verifies the data loading by printing the first few rows of the 'customers' table.
"""

import sqlite3
import pandas as pd
import os
import glob

# Configuration
DB_NAME = "ecommerce.db"
DATA_DIR = "data"

def init_db():
    """Initialize the database from CSV files"""
    
    # Remove existing database if it exists to start fresh
    if os.path.exists(DB_NAME):
        os.remove(DB_NAME)
        print(f"🗑️ Removed existing database: {DB_NAME}")
    
    print(f"🚀 Initializing database: {DB_NAME}")
    conn = sqlite3.connect(DB_NAME)
    
    # Get all CSV files in the data directory
    csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    
    if not csv_files:
        print(f"❌ No CSV files found in {DATA_DIR}/ directory!")
        return

    for file_path in csv_files:
        # Extract table name from filename
        # Format: olist_tablename_dataset.csv -> tablename
        filename = os.path.basename(file_path)
        
        # Determine table name based on file pattern
        if "olist_" in filename and "_dataset.csv" in filename:
            table_name = filename.replace("olist_", "").replace("_dataset.csv", "")
        elif "product_category_name_translation" in filename:
            table_name = "product_category_name_translation"
        else:
            table_name = filename.replace(".csv", "")
            
        try:
            # Read CSV
            df = pd.read_csv(file_path)
            
            # Write to SQLite
            df.to_sql(table_name, conn, if_exists="replace", index=False)
            
            print(f"✅ Loaded: {filename} -> {table_name} table ({len(df)} rows, {len(df.columns)} columns)")
            
        except Exception as e:
            print(f"❌ Error loading {filename}: {e}")
            
    # Verify database content
    print("\n🔍 Verification:")
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        print(f"Tables created: {[t[0] for t in tables]}")
        
    except Exception as e:
        print(f"Error verifying database: {e}")
        
    conn.close()
    print(f"\n✨ Database created successfully: {DB_NAME}")

if __name__ == "__main__":
    init_db()
