import pandas as pd
from visualization_engine import VisualizationEngine

# Test the visualization engine directly
print("🧪 Testing VisualizationEngine directly...")

# Load the sample dataset
file_path = r"e:\New Codes\MP 2.o\02\walmart_sample.csv"
print(f"📂 Loading: {file_path}")

try:
    df = pd.read_csv(file_path)
    print(f"✅ Dataset loaded: {df.shape}")
    print(f"📊 Columns: {list(df.columns)}")
    print(f"🔍 Data types:\n{df.dtypes}")
    print(f"📈 First few rows:\n{df.head()}")
    
    # Initialize visualization engine
    viz_engine = VisualizationEngine()
    
    # Test column analysis
    print("\n🔬 Testing analyze_columns...")
    column_info = viz_engine.analyze_columns(df)
    print(f"✅ Column analysis type: {type(column_info)}")
    print(f"📊 Column info keys: {column_info.keys() if isinstance(column_info, dict) else 'Not a dict'}")
    
    if 'columns' in column_info:
        print(f"🗂️ Number of columns analyzed: {len(column_info['columns'])}")
        for col_name, col_data in column_info['columns'].items():
            print(f"  - {col_name}: {col_data.get('type', 'unknown')}")
    
    # Test visualization types
    print("\n📈 Testing get_available_visualizations...")
    viz_types = viz_engine.get_available_visualizations(df)
    print(f"✅ Visualization types type: {type(viz_types)}")
    print(f"📊 Viz types keys: {viz_types.keys() if isinstance(viz_types, dict) else 'Not a dict'}")
    
    if 'visualizations' in viz_types:
        print(f"🎨 Visualization categories: {viz_types['visualizations'].keys()}")
    
    print("\n✅ Direct testing complete!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()