#!/usr/bin/env python3
"""
Debug the model selection issue
"""

import os
import glob
import json

def debug_model_selection():
    """Debug model selection logic"""
    
    model_storage_path = "models"
    
    # Find all model folders
    model_folders = glob.glob(f"{model_storage_path}/*_*")
    model_folders = [f for f in model_folders if os.path.isdir(f)]
    
    print(f"📂 Found {len(model_folders)} model folders:")
    for folder in model_folders:
        print(f"  - {folder}")
    
    if not model_folders:
        print("❌ No model folders found!")
        return
    
    # Sort by timestamp logic (updated)
    def get_timestamp(folder_path):
        folder_name = os.path.basename(folder_path)
        parts = folder_name.split('_')
        if len(parts) >= 2:
            # Check if last part looks like a timestamp (YYYYMMDD_HHMMSS)
            timestamp_part = parts[-1]
            if len(timestamp_part) == 6 and timestamp_part.isdigit():
                # Get the date part too (second to last)
                if len(parts) >= 3 and len(parts[-2]) == 8 and parts[-2].isdigit():
                    full_ts = f"{parts[-2]}_{timestamp_part}"
                    print(f"📅 {folder_name} -> timestamp: {full_ts}")
                    return full_ts  # YYYYMMDD_HHMMSS format
                else:
                    full_ts = f"20251115_{timestamp_part}"
                    print(f"📅 {folder_name} -> timestamp: {full_ts} (default date)")
                    return full_ts  # Default to today's date
            print(f"⚠️  {folder_name} -> invalid timestamp format")
            return "00000000_000000"  # Invalid format
        print(f"⚠️  {folder_name} -> no timestamp")
        return "00000000_000000"  # No timestamp
    
    print("\n🔍 Extracting timestamps (updated logic):")
    
    # Filter out old models without proper timestamps
    timestamped_models = []
    for folder in model_folders:
        ts = get_timestamp(folder)
        if ts != "00000000_000000":
            timestamped_models.append(folder)
    
    print(f"\n📊 Valid timestamped models: {len(timestamped_models)}")
    for folder in timestamped_models:
        print(f"  - {folder}")
    
    if not timestamped_models:
        print("❌ No valid timestamped models found!")
        return
    
    # Sort by timestamp (most recent first)
    timestamped_models.sort(key=get_timestamp, reverse=True)
    
    print(f"\n📊 Sorted folders (top 5):")
    for folder in timestamped_models[:5]:  # Top 5
        print(f"  - {folder}")
    
    # Check the most recent model
    most_recent = timestamped_models[0]
    print(f"\n🎯 Most recent model: {most_recent}")
    
    # Check metadata
    metadata_path = os.path.join(most_recent, 'metadata.json')
    print(f"📄 Metadata path: {metadata_path}")
    print(f"📄 Metadata exists: {os.path.exists(metadata_path)}")
    
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            print(f"✅ Metadata loaded successfully")
            print(f"🏷️  Model name: {metadata.get('model_name', 'N/A')}")
            print(f"📊 Features: {metadata.get('feature_names', [])}")
            print(f"🎯 Target: {metadata.get('target_column', 'N/A')}")
            print(f"📈 Problem type: {metadata.get('problem_type', 'N/A')}")
        except Exception as e:
            print(f"❌ Error loading metadata: {str(e)}")

if __name__ == "__main__":
    debug_model_selection()