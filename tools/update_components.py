"""
Component ID Update Helper Script

This script helps automate the process of updating component IDs to use the standardized
constants from component_ids.py instead of hardcoded strings.

Usage:
    python tools/update_components.py --file path/to/file.py

The script will:
1. Find all component IDs referenced in the file
2. Determine the appropriate constant from component_ids.py
3. Add the necessary imports
4. Update the component references
"""
import os
import re
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import component ID constants
try:
    from app.interface.constants.component_ids import get_all_component_ids, get_legacy_id_mapping
except ImportError:
    print("Error: Could not import component ID constants.")
    print("Make sure app/interface/constants/component_ids.py exists and contains get_all_component_ids() function.")
    sys.exit(1)

# Regular expressions for finding component IDs
COMPONENT_DEF_PATTERN = r"id=['\"]([a-zA-Z0-9_-]+)['\"]"  # Matches id='component-id' or id="component-id"
CALLBACK_OUTPUT_PATTERN = r"Output\(['\"]([a-zA-Z0-9_-]+)['\"]"  # Matches Output('component-id'
CALLBACK_INPUT_PATTERN = r"Input\(['\"]([a-zA-Z0-9_-]+)['\"]"  # Matches Input('component-id'
CALLBACK_STATE_PATTERN = r"State\(['\"]([a-zA-Z0-9_-]+)['\"]"  # Matches State('component-id'

def extract_component_ids(file_path: Path) -> Set[str]:
    """Extract all component IDs from a file."""
    component_ids = set()
    
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
        
    # Find all component definitions and callback references
    pattern = fr"{COMPONENT_DEF_PATTERN}|{CALLBACK_OUTPUT_PATTERN}|{CALLBACK_INPUT_PATTERN}|{CALLBACK_STATE_PATTERN}"
    matches = re.findall(pattern, content)
    
    # Flatten results and remove empty strings
    for match in matches:
        for item in match:
            if item:
                component_ids.add(item)
    
    return component_ids

def get_constant_name(component_id: str, legacy_mapping: Dict[str, str], all_ids: List[str]) -> Optional[str]:
    """Get the constant name for a component ID."""
    # First check if it's in the legacy mapping
    if component_id in legacy_mapping:
        mapped_id = legacy_mapping[component_id]
        for name, val in sys.modules["app.interface.constants.component_ids"].__dict__.items():
            if name.isupper() and val == mapped_id:
                return name
    
    # If not, try to find the matching constant
    for name, val in sys.modules["app.interface.constants.component_ids"].__dict__.items():
        if name.isupper() and val == component_id:
            return name
    
    # If it's a dynamic report item, suggest using the function
    if component_id.startswith("report-item-"):
        parts = component_id.split("-")
        if len(parts) > 2:
            report_id = parts[2]
            return f"get_report_item_id('{report_id}')"
    
    return None

def update_file(file_path: Path, dry_run: bool = False) -> None:
    """Update a file to use constants instead of string literals."""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    all_ids = get_all_component_ids()
    legacy_mapping = get_legacy_id_mapping()
    component_ids = extract_component_ids(file_path)
    
    # Track which constants we need to import
    constants_to_import = set()
    needs_report_item_func = False
    
    # For each component ID, replace it with the constant
    for component_id in component_ids:
        constant_name = get_constant_name(component_id, legacy_mapping, all_ids)
        
        if constant_name:
            if "get_report_item_id" in constant_name:
                needs_report_item_func = True
            else:
                constants_to_import.add(constant_name)
            
            # Replace in component definitions
            content = re.sub(
                fr"id=['\"]({component_id})['\"]",
                fr"id={constant_name}",
                content
            )
            
            # Replace in callbacks
            content = re.sub(
                fr"Output\(['\"]({component_id})['\"]",
                fr"Output({constant_name}",
                content
            )
            content = re.sub(
                fr"Input\(['\"]({component_id})['\"]",
                fr"Input({constant_name}",
                content
            )
            content = re.sub(
                fr"State\(['\"]({component_id})['\"]",
                fr"State({constant_name}",
                content
            )
    
    # Add import if needed
    if constants_to_import or needs_report_item_func:
        import_stmt = "from app.interface.constants.component_ids import "
        import_list = list(constants_to_import)
        if needs_report_item_func:
            import_list.append("get_report_item_id")
        import_stmt += ", ".join(sorted(import_list))
        
        if "import" in content:
            # Add after the last import
            lines = content.split("\n")
            last_import_idx = 0
            for i, line in enumerate(lines):
                if line.startswith("import ") or line.startswith("from "):
                    last_import_idx = i
            
            # Insert after the last import, with a blank line if needed
            if last_import_idx + 1 < len(lines) and lines[last_import_idx + 1].strip():
                lines.insert(last_import_idx + 1, "")
                lines.insert(last_import_idx + 2, import_stmt)
            else:
                lines.insert(last_import_idx + 1, import_stmt)
            
            content = "\n".join(lines)
        else:
            # Put at the top
            content = import_stmt + "\n\n" + content
    
    # Print changes
    print(f"File: {file_path}")
    print(f"  Found {len(component_ids)} component IDs")
    print(f"  Adding import for {len(constants_to_import)} constants")
    
    if not dry_run:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        print("  Updated successfully!")
    else:
        print("  Dry run, no changes made.")

def find_python_files() -> List[Path]:
    """Find all Python files in the app directory."""
    app_dir = project_root / "app"
    return list(app_dir.glob("**/*.py"))

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Update component IDs to use constants")
    parser.add_argument("--file", help="Path to file to update")
    parser.add_argument("--all", action="store_true", help="Update all files")
    parser.add_argument("--dry-run", action="store_true", help="Don't modify files, just print changes")
    args = parser.parse_args()
    
    if args.file:
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"Error: File {file_path} does not exist")
            sys.exit(1)
        update_file(file_path, args.dry_run)
    elif args.all:
        files = find_python_files()
        print(f"Found {len(files)} Python files")
        for file_path in files:
            update_file(file_path, args.dry_run)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
