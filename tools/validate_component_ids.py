"""
Component ID Validation Script

This script validates component IDs across the application to ensure consistency and 
identify mismatches between component definitions and callback references.

Usage:
    python tools/validate_component_ids.py

Output:
    - List of components defined but not used in callbacks
    - List of callback references to undefined components
    - List of mismatched component IDs (similar but not identical)
"""
import os
import re
import sys
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple

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

def find_python_files() -> List[Path]:
    """Find all Python files in the app directory."""
    app_dir = project_root / "app"
    return list(app_dir.glob("**/*.py"))

def extract_component_definitions(file_paths: List[Path]) -> Set[str]:
    """Extract component definitions from Python files."""
    component_ids = set()
    
    for file_path in file_paths:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            
        # Find all component definitions
        matches = re.findall(COMPONENT_DEF_PATTERN, content)
        component_ids.update(matches)
    
    return component_ids

def extract_callback_references(file_paths: List[Path]) -> Tuple[Set[str], Set[str], Set[str]]:
    """Extract callback references to components from Python files."""
    output_refs = set()
    input_refs = set()
    state_refs = set()
    
    for file_path in file_paths:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            
        # Find all callback references
        output_matches = re.findall(CALLBACK_OUTPUT_PATTERN, content)
        input_matches = re.findall(CALLBACK_INPUT_PATTERN, content)
        state_matches = re.findall(CALLBACK_STATE_PATTERN, content)
        
        output_refs.update(output_matches)
        input_refs.update(input_matches)
        state_refs.update(state_matches)
    
    return output_refs, input_refs, state_refs

def find_similar_ids(id1: str, id_list: List[str]) -> List[str]:
    """Find similar component IDs based on word content."""
    words1 = set(id1.lower().replace("-", " ").split())
    similar_ids = []
    
    for id2 in id_list:
        if id1 == id2:
            continue
            
        words2 = set(id2.lower().replace("-", " ").split())
        # If there's significant word overlap
        intersection = words1.intersection(words2)
        if len(intersection) >= min(len(words1), len(words2)) / 2:
            similar_ids.append(id2)
    
    return similar_ids

def main():
    """Main validation function."""
    print("\n===== Component ID Validation =====\n")
    
    # Get files to analyze
    python_files = find_python_files()
    print(f"Found {len(python_files)} Python files to analyze")
    
    # Extract component definitions and callback references
    defined_components = extract_component_definitions(python_files)
    output_refs, input_refs, state_refs = extract_callback_references(python_files)
    
    # Get all callback references
    all_refs = output_refs.union(input_refs).union(state_refs)
    
    # Get standard component IDs
    standard_ids = get_all_component_ids()
    legacy_mapping = get_legacy_id_mapping()
    
    print(f"\nDefined components: {len(defined_components)}")
    print(f"Callback references: {len(all_refs)}")
    print(f"Standard component IDs: {len(standard_ids)}")
    
    # Find mismatches
    undefined_refs = all_refs - defined_components
    unused_components = defined_components - all_refs
    nonstandard_ids = defined_components - set(standard_ids)
    
    # Print results
    print("\n----- Validation Results -----\n")
    
    print("1. Callback references to undefined components:")
    if undefined_refs:
        for ref in sorted(undefined_refs):
            similar = find_similar_ids(ref, list(defined_components))
            print(f"  - '{ref}' (not defined)")
            if similar:
                print(f"    Similar defined components: {', '.join([f'{s}' for s in similar])}")
    else:
        print("  None found! ")
    
    print("\n2. Defined components not used in callbacks:")
    if unused_components:
        for comp in sorted(unused_components):
            print(f"  - '{comp}'")
    else:
        print("  None found! ")
    
    print("\n3. Non-standard component IDs:")
    if nonstandard_ids:
        for comp in sorted(nonstandard_ids):
            mapped_id = legacy_mapping.get(comp)
            if mapped_id:
                print(f"  - '{comp}' (should be '{mapped_id}')")
            else:
                similar = find_similar_ids(comp, standard_ids)
                if similar:
                    print(f"  - '{comp}' (similar to: {', '.join([f'{s}' for s in similar])})")
                else:
                    print(f"  - '{comp}' (no standard equivalent)")
    else:
        print("  None found! ")
    
    print("\n----- Suggested Fixes -----\n")
    
    if undefined_refs or nonstandard_ids:
        print("Components to standardize:")
        for comp in sorted(undefined_refs.union(nonstandard_ids)):
            mapped_id = legacy_mapping.get(comp)
            if mapped_id:
                print(f"  - Replace '{comp}' with '{mapped_id}'")
            else:
                similar = find_similar_ids(comp, standard_ids)
                if similar:
                    print(f"  - Consider replacing '{comp}' with one of: {', '.join([f'{s}' for s in similar])}")
                else:
                    words = comp.split('-')
                    if len(words) >= 2:
                        media_type = words[0] if words[0] in ['image', 'audio', 'video'] else None
                        if not media_type and words[-1] in ['image', 'audio', 'video']:
                            media_type = words[-1]
                        
                        if media_type:
                            other_words = [w for w in words if w != media_type]
                            suggestion = f"{media_type}-{'-'.join(other_words)}"
                            print(f"  - Consider renaming '{comp}' to '{suggestion}'")
                        else:
                            print(f"  - Add '{comp}' to component_ids.py if it's a legitimate component")
    
    print("\nRun this script regularly to ensure component ID consistency across the application.")
    print("\n===== Validation Complete =====\n")

if __name__ == "__main__":
    main()
