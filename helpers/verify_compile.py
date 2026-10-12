#!/usr/bin/env python3
"""
Verify all Python files in the public repository compile without errors.
"""

import os
import sys
from pathlib import Path

def verify_file(filepath):
    """Verify a single Python file compiles"""
    try:
        compile(Path(filepath).read_bytes(), str(filepath), "exec")
        return True, None
    except Exception as e:
        return False, str(e)

def verify_directory(directory):
    """Verify all Python files in directory"""
    results = {'passed': [], 'failed': []}
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                filepath = Path(root) / file
                success, error = verify_file(str(filepath))
                
                if success:
                    results['passed'].append(filepath)
                    print(f"[OK] {filepath.relative_to(directory)}")
                else:
                    results['failed'].append((filepath, error))
                    print(f"[FAILED] {filepath.relative_to(directory)}")
                    print(f"  Error: {error}")
    
    return results

if __name__ == '__main__':
    ours_dir = Path(__file__).resolve().parent.parent
    
    print("="*80)
    print("Verifying Python files compile in the public repository")
    print("="*80)
    print(f"Directory: {ours_dir}")
    print()
    
    results = verify_directory(ours_dir)
    
    print()
    print("="*80)
    print("COMPILATION RESULTS")
    print("="*80)
    print(f"Passed: {len(results['passed'])} files")
    print(f"Failed: {len(results['failed'])} files")
    
    if results['failed']:
        print("\nFailed files:")
        for filepath, error in results['failed']:
            print(f"  - {filepath.relative_to(ours_dir)}")
            print(f"    {error}")
        sys.exit(1)
    else:
        print("\n[SUCCESS] All Python files compile successfully!")
        sys.exit(0)