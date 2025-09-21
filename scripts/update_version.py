#!/usr/bin/env python3
"""Update version in pyproject.toml"""
import sys
import re

def update_version(new_version):
    with open('pyproject.toml', 'r') as f:
        content = f.read()
    
    # Update version line
    content = re.sub(
        r'version = "[^"]*"',
        f'version = "{new_version}"',
        content
    )
    
    with open('pyproject.toml', 'w') as f:
        f.write(content)
    
    print(f"Updated version to {new_version}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python update_version.py <version>")
        sys.exit(1)
    
    update_version(sys.argv[1])