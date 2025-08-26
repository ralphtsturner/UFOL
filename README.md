# UFOL
Basically OpenF is too difficult to write so i made a python thingy instead.

# UFOL - Universal File Operations Library v2.0

[![Python Version](https://img.shields.io/badge/python-3.7%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

UFOL (Universal File Operations Library) is a **Python library for ultra-simple, fluent, and high-level file and directory operations**. Designed to make file operations feel like natural language, UFOL handles everything from single-file operations to batch processing, compression, backups, and lightweight archiving.

---

## Features

### **Single File Operations**
- Create, read, write, append files
- Rename, move, copy, delete, backup files
- Count lines, words, characters
- Compute file hash (`md5`, `sha256`, etc.)
- Text transformations: replace, regex replace, add line numbers, sort, unique lines
- Simple compression & decompression (RLE)

### **Directory Operations**
- Create, delete, move, copy directories
- List files, subdirectories, recursive search
- Count files and subdirectories
- Check if empty, clean up temporary files
- Backup entire directories

### **Batch File Operations (Files collection)**
- Filter by type, extension, size, age, text content
- Batch copy, move, delete, rename, text replace, compress
- Analyze file collections: total size, count, type breakdown
- Find duplicate files
- Iterate and map functions over files

### **Archive Operations**
- Add individual files or entire folders
- Create simple custom archive format
- Extract archive to a specified directory

### **Convenience Utilities**
- `quick_copy`, `quick_move`, `quick_delete`
- Fluent, chainable API for easy one-liners

---

## Installation

UFOL is a pure Python library. Clone the repository or download the `.py` file:

```bash
git clone https://github.com/<your-username>/UFOL.git
cd UFOL
```

You can then import it in your project:

```python
from ufol import File, Directory, Files, Archive, quick_copy, quick_move, quick_delete
```

Requirements:

Python 3.7+

Standard libraries only (os, shutil, pathlib, hashlib, datetime, etc.)

Usage Examples

Single File Operations

```python
from ufol import File

# Create a new file
f = File("example.txt").create("Hello World!")

# Append text
f.append("\nAppended line")

# Read content
print(f.read())

# Backup file
backup = f.backup()
print(f"Backup created: {backup}")

# Compress and decompress
compressed = f.compress()
decompressed = compressed.decompress()
```

Directory Operations

```python
from ufol import Directory

# Create a directory
d = Directory("my_folder").create()

# Copy contents from another folder
d.copy_from("source_folder")

# Cleanup temporary files
d.cleanup(patterns=["*.tmp", "*.pyc"], older_than=7)

Batch File Operations

from ufol import Files

# Get all Python files in a folder
fs = Files().in_dir("my_folder").extension(".py")

# Replace text in all files
fs.replace_in_all("foo", "bar")

# Analyze file collection
analysis = fs.analyze()
print(analysis)
```

Archives

```python
from ufol import Archive

# Create an archive
archive = Archive("data.ufol").add_folder("my_folder").create()

# Extract archive
archive.extract_to("extracted_folder")

Quick Operations

from ufol import quick_copy, quick_move, quick_delete

quick_copy("source.txt", "destination.txt")
quick_move("old_folder", "new_folder")
quick_delete("temp.txt", "old_folder")
```

Demo Script

UFOL includes a self-contained demo:

```
python ufol.py
```

### It will:

- Create a demo folder
- Add sample files
- Perform text operations, backup, compression
- Display analysis
- Cleanup the demo folder automatically

### API Overview

**Class / Function	Description**

```File(path)```	Single file operations

```Directory(path)```	Directory operations

```Files(file_list=None)```	Batch file operations

```Archive(path)```	Lightweight archive handling

```file(path)``` Convenience: create File

```directory(path)``` Convenience: create Directory

```files(*paths)```	Convenience: create Files collection

```archive(path)```	Convenience: create Archive

```quick_copy(source, dest)```	Copy file/directory quickly

```quick_move(source, dest)```	Move file/directory quickly

```quick_delete(*paths)```	Delete file/directory quickly

### Notes & Limitations

- quick_type file detection is heuristic; some binary files may be misclassified as text.
- Compression uses Run-Length Encoding (RLE)—simple but not space-efficient for all data types.
- Archive format is custom, lightweight, and not compatible with standard ZIP.
- Contributing
- Contributions, bug reports, and feature requests are welcome!
- Fork the repository
- Create a feature branch
- Submit a pull request

**License**

This project is licensed under the **MIT License**. See **LICENSE**
for details.

Author

## Ralph – Developer
