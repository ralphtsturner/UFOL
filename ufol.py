#!/usr/bin/env python3
"""
UFOL - Universal File Operations Library v2.0
==============================================
Ultra-simple API for complex file operations. Makes everything feel like natural language.

Usage examples:
    File('test.txt').create().write('Hello World!')
    Directory('backup').create().copy_from('/source')
    Files().in_dir('.').type('python').bigger_than('10KB').delete()
    Archive('data.zip').add_folder('/data').compress()
"""

import os
import sys
import io
import struct
import hashlib
import threading
import queue
import time
import gc
import shutil
import json
import re
from typing import Any, Dict, List, Tuple, Union, Optional, Callable, Iterator
from pathlib import Path
from collections import defaultdict, namedtuple
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta


# Core exceptions
class UFOLError(Exception): pass

# Data structures
FileInfo = namedtuple('FileInfo', ['path', 'size', 'modified', 'type', 'hash'])
OperationResult = namedtuple('OperationResult', ['success', 'message', 'data'])

# Size parsing helper - FIXED
def parse_size(size_str):
    """Parse size strings like '10KB', '5MB', '1GB'"""
    if isinstance(size_str, (int, float)):
        return size_str
    
    units = {'B': 1, 'KB': 1024, 'MB': 1024**2, 'GB': 1024**3, 'TB': 1024**4}
    size_str = str(size_str).upper().strip()
    
    # Check each unit to see if the string ends with it
    for unit, multiplier in sorted(units.items(), key=lambda x: len(x[0]), reverse=True):
        if size_str.endswith(unit):
            number_part = size_str[:-len(unit)].strip()
            try:
                return float(number_part) * multiplier
            except ValueError:
                break
    
    # If no unit found, try to parse as plain number
    try:
        return float(size_str)
    except ValueError:
        raise UFOLError(f"Invalid size format: {size_str}")

# Fast file type detection
FILE_SIGNATURES = {
    b'\xFF\xD8\xFF': 'jpeg', b'\x89\x50\x4E\x47': 'png', b'GIF8': 'gif',
    b'BM': 'bmp', b'RIFF': 'webp', b'PK\x03\x04': 'zip', b'\x1F\x8B': 'gzip',
    b'%PDF': 'pdf', b'ID3': 'mp3', b'\xFF\xFB': 'mp3', b'ftyp': 'mp4',
    b'MZ': 'exe', b'\x7FELF': 'elf'
}

def quick_type(path):
    """Super fast file type detection"""
    try:
        with open(path, 'rb') as f:
            header = f.read(32)
        
        for sig, ftype in FILE_SIGNATURES.items():
            if header.startswith(sig) or (ftype == 'mp4' and len(header) > 8 and header[4:8] == sig):
                return ftype
        
        ext = Path(path).suffix.lower()
        ext_map = {
            '.py': 'python', '.js': 'javascript', '.html': 'html', '.css': 'css',
            '.json': 'json', '.xml': 'xml', '.txt': 'text', '.md': 'markdown',
            '.jpg': 'jpeg', '.jpeg': 'jpeg', '.png': 'png', '.gif': 'gif'
        }
        if ext in ext_map:
            return ext_map[ext]
        
        # Check if text
        if b'\x00' not in header and sum(1 for b in header if 32 <= b <= 126 or b in (9,10,13)) / len(header) > 0.7:
            return 'text'
        
        return 'binary'
    except:
        return 'unknown'


class File:
    """Fluent API for single file operations"""
    
    def __init__(self, path):
        self.path = Path(path)
        self._content = None
        self._encoding = 'utf-8'
    
    # Creation & Existence
    def create(self, content='', encoding='utf-8'):
        """Create file with optional content"""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, 'w', encoding=encoding) as f:
            f.write(content)
        return self
    
    def create_if_missing(self, content=''):
        """Create only if doesn't exist"""
        if not self.exists():
            self.create(content)
        return self
    
    def exists(self):
        """Check if file exists"""
        return self.path.exists()
    
    def touch(self):
        """Touch file (update timestamp or create empty)"""
        self.path.touch()
        return self
    
    # Reading & Writing
    def read(self, encoding='utf-8'):
        """Read file content"""
        try:
            with open(self.path, 'r', encoding=encoding) as f:
                return f.read()
        except UnicodeDecodeError:
            with open(self.path, 'rb') as f:
                return f.read()
    
    def read_bytes(self):
        """Read file as bytes"""
        with open(self.path, 'rb') as f:
            return f.read()
    
    def read_lines(self, encoding='utf-8'):
        """Read file as list of lines"""
        return self.read(encoding).splitlines()
    
    def write(self, content, encoding='utf-8', append=False):
        """Write content to file"""
        mode = 'a' if append else 'w'
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, mode, encoding=encoding) as f:
            f.write(str(content))
        return self
    
    def append(self, content, encoding='utf-8'):
        """Append content to file"""
        return self.write(content, encoding, append=True)
    
    def write_bytes(self, data):
        """Write bytes to file"""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, 'wb') as f:
            f.write(data)
        return self
    
    def write_lines(self, lines, encoding='utf-8'):
        """Write list of lines to file"""
        return self.write('\n'.join(str(line) for line in lines), encoding)
    
    # File Operations
    def copy_to(self, destination):
        """Copy file to destination"""
        dest_path = Path(destination)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.path, dest_path)
        return File(dest_path)
    
    def move_to(self, destination):
        """Move file to destination"""
        dest_path = Path(destination)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(self.path, dest_path)
        self.path = dest_path
        return self
    
    def rename_to(self, new_name):
        """Rename file"""
        new_path = self.path.parent / new_name
        self.path.rename(new_path)
        self.path = new_path
        return self
    
    def delete(self):
        """Delete file"""
        if self.path.exists():
            self.path.unlink()
        return self
    
    def backup(self, suffix=None):
        """Create backup copy"""
        if suffix is None:
            suffix = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_path = self.path.with_suffix(f'.{suffix}{self.path.suffix}')
        return self.copy_to(backup_path)
    
    # File Info
    def size(self):
        """Get file size in bytes"""
        return self.path.stat().st_size if self.exists() else 0
    
    def size_human(self):
        """Get human-readable file size"""
        size = self.size()
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size < 1024:
                return f"{size:.1f}{unit}"
            size /= 1024
        return f"{size:.1f}PB"
    
    def modified(self):
        """Get modification time"""
        return datetime.fromtimestamp(self.path.stat().st_mtime)
    
    def age_days(self):
        """Get file age in days"""
        return (datetime.now() - self.modified()).days
    
    def type(self):
        """Get file type"""
        return quick_type(self.path)
    
    def extension(self):
        """Get file extension"""
        return self.path.suffix.lower()
    
    def name(self):
        """Get filename without extension"""
        return self.path.stem
    
    def basename(self):
        """Get full filename"""
        return self.path.name
    
    def directory(self):
        """Get parent directory as Directory object"""
        return Directory(self.path.parent)
    
    # Content Analysis
    def line_count(self):
        """Count lines in file"""
        try:
            return len(self.read_lines())
        except:
            return 0
    
    def word_count(self):
        """Count words in file"""
        try:
            return len(self.read().split())
        except:
            return 0
    
    def char_count(self):
        """Count characters in file"""
        try:
            return len(self.read())
        except:
            return 0
    
    def hash(self, algorithm='md5'):
        """Calculate file hash"""
        hasher = hashlib.new(algorithm)
        with open(self.path, 'rb') as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
        return hasher.hexdigest()
    
    # Transformations
    def replace_text(self, old, new, count=-1):
        """Replace text in file"""
        content = self.read()
        content = content.replace(old, new, count)
        return self.write(content)
    
    def replace_regex(self, pattern, replacement):
        """Replace using regex"""
        content = self.read()
        content = re.sub(pattern, replacement, content)
        return self.write(content)
    
    def add_line_numbers(self):
        """Add line numbers to file"""
        lines = self.read_lines()
        numbered = [f"{i+1:4d}: {line}" for i, line in enumerate(lines)]
        return self.write_lines(numbered)
    
    def sort_lines(self, reverse=False):
        """Sort lines in file"""
        lines = self.read_lines()
        lines.sort(reverse=reverse)
        return self.write_lines(lines)
    
    def unique_lines(self):
        """Remove duplicate lines"""
        lines = self.read_lines()
        unique = list(dict.fromkeys(lines))  # Preserves order
        return self.write_lines(unique)
    
    # Compression
    def compress(self):
        """Simple RLE compression"""
        data = self.read_bytes()
        compressed = self._rle_compress(data)
        compressed_file = File(str(self.path) + '.compressed')
        compressed_file.write_bytes(compressed)
        return compressed_file
    
    def decompress(self):
        """Decompress RLE file"""
        data = self.read_bytes()
        decompressed = self._rle_decompress(data)
        decompressed_file = File(str(self.path).replace('.compressed', ''))
        decompressed_file.write_bytes(decompressed)
        return decompressed_file
    
    @staticmethod
    def _rle_compress(data):
        """Run-length encoding"""
        if not data: return b''
        result, current, count = bytearray(), data[0], 1
        for byte in data[1:]:
            if byte == current and count < 255:
                count += 1
            else:
                result.extend([count, current])
                current, count = byte, 1
        result.extend([count, current])
        return bytes(result)
    
    @staticmethod
    def _rle_decompress(data):
        """Run-length decoding"""
        result = bytearray()
        for i in range(0, len(data), 2):
            if i + 1 < len(data):
                count, byte = data[i], data[i+1]
                result.extend([byte] * count)
        return bytes(result)
    
    # Chaining helpers
    def if_exists(self, func):
        """Execute function only if file exists"""
        if self.exists():
            func(self)
        return self
    
    def if_not_exists(self, func):
        """Execute function only if file doesn't exist"""
        if not self.exists():
            func(self)
        return self
    
    def __str__(self):
        return str(self.path)


class Directory:
    """Fluent API for directory operations"""
    
    def __init__(self, path):
        self.path = Path(path)
    
    # Creation & Existence
    def create(self, parents=True, exist_ok=True):
        """Create directory"""
        self.path.mkdir(parents=parents, exist_ok=exist_ok)
        return self
    
    def exists(self):
        """Check if directory exists"""
        return self.path.exists() and self.path.is_dir()
    
    def delete(self, force=False):
        """Delete directory"""
        if self.exists():
            if force:
                shutil.rmtree(self.path)
            else:
                self.path.rmdir()  # Only if empty
        return self
    
    def empty(self):
        """Empty directory (keep directory, delete contents)"""
        if self.exists():
            for item in self.path.iterdir():
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()
        return self
    
    # Content Access
    def files(self):
        """Get Files collection for this directory"""
        return Files().in_dir(self.path)
    
    def directories(self):
        """Get subdirectories as list of Directory objects"""
        if not self.exists():
            return []
        return [Directory(p) for p in self.path.iterdir() if p.is_dir()]
    
    def list_files(self, pattern='*'):
        """List files matching pattern"""
        if not self.exists():
            return []
        return list(self.path.glob(pattern))
    
    def list_all(self, pattern='*'):
        """List all items matching pattern"""
        if not self.exists():
            return []
        return list(self.path.glob(pattern))
    
    # Operations
    def copy_to(self, destination):
        """Copy directory to destination"""
        dest_path = Path(destination)
        shutil.copytree(self.path, dest_path)
        return Directory(dest_path)
    
    def move_to(self, destination):
        """Move directory to destination"""
        dest_path = Path(destination)
        shutil.move(self.path, dest_path)
        self.path = dest_path
        return self
    
    def copy_from(self, source):
        """Copy contents from source directory"""
        source_path = Path(source)
        if source_path.exists():
            self.create()
            for item in source_path.iterdir():
                if item.is_dir():
                    shutil.copytree(item, self.path / item.name)
                else:
                    shutil.copy2(item, self.path / item.name)
        return self
    
    def backup(self, suffix=None):
        """Create backup of directory"""
        if suffix is None:
            suffix = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_path = Path(f"{self.path}_{suffix}")
        return self.copy_to(backup_path)
    
    # Info
    def size(self):
        """Get total size of directory"""
        if not self.exists():
            return 0
        total = 0
        for item in self.path.rglob('*'):
            if item.is_file():
                try:
                    total += item.stat().st_size
                except:
                    pass
        return total
    
    def file_count(self, recursive=True):
        """Count files in directory"""
        if not self.exists():
            return 0
        pattern = '**/*' if recursive else '*'
        return sum(1 for p in self.path.glob(pattern) if p.is_file())
    
    def dir_count(self, recursive=True):
        """Count subdirectories"""
        if not self.exists():
            return 0
        pattern = '**/*' if recursive else '*'
        return sum(1 for p in self.path.glob(pattern) if p.is_dir())
    
    def is_empty(self):
        """Check if directory is empty"""
        return not self.exists() or not any(self.path.iterdir())
    
    # Utilities
    def find_duplicates(self):
        """Find duplicate files in directory"""
        return Files().in_dir(self.path).find_duplicates()
    
    def cleanup(self, patterns=None, older_than=None):
        """Clean up temporary files"""
        if patterns is None:
            patterns = ['*.tmp', '*.temp', '*~', '*.pyc', '__pycache__']
        
        cleaned = []
        for pattern in patterns:
            for file_path in self.path.rglob(pattern):
                try:
                    if older_than and File(file_path).age_days() < older_than:
                        continue
                    if file_path.is_file():
                        file_path.unlink()
                    elif file_path.is_dir():
                        shutil.rmtree(file_path)
                    cleaned.append(str(file_path))
                except:
                    pass
        
        return cleaned
    
    def __str__(self):
        return str(self.path)


class Files:
    """Fluent API for multiple file operations"""
    
    def __init__(self, file_list=None):
        self.files = [Path(f) for f in file_list] if file_list else []
        self._filters = []
    
    # Building file collections
    def add(self, *paths):
        """Add files to collection"""
        for path in paths:
            self.files.append(Path(path))
        return self
    
    def in_dir(self, directory, recursive=True):
        """Get files in directory"""
        dir_path = Path(directory)
        pattern = '**/*' if recursive else '*'
        self.files = [p for p in dir_path.glob(pattern) if p.is_file()]
        return self
    
    def matching(self, pattern):
        """Filter by filename pattern"""
        import fnmatch
        self.files = [f for f in self.files if fnmatch.fnmatch(f.name, pattern)]
        return self
    
    def type(self, file_type):
        """Filter by file type"""
        self.files = [f for f in self.files if quick_type(f) == file_type]
        return self
    
    def extension(self, ext):
        """Filter by extension"""
        if not ext.startswith('.'):
            ext = '.' + ext
        self.files = [f for f in self.files if f.suffix.lower() == ext.lower()]
        return self
    
    def bigger_than(self, size):
        """Filter files bigger than size"""
        size_bytes = parse_size(size)
        self.files = [f for f in self.files if f.stat().st_size > size_bytes]
        return self
    
    def smaller_than(self, size):
        """Filter files smaller than size"""
        size_bytes = parse_size(size)
        self.files = [f for f in self.files if f.stat().st_size < size_bytes]
        return self
    
    def older_than(self, days):
        """Filter files older than days"""
        cutoff = datetime.now() - timedelta(days=days)
        self.files = [f for f in self.files 
                     if datetime.fromtimestamp(f.stat().st_mtime) < cutoff]
        return self
    
    def newer_than(self, days):
        """Filter files newer than days"""
        cutoff = datetime.now() - timedelta(days=days)
        self.files = [f for f in self.files 
                     if datetime.fromtimestamp(f.stat().st_mtime) > cutoff]
        return self
    
    def containing_text(self, text):
        """Filter files containing text"""
        matching_files = []
        for f in self.files:
            try:
                if quick_type(f) in ['text', 'python', 'javascript', 'html', 'css']:
                    with open(f, 'r', encoding='utf-8', errors='ignore') as file:
                        if text in file.read():
                            matching_files.append(f)
            except:
                pass
        self.files = matching_files
        return self
    
    # Batch operations
    def copy_to(self, destination):
        """Copy all files to destination"""
        dest_dir = Directory(destination).create()
        results = []
        for f in self.files:
            try:
                File(f).copy_to(dest_dir.path / f.name)
                results.append(OperationResult(True, f"Copied {f.name}", None))
            except Exception as e:
                results.append(OperationResult(False, f"Failed to copy {f.name}: {e}", None))
        return results
    
    def move_to(self, destination):
        """Move all files to destination"""
        dest_dir = Directory(destination).create()
        results = []
        for f in self.files:
            try:
                File(f).move_to(dest_dir.path / f.name)
                results.append(OperationResult(True, f"Moved {f.name}", None))
            except Exception as e:
                results.append(OperationResult(False, f"Failed to move {f.name}: {e}", None))
        return results
    
    def delete(self):
        """Delete all files"""
        results = []
        for f in self.files:
            try:
                File(f).delete()
                results.append(OperationResult(True, f"Deleted {f.name}", None))
            except Exception as e:
                results.append(OperationResult(False, f"Failed to delete {f.name}: {e}", None))
        return results
    
    def rename_all(self, pattern, replacement):
        """Rename files using pattern replacement"""
        results = []
        for f in self.files:
            try:
                new_name = re.sub(pattern, replacement, f.name)
                if new_name != f.name:
                    File(f).rename_to(new_name)
                    results.append(OperationResult(True, f"Renamed {f.name} to {new_name}", None))
            except Exception as e:
                results.append(OperationResult(False, f"Failed to rename {f.name}: {e}", None))
        return results
    
    def replace_in_all(self, old_text, new_text):
        """Replace text in all files"""
        results = []
        for f in self.files:
            try:
                if quick_type(f) in ['text', 'python', 'javascript', 'html', 'css']:
                    File(f).replace_text(old_text, new_text)
                    results.append(OperationResult(True, f"Replaced text in {f.name}", None))
            except Exception as e:
                results.append(OperationResult(False, f"Failed to replace in {f.name}: {e}", None))
        return results
    
    def compress_all(self):
        """Compress all files"""
        results = []
        for f in self.files:
            try:
                File(f).compress()
                results.append(OperationResult(True, f"Compressed {f.name}", None))
            except Exception as e:
                results.append(OperationResult(False, f"Failed to compress {f.name}: {e}", None))
        return results
    
    # Analysis
    def total_size(self):
        """Get total size of all files"""
        return sum(f.stat().st_size for f in self.files)
    
    def count(self):
        """Get count of files"""
        return len(self.files)
    
    def analyze(self):
        """Get analysis of file collection"""
        if not self.files:
            return {'count': 0, 'total_size': 0, 'types': {}}
        
        types = defaultdict(int)
        total_size = 0
        
        for f in self.files:
            try:
                ftype = quick_type(f)
                types[ftype] += 1
                total_size += f.stat().st_size
            except:
                pass
        
        return {
            'count': len(self.files),
            'total_size': total_size,
            'types': dict(types),
            'average_size': total_size / len(self.files) if self.files else 0
        }
    
    def find_duplicates(self):
        """Find duplicate files"""
        hashes = defaultdict(list)
        
        for f in self.files:
            try:
                file_hash = File(f).hash()
                hashes[file_hash].append(str(f))
            except:
                pass
        
        return {h: paths for h, paths in hashes.items() if len(paths) > 1}
    
    # Iteration
    def each(self, func):
        """Apply function to each file"""
        for f in self.files:
            func(File(f))
        return self
    
    def map(self, func):
        """Map function over files"""
        return [func(File(f)) for f in self.files]
    
    def filter(self, func):
        """Filter files by function"""
        self.files = [f for f in self.files if func(File(f))]
        return self
    
    def paths(self):
        """Get list of file paths"""
        return [str(f) for f in self.files]
    
    def names(self):
        """Get list of file names"""
        return [f.name for f in self.files]
    
    def __len__(self):
        return len(self.files)
    
    def __iter__(self):
        return (File(f) for f in self.files)


class Archive:
    """Simple archive operations"""
    
    def __init__(self, path):
        self.path = Path(path)
        self.files_to_add = []
    
    def add_file(self, file_path):
        """Add file to archive"""
        self.files_to_add.append(Path(file_path))
        return self
    
    def add_folder(self, folder_path):
        """Add entire folder to archive"""
        folder = Path(folder_path)
        if folder.exists():
            for f in folder.rglob('*'):
                if f.is_file():
                    self.files_to_add.append(f)
        return self
    
    def create(self):
        """Create simple archive (just concatenated files with headers)"""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.path, 'wb') as archive:
            for file_path in self.files_to_add:
                try:
                    # Simple format: [name_length][name][content_length][content]
                    name = str(file_path.name).encode('utf-8')
                    content = file_path.read_bytes()
                    
                    archive.write(len(name).to_bytes(4, 'big'))
                    archive.write(name)
                    archive.write(len(content).to_bytes(8, 'big'))
                    archive.write(content)
                except:
                    pass
        
        return self
    
    def extract_to(self, destination):
        """Extract archive to destination"""
        dest_dir = Directory(destination).create()
        
        with open(self.path, 'rb') as archive:
            while True:
                try:
                    # Read name
                    name_len_bytes = archive.read(4)
                    if not name_len_bytes:
                        break
                    
                    name_len = int.from_bytes(name_len_bytes, 'big')
                    name = archive.read(name_len).decode('utf-8')
                    
                    # Read content
                    content_len = int.from_bytes(archive.read(8), 'big')
                    content = archive.read(content_len)
                    
                    # Write file
                    File(dest_dir.path / name).write_bytes(content)
                    
                except:
                    break
        
        return dest_dir


# Ultra-simple convenience functions
def file(path):
    """Create File object"""
    return File(path)

def directory(path):
    """Create Directory object"""
    return Directory(path)

def files(*paths):
    """Create Files collection"""
    return Files(paths if paths else None)

def archive(path):
    """Create Archive object"""
    return Archive(path)

# Quick operations
def quick_copy(source, destination):
    """Quick file/directory copy"""
    if Path(source).is_dir():
        return Directory(source).copy_to(destination)
    else:
        return File(source).copy_to(destination)

def quick_move(source, destination):
    """Quick file/directory move"""
    if Path(source).is_dir():
        return Directory(source).move_to(destination)
    else:
        return File(source).move_to(destination)

def quick_delete(*paths):
    """Quick delete files/directories"""
    results = []
    for path in paths:
        try:
            p = Path(path)
            if p.is_dir():
                Directory(p).delete(force=True)
            else:
                File(p).delete()
            results.append(f"Deleted {path}")
        except Exception as e:
            results.append(f"Failed to delete {path}: {e}")
    return results

def main():
    print("=== UFOL v2.0 Demo ===")
    
    # Create a demo directory
    demo_dir = directory("demo_folder").create()
    print(f"Created demo directory: {demo_dir}")
    
    # Create some demo files
    file1 = file(demo_dir.path / "hello.txt").create("Hello World!")
    file2 = file(demo_dir.path / "numbers.txt").create("\n".join(map(str, range(10))))
    file3 = file(demo_dir.path / "script.py").create("print('This is a demo script')")
    
    print("Created files:")
    for f in [file1, file2, file3]:
        print(f" - {f.basename()} ({f.size_human()})")
    
    # Reading and writing
    content = file1.read()
    print(f"\nContent of {file1.basename()}: {content}")
    
    file1.append("\nThis is appended text.")
    print(f"Appended to {file1.basename()}. New content:")
    print(file1.read())
    
    # Directory analysis
    print(f"\nDirectory size: {demo_dir.size()} bytes")
    print(f"Files in directory: {demo_dir.file_count()}")
    
    # Files collection demo
    fs = files().in_dir(demo_dir.path).type("text")
    print(f"\nText files found: {fs.names()}")
    
    # Replace text in all text files
    fs.replace_in_all("Hello", "Hi")
    print(f"After text replacement in text files:")
    for f in fs:
        print(f" - {f.basename()}: {f.read()}")
    
    # Backup a file
    backup_file = file1.backup()
    print(f"\nBackup created: {backup_file.basename()}")
    
    # Compression & decompression demo
    compressed = file2.compress()
    print(f"Compressed {file2.basename()} -> {compressed.basename()}")
    
    decompressed = compressed.decompress()
    print(f"Decompressed -> {decompressed.basename()}")
    
    # Cleanup demo
    print("\nCleaning up demo folder...")
    quick_delete(demo_dir.path)
    print("Demo folder deleted.")

if __name__ == "__main__":
    main()