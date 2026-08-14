# Windows Path Handling Blocker — Technical Analysis

**Date:** 2026-08-14  
**Severity:** HIGH — Blocks all CLI-based measurement campaigns on Windows  
**Status:** Identified, documented, requires infrastructure refactoring

## Problem Summary

All IPMNIST and slowly_changing_regression measurement execution is blocked by Windows compatibility issues in the atomic file write infrastructure (`upgd_ipmnist_v3.py`).

```
Error: FileNotFoundError: [Errno 2] No such file or directory: 'E:\\'
  File: upgd_ipmnist_v3.py:603, in _open_parent_directory
    directory_fd = os.open(root, _DIRECTORY_OPEN_FLAGS)
```

## Root Cause Analysis

### 1. POSIX-Specific File Descriptor Model

The atomic write infrastructure in `upgd_ipmnist_v3.py` uses POSIX directory file descriptors:

```python
def _open_parent_directory(path: Path, *, create: bool) -> tuple[Path, int]:
    """Open a stable parent descriptor without following symlink components."""
    destination = _lexical_absolute(path)
    _require(destination != destination.parent, "filesystem path must name a file")
    root = destination.anchor or os.sep  # e.g., "E:\" on Windows
    directory_fd = os.open(root, _DIRECTORY_OPEN_FLAGS)  # FAILS on Windows
```

**Issue:** Windows doesn't support opening drive letters as file descriptors. The `os.open()` call expects a file path, not a drive root.

### 2. Cascading Dependencies on Directory FDs

Once a directory FD is opened, it's used extensively throughout the write pipeline:

- `os.stat(name, dir_fd=directory_fd)` - Stat files relative to directory
- `os.unlink(name, dir_fd=directory_fd)` - Delete files relative to directory
- `os.fsync(directory_fd)` - Sync directory metadata
- `os.open(candidate, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode=0o400, dir_fd=directory_fd)`

**Impact:** All five functions using `dir_fd` must be refactored for cross-platform compatibility.

### 3. Atomic Publish Guarantees

The infrastructure's safety guarantees depend on directory FD semantics:

1. **Inode stability** - FD holds parent inode stable across concurrent operations
2. **Rename atomicity** - `os.rename(src, dst, src_dir_fd=..., dst_dir_fd=...)` atomic within snapshot
3. **Symlink rejection** - `os.stat(..., follow_symlinks=False, dir_fd=...)` prevents TOCTOU
4. **Concurrent substitution detection** - Re-verify inode/size after publication

These guarantees must be preserved in any refactored solution.

## Failed Mitigation Attempts

### Attempt 1: Fallback to cwd

```python
try:
    directory_fd = os.open(root, _DIRECTORY_OPEN_FLAGS)
except FileNotFoundError:
    # Fallback: save cwd, chdir, operate, restore
    old_cwd = os.getcwd()
    try:
        os.chdir(destination.parent)
        # ... operations on cwd ...
    finally:
        os.chdir(old_cwd)
```

**Result:** FAILED - The `dir_fd` semantic is baked throughout the codebase; a cwd fallback is fragile and reintroduces TOCTOU races.

## Recommended Solutions

### Option 1: Conditional Directory FD (Minimal)

Detect Windows and use path-based operations instead:

```python
import platform
import os
from pathlib import Path

def _open_parent_directory_cross_platform(path: Path, *, create: bool):
    """Return (destination, directory_fd_or_None) based on platform."""
    destination = _lexical_absolute(path)
    if platform.system() == "Windows":
        # Windows: return None for directory_fd, use path-based ops
        parent = destination.parent
        if create:
            parent.mkdir(parents=True, exist_ok=True)
        return destination, None  # Sentinel
    else:
        # Unix: use directory FD as before
        root = destination.anchor or os.sep
        directory_fd = os.open(root, _DIRECTORY_OPEN_FLAGS)
        return destination, directory_fd

def _stat_file_cross_platform(path, dir_fd=None):
    """Wrapper for os.stat() that works with or without dir_fd."""
    if dir_fd is None:  # Windows
        return os.stat(path, follow_symlinks=False)
    else:  # Unix
        return os.stat(path.name, dir_fd=dir_fd, follow_symlinks=False)
```

**Pros:**
- Minimal code change (~50 lines)
- Preserves Unix behavior entirely
- Clear branching for each platform

**Cons:**
- Loses atomic guarantees on Windows (no true directory anchor)
- TOCTOU race possible if concurrent writers exist
- Needs testing across many edge cases

### Option 2: Pure Path-Based Atomicity (Recommended)

Refactor to use pathlib + tempfile without directory FDs:

```python
import tempfile
from pathlib import Path

def atomic_write_new(path: Path, data: bytes) -> Path:
    """Atomically publish immutable bytes, cross-platform safe."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create temporary file in same directory (ensures same filesystem)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, 
        delete=False, 
        prefix=f".{path.name}.",
        suffix=".tmp"
    ) as tmp:
        tmp_path = Path(tmp.name)
        tmp.write(tmp.read_only_mode_bytes(data))
    
    try:
        # Change mode to read-only
        tmp_path.chmod(0o444)
        
        # Atomic rename (cross-platform)
        # On POSIX: atomic even with concurrent writers
        # On Windows: all-or-nothing in NTFS
        tmp_path.replace(path)
        
        # Sync (platform-specific effectiveness)
        try:
            os.fsync(path.open().fileno())
        except (OSError, NotImplementedError):
            pass  # Windows fsync may fail; acceptable
            
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    
    return path
```

**Pros:**
- True cross-platform atomicity (uses OS-native rename + NTFS guarantees)
- Simpler code, fewer edge cases
- No directory FD dependency

**Cons:**
- Atomic guarantees differ slightly between OS (acceptable for immutable ledger)
- Requires comprehensive testing
- ~150 lines of refactored atomicity

### Option 3: Feature Detection + Delegation (Most Robust)

Detect platform capabilities at import time:

```python
import platform
_IS_WINDOWS = platform.system() == "Windows"
_SUPPORTS_DIR_FD = not _IS_WINDOWS and hasattr(os, 'open')

if _SUPPORTS_DIR_FD:
    from ._posix_atomicity import atomic_write_new, atomic_read_stable
else:
    from ._cross_platform_atomicity import atomic_write_new, atomic_read_stable
```

**Pros:**
- Preserves POSIX guarantees where available
- Uses best available method per platform
- Clean separation of concerns

**Cons:**
- Duplicates code across two modules
- More maintenance burden

## Deployment Recommendation

**Immediate (Session):**
1. Document blocker clearly (DONE)
2. Mark measurement tasks as "blocked pending path fix"
3. Keep implemented arms in staged status

**Short-term (Next Sprint):**
1. Implement Option 2 (path-based atomicity) - most robust
2. Add cross-platform test suite (Windows + Linux CI)
3. Validate with round-trip measurement (smoke test)

**Timeline:** ~4-6 hours development + testing

## Impact Summary

| Component | Status | Blocked? |
|-----------|--------|----------|
| ARM implementations | ✓ Complete | No (code ready) |
| Unit tests | ✓ Passing (10/10) | No |
| Registry registration | ✓ Complete | No |
| 60-task screening | ✗ Blocked | Yes (path issue) |
| 200-task confirmation | ✗ Blocked | Yes (path issue) |
| slowly_changing_regression | ✗ Blocked | Yes (path issue) |
| rule_discovery automation | ✗ Blocked | Yes (path issue) |

**Critical Path:** Fix path handling → Unblock all measurement campaigns

## References

- **POSIX file descriptor semantics:** man dirfd(2), man open(2)
- **Windows NTFS atomicity:** Windows Dev Center - Transactional NTFS (deprecated, but NTFS guarantees remain)
- **Python pathlib:** https://docs.python.org/3/library/pathlib.html
- **Python tempfile:** https://docs.python.org/3/library/tempfile.html
