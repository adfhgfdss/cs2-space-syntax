"""Minimal VPK reader, enough to pull small text files out of a CS2 install.

Source2Viewer would do this too, but its download is blocked from this
machine and the VPK directory format is public. Only the pieces we need are
implemented: walk the directory tree and extract entries, either inline in
the _dir vpk or from a numbered pak.

VPK v1 header: signature, version, tree_size
VPK v2 header: signature, version, tree_size, then four section sizes
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

SIGNATURE = 0x55AA1234
TERMINATOR = 0xFFFF
# Entries whose data lives in the _dir vpk itself carry this archive index.
INLINE_ARCHIVE = 0x7FFF


def read_cstring(buf: bytes, pos: int) -> tuple[str, int]:
    end = buf.index(b"\x00", pos)
    return buf[pos:end].decode("utf-8", "replace"), end + 1


class Vpk:
    def __init__(self, dir_path: Path) -> None:
        self.dir_path = Path(dir_path)
        self.header = self.dir_path.read_bytes()
        signature, version = struct.unpack_from("<II", self.header, 0)
        if signature != SIGNATURE:
            raise ValueError(f"not a vpk: 0x{signature:08X}")
        if version not in (1, 2):
            raise ValueError(f"unsupported vpk version {version}")
        self.version = version
        (self.tree_size,) = struct.unpack_from("<I", self.header, 8)
        self.header_size = 12 if version == 1 else 28
        self.tree = self.header[self.header_size:self.header_size + self.tree_size]
        self.data_start = self.header_size + self.tree_size
        self._pak_cache: dict[int, bytes] = {}

    def entries(self) -> list[dict]:
        out: list[dict] = []
        pos = 0
        while True:
            ext, pos = read_cstring(self.tree, pos)
            if ext == "":
                break
            while True:
                folder, pos = read_cstring(self.tree, pos)
                if folder == "":
                    break
                while True:
                    name, pos = read_cstring(self.tree, pos)
                    if name == "":
                        break
                    crc, preload_bytes, archive, offset, length, term = (
                        struct.unpack_from("<IHHIIH", self.tree, pos)
                    )
                    pos += 18
                    if term != TERMINATOR:
                        raise ValueError("bad entry terminator")
                    preload = self.tree[pos:pos + preload_bytes]
                    pos += preload_bytes
                    full = f"{folder}/{name}.{ext}" if folder else f"{name}.{ext}"
                    out.append({
                        "path": full, "crc": crc, "preload": preload,
                        "archive": archive, "offset": offset, "length": length,
                    })
        return out

    def _pak(self, index: int) -> bytes:
        if index not in self._pak_cache:
            pak_path = self.dir_path.parent / f"{self.dir_path.stem[:-4]}_{index:03d}.vpk"
            self._pak_cache[index] = pak_path.read_bytes()
        return self._pak_cache[index]

    def read(self, entry: dict) -> bytes:
        """Return the full file contents for one entry."""
        body = b""
        if entry["length"]:
            if entry["archive"] == INLINE_ARCHIVE:
                start = self.data_start + entry["offset"]
                body = self.header[start:start + entry["length"]]
            else:
                blob = self._pak(entry["archive"])
                body = blob[entry["offset"]:entry["offset"] + entry["length"]]
        if entry["preload"]:
            body = entry["preload"] + body
        return body


def main() -> None:
    vpk_path = Path(sys.argv[1])
    needle = sys.argv[2].lower() if len(sys.argv) > 2 else "overviews/"
    out_dir = Path(sys.argv[3]) if len(sys.argv) > 3 else None

    vpk = Vpk(vpk_path)
    print(f"{vpk_path.name}: vpk v{vpk.version}, "
          f"tree {vpk.tree_size:,} bytes, {vpk.dir_path.stat().st_size:,} bytes on disk")

    matches = [e for e in vpk.entries() if needle in e["path"].lower()]
    print(f"entries matching {needle!r}: {len(matches)}")
    for entry in matches[:40]:
        where = ("inline" if entry["archive"] == INLINE_ARCHIVE
                 else f"pak {entry['archive']:03d}")
        print(f"  {entry['path']}  [{where}, {entry['length']} bytes]")

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        for entry in matches:
            target = out_dir / Path(entry["path"]).name
            target.write_bytes(vpk.read(entry))
            print(f"  wrote {target}")


if __name__ == "__main__":
    main()
