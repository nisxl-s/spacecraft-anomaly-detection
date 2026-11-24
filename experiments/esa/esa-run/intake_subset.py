import hashlib
import os
import json
import struct
import subprocess
import tempfile
import zlib
from pathlib import Path

import pandas as pd

from ingest import CHANNELS, checksum, channel_series, save_channel

root = Path(__file__).resolve().parent
uploads = root.parent / "upload"
partial = Path(os.environ.get("ESA_MISSION1_ARCHIVE", root.parent / "ESA-Mission1.zip.part"))
fit = root / "fit"
fit.mkdir(exist_ok=True)
files = [f"{channel}.zip" for channel in CHANNELS] + ["channels.csv", "anomaly_types.csv", "labels.csv"]
hashes = {name: checksum(uploads / name) for name in files}
matched = set()
with partial.open("rb") as stream:
    while header := stream.read(30):
        signature, version, flags, method, mtime, mdate, crc, size, plain_size, name_size, extra_size = struct.unpack("<4s5H3I2H", header)
        if signature != b"PK\x03\x04" or flags & 8:
            raise ValueError("Unexpected ZIP member header")
        member = stream.read(name_size).decode()
        stream.seek(extra_size, 1)
        name = member.rsplit("/", 1)[-1]
        if name not in hashes:
            stream.seek(size, 1)
            continue
        data = stream.read(size)
        if method == 9:
            # Rebuild a one-member ZIP so Info-ZIP can decode Deflate64.
            member_name = name.encode()
            local = struct.pack("<4s5H3I2H", signature, version, flags, method, mtime, mdate,
                                crc, size, plain_size, len(member_name), 0) + member_name + data
            central = struct.pack("<4s6H3I5H2I", b"PK\x01\x02", version, version, flags, method,
                                  mtime, mdate, crc, size, plain_size, len(member_name),
                                  0, 0, 0, 0, 0, 0) + member_name
            end = struct.pack("<4s4H2IH", b"PK\x05\x06", 0, 0, 1, 1, len(central), len(local), 0)
            with tempfile.NamedTemporaryFile(suffix=".zip") as tmp:
                tmp.write(local + central + end)
                tmp.flush()
                data = subprocess.check_output(["unzip", "-p", tmp.name, name])
        elif method == 8:
            data = zlib.decompress(data, -15)
        elif method != 0:
            raise ValueError(f"Unsupported ZIP method {method}")
        if hashlib.sha256(data).hexdigest() != hashes[name]:
            raise ValueError(f"Upload differs from retained download: {name}")
        matched.add(name)
        print(f"Matched retained release bytes for {name}", flush=True)

if set(files) - {"labels.csv"} - matched:
    raise ValueError("Missing byte comparison for telemetry or metadata")
rows = []
for channel in CHANNELS:
    frame = pd.read_pickle(uploads / f"{channel}.zip", compression="zip")
    if not isinstance(frame, pd.DataFrame) or not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError(f"Unexpected telemetry format for {channel}")
    row = save_channel(fit, channel, channel_series(frame, channel), "fit")
    rows.append(row)
    print(f"{channel}: {row['eligible_targets']}", flush=True)
result = {"data_source": {"kind": "esa_mission1_uploaded_subset",
                          "matched_retained_download": sorted(matched),
                          "publisher_archive_digest_verified": False},
          "source_member_sha256": hashes, "phase": "fit", "channels": rows}
(fit / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
