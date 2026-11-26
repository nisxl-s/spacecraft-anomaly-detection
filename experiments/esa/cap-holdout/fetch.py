"""Retrieve selected verified members of the ESA Mission 2 v2 release."""
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parent
URL = "https://zenodo.org/api/records/15237121/files/ESA-Mission1.zip/content"
SIZE = 3776246073


def fetch(info, end):
    start = info.header_offset
    part = ROOT / f"{Path(info.filename).name}.range"
    headers = part.with_suffix(part.suffix + ".headers")
    subprocess.run(["curl", "-fsSL", "--retry", "2", "--max-time", "120",
                    "-r", f"{start}-{end}", "--max-filesize", str(end-start+1),
                    "-D", str(headers), URL, "-o", str(part)], check=True)
    expected = f"bytes {start}-{end}/{SIZE}"
    actual = re.findall(r"^content-range:\s*(.+?)\s*$", headers.read_text(), flags=re.I | re.M)
    if actual[-1:] != [expected] or part.stat().st_size != end-start+1:
        raise ValueError(f"Incorrect range for {info.filename}: {actual}")
    return info, part


def main(names):
    sparse = ROOT / "directory.zip"
    with ZipFile(sparse) as archive:
        entries = sorted(archive.infolist(), key=lambda x: x.header_offset)
    tasks = [(x, entries[i+1].header_offset-1) for i,x in enumerate(entries[:-1])
             if Path(x.filename).name in names and ("/channels/" in x.filename or x.filename.count("/") == 1)]
    if len(tasks) != len(names):
        raise ValueError("Missing or ambiguous member")
    with ThreadPoolExecutor(max_workers=3) as executor:
        pending = [executor.submit(fetch, *task) for task in tasks]
        for future in pending:
            info, part = future.result()
            out = ROOT / "inputs" / Path(info.filename).name
            out.parent.mkdir(exist_ok=True)
            with sparse.open("r+b") as stream, part.open("rb") as data:
                stream.seek(info.header_offset)
                import shutil
                shutil.copyfileobj(data, stream)
            with out.open("wb") as output:
                subprocess.run(["unzip", "-p", str(sparse), info.filename], stdout=output, check=True)
            part.unlink()
            print(f"CRC verified: {out.name} ({out.stat().st_size:,} bytes)", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
