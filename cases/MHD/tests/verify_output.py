"""Check actual writer output; no third-party Python dependencies."""
import csv
import json
import math
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

directory = Path(sys.argv[1])
metadata = json.loads((directory / "metadata.json").read_text())
count = metadata["nx"] * metadata["ny"] * metadata["nz"]
pvd = ET.parse(directory / "fields.pvd")
datasets = pvd.findall(".//DataSet")
assert len(datasets) == 2
for entry in datasets:
    root = ET.parse(directory / entry.attrib["file"])
    arrays = {a.attrib["Name"]: a for a in root.findall(".//PointData/DataArray")}
    assert set(arrays) == {"rho", "velocity", "B", "current", "divB", "kinetic_energy", "magnetic_energy"}
    for array in arrays.values():
        values = [float(v) for v in array.text.split()]
        assert len(values) == count * int(array.attrib["NumberOfComponents"])
        assert all(math.isfinite(v) for v in values)
    if float(entry.attrib["timestep"]) == 0:
        for name in ("kinetic_energy", "magnetic_energy"):
            energy = sum(map(float, arrays[name].text.split())) / count
            expected = 0.5 * metadata["U0"] ** 2 - metadata["U0"] ** 2 * metadata["B0"] ** 2 / (32 / 3) if name == "kinetic_energy" else 0.5 * metadata["B0"] ** 2
            assert abs(energy - expected) < 1e-10
rows = list(csv.DictReader((directory / "statistics.csv").open()))
assert len(rows) == 2 and int(rows[0]["step"]) == 0 and int(rows[-1]["step"]) == 100
assert all(math.isfinite(float(value)) for row in rows for value in row.values())
assert all(abs(float(row["E"]) - float(row["K"]) - float(row["M"])) < 1e-15 for row in rows)
print("PASS VTI/PVD/CSV/JSON structure, array sizes, finite values, and energy consistency")
