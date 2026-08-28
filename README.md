# PX-034: Conflict-Aware CTI Routing

Replacement experiment scaffold for the superseded June 2026 source-conflict gate. The detector is option-blind by construction and operates on CTIConnect question/evidence packets.

See [the replacement protocol](docs/PX034_REPLACEMENT_PROTOCOL.md) and [data contracts](data/README.md).

Run the local checks with:

```bash
python3 -m unittest discover -s tests -v
```

Import a hash-verified official CTIConnect checkout with:

```bash
python3 -m px034 import-cticonnect --cticonnect-root /path/to/CTIConnect --output-dir data/cticonnect
```

Instrument retrieval with `python3 -m px034 run-retrieval --help`. Dense vanilla/EtR/DtR execution uses the official CTIConnect classes and requires the packages in `requirements-retrieval.txt`; CSKG retrieval runs offline over the index shipped by CTIConnect.
