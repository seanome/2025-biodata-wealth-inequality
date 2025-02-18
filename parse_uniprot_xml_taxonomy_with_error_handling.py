import gzip
import logging
from pathlib import Path
from typing import Dict, Generator, List, Optional, Set
import json
from datetime import datetime

import polars as pl
from lxml import etree

# Set up logging with both file and console handlers
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("uniprot_parser.log"), logging.StreamHandler()],
)

# Define XML namespaces
UNIPROT_NS = "{https://uniprot.org/uniprot}"


class ParsingError(Exception):
    """Custom exception for parsing errors."""

    pass


def get_lineage_set(lineage_elem) -> Set[str]:
    """Extract taxonomy lineage from XML element."""
    if lineage_elem is None:
        return set()
    return {taxon.text for taxon in lineage_elem.findall(f"{UNIPROT_NS}taxon")}


def classify_organism(lineage: Set[str]) -> str:
    """Classify organism based on its taxonomic lineage."""
    if "Viridiplantae" in lineage:
        return "Plant"
    elif "Metazoa" in lineage:
        return "Animal"
    elif "Fungi" in lineage:
        return "Fungi"
    elif "Eukaryota" in lineage:
        return "other Eukaryota"
    elif "Bacteria" in lineage:
        return "Bacteria"
    elif "Archaea" in lineage:
        return "Archaea"
    elif "Viruses" in lineage:
        return "Viruses"
    return "other"


def get_scientific_name(organism_elem) -> Optional[str]:
    """Extract scientific name from organism element."""
    try:
        sci_name = organism_elem.find(f'.//{UNIPROT_NS}name[@type="scientific"]')
        if sci_name is not None:
            return sci_name.text
    except Exception as e:
        logging.warning(f"Error extracting scientific name: {e}")
    return None


def get_organism_id(organism_elem) -> Optional[int]:
    """Extract NCBI Taxonomy ID from organism element."""
    try:
        for db_ref in organism_elem.findall(f".//{UNIPROT_NS}dbReference"):
            if db_ref.get("type") == "NCBI Taxonomy":
                return int(db_ref.get("id"))
    except Exception as e:
        logging.warning(f"Error extracting organism ID: {e}")
    return None


def has_pdb_structure(elem) -> bool:
    """Check if entry has PDB database references (excluding AlphaFold)."""
    try:
        for db_ref in elem.findall(f".//{UNIPROT_NS}dbReference"):
            if db_ref.get("type") == "PDB":
                return True
    except Exception as e:
        logging.warning(f"Error checking PDB structure: {e}")
    return False


def create_entry_dict(elem) -> Optional[Dict]:
    """Create dictionary from XML entry element."""
    try:
        organism = elem.find(f".//{UNIPROT_NS}organism")
        if organism is None:
            return None

        sci_name = get_scientific_name(organism)
        if sci_name is None:
            return None

        lineage = get_lineage_set(organism.find(f".//{UNIPROT_NS}lineage"))

        return {
            "organism": sci_name,
            "organism_id": get_organism_id(organism),
            "lineage": "; ".join(sorted(lineage)),
            "type": classify_organism(lineage),
            "reviewed": elem.get("dataset") == "Swiss-Prot",
            "unreviewed": elem.get("dataset") == "TrEMBL",
            "has_structure": has_pdb_structure(elem),
        }
    except Exception as e:
        logging.warning(f"Error creating entry dictionary: {e}")
        return None


def create_recovery_point(file_path: str, entry_count: int):
    """Create a recovery point file."""
    recovery_data = {
        "entry_count": entry_count,
        "timestamp": datetime.now().isoformat(),
    }
    recovery_path = Path(file_path).with_suffix(".recovery")
    with open(recovery_path, "w") as f:
        json.dump(recovery_data, f)


def get_recovery_point(file_path: str) -> Optional[int]:
    """Get the last recovery point if it exists."""
    recovery_path = Path(file_path).with_suffix(".recovery")
    if recovery_path.exists():
        with open(recovery_path) as f:
            data = json.load(f)
            return data["entry_count"]
    return None


def clear_element(elem):
    """Clear element and its ancestors from memory."""
    try:
        elem.clear()
        while elem.getprevious() is not None:
            del elem.getparent()[0]
    except Exception as e:
        logging.warning(f"Error clearing element: {e}")


def process_batch(entries: List[Dict]) -> pl.LazyFrame:
    """Convert batch of entries to LazyFrame and perform aggregation."""
    try:
        return (
            pl.LazyFrame(
                entries,
                schema={
                    "organism": pl.Utf8,
                    "organism_id": pl.Int64,
                    "lineage": pl.Utf8,
                    "type": pl.Utf8,
                    "reviewed": pl.Boolean,
                    "unreviewed": pl.Boolean,
                    "has_structure": pl.Boolean,
                },
            )
            .group_by(["organism", "organism_id", "lineage", "type"])
            .agg(
                [
                    pl.col("reviewed").sum().alias("reviewed_count"),
                    pl.col("unreviewed").sum().alias("unreviewed_count"),
                    pl.col("has_structure").sum().alias("pdb_structures_count"),
                ]
            )
        )
    except Exception as e:
        logging.error(f"Error processing batch: {e}")
        raise


def parse_uniprot_xml(
    xml_file: str,
    batch_size: int = 10000,
    n_batches=None,
    chunk_size: int = 50_000_000,
    force_restart: bool = False,
) -> Generator[List[Dict], None, None]:
    """Parse UniProt XML file in batches with error handling and recovery."""
    batch = []
    entry_count = 0
    batch_count = 0
    error_count = 0
    max_consecutive_errors = (
        1000  # Maximum number of consecutive errors before stopping
    )

    # Handle recovery point
    recovery_path = Path(xml_file).with_suffix(".recovery")
    if force_restart:
        if recovery_path.exists():
            recovery_path.unlink()  # Delete recovery file
            logging.info("Forced restart: Deleted existing recovery point")
        last_processed = None
        entry_count = 0
    else:
        last_processed = get_recovery_point(xml_file)
        if last_processed:
            logging.info(f"Resuming from entry {last_processed:,}")
            entry_count = last_processed
        else:
            entry_count = 0

    try:
        with gzip.open(xml_file, "rb") as f:
            while True:
                try:
                    # Read a chunk of the file
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break

                    # Ensure chunk ends at a complete entry
                    while not chunk.strip().endswith(b"</entry>"):
                        next_line = f.readline()
                        if not next_line:
                            break
                        chunk += next_line

                    # Create parser for this chunk
                    parser = etree.XMLPullParser(
                        events=("end",), tag=f"{UNIPROT_NS}entry", recover=True
                    )
                    parser.feed(b"<root>")  # Add artificial root
                    parser.feed(chunk)
                    parser.feed(b"</root>")  # Close root

                    # Process events
                    for event, elem in parser.read_events():
                        entry_count += 1

                        # Skip already processed entries when resuming
                        if last_processed and entry_count <= last_processed:
                            clear_element(elem)
                            continue

                        try:
                            if entry_dict := create_entry_dict(elem):
                                batch.append(entry_dict)
                                error_count = 0  # Reset error count on success

                            if len(batch) >= batch_size:
                                batch_count += 1
                                if n_batches is not None and batch_count > n_batches:
                                    return
                                yield batch
                                batch = []

                            if entry_count % 100000 == 0:
                                logging.info(f"Processed {entry_count:,} entries")
                                create_recovery_point(xml_file, entry_count)

                        except Exception as e:
                            error_count += 1
                            logging.error(f"Error processing entry {entry_count}: {e}")
                            if error_count >= max_consecutive_errors:
                                raise ParsingError(
                                    f"Too many consecutive errors ({error_count})"
                                )
                        finally:
                            clear_element(elem)

                except etree.XMLSyntaxError as e:
                    logging.error(f"XML syntax error in chunk: {e}")
                    error_count += 1
                    if error_count >= max_consecutive_errors:
                        raise ParsingError(
                            f"Too many consecutive errors ({error_count})"
                        )
                    continue

        if batch:
            yield batch

    except Exception as e:
        logging.error(f"Fatal error during parsing: {e}")
        create_recovery_point(xml_file, entry_count)
        raise


def process_uniprot_file(
    input_file: str,
    output_file: str,
    batch_size: int = 10000,
    n_batches=None,
    force_restart: bool = False,
):
    """Process UniProt XML file and write to Parquet with error handling."""
    logging.info(f"Starting to process {input_file}")
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    try:
        # Process all batches and concatenate into a single LazyFrame
        all_data = pl.concat(
            [
                process_batch(batch)
                for batch in parse_uniprot_xml(
                    input_file, batch_size, n_batches, force_restart=force_restart
                )
            ]
        )

        # Groupby organism and aggregate
        all_data = (
            all_data.group_by("organism")
            .agg(
                pl.col("reviewed_count").sum(),
                pl.col("unreviewed_count").sum(),
                pl.col("pdb_structures_count").sum(),
                pl.col("organism_id").first(),
                pl.col("lineage").first(),
                pl.col("type").first(),
            )
            # Add taxonomic classifications
            .with_columns(
                [
                    pl.col("type")
                    .map_dict(
                        {
                            "Bacteria": "Microbial",
                            "Archaea": "Microbial",
                            "Viruses": "Microbial",
                            "Plant": "Plant",
                            "Animal": "Animal",
                            "Fungi": "Fungi",
                            "other Eukaryota": "Other Eukaryota",
                            "other": "Other",
                        }
                    )
                    .alias("type_merge_microbes"),
                    pl.col("type")
                    .map_dict(
                        {
                            "Bacteria": "Cellular Life",
                            "Archaea": "Cellular Life",
                            "Viruses": "Non-cellular Life",
                            "Plant": "Cellular Life",
                            "Animal": "Cellular Life",
                            "Fungi": "Cellular Life",
                            "other Eukaryota": "Cellular Life",
                            "other": "Cellular Life",
                        }
                    )
                    .alias("superdomain"),
                    pl.col("type")
                    .map_dict(
                        {
                            "Bacteria": "Bacteria",
                            "Archaea": "Archaea",
                            "Viruses": "Viruses",
                            "Plant": "Eukaryota",
                            "Animal": "Eukaryota",
                            "Fungi": "Eukaryota",
                            "other Eukaryota": "Eukaryota",
                            "other": "Eukaryota",
                        }
                    )
                    .alias("domain"),
                    pl.col("type")
                    .map_dict(
                        {
                            "Bacteria": "Monera",
                            "Archaea": "Monera",
                            "Viruses": "Viruses",
                            "Plant": "Plant",
                            "Animal": "Animal",
                            "Fungi": "Fungi",
                            "other Eukaryota": "Other Eukaryota",
                            "other": "Other",
                        }
                    )
                    .alias("kingdom"),
                ]
            )
            .sort(["reviewed_count", "unreviewed_count"], descending=True)
        )

        # Write to parquet file
        all_data.sink_parquet(
            output_file,
            compression="zstd",
            compression_level=22,
            statistics=True,
            row_group_size=100000,
        )

        logging.info(f"Processing complete. Results written to {output_file}")

    except Exception as e:
        logging.error(f"Error during file processing: {e}")
        raise
