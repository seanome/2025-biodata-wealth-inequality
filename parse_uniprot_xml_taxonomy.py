import gzip
import logging
from pathlib import Path
from typing import Dict, Generator, List, Optional, Set

import polars as pl
from lxml import etree

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Define XML namespaces
UNIPROT_NS = "{https://uniprot.org/uniprot}"


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
    sci_name = organism_elem.find(f'.//{UNIPROT_NS}name[@type="scientific"]')
    if sci_name is not None:
        return sci_name.text
    return None


def get_organism_id(organism_elem) -> Optional[int]:
    """Extract NCBI Taxonomy ID from organism element."""
    for db_ref in organism_elem.findall(f".//{UNIPROT_NS}dbReference"):
        if db_ref.get("type") == "NCBI Taxonomy":
            return int(db_ref.get("id"))
    return None


def has_pdb_structure(elem) -> bool:
    """Check if entry has PDB database references (excluding AlphaFold)."""
    for db_ref in elem.findall(f".//{UNIPROT_NS}dbReference"):
        if db_ref.get("type") == "PDB":
            # Count explicit PDB references, not AlphaFold or other structure DBs
            return True
    return False


def create_entry_dict(elem) -> Optional[Dict]:
    """Create dictionary from XML entry element."""
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


def clear_element(elem):
    """Clear element and its ancestors from memory."""
    elem.clear()
    while elem.getprevious() is not None:
        del elem.getparent()[0]


def process_batch(entries: List[Dict]) -> pl.LazyFrame:
    """Convert batch of entries to LazyFrame and perform aggregation."""
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
                pl.col("has_structure")
                .sum()
                .alias("pdb_structures_count"),  # Renamed to be explicit
            ]
        )
    )


def parse_uniprot_xml(
    xml_file: str, batch_size: int = 10000, n_batches=None
) -> Generator[List[Dict], None, None]:
    """Parse UniProt XML file in batches."""
    batch = []
    entry_count = 0

    batch_count = 0

    with gzip.open(xml_file, "rb") as f:
        context = etree.iterparse(f, events=("end",), tag=f"{UNIPROT_NS}entry")

        
        for event, elem in context:
            entry_count += 1

            if entry_dict := create_entry_dict(elem):
                batch.append(entry_dict)

            if len(batch) >= batch_size:
                batch_count += 1
                if n_batches is not None and batch_count > n_batches:
                    return batch
                yield batch
                batch = []

            if entry_count % 100000 == 0:
                logging.info(f"Processed {entry_count:,} entries")

            clear_element(elem)

        if batch:
            batch_count += 1
            if n_batches is not None and batch_count > n_batches:
                return batch
            yield batch
            # Early exit


def process_uniprot_file(
    input_file: str, output_file: str, batch_size: int = 10000, n_batches=None
):
    """Process UniProt XML file and write to Parquet."""
    logging.info(f"Starting to process {input_file}")
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    # Process all batches and concatenate into a single LazyFrame
    all_data = pl.concat(
        [
            process_batch(batch)
            for batch in parse_uniprot_xml(input_file, batch_size, n_batches)
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
