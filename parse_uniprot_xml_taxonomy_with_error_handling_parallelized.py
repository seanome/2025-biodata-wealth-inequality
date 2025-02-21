import gzip
import logging
from pathlib import Path
from typing import Dict, Generator, List, Optional, Set
import json
from datetime import datetime
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import math
import os

import click
import polars as pl
from lxml import etree

# Keep existing constants and helper functions...
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


def add_taxonomic_classifications(df: pl.LazyFrame) -> pl.LazyFrame:
    """Add taxonomic classification columns to the DataFrame."""
    return df.with_columns([
        pl.col("type")
        .replace({
            "Bacteria": "Microbial",
            "Archaea": "Microbial",
            "Viruses": "Microbial",
            "Plant": "Plant",
            "Animal": "Animal",
            "Fungi": "Fungi",
            "other Eukaryota": "Other Eukaryota",
            "other": "Other",
        })
        .alias("type_merge_microbes"),
        
        pl.col("type")
        .replace({
            "Bacteria": "Cellular Life",
            "Archaea": "Cellular Life",
            "Viruses": "Non-cellular Life",
            "Plant": "Cellular Life",
            "Animal": "Cellular Life",
            "Fungi": "Cellular Life",
            "other Eukaryota": "Cellular Life",
            "other": "Cellular Life",
        })
        .alias("superdomain"),
        
        pl.col("type")
        .replace({
            "Bacteria": "Bacteria",
            "Archaea": "Archaea",
            "Viruses": "Viruses",
            "Plant": "Eukaryota",
            "Animal": "Eukaryota",
            "Fungi": "Eukaryota",
            "other Eukaryota": "Eukaryota",
            "other": "Eukaryota",
        })
        .alias("domain"),
        
        pl.col("type")
        .replace({
            "Bacteria": "Monera",
            "Archaea": "Monera",
            "Viruses": "Viruses",
            "Plant": "Plant",
            "Animal": "Animal",
            "Fungi": "Fungi",
            "other Eukaryota": "Other Eukaryota",
            "other": "Other",
        })
        .alias("kingdom"),
    ])

def get_file_size(file_path: str) -> int:
    """Get the size of a gzipped file."""
    with gzip.open(file_path, 'rb') as f:
        f.seek(0, 2)  # Seek to end of file
        return f.tell()


def process_chunk_file(file_path: str, start_pos: int, end_pos: int) -> List[Dict]:
    """Process a single chunk of the XML file using streaming parser."""
    process_id = os.getpid()
    print(f"Processing chunk in process {process_id} from {start_pos} to {end_pos}")
    results = []
    
    with gzip.open(file_path, 'rb') as f:
        # Seek to start position
        f.seek(start_pos)
        
        # If not starting at beginning, find next complete entry
        if start_pos > 0:
            while not f.readline().strip().startswith(b'<entry'):
                pass
        
        # Create a streaming parser
        parser = etree.XMLPullParser(events=('end',), tag=f"{UNIPROT_NS}entry")
        current_pos = f.tell()
        
        while current_pos < end_pos:
            # Read in smaller chunks to avoid memory issues
            chunk = f.read(min(8192, end_pos - current_pos))  # 8KB chunks
            if not chunk:
                break
                
            parser.feed(chunk)
            
            # Process any completed entries
            for event, elem in parser.read_events():
                try:
                    if entry_dict := create_entry_dict(elem):
                        results.append(entry_dict)
                finally:
                    # Clean up the element immediately
                    elem.clear()
                    while elem.getprevious() is not None:
                        del elem.getparent()[0]
            
            current_pos = f.tell()
    
    logging.info(f"Process {process_id} completed chunk with {len(results)} results")
    return results

def get_chunk_ranges(file_size: int, num_processes: int) -> List[tuple]:
    """Calculate chunk ranges based on number of processes."""
    # Aim for chunks that are roughly equal in size
    # For a 250GB file with 12 processes, each process would handle about 20GB
    chunk_size = math.ceil(file_size / num_processes)
    ranges = []
    start = 0
    
    while start < file_size:
        end = min(start + chunk_size, file_size)
        ranges.append((start, end))
        start = end
    
    return ranges


def save_intermediate_results(
    results: List[Dict],
    output_file: str,
    chunk_num: int,
    total_chunks: int
):
    """Save intermediate results to parquet file."""
    if not results:
        return
        
    output_path = Path(output_file)
    intermediate_file = output_path.with_stem(f"{output_path.stem}_part_{chunk_num}")
    
    # Process batch using existing functions
    processed_data = (
        process_batch(results)
        .group_by(["organism", "organism_id", "lineage", "type"])
        .agg([
            pl.col("reviewed_count").sum(),
            pl.col("unreviewed_count").sum(),
            pl.col("pdb_structures_count").sum(),
        ])
    )
    
    processed_data = add_taxonomic_classifications(processed_data)
    
    # Save to parquet
    (
        processed_data
        .collect(streaming=True)
        .write_parquet(
            intermediate_file,
            compression="zstd",
            compression_level=22,
            statistics=True,
            row_group_size=100000,
        )
    )
    
    logging.info(f"Saved intermediate results {chunk_num}/{total_chunks}")

def combine_intermediate_results(output_file: str):
    """Combine all intermediate results into final output file."""
    output_path = Path(output_file)
    all_parts = list(output_path.parent.glob(f"{output_path.stem}_part_*"))
    
    if not all_parts:
        logging.warning("No intermediate results found to combine")
        return
        
    # Combine all parts
    combined_data = pl.concat([pl.scan_parquet(part) for part in all_parts])
    
    # Final grouping and sorting
    (
        combined_data
        .group_by("organism")
        .agg([
            pl.col("reviewed_count").sum(),
            pl.col("unreviewed_count").sum(),
            pl.col("pdb_structures_count").sum(),
            pl.col("organism_id").first(),
            pl.col("lineage").first(),
            pl.col("type").first(),
        ])
        .sort(["reviewed_count", "unreviewed_count"], descending=True)
        .collect(streaming=True)
        .write_parquet(
            output_file,
            compression="zstd",
            compression_level=22,
            statistics=True,
            row_group_size=100000,
        )
    )
    
    # Clean up intermediate files
    for part_file in all_parts:
        part_file.unlink()
    
    logging.info(f"Combined all results into {output_file}")

# Example usage
def worker(x):
    """Worker function that must be defined at module level."""
    import time
    import os
    pid = os.getpid()
    time.sleep(1)  # Simulate work
    return f"Process {pid} processed {x}"

def simple_parallel_test():
    """Simple test to verify multiprocessing is working."""
    with ProcessPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(worker, range(8)))
    for result in results:
        print(result)

def process_xml_chunk(chunk_data: bytes) -> List[Dict]:
    """Process a chunk of XML data containing complete entries."""
    results = []
    
    # Wrap the chunk in a root element to make it valid XML
    wrapped_chunk = b'<?xml version="1.0" encoding="UTF-8"?><root>' + chunk_data + b'</root>'
    
    # Create a memory-efficient iterative parser
    context = etree.iterparse(io.BytesIO(wrapped_chunk), events=('end',), tag=f"{UNIPROT_NS}entry")
    
    for event, elem in context:
        try:
            if entry_dict := create_entry_dict(elem):
                results.append(entry_dict)
        finally:
            # Clear element to free memory
            elem.clear()
            while elem.getprevious() is not None:
                del elem.getparent()[0]
    
    return results

def split_xml_file(file_path: str, num_chunks: int) -> List[bytes]:
    """Split XML file into chunks at entry boundaries."""
    chunks = []
    current_chunk = []
    chunk_size = 0
    entry_count = 0
    
    with gzip.open(file_path, 'rb') as f:
        # Skip to first entry
        for line in f:
            if line.strip().startswith(b'<entry'):
                current_chunk.append(line)
                chunk_size += len(line)
                break
        
        # Process rest of file
        for line in f:
            if line.strip().startswith(b'<entry'):
                entry_count += 1
                if entry_count % num_chunks == 0:
                    # Start new chunk
                    chunks.append(b''.join(current_chunk))
                    current_chunk = []
                    chunk_size = 0
            
            current_chunk.append(line)
            chunk_size += len(line)
    
    # Add final chunk if any
    if current_chunk:
        chunks.append(b''.join(current_chunk))
    
    return chunks
import subprocess
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import os
from lxml import etree
import click
import polars as pl

UNIPROT_NS = "{https://uniprot.org/uniprot}"

def process_chunk(chunk_start: int, chunk_size: int, input_file: str) -> List[Dict]:
    """Process a chunk of the uncompressed XML file."""
    pid = os.getpid()
    results = []
    
    context = etree.iterparse(input_file, events=('end',), tag=f"{UNIPROT_NS}entry")
    
    # Forward to our chunk start
    with open(input_file, 'rb') as f:
        f.seek(chunk_start)
        chunk_bytes_read = 0
        
        # Process entries until we hit our chunk size
        for event, elem in context:
            try:
                if entry_dict := create_entry_dict(elem):
                    results.append(entry_dict)
                chunk_bytes_read += len(etree.tostring(elem))
                if chunk_bytes_read >= chunk_size:
                    break
            finally:
                elem.clear()
                while elem.getprevious() is not None:
                    del elem.getparent()[0]
    
    logging.info(f"Process {pid} processed chunk of {len(results)} entries")
    return results

def parallel_process_uniprot(input_gz: str, output_file: str, num_processes: int):
    """Process UniProt XML file in parallel using actual multiprocessing."""
    # First decompress the file
    uncompressed_file = input_gz.replace('.gz', '')
    logging.info(f"Decompressing {input_gz} to {uncompressed_file}")
    
    with open(uncompressed_file, 'wb') as out_file:
        subprocess.run(['gunzip', '-c', input_gz], stdout=out_file)
    
    # Get file size and calculate chunks
    file_size = os.path.getsize(uncompressed_file)
    chunk_size = file_size // num_processes
    
    # Process in parallel
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = []
        for i in range(num_processes):
            start = i * chunk_size
            size = chunk_size if i < num_processes - 1 else (file_size - start)
            future = executor.submit(process_chunk, start, size, uncompressed_file)
            futures.append((i, future))
            
        # Process results as they complete
        for i, future in futures:
            try:
                chunk_results = future.result()
                if chunk_results:
                    save_intermediate_results(chunk_results, output_file, i, num_processes)
            except Exception as e:
                logging.error(f"Error processing chunk {i}: {e}")
    
    # Clean up uncompressed file
    os.unlink(uncompressed_file)
    
    # Combine results
    combine_intermediate_results(output_file)

@click.command()
@click.argument('uniprot_xml_gz')
@click.argument('output_parquet')
@click.option('--num-processes', type=int, help='Number of processes to use')
def main(uniprot_xml_gz: str, output_parquet: str, num_processes: Optional[int] = None):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(processName)s - %(levelname)s - %(message)s",
    )
    
    if num_processes is None:
        num_processes = max(1, os.cpu_count() - 1)
    
    mp.set_start_method('spawn')
    logging.info(f"Starting processing with {num_processes} processes")
    
    parallel_process_uniprot(
        input_gz=uniprot_xml_gz,
        output_file=output_parquet,
        num_processes=num_processes
    )


if __name__ == "__main__":
    main()