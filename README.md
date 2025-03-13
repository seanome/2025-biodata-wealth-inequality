# 2025-biodata-wealth-inequality
Code to create figures showing wealth inequality of biological sequence data in the Protein Data Bank (PDB), UniProtKB/SwissProt, UniProtKB/TREMBL (AlphaFoldDB), and predicted number of proteins on Earth.

## Figures created

The figures created are the "packcircles" below the graph in the following figure:

![](figures/the-known-protein-universe-is-limited-and-biased.png)

See [`figures/`](figures/) for each individual figure.


## Parsing UniProt TREMBL entries

We used the [uniprot](https://github.com/heuermh/dishevelled-bio/tree/master/protein/src/main/java/org/dishevelled/bio/protein/uniprot) module from [heuermh/dishevelled-bio](https://github.com/heuermh/dishevelled-bio) to parse the gigantic 219 GB `uniprot_trembl.xml.gz` file into entries. See [`notebooks/01_parse_uniprot_trembl_xml.ipynb`](notebooks/01_parse_uniprot_trembl_xml.ipynb) for more.
