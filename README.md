# mcp-tme-servers

This project lets you ask an AI assistant to analyze your bulk
RNAseq data and characterize the tumor-microenvironment (TME) of your samples 
just by describing what you want in plain English, instead of writing complex code yourself. It does this by wrapping different advanced bioinformatic algorithms and packages that aim to characterize the TME that the AI can call
on your behalf, using the [Model Context Protocol](https://modelcontextprotocol.io/)
(MCP):

| Server | What it's for | Package |
|---|---|---|
| multideconv | Bulk RNAseq cell-type deconvolution | [multideconv](https://github.com/VeraPancaldiLab/multideconv) (package) |
| pipeML | Training models to predict patient outcomes (e.g. treatment response) | [pipeML](https://github.com/VeraPancaldiLab/pipeML) (package) |
| CellTFusion | Inferring Tumor-microenvironment (TME) states across patients | [CellTFusion](https://github.com/VeraPancaldiLab/CellTFusion) (package) |
| RaCInG | Reconstructing how different cell types are signaling to each other | [RaCInG](https://github.com/VeraPancaldiLab/RaCInG_package) (package) |
| Bioinfo-tools | General bioinformatic analysis | [Bioinfo-tools](https://github.com/mhurtado13/Bioinfo-tools) (package, `BioinfoTools`) |

## Requirements

- Python >= 3.12.
- MCP Python SDK 2.x, installed automatically with this package.
- R >= 4.3.3 (CellTFusion), see "R version" below.
- RaCInG >= 0.3.1: its deconvolution uses multideconv's default set of methods (Quantiseq,
  EpiDISH, DeconRNASeq, DWLS and CIBERSORTx), as multideconv and CellTFusion do. An older
  installed RaCInG is not updated automatically: reinstall it.
- For CIBERSORTx (in the default methods of multideconv, CellTFusion and RaCInG): Docker, which
  multideconv uses to run it, and a free CIBERSORTx account (email and API token). Without them,
  remove CBSX from the methods.

**Recommended:** run this from a dedicated conda/mamba environment with its
own R, to avoid version conflicts with other projects. 

## Automatic R package installation

`multideconv`, `pipeML`, `CellTFusion`, `RaCInG`, and `BioinfoTools` are
installed automatically, on first use, if missing:

| Package | Install command |
|---|---|
| multideconv | `pak::pkg_install("VeraPancaldiLab/multideconv")` |
| pipeML | `pak::pkg_install("VeraPancaldiLab/pipeML")` |
| CellTFusion | `pak::pkg_install("VeraPancaldiLab/CellTFusion")` |
| RaCInG | `remotes::install_github("VeraPancaldiLab/RaCInG_package")` |
| BioinfoTools | `pak::pkg_install("mhurtado13/Bioinfo-tools")` |

## How it works

When you ask for something, the AI picks the matching tool, translates your
request into a call to the corresponding algorithm or package, runs it in a separate
process, and hands the result back to you -- so you get the analysis without
writing or running any code yourself. 

## Installation

### 1. Install `uv`

`uv` is the Python package/dependency manager this project uses; it also
fetches a matching Python interpreter (>= 3.12) automatically if your system
doesn't have one.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

(Windows: `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`.)

### 2. Clone the repo

```bash
git clone https://github.com/mhurtado13/mcp-tme-servers.git
cd mcp-tme-servers
```

### 3. Install Python dependencies

```bash
uv sync
```

### 4. (Optional) Run the test suite

```bash
uv run pytest        # regression tests against each package's own example data
```

This shells out to real R and will auto-install any of the five R packages
that are still missing (see "Automatic R package installation" above), so
the first run is slower and needs internet access, R itself must already be
on `PATH`, and `pak`/`remotes` don't need to be installed beforehand -- they
get bootstrapped automatically too if missing.

### 6. Register with Claude Code

Replace `/path/to/mcp-tme-servers` below with wherever you cloned the repo
in step 2 (e.g. the output of `pwd` if you're still in that directory).

```bash
claude mcp add multideconv-server --scope user -- uv --directory /path/to/mcp-tme-servers run mcp-multideconv-server
claude mcp add pipeML-server --scope user -- uv --directory /path/to/mcp-tme-servers run mcp-pipeml-server
claude mcp add CellTFusion-server --scope user -- uv --directory /path/to/mcp-tme-servers run mcp-celltfusion-server
claude mcp add RaCInG-server --scope user -- uv --directory /path/to/mcp-tme-servers run mcp-racing-server
claude mcp add Bioinfo-tools-server --scope user -- uv --directory /path/to/mcp-tme-servers run mcp-bioinfotools-server
```

`--scope user` registers each server for every Claude Code session on this
machine (not just when run from inside this repo's directory). Verify with
`claude mcp list`, or `/mcp` inside a Claude Code session.

**R version.** CellTFusion requires R >= 4.3.3. The servers run whatever `Rscript` is first on
`PATH`, unless the `MCP_TME_RSCRIPT` environment variable points to another one. If the R on
`PATH` is older, register the servers with `-e MCP_TME_RSCRIPT=/path/to/R-4.3.3/bin/Rscript`
(before the `--`), e.g.

```bash
claude mcp add CellTFusion-server --scope user -e MCP_TME_RSCRIPT=/opt/R/4.3.3/bin/Rscript -- uv --directory /path/to/mcp-tme-servers run mcp-celltfusion-server
```

and run the tests with the same variable (`MCP_TME_RSCRIPT=/opt/R/4.3.3/bin/Rscript uv run pytest`).
Install the R packages with that same R (`/opt/R/4.3.3/bin/R CMD INSTALL ...`).

## What and how to ask? Here some examples

- "I have RNA-seq counts from 40 tumor samples in tumor_counts.csv. Can you estimate how much of each immune and stromal cell type is present in each sample?"

- "I have multiple estimates for B cells, can you tell me how these different estimates differ biologically?"

- "I have single cell data, can you use it to create a reference signature and then use it to calculate cell abundances of my samples?"

- "Can you build a model that is able to predict immunotherapy response based on the cell abundances of my patients? I want to see if these features are predictive of treatment"

- "I only have raw RNA-seq counts for my tumor samples. Can you give me a bigger-picture view of the tumor microenvironment — not just what cell types are there, but what regulatory state they seem to be in, and whether it looks more 'immune-active' or 'immune-suppressed' overall?"

-  "I want to study how different cell types in my tumor samples might be signaling to each other. Can you tell me which cell types are 'talking' to which other cell types the most in each of my patients?

- "I would like to know which transcription factors are more active in my patients who have responded to this specific treatment"

- "I have a list of genes that go up or down between responders and non-responders. Can you tell me which transcription factors are likely driving that, and whether it's actually meaningful?"

- "Can you tell me how active different biological pathways are in each of my samples, based on the expression data?"

- "I have RNA-seq counts from two groups of patients — responders and non-responders. Can you find which genes are significantly different between them?"

- "I have a list of genes that came up as significant. Can you tell me what biological processes or pathways they're associated with?"

- "I have patient survival data along with a feature score for each patient. Can you check whether these features are related to survival?"

## Issues

If you encounter any problems or have questions about the package, we
encourage you to open an issue
[here](https://github.com/mhurtado13/mcp-tme-servers/issues). We'll do our
best to assist you!

## Authors

`mcp-tme-servers` was developed by [Marcelo
Hurtado](https://github.com/mhurtado13), part of the [Pancaldi lab](https://github.com/VeraPancaldiLab). Currently, Marcelo is the primary
maintainer of this project.

## Citing `mcp-tme-servers`

If you use `mcp-tme-servers` in a scientific publication, we would
appreciate citation to the:

*XX, et al. (XX). mcp-tme-servers -- XX. XX. <https://doi.org/XX>*

Since `mcp-tme-servers` is a wrapper around several existing R packages and
methods rather than a single self-contained tool, **please also cite each of
the specific packages and methods actually used in your analysis** -- not
just this project.

If you're not sure exactly which methods to cite, just ask the AI assistant!. For example:

> **You:** "I finished my analysis, which methods or packages you used? What should I
> cite in my project?"
>
> **AI:** "For this analysis, you should cite:
>
> - This toolkit: *XX, et al. (XX). mcp-tme-servers -- XX. XX.
>   <https://doi.org/XX>*
> - **multideconv** (the deconvolution pipeline): Hurtado, M., Khajavi, L.,
>   Essabbar, A., & Pancaldi, V. (2025). multideconv -- Integrative pipeline
>   for cell type deconvolution from bulk RNAseq using first and second
>   generation methods. bioRxiv. <https://doi.org/10.1101/2025.04.29.651220>
> - **omnideconv** (covering the deconvolution methods): Benchmarking 
>   second-generation methods for cell-type deconvolution of transcriptomic data.
>   Dietrich, Alexander and Merotto, Lorenzo and Pelz, Konstantin and Eder, Bernhard
>   and Zackl, Constantin and Reinisch, Katharina and Edenhofer, Frank and Marini
>   Federico and Sturm, Gregor and List, Markus and Finotello, Francesca. (2024)
>   https://doi.org/>10.1101/2024.06.10.598226
> - **CollecTRI** (the TF-target network used): Müller-Dott, S. et al.
>   (2023). Expanding the coverage of regulons from high-confidence prior
>   knowledge for accurate estimation of transcription factor activities.
>   bioRxiv. <https://doi.org/10.1101/2023.03.30.534849>
> - **decoupleR** (the scoring framework behind both the TF and pathway
>   activity steps): Badia-i-Mompel, P. et al. (2022). decoupleR: ensemble
>   of computational methods to infer biological activities from omics
>   data. Bioinformatics Advances. <https://doi.org/10.1093/bioadv/vbac016>
> - **PROGENy** (the pathway resource used): Schubert, M. et al. (2018).
>   Perturbation-response genes reveal signaling footprints in cancer gene
>   expression. Nature Communications.
>   <https://doi.org/10.1038/s41467-017-02391-6>"
