### Overview
This repository contains two Python scripts for working with PDFs from PubMed Central (PMC). The first script searches for PMC articles via the [Entrez API](https://www.ncbi.nlm.nih.gov/books/NBK25499/) and downloads the available PDFs. The second script converts those PDFs into PNG images using [`pdf2image`](https://pypi.org/project/pdf2image/) and [Poppler](https://poppler.freedesktop.org/).

### Setup & Requirements
   
1. **Install dependencies** (ideally in a virtual environment):
   ```bash
   pip install -r requirements.txt
   ```

2. **Poppler Installation**:  
   - If you’re on Windows or macOS, you may need to install Poppler or specify a path to its binaries. See the [pdf2image documentation](https://pypi.org/project/pdf2image/) for details.

### 1. Download PMC PDFs

#### Script: `download_pmc_pdfs.py`

**Description**  
Searches PMC (PubMed Central) with a given query and downloads up to a specified number of PDFs to a chosen directory.

**Usage**  
```bash
python download_pmc_pdfs.py \
    --search_term "icd coding AND cc by license[filter]" \
    --max_results 30 \
    --outdir pmc_pdfs \
    --email "your_email@example.com"
```

**Arguments**  

| Argument       | Description                                                     | Default                                      |
|----------------|-----------------------------------------------------------------|----------------------------------------------|
| `--search_term`  | PMC search term (e.g., "icd coding AND cc by license[filter]") | "icd coding AND cc by license[filter]"       |
| `--max_results`  | Maximum number of results to retrieve                         | 30                                           |
| `--outdir`       | Directory to save downloaded PDFs                             | "pmc_pdfs"                                   |
| `--email`        | Email address required by NCBI to use Entrez                  | "your_email@example.com"                     |

---

### 2. Convert PDFs to PNGs

#### Script: `pdf_to_png.py`

**Description**  
Converts each PDF in a specified folder into PNG images, creating a separate subfolder for each PDF.

**Usage**  
```bash
python pdf_to_png.py \
    --input_folder pmc_pdfs \
    --output_folder_base output_images \
    --poppler_path "poppler-24.08.0/Library/bin/"
```

**Arguments**  

| Argument             | Description                                                      | Default |
|----------------------|------------------------------------------------------------------|---------|
| `--input_folder`       | Path to the folder containing PDF files                         | (required) |
| `--output_folder_base` | Base folder in which subfolders (one per PDF) will be created   | (required) |
| `--poppler_path`       | Path to the Poppler binaries (if not in system PATH)            | None    |

---

### Example Workflow

1. **Download PDFs**  
   - Adjust the `search_term`, `max_results`, etc. as needed:
     ```bash
     python download_pmc_pdfs.py \
       --search_term "cancer AND cc by license[filter]" \
       --max_results 10 \
       --outdir pmc_pdfs \
       --email "myemail@example.com"
     ```
   - This will create a folder `pmc_pdfs` with the downloaded PDFs.

2. **Convert PDFs to PNGs**  
   - Use the downloaded PDFs as input:
     ```bash
     python pdf_to_png.py \
       --input_folder pmc_pdfs \
       --output_folder_base output_images \
       --poppler_path "poppler-24.08.0/Library/bin/"
     ```
   - For each PDF in `pmc_pdfs`, a subfolder inside `output_images` will be created, and the script will save each page as a PNG.

---

### Notes
- Be mindful of the [NCBI usage guidelines](https://www.ncbi.nlm.nih.gov/books/NBK25497/) — keep request rates low to avoid overloading their servers (the example code includes a small `time.sleep(3)` between downloads).
- Ensure that your `poppler_path` is set correctly on Windows/macOS when running `pdf_to_png.py`.
