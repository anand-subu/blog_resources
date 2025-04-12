import os
import time
import argparse
import requests
from Bio import Entrez

def search_pmc(query, max_results=20):
    """
    Searches PMC for the query (including license filters) and returns a list of PMC IDs.

    Args:
        query (str): The query string to be used for searching PMC.
        max_results (int, optional): Maximum number of results to fetch. Defaults to 20.

    Returns:
        list: A list of PMC IDs (strings).
    """
    handle = Entrez.esearch(db="pmc", term=query, retmax=max_results)
    record = Entrez.read(handle)
    handle.close()
    return record.get("IdList", [])

def download_pmc_pdf(pmc_id, outdir="pmc_pdfs"):
    """
    Attempts to download the PDF for the specified PMC ID.

    Args:
        pmc_id (str): The PMC ID (e.g., "1234567") to download.
        outdir (str, optional): Output directory to save the downloaded PDF. Defaults to "pmc_pdfs".

    Returns:
        tuple: A tuple (pdf_path, final_url) where
               - pdf_path is the local path to the downloaded PDF, or None if download failed.
               - final_url is the final URL after redirects, or None if download failed.
    """
    os.makedirs(outdir, exist_ok=True)
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36"
        )
    }

    pdf_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmc_id}/pdf"
    print(f"Requesting PDF for PMC{pmc_id} via: {pdf_url}")

    try:
        response = requests.get(pdf_url, headers=headers, allow_redirects=True, timeout=30)
        final_url = response.url
        print(f"PMC{pmc_id} final URL after redirects: {final_url}")

        content_type = response.headers.get("Content-Type", "").lower()
        if "pdf" not in content_type:
            print(f"Warning: The final content for PMC{pmc_id} is not PDF (Content-Type={content_type}).")
            return None, None

        pdf_path = os.path.join(outdir, f"PMC{pmc_id}.pdf")
        with open(pdf_path, "wb") as f:
            f.write(response.content)
        print(f"Downloaded PMC{pmc_id} -> {pdf_path}")

        return pdf_path, final_url

    except Exception as e:
        print(f"Error downloading PDF for PMC{pmc_id}: {e}")
        return None, None

def main():
    """
    The main entry point that sets up argument parsing, searches for articles in PMC,
    and downloads their PDFs.
    """
    parser = argparse.ArgumentParser(
        description="Search PMC for a given query and download corresponding PDFs."
    )
    parser.add_argument(
        "--search_term",
        type=str,
        default="icd coding AND cc by license[filter]",
        help="Search term to use in PMC queries."
    )
    parser.add_argument(
        "--max_results",
        type=int,
        default=30,
        help="Maximum number of search results to retrieve."
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="pmc_pdfs",
        help="Directory to save downloaded PDFs."
    )
    parser.add_argument(
        "--email",
        type=str,
        default="your_email@example.com",
        help="Email address required by NCBI to use Entrez."
    )

    args = parser.parse_args()
    # Set the email required by NCBI
    Entrez.email = args.email

    print(f"Searching for articles with query: '{args.search_term}'")
    pmc_ids = search_pmc(args.search_term, max_results=args.max_results)
    print(f"Found {len(pmc_ids)} PMC IDs: {pmc_ids}")

    # Dictionary to map "file_name" => "downloaded_from_url"
    downloaded_files = {}

    for pmc_id in pmc_ids:
        pdf_path, final_url = download_pmc_pdf(pmc_id, outdir=args.outdir)
        if pdf_path and final_url:
            file_name = os.path.basename(pdf_path)
            downloaded_files[file_name] = final_url
        # Give some time between subsequent requests
        time.sleep(3)

    with open("downloaded_files_map.json", "w") as f:
        f.write(json.dumps(downloaded_files))

if __name__ == "__main__":
    main()
