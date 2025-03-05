import fitz  # PyMuPDF
import re

"""
pdffonts "Joseph - Font Test - MC-US-10336-External-Flu- RSV RUO deck (1).pdf"
"""
def analyze_font(pdf_path: str, brand_font: str):
    """
    Analyzes the fonts used in a local PDF file and identifies text
    that does not match the specified brand font. Words using the same
    non-matching font are grouped together to reduce verbosity.

    Parameters:
        pdf_path (str): Path to the local PDF file.
        brand_font (str): The expected font. Words using a different font will be flagged.

    Returns:
        dict: A dictionary mapping each page number to a list of text blocks with incorrect fonts.
    """

    try:
        doc = fitz.open(pdf_path)  # Open the local PDF file
    except Exception as e:
        print(f"Error opening PDF: {e}")
        return {}

    font_to_filter = brand_font
    fonts_by_page = {}

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text_page = page.get_text("dict")  # Extract text in a structured format
        fonts_by_page[page_num] = []

        for block in text_page.get("blocks", []):
            if "lines" in block:
                for line in block["lines"]:
                    current_font = None
                    current_text = []
                    bbox_start = None

                    for span in line["spans"]:
                        if "font" in span:
                            if re.search(font_to_filter, span["font"]):  # Skip matching font
                                continue

                            if current_font != span["font"]:
                                # Save previous grouped text before switching fonts
                                if current_text:
                                    fonts_by_page[page_num].append({
                                        "font": current_font,
                                        "text": " ".join(current_text),
                                        "bbox": bbox_start
                                    })
                                
                                # Start a new group with the current font
                                current_font = span["font"]
                                current_text = [span["text"]]
                                bbox_start = span["bbox"]
                            else:
                                # Continue grouping words with the same font
                                current_text.append(span["text"])
                                bbox_start = [
                                    min(bbox_start[0], span["bbox"][0]),  # x0
                                    min(bbox_start[1], span["bbox"][1]),  # y0
                                    max(bbox_start[2], span["bbox"][2]),  # x1
                                    max(bbox_start[3], span["bbox"][3]),  # y1
                                ]

                    # Append the last grouped entry
                    if current_text:
                        fonts_by_page[page_num].append({
                            "font": current_font,
                            "text": " ".join(current_text),
                            "bbox": bbox_start
                        })

    return fonts_by_page


# Example usage
pdf_path = "Joseph - Font Test - MC-US-10336-External-Flu- RSV RUO deck (1).pdf"  # Change this to the path of your PDF file
brand_font = "Arial"  # Change this to your expected font

font_issues = analyze_font(pdf_path, brand_font)

# Print the results in a more concise format
for page, text_blocks in font_issues.items():
    if text_blocks:
        print(f"\nPage {page + 1} has text with incorrect fonts:")
        for block in text_blocks:
            print(f"  - Text: '{block['text']}' (Font: {block['font']}) at {block['bbox']}")
    else:
        print(f"\nPage {page + 1} is all in the correct font.")


def extract_fonts(pdf_path):
    """ Extracts and displays all fonts used in the PDF, including real names if available. """
    doc = fitz.open(pdf_path)
    fonts = {}

    for page_num in range(len(doc)):
        font_list = doc.get_page_fonts(page_num)
        for font in font_list:
            font_ref = font[0]  # Font reference in PDF
            font_name = font[3]  # Font name (may be CIDFont+F#)
            is_embedded = font[4]  # Boolean flag indicating if the font is embedded

            # Extract font details safely
            font_info = doc.extract_font(font_ref)
            if font_info:  
                real_font_name = font_info[1] if len(font_info) > 1 else "Unknown"
            else:
                real_font_name = "Unknown"

            if font_name not in fonts:
                fonts[font_name] = {
                    "real_name": real_font_name,
                    "embedded": is_embedded
                }

    # Print extracted font information
    for font_name, details in fonts.items():
        status = "Embedded" if details["embedded"] else "Not Embedded"
        print(f"PDF Font Name: {font_name} - Actual Font: {details['real_name']} - {status}")



# Example usage
pdf_path = "Joseph - Font Test - MC-US-10336-External-Flu- RSV RUO deck (1).pdf"  # Change to your PDF file
extract_fonts(pdf_path)

def extract_fonts_clean(pdf_path):
    """ Extracts and displays all fonts used in the PDF, ignoring binary data. """
    doc = fitz.open(pdf_path)
    
    for page_num in range(len(doc)):
        font_list = doc.get_page_fonts(page_num)
        print(f"\n📄 Page {page_num + 1} Fonts:")
        
        for font in font_list:
            font_ref = font[0]  # Font reference in PDF
            font_name = font[3]  # Font name (CIDFont+Fx)
            is_embedded = font[4]  # Boolean flag

            # Extract font details safely
            font_info = doc.extract_font(font_ref)
            
            # Unpack font_info and avoid printing raw font file data
            if font_info:
                font_base_name = font_info[1] if len(font_info) > 1 else "Unknown"  # Font name
                encoding = font_info[2] if len(font_info) > 2 else "Unknown"  # Encoding
            else:
                font_base_name, encoding = "Unknown", "Unknown"

            print(f"  🔹 PDF Font Name: {font_name} (Embedded: {is_embedded})")
            print(f"  🔹 Actual Font Name: {font_base_name}")
            print(f"  🔹 Encoding: {encoding}")
            
extract_fonts_clean("Joseph - Font Test - MC-US-10336-External-Flu- RSV RUO deck (1).pdf")