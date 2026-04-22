import sys
import os
import json

import re  # Import regex module
import logging  # Import logging module for better debug information

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Add the 'planning' directory to the system path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
sys.path.extend([parent_dir, grandparent_dir])  # Add two levels up for 'planning' directory

from item import Item
from bin import Bin

def visualize_upto_partno(json_file, partno):
    """
    Visualizes items up to a specified part number from a JSON file.
    
    Parameters:
    - json_file (str): Path to the JSON file containing item data.
    - partno (int): The part number up to which items will be visualized (inclusive).
    
    Behavior:
    - Attempts to extract bin dimensions from the JSON filename.
    - If extraction fails, computes bin size based on item positions and dimensions.
    - Creates a Bin object and populates it with Item objects.
    - Visualizes the bin and items using PainterPlot.
    """
    
    # ------------------------------
    # 1. Load Items from JSON
    # ------------------------------
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        logging.error(f"JSON file not found: {json_file}")
        return
    except json.JSONDecodeError:
        logging.error(f"Invalid JSON format in file: {json_file}")
        return
    
    # Ensure data is sorted and slice up to the specified partno
    subset = data[:partno+1]  # partno is inclusive
    logging.info(f"Loaded {len(subset)} items up to partno {partno}.")
    
    # ------------------------------
    # 2. Extract Bin Dimensions
    # ------------------------------
    def extract_bin_dimensions(filename):
        """
        Extracts bin dimensions from the filename.
        
        Expected filename format with fixed three-digit dimensions:
        'bin<width><height><depth>_seed<seed>.json'
        Example: 'bin495495495_seed5.json' -> [495, 495, 495]
        
        Returns:
        - (width, height, depth) as integers if extraction is successful.
        - None if extraction fails.
        """
        # Fixed three-digit pattern
        pattern_fixed = r'bin(\d{3})(\d{3})(\d{3})_seed\d+\.json$'
        match_fixed = re.search(pattern_fixed, filename)
        if match_fixed:
            width, height, depth = map(int, match_fixed.groups())
            return width, height, depth
        
        # Delimited pattern (e.g., 'bin495x495x495_seed5.json')
        pattern_delim = r'bin(\d+)x(\d+)x(\d+)_seed\d+\.json$'
        match_delim = re.search(pattern_delim, filename)
        if match_delim:
            width, height, depth = map(int, match_delim.groups())
            return width, height, depth
        
        # If no pattern matches, return None
        return None
    
    filename = os.path.basename(json_file)
    bin_dimensions = extract_bin_dimensions(filename)
    
    if bin_dimensions:
        width, height, depth = bin_dimensions
        logging.info(f"Extracted bin dimensions from filename: width={width}, height={height}, depth={depth}")
    else:
        # If extraction fails, compute bin size based on items
        logging.warning("Failed to extract bin dimensions from filename. Computing based on item positions.")
        if not subset:
            logging.error("No items available to compute bin dimensions.")
            return
        max_x, max_y, max_z = 0, 0, 0
        for it in subset:
            x, y, z = it.get("b_position", [0, 0, 0])
            w, h, d = it.get("width", 0), it.get("height", 0), it.get("depth", 0)
            max_x = max(max_x, x + w)
            max_y = max(max_y, y + h)
            max_z = max(max_z, z + d)
        width, height, depth = max_x, max_y, max_z
        logging.info(f"Computed bin dimensions: width={width}, height={height}, depth={depth}")
    
    # ------------------------------
    # 3. Create Bin Object
    # ------------------------------
    bin_obj = Bin(
        width=width,
        height=height,
        depth=depth,
        name='visual_bin',
        unit='mm',
        max_weight=1000000  # Adjust as needed
    )
    logging.info("Created Bin object.")
    
    # ------------------------------
    # 4. Populate Bin with Items
    # ------------------------------
    # Identify the last partno in the subset for coloring
    if subset:
        last_partno = subset[-1].get("partno")
    else:
        last_partno = None  # Handle empty subset
    
    for it in subset:
        # Determine item color
        color = 'blue'
        if last_partno is not None and str(it.get('partno')) == str(last_partno):
            color = 'red'  # Highlight the last item
        
        # Create Item object
        try:
            item_obj = Item(
                partno=it["partno"],
                name=it["name"],
                objshape=it["objshape"],
                width=it["width"],
                height=it["height"],
                depth=it["depth"],
                rotation_quat=it["rotation_quat"],
                priority=it["priority"],
                updown=it["updown"],
                weight=it["weight"],
                loadbear=it["loadbear"],
                unit=it["unit"],
                b_position=it["b_position"],
            )
            bin_obj.store(item_obj)
        except KeyError as e:
            logging.error(f"Missing key {e} in item: {it}")
            continue  # Skip this item and continue with others
        
    # ------------------------------
    # 5. Visualize Using PainterPlot
    # ------------------------------
    try:
        from utils.painter.painter_plot import PainterPlot  # Ensure this import works
    except ImportError as e:
        logging.error(f"Failed to import PainterPlot: {e}")
        return
    
    painter = PainterPlot(bin_obj)
    painter.plotBoxAndItems(
        title=f"upto_partno_{partno}",
        alpha=0.2,
        write_num=True,
        fontsize=8,
        save=False,
        show=True
    )
    logging.info("Visualization complete.")

if __name__ == "__main__":
    # Example JSON file path
    json_file = "planning/data/Item_data/trainset/bin495495400_seed20250103.json"
    # Visualize items up to partno=6, with the 6th item highlighted in red
    visualize_upto_partno(json_file, partno=8)
