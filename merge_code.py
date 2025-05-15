import os
import re

# --- Configuration ---
OUTPUT_FILENAME = "main.cpp"
DIRECTORIES_TO_PROCESS = [
    "sim/include", # Headers first, especially from a dedicated include dir
    "sim/common",
    "sim/simx",
]
FILE_EXTENSIONS = (".h", ".cpp")

# --- Globals ---
included_headers_content = set() # To store content of headers already added
processed_header_files = set() # To keep track of header file names already processed

# --- Helper Functions ---
def get_include_guard_name(content):
    """Tries to extract the include guard macro name from header content."""
    match = re.search(r"#ifndef\s+(\w+)\s*\n\s*#define\s+\1", content)
    if match:
        return match.group(1)
    return None

def is_pragma_once_present(content):
    """Checks if #pragma once is present."""
    return "#pragma once" in content

def process_file_content(filepath, content):
    """
    Processes the content of a file.
    For headers, it checks include guards.
    For all files, it comments out local includes that will be merged.
    """
    global included_headers_content
    global processed_header_files

    filename = os.path.basename(filepath)
    is_header = filepath.endswith(".h")
    output_lines = []

    if is_header:
        # Check if this header file (by name) has already been processed
        if filename in processed_header_files:
            print(f"    // Header {filename} already processed, skipping content.")
            return f"// --- Content of {filepath} (already included/processed) ---\n\n"

        guard_name = get_include_guard_name(content)
        has_pragma_once = is_pragma_once_present(content)

        # Use a hash of the content (excluding guards) to check for true duplicates if guards are missing/inconsistent
        # This is a more robust way if include guards are not perfectly managed or if we want to avoid adding
        # the same semantic content multiple times even if filenames differ but guards were copied.
        # For simplicity here, we'll primarily rely on filename and guards/pragma once.

        # Mark this header as processed
        processed_header_files.add(filename)

        output_lines.append(f"// --- Start of content from {filepath} ---")
        if guard_name:
            output_lines.append(f"// Original include guard: {guard_name}")
        elif has_pragma_once:
            output_lines.append("// Original pragma: #pragma once")

        # We don't add the #ifndef/#define/#endif or #pragma once to the merged file
        # as the goal is to include the content directly.
        # The script will manage uniqueness.
        lines = content.splitlines()
        if guard_name:
            # Skip the #ifndef, #define
            lines_to_add = []
            in_guarded_section = False
            define_skipped = False
            for line in lines:
                if not define_skipped and f"#ifndef {guard_name}" in line:
                    continue
                if not define_skipped and f"#define {guard_name}" in line:
                    define_skipped = True
                    in_guarded_section = True
                    continue
                if in_guarded_section and f"#endif // {guard_name}" in line or f"#endif /* {guard_name} */" in line or (line.strip() == "#endif" and guard_name): # also check for simple #endif if guard was present
                    in_guarded_section = False
                    continue
                lines_to_add.append(line)
            content_to_add = "\n".join(lines_to_add)
        elif has_pragma_once:
            content_to_add = "\n".join(line for line in lines if "#pragma once" not in line)
        else:
            content_to_add = content
    else: # .cpp file
        output_lines.append(f"// --- Start of content from {filepath} ---")
        content_to_add = content

    # Process lines to comment out local includes
    final_content_lines = []
    for line in content_to_add.splitlines():
        stripped_line = line.strip()
        # Regex to find #include "local_header.h"
        match = re.match(r'#include\s*"([^"]+)"', stripped_line)
        if match:
            included_file = match.group(1)
            # Check if this included file is one of the headers we are merging
            # This requires knowing all header filenames beforehand, or checking against processed_header_files
            # For simplicity, we can check if its basename is in processed_header_files
            # A more robust check would be to collect all .h basenames from the target dirs first.
            potential_path_check = False
            for d in DIRECTORIES_TO_PROCESS:
                if os.path.exists(os.path.join(d, included_file)):
                    potential_path_check = True
                    break
            if included_file in processed_header_files or potential_path_check:
                final_content_lines.append(f"// MERGED_LOCALLY: {line}")
                print(f"    // Commented out local include: {line.strip()} (from {filepath})")
                continue
        final_content_lines.append(line)

    output_lines.append("\n".join(final_content_lines))
    output_lines.append(f"// --- End of content from {filepath} ---\n\n")
    return "\n".join(output_lines)


def main():
    all_filepaths = []
    header_filepaths = []
    source_filepaths = []

    print("Scanning for files...")
    for directory in DIRECTORIES_TO_PROCESS:
        if not os.path.isdir(directory):
            print(f"Warning: Directory '{directory}' not found. Skipping.")
            continue
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(FILE_EXTENSIONS):
                    filepath = os.path.join(root, file)
                    if filepath.endswith(".h"):
                        header_filepaths.append(filepath)
                    elif filepath.endswith(".cpp"):
                        source_filepaths.append(filepath)

    # Prioritize headers from sim/include
    sorted_header_filepaths = sorted(
        header_filepaths,
        key=lambda p: 0 if p.startswith("sim/include") else 1
    )

    # All files in desired processing order
    all_filepaths.extend(sorted_header_filepaths)
    all_filepaths.extend(sorted(source_filepaths)) # Sort other sources alphabetically for consistency

    print(f"\nFound {len(all_filepaths)} files to merge.")
    for fp in all_filepaths:
        print(f"  - {fp}")

    # Collect all header basenames first for the local include check
    global processed_header_files # Ensure we modify the global
    # We will add to processed_header_files as we go, but having a list of *all*
    # potential local headers is good for the commenting-out logic.
    # However, the current logic adds to processed_header_files during the processing
    # of each header, which should generally work for subsequent files.

    with open(OUTPUT_FILENAME, "w", encoding="utf-8") as outfile:
        outfile.write(f"// Merged C++ file generated by script\n")
        outfile.write(f"// Total files merged: {len(all_filepaths)}\n")
        outfile.write("// WARNING: Review this file carefully for compilation and logic errors.\n")
        outfile.write("// Issues like multiple main() functions, global variable redefinitions,\n")
        outfile.write("// and order-dependent declarations might need manual fixing.\n\n")

        # Keep track of unique include lines (for standard/external libs)
        unique_global_includes = set()
        temp_global_includes_content = []

        # First pass: Collect all unique global includes from all files
        print("\nPhase 1: Collecting global includes...")
        for filepath in all_filepaths:
            try:
                with open(filepath, "r", encoding="utf-8") as infile:
                    for line in infile:
                        stripped_line = line.strip()
                        if stripped_line.startswith("#include <") or \
                           (stripped_line.startswith("#include \"") and \
                            not any(h_base in stripped_line for h_base in [os.path.basename(hfp) for hfp in header_filepaths])): # Crude check for non-local
                            if stripped_line not in unique_global_includes:
                                unique_global_includes.add(stripped_line)
                                temp_global_includes_content.append(stripped_line)
            except Exception as e:
                print(f"Error reading {filepath}: {e}")
                outfile.write(f"\n// ERROR: Could not read file {filepath} due to {e}\n\n")

        # Write unique global includes at the top
        if temp_global_includes_content:
            outfile.write("// --- Unique Global Includes (from all files) ---\n")
            for inc_line in sorted(list(unique_global_includes)): # Sort for consistency
                outfile.write(inc_line + "\n")
            outfile.write("// --- End of Unique Global Includes ---\n\n")

        # Second pass: Process and append file contents
        print("\nPhase 2: Merging file contents...")
        for filepath in all_filepaths:
            print(f"  Processing {filepath}...")
            try:
                with open(filepath, "r", encoding="utf-8") as infile:
                    content = infile.read()
                
                # For headers, check if already effectively included
                is_header = filepath.endswith(".h")
                if is_header:
                    base_fname = os.path.basename(filepath)
                    # If header was already processed by name, skip its raw content append
                    # The `process_file_content` will return a comment if it's a known header.
                    # The more crucial check is inside `process_file_content` using `processed_header_files`.

                processed_content = process_file_content(filepath, content)
                outfile.write(processed_content)

            except Exception as e:
                print(f"Error processing {filepath}: {e}")
                outfile.write(f"\n// ERROR: Could not process file {filepath} due to {e}\n\n")

    print(f"\nSuccessfully merged files into '{OUTPUT_FILENAME}'.")
    print("IMPORTANT: Please review the generated file for any issues.")

if __name__ == "__main__":
    # Create dummy directories and files for testing if they don't exist
    # In a real scenario, these directories (sim/simx, sim/common, sim/include) should exist.
    if not os.path.exists("sim/include"): os.makedirs("sim/include")
    if not os.path.exists("sim/common"): os.makedirs("sim/common")
    if not os.path.exists("sim/simx"): os.makedirs("sim/simx")

    # Example: Create dummy files for a quick test
    # open("sim/include/config.h", "w").write("#ifndef CONFIG_H\n#define CONFIG_H\nconst int VERSION = 1;\n#endif // CONFIG_H\n")
    # open("sim/common/utils.h", "w").write("#pragma once\n#include <string>\nstd::string get_util_name();\n")
    # open("sim/common/utils.cpp", "w").write("#include \"utils.h\"\n#include <iostream>\nstd::string get_util_name() { return \"UtilityModule\"; }\n")
    # open("sim/simx/main_sim.cpp", "w").write("#include \"../include/config.h\"\n#include \"../common/utils.h\"\n#include <iostream>\nint main() { std::cout << \"Version: \" << VERSION << \" Name: \" << get_util_name() << std::endl; return 0; }\n")

    main()