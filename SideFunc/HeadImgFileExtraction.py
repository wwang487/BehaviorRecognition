import os
import shutil
import re

def find_and_copy_images(file_folder, file_suffix, species, image_folder, image_suffix, image_save_folder):
    """
    Searches through files in file_folder for lines containing specified species and related data.
    Copies corresponding images from image_folder to image_save_folder.
    
    :param file_folder: Path to the folder containing the files to search.
    :param file_suffix: Suffix of the files to search.
    :param species: Species to search for in the files.
    :param image_folder: Path to the folder containing the images.
    :param image_suffix: Suffix of the image files.
    :param image_save_folder: Path to the folder where images will be saved.
    """

    # Create save folder if it doesn't exist
    if not os.path.exists(image_save_folder):
        os.makedirs(image_save_folder)

    # Define a regex pattern for matching lines
    # This pattern assumes that species names do not contain digits or commas
    pattern = re.compile(r'([^\d,]+),(\d+\.\d{3}),\s*(\d+\.\d{3}),\s*(\d+\.\d{3}),\s*(\d+\.\d{3}),\s*(\d+\.\d{3})')

    # Iterate over all files in the given file folder
    for file_name in os.listdir(file_folder):
        if file_name.endswith(file_suffix):
            file_path = os.path.join(file_folder, file_name)

            # Read through each line of the file
            with open(file_path, 'r') as file:
                for line in file:
                    # Find all matches of the pattern
                    matches = pattern.finditer(line)

                    for match in matches:
                        # Check if the species name matches
                        if match.group(1).strip().lower() == species.lower():
                            # Species matched, prepare to copy corresponding image
                            base_file_name = os.path.splitext(file_name)[0]
                            image_file_name = f"{base_file_name}{image_suffix}"
                            image_file_path = os.path.join(image_folder, image_file_name)

                            # Check if the image file exists
                            if os.path.exists(image_file_path):
                                # Copy the image to the save folder
                                save_path = os.path.join(image_save_folder, image_file_name)
                                shutil.copy2(image_file_path, save_path)
                                print(f"Copied image: {image_file_name} to {image_save_folder}")
                                break  # If one species match is found, no need to check further in this line
                            else:
                                print(f"Image file does not exist: {image_file_path}")
                        # If species doesn't match, continue to next match
                    # If no matches found in the line, continue to the next line


if __name__ == '__main__':

   # Example usage
   find_and_copy_images(
	file_folder='/media/wlp/chapter2/Demo_Res_Dane_Ashland_2020/Spe/Box/1/',
	file_suffix='.txt',
	species='deer',
	image_folder='/media/wlp/chapter2/Dane_Ashland_2020/1/',
	image_suffix='.jpg',
	image_save_folder='/media/wlp/chapter2/Demo_Res_Dane_Ashland_2020/Head/Orig_Imgs/30/'
    )
