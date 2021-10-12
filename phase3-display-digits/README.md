# phase3-display-digits

Recognize the digits in display for Petrobras competition using machine learning (neural networks).

Documentation for the 2020 competition: https://docs.google.com/document/d/1D75AwIt3nXUpGOzEO9Ix86ihv40ZXSWMN8WccYshG-o/edit#heading=h.bixnuw3ag4j6

Old documentation for the 2019 competition: https://docs.google.com/document/d/1b1R3ncgDhGfIEpoF7hteZg4nThMeSrUQM2pqORqP4Ww/edit

# Dependencies

- data_aug - it's part of this git project
- opencv (cv2) - we use it just for imread and imwrite


# Install

In folder source_digits_images, unzip the 30 MB file small_images.zip and create folder small_images with 3900 PNG files. But do not allow the PNGs to be located at source_digits_images\small_images\small_images. The PNGs must be at folder source_digits_images\small_images.

Do not delete small_images.zip to avoid git pull retrieving a new copy of it from the git repository.

Edit and run the main script dataset_task3_digits.py

# Generating the source images

In folder source_digits_images there are two Matlab scripts to generate the 3900 PNG images in source_digits_images\small_images, which are used by dataset_task3_digits.py. 