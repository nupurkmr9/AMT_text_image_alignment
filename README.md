# AMT_text_image_alignment

Run MTurk experiments for image-alignment and text-alignment in model customization methods e.g. Custom Diffusion, DreamBooth, Textual-Inversion. 

## Synopsis

1. Text-alignment: each trial pits two images generated by two different methods for the same concept and prompt. The user selects the image which aligns more with the prompt (shown along with generated images). 
2. Image-alignment: each trial pits two images generated by two different methods for the same concept and prompt. The user selects the image with more similar target concept (three images of the target concept are shown along with generated images). 


## Requirements
Python

## Usage
- Put all images to test in a web accessible folder. This folder should have subfolders for the results of each algorithm you would like to test (names of subfolders are specified in `opt.which_algs_paths`) against the baseline algorithm (specified in `opt.gt_path`). Images should be named "0.jpg", "1.jpg", etc, in consecutive order up to some total number of images N (or they can be named differently, but you will have to specify a lambda function in `opt['filename']`). 
- Practice test: set `opt["Npractice"]` to provide practice examples in the beginning of each HIT. Set all other parameters corresponding to `practice`. It basically selects images/prompts from the training dataset. Set it to `opt["Npractice"]=0` to disable this.
- Set experiment parameters by modifying `opt` in `getOpts` function.
- Text-alignment: `python mk_expt_prompt.py -n EXPT_NAME` to generate data csv and index.html for Turk. This will also save a pkl file for the pairs selected. The same set of pairs are then selected for image-alignment experiment. 
- Image-alignment: `python mk_expt_image.py -n EXPT_NAME` to generate data csv and index.html for Turk.

- Create experiment using AMT website or command line tools. For the former option, paste contents of index.html into HIT html code. Upload HIT data from the generated csv.
- After collecting results, run `python process_csv.py -f CSV_FILENAME --N_imgs NUMBER_IMAGES --N_practice NUMBER_PRACTICE`. This will compute and run bootstrap statistics.

## Features
- If multiple algorithms are specified in `opt['which_algs_paths']`, then each HIT tests all algorithms randomly i.i.d. from this list.
- If `opt['paired']` is true always for this setup.
- See `getDefaultOpts()` for documentation on more features


## Citation

This code is modified from <a href="https://github.com/phillipi/AMT_Real_vs_Fake">. This was done as part of the Custom Diffusion project and Bingliang Zhang is one of the contributors to this. 
