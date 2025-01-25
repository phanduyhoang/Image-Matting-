# Image-Matting-
this project was done thanks to this paper
https://augmentedperception.github.io/total_relighting/total_relighting_paper.pdf
alpha matte result is in deep-image-matting-network-good-result.ipynb

please read it for more technical insights
the refined version trained on 27 images (due to the lack of high quality ground truth mask). As the image resolution became higher the result became better 
For input you will need a RGB image and its trimap (4 channels). I also create a model specifically for trimap extraction. Which needs to be refined on larger data an train on higher resolution images. you can find the implementation in trimap.py (or in some notebook I don't remember). I trained on 256x256 px images which doesn't give very goood result
