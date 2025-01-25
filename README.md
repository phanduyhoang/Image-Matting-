# Image-Matting

This project was developed based on the paper:  
[**Total Relighting: Learning to Relight Images for Realistic Lighting Transfer**](https://augmentedperception.github.io/total_relighting/total_relighting_paper.pdf)  

The alpha matte result is available in:  
`deep-image-matting-network-good-result.ipynb`  

For more technical insights, please refer to the paper and the mentioned notebook.  

## Refined Version

- Trained on **27 images** (due to the lack of high-quality ground truth masks).  
- As the image resolution increased, the results improved.  

### Input Requirements

- **RGB Image**  
- **Trimap (4 channels)**  

## Trimap Extraction Model

A separate model was developed specifically for trimap extraction.  

### Details:
- Requires refinement on larger datasets.  
- Needs training on higher-resolution images.  
- Current training was conducted on **256x256 px** images, resulting in suboptimal performance.  

Implementation can be found in:  
- `trimap.py`  
- (Or in some notebook, exact file unknown).  

---
Feel free to explore the implementation and suggest improvements!
