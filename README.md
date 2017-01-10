# Inceptionv3 trained on OpenImages dataset

An inceptionv3 model trained on the OpenImages (with weights) , ready to run in Tensorflow.


### Running it

* As a standalone file, run `python tag.py <image>`
* To use as a module, call predict_on_image(image_path, return_dict={}), to receive the results in `return_dict`

### Citations

If you use the OpenImages dataset in your work, please cite it as:

APA-style citation: "Krasin I., Duerig T., Alldrin N., Veit A., Abu-El-Haija S., Belongie S., Cai D., Feng Z., Ferrari V., Gomes V., Gupta A., Narayanan D., Sun C., Chechik G, Murphy K. OpenImages: A public dataset for large-scale multi-label and multi-class image classification, 2016. Available from https://github.com/openimages".

BibTeX
```
@article{openimages,
  title={OpenImages: A public dataset for large-scale multi-label and multi-class image classification.},
  author={Krasin, Ivan and Duerig, Tom and Alldrin, Neil and Veit, Andreas and Abu-El-Haija, Sami
    and Belongie, Serge and Cai, David and Feng, Zheyun and Ferrari, Vittorio and Gomes, Victor
    and Gupta, Abhinav and Narayanan, Dhyanesh and Sun, Chen and Chechik, Gal and Murphy, Kevin},
  journal={Dataset available from https://github.com/openimages},
  year={2016}
}
```
