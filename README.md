# Bioinfo2024
Classification of bio-informatic sentences extracted from scientific articles. Use of classic ML and neural networks. <br>
The project comes with a report detailing methods and results interpreted from our experiences.
This project was realized with Ludovic Guyader, who was mainly tasked with the creation of the word2vec model, whereas i was tasked with the classification part.

test_hyper_param.ipynb -> test our w2v model hyperparameters
pre_processing_baseline_model.ipynb -> pre-processing of our data
fun_encoding.py, fun_features_engineering.py, fun_metrics.py -> def functions and libraries
Classification.ipynb -> classifiers for 2 W2V models
All .ipynb can be executed independently

Folder : 
proj_100d_external -> contains .bin and 2 extra files for the pre-trained word embedding
models -> should fill up with test_hyper_param.ipynb

Code is not optimized for NVidia (no Cuda). Everything works with the CPU in a decent amount of time.

If there is any trouble at all with any files, contact us via e-mail (paul.oriat@universite-paris-saclay.fr, paul.oriat@gmail.com)
