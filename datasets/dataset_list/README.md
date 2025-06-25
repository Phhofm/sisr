# Datasets

A collection of dataset links that I think might be useful for sisr training dataset preparation/curation.  

Note from me as a model trainer: Please make sure, if you create your own curated versions or mixup curated versions of these datasets, to only include the training images. Do not include val or testing parts of these datasets.  
Reason is simple: I (or others) create val sets out of these val or test parts. Models trained on these images will naturally perform better and such a usage will diminish the actualy usefulness or fairness of testing models against those val sets.  

Therefore number of images corresponds to # images that can be used for training. For example, iNaturalist has 2'686'843 train images, but would also have 100'000 val and 500'000 test images additionally. If i know the training # images, ill list that, if not, ill list the whole dataset # number, if i know it.        

My own released Datasets:

Other datasets:
  
<img src="https://github.com/NVlabs/ffhq-dataset/blob/master/ffhq-teaser.png?raw=true" width="256">   

[FFHQ (Flickr-Faces-HQ)](https://github.com/NVlabs/ffhq-dataset?tab=readme-ov-file)  
Topic: Faces  
Number of images: 70K 
Resolution: 1024x1024  
Format: PNG  
License: Mixed Licenses, all of these licenses allow free use, redistribution, and adaptation for non-commercial purposes  




<img src="https://production-media.paperswithcode.com/datasets/Places-0000003475-4b6da14b.jpg" width="256">   

[Places](http://places.csail.mit.edu/)  
Topic: Scenes  
Number of images: 2.5 million images from 205 scene categories  



 
<img src="https://raw.githubusercontent.com/tkarras/progressive_growing_of_gans/master/representative_image_512x256.png" width="256">   

[CelebA-HQ](https://github.com/tkarras/progressive_growing_of_gans)  
Topic: Faces   
Number of images: 30K  
Resolution: 1024x1024  




<img src="https://production-media.paperswithcode.com/datasets/Screen_Shot_2021-01-28_at_2.11.08_PM.png" width="256">   

[Flickr30k](https://shannon.cs.illinois.edu/DenotationGraph/)  
Number of images: 31K  
License: Flickr Terms of Use  




<img src="https://www.robots.ox.ac.uk/~vgg/data/dtd/thumbs/grid/grid_0135.jpg" width="256">   

[Describable Textures Dataset (DTD)](https://www.robots.ox.ac.uk/~vgg/data/dtd/)  
Topic: Textures   
Number of images: 5'640   
License: Flickr Terms of Use  




<img src="https://production-media.paperswithcode.com/datasets/LFW-0000000022-7647ef6f_M2DdqYg.jpg" width="256">   

[LFW (Labeled Faces in the Wild)](https://www.kaggle.com/datasets/jessicali9530/lfw-dataset)  
Topic: Faces    
Number of images: 13'233     





<img src="https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/static/img/food-101.jpg" width="256">   

[Food-101](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/)  
Topic: Food    
Number of images: 750 (training)   





<img src="https://www.researchgate.net/profile/Liang-Xiao-8/publication/334695033/figure/fig1/AS:784753015943168@1564111107129/Sample-images-from-PACS-dataset-Each-row-represents-a-domain-and-each-column-represents.ppm" width="256">   

[PACS](https://huggingface.co/datasets/flwrlabs/pacs)  
Topic: Photo, Art Painting, Cartoon, Sketch       
Number of images: 9991 (1670 Photos, 2048 Art Painting, 2344 Cartoon, 3929 Sketch)     




<img src="https://web.archive.org/web/20250609231049im_/https://production-media.paperswithcode.com/datasets/Screen_Shot_2021-01-28_at_9.34.07_PM.png" width="256">   

[DIV2K](https://web.archive.org/web/20250603150533/https://data.vision.ee.ethz.ch/cvl/DIV2K/)   
Topic: Scenes          
Number of images: 800   





<img src="https://ieeexplore.ieee.org/mediastore/IEEE/content/media/8982559/9021948/9021973/502300d512-fig-1-source-large.gif" width="256">   

[DIV8K](https://huggingface.co/datasets/Iceclear/DIV8K_TrainingSet)   
Topic: Scenes    
Number of train images: 1304   







<img src="https://github.com/visipedia/inat_comp/blob/master/2021/assets/inat_2021_banner.jpg?raw=true" width="256">   

[iNaturalist](https://github.com/visipedia/inat_comp/tree/master/2021)   
Topic: Nature Photography (Plants, Animals)    
Number of train images: 2'686'843   
Note: There are different versions of this dataset: iNaturalist2021, iNaturalist2019, iNaturalist2018, and iNaturalist2017. This one links to iNaturalist2021.   





<img src="https://web.archive.org/web/20250120113458if_/https://cs.stanford.edu/people/karpathy/cnnembed/cnn_embed_full_1k.jpg" width="256">   

[ImageNet](https://web.archive.org/web/20250620052248/https://www.image-net.org/index.php)   
Topic: Web images       
Number of images: 14'197'122   
Free to researchers for non-commercial use    
Personal note: Be careful about image quality, filtering might be needed.    




<img src="https://web.archive.org/web/20250120113458if_/https://cs.stanford.edu/people/karpathy/cnnembed/cnn_embed_full_1k.jpg" width="256">   

[ImageNet](https://web.archive.org/web/20250620052248/https://www.image-net.org/index.php)   
Topic: Web images       
Number of images: 14'197'122   
Free to researchers for non-commercial use    
Personal note: Be careful about image quality, filtering might be needed.   