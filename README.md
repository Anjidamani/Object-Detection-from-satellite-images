# Object-Detection-from-satellite-images
Introduction This project detects target object from satellite images and extrats colour of detected target objec.

Steps

(1) Data prepration:(1) Downlod NWPUVHR10dataset.Use Labelling tool for annotation of images.
(2) Training:(1)Download darkflow then do changes into tiny-yolo-voc.cfg and labels.txt according to number of classes of objects.And do training using yolo-weights(available at official website of yolo) 
(3) Testing : Test images using demoyolocustomobject.py
