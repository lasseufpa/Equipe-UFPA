### How to use the classifier ###

We have two functions.


Now he have the following:
1) the function called` upper_digit, down_digit = mnist_classifier(drone_image)`
   which uses the classifier in only one image
2) the function called 
  `upper_digit_by_voting, down_digit_by_voting = mnist_classifier_by_voting(drone_images)` 
   which will the classifier in several images (you can use how much images 
   that you want) and select the most voted. 
   This is to minimize errors (by a bad capture as an example)



About 2)

The `drone_images` in` mnist_classifier_by_voting(drone_images)` is a list of 
the images captured by the drone


### Example of using the classifier with four images: ###

     drone_images = [] # It's the list of images
     #loading the images into the list
     drone_images.append(drone_image_1) #first image
     drone_images.append(drone_image_2) #second image
     drone_images.append(drone_image_3) #third image

     #Use the list as input to the classifier
     upper_digit_by_voting, down_digit_by_voting = mnist_classifier_by_voting(drone_images)


## Necessary Libraries ##
      
      torch
      torchvision
      skbuild
      opencv
      numpy
      