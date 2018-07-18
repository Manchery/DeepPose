# image size [cols,rows]
SIZE=[256,212]

# num of joints
JOINTS=15

# idx (starts from 0) of two ends of body diameter, such as rightshoulder and lefthip 
DIAM1=3
DIAM2=12

# idx (starts from 0) of two ends of limbs, such as righthand and rightelbow
LIMBS=[(0,1),(1,2),\
       (3,4),(4,5),\
       (6,7),(7,8),\
       (9,10),(10,11),\
       (12,13),(13,14)]

# size of stage2 bounding box : alpha * diameter
alpha=0.8

# path to images dir (end with /)

IMPATH="./data/K2HPD/depth_data/depth_images/"

# path to dataset dir (end with /)

DSPATH="./data/K2HPD/"
