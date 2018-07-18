set -e

ROOT=.
CUR=$ROOT/models/deeppose

#--------------------------------------------

rm -rf $CUR/train_data_lmdb

rm -f $CUR/mean.npy

#find $DATA/train -type f -exec echo {} \; > $CUR/temp.txt

#sed "s/$/ 0/" $CUR/temp.txt > $CUR/train_images.txt

convert_imageset_multilabel -resize_height=227 -resize_width=227 "" $CUR/train_images.txt $CUR/train_data_lmdb

compute_image_mean $CUR/train_data_lmdb $CUR/train_mean.binaryproto

#--------------------------------------------

rm -rf $CUR/test_data_lmdb

#find $DATA/test -type f -exec echo {} \; > $CUR/temp.txt

#sed "s/$/ 0/" $CUR/temp.txt > $CUR/test_images.txt

convert_imageset_multilabel -resize_height=227 -resize_width=227 "" $CUR/test_images.txt $CUR/test_data_lmdb

compute_image_mean $CUR/test_data_lmdb $CUR/test_mean.binaryproto

#rm $CUR/temp.txt
#rm $CUR/train_images.txt
#rm $CUR/test_images.txt
