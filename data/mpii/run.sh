NAME="mpii-240"
NEW_NAME="mpii-240-1"
python get_meta.py --image_size=240 --input_dir=/home/yi/Downloads/$NAME --meta_file=./$NAME/meta.pkl --second_meta_file=./$NAME/meta_2.pkl --image_diff_file=./$NAME/diff.txt --discontinue_file=./$NAME/discontinue.txt
python get_image_diff.py --image_size=240 --input_dir=/home/yi/Downloads/$NAME --meta_file=./$NAME/meta.pkl --second_meta_file=./$NAME/meta_2.pkl --image_diff_file=./$NAME/diff.txt --discontinue_file=./$NAME/discontinue.txt
python get_discontinue.py --image_size=240 --input_dir=/home/yi/Downloads/$NAME --meta_file=./$NAME/meta.pkl --second_meta_file=./$NAME/meta_2.pkl --image_diff_file=./$NAME/diff.txt --discontinue_file=./$NAME/discontinue.txt
python split_folder.py --image_size=240 --split_dir=/home/yi/Downloads/$NEW_NAME --split_round=1 --meta_file=./$NAME/meta.pkl --second_meta_file=./$NAME/meta_2.pkl --image_diff_file=./$NAME/diff.txt --discontinue_file=./$NAME/discontinue.txt
