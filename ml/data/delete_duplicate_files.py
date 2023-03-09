import os

input_file = '/org/temp/anon/data/new_images_5M/train.csv'
input_dir = '/org/temp/anon/data/new_images_10M/train_img/'



i = 0
for line in open(input_file).readlines():
    i = i+1
    if i == 1: continue
    file_name = line.split(",")[0]
    src_file = os.path.join(input_dir, file_name)
    print(src_file)
    #shutils.copy(src_file, "test_img")
    #os.system("ln -s %s %s" % (src_file, "test_img_50K"))
    os.system("rm %s" % (src_file))