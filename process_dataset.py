# processing the raw data of the video datasets (Something-something and jester)
# generate the meta files:
#   category.txt:               the list of categories.
#   train_videofolder.txt:      each row contains [videoname num_frames classIDX]
#   val_videofolder.txt:        same as above
#
# Bolei Zhou, Dec.2 2017
#
#
import os
import pdb



p_d_root_path = "/data/sjd/d/p_d/TRN-pytorch"
raw_root_path = "/data/sjd/d/video"


dataset_name = 'something-something-v1' # 'jester-v1'
# with open('%s-labels.csv'% dataset_name) as f:
with open(os.path.join(raw_root_path,dataset_name,'labels.csv')) as f:
    print("os.path.join(raw_root_path,dataset_name,'labels.csv'): \n",os.path.join(raw_root_path,dataset_name,'labels.csv'))
    lines = f.readlines()
categories = []
for line in lines:
    line = line.rstrip()
    categories.append(line)
categories = sorted(categories)
# with open('category.txt','w') as f:
with open(os.path.join(p_d_root_path,dataset_name,'category.txt'),'w') as f:
    f.write('\n'.join(categories))

dict_categories = {}
for i, category in enumerate(categories):
    dict_categories[category] = i

print("dict_categories: \n",dict_categories)

# files_input = ['%s-validation.csv'%dataset_name,'%s-train.csv'%dataset_name]
files_input = [os.path.join(raw_root_path,dataset_name,'validation.csv'),os.path.join(raw_root_path,dataset_name,'train.csv')]
# files_output = ['val_videofolder.txt','train_videofolder.txt']
files_output = [os.path.join(p_d_root_path,dataset_name,'val_videofolder.txt'),os.path.join(p_d_root_path,dataset_name,'train_videofolder.txt')]
for (filename_input, filename_output) in zip(files_input, files_output):
    with open(filename_input) as f:
        print("filename_input: \n",filename_input)
        lines = f.readlines()
    folders = []
    idx_categories = []
    for line in lines:
        # print("line: \n",line )
        line = line.rstrip()
        items = line.split(';')
        folders.append(items[0])
        # idx_categories.append(os.path.join(dict_categories[items[1]]))
        idx_categories.append(dict_categories[items[1]])
    output = []
    for i in range(len(folders)):
        curFolder = folders[i]
        curIDX = idx_categories[i]
        # counting the number of frames in each video folders
        # dir_files = os.listdir(os.path.join('20bn-%s'%dataset_name, curFolder))

        dir_files = os.listdir(os.path.join(os.path.join(raw_root_path,dataset_name,'unzip','20bn-%s'%dataset_name), curFolder))

        output.append('%s %d %d'%(curFolder, len(dir_files), curIDX))
        print('%d/%d'%(i, len(folders)))
    with open(filename_output,'w') as f:
        f.write('\n'.join(output))
