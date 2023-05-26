import OSToolBox as ostb
import shutil


with open('/home/willalbert/Documents/inputs/data_3d_semantics/train/2013_05_28_drive_train.txt', 'r') as train_file:
    train_lines = train_file.readlines()
train_file.close()

with open('/home/willalbert/Documents/inputs/data_3d_semantics/train/2013_05_28_drive_val.txt', 'r') as val_file:
    val_lines = val_file.readlines()
val_file.close()


for line in train_lines:
    print()
    seq = line[43:45]
    print(seq)
    original = "/home/willalbert/Documents/inputs/"+line[0:-1]
    target = "/home/willalbert/Documents/inputs/training/sequence/"+seq+"/"+line[58:-1]
    print(original)
    print(target)
    shutil.copyfile(original, target)


for line in val_lines:
    print()
    seq = line[43:45]
    print(seq)
    original = "/home/willalbert/Documents/inputs/"+line[0:-1]
    target = "/home/willalbert/Documents/inputs/validation/sequence/"+seq+"/"+line[58:-1]
    print(original)
    print(target)
    shutil.copyfile(original, target)
