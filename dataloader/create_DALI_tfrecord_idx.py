from subprocess import call
import os.path

tfrecord = "/home/guyuchao/ssd/dataset/lsun-master/lsun_torch_tfrecord2/-r08.tfrecords"
tfrecord_idx = "/home/guyuchao/ssd/dataset/lsun-master/lsun_torch_tfrecord2/-r08.idx"
tfrecord2idx_script = "/home/guyuchao/ssd/PycharmProjects/progressive/dataset_tool/tfrecord2idx"

if not os.path.exists("idx_files"):
    os.mkdir("idx_files")

if not os.path.isfile(tfrecord_idx):
    call([tfrecord2idx_script, tfrecord, tfrecord_idx])
