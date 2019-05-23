from timeit import default_timer as timer
from nvidia.dali.plugin.pytorch import DALIGenericIterator
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types
import nvidia.dali.tfrecord as tfrec

class CommonPipeline(Pipeline):
    def __init__(self, batch_size, num_threads, device_id,size=1024):
        super(CommonPipeline, self).__init__(batch_size, num_threads, device_id)

        self.decode = ops.nvJPEGDecoder(device = "mixed", output_type = types.RGB)
        self.cmn = ops.NormalizePermute(device = "gpu",
                                    height=size,
                                    width=size,
                                    output_dtype = types.FLOAT,
                                    image_type = types.RGB,
                                    mean=[0.5 * 255, 0.5 * 255, 0.5 * 255],
                                    std=[0.5 * 255, 0.5 * 255, 0.5 * 255])
    def base_define_graph(self, inputs):
        images = self.decode(inputs)
        output = self.cmn(images)
        return output

class TFRecordPipeline(CommonPipeline):
    def __init__(self, batch_size, num_threads, device_id,size=1024,
                 path="/home/guyuchao/ssd/dataset/cityscape/leftImg8bit/dalirecord/dataset-r10.tfrecords",
                 index_path="/home/guyuchao/ssd/dataset/cityscape/leftImg8bit/dalirecord/dataset-r10.idx"):
        super(TFRecordPipeline, self).__init__(batch_size, num_threads, device_id,size)
        self.input = ops.TFRecordReader(path = path,
                                        index_path = index_path,
                                        features = {"image/encoded" : tfrec.FixedLenFeature((), tfrec.string, "")})

    def define_graph(self):
        inputs = self.input(name="Reader")
        images = inputs["image/encoded"]
        return self.base_define_graph(images)

class Lsun_Loader(object):
    def __init__(self,device_idx=1):
        self.batch_table = {4: 128, 8: 128, 16: 128, 32: 64, 64: 32, 128: 16, 256: 8}
        self.device_idx=device_idx
        #self.update(resolution_level=2)

    def update(self,resolution_level,batch_size):
        '''


        :param resolution_level:
            resolution_level:
                2-2,3-8,4-16,5-32,6-64,7-128,8-256
        :return:
        '''
        assert resolution_level>=2 and resolution_level<=8,"res error"
        self.batchsize=int(self.batch_table[pow(2,resolution_level)])
        self.pipeline=TFRecordPipeline(batch_size=batch_size,
                                       size=pow(2,resolution_level),
                                       num_threads=4,
                                       device_id = self.device_idx,
                                       path="/home/guyuchao/ssd/dataset/lsun-master/lsun_torch_tfrecord2/-r%02d.tfrecords"%resolution_level,
                                       index_path="/home/guyuchao/ssd/dataset/lsun-master/lsun_torch_tfrecord2/-r%02d.idx"%resolution_level)
        self.pipeline.build()

        self.dali_iter=DALIGenericIterator([self.pipeline], ["image/encoded"], self.pipeline.epoch_size("Reader"),auto_reset=True)


    def get_batch(self):
        #self.dali_iter.reset()
        return self.dali_iter.next()[0]["image/encoded"]

'''
test_batch_size = 64

def speedtest(pipeclass, batch, n_threads):
    pipe = pipeclass(batch, n_threads, 1)
    pipe.build()
    # warmup
    for i in range(5):
        pipe.run()
    # test
    n_test = 20
    t_start = timer()
    for i in range(n_test):
        pipe.run()
    t = timer() - t_start
    print("Speed: {} imgs/s".format((n_test * batch)/t))
speedtest(TFRecordPipeline,test_batch_size,4)
'''

'''


batch_size = 16

pipe = TFRecordPipeline(batch_size=batch_size, num_threads=4, device_id = 1)
pipe.build()
#pipout=pipe.run()


dali_iter = DALIGenericIterator([pipe], ["image/encoded"], pipe.epoch_size("Reader"))
print(dali_iter.next()[0]["image/encoded"].shape)
dsa
'''
#Lsun_Loader=Lsun_Loader(device_idx=1)
if __name__=="__main__":
    Lsun_Loader.update(2,512)
    i=0
    from time import sleep
    while True:
        i+=512
        Lsun_Loader.get_batch()
        print(i)
        if i>150000:
            print(Lsun_Loader.get_batch().max())
            dsa
        sleep(0.02)
