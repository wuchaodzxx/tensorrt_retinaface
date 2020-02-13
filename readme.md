# TensorRT ONNX

This is an lib which target at inference model using TensorRT and model from onnx format. Nowdays massive model are trained using pytorch and it's easy to export into onnx format, with the benefit of huge acceleration by TensorRT, combines onnx && TensorRT would be a very important in production.

Currently, we support models as:

- [x] Refinenet segmentation example (theoretical all segmentation models can be supported );
- [x] Retinaface fastest face detection and landmark estimation;
- [ ] FasterRCNN;
- [ ] MaskRCNN;
- [x] YoloV3;


## Install

All you need is install TensorRT 6.0 and install on:

```
# move it into ~/TensorRT (or soft link)
./build_all.sh
```


## Demo

1. **Refinenet Segmentation**:

   the refinenet body part segmentation onnx model can be download from 链接: https://pan.baidu.com/s/1zpUAKY2CwVYaKi9QnWi66Q 提取码: aium . The training framework from [here](https://github.com/DrSleep/light-weight-refinenet), you can training your down dataset and replace the example onnx model to yours, it will works as you expected.
   
   ```
   ./examples/demo_refinenet no_have_serialize_txt float32 save_serialize_name here_your_video_file_name_or_cam here_your_onnxmodel_name
./trt_engine_refinenetseg refinenet_seg.trt /media/fagangjin/wd/permanent/datasets/TestVideos/elonmask.mp4
   ```
   
2. **RetinaFace Landmark Detection**:

   this model already included under the `models` folder. you can convert it to trt first, then load directly into infer engine for inference. Make sure you have TensorRT 6 installed, and you shuold get onnx model from here: http://manaai.cn/aicodes_detail3.html?id=46 

   ```
   ./build_all.sh
   ./build/examples/trt_engine_retinaface ../models/retinaface.trt ./images/0--Parade_0_Parade_marchingband_1_657.jpg   
   ```

   

## Build

to build the lib and examples, you gonna need TensorRT download, we find TensorRT from `~/tensorrt` by default. Then run:

```
./build_all.sh
```





## Copyright

All rights reserved by Fagang Jin, codes released under Apache License.

 