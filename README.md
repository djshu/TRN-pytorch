
#### environment
python3.6 pytorch 0.3.1

#### note
always use git clone --recursive https://github.com/djshu/TRN-pytorch to clone this project Otherwise you will not be able to use the inception series CNN architecture.

## S0, BNInception RelationModuleMultiScaleWithClassifier_h_after_sum
### train
```bash
CUDA_VISIBLE_DEVICES=6,7  python -u main.py --dataset something-something-v1 --modality RGB \
                     --arch BNInception --num_segments 8 \
                     --consensus_type RelationModuleMultiScaleWithClassifier_h_after_sum --batch-size 32 --eval_freq 5\
                      > /data/sjd/d/p_d/TRN-pytorch/something-something-v1/log/log-something-something-v1-RelationModuleMultiScaleWithClassifier_h_after_sum

```

### test
```bash
CUDA_VISIBLE_DEVICES=7 python -u test_models.py  --dataset something-something-v1 --modality RGB \
/data/sjd/d/p_d/TRN-pytorch/something-something-v1/model/TRN_something-something-v1_RGB_BNInception_RelationModuleMultiScaleWithClassifier_h_after_sum_segment8_best.pth.tar   \
   --arch BNInception --crop_fusion_type RelationModuleMultiScaleWithClassifier_h_after_sum --test_segments 8 > \
   /data/sjd/d/p_d/TRN-pytorch/something-something-v1/log/log-something-something-v1-BNInception-RelationModuleMultiScaleWithClassifier_h_after_sum-test

```


## S1 , BNInception TRNmultiscale
### train
```bash
CUDA_VISIBLE_DEVICES=2,3 python -u main.py --dataset something-something-v1 --modality RGB \
                     --arch BNInception --num_segments 8 \
                     --consensus_type TRNmultiscale --batch-size 32 --eval_freq 5\
                      >/data/sjd/d/p_d/TRN-pytorch/something-something-v1/log/log-something-something-v1

```

### test
```bash
CUDA_VISIBLE_DEVICES=1 python -u test_models.py  --dataset something-something-v1 --modality RGB \
/data/sjd/d/p_d/TRN-pytorch/something-something-v1/model/TRN_something-something-v1_RGB_BNInception_TRNmultiscale_segment8_best.pth.tar \
   --arch BNInception --crop_fusion_type TRNmultiscale --test_segments 8 > \
   /data/sjd/d/p_d/TRN-pytorch/something-something-v1/log/log-something-something-v1-BNInception-TRNmultiscale-test

```


## S2, BNInception RelationModuleMultiScaleWithClassifier
### train
```bash
CUDA_VISIBLE_DEVICES=6,7 python -u main.py something-something-v1 RGB \
                     --arch BNInception --num_segments 8 \
                     --consensus_type RelationModuleMultiScaleWithClassifier --batch-size 32 --eval_freq 5\
                      > /data/sjd/d/p_d/TRN-pytorch/something-something-v1/log/log-something-something-v1-RelationModuleMultiScaleWithClassifier

```

### test
```bash
CUDA_VISIBLE_DEVICES=6 python -u test_models.py  --dataset something-something-v1 --modality RGB \
/data/sjd/d/p_d/TRN-pytorch/something-something-v1/model/TRN_something-something-v1_RGB_BNInception_RelationModuleMultiScaleWithClassifier_segment8_best.pth.tar  \
   --arch BNInception --crop_fusion_type RelationModuleMultiScaleWithClassifier --test_segments 8 > \
   /data/sjd/d/p_d/TRN-pytorch/something-something-v1/log/log-something-something-v1-BNInception-RelationModuleMultiScaleWithClassifier-test

```


## S3, BNInception MultiScaleLSTM
### train
```bash
CUDA_VISIBLE_DEVICES=0,1 python -u main.py --dataset something-something-v1 --modality RGB \
                     --arch BNInception --num_segments 8 \
                     --consensus_type MultiScaleLSTM --batch-size 32 --eval_freq 5\
                      > /data/sjd/d/p_d/TRN-pytorch/something-something-v1/log/log-something-something-v1-MultiScaleLSTM

```

### test
```bash
CUDA_VISIBLE_DEVICES=7 python -u test_models.py  --dataset  something-something-v1  --modality RGB \
/data/sjd/d/p_d/TRN-pytorch/something-something-v1/model/TRN_something-something-v1_RGB_BNInception_MultiScaleLSTM_segment8_best.pth.tar \
   --arch BNInception --crop_fusion_type MultiScaleLSTM --test_segments 8 \
   > /data/sjd/d/p_d/TRN-pytorch/something-something-v1/log/log-something-something-v1-BNInception-MultiScaleLSTM-test

```

## S4, RelationModuleMultiScaleLSTM_RelationModuleMultiScale_num_layer2
### train
```bash
CUDA_VISIBLE_DEVICES=6,7 python -u main.py --dataset something-something-v1 --modality RGB \
                     --arch BNInception --num_segments 8 \
                     --consensus_type RelationModuleMultiScaleLSTM_RelationModuleMultiScale_num_layer2 --batch-size 32 --eval_freq 5\
                      > /data/sjd/d/p_d/TRN-pytorch/something-something-v1/log/log-something-something-v1-RelationModuleMultiScaleLSTM_RelationModuleMultiScale_num_layer2

```

### test
```bash
CUDA_VISIBLE_DEVICES=7 python -u test_models.py  --dataset  something-something-v1  --modality RGB \
/data/sjd/d/p_d/TRN-pytorch/something-something-v1/model/TRN_something-something-v1_RGB_BNInception_RelationModuleMultiScaleLSTM_RelationModuleMultiScale_num_layer2_segment8_best.pth.tar  \
   --arch BNInception --crop_fusion_type RelationModuleMultiScaleLSTM_RelationModuleMultiScale_num_layer2 --test_segments 8 \
   > /data/sjd/d/p_d/TRN-pytorch/something-something-v1/log/log-something-something-v1-BNInception-RelationModuleMultiScaleLSTM_RelationModuleMultiScale_num_layer2-test

```

## S5, RelationModuleMultiScaleLSTM_RelationModuleMultiScale_num_layer1
### train
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -u main.py --dataset something-something-v1 --modality RGB \
                     --arch BNInception --num_segments 8 \
                     --consensus_type RelationModuleMultiScaleLSTM_RelationModuleMultiScale_num_layer1 --batch-size 32 --eval_freq 5\
                      > /data/sjd/d/p_d/TRN-pytorch/something-something-v1/log/log-something-something-v1-RelationModuleMultiScaleLSTM_RelationModuleMultiScale_num_layer1

```

### test
```bash
CUDA_VISIBLE_DEVICES=0 python -u test_models.py  --dataset  something-something-v1  --modality RGB \
/data/sjd/d/p_d/TRN-pytorch/something-something-v1/model/TRN_something-something-v1_RGB_BNInception_RelationModuleMultiScaleLSTM_RelationModuleMultiScale_num_layer1_segment8_best.pth.tar  \
   --arch BNInception --crop_fusion_type RelationModuleMultiScaleLSTM_RelationModuleMultiScale_num_layer1 --test_segments 8 \
   > /data/sjd/d/p_d/TRN-pytorch/something-something-v1/log/log-something-something-v1-BNInception-RelationModuleMultiScaleLSTM_RelationModuleMultiScale_num_layer1-test

```

## S6, TSN BNInception avg
### train
```bash
CUDA_VISIBLE_DEVICES=6,7 python -u main.py --dataset something-something-v1 --modality RGB \
                     --arch BNInception --num_segments 8 \
                     --consensus_type avg --batch-size 32 --eval_freq 5\
                      >/data/sjd/d/p_d/TRN-pytorch/something-something-v1/log/log-something-something-v1-tsn-avg
```


### test
```bash
CUDA_VISIBLE_DEVICES=7 python -u test_models.py  --dataset something-something-v1 --modality RGB \
/data/sjd/d/p_d/TRN-pytorch/something-something-v1/model/TRN_something-something-v1_RGB_BNInception_avg_segment8_best.pth.tar  \
   --arch BNInception --crop_fusion_type avg --test_segments 8 > \
   /data/sjd/d/p_d/TRN-pytorch/something-something-v1/log/log-something-something-v1-BNInception-avg-test
```



## S7 resnet101
### train
```bash
CUDA_VISIBLE_DEVICES=1,2,3,4 python -u main.py something-something-v1 RGB \
                     --arch resnet101 --num_segments 8 \
                     --consensus_type TRNmultiscale --batch-size 32 --eval_freq 5\
                      >/data/sjd/d/p_d/TRN-pytorch/something-something-v1/log/log-something-something-v1-resnet101
```


### test
```bash
CUDA_VISIBLE_DEVICES=0 python -u test_models.py  --dataset  something-something-v1  --modality RGB \
/data/sjd/d/p_d/TRN-pytorch/something-something-v1/model/TRN_something-something-v1_RGB_resnet101_TRNmultiscale_segment8_best.pth.tar  \
   --arch resnet101 --crop_fusion_type TRNmultiscale --test_segments 8 \
   > /data/sjd/d/p_d/TRN-pytorch/something-something-v1/log/log-something-something-v1-resnet101-bs16-TRNmultiscale-test
```


### Reference:
B. Zhou, A. Andonian, and A. Torralba. Temporal Relational Reasoning in Videos. European Conference on Computer Vision (ECCV), 2018. [PDF](https://arxiv.org/pdf/1711.08496.pdf)
```
@article{zhou2017temporalrelation,
    title = {Temporal Relational Reasoning in Videos},
    author = {Zhou, Bolei and Andonian, Alex and Oliva, Aude and Torralba, Antonio},
    journal={European Conference on Computer Vision},
    year={2018}
}
```

### Acknowledgement
this repository is based on https://github.com/metalbubble/TRN-pytorch, and some modification is added. We thank Bolei Zhou for releasing the codebase. Something-something dataset and Jester dataset are from [TwentyBN](https://www.twentybn.com/), we really appreciate their effort to build such nice video datasets. Please refer to [their dataset website](https://www.twentybn.com/datasets/something-something) for the proper usage of the data.

