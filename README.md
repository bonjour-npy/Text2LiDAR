# Text2LiDAR

## Basic Information

[ECCV2024] Text2LiDAR: Text-guided LiDAR Point Clouds Generation via Equirectangular Transformer

You can download the prepared nuLiDARtext directly from this link:  
https://pan.baidu.com/s/15SQPlxt_zZJQScNdKLC_YQ and the password is 1661.

You can also download nuScenes from this link:  
https://nuscenes.org/nuscenes#download and replace the scene.json file with nuLiDARtext.json.

## Supplementary

You can refer to this [CSDN blog](https://blog.csdn.net/Miller_em/article/details/132025409) to download dataset from pan.baidu.com via bash without UI.

You can refer to this [nuSences 数据集解析以及 nuScenes devkit 的使用](https://blog.csdn.net/zyw2002/article/details/128292100) to learn how to use nuLIDARtext dataset which is based on nuScenes.

You can also refer to this [nuScenes 数据集一文弄懂](https://mp.weixin.qq.com/s/e471aEFUOhsYPbmvQKIZLg) to gain a better understand of nuScenes dataset.

## Analysis of Dataset

- `log.json`：共有两辆数据采集车，编号分别为 `n008` 和 `n015`，在 log.json 中记录了每辆车进行数据采集的日期和地点，共 68 个 log records。
- `map.json`：存储四张地图的信息以及指向每张地图对应的日志数据。
- `scene.json`：共有 850 个场景（scene），每个场景是一个 20s 长的帧序列。在 scene.json 中记录了每个场景对应的描述、采集的样本数以及对应的 log 记录，并且以外键的形式额外提供了每个场景的第一个样本的 token 以及最后一个样本的 token。一个 scene 对应一个 log，一个 log 可以对应多个 scene。
- `sample.json`：记录所有场景对应的关键 sample 帧（LiDAR 数据），包含当前关键帧的 token、前一关键帧的 token 以及后一关键帧的 token，形成链表结构表示时间序列。共计 3,4149 个 LiDAR sample 关键帧。
- `sample_data.json`：记录采集的所有数据对应的真实数据文件名称，并以链表的形式形成时间序列。包括 Camera data、Radar data 以及 LiDAR data 的所有 sample 关键帧以及 sweep 非关键帧。其中，LiDAR 数据的关键帧就是 `sample.json` 文件中的 record（时间戳与 `sample.json` 中的时间戳基本相同），非关键帧的所有数据以及关键帧的 LiDAR 之外的其他数据围绕关键帧的 LiDAR 数据展开存储（时间戳是在 LiDAR 关键帧的前后）。共计 2,631,083 个 data。
- `category.json`：23 个类别，定义了物体的分类，如人、动物、载具、可移动物体或不可移动物体等。
- `attribute.json`：8 个属性，定义了物体的状态，attribute 在 category 不变的情况下改变，如载具是运动状态还是静止状态。
- `visibility.json`：4 个等级，定义了 `sample_annotation.json` 中每个 annotation 的可见度。
- `instance.json`：存储某一 scene 中标注的实例信息，如载具、行人等，每个 instance 可能在同一 scene 的不同 sample 中都有标注。
- `sample_annotation.json`：记录从 sample 关键帧中标注的 bounding box 信息，以及对应的实例（instance）、属性（attribute）以及可见度（visibility）。一个 sample 关键帧可以拥有多个 annotation。
- nuLiDARtext 和 nuScenes 的配对文本描述主要存储于 `scene.json` 中，可以与 nuScenes 中的 `scene.json` 进行比较，查看 nuLiDARtext 的作者进行的针对性修改。