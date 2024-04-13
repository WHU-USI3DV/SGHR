### Register a set of point clouds

#### Step I: Download yoho checkpoints:
Save [back ckpt](https://github.com/HpWang-whu/YOHO/blob/master/model/Backbone/best_val_checkpoint.pth) to ```yoho/ckpts/backbone```
Save [yoho ckpt](https://github.com/HpWang-whu/YOHO/blob/master/model/PartI_train/model_best.pth) to ```yoho/ckpts/yoho```

#### Step II: Run SGHR on your raw point clouds.
You can just save your point clouds (.ply/.pcd/.npy) to somewhere like ```data/demo```. 
Then just run:
```
python demo.py --pcdir data/demo
```
The registration results will be saved to:
- ```data/demo/registration/kpts```: sampled kpts of the point clouds in "pc_dir" for yoho feature extraction;
- ```data/demo/registration/yoho```: yoho features of the point clouds in "pc_dir";
- ```data/demo/registration/multi_reg```: overlap.txt saving estimated overlap ratio, pose.txt saving the estimated pc pose.