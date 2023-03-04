## Test SGHR on your own dataset

#### Step I: organize your dataset and log it into ```dataops/dataset```
You can follow [YOHO](https://github.com/HpWang-whu/YOHO/tree/master/others) to organize your own dataset, place the dataset files to ```data```, and log it into ```dataops/dataset```.

#### Step II: prepare yoho-desc
Extract YOHO-Desc on (randomly sampled) keypoints following [YOHO](https://github.com/HpWang-whu/YOHO/tree/master/others). Note that the extracted YOHO-Desc is rotation-equivariant but we only need the rotation-invariant yoho-desc.
We thus provide a script to convert YOHO-Desc to yoho-desc, you can run it as follows:
```
python data/convert.py --dataset \<your dataset name\> --yoho_desc_dir \<folder of YOHO_FCGF\>
```
where \<your dataset name\> is the name of your dataset (the folder name saved to '''data''' and the name logged in ```dataops/dataset```),  \<folder of YOHO_FCGF\> is the folder where YOHO-Desc is saved by [YOHO](https://github.com/HpWang-whu/YOHO/tree/master/others) and it is typically the folder end with ```YOHO_FCGF/Testset```.

#### Step III: conduct multiview registration
- Extract global features
```
python Test.py --dataset \<your dataset name\>
```
- Construct sparse graph and condut multiview registration
```
python Test_cycle.py --dataset \<your dataset name\> --topk *** --inlierd ***
```
where topk means use top-k scored edges in sparse graph construction, inlierd means the inlier threshold in RANSAC.