# Dataset preparation

If you want to reproduce the results in the paper for benchmark evaluation and training, you will need to setup dataset.


### Rock-COCO (Scaled Front Dataset or Hammer-Rocks Dataset)
- Download the images and annotations from [Scaled Front Dataset and Hammer Rocks Dataset](http://datos.uchile.cl/dataset.xhtml?persistentId=doi:10.34691/FK2/1GQBHK).
- Place the data (or create symlinks) to make the data folder like:

  ~~~
  ${RockyCenterNet_ROOT}
  |-- data
  `-- |-- rock_hammer_dataset_v1_v2
      `-- |-- annotations
          |   |-- train.json
          |   |-- val.json
          |   |-- test.json
          |---|-- train
          |---|-- val
          `---|-- test
