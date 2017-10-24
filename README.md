# Show and Tell: A Neural Image Caption Generator*

**Rewritten version of [tensorflow/models/im2txt](https://github.com/tensorflow/models/tree/master/research/im2txt)**

A TensorFlow implementation* of the image-to-text model described in the paper:

"Show and Tell: Lessons learned from the 2015 MSCOCO Image Captioning Challenge."

Oriol Vinyals, Alexander Toshev, Samy Bengio, Dumitru Erhan.

IEEE transactions on pattern analysis and machine intelligence (2016).

Full text available at: http://arxiv.org/abs/1609.06647

**Some key elements was changed:*

*Advantages*
* CNN removed from training loop (train is ~150 times faster)
* Less memory needed for same batch
* Model written with `tf.estimator.Estimator` and `tf.dataset.Dataset`

*Disadvantages*
* Model spited in 2 parts
* CNN removed from training loop (paper's finetun not possible, CNN is not trainable)


## Preprocessing 
```bash
python3 preprocess.py
```

##Training 

150 epoch training will take 8h with 1024 bath on GTX1080 
```bash
python3 im2txt.py train --optimizer Adagrad  --initial_learning_rate 4 --num_epochs_per_decay 20  --batch_size 1024 --max_train_epochs 150 --save_checkpoints_steps 1000 --log_step_count_steps 500
```

Train loss
![Train loss](doc/Screen%20Shot%202017-10-21%20at%201.03.46%20PM.png)


## Testing
Test last model (just past output in Markdown editor)
```bash
python3 im2txt.py test --test_urls https://cdn132.picsart.com/245236991011202.jpg,https://cdn141.picsart.com/245236965014202.jpg
```

force run on CPU
```bash
CUDA_VISIBLE_DEVICES=""; python3 im2txt.py test ...
``` 


![]( https://cdn132.picsart.com/245236991011202.jpg )
 `<S> a woman in a dress standing on a street corner . </S> (-11.629340171813965)`

![]( https://cdn141.picsart.com/245236965014202.jpg )
 `<S> a man sitting on a bench in a room . </S> (-8.756865501403809)`
