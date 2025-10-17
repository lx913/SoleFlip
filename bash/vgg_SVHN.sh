for i in {1..3}
do
    echo "运行第 $i 次"
    python train_clean_model.py -dataset SVHN -backbone vgg -device 1 -batch_size 512 -epochs 200 -lr 1e-1 -weight_decay 1e-3 -model_num $i -optimizer SGD
done
