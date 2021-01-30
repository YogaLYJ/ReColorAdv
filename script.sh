for i in $( seq 0 0 )
do   
    CUDA_VISIBLE_DEVICES=$1 python attack_cifar.py \
    --dataset ../cifar10test/cifar10_test_${i} \
    --gt ../cifar10test/cifar10_test_${i}.csv \
    --model $2 \
    --save_fig \
    --trial 1
done