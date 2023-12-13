python test.py --multiprocessing_distributed \
    --test_data_path /path/to/avsbench/test/ \
    --test_gt_path /path/to/avsbench/test/gt_masks \
    --experiment_name wsavs_avsbench \
    --model 'wsavs' \
    --imgnet_type resnet50 --audnet_type resnet50 \
    --trainset 'avsbench' \
    --testset 'avsbench' \
    --save_pred_masks