run:
	python train.py --batch_size 32 --checkpoint_interval 10

test:
	python test.py --img_path ./outputs/blurry-headshot.jpg

board:
	tensorboard --logdir=runs
