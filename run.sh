python src/train.py --config config/isic2016.yaml;
python src/train.py --config config/isic2017.yaml;
python src/train.py --config config/isic2018.yaml;
python src/test_new.py --config config/isic2016.yaml;
python src/test_new.py --config config/isic2017.yaml;
python src/test_new.py --config config/isic2018.yaml;
shutdown -h now;