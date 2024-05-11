python src/train.py --config config/isic2016.yaml;
python src/train.py --config config/isic2017.yaml;
python src/train.py --config config/isic2018.yaml;
python src/test_new.py --config config/isic2016.yaml;
python src/test_new.py --config config/isic2017.yaml;
python src/test_new.py --config config/isic2018.yaml;
cd /autodl-fs/data/xboundformer/ && zip -r result.zip result && rm -rf result;
shutdown -h now;