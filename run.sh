move_file_to_folder() {
    local file="$1"  # 待移动的文件
    local dest_folder="$2"  # 目标文件夹

    # 检查目标文件夹是否存在，如果不存在则创建
    if [ ! -d "$dest_folder" ]; then
        echo "目标文件夹不存在，将创建文件夹..."
        mkdir -p "$dest_folder"
    fi

    # 提取文件名和扩展名
    filename=$(basename -- "$file")
    extension="${filename##*.}"
    filename="${filename%.*}"
    new_filename="$filename.$extension"
    # 判断文件是否存在
    if [ ! -e "$dest_folder/$new_filename" ]; then
        mv "$file" "$dest_folder/$new_filename"
        echo "文件移动完成"
        return
    fi

    # 如果目标文件夹中已存在同名文件，重命名为name(xx).extension 形式
    index=1
    while [ -e "$dest_folder/$filename($index).$extension" ]; do
        ((index++))
    done
    new_filename="$filename($index).$extension"
    # 移动文件
    mv "$file" "$dest_folder/$new_filename"
    echo "文件移动完成"
}

python src/train.py --config config/isic2016.yaml;
python src/train.py --config config/isic2017.yaml;
python src/train.py --config config/isic2018.yaml;
python src/test_new.py --config config/isic2016.yaml;
python src/test_new.py --config config/isic2017.yaml;
python src/test_new.py --config config/isic2018.yaml;
zip -r result.zip result && move_file_to_folder result.zip /autodl-fs/data/xboundformer/;
shutdown -h now;