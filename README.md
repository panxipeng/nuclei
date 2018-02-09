# Nuclei
[2018 Data Science Bowl](https://www.kaggle.com/c/data-science-bowl-2018) project

## Step-by-step from the box
1. Create folders on the project directory:

     ![](https://2.downloader.disk.yandex.ru/disk/25bf8c08781823b1a3771a60caa06d7c2ef4bdcddda745074cc0882071881dfa/5a7b666b/yPLnXPPB9QLsi-9-tOYQ6ScDHH8D12pyiktHGtE1v89xWr6kowrlvmx5MceAnSu6nEckoD2W6pGj3Wje1wrQmw%3D%3D?uid=0&filename=2018-02-07_19-48-49.png&disposition=inline&hash=&limit=0&content_type=image%2Fpng&fsize=1284&hid=c108547a1f013eb931935998c6199ef8&media_type=image&tknv=v2&etag=558db41ed72ed96fbe7ad2e68287fc76)

2. Download [data set](https://www.kaggle.com/c/data-science-bowl-2018/data) from competition page.
3. Unpack data set:

    * stage1_train.zip into __../data/images/train__
    * stage1_test.zip into __../data/images/test__

4. Run __../src/utils/make_npys.py__. This will create folder __../out_files/npy/128_128_split__.
5. Run __../src/main.py__. This will run the process of fitting model and predicting masks on test images, and create the submission file
at the __../sub folder__
