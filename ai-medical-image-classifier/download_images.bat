@echo off
echo Creating directories for detector dataset...

mkdir "data\detector_dataset\train\not_chest_xray" 2>nul
mkdir "data\detector_dataset\test\not_chest_xray" 2>nul

echo Downloading sample non-chest images for detector training...
echo This may take a few minutes...

REM Download dog images from dog.ceo API
curl -L "https://images.dog.ceo/breeds/labrador/n02099712_100.jpg" -o "data\detector_dataset\train\not_chest_xray\train_dog_1.jpg"
curl -L "https://images.dog.ceo/breeds/pug/n02110958_100.jpg" -o "data\detector_dataset\train\not_chest_xray\train_dog_2.jpg"
curl -L "https://images.dog.ceo/breeds/beagle/n02088364_100.jpg" -o "data\detector_dataset\train\not_chest_xray\train_dog_3.jpg"
curl -L "https://images.dog.ceo/breeds/boxer/n02108089_100.jpg" -o "data\detector_dataset\train\not_chest_xray\train_dog_4.jpg"
curl -L "https://images.dog.ceo/breeds/golden/n02099601_100.jpg" -o "data\detector_dataset\train\not_chest_xray\train_dog_5.jpg"

REM Download cat images from thecatapi.com
curl -L "https://cdn2.thecatapi.com/images/4fs.jpg" -o "data\detector_dataset\train\not_chest_xray\train_cat_1.jpg"
curl -L "https://cdn2.thecatapi.com/images/9cc.jpg" -o "data\detector_dataset\train\not_chest_xray\train_cat_2.jpg"
curl -L "https://cdn2.thecatapi.com/images/d7g.jpg" -o "data\detector_dataset\train\not_chest_xray\train_cat_3.jpg"
curl -L "https://cdn2.thecatapi.com/images/2pb.jpg" -o "data\detector_dataset\train\not_chest_xray\train_cat_4.jpg"
curl -L "https://cdn2.thecatapi.com/images/3dj.jpg" -o "data\detector_dataset\train\not_chest_xray\train_cat_5.jpg"

REM Download random images from picsum.photos
curl -L "https://picsum.photos/224/224?random=1" -o "data\detector_dataset\train\not_chest_xray\train_random_1.jpg"
curl -L "https://picsum.photos/224/224?random=2" -o "data\detector_dataset\train\not_chest_xray\train_random_2.jpg"
curl -L "https://picsum.photos/224/224?random=3" -o "data\detector_dataset\train\not_chest_xray\train_random_3.jpg"
curl -L "https://picsum.photos/224/224?random=4" -o "data\detector_dataset\train\not_chest_xray\train_random_4.jpg"
curl -L "https://picsum.photos/224/224?random=5" -o "data\detector_dataset\train\not_chest_xray\train_random_5.jpg"

REM Download test images
curl -L "https://picsum.photos/224/224?random=6" -o "data\detector_dataset\test\not_chest_xray\test_random_1.jpg"
curl -L "https://picsum.photos/224/224?random=7" -o "data\detector_dataset\test\not_chest_xray\test_random_2.jpg"
curl -L "https://images.dog.ceo/breeds/terrier/n02093754_100.jpg" -o "data\detector_dataset\test\not_chest_xray\test_dog_1.jpg"
curl -L "https://cdn2.thecatapi.com/images/7dj.jpg" -o "data\detector_dataset\test\not_chest_xray\test_cat_1.jpg"
curl -L "https://picsum.photos/224/224?random=8" -o "data\detector_dataset\test\not_chest_xray\test_random_3.jpg"

echo Download complete!
echo Checking downloaded files...

dir "data\detector_dataset\train\not_chest_xray" /b
echo.
echo Training images count:
dir "data\detector_dataset\train\not_chest_xray" /b | find /c ".jpg"

echo.
dir "data\detector_dataset\test\not_chest_xray" /b
echo.
echo Test images count:
dir "data\detector_dataset\test\not_chest_xray" /b | find /c ".jpg"

echo.
echo Dataset preparation complete!
pause
