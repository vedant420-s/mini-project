# PowerShell script to download sample non-chest images for detector training

# Get the script directory and construct absolute paths
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$trainDir = Join-Path $scriptDir "data\detector_dataset\train\not_chest_xray"
$testDir = Join-Path $scriptDir "data\detector_dataset\test\not_chest_xray"

# Create directories if they don't exist
New-Item -ItemType Directory -Path $trainDir -Force
New-Item -ItemType Directory -Path $testDir -Force

# List of image URLs (free public images)
$imageUrls = @(
    # Dog images from dog.ceo
    "https://images.dog.ceo/breeds/hound-afghan/n02088094_1003.jpg",
    "https://images.dog.ceo/breeds/labrador/n02099712_100.jpg",
    "https://images.dog.ceo/breeds/pug/n02110958_100.jpg",
    "https://images.dog.ceo/breeds/beagle/n02088364_100.jpg",
    "https://images.dog.ceo/breeds/boxer/n02108089_100.jpg",
    "https://images.dog.ceo/breeds/golden/n02099601_100.jpg",
    "https://images.dog.ceo/breeds/terrier/n02093754_100.jpg",
    "https://images.dog.ceo/breeds/spaniel/n02102177_100.jpg",
    "https://images.dog.ceo/breeds/retriever/n02099601_200.jpg",
    "https://images.dog.ceo/breeds/collie/n02106030_100.jpg",

    # Cat images from thecatapi.com
    "https://cdn2.thecatapi.com/images/4fs.jpg",
    "https://cdn2.thecatapi.com/images/9cc.jpg",
    "https://cdn2.thecatapi.com/images/MTY3ODIyMQ.jpg",
    "https://cdn2.thecatapi.com/images/d7g.jpg",
    "https://cdn2.thecatapi.com/images/2pb.jpg",
    "https://cdn2.thecatapi.com/images/3dj.jpg",
    "https://cdn2.thecatapi.com/images/7dj.jpg",
    "https://cdn2.thecatapi.com/images/8dj.jpg",
    "https://cdn2.thecatapi.com/images/9dj.jpg",
    "https://cdn2.thecatapi.com/images/10dj.jpg",

    # Random objects from picsum.photos
    "https://picsum.photos/224/224?random=1",
    "https://picsum.photos/224/224?random=2",
    "https://picsum.photos/224/224?random=3",
    "https://picsum.photos/224/224?random=4",
    "https://picsum.photos/224/224?random=5",
    "https://picsum.photos/224/224?random=6",
    "https://picsum.photos/224/224?random=7",
    "https://picsum.photos/224/224?random=8",
    "https://picsum.photos/224/224?random=9",
    "https://picsum.photos/224/224?random=10",

    # More random images
    "https://picsum.photos/224/224?random=11",
    "https://picsum.photos/224/224?random=12",
    "https://picsum.photos/224/224?random=13",
    "https://picsum.photos/224/224?random=14",
    "https://picsum.photos/224/224?random=15",
    "https://picsum.photos/224/224?random=16",
    "https://picsum.photos/224/224?random=17",
    "https://picsum.photos/224/224?random=18",
    "https://picsum.photos/224/224?random=19",
    "https://picsum.photos/224/224?random=20"
)

Write-Host "Downloading sample non-chest images for detector training..."
Write-Host "This may take a few minutes..."

# Download images to training folder (first 30)
for ($i = 0; $i -lt 30 -and $i -lt $imageUrls.Length; $i++) {
    $url = $imageUrls[$i]
    $filename = "train_image_$($i+1).jpg"
    $outputPath = Join-Path $trainDir $filename

    try {
        Invoke-WebRequest -Uri $url -OutFile $outputPath -TimeoutSec 10
        Write-Host "Downloaded: $filename"
    } catch {
        Write-Host "Failed to download: $filename"
    }
}

# Download images to test folder (next 10)
for ($i = 30; $i -lt 40 -and $i -lt $imageUrls.Length; $i++) {
    $url = $imageUrls[$i]
    $filename = "test_image_$($i-29).jpg"
    $outputPath = Join-Path $testDir $filename

    try {
        Invoke-WebRequest -Uri $url -OutFile $outputPath -TimeoutSec 10
        Write-Host "Downloaded: $filename"
    } catch {
        Write-Host "Failed to download: $filename"
    }
}

Write-Host "Download complete!"
Write-Host "Training images: $(Get-ChildItem $trainDir | Measure-Object | Select-Object -ExpandProperty Count)"
Write-Host "Test images: $(Get-ChildItem $testDir | Measure-Object | Select-Object -ExpandProperty Count)"
