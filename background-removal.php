<?php

use Codewithkyrian\Transformers\Models\Auto\AutoModel;
use Codewithkyrian\Transformers\Processors\AutoProcessor;
use Codewithkyrian\Transformers\Transformers;
use Codewithkyrian\Transformers\Utils\Image;
use Codewithkyrian\Transformers\Utils\ImageDriver;

require 'vendor/autoload.php';

Transformers::setup()->setImageDriver(ImageDriver::IMAGICK);

$model = AutoModel::fromPretrained(modelNameOrPath: "briaai/RMBG-1.4");
$processor = AutoProcessor::fromPretrained(modelNameOrPath: 'briaai/RMBG-1.4');
$url = __DIR__ . '/woman-with-bag.jpg';
$image = Image::read($url);
['pixel_values' => $pixelValues] = $processor($image);

['output' => $output] = $model(['input' => $pixelValues]);

$mask = Image::fromTensor($output[0]->multiply(255))
    ->resize($image->width(), $image->height());
$mask->save(__DIR__ . '/mask.png');

$maskedImage = $image->applyMask($mask);

$maskedImage->save('masked.png');
