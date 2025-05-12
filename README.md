# Binary Semantic Segmentation in PyTorch

You can use [uv](https://github.com/astral-sh/uv) for the environment.
```
uv sync
```

## Dataset

You can use [CVAT](https://github.com/cvat-ai/cvat) to annotate images and export them as [CamVid format](https://docs.cvat.ai/docs/manual/advanced/formats/format-camvid/). You can download sample balloon dataset using `python download_dataset.py`.

```
data_dir
├── default
│   ├── <image-file1>.png
│   ├── <image-file2>.png
│   └── ...
└── defaultannot
    ├── <annot-file1>.png
    ├── <annot-file2>.png
    └── ...
```

Modify `config/create_dataset.yaml`

`python create_dataset.py`
- Split dataset as train, val, test
- Copy images & binary masks to appropriate directories

## Training

Modify `config/main.yaml`

`python train.py`

## Evaluation (With Mask)

Modify `config/main.yaml`

`python eval.py`

## Prediction (Without Mask, Only Input Images)

Modify `config/predict.yaml`

`python predict.py`

## Gradio

Modify `app.py` if you want to change the behavior

`python app.py`