# BertFold

BertFold is a 16 layers BERT model to predict the distance map of protein 3D structure. ProtBert is used as a pre-trained model.

More details are in my article.

## Getting Started

Download pre processed [dataset](https://www.dropbox.com/sh/hhg2jo9ojafn2a0/AAA-ugVoIkIE1wYOWf2fILoua?dl=0).

```
mkdir -p data/ProteinNet/casp12
mv *.pqt data/ProteinNet/casp12/
```

Install [requirements](./requirements.txt). You may have to install [apex](https://github.com/NVIDIA/apex) and [torch-scatter](https://github.com/rusty1s/pytorch_scatter) manually.

Run train script.

```
cd src
python run_train.py params/001.yaml
```

## Results

Once finishing train the model, predicted distance map is available by using the [visualization script](./src/vis_contacts.py)

![1ZU4](images/1ZU4.jpg)