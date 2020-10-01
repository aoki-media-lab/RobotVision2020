# For contributor

## コードに関して

blackとisortを使ってコードを整形してからコミットするようにする．
flake8を用いて論理エラー等も見つけて取り除くようにしておく．

### Installation

```bash
pip install black isort
```

### Formatting

```bash
black .
isort -rc .
```
