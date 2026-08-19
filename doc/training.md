# ScienceBeam Parser Model Training

ScienceBeam Parser uses machine learning models to parse documents.
Pre-trained models are provided and referenced by the configuration.
In order to get performance for the particular domain, it may be necessary to train
models with domain specific documents.

For the sequence model (`delft` etc) the general workflow looks like:

- Generate training data
- Annotate generated training data
- Train model (using `sciencebeam-trainer-delft`)
- Use and evaluate model:
  - Configure new model in `sciencebeam-parser`
  - Convert documents
  - Evaluate converted documents (using `sciencebeam-judge`)

## Generate training data

The training data for the sequential models follows the GROBID training data format.

Note: all commands below also support input and output files in google cloud storage (using `gs://` for the path url)

### Generate `tei` training data for the sequence models

Currently training data will be generated for the following models:

- `segmentation`
- `header`
- `affiliation_address`
- `fulltext`
- `reference_segmenter`
- `citation` (references)
- `figure`
- `table`
- `name` (author names for `header` and `citations`)

```bash
python -m sciencebeam_parser.training.cli.generate_data \
    --source-path="test-data/*.pdf" \
    --output-path="./data/generated-training-data"
```

Using the configured models to pre-annotate the training data:

```bash
python -m sciencebeam_parser.training.cli.generate_data \
    --use-model \
    --source-path="test-data/*.pdf" \
    --output-path="./data/generated-training-data"
```

Note: as the models are hierachical, the parent model needs to be used
  in order to generate data for the child model.
  For example the `segmentation` model will be required for the `header` model.

The output could also be organised into a folder structure by model and type of file:

```bash
python -m sciencebeam_parser.training.cli.generate_data \
    --use-model \
    --use-directory-structure \
    --source-path="test-data/*.pdf" \
    --output-path="./data/generated-training-data"
```

Additionally the `--gzip` argument can be passed in, resulting in gzip (`.gz`) compressed output files.

### Annotating `tei` training data for the sequence models

After the `tei` training data has been generated, it should get reviewed and manually annotated.
Alternatively it can auto-annotated using [sciencebeam-trainer-grobid-tools](https://gitlab.coko.foundation/sciencebeam/sciencebeam-trainer-grobid-tools).

### Generate `delft` training data for the sequence models

From the annotated `tei` training data, we can generate the `delft` training data used for training.

It will do one of the following:

- For models with only `tei` XML files (no layout feature), it will parse the `tei` and generate data using the data generator.
- For models with additional layout data files, it will align the parsed `tei` with the layout data file and add the label to it.

The output matches GROBID's column layout for the model, so it can be mixed with
GROBID's own corpus. The expected layout per model is recorded in
[`grobid_column_layout.yml`](../sciencebeam_parser/resources/grobid_column_layout.yml),
and generating data for a model with no entry there fails rather than guessing.
`python -m sciencebeam_parser.training.cli.check_grobid_column_layout` re-checks
that file against GROBID's published corpora; it downloads them, so it is run by
hand rather than in CI.

Pass `--include-extra-columns` to also emit the columns this project adds on top
of GROBID's layout. Today that is the `segmentation` model's `whole_line_text`,
which `delft` models read as a text feature and `wapiti` templates do not
reference. Training data generated with the flag cannot be mixed with GROBID's
`segmentation` corpus, since it is one column wider.

#### Example command for `segmentation` model

```bash
python -m sciencebeam_parser.training.cli.generate_delft_data \
    --model-name="segmentation" \
    --tei-source-path="data/generated-training-data/segmentation/corpus/tei/*.tei.xml" \
    --raw-source-path="data/generated-training-data/segmentation/corpus/raw/" \
    --delft-output-path="./data/generated-training-data/delft/segmentation/corpus/segmentation.data"
```

#### Example command for `header` model

```bash
python -m sciencebeam_parser.training.cli.generate_delft_data \
    --model-name="header" \
    --tei-source-path="data/generated-training-data/header/corpus/tei/*.tei.xml" \
    --raw-source-path="data/generated-training-data/header/corpus/raw/" \
    --delft-output-path="./data/generated-training-data/delft/header/corpus/header.data"
```

#### Example command for `fulltext` model

```bash
python -m sciencebeam_parser.training.cli.generate_delft_data \
    --model-name="fulltext" \
    --tei-source-path="data/generated-training-data/fulltext/corpus/tei/*.tei.xml" \
    --raw-source-path="data/generated-training-data/fulltext/corpus/raw/" \
    --delft-output-path="./data/generated-training-data/delft/fulltext/corpus/fulltext.data"
```

#### Example command for `figure` model

```bash
python -m sciencebeam_parser.training.cli.generate_delft_data \
    --model-name="figure" \
    --tei-source-path="data/generated-training-data/figure/corpus/tei/*.tei.xml" \
    --raw-source-path="data/generated-training-data/figure/corpus/raw/" \
    --delft-output-path="./data/generated-training-data/delft/figure/corpus/figure.data"
```

#### Example command for `table` model

```bash
python -m sciencebeam_parser.training.cli.generate_delft_data \
    --model-name="table" \
    --tei-source-path="data/generated-training-data/table/corpus/tei/*.tei.xml" \
    --raw-source-path="data/generated-training-data/table/corpus/raw/" \
    --delft-output-path="./data/generated-training-data/delft/table/corpus/table.data"
```

#### Example command for `reference_segmenter` model

```bash
python -m sciencebeam_parser.training.cli.generate_delft_data \
    --model-name="reference_segmenter" \
    --tei-source-path="data/generated-training-data/reference-segmenter/corpus/tei/*.tei.xml" \
    --raw-source-path="data/generated-training-data/reference-segmenter/corpus/raw/" \
    --delft-output-path="./data/generated-training-data/delft/reference-segmenter/corpus/reference-segmenter.data"
```

#### Example command for `affiliation_address` model

```bash
python -m sciencebeam_parser.training.cli.generate_delft_data \
    --model-name="affiliation_address" \
    --tei-source-path="data/generated-training-data/affiliation-address/corpus/*.tei.xml" \
    --delft-output-path \
    "./data/generated-training-data/delft/affiliation-address/corpus/affiliation-address.data"
```

#### Example command for `name` model

```bash
python -m sciencebeam_parser.training.cli.generate_delft_data \
    --model-name="name_header" \
    --tei-source-path="data/generated-training-data/name/header/corpus/*.tei.xml" \
    --delft-output-path \
    "./data/generated-training-data/delft/name/header/corpus/name.data"
```

```bash
python -m sciencebeam_parser.training.cli.generate_delft_data \
    --model-name="name_citation" \
    --tei-source-path="data/generated-training-data/name/citation/corpus/*.tei.xml" \
    --delft-output-path \
    "./data/generated-training-data/delft/name/citation/corpus/name.data"
```

#### Example command for `citation` model

```bash
python -m sciencebeam_parser.training.cli.generate_delft_data \
    --model-name="citation" \
    --tei-source-path="data/generated-training-data/citation/corpus/*.tei.xml" \
    --delft-output-path \
    "./data/generated-training-data/delft/citation/corpus/citation.data"
```
